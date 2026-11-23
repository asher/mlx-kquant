"""Routed GLU gathers whose gate and up stacks are in different codecs.

moe_glu_gather_kq and moe_glu_gather_shexp_kq take up_kquant_type for an up
stack in another codec than the gate stack (per-tensor quant recipes give
the two their own codecs). A split pair runs the same kernel forms a
single-codec block of either codec runs at each row count: the c32 step at
one row where a codec has it, the dedupe pairs at 2 to 8 rows, the grouped
kernels at 5 to 16 rows on GPUs with tensor-op units, and the plain
kernels past that. Each case checks every form against a float reference
built from the dequantized stacks, with the clamped SwiGLU act
(silu_limit) and a q6_k shared expert, at the forms the default pick takes
and at the forms the levers force. moe_glu_pair_has_kernel names the pairs
with kernels, and a pair without them raises at the call.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pytest
from kqref import GT, quants, synth_wire

GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
# codec: (gguf type, weights per block, bytes per block)
CODECS = {
    "iq2_xxs": (GT.IQ2_XXS, 256, 66),
    "stq1_0": (GT.STQ1_0, 256, 42),
    "q6_k": (GT.Q6_K, 256, 210),
}
PAIRS = (("iq2_xxs", "stq1_0"), ("stq1_0", "iq2_xxs"))
SHEXP = "q6_k"
E, N, K, R = 8, 256, 512, 4  # experts, expert width, input width, routed slots
ROWS = (1, 2, 8, 16, 64, 512)
DTYPES = ("bfloat16", "float16")
REL_BOUND = 2e-3
# Lever sets of the child runs: the default pick, the grouped and c32
# kernels forced on (every GPU runs them), and the plain kernels alone.
RUNS = {
    "default": {},
    "forced": {"KQ_MOE_GROUP": "1", "KQ_MOE_C32": "1"},
    "plain": {"KQ_MOE_GROUP": "0", "KQ_MOE_C32": "0", "KQ_MOE_DEDUP": "0"},
}


def _wire(rng, codec, rows):
    _gt, wpb, bpb = CODECS[codec]
    if codec == "q6_k":
        return _kq_quantize(rng, (rows, K), codec)
    nb = K // wpb
    # stq1_0 weights are d times -1, 0 or 1: a larger d gives its products
    # the size of the iq2_xxs ones, so the clamp engages on either codec.
    d = (0.2, 0.5) if codec == "stq1_0" else (0.004, 0.01)
    return synth_wire(rng, codec, bpb, rows * nb, d_range=d).reshape(rows, nb * bpb)


def _kq_quantize(rng, shape, codec):
    import mlx.core as mx

    import mlx_kquant as kq

    w = mx.array((rng.standard_normal(shape) * 0.02).astype(np.float32))
    wq, _ = kq.quantize(w, codec)
    mx.eval(wq)
    return np.array(wq)


def _deq(wire, codec):
    w = np.ascontiguousarray(wire)
    lead = w.shape[:-1]
    flat = w.reshape(-1, w.shape[-1])
    return (
        quants.dequantize(flat, CODECS[codec][0]).astype(np.float32).reshape(*lead, -1)
    )


@pytest.fixture(scope="module")
def wires(tmp_path_factory):
    path = os.path.join(str(tmp_path_factory.mktemp("glu_split")), "wires.npz")
    rng = np.random.default_rng(31)
    out = {}
    for codec in ("iq2_xxs", "stq1_0"):
        out[f"w/{codec}"] = np.stack([_wire(rng, codec, N) for _ in range(E)])
    out["sg"] = _wire(rng, SHEXP, N)
    out["su"] = _wire(rng, SHEXP, N)
    for t in ROWS:
        out[f"x{t}"] = rng.standard_normal((t, K)).astype(np.float32)
        ind = rng.integers(0, E, size=(t, R)).astype(np.uint32)
        ind[:, 0] = 3  # rows share an expert, which the dedupe and grouped forms pair
        out[f"i{t}"] = ind
    # The clamp bound: a fifth of the routed gate products sit above it.
    g = out["x64"] @ _deq(out["w/iq2_xxs"], "iq2_xxs")[3].T
    out["limit"] = np.float32(np.quantile(g, 0.8))
    np.savez(path, **out)
    return path


def _calls():
    """(key, gate codec, up codec, shared expert, rows, dtype)."""
    return [
        (f"{g}/{u}/{'sx' if sx else 'ns'}/{t}/{dt}", g, u, sx, t, dt)
        for g, u in PAIRS
        for sx in (False, True)
        for t in ROWS
        for dt in DTYPES
    ]


def child(wire_path, out_path):
    """Run every call of _calls; a marker line on stderr precedes each."""
    import mlx.core as mx

    import mlx_kquant as kq

    z = np.load(wire_path)
    limit = float(z["limit"])
    res = {}
    for key, g, u, sx, t, dt in _calls():
        sys.stderr.write(f"== {key}\n")
        sys.stderr.flush()
        x = mx.array(z[f"x{t}"]).astype(getattr(mx, dt))
        idx = mx.array(z[f"i{t}"])
        gw, uw = mx.array(z[f"w/{g}"]), mx.array(z[f"w/{u}"])
        if sx:
            y = kq.moe_glu_gather_shexp_kq(
                x,
                gw,
                uw,
                mx.array(z["sg"]),
                mx.array(z["su"]),
                g,
                idx,
                act="silu_limit",
                shexp_kquant_type=SHEXP,
                limit=limit,
                up_kquant_type=u,
            )
        else:
            y = kq.moe_glu_gather_kq(
                x, gw, uw, g, idx, act="silu_limit", limit=limit, up_kquant_type=u
            )
        mx.eval(y)
        res[key] = np.array(y.astype(mx.float32))
    np.savez(out_path, **res)


_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
import test_moe_glu_split as m
m.child(sys.argv[2], sys.argv[3])
"""


@pytest.fixture(scope="module", params=sorted(RUNS))
def runs(request, wires, tmp_path_factory):
    env = {"KQ_MOE_NX_LOG": "1", **RUNS[request.param]}
    out = os.path.join(str(tmp_path_factory.mktemp(request.param)), "out.npz")
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, TESTS_DIR, wires, out],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    assert r.returncode == 0, f"child exited {r.returncode}\n{r.stdout}\n{r.stderr}"
    return request.param, np.load(out), r.stderr


def _kernels(log):
    """Per call key, the GLU kernel names it launched (KQ_MOE_NX_LOG names a
    kernel at its first launch only, so later calls of one kernel list none)."""
    out, key = {}, None
    for ln in log.splitlines():
        if ln.startswith("== "):
            key = ln[3:]
            out[key] = []
        m = re.match(r"\[kq_moe\] (\S+)", ln)
        if m and key is not None:
            out[key].append(m.group(1))
    return out


def _silu_limit(g, u, limit):
    g = np.minimum(g, limit)
    return g / (1.0 + np.exp(-g)) * np.clip(u, -limit, limit)


def _reference(z, g, u, sx, t, dt):
    import mlx.core as mx

    limit = float(z["limit"])
    x = np.array(mx.array(z[f"x{t}"]).astype(getattr(mx, dt)).astype(mx.float32))
    ind = z[f"i{t}"]
    wg, wu = _deq(z[f"w/{g}"], g), _deq(z[f"w/{u}"], u)
    ref = np.zeros((t, R + int(sx), N), dtype=np.float32)
    for e in np.unique(ind):
        rows, slots = np.nonzero(ind == e)
        xr = x[rows]
        ref[rows, slots] = _silu_limit(xr @ wg[e].T, xr @ wu[e].T, limit)
    if sx:
        sg, su = _deq(z["sg"], SHEXP), _deq(z["su"], SHEXP)
        ref[:, R] = _silu_limit(x @ sg.T, x @ su.T, limit)
    # the kernels store in the call's dtype
    return np.array(mx.array(ref).astype(getattr(mx, dt)).astype(mx.float32))


def _rel(got, ref):
    return float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-6))


@GPU
def test_glu_split_against_reference(wires, runs):
    """Every pair, row count and dtype against the float reference, with and
    without the shared expert, under each lever set."""
    _name, outs, _log = runs
    z = np.load(wires)
    for key, g, u, sx, t, dt in _calls():
        got = outs[key]
        assert np.isfinite(got).all(), key
        rel = _rel(got, _reference(z, g, u, sx, t, dt))
        assert rel < REL_BOUND, (key, rel)


@GPU
def test_glu_split_clamp_engaged(wires):
    """The bound clamps a share of the gate products, so the act's clamp is
    on the checked path."""
    z = np.load(wires)
    for codec in ("iq2_xxs", "stq1_0"):
        g = z["x64"] @ _deq(z[f"w/{codec}"], codec)[3].T
        assert 0.05 < float(np.mean(g > z["limit"])) < 0.5, codec


def _form(name):
    if "_grp_" in name:
        return "grp"
    if "_dd_" in name:
        return "dd"
    if "_c32" in name:
        return "c32"
    return "plain"


# The forms each lever set allows at each row count. The default pick runs
# c32 at one row on GPUs that take it and the plain kernels elsewhere, and the
# grouped kernels from 5 rows to 16 on GPUs with tensor-op units, the dedupe
# and plain kernels without them.
ALLOWED = {
    "plain": {t: {"plain"} for t in ROWS},
    "forced": {
        1: {"c32"},
        2: {"dd"},
        8: {"grp"},
        16: {"grp"},
        64: {"c32"},
        512: {"c32"},
    },
    "default": {
        1: {"c32", "plain"},
        2: {"dd"},
        8: {"grp", "dd"},
        16: {"grp", "plain"},
        64: {"plain"},
        512: {"plain"},
    },
}
NEEDED = {
    "plain": {"plain"},
    "forced": {"c32", "dd", "grp"},
    "default": {"dd", "plain"},
}


@GPU
def test_glu_split_forms(runs):
    """The split pairs launch their own kernels, under the stem
    "<gate>_up_<up>", in the forms each lever set allows at each row count,
    and every form the set forces shows up."""
    name, _outs, log = runs
    seen = set()
    for key, names in _kernels(log).items():
        g, u, _sx, t = key.split("/")[:4]
        for k in names:
            if "_moe_glu_gather" not in k:
                continue
            assert f"kq_{g}_up_{u}_" in k, (key, k)
            assert _form(k) in ALLOWED[name][int(t)], (name, key, k)
            seen.add(_form(k))
    assert NEEDED[name] <= seen, (name, seen)


def test_glu_pair_cap():
    """moe_glu_pair_has_kernel: the split pairs with and without a q6_k
    shared expert; a pair of one codec answers as codec_has_moe_glu; a pair
    with no kernels, or a shared expert the pairs have no kernels for, does
    not."""
    import mlx_kquant as kq

    for g, u in PAIRS:
        assert kq.moe_glu_pair_has_kernel(g, u)
        assert kq.moe_glu_pair_has_kernel(g, u, "q6_k", "q6_k")
        assert kq.moe_glu_pair_has_kernel(g, u, "q8_0", "q8_0")
        assert not kq.moe_glu_pair_has_kernel(g, u, "iq4_xs", "iq4_xs")
        assert not kq.moe_glu_pair_has_kernel(g, u, "q6_k", "q8_0")
    for c in ("iq2_xxs", "q4_k", "stq1_0"):
        assert kq.moe_glu_pair_has_kernel(c, c) == kq.codec_has_moe_glu(c)
    assert kq.moe_glu_pair_has_kernel("iq2_xxs", "iq2_xxs", "q6_k", "q6_k")
    assert not kq.moe_glu_pair_has_kernel("q4_k", "iq2_xxs")
    assert not kq.moe_glu_pair_has_kernel("iq2_xxs", "nope")


@GPU
def test_glu_split_unsupported_raises(wires):
    """A pair with no kernels, or a split pair with an unsupported shared
    expert, raises at the call and names the codecs."""
    import mlx.core as mx

    import mlx_kquant as kq

    z = np.load(wires)
    x = mx.array(z["x2"]).astype(mx.bfloat16)
    idx = mx.array(z["i2"])
    q4 = mx.array(_kq_quantize(np.random.default_rng(5), (E, N, K), "q4_k"))
    iq2 = mx.array(z["w/iq2_xxs"])
    with pytest.raises(ValueError, match="q4_k.*iq2_xxs"):
        kq.moe_glu_gather_kq(x, q4, iq2, "q4_k", idx, up_kquant_type="iq2_xxs")
    with pytest.raises(ValueError, match="iq2_xxs.*stq1_0"):
        kq.moe_glu_gather_shexp_kq(
            x,
            iq2,
            mx.array(z["w/stq1_0"]),
            iq2[0],
            iq2[0],
            "iq2_xxs",
            idx,
            shexp_kquant_type="iq2_xxs",
            up_kquant_type="stq1_0",
        )
