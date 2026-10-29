"""Shared experts k routed experts wide in the fused MoE gathers.

A shared expert of width k * N runs as k shared slots. moe_glu_gather_shexp_kq
reads rows [j N, (j + 1) N) of the shared gate and up tensors [k N, row
bytes] for shared slot j, and gather_qmv_mix_kq reads columns [j N, (j + 1) N)
of each row of the shared down tensor [hidden, row bytes of k N]. Each case
checks both ops against a float reference from gguf dequantize, and the whole
block against the dense shared expert (one matmul of width k N per
projection), over k = 1, 2 and 4, T = 1, 2 and 33, and each kernel form:
the default pick, the forced loop, slot-parallel and dedupe forms
(KQ_MOE_MIX_FORM), the per-row gathers (KQ_MOE_DEDUP=0), the wide GLU
gathers (KQ_MOE_NX=16) and the half-dot gathers (KQ_MOE_HALF=1), all on
the Ext kernels (KQ_MOE_QF=0), and the qf kernels (KQ_MOE_QF=1). The forms
latch their levers once per process, so each runs in a child.

The slot-parallel down runs 64 * S threads per threadgroup. When the
pipeline takes fewer, it runs the loop kernel. KQ_MOE_SP_THREAD_CAP lowers
that limit, so a test at S = 8 under a cap of 256 takes the fallback on any
GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from kqref import GT, quants

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
GTYPE = {
    "q4_1": GT.Q4_1,
    "q4_k": GT.Q4_K,
    "q6_k": GT.Q6_K,
    "q8_0": GT.Q8_0,
    "iq2_s": GT.IQ2_S,
    "iq2_xs": GT.IQ2_XS,
    "iq4_xs": GT.IQ4_XS,
    "iq4_nl": GT.IQ4_NL,
    "q2_0": GT.Q2_0,
}
E, H, R = 8, 512, 6  # experts, hidden size, routed slots
ROWS = (1, 2, 33)
REL_BOUND = 2e-3
HALF_REL_BOUND = 6e-3
# (name, gate/up stack codec, shexp gate codec, shexp up codec, down stack
# codec, shexp down codec, k, N)
CASES = [
    ("q4_1_k1", "q4_1", "q4_1", "q4_1", "q4_1", "q4_1", 1, 512),
    ("q4_1_k2", "q4_1", "q4_1", "q4_1", "q4_1", "q4_1", 2, 512),
    ("q4_1_k4", "q4_1", "q4_1", "q4_1", "q4_1", "q4_1", 4, 512),
    ("q4_k_q4_k_k2", "q4_k", "q4_k", "q4_k", "q4_k", "q4_k", 2, 512),
    ("q4_k_q6_k_k2", "q4_k", "q4_k", "q4_k", "q6_k", "q6_k", 2, 512),
    ("q4_1_up_q6_k_k2", "q4_1", "q6_k", "q6_k", "q4_1", "q6_k", 2, 512),
    ("q4_1_up_q8_0_k2", "q4_1", "q8_0", "q8_0", "q4_1", "q8_0", 2, 512),
    ("q6_k_k2", "q6_k", "q6_k", "q6_k", "q6_k", "q6_k", 2, 512),
    ("q8_0_k2", "q8_0", "q8_0", "q8_0", "q8_0", "q8_0", 2, 512),
    ("q4_1_k2_n64", "q4_1", "q4_1", "q4_1", "q4_1", "q4_1", 2, 64),
    ("iq2_s_k2", "iq2_s", "iq2_s", "iq2_s", "iq2_s", "iq2_s", 2, 512),
    ("iq2_s_split_k2", "iq2_s", "iq4_xs", "q6_k", "iq2_s", "q8_0", 2, 512),
    ("iq2_xs_up_q6_k_k2", "iq2_xs", "q6_k", "q6_k", "iq2_xs", "q6_k", 2, 512),
    ("q2_0_iq4_nl_k2", "q2_0", "q2_0", "q2_0", "q2_0", "iq4_nl", 2, 512),
]
HALF_CASES = ("iq2_xs_up_q6_k_k2",)
ARMS = {
    "default": {},
    "loop": {"KQ_MOE_MIX_FORM": "loop"},
    "sp": {"KQ_MOE_MIX_FORM": "sp"},
    "dd": {"KQ_MOE_MIX_FORM": "dd"},
    "nodedup": {"KQ_MOE_DEDUP": "0"},
    "nx16": {"KQ_MOE_NX": "16"},
    "half": {"KQ_MOE_HALF": "1"},
    "qf": {"KQ_MOE_QF": "1"},
}


def _wire_path(tmp_root):
    return os.path.join(tmp_root, "shexp_multi_wires.npz")


def _quantize(rng, shape, codec):
    import mlx.core as mx

    import mlx_kquant as kq

    w = mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32))
    im = mx.ones((shape[-1],), dtype=mx.float32) if codec.startswith("iq") else None
    wq, _ = kq.quantize(w, codec, im)
    mx.eval(wq)
    return np.array(wq)


def _make_wires(path):
    rng = np.random.default_rng(17)
    out = {}
    for name, cg, sg, su, cd, sd, k, n in CASES:
        out[f"{name}/gw"] = _quantize(rng, (E, n, H), cg)
        out[f"{name}/uw"] = _quantize(rng, (E, n, H), cg)
        out[f"{name}/dw"] = _quantize(rng, (E, H, n), cd)
        out[f"{name}/sg"] = _quantize(rng, (k * n, H), sg)
        out[f"{name}/su"] = _quantize(rng, (k * n, H), su)
        out[f"{name}/sd"] = _quantize(rng, (H, k * n), sd)
        for t in ROWS:
            out[f"{name}/x{t}"] = (rng.standard_normal((t, H)) * 0.1).astype(np.float16)
            out[f"{name}/h{t}"] = (rng.standard_normal((t, R + k, n)) * 0.1).astype(
                np.float16
            )
            ind = rng.integers(0, E, size=(t, R)).astype(np.uint32)
            # rows share experts, which the dedupe forms pair
            ind[:, 0] = 3
            out[f"{name}/i{t}"] = ind
            out[f"{name}/s{t}"] = rng.uniform(0.05, 0.9, size=(t, R + k)).astype(
                np.float32
            )
    np.savez(path, **out)


@pytest.fixture(scope="module")
def wires(tmp_path_factory):
    path = _wire_path(str(tmp_path_factory.mktemp("shexp_multi")))
    _make_wires(path)
    return path


def child(wire_path, out_path, names):
    """Run both ops for each named case and save the outputs."""
    import mlx.core as mx

    import mlx_kquant as kq

    z = np.load(wire_path)
    res = {}
    for name, cg, sg, su, cd, sd, _k, _n in CASES:
        if name not in names:
            continue
        a = {key: mx.array(z[f"{name}/{key}"]) for key in ("gw", "uw", "dw")}
        a.update({key: mx.array(z[f"{name}/{key}"]) for key in ("sg", "su", "sd")})
        for t in ROWS:
            x = mx.array(z[f"{name}/x{t}"])
            h = mx.array(z[f"{name}/h{t}"])
            ind = mx.array(z[f"{name}/i{t}"])
            sc = mx.array(z[f"{name}/s{t}"])
            glu = kq.moe_glu_gather_shexp_kq(
                x,
                a["gw"],
                a["uw"],
                a["sg"],
                a["su"],
                cg,
                ind,
                shexp_kquant_type=sg,
                shexp_up_kquant_type=su,
            )
            mix = kq.gather_qmv_mix_kq(
                h, a["dw"], a["sd"], cd, ind, sc, shexp_kquant_type=sd
            )
            unit = kq.gather_qmv_mix_kq(
                h, a["dw"], a["sd"], cd, ind, sc[:, :R], shexp_kquant_type=sd
            )
            block = kq.gather_qmv_mix_kq(
                glu, a["dw"], a["sd"], cd, ind, sc[:, :R], shexp_kquant_type=sd
            )
            for key, v in (("glu", glu), ("mix", mix), ("unit", unit)):
                res[f"{name}/{key}{t}"] = v
            res[f"{name}/block{t}"] = block
    mx.eval(res)
    np.savez(out_path, **{k: np.array(v.astype(mx.float32)) for k, v in res.items()})


_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
import json
import test_moe_shexp_multi as m
m.child(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
"""


def _run(wire_path, out_path, names, env):
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            TESTS_DIR,
            wire_path,
            out_path,
            json.dumps(names),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "KQ_MOE_NX_LOG": "1", "KQ_MOE_QF": "0", **env},
    )
    assert r.returncode == 0, f"child exited {r.returncode}\n{r.stdout}\n{r.stderr}"
    return np.load(out_path), r.stderr


def _deq(wire, codec):
    w = np.ascontiguousarray(wire)
    lead = w.shape[:-1]
    flat = w.reshape(-1, w.shape[-1])
    return quants.dequantize(flat, GTYPE[codec]).astype(np.float32).reshape(*lead, -1)


def _silu_glu(g, u):
    with np.errstate(over="ignore"):
        return g / (1.0 + np.exp(-g)) * u


_REFS = {}


def _refs(wire_path, name):
    """Float references for one case: GLU [T, R + k, N], mix and mix with
    unit shared weights [T, H] at each row count, and the deq tensors."""
    key = (wire_path, name)
    if key in _REFS:
        return _REFS[key]
    case = next(c for c in CASES if c[0] == name)
    _, cg, sg, su, cd, sd, k, n = case
    z = np.load(wire_path)
    gw, uw = _deq(z[f"{name}/gw"], cg), _deq(z[f"{name}/uw"], cg)
    dw = _deq(z[f"{name}/dw"], cd)
    sgw, suw = _deq(z[f"{name}/sg"], sg), _deq(z[f"{name}/su"], su)
    sdw = _deq(z[f"{name}/sd"], sd)
    out = {}
    for t in ROWS:
        x = z[f"{name}/x{t}"].astype(np.float32)
        h = z[f"{name}/h{t}"].astype(np.float32)
        ind, sc = z[f"{name}/i{t}"], z[f"{name}/s{t}"]
        routed = np.stack(
            [
                np.stack([_silu_glu(x[r] @ gw[e].T, x[r] @ uw[e].T) for e in ind[r]])
                for r in range(t)
            ]
        )
        shared = _silu_glu(x @ sgw.T, x @ suw.T)  # [T, k N], one dense matmul
        out[f"glu{t}"] = np.concatenate([routed, shared.reshape(t, k, n)], axis=1)
        routed_mix = np.stack(
            [
                sum(sc[r, s] * (h[r, s] @ dw[ind[r, s]].T) for s in range(R))
                for r in range(t)
            ]
        )
        shared_slots = [h[:, R + j] @ sdw[:, j * n : (j + 1) * n].T for j in range(k)]
        out[f"mix{t}"] = routed_mix + sum(
            sc[:, R + j, None] * shared_slots[j] for j in range(k)
        )
        out[f"unit{t}"] = routed_mix + sum(shared_slots)
        # The whole block: routed experts plus the dense shared expert.
        out[f"block{t}"] = (
            np.stack(
                [
                    sum(sc[r, s] * (routed[r, s] @ dw[ind[r, s]].T) for s in range(R))
                    for r in range(t)
                ]
            )
            + shared @ sdw.T
        )
    _REFS[key] = out
    return out


def _rel(got, ref):
    return float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-6))


def _check(outs, wire_path, names, bound):
    for name in names:
        ref = _refs(wire_path, name)
        for t in ROWS:
            for key in ("glu", "mix", "unit"):
                got = outs[f"{name}/{key}{t}"]
                assert np.isfinite(got).all(), (name, key, t)
                rel = _rel(got, ref[f"{key}{t}"])
                assert rel < bound, (name, key, t, rel)
            # the block runs on the fp16 GLU output, one rounding more
            rel = _rel(outs[f"{name}/block{t}"], ref[f"block{t}"])
            assert rel < 3 * bound, (name, "block", t, rel)


@pytest.mark.parametrize("arm", [a for a in ARMS if a != "half"])
def test_shexp_multi_against_reference(arm, wires, tmp_path):
    """Every case, every row count, against the float reference under one
    kernel form."""
    names = [c[0] for c in CASES]
    outs, _ = _run(wires, str(tmp_path / f"{arm}.npz"), names, ARMS[arm])
    _check(outs, wires, names, REL_BOUND)


def test_shexp_multi_half_dot(wires, tmp_path):
    """KQ_MOE_HALF=1 serves the routed slots of a grid-codec stack on the
    half-dot kernels; the shared slots keep the float path."""
    names = list(HALF_CASES)
    outs, log = _run(wires, str(tmp_path / "half.npz"), names, ARMS["half"])
    assert "_moe_glu_gather_shexp_h_" in log and "_gather_qmv_mix_h" in log, log
    _check(outs, wires, names, HALF_REL_BOUND)


def test_shexp_multi_loop_and_sp_bit_identical(wires, tmp_path):
    """The slot-parallel down replays the loop kernel's score chain, so the
    two forms give the same bytes with k shared slots too."""
    names = [c[0] for c in CASES]
    loop, _ = _run(wires, str(tmp_path / "loop.npz"), names, ARMS["loop"])
    sp, log = _run(wires, str(tmp_path / "sp.npz"), names, ARMS["sp"])
    assert "_gather_qmv_mix_sp_" in log, log
    for key in loop.files:
        if "/mix" in key or "/unit" in key:
            assert np.array_equal(loop[key], sp[key]), key


_GUARD = r"""
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq

rng = np.random.default_rng(29)
E, H, N, R, k = 16, 1024, 512, 6, 2
S = R + k
def q(shape):
    w = mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32))
    wq, _ = kq.quantize(w, "q4_1")
    mx.eval(wq)
    return wq
dw, sdw, nsw = q((E, H, N)), q((H, k * N)), q((E, H, N))
h = mx.array((rng.standard_normal((1, S, N)) * 0.1).astype(np.float16))
ind = mx.array(rng.integers(0, E, size=(1, R)).astype(np.uint32))
ind_ns = mx.array(rng.integers(0, E, size=(1, S)).astype(np.uint32))
sc = mx.array(rng.uniform(0.05, 0.9, size=(1, S)).astype(np.float32))
mix = kq.gather_qmv_mix_kq(h, dw, sdw, "q4_1", ind, sc)
ns = kq.gather_qmv_mix_ns_kq(h, nsw, "q4_1", ind_ns, sc)
# one shared slot, the width the qf mix takes
sdw1 = q((H, N))
ind1 = mx.array(rng.integers(0, E, size=(1, S - 1)).astype(np.uint32))
mix1 = kq.gather_qmv_mix_kq(h, dw, sdw1, "q4_1", ind1, sc)
mx.eval(mix, ns, mix1)
np.savez(sys.argv[1], mix=np.array(mix.astype(mx.float32)),
         ns=np.array(ns.astype(mx.float32)), h=np.array(h.astype(mx.float32)),
         dw=np.array(dw), sdw=np.array(sdw), nsw=np.array(nsw),
         ind=np.array(ind), ind_ns=np.array(ind_ns), sc=np.array(sc),
         mix1=np.array(mix1.astype(mx.float32)), sdw1=np.array(sdw1),
         ind1=np.array(ind1))
"""


def _guard_run(tmp_path, env, arm):
    out = tmp_path / f"guard_{arm}.npz"
    r = subprocess.run(
        [sys.executable, "-c", _GUARD, str(out)],
        capture_output=True,
        text=True,
        env={**os.environ, "KQ_MOE_NX_LOG": "1", **env},
    )
    assert r.returncode == 0, f"guard child {arm}\n{r.stdout}\n{r.stderr}"
    z = np.load(out)
    n, k = 512, 2
    h, ind, ind_ns, sc = z["h"][0], z["ind"][0], z["ind_ns"][0], z["sc"][0]
    dw, sdw, nsw = _deq(z["dw"], "q4_1"), _deq(z["sdw"], "q4_1"), _deq(z["nsw"], "q4_1")
    ref = sum(sc[s] * (h[s] @ dw[ind[s]].T) for s in range(R))
    ref = ref + sum(
        sc[R + j] * (h[R + j] @ sdw[:, j * n : (j + 1) * n].T) for j in range(k)
    )
    ref_ns = sum(sc[s] * (h[s] @ nsw[ind_ns[s]].T) for s in range(R + k))
    sdw1, ind1, last = _deq(z["sdw1"], "q4_1"), z["ind1"][0], R + k - 1
    ref1 = sum(sc[s] * (h[s] @ dw[ind1[s]].T) for s in range(last))
    ref1 = ref1 + sc[last] * (h[last] @ sdw1.T)
    assert _rel(z["mix"][0], ref) < REL_BOUND
    assert _rel(z["ns"][0], ref_ns) < REL_BOUND
    assert _rel(z["mix1"][0], ref1) < REL_BOUND
    return r.stderr


def _qf_stems(qf):
    """The one-slot mix and the mix_ns stems; the two-slot mix stays on the
    Ext kernel, since the qf kernels take one shared slot."""
    if qf == "1":
        return "kq_q4_1_qf_sx_q4_1_gather_qmv_mix", "kq_q4_1_qf_gather_qmv_mix_ns"
    return "kq_q4_1_gather_qmv_mix", "kq_q4_1_gather_qmv_mix_ns"


@pytest.mark.parametrize("qf", ["0", "1"])
def test_slot_parallel_guard_declines_under_thread_cap(tmp_path, qf):
    """At S = 8 the slot-parallel downs need 512 threads. Under a cap of 256
    both guards decline, and the loop kernels serve the calls."""
    log = _guard_run(
        tmp_path, {"KQ_MOE_SP_THREAD_CAP": "256", "KQ_MOE_QF": qf}, f"cap{qf}"
    )
    mix1, ns = _qf_stems(qf)
    assert "kq_q4_1_gather_qmv_mix_float16_t" in log, log
    assert mix1 + "_float16_t" in log, log
    assert ns + "_float16_t" in log, log
    assert "_sp_" not in log, log


@pytest.mark.parametrize("qf", ["0", "1"])
def test_slot_parallel_at_eight_slots(tmp_path, qf):
    """Without the cap, S = 8 runs the slot-parallel downs where the
    pipeline takes 512 threads, and the loop kernels where it does not."""
    log = _guard_run(tmp_path, {"KQ_MOE_QF": qf}, f"nocap{qf}")
    sp_lines = [ln for ln in log.splitlines() if "_sp_" in ln]
    loop = "kq_q4_1_gather_qmv_mix_float16_t" in log
    if sp_lines:
        for ln in sp_lines:
            assert int(ln.rsplit("max ", 1)[1].rstrip(")")) >= 512, ln
        assert "kq_q4_1_gather_qmv_mix_sp_float16_t" in log, log
        mix1, ns = _qf_stems(qf)
        assert mix1 + "_sp_float16_t" in log, log
        assert ns + "_sp_float16_t" in log, log
    else:
        assert loop, log


def _q4_1(rng, shape):
    return _quantize(rng, shape, "q4_1")


def test_shexp_multi_weight_views_at_offsets():
    """Expert stacks and shared-expert tensors that are row-major views at a
    nonzero offset into a larger buffer run in place and give the bytes of
    the same weights in their own buffers."""
    import mlx.core as mx

    import mlx_kquant as kq

    rng = np.random.default_rng(41)
    n, k, t = 512, 2, 2
    big_g = mx.array(_q4_1(rng, (E + 1, n, H)))
    big_u = mx.array(_q4_1(rng, (E + 2, n, H)))
    big_d = mx.array(_q4_1(rng, (E + 1, H, n)))
    big_sg = mx.array(_q4_1(rng, ((k + 1) * n, H)))
    big_su = mx.array(_q4_1(rng, ((k + 2) * n, H)))
    big_sd = mx.array(_q4_1(rng, (H + 8, k * n)))
    mx.eval(big_g, big_u, big_d, big_sg, big_su, big_sd)
    views = {
        "gw": big_g[1:],
        "uw": big_u[2:],
        "dw": big_d[1:],
        "sg": big_sg[n:],
        "su": big_su[2 * n :],
        "sd": big_sd[8:],
    }
    own = {key: mx.array(np.array(v)) for key, v in views.items()}
    x = mx.array((rng.standard_normal((t, H)) * 0.1).astype(np.float16))
    h = mx.array((rng.standard_normal((t, R + k, n)) * 0.1).astype(np.float16))
    ind = mx.array(rng.integers(0, E, size=(t, R)).astype(np.uint32))
    sc = mx.array(rng.uniform(0.05, 0.9, size=(t, R)).astype(np.float32))

    def run(w):
        glu = kq.moe_glu_gather_shexp_kq(
            x, w["gw"], w["uw"], w["sg"], w["su"], "q4_1", ind
        )
        mix = kq.gather_qmv_mix_kq(h, w["dw"], w["sd"], "q4_1", ind, sc)
        mx.eval(glu, mix)
        return np.array(glu.astype(mx.float32)), np.array(mix.astype(mx.float32))

    for a, b in zip(run(views), run(own), strict=True):
        assert np.array_equal(a, b)
    # the views reach the reference too
    sgw = _deq(np.array(own["sg"]), "q4_1")
    suw = _deq(np.array(own["su"]), "q4_1")
    xf = np.array(x.astype(mx.float32))
    shared = _silu_glu(xf @ sgw.T, xf @ suw.T).reshape(t, k, n)
    glu, _ = run(views)
    assert _rel(glu[:, R:], shared) < REL_BOUND


def test_shexp_multi_declines():
    """Shapes the kernels cannot serve raise, and the probes say so first."""
    import mlx.core as mx

    import mlx_kquant as kq

    rng = np.random.default_rng(43)
    n, t = 512, 1
    gw = mx.array(_q4_1(rng, (E, n, H)))
    x = mx.zeros((t, H), dtype=mx.float16)
    ind = mx.zeros((t, R), dtype=mx.uint32)
    # gate rows not a whole number of expert widths
    sg = mx.array(_q4_1(rng, (n + n // 2, H)))
    with pytest.raises(ValueError, match="whole k"):
        kq.moe_glu_gather_shexp_kq(x, gw, gw, sg, sg, "q4_1", ind)
    # gate and up of different widths
    s1, s2 = mx.array(_q4_1(rng, (n, H))), mx.array(_q4_1(rng, (2 * n, H)))
    with pytest.raises(ValueError, match="gate/up widths"):
        kq.moe_glu_gather_shexp_kq(x, gw, gw, s1, s2, "q4_1", ind)

    dw = mx.array(_q4_1(rng, (E, H, n)))
    h = mx.zeros((t, R + 2, n), dtype=mx.float16)
    sc = mx.ones((t, R), dtype=mx.float32)
    # down width not a whole number of expert widths
    sd = mx.array(_q4_1(rng, (H, n + n // 2)))
    with pytest.raises(ValueError, match="whole k"):
        kq.gather_qmv_mix_kq(h, dw, sd, "q4_1", ind, sc)
    # indices that do not leave k shared slots
    sd2 = mx.array(_q4_1(rng, (H, 2 * n)))
    with pytest.raises(ValueError, match="S - k"):
        kq.gather_qmv_mix_kq(h, dw, sd2, "q4_1", ind[:, : R - 1], sc)
    # a 640-wide expert with a q6_k shared down: slot 1 starts at weight 640,
    # inside a 256-weight block
    dw640 = mx.array(_q4_1(rng, (E, H, 640)))
    sd640 = mx.array(_quantize(rng, (H, 1280), "q6_k"))
    h640 = mx.zeros((t, R + 2, 640), dtype=mx.float16)
    with pytest.raises(ValueError, match="whole number of 'q6_k' blocks"):
        kq.gather_qmv_mix_kq(
            h640, dw640, sd640, "q4_1", ind, sc, shexp_kquant_type="q6_k"
        )

    assert kq.shexp_mix_combo_has_kernel("q4_1", "q6_k", 2, 512)
    assert not kq.shexp_mix_combo_has_kernel("q4_1", "q6_k", 2, 640)
    assert kq.shexp_mix_combo_has_kernel("q4_1", "q4_1", 2, 640)
    assert not kq.shexp_mix_combo_has_kernel("q4_1", "q4_1", 2)
    assert not kq.shexp_mix_combo_has_kernel("q4_1", "q4_1", 0, 512)
    assert kq.shexp_mix_combo_has_kernel("q4_1", "q4_1")
    assert kq.shexp_glu_combo_has_kernel("q4_1", "q4_1", "q4_1", 2)
    assert not kq.shexp_glu_combo_has_kernel("q4_1", "q4_1", "q4_1", 0)
    assert kq.shexp_glu_combo_has_kernel("q4_1", "q4_1", "q4_1")
    assert kq.shexp_mix_slot_parallel("q4_1", "q4_1", 2, 512)
    assert not kq.shexp_mix_slot_parallel("q6_k", "q6_k", 2, 512)


def test_shexp_multi_lever_declines(monkeypatch):
    """KQ_MOE_SHEXP_MULTI=0 declines a shared expert wider than one routed
    expert, read per call; one routed width still runs."""
    import mlx.core as mx

    import mlx_kquant as kq

    rng = np.random.default_rng(47)
    n, t = 512, 1
    gw = mx.array(_q4_1(rng, (E, n, H)))
    dw = mx.array(_q4_1(rng, (E, H, n)))
    s1, s2 = mx.array(_q4_1(rng, (n, H))), mx.array(_q4_1(rng, (2 * n, H)))
    d1, d2 = mx.array(_q4_1(rng, (H, n))), mx.array(_q4_1(rng, (H, 2 * n)))
    x = mx.zeros((t, H), dtype=mx.float16)
    ind = mx.zeros((t, R), dtype=mx.uint32)
    sc = mx.ones((t, R), dtype=mx.float32)
    monkeypatch.setenv("KQ_MOE_SHEXP_MULTI", "0")
    assert not kq.shexp_glu_combo_has_kernel("q4_1", "q4_1", "q4_1", 2)
    assert not kq.shexp_mix_combo_has_kernel("q4_1", "q4_1", 2, n)
    assert kq.shexp_mix_combo_has_kernel("q4_1", "q4_1", 1, n)
    with pytest.raises(ValueError, match="KQ_MOE_SHEXP_MULTI"):
        kq.moe_glu_gather_shexp_kq(x, gw, gw, s2, s2, "q4_1", ind)
    with pytest.raises(ValueError, match="KQ_MOE_SHEXP_MULTI"):
        kq.gather_qmv_mix_kq(
            mx.zeros((t, R + 2, n), dtype=mx.float16), dw, d2, "q4_1", ind, sc
        )
    glu = kq.moe_glu_gather_shexp_kq(x, gw, gw, s1, s1, "q4_1", ind)
    mix = kq.gather_qmv_mix_kq(
        mx.zeros((t, R + 1, n), dtype=mx.float16), dw, d1, "q4_1", ind, sc
    )
    mx.eval(glu, mix)
    assert glu.shape == (t, R + 1, n) and mix.shape == (t, H)
    monkeypatch.delenv("KQ_MOE_SHEXP_MULTI")
    assert kq.shexp_glu_combo_has_kernel("q4_1", "q4_1", "q4_1", 2)
