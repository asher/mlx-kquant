"""Fused-MoE launch-shape A/Bs: slot-parallel mix_ns (_sp), the Ext
gather simdgroups-per-threadgroup pick (KQ_MOE_SG) and the verify-width
dedupe gathers (KQ_MOE_DEDUP).

These variants restructure parallelism only (the slot loop spreads onto
simdgroup pairs; more rows share one staged LUT; one owner per expert
dots the pair of rows that share it) and must be bit-identical to the
base launch. Each arm runs in a subprocess so the live-read env latches
(KQ_MOE_SP, KQ_MOE_SG, KQ_MOE_DEDUP) see the variable from the first
dispatch.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

_SNIPPET = r"""
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq

codec, out_path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(11)
E, N, K, T, S = 32, 64, 512, 2, 6
bpb = {"q2_k": 84, "q4_k": 144, "q8_0": 34, "iq2_xxs": 66, "iq2_xs": 74}[codec]
wpb = 32 if codec == "q8_0" else 256
nb = E * N * (K // wpb)
wire = rng.integers(0, 256, size=(nb, bpb), dtype=np.uint8)
d = rng.uniform(0.004, 0.01, nb).astype(np.float16)
off = 80 if codec == "q2_k" else 0
wire[:, off:off + 2] = d.view(np.uint8).reshape(nb, 2)
dmin_off = {"q2_k": 82, "q4_k": 2}.get(codec)
if dmin_off is not None:
    dm = rng.uniform(0.001, 0.004, nb).astype(np.float16)
    wire[:, dmin_off:dmin_off + 2] = dm.view(np.uint8).reshape(nb, 2)
w = mx.array(wire.reshape(E, N, (K // wpb) * bpb))
h = mx.array((rng.standard_normal((T, S, K)) * 0.05).astype(np.float16))
x = mx.array((rng.standard_normal((T, K)) * 0.05).astype(np.float16))
inds = mx.array(rng.integers(0, E, size=(T, S)).astype(np.uint32))
sc = mx.array(rng.uniform(0.05, 0.9, size=(T, S)).astype(np.float32))
mix = kq.gather_qmv_mix_ns_kq(h, w, codec, inds, sc)
glu = kq.moe_glu_gather_kq(x, w, w, codec, inds, act="silu")
mx.eval(mix, glu)
np.savez(out_path, mix=np.array(mix.astype(mx.float32)),
         glu=np.array(glu.astype(mx.float32)))
"""


@pytest.mark.parametrize("codec", ["q2_k", "q4_k", "q8_0", "iq2_xxs"])
def test_sp_bit_identical(codec, tmp_path):
    outs = {}
    for arm, env in (
        ("base", {"KQ_MOE_SP": "0", "KQ_MOE_NX": "8"}),
        ("variant", {"KQ_MOE_SP": "1", "KQ_MOE_NX": "8"}),
    ):
        f = tmp_path / f"{arm}.npz"
        subprocess.run(
            [sys.executable, "-c", _SNIPPET, codec, str(f)],
            check=True,
            env={**os.environ, **env},
        )
        outs[arm] = np.load(f)
    for key in ("mix", "glu"):
        a, b = outs["base"][key], outs["variant"][key]
        assert np.array_equal(a, b), f"{codec} {key} not bit-identical"


_WIDE_SNIPPET = r"""
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from mlx_kquant.nn import bytes_per_row

codec, out_path, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = np.random.default_rng(5)
E, N, K, S = 8, 64, 2048, 4
wpb = 32 if codec in ("q4_0", "q5_0") else 256
bpb = bytes_per_row(codec, K) // (K // wpb)
nb = E * N * (K // wpb)
wire = rng.integers(0, 256, size=(nb, bpb), dtype=np.uint8)
d = rng.uniform(0.004, 0.01, nb).astype(np.float16)
d_off = {"q6_k": 208}.get(codec, 0)
wire[:, d_off:d_off + 2] = d.view(np.uint8).reshape(nb, 2)
if codec == "q4_k":
    dm = rng.uniform(0.001, 0.004, nb).astype(np.float16)
    wire[:, 2:4] = dm.view(np.uint8).reshape(nb, 2)
w = mx.array(wire.reshape(E, N, (K // wpb) * bpb))
h = mx.array((rng.standard_normal((T, S, K)) * 0.05).astype(np.float16))
inds = mx.array(rng.integers(0, E, size=(T, S)).astype(np.uint32))
sc = mx.array(rng.uniform(0.05, 0.9, size=(T, S)).astype(np.float32))
mix = kq.gather_qmv_mix_ns_kq(h, w, codec, inds, sc)
mx.eval(mix)
np.save(out_path, np.array(mix.astype(mx.float32)))
"""


@pytest.mark.parametrize("codec", ["q4_k", "q4_0", "q5_0", "iq3_xxs", "q6_k"])
@pytest.mark.parametrize("T", [1, 2, 3])
def test_mix_ns_wide_pick(codec, T, tmp_path):
    """The codec-keyed nx16 mix_ns pick (T <= 2, K >= 2048) dispatches the
    _nx16 kernel for the listed codecs, the slot-parallel kernel otherwise,
    and matches the forced-nx8 launch to summation-order noise."""
    wide = codec != "q6_k" and T <= 2
    outs, names = {}, {}
    for arm, env in (("auto", {}), ("nx8", {"KQ_MOE_NX": "8"})):
        f = tmp_path / f"{arm}.npy"
        r = subprocess.run(
            [sys.executable, "-c", _WIDE_SNIPPET, codec, str(f), str(T)],
            check=True,
            capture_output=True,
            text=True,
            # the per-row pick; verify widths route to the dedupe kernels
            env={**os.environ, **env, "KQ_MOE_NX_LOG": "1", "KQ_MOE_DEDUP": "0"},
        )
        outs[arm] = np.load(f)
        names[arm] = r.stdout + r.stderr
    assert ("_nx16_" in names["auto"]) == wide, names["auto"]
    assert "_sp_" in names["nx8"], names["nx8"]
    a, b = outs["auto"], outs["nx8"]
    scale = np.abs(b).max() + 1e-6
    assert np.abs(a - b).max() / scale < 2e-3, f"{codec} T={T}"


@pytest.mark.parametrize("codec", ["q2_k", "iq2_xxs", "iq2_xs"])
@pytest.mark.parametrize("sg", ["4", "8"])
def test_sg_bit_identical(codec, sg, tmp_path):
    outs = {}
    for arm, env in (
        ("base", {"KQ_MOE_SG": "2"}),
        ("variant", {"KQ_MOE_SG": sg}),
    ):
        f = tmp_path / f"{arm}.npz"
        subprocess.run(
            [sys.executable, "-c", _SNIPPET, codec, str(f)],
            check=True,
            env={**os.environ, **env},
        )
        outs[arm] = np.load(f)
    for key in ("mix", "glu"):
        a, b = outs["base"][key], outs["variant"][key]
        assert np.array_equal(a, b), f"{codec} {key} sg={sg} not bit-identical"


_HALF_SNIPPET = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
import test_moe_glu as tm

codec, sx, dtype = sys.argv[2], sys.argv[3] or None, getattr(mx, sys.argv[4])
res = tm._check_codec(codec, sx=sx, dtype=dtype)
print("RESULT " + json.dumps(
    None if res is None else [(n, float(r), bool(o)) for n, r, o in res]))
"""

# Half dots round each 4-wide product sum to half before the float chunk
# accumulate; the fused outputs land near 1e-3 relative to the f32
# reference against the 2e-3 bound of the float kernels.
HALF_REL_BOUND = 6e-3


@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "iq3_xxs"])
@pytest.mark.parametrize("sx", ["", "q6_k"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_half_dot_dispatch_and_accuracy(codec, sx, dtype):
    """KQ_MOE_HALF=1 routes the grid codecs' glu / qmv / shexp / mix gathers
    to the "_h" kernels (mix_ns has no half form and stays on the float
    kernel) and every op stays within the half-dot bound of the f32
    reference."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(
        [sys.executable, "-c", _HALF_SNIPPET, tests_dir, codec, sx, dtype],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "KQ_MOE_HALF": "1", "KQ_MOE_NX_LOG": "1"},
    )
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, r.stdout + r.stderr
    import json

    res = json.loads(line[-1][len("RESULT ") :])
    if res is None:
        pytest.skip(f"{sx or codec} fixture missing")
    names = r.stderr
    stem = codec if not sx else f"{codec}_sx_{sx}"
    for op in (
        "_moe_glu_gather_h_",
        "_gather_qmv_h",
        "_moe_glu_gather_shexp_h_",
        "_gather_qmv_mix_h",
    ):
        want = (
            f"kq_{codec}{op}"
            if op in ("_moe_glu_gather_h_", "_gather_qmv_h")
            else f"kq_{stem}{op}"
        )
        assert want in names, (want, names)
    assert f"kq_{codec}_gather_qmv_mix_ns" in names and "mix_ns_h" not in names, names
    for name, rel, ok in res:
        bound = HALF_REL_BOUND if name in ("glu", "qmv", "shexp", "mix") else None
        if bound is None:
            assert ok, (name, rel)
        else:
            assert rel < bound, (name, rel)


_DD_SNIPPET = r"""
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from mlx_kquant.nn import bytes_per_row

codec, scodec, T, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(23)
E, N, K, S = 24, 256, 512, 4
def wire(e, n, k, c):
    w = mx.array((rng.standard_normal((e, n, k)) * 0.1).astype(np.float32))
    im = mx.ones((k,), dtype=mx.float32) if c.startswith("iq") else None
    wq, _ = kq.quantize(w, c, im)
    return wq
gw, uw = wire(E, N, K, codec), wire(E, N, K, codec)
dw = wire(E, K, N, codec)
sgw, suw = wire(1, N, K, scodec)[0], wire(1, N, K, scodec)[0]
sdw = wire(1, K, N, scodec)[0]
x = mx.array((rng.standard_normal((T, K)) * 0.05).astype(np.float16))
# rows share experts (one expert on every row, so T > 2 exercises a
# second owner; one shared by rows 1..T-1), the rest are random
base = rng.integers(0, E, size=(T, S)).astype(np.uint32)
base[:, 0] = 3
base[1:, 1] = base[0, 1]
inds = mx.array(base)
sc = mx.array(rng.uniform(0.05, 0.9, size=(T, S)).astype(np.float32))
scs = mx.array(rng.uniform(0.05, 0.9, size=(T, S + 1)).astype(np.float32))
h = mx.array((rng.standard_normal((T, S, N)) * 0.05).astype(np.float16))
hs = mx.array((rng.standard_normal((T, S + 1, N)) * 0.05).astype(np.float16))
outs = {
    "glu": kq.moe_glu_gather_kq(x, gw, uw, codec, inds, act="silu"),
    "mix_ns": kq.gather_qmv_mix_ns_kq(h, dw, codec, inds, sc),
    "shexp": kq.moe_glu_gather_shexp_kq(x, gw, uw, sgw, suw, codec, inds, act="silu",
                                        shexp_kquant_type=scodec),
    "mix": kq.gather_qmv_mix_kq(
        hs, dw, sdw, codec, inds, scs, shexp_kquant_type=scodec),
}
mx.eval(*outs.values())
np.savez(out_path, **{k: np.array(v.astype(mx.float32)) for k, v in outs.items()})
"""


@pytest.mark.parametrize(
    "codec,scodec",
    [("q4_k", "q4_k"), ("iq2_xs", "q6_k"), ("iq3_xxs", "q5_k"), ("q6_k", "q6_k")],
)
@pytest.mark.parametrize("T", [2, 3, 4])
def test_dedup_bit_identical(codec, scodec, T, tmp_path):
    """KQ_MOE_DEDUP=0 (per-pair kernels) vs the default dedupe kernels at
    verify widths: identical bytes on the gate/up, shexp gate/up and both
    score-mixed down gathers, and the "_dd" names in the dispatch log."""
    outs = {}
    logs = {}
    for arm in ("0", "1"):
        out = tmp_path / f"dd{arm}.npz"
        r = subprocess.run(
            [sys.executable, "-c", _DD_SNIPPET, codec, scodec, str(T), str(out)],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "KQ_MOE_DEDUP": arm, "KQ_MOE_NX_LOG": "1"},
        )
        outs[arm] = np.load(out)
        logs[arm] = r.stderr
    for name in ("glu", "mix_ns", "shexp", "mix"):
        a, b = outs["0"][name], outs["1"][name]
        assert np.isfinite(a).all() and np.abs(a).max() > 0, name
        if name == "mix" and codec in ("q6_k", "q8_0"):
            # the per-row reference is the tuned uniform-codec kernel, whose
            # fine tiling sums in another order; the dedupe form is Ext
            assert np.allclose(a, b, rtol=1e-3, atol=1e-5), (name, np.abs(a - b).max())
        else:
            assert np.array_equal(a, b), (name, np.abs(a - b).max())
    stem = codec if scodec == codec else f"{codec}_sx_{scodec}"
    if codec in ("q6_k", "q8_0"):
        stem = codec + "_ext"
    ustem = codec + "_ext" if codec in ("q6_k", "q8_0") else codec
    for want in (
        f"kq_{ustem}_moe_glu_gather_dd_silu",
        f"kq_{codec}_gather_qmv_mix_ns_dd",
        f"kq_{stem}_moe_glu_gather_shexp_dd_silu",
        f"kq_{stem}_gather_qmv_mix_dd",
    ):
        assert want in logs["1"], (want, logs["1"])
    assert "_dd" not in logs["0"], logs["0"]
