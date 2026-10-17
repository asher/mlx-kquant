"""Fused-MoE launch-shape A/Bs: slot-parallel mix and mix_ns (_sp), the
Ext gather simdgroups-per-threadgroup pick (KQ_MOE_SG), the verify-width
dedupe gathers (KQ_MOE_DEDUP) and the form of a score-mixed down gather
(KQ_MOE_MIX_FORM).

These variants restructure parallelism only (the slot loop spreads onto
simdgroup pairs; more rows share one staged LUT; one owner per expert
dots the pair of rows that share it) and must be bit-identical to the
base launch. Each arm runs in a subprocess so the live-read env latches
(KQ_MOE_SP, KQ_MOE_SG, KQ_MOE_DEDUP, KQ_MOE_MIX_FORM) see the variable from
the first dispatch.
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


def _run_child(argv, env, what):
    """Run one arm in a subprocess and surface its output if it dies.

    check=True raises CalledProcessError, whose message carries the command
    and the exit status but not the child's stderr, so a crash in the child
    reaches CI as an unreadable wall of quoted source. Assert instead and
    put both streams in the message.
    """
    r = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    assert r.returncode == 0, (
        f"{what} exited {r.returncode}\n"
        f"--- child stdout ---\n{r.stdout}\n"
        f"--- child stderr ---\n{r.stderr}"
    )
    return r


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
        r = _run_child(
            [sys.executable, "-c", _WIDE_SNIPPET, codec, str(f), str(T)],
            # the per-row pick; verify widths route to the dedupe kernels
            {**env, "KQ_MOE_NX_LOG": "1", "KQ_MOE_DEDUP": "0"},
            f"wide {codec} T={T} arm {arm}",
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
res = tm._check_codec(codec, sx=sx, dtype=dtype, x_scale=float(sys.argv[5]))
print("RESULT " + json.dumps(
    None if res is None else [(n, float(r), bool(o)) for n, r, o in res]))
"""

# Half dots stage the grid and the activations in half and sum each
# chunk's products in float; the fused outputs land near 2e-4 relative to
# the f32 reference on float16 and near 2e-3 on bfloat16, where the float
# kernels are bound at 2e-3.
HALF_REL_BOUND = 6e-3
HALF_OPS = ("glu", "qmv", "shexp", "mix")


def _half_run(codec, sx, dtype, x_scale):
    """KQ_MOE_HALF=1 results of test_moe_glu._check_codec in a child, and
    the kernel names it logged."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    r = _run_child(
        [
            sys.executable,
            "-c",
            _HALF_SNIPPET,
            tests_dir,
            codec,
            sx,
            dtype,
            str(x_scale),
        ],
        {"KQ_MOE_HALF": "1", "KQ_MOE_NX_LOG": "1"},
        f"half {codec} sx={sx} {dtype} x_scale={x_scale}",
    )
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, r.stdout + r.stderr
    import json

    res = json.loads(line[-1][len("RESULT ") :])
    if res is None:
        pytest.skip(f"{sx or codec} fixture missing")
    return res, r.stderr


@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "iq3_xxs"])
@pytest.mark.parametrize("sx", ["", "q6_k"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_half_dot_dispatch_and_accuracy(codec, sx, dtype):
    """KQ_MOE_HALF=1 routes the grid codecs' glu / qmv / shexp / mix gathers
    to the "_h" kernels (mix_ns has no half form and stays on the float
    kernel) and every op stays within the half-dot bound of the f32
    reference."""
    res, names = _half_run(codec, sx, dtype, 0.1)
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
        if name in HALF_OPS:
            assert rel < HALF_REL_BOUND, (name, rel)
        else:
            assert ok, (name, rel)


@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "iq3_xxs"])
def test_half_dot_range(codec):
    """Activations of standard deviation 2e3 stay finite under
    KQ_MOE_HALF=1: a grid code (up to 62) times such an activation passes
    65504, so each product and the chunk sum run in float. bfloat16 keeps
    the GLU outputs in range."""
    res, names = _half_run(codec, "", "bfloat16", 2e3)
    for op in ("_moe_glu_gather_h_", "_gather_qmv_h"):
        assert f"kq_{codec}{op}" in names, (op, names)
    for name, rel, ok in res:
        if name in HALF_OPS:
            assert rel < HALF_REL_BOUND, (name, rel)
        else:
            assert ok, (name, rel)


_DD_SNIPPET = r"""
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from mlx_kquant.nn import bytes_per_row

codec, scodec, T, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(23)
# Few experts and an eval per tensor: an iq2_xs encode of one expert slab
# runs over a second of GPU work, and left lazy all six land in the same
# command buffer as the gathers below, which is long enough to trip the
# command buffer watchdog on a slow GPU. The encode is fixture cost, not
# what this test asserts on, and fewer experts means rows share more, which
# is the dedupe path.
E, N, K, S = 8, 256, 512, 4
def wire(e, n, k, c):
    w = mx.array((rng.standard_normal((e, n, k)) * 0.1).astype(np.float32))
    im = mx.ones((k,), dtype=mx.float32) if c.startswith("iq") else None
    wq, _ = kq.quantize(w, c, im)
    mx.eval(wq)
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
        r = _run_child(
            [sys.executable, "-c", _DD_SNIPPET, codec, scodec, str(T), str(out)],
            # this shape underfills the device, where the slot-parallel
            # form goes first: force the pair form on the mixed gathers
            {"KQ_MOE_DEDUP": arm, "KQ_MOE_NX_LOG": "1", "KQ_MOE_MIX_FORM": "dd"},
            f"dedup {codec}/{scodec} T={T} arm KQ_MOE_DEDUP={arm}",
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


_FORM_SNIPPET = r"""
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq

codec, scodec, T, N, out_path = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
rng = np.random.default_rng(31)
E, K, S = 6, 256, 5
def wire(n, c):
    w = mx.array((rng.standard_normal((n, K)) * 0.1).astype(np.float32))
    im = mx.ones((K,), dtype=mx.float32) if c.startswith("iq") else None
    wq, _ = kq.quantize(w, c, im)
    mx.eval(wq)
    return wq
dw = wire(E * N, codec).reshape(E, N, -1)
sdw = wire(N, scodec)
base = rng.integers(0, E, size=(T, S)).astype(np.uint32)
base[:, 0] = 3
inds = mx.array(base)
sc = mx.array(rng.uniform(0.05, 0.9, size=(T, S)).astype(np.float32))
scs = mx.array(rng.uniform(0.05, 0.9, size=(T, S + 1)).astype(np.float32))
h = mx.array((rng.standard_normal((T, S, K)) * 0.05).astype(np.float16))
hs = mx.array((rng.standard_normal((T, S + 1, K)) * 0.05).astype(np.float16))
outs = {
    "mix_ns": kq.gather_qmv_mix_ns_kq(h, dw, codec, inds, sc),
    # one score per slot, then routed scores only (the shared slot at 1)
    "mix": kq.gather_qmv_mix_kq(
        hs, dw, sdw, codec, inds, scs, shexp_kquant_type=scodec),
    "mix_unit": kq.gather_qmv_mix_kq(
        hs, dw, sdw, codec, inds, sc, shexp_kquant_type=scodec),
}
mx.eval(*outs.values())
np.savez(out_path, **{k: np.array(v.astype(mx.float32)) for k, v in outs.items()})
"""


def _form_arms(codec, scodec, T, N, tmp_path, arms):
    outs, logs = {}, {}
    for arm in arms:
        out = tmp_path / f"form_{arm}.npz"
        env = {"KQ_MOE_NX_LOG": "1"}
        if arm != "default":
            env["KQ_MOE_MIX_FORM"] = arm
        r = _run_child(
            [
                sys.executable,
                "-c",
                _FORM_SNIPPET,
                codec,
                scodec,
                str(T),
                str(N),
                str(out),
            ],
            env,
            f"mix form {codec}/{scodec} T={T} N={N} arm {arm}",
        )
        outs[arm] = np.load(out)
        logs[arm] = r.stderr
    return outs, logs


@pytest.mark.parametrize(
    "codec,scodec",
    [
        ("q2_0", "q2_0"),
        ("q2_0", "iq4_nl"),
        ("q4_k", "q4_k"),
        ("iq2_xs", "q6_k"),
        ("iq3_xxs", "q5_k"),
    ],
)
@pytest.mark.parametrize("T", [1, 2, 5])
def test_mix_sp_bit_identical(codec, scodec, T, tmp_path):
    """KQ_MOE_MIX_FORM=loop against sp on both score-mixed down gathers:
    identical bytes with one score per slot and with routed scores only,
    and the slot-parallel kernels serve the default on a grid that
    underfills the device."""
    outs, logs = _form_arms(codec, scodec, T, 256, tmp_path, ("loop", "sp", "default"))
    for name in ("mix_ns", "mix", "mix_unit"):
        a = outs["loop"][name]
        assert np.isfinite(a).all() and np.abs(a).max() > 0, name
        for arm in ("sp", "default"):
            b = outs[arm][name]
            assert np.array_equal(a, b), (name, arm, np.abs(a - b).max())
    stem = codec if scodec == codec else f"{codec}_sx_{scodec}"
    for arm in ("sp", "default"):
        for want in (
            f"kq_{codec}_gather_qmv_mix_ns_sp_",
            f"kq_{stem}_gather_qmv_mix_sp_",
        ):
            assert want in logs[arm], (arm, want, logs[arm])
    assert "_sp_" not in logs["loop"] and "_dd" not in logs["loop"], logs["loop"]


def test_mix_form_wide_grid_keeps_pair(tmp_path):
    """A grid that fills the device stays on the dedupe row-pair kernels."""
    outs, logs = _form_arms("q4_0", "q4_0", 2, 8192, tmp_path, ("default", "loop"))
    for want in ("kq_q4_0_gather_qmv_mix_ns_dd", "kq_q4_0_gather_qmv_mix_dd"):
        assert want in logs["default"], (want, logs["default"])
    assert "_sp_" not in logs["default"], logs["default"]
    for name in ("mix_ns", "mix", "mix_unit"):
        a, b = outs["loop"][name], outs["default"][name]
        assert np.array_equal(a, b), (name, np.abs(a - b).max())


_SP_COVER_SNIPPET = r"""
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

rng = np.random.default_rng(7)
N, K, S = 64, 256, 3
def wire(c, lead):
    bpb, wpb = CODEC_GEOMETRY[c][2:]
    size = (*lead, N, (K // wpb) * bpb)
    return mx.array(rng.integers(0, 256, size=size, dtype=np.uint8))
h = mx.zeros((1, S + 1, K), dtype=mx.float16)
inds = mx.array(np.arange(S, dtype=np.uint32)[None])
for c in kq.codecs():
    for sc in dict.fromkeys((c, "q5_k", "q6_k", "q8_0", "iq4_nl")):
        if not kq.shexp_mix_combo_has_kernel(c, sc):
            continue
        y = kq.gather_qmv_mix_kq(
            h, wire(c, (4,)), wire(sc, ()), c, inds, mx.ones((1, S + 1)),
            shexp_kquant_type=sc)
        mx.eval(y)
        print("PAIR", c, sc, int(kq.shexp_mix_slot_parallel(c, sc)))
"""


def test_shexp_mix_slot_parallel_matches_dispatch():
    """Every codec combination shexp_mix_slot_parallel lists dispatches its
    slot-parallel kernel at one row, and the others do not."""
    r = _run_child(
        [sys.executable, "-c", _SP_COVER_SNIPPET],
        {"KQ_MOE_NX_LOG": "1"},
        "slot-parallel cover",
    )
    pairs = [ln.split()[1:] for ln in r.stdout.splitlines() if ln.startswith("PAIR")]
    assert len(pairs) > 40, r.stdout
    assert ["q2_0", "iq4_nl", "1"] in pairs
    assert ["q6_k", "q6_k", "0"] in pairs and ["q8_0", "q8_0", "0"] in pairs
    for codec, scodec, listed in pairs:
        stem = codec if scodec == codec else f"{codec}_sx_{scodec}"
        name = f"kq_{stem}_gather_qmv_mix_sp_"
        assert (name in r.stderr) == (listed == "1"), (codec, scodec, listed)


C32_CODECS = ["iq2_s", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m"]

_C32_SNIPPET = r"""
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq

codec, T, dtype = sys.argv[1], int(sys.argv[2]), getattr(mx, sys.argv[3])
out_path = sys.argv[4]
rng = np.random.default_rng(31)
E, N, K, S = 8, 256, 1024, 4
def wire(shape, c=codec):
    w = mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32))
    wq, _ = kq.quantize(w, c, mx.ones((shape[-1],), dtype=mx.float32))
    mx.eval(wq)
    return wq
gw, uw, dw = wire((E, N, K)), wire((E, N, K)), wire((E, K, N))
sg, su, sd = wire((N, K)), wire((N, K)), wire((K, N))
qg, qu, qd = wire((N, K), "q6_k"), wire((N, K), "q6_k"), wire((K, N), "q6_k")
def act(shape):
    a = (rng.standard_normal(shape) * 0.05).astype(np.float32)
    return mx.array(a).astype(dtype)
x, h, hs = act((T, K)), act((T, S, N)), act((T, S + 1, N))
inds = mx.array(rng.integers(0, E, size=(T, S)).astype(np.uint32))
sc = mx.array(rng.uniform(0.05, 0.9, size=(T, S)).astype(np.float32))
sx = {"shexp_kquant_type": "q6_k"}
out = {
    "glu": kq.moe_glu_gather_kq(x, gw, uw, codec, inds, act="silu"),
    "qmv": kq.gather_qmv_kq(h, dw, codec, inds),
    "mix_ns": kq.gather_qmv_mix_ns_kq(h, dw, codec, inds, sc),
    "shexp": kq.moe_glu_gather_shexp_kq(x, gw, uw, sg, su, codec, inds),
    "shexp_sx": kq.moe_glu_gather_shexp_kq(x, gw, uw, qg, qu, codec, inds, **sx),
    "mix": kq.gather_qmv_mix_kq(hs, dw, sd, codec, inds, sc),
    "mix_sx": kq.gather_qmv_mix_kq(hs, dw, qd, codec, inds, sc, **sx),
    "dense": kq.quantized_matmul(
        x[:1], gw[0], mx.zeros((1,), mx.uint8), codec, transpose=True),
}
mx.eval(out)
np.savez(out_path, **{k: np.array(v.astype(mx.float32)) for k, v in out.items()})
"""


def _c32_run(codec, T, dtype, env, tmp_path, arm):
    f = tmp_path / f"{codec}-{arm}.npz"
    r = _run_child(
        [sys.executable, "-c", _C32_SNIPPET, codec, str(T), dtype, str(f)],
        {**env, "KQ_MOE_NX_LOG": "1", "KQ_QMM_LOG": "1"},
        f"c32 {codec} {arm} T={T} {dtype}",
    )
    return np.load(f), r.stderr


def _c32_class(codec):
    """This GPU's c32 defaults for codec: (MoE on, dense on, GLU suffix)."""
    import mlx.core as mx
    from mlx_kquant._ext import _moe_c32_nx, _moe_c32_on, _qmv_c32_on

    arch = mx.device_info().get("architecture", "")
    nx = _moe_c32_nx(arch)
    return (
        _moe_c32_on(arch, codec),
        _qmv_c32_on(arch, codec),
        "" if nx == 8 else f"_nx{nx}",
    )


def _c32_names(codec, dtype, glu, down):
    """The c32 kernel names of the snippet's seven gathers at one row, with
    GLU suffix glu and down suffix down ("" or "_nx16"); the score-mixed
    downs take the slot-parallel form at 8 lanes."""
    t = f"_{dtype}_t"
    sx = f"{codec}_sx_q6_k"
    mix = "_sp_c32" + t if down == "" else "_c32" + down + t
    return [
        f"kq_{codec}_moe_glu_gather_silu_c32{glu}{t}",
        f"kq_{codec}_gather_qmv_c32{down}{t}",
        f"kq_{codec}_gather_qmv_mix_ns{mix}",
        f"kq_{codec}_moe_glu_gather_shexp_silu_c32{glu}{t}",
        f"kq_{sx}_moe_glu_gather_shexp_silu_c32{glu}{t}",
        f"kq_{codec}_gather_qmv_mix{mix}",
        f"kq_{sx}_gather_qmv_mix{mix}",
    ]


def _assert_close(got, ref):
    for key in ref.files:
        rel = np.abs(got[key] - ref[key]).max() / np.abs(ref[key]).max()
        assert rel < 2e-3, (key, rel)


@pytest.mark.parametrize("codec", C32_CODECS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_c32_dispatch(codec, dtype, tmp_path):
    """At one row the grid codecs' gathers (gate/up, down, both score-mixed
    downs, the shared-expert GLU) and the one-row quantized_matmul take the
    c32 kernels where this GPU's class does (_moe_c32_on, _qmv_c32_on): the
    GLU gathers at the class width, the downs at 8 lanes, the dense mat-vec
    at the class width. KQ_MOE_C32 and KQ_QMV_C32 force either way, and
    KQ_MOE_NX or KQ_QMV_C32 of 8 or 16 force a width. The forms differ only
    in summation order."""
    moe_on, qmv_on, gsfx = _c32_class(codec)
    t = f"_{dtype}_t"
    dense = f"kq_{codec}_qmv_c32"
    old, names = _c32_run(
        codec, 1, dtype, {"KQ_MOE_C32": "0", "KQ_QMV_C32": "0"}, tmp_path, "old"
    )
    assert "_c32" not in names, names

    new, names = _c32_run(codec, 1, dtype, {}, tmp_path, "default")
    for name in _c32_names(codec, dtype, gsfx, ""):
        assert (name in names) == moe_on, (name, names)
    assert (f"{dense}{gsfx}{t}" in names) == qmv_on, names
    _assert_close(new, old)

    on = {"KQ_MOE_C32": "1", "KQ_QMV_C32": "1"}
    forced, names = _c32_run(codec, 1, dtype, on, tmp_path, "forced")
    for name in _c32_names(codec, dtype, gsfx, "") + [f"{dense}{gsfx}{t}"]:
        assert name in names, (name, names)
    _assert_close(forced, old)

    for nx, sfx in (("8", ""), ("16", "_nx16")):
        env = {"KQ_MOE_C32": "1", "KQ_MOE_NX": nx, "KQ_QMV_C32": nx}
        got, names = _c32_run(codec, 1, dtype, env, tmp_path, f"nx{nx}")
        for name in _c32_names(codec, dtype, sfx, sfx) + [f"{dense}{sfx}{t}"]:
            assert name in names, (name, names)
        _assert_close(got, old)


@pytest.mark.parametrize("codec", C32_CODECS)
def test_c32_stands_down(codec, tmp_path):
    """A forced KQ_MOE_NX=32 keeps the 16-weight gathers, as do two rows
    unless KQ_MOE_C32=1 forces c32 there (the dedupe forms, which have no
    c32 kernels, then stand down)."""
    on = {"KQ_MOE_C32": "1"}
    _, names = _c32_run(
        codec, 1, "bfloat16", {**on, "KQ_MOE_NX": "32"}, tmp_path, "nx32"
    )
    assert "gather_qmv_c32" not in names and "gather_silu_c32" not in names
    assert f"kq_{codec}_moe_glu_gather_silu_nx32_" in names, names
    _, names = _c32_run(codec, 2, "bfloat16", {}, tmp_path, "t2")
    assert "gather_silu_c32" not in names and "gather_qmv_c32" not in names
    _, names = _c32_run(
        codec, 2, "bfloat16", {**on, "KQ_MOE_DEDUP": "0"}, tmp_path, "t2c32"
    )
    _, _, gsfx = _c32_class(codec)
    for name in _c32_names(codec, "bfloat16", gsfx, "")[:2]:
        assert name in names, (name, names)


_C32_LIVE_SNIPPET = r"""
import os
import numpy as np
import mlx.core as mx
import mlx_kquant as kq

rng = np.random.default_rng(5)
w = mx.array((rng.standard_normal((4, 256, 256)) * 0.1).astype(np.float32))
wq, _ = kq.quantize(w, "iq2_s", mx.ones((256,), dtype=mx.float32))
x = mx.ones((1, 1, 256), dtype=mx.bfloat16)
inds = mx.array(np.array([[1]], np.uint32))
os.environ.pop("KQ_MOE_C32", None)
mx.eval(kq.gather_qmv_kq(x, wq, "iq2_s", inds))
os.environ["KQ_MOE_C32"] = "0"
mx.eval(kq.gather_qmv_kq(x, wq, "iq2_s", inds))
"""


def test_c32_reads_live():
    """KQ_MOE_C32 takes effect on the next call, also when the process
    started without it."""
    r = _run_child(
        [sys.executable, "-c", _C32_LIVE_SNIPPET],
        {"KQ_MOE_NX_LOG": "1"},
        "c32 live read",
    )
    assert "kq_iq2_s_gather_qmv_bfloat16_t" in r.stderr, r.stderr
    assert "kq_iq2_s_gather_qmv_c32_bfloat16_t" in r.stderr, r.stderr


_C32_REF_SNIPPET = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
import test_moe_glu as tm

res = tm._check_codec(sys.argv[3], dtype=getattr(mx, sys.argv[2]))
print("RESULT " + json.dumps(
    None if res is None else [(n, float(r), bool(o)) for n, r, o in res]))
"""


@pytest.mark.parametrize("codec", C32_CODECS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_c32_accuracy(codec, dtype):
    """The 32-weight-step kernels, forced at the reference test's three rows,
    stay inside the gather bound of the f32 dequantized reference for the
    gate/up, down, shared-expert GLU and shared-expert mix gathers."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    r = _run_child(
        [sys.executable, "-c", _C32_REF_SNIPPET, tests_dir, dtype, codec],
        {"KQ_MOE_C32": "1", "KQ_MOE_DEDUP": "0", "KQ_MOE_NX_LOG": "1"},
        f"c32 reference {codec} {dtype}",
    )
    for op in (
        "moe_glu_gather_silu_c32",
        "gather_qmv_c32",
        "moe_glu_gather_shexp_silu_c32",
        "gather_qmv_mix_sp_c32",
    ):
        assert f"kq_{codec}_{op}" in r.stderr, (op, r.stderr)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout + r.stderr
    import json

    res = json.loads(line[-1][len("RESULT ") :])
    if res is None:
        pytest.skip(f"{codec} reference unavailable")
    for name, rel, ok in res:
        assert ok, (name, rel)
