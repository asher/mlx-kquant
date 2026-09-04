"""Fused-MoE launch-shape A/Bs: slot-parallel mix_ns (_sp) and the Ext
gather simdgroups-per-threadgroup pick (KQ_MOE_SG).

Both variants restructure parallelism only (the slot loop spreads onto
simdgroup pairs; more rows share one staged LUT) and must be bit-identical
to the base launch. Each arm runs in a subprocess so the live-read env
latches (KQ_MOE_SP, KQ_MOE_SG) see the variable from the first dispatch.
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
            env={**os.environ, **env, "KQ_MOE_NX_LOG": "1"},
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
