"""Slot-parallel mix_ns (_sp) A/B.

The variant restructures parallelism only (the slot loop spreads onto
simdgroup pairs) and must be bit-identical to the loop kernel. Each arm
runs in a subprocess so the live-read env latch (KQ_MOE_SP) sees the
variable from the first dispatch.
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
bpb = {"q2_k": 84, "q4_k": 144, "q8_0": 34, "iq2_xxs": 66}[codec]
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
        ("base", {"KQ_MOE_SP": "0"}),
        ("variant", {"KQ_MOE_SP": "1"}),
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
