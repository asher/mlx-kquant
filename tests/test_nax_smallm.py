#!/usr/bin/env python3
"""Small-M NAX qmm routing validation (q6_k, q8_0, q4_k, q5_k).

The batch-decode M range routes these codecs through three regimes: the mv
paths (up to a per-codec crossover at M 6-8), the double-buffered BM=32 NAX
tile (crossover+1 through 32), and the classic BM=64 NAX tile (M >= 33).
This sweeps M across every seam and bounds each result against a
dequantize-based float32 reference, on both an aligned and a ragged N. On
non-NAX GPUs the small-M route falls back to the mv paths and BM stays 64;
the numeric contract is identical, so the assertions hold on any Metal
device.

Run locally on GPU (per-phase NAX gate, not hosted CI).
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="NAX/GPU-only routing",
)

K = 1024
# Every routing seam: mv tail (2, 6, 8), per-codec qmm crossover (7-9),
# NAX BM=32 body/edges (12, 13, 16, 24, 31, 32), BM=64 handoff (33, 64).
MS = [2, 6, 7, 8, 9, 12, 13, 16, 24, 31, 32, 33, 64]


@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", ["q6_k", "q8_0", "q4_k", "q5_k"])
def test_smallm_routing(codec, n_out):
    mx.random.seed(11)
    wf = mx.random.normal((n_out, K)) * 0.1
    w, s = kq.quantize(wf, codec)
    ref_w = kq.dequantize(w, s, codec).astype(mx.float32)
    mx.eval(w, s, ref_w)
    for m in MS:
        x = (mx.random.normal((m, K)) * 0.5).astype(mx.bfloat16)
        y = kq.quantized_matmul(x, w, s, codec, transpose=True)
        y = y.astype(mx.float32)
        ref = x.astype(mx.float32) @ ref_w.T
        mx.eval(y, ref)
        err = float((mx.abs(y - ref)).max() / (mx.abs(ref).max() + 1e-6))
        assert err < 2e-2, f"{codec} N{n_out} M{m}: rel err {err:.3e}"
