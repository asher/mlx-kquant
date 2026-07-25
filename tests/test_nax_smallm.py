#!/usr/bin/env python3
"""Small-M NAX qmm routing validation (q6_k).

The batch-decode M range routes q6_k through three regimes: mv_ext (M 2-8),
the double-buffered BM=32 NAX tile (M 9-32), and the classic BM=64 NAX tile
(M >= 33). This sweeps M across every seam and bounds each result against a
dequantize-based float32 reference, on both an aligned and a ragged N. On
non-NAX GPUs the M 9-12 route falls back to mv_ext and BM stays 64; the
numeric contract is identical, so the assertions hold on any Metal device.

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
# Every routing seam: mv_ext tail (2, 8), NAX BM=32 entry/body/edges
# (9, 12, 13, 16, 24, 31, 32), BM=64 handoff (33, 64).
MS = [2, 8, 9, 12, 13, 16, 24, 31, 32, 33, 64]


@pytest.mark.parametrize("n_out", [1024, 1000])
def test_q6k_smallm_routing(n_out):
    mx.random.seed(11)
    wf = mx.random.normal((n_out, K)) * 0.1
    w, s = kq.quantize(wf, "q6_k")
    ref_w = kq.dequantize(w, s, "q6_k").astype(mx.float32)
    mx.eval(w, s, ref_w)
    for m in MS:
        x = (mx.random.normal((m, K)) * 0.5).astype(mx.bfloat16)
        y = kq.quantized_matmul(x, w, s, "q6_k", transpose=True)
        y = y.astype(mx.float32)
        ref = x.astype(mx.float32) @ ref_w.T
        mx.eval(y, ref)
        err = float((mx.abs(y - ref)).max() / (mx.abs(ref).max() + 1e-6))
        assert err < 2e-2, f"N{n_out} M{m}: rel err {err:.3e}"
