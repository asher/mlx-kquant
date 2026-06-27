#!/usr/bin/env python3
"""Synthetic (GGUF-free) quantized-matmul validation for EVERY wired codec.

Companion to test_matmul.py: that one needs a real GGUF (KQUANT_TEST_GGUF) and
only exercises whatever codecs a given file contains, so legacy (q4_0..q5_1) and
several IQ codecs are usually absent. Here the wire bytes are minted in-process
by kq.quantize (the IQ encode runs CPU-only), giving every codec coverage in CI
with no model download.

The oracle is gguf-py's INDEPENDENT numpy decoder (NOT kq.dequantize), so a
shared dequant bug cannot cancel out of both sides:

    quantized_matmul(x, w, transpose=True)  ~=  x @ gguf_py.dequantize(w).T

The M-sweep covers M=1 (qmv), M in [2,8] (the verify mv_ext kernel) and a large
M (qmm). KQ_VERIFY_EXT=0 forces the pre-mv_ext fallback (verify_qmv for K/legacy,
per-row dispatch_qmv for IQ); run the suite under that env to A/B both paths -
both must match the oracle.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

# codec name (lowercase ggml enum) -> gguf type
CODECS = {
    "q4_0": GT.Q4_0,
    "q4_1": GT.Q4_1,
    "q5_0": GT.Q5_0,
    "q5_1": GT.Q5_1,
    "q8_0": GT.Q8_0,
    "q2_k": GT.Q2_K,
    "q3_k": GT.Q3_K,
    "q4_k": GT.Q4_K,
    "q5_k": GT.Q5_K,
    "q6_k": GT.Q6_K,
    "iq4_nl": GT.IQ4_NL,
    "iq4_xs": GT.IQ4_XS,
    "iq3_s": GT.IQ3_S,
    "iq3_xxs": GT.IQ3_XXS,
    "iq2_xxs": GT.IQ2_XXS,
    "iq2_xs": GT.IQ2_XS,
    "iq2_s": GT.IQ2_S,
    "iq1_s": GT.IQ1_S,
    "iq1_m": GT.IQ1_M,
}
# ggml marks these imatrix-required; kq.quantize rejects them without one.
REQ_IMAT = {"iq2_xxs", "iq2_xs", "iq1_s"}
N, K = 256, 512
MS = (1, 2, 3, 4, 8)


@pytest.mark.parametrize("codec", list(CODECS))
def test_matmul_synth(codec):
    gtype = CODECS[codec]
    rng = np.random.default_rng(0)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    imat = None
    if codec in REQ_IMAT:
        imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    wq, _ = kq.quantize(mx.array(w_np), codec, imatrix=imat)
    mx.eval(wq)
    packed = np.ascontiguousarray(np.array(wq).astype(np.uint8))
    w = mx.array(packed)
    scales = mx.zeros((1,), dtype=mx.uint8)
    # independent oracle: gguf-py decode + numpy f32 matmul
    deq = quants.dequantize(packed, gtype).astype(np.float32)
    for M in MS:
        x_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        x = mx.array(x_np).astype(mx.float16)
        got = kq.quantized_matmul(x, w, scales, codec, transpose=True)
        mx.eval(got)
        g = np.array(got).astype(np.float32)
        r = np.array(x).astype(np.float32) @ deq.T
        diff = np.abs(g - r)
        max_abs = float(diff.max())
        max_rel = float((diff / (np.abs(r) + 1e-3)).max())
        assert max_rel < 5e-2 or max_abs < 5e-3, (
            f"{codec} M={M}: max_rel={max_rel:.3e} max_abs={max_abs:.3e}"
        )
