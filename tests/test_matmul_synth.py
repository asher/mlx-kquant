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

The M-sweep covers M=1 (qmv), M in [2,8] (the verify mv_ext kernel, or per-row
qmv where NAX GPUs route this small N to it) and a large M (qmm). A second arm
forces mv_ext through KQ_QMM_ROUTE so it meets the oracle on every GPU.
KQ_VERIFY_EXT=0 forces the pre-mv_ext fallback (verify_qmv for K/legacy, per-row
dispatch_qmv for IQ). Run the suite under that env to A/B both paths, which must
both match the oracle.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from kqref import GT, quants

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
    "stq1_0": GT.STQ1_0,
    "pq2_0": GT.PQ2_0,
    "q2_0": GT.Q2_0,
    "ptq1_0": GT.PTQ1_0,
}
# ggml marks these imatrix-required; kq.quantize rejects them without one.
REQ_IMAT = {"iq2_xxs", "iq2_xs", "iq1_s"}
N, K = 256, 512
# 1 takes qmv, 2 through 8 take the verify mv_ext kernel (or per-row qmv
# on NAX GPUs), and 64 takes qmm. Without a large M the block loaders
# never run, so the prefill path of every codec goes untested.
MS = (1, 2, 3, 4, 8, 64)


@pytest.mark.parametrize("route", ["default", "mv_ext"])
@pytest.mark.parametrize("codec", list(CODECS))
def test_matmul_synth(codec, route, monkeypatch):
    if route != "default":
        monkeypatch.setenv("KQ_QMM_ROUTE", route)
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
        assert max_rel < 5e-2 or max_abs < 1e-2, (
            f"{codec} M={M}: max_rel={max_rel:.3e} max_abs={max_abs:.3e}"
        )


# Codecs with a 32-weight block. They accept any K that 32 divides, so they
# reach the plain qmv path. The superblock codecs need K that 256 divides,
# which always takes the fast path.
BLOCK32 = ("q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl")
# 288 is 9 blocks of 32. It is not a multiple of qmv_fast_k_align(), so the
# kernel must guard the tail. K=512 above only covers the fast path.
K_TAIL = 288


@pytest.mark.parametrize("codec", BLOCK32)
def test_matmul_synth_tail(codec):
    """Validate the plain qmv path, where the block count leaves a tail."""
    gtype = CODECS[codec]
    rng = np.random.default_rng(0)
    w_np = (rng.standard_normal((N, K_TAIL)) * 0.1).astype(np.float32)
    wq, _ = kq.quantize(mx.array(w_np), codec)
    mx.eval(wq)
    packed = np.ascontiguousarray(np.array(wq).astype(np.uint8))
    w = mx.array(packed)
    scales = mx.zeros((1,), dtype=mx.uint8)
    deq = quants.dequantize(packed, gtype).astype(np.float32)
    for M in MS:
        x_np = (rng.standard_normal((M, K_TAIL)) * 0.1).astype(np.float32)
        x = mx.array(x_np).astype(mx.float16)
        got = kq.quantized_matmul(x, w, scales, codec, transpose=True)
        mx.eval(got)
        g = np.array(got).astype(np.float32)
        r = np.array(x).astype(np.float32) @ deq.T
        diff = np.abs(g - r)
        max_abs = float(diff.max())
        max_rel = float((diff / (np.abs(r) + 1e-3)).max())
        assert max_rel < 5e-2 or max_abs < 1e-2, (
            f"{codec} M={M}: max_rel={max_rel:.3e} max_abs={max_abs:.3e}"
        )


# q6_k at M 1 runs the split-K mat-vec when 4 divides N, and the per-row
# kernels otherwise or under KQ_QMV_SPLITK=0. K 256 leaves seven of the
# eight simdgroups without a superblock, and K 5376 splits 21 superblocks
# unevenly. Both arms meet the oracle in every dtype and agree bit for bit
# where N rules the split out. The op promotes float32 x to bfloat16. The
# tolerance scales with the output because the CPU path quantizes x to 8
# bits, which costs it about 1% of the largest output at these K.
SPLITK_SHAPES = ((256, 256), (260, 5376), (1024, 4096), (258, 512))
# Levers that move an M=1 call off the qmv route or change its tiling.
# KQ_VERIFY_EXT and KQ_QMM_SPLITK are read once per process, so a set one
# skips the tests.
QMV_LEVERS = (
    "KQ_QMV_FINE",
    "KQ_QMV_SPLITK",
    "KQ_NAX_QMV",
    "KQ_VERIFY_NAX",
    "KQ_VERIFY_MMA",
    "KQ_QMM_SPLITK_NAX",
    "KQ_DISABLE_NAX",
    "KQ_FORCE_QMM_MIN_M",
    "KQ_QMM_ROUTE",
    "KQ_QMM_ROUTE_STRICT",
)
# Explicit mantissa bits of the output (float32 x is promoted to bfloat16).
MANTISSA = {mx.float32: 7, mx.float16: 10, mx.bfloat16: 7}


def _clear_qmv_levers(monkeypatch):
    if any(os.environ.get(k) is not None for k in ("KQ_VERIFY_EXT", "KQ_QMM_SPLITK")):
        pytest.skip("a process-static routing lever is set")
    for k in QMV_LEVERS:
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("n, k", SPLITK_SHAPES)
def test_q6_k_qmv_splitk(n, k, dtype, monkeypatch):
    _clear_qmv_levers(monkeypatch)
    rng = np.random.default_rng(1)
    w_np = (rng.standard_normal((n, k)) * 0.1).astype(np.float32)
    wq, _ = kq.quantize(mx.array(w_np), "q6_k")
    mx.eval(wq)
    packed = np.ascontiguousarray(np.array(wq).astype(np.uint8))
    w = mx.array(packed)
    scales = mx.zeros((1,), dtype=mx.uint8)
    deq = quants.dequantize(packed, GT.Q6_K).astype(np.float32)
    x_np = (rng.standard_normal((1, k)) * 0.1).astype(np.float32)
    x = mx.array(x_np).astype(dtype)
    r = np.array(x.astype(mx.float32)) @ deq.T
    outs = {}
    for lever in ("1", "0"):
        monkeypatch.setenv("KQ_QMV_SPLITK", lever)
        got = kq.quantized_matmul(x, w, scales, "q6_k", transpose=True)
        mx.eval(got)
        outs[lever] = got
        err = float(np.abs(np.array(got.astype(mx.float32)) - r).max())
        assert err < 2e-2 * float(np.abs(r).max()), (
            f"N={n} K={k} KQ_QMV_SPLITK={lever}: max_abs={err:.3e}"
        )
    if n % 4:
        assert bool(mx.array_equal(outs["1"], outs["0"]).item())
    # The two kernels differ only in float32 summation order, so their
    # outputs agree to one unit in the last place.
    a = np.array(outs["1"].astype(mx.float32))
    b = np.array(outs["0"].astype(mx.float32))
    ulp = np.maximum(np.abs(a), np.abs(b)) * 2.0 ** -MANTISSA[dtype]
    assert np.all(np.abs(a - b) <= ulp + 1e-5 * float(np.abs(b).max()))


def _q6_k_block(ql, qh, scale, d):
    return np.concatenate(
        [
            np.full(128, ql, np.uint8),
            np.full(64, qh, np.uint8),
            np.full(16, scale, np.int8).view(np.uint8),
            np.array([d], np.float16).view(np.uint8),
        ]
    )


# The reduction order shows which kernel ran. With x all ones, superblocks
# 0 and 2 contribute +B and -B (B = 3937 * 2**23, exact in float32) and
# superblock 1 contributes 256. The per-row kernels cancel +B and -B inside
# each lane and return 256. The split-K kernel sums superblocks 0 and 1 in
# one simdgroup, where 256 is below half a unit of B, and superblock 2 in
# another, so it returns 0.
@pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or mx.default_device() != mx.gpu,
    reason="Metal kernel routing",
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_q6_k_qmv_splitk_routing(dtype, monkeypatch):
    _clear_qmv_levers(monkeypatch)
    row = np.concatenate(
        [
            _q6_k_block(0xFF, 0xFF, 127, 32768.0),
            _q6_k_block(0x11, 0xAA, 1, 1.0),
            _q6_k_block(0x11, 0x00, 127, 32768.0),
        ]
    )
    scales = mx.zeros((1,), dtype=mx.uint8)
    x = mx.ones((1, 768), dtype=dtype)

    def run(n, lever=None, fine=None):
        for name, v in (("KQ_QMV_SPLITK", lever), ("KQ_QMV_FINE", fine)):
            if v is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, v)
        w = mx.array(np.tile(row, (n, 1)))
        y = kq.quantized_matmul(x, w, scales, "q6_k", transpose=True)
        return np.array(y.astype(mx.float32)).ravel().tolist()

    assert run(4) == [0.0] * 4
    assert run(4, lever="1") == [0.0] * 4
    assert run(4, lever="0") == [256.0] * 4
    assert run(4, fine="1") == [256.0] * 4
    assert run(4, fine="0") == [256.0] * 4
    assert run(6) == [256.0] * 6
