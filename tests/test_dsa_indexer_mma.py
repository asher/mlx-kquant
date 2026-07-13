#!/usr/bin/env python3
"""Tensor-op indexer arms: kq.dsa_indexer_scores (MMA dispatch),
kq.dsa_indexer_qat_quant and kq.dsa_indexer_scores_q.

Contracts:
  * qat-quant emit: codes * scales reproduces kq.dsa_indexer_qat(x)
    bit-exactly, except negatives snapped to zero re-dequantize as +0.0
    where the round-trip stores -0.0 (int8 has no signed zero;
    value-equal, and downstream scores are unaffected -- asserted here).
  * scores_q(codes, scales) is BIT-identical to dsa_indexer_scores on the
    dequantized codes: the int8 segment MMA accumulates in int32 (exact)
    and every rescaled product is a dyadic rational, so the fp32
    accumulation sees the same values in the same order.
  * the tensor-op f16 dispatch (default on NAX hardware) matches the fp32
    reference within the same bounds as the simdgroup kernel, and
    KQ_DISABLE_INDEXER_MMA=1 falls back to the simdgroup kernel at
    dispatch time (read per call, no rebuild).

Covers both weight layouts, both head counts (64 and 32), causal on/off,
explicit and derived q_offset, and the skip-store prefill wiring.

Tensor-op kernels are Metal-only and NAX-gated: skipped under
KQUANT_FORCE_CPU.

Usage: test_dsa_indexer_mma.py
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq DSA indexer kernels are Metal-only; no CPU path.",
)

REL_BOUND = {mx.bfloat16: 5e-3, mx.float16: 2e-3}

D = 128


def _make_qkw(B, H, M, N, dtype, weights_lh, seed):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((B, H, M, D)) * 0.4).astype(dtype)
    k = mx.array(rng.standard_normal((B, 1, N, D)) * 0.4).astype(dtype)
    wshape = (B, M, H) if weights_lh else (B, H, M, 1)
    w = mx.array(np.abs(rng.standard_normal(wshape)) * 0.2).astype(dtype)
    return q, k, w


def _ref_scores(q, k, w, causal, q_offset):
    qf = np.array(q.astype(mx.float32))
    kf = np.array(k.astype(mx.float32))
    wf = np.array(w.astype(mx.float32))
    B, H, M, _ = qf.shape
    N = kf.shape[2]
    if wf.ndim == 4:
        wf = wf[..., 0].transpose(0, 2, 1)
    off = q_offset if q_offset >= 0 else N - M
    out = np.zeros((B, 1, M, N), dtype=np.float32)
    for b in range(B):
        dots = np.einsum("hmd,nd->hmn", qf[b], kf[b, 0])
        out[b, 0] = np.einsum("hmn,mh->mn", np.maximum(dots, 0.0), wf[b])
        if causal:
            m_idx = np.arange(M)[:, None]
            n_idx = np.arange(N)[None, :]
            out[b, 0][n_idx > off + m_idx] = -np.inf
    return out


def _quantize(x):
    """QAT-emit wire form for a [B, H, L, 128] (or [B, 1, N, 128]) array."""
    codes, scales = kq.dsa_indexer_qat_quant(x.reshape(-1, D))
    shape = list(x.shape)
    return codes.reshape(x.shape), scales.reshape(shape[:-1] + [4])


def _dequantize(codes, scales):
    c = codes.astype(mx.float32)
    s = mx.repeat(scales, 32, axis=-1)
    return c * s


# ------------------------------------------------------- qat quant emit


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_dsa_indexer_qat_quant_bit_identity(dtype):
    rng = np.random.default_rng(5)
    x = mx.array(rng.standard_normal((3000, D)) * 0.7).astype(dtype)
    deq = kq.dsa_indexer_qat(x)
    codes, scales = kq.dsa_indexer_qat_quant(x)
    mx.eval(deq, codes, scales)
    assert codes.shape == x.shape and codes.dtype == mx.int8
    assert scales.shape == (3000, 4) and scales.dtype == mx.float32

    c = np.array(codes)
    assert c.min() >= -12 and c.max() <= 12
    redeq_mx = _dequantize(codes, scales).astype(dtype)
    if dtype == mx.bfloat16:
        # numpy has no bf16; the bf16 -> f32 injection is exact, so f32
        # bit patterns compare bf16 bits faithfully.
        redeq = np.array(redeq_mx.astype(mx.float32))
        d = np.array(deq.astype(mx.float32))
        width = np.uint32
    else:
        redeq = np.array(redeq_mx)
        d = np.array(deq)
        width = np.uint16 if dtype == mx.float16 else np.uint32
    bit_eq = redeq.view(width) == d.view(width)
    # Bit-exact except -0.0 (negatives snapped to zero), which is +0.0 in
    # the emitted form.
    assert np.all(bit_eq | ((redeq == 0) & (d == 0)))
    assert np.array_equal(redeq, d)  # value-equal everywhere


def test_dsa_indexer_qat_quant_rejects_bad_shapes():
    with pytest.raises(ValueError):
        kq.dsa_indexer_qat_quant(mx.zeros((8, 64), dtype=mx.float16))


# --------------------------------------------------------- scores_q arm

SCORE_CASES = [
    # (name, H, M, N, causal, q_offset, weights_lh)
    ("prefill_causal", 64, 128, 640, True, -1, True),
    ("prefill_offset", 64, 64, 640, True, 300, True),
    ("noncausal_lh", 64, 64, 256, False, -1, True),
    ("noncausal_hl", 64, 64, 256, False, -1, False),
    ("heads32", 32, 128, 320, True, -1, True),
    ("unaligned_pad", 64, 192, 448, True, -1, True),
]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("case", SCORE_CASES, ids=[c[0] for c in SCORE_CASES])
def test_dsa_indexer_scores_q_matches_dequant_scores(case, dtype):
    """scores_q(codes, scales) == scores(dequant(codes, scales)) bitwise."""
    name, H, M, N, causal, q_offset, weights_lh = case
    q, k, w = _make_qkw(1, H, M, N, dtype, weights_lh, seed=7)
    qd = kq.dsa_indexer_qat(q.reshape(-1, D)).reshape(q.shape)
    kd = kq.dsa_indexer_qat(k.reshape(-1, D)).reshape(k.shape)
    qc, qs = _quantize(q)
    kc, ks = _quantize(k)

    ref = kq.dsa_indexer_scores(qd, kd, w, causal=causal, causal_q_offset=q_offset)
    got = kq.dsa_indexer_scores_q(
        qc, qs, kc, ks, w, causal=causal, causal_q_offset=q_offset
    )
    mx.eval(ref, got)
    assert got.shape == (1, 1, M, N) and got.dtype == dtype

    a = np.array(ref.astype(mx.float32))
    b = np.array(got.astype(mx.float32))
    masked = np.isneginf(a)
    assert np.array_equal(masked, np.isneginf(b)), name
    assert np.array_equal(a[~masked], b[~masked]), f"{name}: not bit-identical"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dsa_indexer_scores_q_vs_fp32_reference(dtype):
    """Independent check against the numpy reference on dequantized values."""
    q, k, w = _make_qkw(1, 64, 128, 640, dtype, True, seed=11)
    qd = kq.dsa_indexer_qat(q.reshape(-1, D)).reshape(q.shape)
    kd = kq.dsa_indexer_qat(k.reshape(-1, D)).reshape(k.shape)
    qc, qs = _quantize(q)
    kc, ks = _quantize(k)
    got = kq.dsa_indexer_scores_q(qc, qs, kc, ks, w, causal=True)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    ref = _ref_scores(qd, kd, w, True, -1)
    masked = np.isneginf(ref)
    assert np.all(np.isneginf(g[masked]) | (g[masked] <= -60000))
    rel = np.linalg.norm(g[~masked] - ref[~masked]) / (
        np.linalg.norm(ref[~masked]) + 1e-6
    )
    assert rel < REL_BOUND[dtype], f"rel {rel:.3e}"


def test_dsa_indexer_scores_q_fp32_weights():
    """fp32 weights accepted; output falls back to fp16."""
    q, k, w = _make_qkw(1, 64, 64, 256, mx.float16, True, seed=13)
    qc, qs = _quantize(q)
    kc, ks = _quantize(k)
    got32 = kq.dsa_indexer_scores_q(qc, qs, kc, ks, w.astype(mx.float32), causal=False)
    got16 = kq.dsa_indexer_scores_q(qc, qs, kc, ks, w, causal=False)
    mx.eval(got32, got16)
    assert got32.dtype == mx.float16
    # fp16 weights are exact in fp32, so both runs see identical values.
    assert np.array_equal(np.array(got32), np.array(got16))


def test_dsa_indexer_scores_q_skip_store_e2e():
    """Production prefill wiring on the quant arm: skip-store + prefix topk."""
    M = N = 640
    topk = 512
    q, k, w = _make_qkw(1, 64, M, N, mx.float16, True, seed=17)
    qc, qs = _quantize(q)
    kc, ks = _quantize(k)
    scores = kq.dsa_indexer_scores_q(
        qc,
        qs,
        kc,
        ks,
        w,
        causal=True,
        unused_causal_prefix_topk=topk,
        skip_causal_future_store=True,
    )
    got = kq.dsa_topk_indices(scores, topk, bucketed=True, causal_valid_prefix=True)
    mx.eval(got)
    vals = np.array(scores.astype(mx.float32))
    sel = np.array(got)
    for m in range(0, M, 97):
        valid = m + 1
        row = vals[0, 0, m]
        if valid <= topk:
            expect = np.zeros(topk, dtype=np.uint32)
            expect[:valid] = np.arange(valid, dtype=np.uint32)
            np.testing.assert_array_equal(np.sort(sel[0, 0, m]), np.sort(expect))
            continue
        v = row[:valid]
        thr = np.partition(v, -topk)[-topk]
        assert np.all(v[sel[0, 0, m]] >= thr)


def test_dsa_indexer_scores_q_rejects_bad_inputs():
    qc = mx.zeros((1, 64, 64, D), dtype=mx.int8)
    qs = mx.zeros((1, 64, 64, 4), dtype=mx.float32)
    kc = mx.zeros((1, 1, 256, D), dtype=mx.int8)
    ks = mx.zeros((1, 1, 256, 4), dtype=mx.float32)
    w = mx.zeros((1, 64, 64), dtype=mx.float16)
    with pytest.raises(ValueError):  # M % 64 != 0
        kq.dsa_indexer_scores_q(qc[:, :, :32], qs[:, :, :32], kc, ks, w[:, :32])
    with pytest.raises(ValueError):  # wrong codes dtype
        kq.dsa_indexer_scores_q(qc.astype(mx.int32), qs, kc, ks, w)
    with pytest.raises(ValueError):  # scales shape mismatch
        kq.dsa_indexer_scores_q(qc, qs[:, :, :, :2], kc, ks, w)


# ------------------------------------------------- f16 dispatch fallback


def _score_case(dtype, seed):
    q, k, w = _make_qkw(1, 64, 128, 640, dtype, True, seed=seed)
    return q, k, w


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dsa_indexer_scores_mma_vs_simdgroup(dtype):
    """Both dispatch arms meet the fp32 reference bound; the env switch
    selects the simdgroup kernel at dispatch time."""
    q, k, w = _score_case(dtype, 19)
    ref = _ref_scores(q, k, w, True, -1)

    outs = {}
    for arm, env in (("mma", None), ("simdgroup", "1")):
        if env is None:
            os.environ.pop("KQ_DISABLE_INDEXER_MMA", None)
        else:
            os.environ["KQ_DISABLE_INDEXER_MMA"] = env
        try:
            got = kq.dsa_indexer_scores(q, k, w, causal=True)
            mx.eval(got)
        finally:
            os.environ.pop("KQ_DISABLE_INDEXER_MMA", None)
        g = np.array(got.astype(mx.float32))
        masked = np.isneginf(ref)
        assert np.all(np.isneginf(g[masked]) | (g[masked] <= -60000)), arm
        rel = np.linalg.norm(g[~masked] - ref[~masked]) / (
            np.linalg.norm(ref[~masked]) + 1e-6
        )
        assert rel < REL_BOUND[dtype], f"{arm} {dtype}: rel {rel:.3e}"
        outs[arm] = g

    # The two arms agree far more tightly than either agrees with the fp16
    # inputs' quantization noise floor (same math, different MMA tiling).
    masked = np.isneginf(ref)
    rel = np.linalg.norm(outs["mma"][~masked] - outs["simdgroup"][~masked]) / (
        np.linalg.norm(outs["simdgroup"][~masked]) + 1e-6
    )
    assert rel < REL_BOUND[dtype]


def test_dsa_indexer_scores_heads32_mma():
    q, k, w = _make_qkw(1, 32, 64, 320, mx.float16, True, seed=23)
    got = kq.dsa_indexer_scores(q, k, w, causal=False)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    ref = _ref_scores(q, k, w, False, -1)
    rel = np.linalg.norm(g - ref) / (np.linalg.norm(ref) + 1e-6)
    assert rel < REL_BOUND[mx.float16]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
