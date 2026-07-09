#!/usr/bin/env python3
"""kq.dsa_indexer_scores / kq.dsa_topk_indices validation.

Scores: independent f32 numpy reference of the lightning-indexer math
(sum_h relu(q_h . k_n) * w_h_m), both weight layouts, causal on/off,
explicit and derived (-1) q_offset. Masked -inf entries are compared
exactly; finite entries by relative norm.

Top-k: the radix select is checked with an exact tie-aware criterion on the
16-bit score values themselves -- every selected index must score >= the
topk-th largest value, and every index scoring strictly above it must be
selected. That is the full correctness contract (intra-row order is
unspecified); a plain sorted-reference overlap check would flag legitimate
threshold ties. The causal_valid_prefix arm covers both the identity-prefix
fast path (valid <= topk) and the clamped radix path in one [1,1,L,K] grid,
with poison values planted beyond each row's causal horizon.

The E2E case chains dsa_indexer_scores(skip_causal_future_store=True,
unused_causal_prefix_topk=topk) into dsa_topk_indices(causal_valid_prefix=
True) -- the production prefill wiring, where unwritten score tiles must
never influence the selection.

Metal-only kernels: skipped under KQUANT_FORCE_CPU.

Usage: test_dsa_indexer.py
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

D = 128  # indexer head dim (fixed by the kernel contract)


# ---------------------------------------------------------------- scores


def _ref_scores(q, k, w, causal, q_offset):
    """f32 reference on the exact (already dtype-cast) input values."""
    qf = np.array(q.astype(mx.float32))
    kf = np.array(k.astype(mx.float32))
    wf = np.array(w.astype(mx.float32))
    B, H, M, _ = qf.shape
    N = kf.shape[2]
    if wf.ndim == 4:  # [B, H, M, 1] -> [B, M, H]
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


def _make_qkw(B, H, M, N, dtype, weights_lh, seed):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((B, H, M, D)) * 0.4).astype(dtype)
    k = mx.array(rng.standard_normal((B, 1, N, D)) * 0.4).astype(dtype)
    wshape = (B, M, H) if weights_lh else (B, H, M, 1)
    w = mx.array(np.abs(rng.standard_normal(wshape)) * 0.2).astype(dtype)
    return q, k, w


SCORE_CASES = [
    # (name, M, N, causal, q_offset, weights_lh)
    ("prefill_causal", 128, 640, True, -1, True),
    ("prefill_offset", 64, 640, True, 300, True),
    ("noncausal_lh", 64, 256, False, -1, True),
    ("noncausal_hl", 64, 256, False, -1, False),
    ("decode_pad64", 64, 640, False, -1, True),
]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("case", SCORE_CASES, ids=[c[0] for c in SCORE_CASES])
def test_dsa_indexer_scores(case, dtype):
    name, M, N, causal, q_offset, weights_lh = case
    q, k, w = _make_qkw(1, 64, M, N, dtype, weights_lh, seed=3)
    if name == "decode_pad64":
        # The decode-v1 shape: one real query row broadcast to fill M=64.
        q = mx.broadcast_to(q[:, :, :1], q.shape)
        w = mx.broadcast_to(w[:, :1], w.shape)
    got = kq.dsa_indexer_scores(q, k, w, causal=causal, causal_q_offset=q_offset)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    ref = _ref_scores(q, k, w, causal, q_offset)

    masked = np.isneginf(ref)
    assert np.all(np.isneginf(g[masked]) | (g[masked] <= -60000)), name
    rel = np.linalg.norm(g[~masked] - ref[~masked]) / (
        np.linalg.norm(ref[~masked]) + 1e-6
    )
    assert rel < REL_BOUND[dtype], f"{name} {dtype}: rel {rel:.3e}"

    if name == "decode_pad64":
        # All 64 padded rows share one query: identical scores per row.
        assert np.allclose(g[0, 0], g[0, 0, 0][None, :], atol=0), name


# ----------------------------------------------------------------- top-k


def _check_topk_row(vals, sel, topk, valid):
    """Exact tie-aware contract for one row.

    vals: f32-exact copy of the row's 16-bit scores; sel: kernel indices;
    valid: causal scan limit (len(vals) when unclamped).
    """
    if valid <= topk:
        expect = np.zeros(topk, dtype=np.uint32)
        expect[:valid] = np.arange(valid, dtype=np.uint32)
        np.testing.assert_array_equal(np.sort(sel), np.sort(expect))
        return
    assert np.all(sel < valid), "selected beyond the causal horizon"
    assert len(np.unique(sel)) == topk, "duplicate selections"
    v = vals[:valid]
    thr = np.partition(v, -topk)[-topk]
    assert np.all(v[sel] >= thr), "selected below the top-k threshold"
    above = np.flatnonzero(v > thr)
    assert np.isin(above, sel).all(), "missed a strictly-above-threshold idx"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bucketed", [False, True], ids=["scan", "bucketed"])
def test_dsa_topk_indices_decode(dtype, bucketed):
    rng = np.random.default_rng(11)
    K, topk = 1500, 512
    # Heavy ties: quantized normal values so the threshold bucket is fat.
    scores = mx.array(np.round(rng.standard_normal((1, 1, 1, K)) * 8) / 8).astype(dtype)
    got = kq.dsa_topk_indices(scores, topk, bucketed=bucketed)
    mx.eval(got)
    assert got.shape == (1, 1, 1, topk) and got.dtype == mx.uint32
    vals = np.array(scores.astype(mx.float32)).reshape(-1)
    _check_topk_row(vals, np.array(got).reshape(-1), topk, K)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dsa_topk_indices_causal_prefix(dtype):
    rng = np.random.default_rng(13)
    B, L, K, topk = 1, 640, 640, 512
    raw = rng.standard_normal((B, 1, L, K))
    scores = mx.array(raw).astype(dtype)
    got = kq.dsa_topk_indices(scores, topk, bucketed=True, causal_valid_prefix=True)
    mx.eval(got)
    vals = np.array(scores.astype(mx.float32))
    sel = np.array(got)
    for q in range(L):
        valid = min(K, max(0, K - L + q + 1))
        row = vals[0, 0, q].copy()
        row[valid:] = 100.0  # poison: must never be selected
        _check_topk_row(row, sel[0, 0, q], topk, valid)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dsa_indexer_e2e_prefill(dtype):
    """Production wiring: skip-store scores -> causal-prefix top-k."""
    M, N, topk = 640, 640, 512
    q, k, w = _make_qkw(1, 64, M, N, dtype, weights_lh=True, seed=17)
    scores = kq.dsa_indexer_scores(
        q,
        k,
        w,
        causal=True,
        unused_causal_prefix_topk=topk,
        skip_causal_future_store=True,
    )
    got = kq.dsa_topk_indices(scores, topk, bucketed=True, causal_valid_prefix=True)
    mx.eval(got)
    # Judge against the kernel's own 16-bit scores; each row's causal scan
    # range [0, m] is always written (only never-read tiles skip-store).
    vals = np.array(scores.astype(mx.float32))
    sel = np.array(got)
    for m in range(M):
        valid = m + 1  # q_offset = N - M = 0
        _check_topk_row(vals[0, 0, m], sel[0, 0, m], topk, valid)


def test_dsa_indexer_rejects_bad_shapes():
    q = mx.zeros((1, 64, 64, 128), dtype=mx.float16)
    k = mx.zeros((1, 1, 256, 128), dtype=mx.float16)
    w = mx.zeros((1, 64, 64), dtype=mx.float16)
    with pytest.raises(ValueError):  # M % 64 != 0
        kq.dsa_indexer_scores(q[:, :, :63], k, w[:, :63])
    with pytest.raises(ValueError):  # bad head count
        kq.dsa_indexer_scores(q[:, :16], k, w[:, :, :16])
    scores = mx.zeros((1, 1, 1, 1024), dtype=mx.float16)
    with pytest.raises(ValueError):  # unsupported topk
        kq.dsa_topk_indices(scores, 100)
    with pytest.raises(ValueError):  # K < topk
        kq.dsa_topk_indices(scores[..., :256], 512)


# --------------------------------------------------------- decode scores


def _ref_scores_decode(q, k, w, q_offset, ratio):
    """f32 reference of the fused decode score with make_mask visibility.

    Masked entries carry the dtype's finite min, matching the kernel."""
    qf = np.array(q.astype(mx.float32))
    kf = np.array(k.astype(mx.float32))
    wf = np.array(w.astype(mx.float32))
    B, _, QL, _ = qf.shape
    P = kf.shape[1]
    dots = np.einsum("bhjd,bpd->bhjp", qf, kf)
    out = np.einsum("bjh,bhjp->bjp", wf, np.maximum(dots, 0.0))[:, None]
    mask = np.zeros(out.shape, dtype=bool)
    if QL > 1:
        for j in range(QL):
            vlim = min(P, (q_offset + j + 1) // ratio)
            mask[:, :, j, vlim:] = True
    return out, mask


def _make_decode_qkw(B, QL, P, dtype, seed):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((B, 64, QL, D)) * 0.4).astype(dtype)
    k = mx.array(rng.standard_normal((B, P, D)) * 0.4).astype(dtype)
    # Signed weights: relu is on the dot, not the weighted sum.
    w = mx.array(rng.standard_normal((B, QL, 64)) * 0.2).astype(dtype)
    return q, k, w


DECODE_SCORE_CASES = [
    # (name, QL, P, q_offset, ratio)
    ("decode_ql1", 1, 1024, 4095, 4),
    ("verify_ql2_min_p", 2, 512, 2047, 4),
    ("verify_ql3_ragged", 3, 1027, 4106, 4),
    ("verify_ql4_clipped", 4, 517, 2066, 4),
    ("tiny_pool", 1, 33, 131, 4),
    ("deep_pool", 2, 16384, 65535, 4),
]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "case", DECODE_SCORE_CASES, ids=[c[0] for c in DECODE_SCORE_CASES]
)
def test_dsa_indexer_score_decode(case, dtype):
    name, QL, P, q_offset, ratio = case
    q, k, w = _make_decode_qkw(1, QL, P, dtype, seed=23)
    got = kq.dsa_indexer_score_decode(q, k, w, q_offset, ratio)
    mx.eval(got)
    assert got.shape == (1, 1, QL, P)
    g = np.array(got.astype(mx.float32))
    ref, mask = _ref_scores_decode(q, k, w, q_offset, ratio)
    if mask.any():
        assert np.all(g[mask] <= -60000), name
    rel = np.linalg.norm(g[~mask] - ref[~mask]) / (np.linalg.norm(ref[~mask]) + 1e-6)
    assert rel < REL_BOUND[dtype], f"{name} {dtype}: rel {rel:.3e}"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dsa_indexer_e2e_decode(dtype):
    """Production decode wiring: fused scores -> bucketed top-k."""
    QL, P, topk = 3, 4096, 512
    q, k, w = _make_decode_qkw(1, QL, P, dtype, seed=29)
    q_offset, ratio = P * 4 - QL, 4
    scores = kq.dsa_indexer_score_decode(q, k, w, q_offset, ratio)
    got = kq.dsa_topk_indices(scores, topk, bucketed=True)
    mx.eval(got)
    assert got.shape == (1, 1, QL, topk)
    vals = np.array(scores.astype(mx.float32))
    sel = np.array(got)
    for j in range(QL):
        _check_topk_row(vals[0, 0, j], sel[0, 0, j], topk, P)


def test_dsa_indexer_score_decode_rejects_bad_shapes():
    q = mx.zeros((1, 64, 1, 128), dtype=mx.float16)
    k = mx.zeros((1, 256, 128), dtype=mx.float16)
    w = mx.zeros((1, 1, 64), dtype=mx.float16)
    with pytest.raises(ValueError):  # qL > 4
        kq.dsa_indexer_score_decode(
            mx.zeros((1, 64, 5, 128), dtype=mx.float16),
            k,
            mx.zeros((1, 5, 64), dtype=mx.float16),
            100,
            4,
        )
    with pytest.raises(ValueError):  # bad head count
        kq.dsa_indexer_score_decode(q[:, :32], k, w[:, :, :32], 100, 4)
    with pytest.raises(ValueError):  # keys rank
        kq.dsa_indexer_score_decode(q, k[:, None], w, 100, 4)
    with pytest.raises(ValueError):  # negative q_offset
        kq.dsa_indexer_score_decode(q, k, w, -1, 4)


def main() -> int:
    rc = pytest.main([__file__, "-q"])
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
