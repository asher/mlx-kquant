#!/usr/bin/env python3
"""kq.sdpa_sparse_decode against an independent f32 reference.

The reference concatenates the window rows and the listed pool rows per
query, drops masked and out-of-range entries, and runs a plain f32 softmax
with the per-head sink in the denominator. Covers both float dtypes, both
index dtypes, head dims 128/256/512, L in {1, 2, 4}, B = 2, masks, padded
index slots, a forced split count and the automatic one.

Metal-only kernel (eval_cpu throws): skipped under KQUANT_FORCE_CPU.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq.sdpa_sparse_decode is a Metal-only kernel; no CPU path.",
)


def _ref(q, win, pool, idx, scale, sinks, wmask, smask):
    """f32 numpy reference: q [B,H,L,D], win [B,1,W,D], pool [B,P,D],
    idx [B,L,N]; masks [B|1, L, X] bool or None."""
    B, H, L, D = q.shape
    W, P, N = win.shape[2], pool.shape[1], idx.shape[2]
    out = np.zeros((B, H, L, D), np.float32)
    for b in range(B):
        for l in range(L):
            rows = []
            for j in range(W):
                if wmask is not None and not wmask[b % wmask.shape[0], l, j]:
                    continue
                rows.append(win[b, 0, j])
            for n in range(N):
                if smask is not None and not smask[b % smask.shape[0], l, n]:
                    continue
                r = int(idx[b, l, n])
                if r < 0 or r >= P:
                    continue
                rows.append(pool[b, r])
            keys = np.stack(rows, 0)  # [M, D]
            s = (q[b, :, l] * scale) @ keys.T  # [H, M]
            if sinks is not None:
                s = np.concatenate([sinks[:, None], s], 1)
            s = s - s.max(1, keepdims=True)
            w = np.exp(s)
            w = w / w.sum(1, keepdims=True)
            if sinks is not None:
                w = w[:, 1:]
            out[b, :, l] = w @ keys
    return out


def _case(B, H, L, D, W, P, N, dtype, idtype, sinks, masks, pad, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, L, D)).astype(np.float32)
    win = rng.standard_normal((B, 1, W, D)).astype(np.float32)
    pool = rng.standard_normal((B, P, D)).astype(np.float32)
    idx = (
        np.stack([rng.permutation(P)[:N] for _ in range(B * L)], 0)
        .reshape(B, L, N)
        .astype(np.int64)
    )
    if pad:
        idx[..., -3:] = np.array([-1, P, P + 7]) if idtype == np.int32 else P
    idx = idx.astype(idtype)
    sk = rng.standard_normal(H).astype(np.float32) if sinks else None
    wm = rng.random((1, L, W)) > 0.2 if masks else None
    sm = rng.random((B, L, N)) > 0.3 if masks else None
    if masks:
        wm[:, :, :4] = True  # keep at least a few rows live
    qm = mx.array(q).astype(dtype)
    winm = mx.array(win).astype(dtype)
    poolm = mx.array(pool).astype(dtype)
    # the kernel reads the rounded inputs; the reference must too
    qf = np.array(qm.astype(mx.float32))
    winf = np.array(winm.astype(mx.float32))
    poolf = np.array(poolm.astype(mx.float32))
    skm = mx.array(sk).astype(dtype) if sinks else None
    skf = np.array(skm.astype(mx.float32)) if sinks else None
    scale = D**-0.5
    ref = _ref(qf, winf, poolf, idx, scale, skf, wm, sm)
    kw = {}
    if sinks:
        kw["sinks"] = skm
    if masks:
        kw["win_mask"] = mx.array(wm[0])  # [L, W] form
        kw["sel_mask"] = mx.array(sm)  # [B, L, N] form
    return (qm, winm, poolm, mx.array(idx), scale, kw), ref


def _check(args, ref, tol, **extra):
    q, win, pool, idx, scale, kw = args
    got = kq.sdpa_sparse_decode(q, win, pool, idx, scale, **kw, **extra)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    assert g.shape == ref.shape
    err = float(np.abs(g - ref).max())
    assert err < tol, err


@pytest.mark.parametrize("dtype,tol", [(mx.float16, 3e-3), (mx.bfloat16, 2e-2)])
@pytest.mark.parametrize("idtype", [np.int32, np.uint32])
@pytest.mark.parametrize("D", [128, 256, 512])
def test_decode_matches_reference(dtype, tol, idtype, D):
    args, ref = _case(
        1, 64, 1, D, 128, 700, 512, dtype, idtype, sinks=True, masks=False, pad=False
    )
    _check(args, ref, tol)


@pytest.mark.parametrize("L", [1, 2, 4])
@pytest.mark.parametrize("splits", [0, 1, 7, 32])
def test_masks_padding_batch_and_splits(L, splits):
    args, ref = _case(
        2,
        12,
        L,
        128,
        40,
        96,
        24,
        mx.float16,
        np.int32,
        sinks=True,
        masks=True,
        pad=True,
        seed=L + splits,
    )
    _check(args, ref, 3e-3, splits=splits)


def test_no_sinks_and_window_only():
    args, ref = _case(
        1,
        8,
        1,
        256,
        64,
        16,
        0,
        mx.float16,
        np.int32,
        sinks=False,
        masks=False,
        pad=False,
    )
    q, win, pool, idx, scale, kw = args
    got = kq.sdpa_sparse_decode(q, win, pool, idx, scale)
    mx.eval(got)
    assert float(np.abs(np.array(got.astype(mx.float32)) - ref).max()) < 3e-3


@pytest.mark.parametrize("L", [64, 200])
def test_prefill_block_widths(L):
    args, ref = _case(
        1,
        16,
        L,
        128,
        L + 7,
        96,
        24,
        mx.float16,
        np.int32,
        sinks=True,
        masks=True,
        pad=True,
        seed=L,
    )
    _check(args, ref, 3e-3)


def test_validation():
    q = mx.zeros((1, 4, 1, 128), mx.float16)
    win = mx.zeros((1, 1, 8, 128), mx.float16)
    pool = mx.zeros((1, 8, 128), mx.float16)
    with pytest.raises(ValueError, match="4096"):
        kq.sdpa_sparse_decode(
            mx.zeros((1, 4, 4097, 128), mx.float16),
            win,
            pool,
            mx.zeros((1, 4097, 4), mx.int32),
            0.1,
        )
    idx = mx.zeros((1, 1, 4), mx.int32)
    with pytest.raises(ValueError, match="head_dim"):
        kq.sdpa_sparse_decode(
            mx.zeros((1, 4, 1, 64), mx.float16),
            mx.zeros((1, 1, 8, 64), mx.float16),
            mx.zeros((1, 8, 64), mx.float16),
            idx,
            0.1,
        )
    with pytest.raises(ValueError, match="int32 or uint32"):
        kq.sdpa_sparse_decode(q, win, pool, idx.astype(mx.int64), 0.1)
    with pytest.raises(ValueError, match="sinks"):
        kq.sdpa_sparse_decode(q, win, pool, idx, 0.1, sinks=mx.zeros((3,)))
    with pytest.raises(ValueError, match="win_mask"):
        kq.sdpa_sparse_decode(
            q, win, pool, idx, 0.1, win_mask=mx.zeros((1, 7), mx.bool_)
        )
    with pytest.raises(ValueError, match="splits"):
        kq.sdpa_sparse_decode(q, win, pool, idx, 0.1, splits=40)
