"""Fused hyper-connection glue ops vs float64 numpy references.

The kernels accumulate in f32 and round once at each write, so references
are computed in f64 and compared with an f32-reduction tolerance. The
fused front_expand_reduce is specified bit-identical to hc_expand followed
by hc_front_reduce and is checked exactly against that composition.

Metal-only kernels (eval_cpu throws): skipped under KQUANT_FORCE_CPU.

Usage: test_hc_glue.py
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

HC = 4
MIX = 24
ITERS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="hc glue ops are Metal-only kernels; no CPU path.",
)


def _mk(seed, D=2048, dtype=mx.bfloat16):
    mx.random.seed(seed)
    x = (mx.random.normal((1, 1, HC, D)) * 0.05).astype(dtype)
    fn = (mx.random.normal((MIX, HC * D)) * 0.02).astype(mx.float32)
    scale = mx.array([1.1, 0.9, 1.3], dtype=mx.float32)
    base = (mx.random.normal((MIX,)) * 0.1).astype(mx.float32)
    w = (mx.random.normal((D,)) * 0.1 + 1.0).astype(dtype)
    mx.eval(x, fn, scale, base, w)
    return x, fn, scale, base, w


def _np64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _sinkhorn_ref(mix, ssq, scale, base, D):
    factor = 1.0 / np.sqrt(ssq / (HC * D) + NORM_EPS)
    pre = 1.0 / (1.0 + np.exp(-(mix[:HC] * scale[0] * factor + base[:HC])))
    pre = pre + HC_EPS
    post = 2.0 / (
        1.0 + np.exp(-(mix[HC : 2 * HC] * scale[1] * factor + base[HC : 2 * HC]))
    )
    v = (mix[2 * HC :] * scale[2] * factor + base[2 * HC :]).reshape(HC, HC)
    e = np.exp(v - v.max(axis=1, keepdims=True))
    r = e / (e.sum(axis=1, keepdims=True) + HC_EPS) + HC_EPS
    r = r / (r.sum(axis=0, keepdims=True) + HC_EPS)
    for _ in range(1, ITERS):
        r = r / (r.sum(axis=1, keepdims=True) + HC_EPS)
        r = r / (r.sum(axis=0, keepdims=True) + HC_EPS)
    return pre, post, r


@pytest.mark.parametrize("D", [2048, 4096])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_front_reduce_and_collapse(D, dtype):
    x, fn, scale, base, w = _mk(7, D, dtype)
    mr, ssq = kq.hc_front_reduce(x, fn)
    col, post, comb = kq.hc_sinkhorn_collapse(
        x, mr, ssq, scale, base, w, iters=ITERS, hc_eps=HC_EPS, norm_eps=NORM_EPS
    )
    mx.eval(mr, ssq, col, post, comb)

    xf = _np64(x).reshape(HC * D)
    fnf = _np64(fn)
    mr_ref = fnf @ xf
    ssq_ref = float(xf @ xf)
    assert np.abs(np.array(mr).ravel() - mr_ref).max() < 1e-3
    assert abs(float(ssq.item()) - ssq_ref) / ssq_ref < 1e-5

    pre_ref, post_ref, comb_ref = _sinkhorn_ref(
        mr_ref, ssq_ref, _np64(scale), _np64(base), D
    )
    assert np.abs(np.array(post).ravel() - post_ref).max() < 1e-4
    assert np.abs(np.array(comb).reshape(HC, HC) - comb_ref).max() < 5e-4

    xs = _np64(x).reshape(HC, D)
    collapsed = pre_ref @ xs
    inv = 1.0 / np.sqrt((collapsed * collapsed).mean() + NORM_EPS)
    col_ref = collapsed * inv * _np64(w)
    tol = 2e-2 if dtype == mx.bfloat16 else 5e-3
    denom = np.abs(col_ref).max() + 1e-6
    assert np.abs(_np64(col).ravel() - col_ref).max() / denom < tol


@pytest.mark.parametrize("D", [2048, 4096])
def test_expand(D):
    x, fn, scale, base, w = _mk(11, D)
    mr, ssq = kq.hc_front_reduce(x, fn)
    col, post, comb = kq.hc_sinkhorn_collapse(
        x, mr, ssq, scale, base, w, iters=ITERS, hc_eps=HC_EPS, norm_eps=NORM_EPS
    )
    out = kq.hc_expand(col, x, post, comb)
    mx.eval(out)

    xf = _np64(col).reshape(D)
    rf = _np64(x).reshape(HC, D)
    pf = np.array(post).ravel().astype(np.float64)
    cf = np.array(comb).reshape(HC, HC).astype(np.float64)
    ref = pf[:, None] * xf[None, :] + cf.T @ rf
    denom = np.abs(ref).max() + 1e-6
    assert np.abs(_np64(out).reshape(HC, D) - ref).max() / denom < 2e-2


@pytest.mark.parametrize("D", [2048, 4096])
def test_front_expand_reduce_matches_composition(D):
    # D a multiple of 1024 keeps the fused kernel's reduction order
    # identical to hc_front_reduce, which the bit-exact claim needs.
    x, fn, scale, base, w = _mk(13, D)
    mr, ssq = kq.hc_front_reduce(x, fn)
    col, post, comb = kq.hc_sinkhorn_collapse(
        x, mr, ssq, scale, base, w, iters=ITERS, hc_eps=HC_EPS, norm_eps=NORM_EPS
    )

    h_ref = kq.hc_expand(col, x, post, comb)
    mr_ref, ssq_ref = kq.hc_front_reduce(h_ref, fn)
    h, mr2, ssq2 = kq.hc_front_expand_reduce(col, x, post, comb, fn)
    mx.eval(h_ref, mr_ref, ssq_ref, h, mr2, ssq2)

    assert np.array_equal(
        np.array(h.astype(mx.float32)), np.array(h_ref.astype(mx.float32))
    )
    assert np.array_equal(np.array(mr2), np.array(mr_ref))
    assert np.array_equal(np.array(ssq2), np.array(ssq_ref))


def test_input_validation():
    x, fn, scale, base, w = _mk(17, 2048)
    with pytest.raises(ValueError):
        kq.hc_front_reduce(x[..., :2, :], fn)
    with pytest.raises(ValueError):
        kq.hc_front_reduce(x, fn[:, :100])
    with pytest.raises(ValueError):
        kq.hc_sinkhorn_collapse(
            x,
            mx.zeros((1, 1, MIX)),
            mx.zeros((1, 1, 1)),
            scale,
            base,
            w.astype(mx.float32),
            iters=ITERS,
            hc_eps=HC_EPS,
            norm_eps=NORM_EPS,
        )
