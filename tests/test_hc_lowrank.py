"""Fused low-rank hyper-connection ops vs the eager op-chain reference.

The reference reproduces the gmlx qwen4exp HyperConnection eager path on
dequantized weights, including its rounding points (the down/up qmv
outputs round to the half dtype before their activations). The kernels
accumulate in f32 with different reduction orders, so comparisons carry a
half-grid tolerance rather than bit-exactness.

Metal-only kernels (eval_cpu throws): skipped under KQUANT_FORCE_CPU.

Usage: test_hc_lowrank.py
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType, quants

import mlx_kquant as kq

HC = 4
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="hc lowrank ops are Metal-only kernels; no CPU path.",
)


def _wire(w):
    """f32 [N, K] -> (q8_0 wire uint8 [N, K // 32 * 34], dequantized f32)."""
    n, k = w.shape
    raw = quants.quantize(w.astype(np.float32), GGMLQuantizationType.Q8_0)
    packed = np.ascontiguousarray(raw, dtype=np.uint8).reshape(n, k // 32 * 34)
    deq = quants.dequantize(raw, GGMLQuantizationType.Q8_0).reshape(n, k)
    return mx.array(packed), mx.array(deq.astype(np.float32))


def _mk(seed, R, D, LR, h_dtype, half_dtype):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    h = (mx.random.normal((R, HC, D)) * 1.5).astype(h_dtype)
    if h_dtype != mx.float32:
        h = h.astype(h_dtype)
    gamma = (mx.random.normal((HC * D,)) * 0.3 + 1.0).astype(half_dtype)
    wd, wd_deq = _wire(rng.standard_normal((LR, HC * D)) * 0.05)
    wu, wu_deq = _wire(rng.standard_normal((HC * D, LR)) * 0.2)
    wi = mx.array(rng.standard_normal((HC, HC * D)).astype(np.float32) * 0.02)
    return h, gamma, wd, wd_deq, wu, wu_deq, wi


def _ref(h, gamma, wd_deq, wu_deq, wi, half_dtype):
    """Eager chain on dequantized weights, mirroring gmlx rounding points."""
    R, _, D = h.shape
    xn = mx.fast.rms_norm(h, None, EPS) * gamma.reshape(HC, D)
    xf = xn.reshape(R, HC * D)
    down = (xf.astype(mx.float32) @ wd_deq.T).astype(half_dtype)
    t = down * 0.25
    lo = t * mx.sigmoid(t)
    up = (lo.astype(mx.float32) @ wu_deq.T).astype(half_dtype)
    gate = mx.sigmoid(up).reshape(R, HC, D)
    mixed = (gate.astype(h.dtype) * xn).mean(axis=1)
    inj = 2.0 * mx.sigmoid((xf.astype(mx.float32) @ wi.T) * 0.25)
    return lo, mixed, inj, xn


CASES = [
    (1, 256, 64, mx.float32, mx.bfloat16),
    (4, 256, 64, mx.float32, mx.bfloat16),
    (1, 256, 64, mx.bfloat16, mx.bfloat16),
    (1, 256, 64, mx.float32, mx.float16),
    (1, 256, 64, mx.float16, mx.float16),
    (2, 2560, 320, mx.float32, mx.bfloat16),  # production geometry
]


@pytest.mark.parametrize("R,D,LR,h_dtype,half_dtype", CASES)
def test_hc_lowrank_front(R, D, LR, h_dtype, half_dtype):
    h, gamma, wd, wd_deq, wu, wu_deq, wi = _mk(3, R, D, LR, h_dtype, half_dtype)
    ref_lo, _, ref_inj, ref_xn = _ref(h, gamma, wd_deq, wu_deq, wi, half_dtype)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    assert xn.dtype == h.dtype and xn.shape == (R, HC, D)
    lo, inj = kq.hc_lowrank_front(xn, wd, wi, half_dtype)
    assert lo.dtype == half_dtype and lo.shape == (R, LR)
    assert inj.dtype == mx.float32 and inj.shape == (R, HC)
    np.testing.assert_allclose(
        np.array(inj),
        np.array(ref_inj),
        rtol=5e-4,
        atol=5e-5,
    )
    # lo rounds to the half grid in both paths; different f32 reduction
    # orders can move a value one ulp.
    np.testing.assert_allclose(
        np.array(lo.astype(mx.float32)),
        np.array(ref_lo.astype(mx.float32)),
        rtol=2e-2 if half_dtype == mx.bfloat16 else 3e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("R,D,LR,h_dtype,half_dtype", CASES)
def test_hc_lowrank_epilogue(R, D, LR, h_dtype, half_dtype):
    h, gamma, wd, wd_deq, wu, wu_deq, wi = _mk(7, R, D, LR, h_dtype, half_dtype)
    ref_lo, ref_mixed, _, ref_xn = _ref(h, gamma, wd_deq, wu_deq, wi, half_dtype)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    lo, inj = kq.hc_lowrank_front(xn, wd, wi, half_dtype)
    mixed = kq.hc_lowrank_epilogue(lo, wu, xn)
    assert mixed.dtype == h.dtype and mixed.shape == (R, D)
    # Both paths round the up dot to the half grid before the sigmoid
    # gate; different f32 reduction orders (and the reference's TF32 GEMM)
    # flip near-boundary roundings, each worth up to one gate ulp times
    # xn / 4. Bound the tail and the bulk separately.
    d = np.abs(
        np.array(mixed.astype(mx.float32)) - np.array(ref_mixed.astype(mx.float32))
    )
    scale = float(np.abs(np.array(ref_mixed.astype(mx.float32))).mean())
    assert float(d.max()) < 1e-2 * max(scale, 1.0)
    assert float(np.quantile(d, 0.99)) < 5e-3 * max(scale, 1.0)
    assert float(d.mean()) < 1.5e-3 * max(scale, 1.0)


def test_hc_lowrank_leading_dims():
    """[B, T, 4, D] leading shape round-trips like [R, 4, D]."""
    h, gamma, wd, wd_deq, wu, wu_deq, wi = _mk(11, 4, 256, 64, mx.float32, mx.bfloat16)
    h4 = h.reshape(2, 2, HC, 256)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    xn4 = kq.hc_lowrank_norm(h4, gamma, EPS)
    assert xn4.shape == (2, 2, HC, 256)
    assert mx.array_equal(xn4.reshape(4, HC, 256), xn)
    lo, inj = kq.hc_lowrank_front(xn, wd, wi, mx.bfloat16)
    lo4, inj4 = kq.hc_lowrank_front(xn4, wd, wi, mx.bfloat16)
    assert lo4.shape == (2, 2, 64) and inj4.shape == (2, 2, HC)
    assert mx.array_equal(lo4.reshape(4, 64), lo)
    mixed = kq.hc_lowrank_epilogue(lo, wu, xn)
    mixed4 = kq.hc_lowrank_epilogue(lo4, wu, xn4)
    assert mixed4.shape == (2, 2, 256)
    assert mx.array_equal(mixed4.reshape(4, 256), mixed)


def test_hc_lowrank_bad_shapes():
    h, gamma, wd, _, wu, _, wi = _mk(13, 1, 256, 64, mx.float32, mx.bfloat16)
    with pytest.raises(ValueError):
        kq.hc_lowrank_norm(h[:, :3], gamma, EPS)  # 3 streams
    with pytest.raises(ValueError):
        kq.hc_lowrank_norm(h, gamma[:-1], EPS)  # short gamma
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    with pytest.raises(ValueError):
        kq.hc_lowrank_front(xn, wd[:, :-1], wi, mx.bfloat16)  # bad wire
    with pytest.raises(ValueError):
        kq.hc_lowrank_front(xn, wd, wi[:3], mx.bfloat16)  # 3 inject rows
    with pytest.raises(ValueError):
        kq.hc_lowrank_front(xn, wd, wi, mx.float32)  # non-half lo dtype
    lo, _ = kq.hc_lowrank_front(xn, wd, wi, mx.bfloat16)
    with pytest.raises(ValueError):
        kq.hc_lowrank_epilogue(lo[:, :-2], wu, xn)  # LR mismatch
