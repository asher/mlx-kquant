#!/usr/bin/env python3
"""kq.latent_fp4_pack / latent_fp4_unpack: rows on the DeepSeek-V4.1 latent
QAT grid (E4M3 scale per 16, E2M1 codes) round-trip bit-for-bit; rows off
the grid land on it the way the QAT projects them.

Metal-only kernels (eval_cpu throws): skipped under KQUANT_FORCE_CPU.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq.latent_fp4_pack is a Metal-only kernel; no CPU path.",
)

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def _e4m3_round(v):
    """Nearest E4M3FN value of v > 0 (ties to even), as the QAT does."""
    e = np.clip(np.floor(np.log2(np.maximum(v, 2.0**-9))), -6.0, 8.0)
    q = 2.0 ** (e - 3.0)
    return np.minimum(np.round(v / q) * q, 448.0).astype(np.float32)


def _e2m1_round(v):
    a = np.abs(v)
    q = np.where(
        a <= 0.25,
        0.0,
        np.where(
            a < 0.75,
            0.5,
            np.where(
                a <= 1.25,
                1.0,
                np.where(
                    a < 1.75,
                    1.5,
                    np.where(
                        a <= 2.5,
                        2.0,
                        np.where(a < 3.5, 3.0, np.where(a <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return (np.sign(v) * q).astype(np.float32)


def _latent_qat(x, dtype):
    """The V4.1 latent QAT in numpy: [..., D] fp32 -> on-grid rows in dtype."""
    v = x.reshape(*x.shape[:-1], -1, 16).astype(np.float32)
    amax = np.maximum(np.abs(v).max(-1, keepdims=True), 6.0 * 2.0**-9)
    scale = _e4m3_round(amax / 6.0)
    out = _e2m1_round(np.clip(v / scale, -6.0, 6.0)) * scale
    return mx.array(out.reshape(x.shape)).astype(dtype)


def _e4m3_decode(b):
    e = (b >> 3) & 15
    m = b & 7
    return np.where(e == 0, m * 2.0**-9, (1.0 + m / 8.0) * 2.0 ** (e - 7.0))


def _unpack_np(codes, scales):
    c = np.array(codes).astype(np.uint32)
    lo = c & 15
    hi = c >> 4
    nib = np.stack([lo, hi], -1).reshape(*c.shape[:-1], -1)
    mag = E2M1[nib & 7] * np.where(nib & 8, -1.0, 1.0)
    s = _e4m3_decode(np.array(scales).astype(np.uint32))
    s = np.repeat(s, 16, axis=-1)
    return (mag * s).astype(np.float32)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("D", [16, 128, 512])
def test_on_grid_rows_round_trip_exactly(dtype, D):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 97, D)) * rng.choice([0.01, 1.0, 40.0], (3, 97, 1))
    rows = _latent_qat(x, dtype)
    codes, scales = kq.latent_fp4_pack(rows)
    mx.eval(codes, scales)
    assert codes.dtype == mx.uint8 and scales.dtype == mx.uint8
    assert codes.shape == (3, 97, D // 2) and scales.shape == (3, 97, D // 16)
    back = kq.latent_fp4_unpack(codes, scales, dtype)
    mx.eval(back)
    assert back.dtype == dtype
    assert mx.array_equal(back, rows).item()
    # The numpy reading of the wire form agrees with the kernel's.
    ref = _unpack_np(codes, scales)
    assert np.array_equal(ref, np.array(rows.astype(mx.float32)))


def test_zero_and_tiny_groups():
    D = 64
    rows = mx.zeros((1, 5, D), mx.float16)
    codes, scales = kq.latent_fp4_pack(rows)
    mx.eval(codes, scales)
    assert np.all(np.array(codes) == 0)
    assert np.all(np.array(scales) == 1)  # 2^-9, the QAT floor scale
    back = kq.latent_fp4_unpack(codes, scales)
    assert mx.array_equal(back, rows).item()
    # A group whose scale sits in the E4M3 subnormal range.
    x = np.zeros((1, 1, D), np.float32)
    x[0, 0, :16] = np.linspace(-3, 3, 16) * 2.0**-9
    rows = _latent_qat(x, mx.float16)
    codes, scales = kq.latent_fp4_pack(rows)
    back = kq.latent_fp4_unpack(codes, scales)
    assert mx.array_equal(back, rows).item()


def test_off_grid_rows_take_the_qat_projection():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((2, 33, 128)) * 3.0).astype(np.float32)
    codes, scales = kq.latent_fp4_pack(mx.array(x))
    back = kq.latent_fp4_unpack(codes, scales, mx.float32)
    want = _latent_qat(x, mx.float32)
    mx.eval(back, want)
    assert mx.array_equal(back, want).item()


def test_pack_is_a_fixed_point():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((4, 64, 512)) * 2.0).astype(np.float32)
    c1, s1 = kq.latent_fp4_pack(mx.array(x).astype(mx.float16))
    once = kq.latent_fp4_unpack(c1, s1)
    c2, s2 = kq.latent_fp4_pack(once)
    twice = kq.latent_fp4_unpack(c2, s2)
    assert mx.array_equal(once, twice).item()


def test_validation():
    with pytest.raises(ValueError, match="multiple of 16"):
        kq.latent_fp4_pack(mx.zeros((2, 24), mx.float16))
    with pytest.raises(ValueError, match="fp16/bf16/fp32"):
        kq.latent_fp4_pack(mx.zeros((2, 32), mx.int32))
    codes = mx.zeros((2, 16), mx.uint8)
    with pytest.raises(ValueError, match="uint8"):
        kq.latent_fp4_unpack(codes, mx.zeros((2, 2), mx.int8))
    with pytest.raises(ValueError, match="scales"):
        kq.latent_fp4_unpack(codes, mx.zeros((2, 3), mx.uint8))
    with pytest.raises(ValueError, match="dtype"):
        kq.latent_fp4_unpack(codes, mx.zeros((2, 2), mx.uint8), mx.int32)
