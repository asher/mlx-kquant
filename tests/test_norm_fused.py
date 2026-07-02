"""Fused residual + RMSNorm glue ops vs mx.fast.rms_norm compositions.

References are computed in float32 (the kernels do all math in f32 and round
once at the write), then compared in the op's output dtype against both the
f32 truth and the stock mx.fast.rms_norm composition.
"""

import mlx.core as mx
import pytest

import mlx_kquant as kq

EPS = 1e-6

SHAPES = [
    (1, 2816),  # gemma-4-a4b decode row
    (4, 2816),  # small batch / MTP verify
    (3, 704),  # non-multiple-of-256 width
    (2, 1000),  # width not a multiple of the 256-thread group
    (1, 96),  # width below one loop pass
]

DTYPES = [mx.bfloat16, mx.float16]


def _tol(dtype):
    return 2e-2 if dtype == mx.bfloat16 else 5e-3


def _rel(got, ref):
    got = got.astype(mx.float32)
    denom = mx.abs(ref).max().item() + 1e-6
    return (mx.abs(got - ref).max().item()) / denom


def _rms_ref(x32, w32):
    inv = mx.rsqrt((x32 * x32).mean(axis=-1, keepdims=True) + EPS)
    return x32 * inv * w32


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_add_rmsnorm(dtype, shape):
    t, d = shape
    mx.random.seed(7)
    h = mx.random.normal((t, d)).astype(dtype)
    res = mx.random.normal((t, d)).astype(dtype)
    w = (1.0 + 0.1 * mx.random.normal((d,))).astype(dtype)

    ref = res.astype(mx.float32) + _rms_ref(h.astype(mx.float32), w.astype(mx.float32))
    got = kq.add_rmsnorm(h, res, w, EPS)
    assert got.dtype == dtype
    assert _rel(got, ref) < _tol(dtype)

    stock = res + mx.fast.rms_norm(h, w, EPS)
    assert _rel(got, stock.astype(mx.float32)) < _tol(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_add_rmsnorm_scale(dtype):
    t, d = 2, 2816
    mx.random.seed(11)
    h = mx.random.normal((t, d)).astype(dtype)
    res = mx.random.normal((t, d)).astype(dtype)
    w = (1.0 + 0.1 * mx.random.normal((d,))).astype(dtype)
    sc = mx.array([0.73]).astype(dtype)

    ref = (
        res.astype(mx.float32) + _rms_ref(h.astype(mx.float32), w.astype(mx.float32))
    ) * sc.astype(mx.float32)
    got = kq.add_rmsnorm(h, res, w, EPS, scale=sc)
    assert _rel(got, ref) < _tol(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_multi3(dtype, shape):
    t, d = shape
    mx.random.seed(13)
    x = mx.random.normal((t, d)).astype(dtype)
    ws = [(1.0 + 0.1 * mx.random.normal((d,))).astype(dtype) for _ in range(3)]

    outs = kq.rmsnorm_multi3(x, ws[0], ws[1], ws[2], EPS)
    x32 = x.astype(mx.float32)
    for got, w in zip(outs, ws, strict=True):
        assert got.dtype == dtype
        ref = _rms_ref(x32, w.astype(mx.float32))
        assert _rel(got, ref) < _tol(dtype)
        stock = mx.fast.rms_norm(x, w, EPS)
        assert _rel(got, stock.astype(mx.float32)) < _tol(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm2_add(dtype, shape):
    t, d = shape
    mx.random.seed(17)
    a = mx.random.normal((t, d)).astype(dtype)
    b = mx.random.normal((t, d)).astype(dtype)
    wa = (1.0 + 0.1 * mx.random.normal((d,))).astype(dtype)
    wb = (1.0 + 0.1 * mx.random.normal((d,))).astype(dtype)

    ref = _rms_ref(a.astype(mx.float32), wa.astype(mx.float32)) + _rms_ref(
        b.astype(mx.float32), wb.astype(mx.float32)
    )
    got = kq.rmsnorm2_add(a, wa, b, wb, EPS)
    assert got.dtype == dtype
    assert _rel(got, ref) < _tol(dtype)

    stock = mx.fast.rms_norm(a, wa, EPS) + mx.fast.rms_norm(b, wb, EPS)
    assert _rel(got, stock.astype(mx.float32)) < _tol(dtype)


def test_add_rmsnorm_3d_and_noncontiguous():
    mx.random.seed(19)
    h = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)
    res = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)
    w = (1.0 + 0.1 * mx.random.normal((512,))).astype(mx.bfloat16)

    ref = res.astype(mx.float32) + _rms_ref(h.astype(mx.float32), w.astype(mx.float32))
    got = kq.add_rmsnorm(h, res, w, EPS)
    assert got.shape == (2, 3, 512)
    assert _rel(got, ref) < _tol(mx.bfloat16)

    big = mx.random.normal((2, 6, 512)).astype(mx.bfloat16)
    hs = big[:, 1:4, :]  # sliced view, not row-contiguous
    mx.eval(hs)
    res_s = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)
    ref_s = res_s.astype(mx.float32) + _rms_ref(
        hs.astype(mx.float32), w.astype(mx.float32)
    )
    got_s = kq.add_rmsnorm(hs, res_s, w, EPS)
    assert _rel(got_s, ref_s) < _tol(mx.bfloat16)


def test_validation_errors():
    h = mx.zeros((2, 64), dtype=mx.bfloat16)
    w = mx.ones((64,), dtype=mx.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm(h.astype(mx.float32), h.astype(mx.float32), w, EPS)
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm(h, h, mx.ones((32,), dtype=mx.bfloat16), EPS)
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm(h, h, w.astype(mx.float32), EPS)
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm(h, h, w, EPS, scale=mx.ones((2,), dtype=mx.bfloat16))
    with pytest.raises((ValueError, RuntimeError)):
        kq.rmsnorm2_add(h, w, mx.zeros((2, 32), dtype=mx.bfloat16), w, EPS)
