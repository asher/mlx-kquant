"""Fused residual + RMSNorm glue ops vs mx.fast.rms_norm compositions.

References are computed in float32 (the kernels do all math in f32 and round
once at the write), then compared in the op's output dtype against both the
f32 truth and the stock mx.fast.rms_norm composition. add_rmsnorm_norm rounds
like the unfused ops instead and is compared to their composition exactly.
"""

import os
import tempfile

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


ADD_NORM_WIDTHS = [
    96,  # below one simdgroup of 4-wide reads
    98,  # row not a multiple of 4
    704,
    1000,  # partial last simdgroup
    2816,  # gemma-4-a4b hidden
    4096,  # widest row-sized threadgroup
    4097,  # first 1024-thread width
    4100,  # partial second chunk
    5376,  # gemma-4-31B hidden
    8192,
    10002,  # three chunks, row not a multiple of 4
    16384,  # widest instantiated chunk count
    16400,  # past it: the op builds the unfused ops
]


def _add_norm_ref(h, res, w, wn, eps, scale=None, next_eps=None):
    out = res + mx.fast.rms_norm(h, w, eps)
    if scale is not None:
        out = out * scale
    return out, mx.fast.rms_norm(out, wn, eps if next_eps is None else next_eps)


def _add_norm_operands(t, d, dtype, seed):
    mx.random.seed(seed)
    h = (3.0 * mx.random.normal((t, 1, d))).astype(dtype)
    res = (20.0 * mx.random.normal((t, 1, d))).astype(dtype)
    w = (1.0 + 0.3 * mx.random.normal((d,))).astype(dtype)
    wn = (1.0 + 0.3 * mx.random.normal((d,))).astype(dtype)
    return h, res, w, wn


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", ADD_NORM_WIDTHS)
@pytest.mark.parametrize("scaled", [False, True])
def test_add_rmsnorm_norm_bit_exact(dtype, d, scaled):
    sc = mx.array([0.71]).astype(dtype) if scaled else None
    for t in (1, 3):
        h, res, w, wn = _add_norm_operands(t, d, dtype, seed=d + t)
        got = kq.add_rmsnorm_norm(h, res, w, wn, EPS, scale=sc)
        ref = _add_norm_ref(h, res, w, wn, EPS, scale=sc)
        for g, r in zip(got, ref, strict=True):
            assert g.shape == h.shape and g.dtype == dtype
            assert mx.array_equal(g, r).item()


def test_add_rmsnorm_norm_next_eps_and_views():
    d = 5376
    h, res, w, wn = _add_norm_operands(2, d, mx.bfloat16, seed=23)
    got = kq.add_rmsnorm_norm(h, res, w, wn, EPS, next_eps=1e-5)
    ref = _add_norm_ref(h, res, w, wn, EPS, next_eps=1e-5)
    assert all(mx.array_equal(g, r).item() for g, r in zip(got, ref, strict=True))

    # A lazy transposed view as h: never evaluated before the op reads it.
    ht = (3.0 * mx.random.normal((d, 2))).astype(mx.bfloat16).T
    rt = res.reshape(2, d)
    got = kq.add_rmsnorm_norm(ht, rt, w, wn, EPS)
    ref = _add_norm_ref(ht, rt, w, wn, EPS)
    assert all(mx.array_equal(g, r).item() for g, r in zip(got, ref, strict=True))


@pytest.mark.parametrize("stream", [None, mx.cpu], ids=["default", "cpu"])
def test_add_rmsnorm_norm_scale_keeps_h_shape(stream):
    h, res, w, wn = _add_norm_operands(2, 704, mx.bfloat16, seed=37)
    h, res = h.reshape(2, 704), res.reshape(2, 704)
    sc = mx.array([[[0.6]]]).astype(mx.bfloat16)
    got = kq.add_rmsnorm_norm(h, res, w, wn, EPS, scale=sc, stream=stream)
    ref = _add_norm_ref(h, res, w, wn, EPS, scale=sc.reshape(()))
    for g, r in zip(got, ref, strict=True):
        assert g.shape == h.shape
        assert mx.array_equal(g, r).item()


def test_add_rmsnorm_norm_cpu_stream():
    h, res, w, wn = _add_norm_operands(3, 2816, mx.bfloat16, seed=29)
    sc = mx.array([0.5]).astype(mx.bfloat16)
    got = kq.add_rmsnorm_norm(h, res, w, wn, EPS, scale=sc, stream=mx.cpu)
    with mx.stream(mx.cpu):
        ref = _add_norm_ref(h, res, w, wn, EPS, scale=sc)
    assert all(mx.array_equal(g, r).item() for g, r in zip(got, ref, strict=True))


@pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="the fused kernel is Metal-only; CPU streams build the unfused ops.",
)
@pytest.mark.parametrize("d", [2816, 5376, 16384])
def test_add_rmsnorm_norm_uses_kernel(d):
    h, res, w, wn = _add_norm_operands(1, d, mx.bfloat16, seed=31)
    out = kq.add_rmsnorm_norm(h, res, w, wn, EPS, scale=mx.array([1.0], mx.bfloat16))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "g.dot")
        mx.export_to_dot(path, *out)
        dot = open(path).read()
    assert "KQuantAddRMSNormNorm" in dot
    assert "RMSNorm" not in dot.replace("KQuantAddRMSNormNorm", "")


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
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm_norm(h, h, w, mx.ones((32,), dtype=mx.bfloat16), EPS)
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm_norm(h, h, w, w.astype(mx.float32), EPS)
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm_norm(h, h[:, :32], w, w, EPS)
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm_norm(h, h, w, w, EPS, scale=mx.ones((2,), dtype=mx.bfloat16))
    with pytest.raises((ValueError, RuntimeError)):
        kq.add_rmsnorm_norm(h, h, w, w, EPS, scale=mx.ones((1,), dtype=mx.float32))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", [(1, 40, 64, 128), (2, 3, 4, 64), (5, 256), (1, 7, 128)]
)
def test_rmsnorm_gate(dtype, shape):
    mx.random.seed(11)
    d = shape[-1]
    x = mx.random.normal(shape).astype(dtype)
    g = mx.random.normal(shape).astype(dtype)
    w = (1.0 + 0.1 * mx.random.normal((d,))).astype(dtype)
    ref = _rms_ref(x.astype(mx.float32), w.astype(mx.float32)) * mx.sigmoid(
        g.astype(mx.float32)
    )
    got = kq.rmsnorm_gate(x, w, g, EPS)
    assert got.shape == shape and got.dtype == dtype
    assert _rel(got, ref) < _tol(dtype)
    stock = mx.fast.rms_norm(x, w, EPS) * mx.sigmoid(g)
    assert _rel(got, stock.astype(mx.float32)) < _tol(dtype)


def test_rmsnorm_gate_rejects_bad_args():
    x = mx.random.normal((3, 128)).astype(mx.bfloat16)
    w = mx.ones((128,), dtype=mx.bfloat16)
    with pytest.raises(ValueError):
        kq.rmsnorm_gate(x, w, x[:, :64], EPS)
    with pytest.raises(ValueError):
        kq.rmsnorm_gate(x[:, :96], w[:96], x[:, :96], EPS)
