"""qk_norm_rope against a float64 reference.

The reference normalizes and rotates in float64 on the CPU. The op does its
math in float32 and rounds once, so its output should equal the reference
rounded to the output dtype, within 1 ulp on nearly every element, and be no
less accurate than the unfused composition, which rounds the normalized
value to the input dtype before the rotation.

Inputs come in the layouts a caller passes: packed, sliced from one fused
qkv output, lazily transposed, misaligned for vector loads by the start, the
head stride or the row stride, and with a non-unit head-dim stride.
"""

from __future__ import annotations

import os
import re
import tempfile

import mlx.core as mx
import numpy as np
import pytest
from qmm_probe import force_cpu

import mlx_kquant as kq

EPS = 1e-6
DTYPES = [mx.bfloat16, mx.float16]
KERNEL_DIMS = [64, 96, 128, 256]
# Views that start aligned and have an odd head or row stride, and a packed
# view that starts one element off. Each misses the vector width in one place.
ODD = ["hstride", "lstride", "offset"]
LAYOUTS = ["packed", "qkv", "transposed", "misaligned", "dstride"] + ODD


def _metal():
    if force_cpu():
        return False
    try:
        return kq.metallib_loads()
    except Exception:
        return False


def _tables(L, D, seed):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-60.0, 60.0, (L, D // 2)).astype(np.float32)
    return mx.array(np.cos(ang)), mx.array(np.sin(ang))


def _weights(D, dtype, seed):
    rng = np.random.default_rng(seed)
    w = (1.0 + 0.2 * rng.standard_normal((2, D))).astype(np.float32)
    return mx.array(w[0]).astype(dtype), mx.array(w[1]).astype(dtype)


def _odd_view(x, layout):
    """A view of the values of x [B, L, H, D] in one of the ODD layouts."""
    B, L, H, D = x.shape
    if layout == "hstride":
        return mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1)])[..., :D]
    if layout == "lstride":
        wide = mx.pad(x.reshape(B, L, H * D), [(0, 0), (0, 0), (0, 1)])
        return wide[..., : H * D].reshape(B, L, H, D)
    if layout == "offset":
        return mx.pad(x.reshape(-1), (1, 0))[1:].reshape(B, L, H, D)
    raise ValueError(layout)


def _inputs(B, L, H, D, dtype, layout, seed=0, scale=None):
    """q and k [B, L, H, D] in the given layout, evaluated. Standard normal
    values, or with `scale` magnitudes uniform in [scale / 2, scale] with
    random signs."""
    rng = np.random.default_rng(seed)

    def draw(shape):
        if scale is None:
            x = rng.standard_normal(shape)
        else:
            x = scale * rng.uniform(0.5, 1.0, shape) * rng.choice([-1.0, 1.0], shape)
        return mx.array(x.astype(np.float32)).astype(dtype)

    if layout == "packed":
        q, k = draw((B, L, H, D)), draw((B, L, H, D))
    elif layout == "qkv":
        qkv = draw((B, L, 3 * H * D)).reshape(B, L, 3, H, D)
        q, k = qkv[:, :, 0], qkv[:, :, 1]
    elif layout == "transposed":
        q = draw((B, H, L, D)).transpose(0, 2, 1, 3)
        k = draw((B, H, L, D)).transpose(0, 2, 1, 3)
    elif layout == "misaligned":
        wide = draw((B, L, 2 * H * D + 2))
        q = wide[..., 1 : 1 + H * D].reshape(B, L, H, D)
        k = wide[..., 1 + H * D : 1 + 2 * H * D].reshape(B, L, H, D)
    elif layout == "dstride":
        q = draw((B, L, D, H)).transpose(0, 1, 3, 2)
        k = draw((B, L, D, H)).transpose(0, 1, 3, 2)
    elif layout in ODD:
        q = _odd_view(draw((B, L, H, D)), layout)
        k = _odd_view(draw((B, L, H, D)), layout)
    else:
        raise ValueError(layout)
    mx.eval(q, k)
    return q, k


def _ref(x, w, cos, sin, eps=EPS):
    """float64 rms_norm and interleaved-pair rotation, as [B, H, L, D]."""
    x = np.asarray(x.astype(mx.float32), dtype=np.float64)
    w = np.asarray(w.astype(mx.float32), dtype=np.float64)
    c = np.asarray(cos, dtype=np.float64)[None, :, None, :]
    s = np.asarray(sin, dtype=np.float64)[None, :, None, :]
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
    o = np.empty_like(y)
    o[..., 0::2] = y[..., 0::2] * c - y[..., 1::2] * s
    o[..., 1::2] = y[..., 0::2] * s + y[..., 1::2] * c
    return o.transpose(0, 2, 1, 3)


def _round(ref, dtype):
    """ref rounded once, to nearest even, to dtype, as float64."""
    if dtype == mx.float16:
        return ref.astype(np.float16).astype(np.float64)
    m, e = np.frexp(ref)
    return np.ldexp(np.rint(np.ldexp(m, 8)), e - 8)


def _bits(a):
    b = a.astype(np.int32)
    return np.where(b < 0, -(b & 0x7FFF), b)


def _ulps(got, ref, dtype):
    """Per-element ulp distance of got to ref rounded to dtype."""
    r = mx.array(_round(ref, dtype).astype(np.float32)).astype(dtype)
    g = np.asarray(got.view(mx.int16))
    return np.abs(_bits(g) - _bits(np.asarray(r.view(mx.int16))))


def _composed(q, k, q_w, k_w, cos, sin, eps=EPS):
    """The unfused composition: mx.fast.rms_norm, then the rotation in f32."""

    def rope(x):
        b, n, h, d = x.shape
        x = x.astype(mx.float32).reshape(b, n, h, d // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        c = cos[None, :, None, :]
        s = sin[None, :, None, :]
        return mx.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1).reshape(b, n, h, d)

    dt = q.dtype
    qo = rope(mx.fast.rms_norm(q, q_w, eps)).astype(dt)
    ko = rope(mx.fast.rms_norm(k, k_w, eps)).astype(dt)
    return qo.transpose(0, 2, 1, 3), ko.transpose(0, 2, 1, 3)


def _graph(*outs):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.dot")
        mx.export_to_dot(path, *outs)
        with open(path) as f:
            return f.read()


def _check(q, k, q_w, k_w, cos, sin, against_composed=True, eps=EPS):
    """Run the op, check shape, dtype and accuracy, return the outputs. On
    the GPU a kernel head dim must build the fused primitive, and any other
    head dim the composition."""
    B, L, H, D = q.shape
    qo, ko = kq.qk_norm_rope(q, k, q_w, k_w, cos, sin, eps)
    fused = _metal() and D in KERNEL_DIMS
    assert ("KQuantQkNormRope" in _graph(qo, ko)) == fused
    mx.eval(qo, ko)
    assert qo.shape == (B, H, L, D) and ko.shape == (B, H, L, D)
    assert qo.dtype == q.dtype and ko.dtype == q.dtype
    refs = (_ref(q, q_w, cos, sin, eps), _ref(k, k_w, cos, sin, eps))
    comp = _composed(q, k, q_w, k_w, cos, sin, eps) if against_composed else None
    for i, (got, ref) in enumerate(zip((qo, ko), refs, strict=True)):
        g = np.asarray(got.astype(mx.float32), dtype=np.float64)
        assert np.isfinite(g).all()
        u = _ulps(got, ref, q.dtype)
        assert (u > 1).mean() <= 1e-4, np.bincount(np.minimum(u, 8).ravel())
        if comp is not None:
            c = np.asarray(comp[i].astype(mx.float32), dtype=np.float64)
            assert np.abs(g - ref).max() <= np.abs(c - ref).max()
            assert np.abs(g - ref).mean() <= np.abs(c - ref).mean()
    return qo, ko


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("B", [1, 2])
def test_layouts(dtype, layout, B):
    L, H, D = 77, 3, 128
    q, k = _inputs(B, L, H, D, dtype, layout, seed=B)
    q_w, k_w = _weights(D, dtype, 1)
    cos, sin = _tables(L, D, 2)
    _check(q, k, q_w, k_w, cos, sin)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", KERNEL_DIMS + [32, 80])
def test_head_dims(dtype, D):
    # 32 and 80 have no kernel and run the composition on every stream.
    B, L, H = 2, 45, 5
    q, k = _inputs(B, L, H, D, dtype, "qkv", seed=D)
    q_w, k_w = _weights(D, dtype, 3)
    cos, sin = _tables(L, D, 4)
    _check(q, k, q_w, k_w, cos, sin)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B,L", [(1, 33000), (2, 16896)])
def test_rows_past_int16(dtype, B, L):
    # 2 x 16896 is a batch of two 2048-pixel images flattened to rows.
    H, D = 2, 128
    q, k = _inputs(B, L, H, D, dtype, "qkv", seed=L)
    q_w, k_w = _weights(D, dtype, 5)
    cos, sin = _tables(L, D, 6)
    _check(q, k, q_w, k_w, cos, sin, against_composed=False)


@pytest.mark.parametrize("dtype", DTYPES)
def test_zimage_shape(dtype):
    # One Z-Image main-layer call at 1024 pixels: 4096 image rows and 32
    # caption rows, 30 heads of 128, q and k sliced from the fused qkv.
    B, L, H, D = 1, 4128, 30, 128
    q, k = _inputs(B, L, H, D, dtype, "qkv", seed=9)
    q_w, k_w = _weights(D, dtype, 7)
    cos, sin = _tables(L, D, 8)
    _check(q, k, q_w, k_w, cos, sin)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [1e4, 6e4])
def test_large_activations(dtype, scale):
    # Every square is past the float16 range and must not overflow: the mean
    # square is accumulated in f32 and the output is back at the weight
    # scale.
    B, L, H, D = 2, 33, 4, 128
    q, k = _inputs(B, L, H, D, dtype, "qkv", seed=11, scale=scale)
    q_w, k_w = _weights(D, dtype, 12)
    cos, sin = _tables(L, D, 13)
    _check(q, k, q_w, k_w, cos, sin, against_composed=False)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "scale,eps,zero_row", [(1e-3, 1e-5, False), (None, 0.5, False), (None, 1e-5, True)]
)
def test_eps(dtype, scale, eps, zero_row):
    # At inputs of 1e-3 the mean square is below the Z-Image eps of 1e-5, and
    # eps 0.5 is half the mean square of unit inputs, so in both eps changes
    # every output. A head row of zeros comes out as zeros, since eps keeps
    # its rsqrt finite.
    B, L, H, D = 2, 21, 3, 128
    q, k = _inputs(B, L, H, D, dtype, "packed", seed=19, scale=scale)
    if zero_row:
        q[1, 4, 2] = 0
        k[0, 7, 0] = 0
        mx.eval(q, k)
    q_w, k_w = _weights(D, dtype, 20)
    cos, sin = _tables(L, D, 21)
    qo, ko = _check(q, k, q_w, k_w, cos, sin, against_composed=False, eps=eps)
    if zero_row:
        assert (np.asarray(qo[1, 2, 4].astype(mx.float32)) == 0).all()
        assert (np.asarray(ko[0, 0, 7].astype(mx.float32)) == 0).all()


def test_float32_weights_and_tables():
    # Weights and tables in other float dtypes are used in f32.
    B, L, H, D = 1, 20, 2, 64
    q, k = _inputs(B, L, H, D, mx.bfloat16, "qkv", seed=14)
    q_w, k_w = _weights(D, mx.float32, 15)
    cos, sin = _tables(L, D, 16)
    _check(q, k, q_w, k_w, cos, sin, against_composed=False)
    qo, ko = kq.qk_norm_rope(
        q, k, q_w, k_w, cos.astype(mx.float16), sin.astype(mx.float16), EPS
    )
    ref = _ref(q, q_w, cos.astype(mx.float16), sin.astype(mx.float16))
    assert (_ulps(qo, ref, mx.bfloat16) > 1).mean() <= 1e-4


def test_empty_rows():
    q = mx.zeros((1, 0, 2, 64), dtype=mx.bfloat16)
    w = mx.ones((64,), dtype=mx.bfloat16)
    t = mx.zeros((0, 32))
    qo, ko = kq.qk_norm_rope(q, q, w, w, t, t, EPS)
    mx.eval(qo, ko)
    assert qo.shape == (1, 2, 0, 64) and ko.shape == (1, 2, 0, 64)


def test_rejects_bad_args():
    q = mx.zeros((1, 4, 2, 64), dtype=mx.bfloat16)
    w = mx.ones((64,), dtype=mx.bfloat16)
    c = mx.zeros((4, 32))
    with pytest.raises(ValueError):
        kq.qk_norm_rope(q[..., :63], q[..., :63], w[:63], w[:63], c, c, EPS)
    with pytest.raises(ValueError):
        kq.qk_norm_rope(q, q[:, :3], w, w, c, c, EPS)
    with pytest.raises(ValueError):
        kq.qk_norm_rope(q, q.astype(mx.float16), w, w, c, c, EPS)
    with pytest.raises(ValueError):
        kq.qk_norm_rope(q.astype(mx.float32), q.astype(mx.float32), w, w, c, c, EPS)
    with pytest.raises(ValueError):
        kq.qk_norm_rope(q[0], q[0], w, w, c, c, EPS)
    with pytest.raises(ValueError):
        kq.qk_norm_rope(q, q, w[:32], w, c, c, EPS)
    with pytest.raises(ValueError):
        kq.qk_norm_rope(q, q, w, w, c[:3], c, EPS)
    with pytest.raises(ValueError):
        kq.qk_norm_rope(q, q, w, w, c, c[:, :16], EPS)


def _kernel_names():
    return sorted(n for n in kq.metallib_kernels() if n.startswith("kq_qk_norm_rope_"))


@pytest.mark.skipif(not _metal(), reason="Metal kernel names")
def test_kernel_set():
    want = sorted(
        f"kq_qk_norm_rope_{t}_{d}"
        for t in ("bfloat16_t", "float16_t")
        for d in KERNEL_DIMS
    )
    assert _kernel_names() == want


@pytest.mark.skipif(not _metal(), reason="Metal kernel names")
@pytest.mark.parametrize("name", _kernel_names() if _metal() else [])
def test_every_kernel(name):
    # Each kernel runs both load paths: vector loads on the fused qkv slice,
    # element loads on a misaligned view and on each ODD view. All read the
    # same values and must agree bit for bit.
    m = re.fullmatch(r"kq_qk_norm_rope_(bfloat16|float16)_t_(\d+)", name)
    assert m, name
    dtype = getattr(mx, m.group(1))
    D = int(m.group(2))
    B, L, H = 2, 19, 3
    q_w, k_w = _weights(D, dtype, 17)
    cos, sin = _tables(L, D, 18)
    qa, ka = _inputs(B, L, H, D, dtype, "qkv", seed=D)
    qb = mx.contiguous(qa)
    wide = mx.concatenate(
        [
            mx.zeros((B, L, 1), dtype=dtype),
            mx.concatenate([qa, ka], axis=2).reshape(B, L, 2 * H * D),
            mx.zeros((B, L, 1), dtype=dtype),
        ],
        axis=-1,
    )
    qm = wide[..., 1 : 1 + H * D].reshape(B, L, H, D)
    km = wide[..., 1 + H * D : 1 + 2 * H * D].reshape(B, L, H, D)
    mx.eval(qb, qm, km)
    vec = _check(qa, ka, q_w, k_w, cos, sin)
    strided = _check(qm, km, q_w, k_w, cos, sin)
    packed = _check(qb, ka, q_w, k_w, cos, sin)
    for a, b in zip(vec, strided, strict=True):
        assert mx.array_equal(a, b).item()
    assert mx.array_equal(vec[0], packed[0]).item()
    for layout in ODD:
        qv, kv = _odd_view(qa, layout), _odd_view(ka, layout)
        mx.eval(qv, kv)
        for a, b in zip(vec, _check(qv, kv, q_w, k_w, cos, sin), strict=True):
            assert mx.array_equal(a, b).item()
