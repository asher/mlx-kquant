"""kda_conv: fused causal short conv (silu, optional per-head l2 norm with a
folded scale, tail state) against the eager composition in fp32.

The kernel is a plain simdgroup kernel (no tensor-op requirement); under
KQUANT_FORCE_CPU=1 the same cases run the CPU path.
"""

import mlx.core as mx
import pytest

import mlx_kquant as kq

EPS = 1e-6
REL_BOUND = {mx.bfloat16: 1.5e-2, mx.float16: 5e-3}


def _rel(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    return float(mx.linalg.norm(af - bf) / (mx.linalg.norm(bf) + 1e-9))


def _ref(x, state, w, D, scale):
    """Eager composition in fp32: concat, depthwise conv, silu, rms per head."""
    B, T, C = x.shape
    K = w.shape[1]
    xin = mx.concatenate([state, x], axis=1).astype(mx.float32)
    w32 = w.reshape(w.shape[0], K).astype(mx.float32)
    y = mx.zeros((B, T, C), dtype=mx.float32)
    for j in range(K):
        y = y + w32[:, j] * xin[:, j : j + T]
    y = y * mx.sigmoid(y)
    if scale != 0.0:
        yh = y.reshape(B, T, C // D, D)
        yh = scale * yh * mx.rsqrt((yh * yh).mean(-1, keepdims=True) + EPS)
        y = yh.reshape(B, T, C)
    tail = xin[:, T:]
    return y, tail


def _make(B, T, C, K, dtype, seed):
    key = mx.random.key(seed)
    k0, k1, k2 = mx.random.split(key, 3)
    x = mx.random.normal((B, T, C), key=k0).astype(dtype)
    state = mx.random.normal((B, K - 1, C), key=k1).astype(dtype)
    w = (mx.random.normal((C, K, 1), key=k2) * 0.5).astype(dtype)
    mx.eval(x, state, w)
    return x, state, w


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 512, 4, 128, 0.5),  # glm5 shape class, normed q/k
        (1, 64, 512, 4, 128, 0.0),  # v: no norm
        (2, 37, 256, 4, 64, 2.0),  # batched, odd T, head 64
        (1, 5, 256, 3, 256, 1.0),  # 3 taps, head 256
        (1, 1, 384, 4, 128, 0.7),  # decode-shaped T = 1
        (2, 2, 256, 4, 128, 0.0),  # T shorter than the carried rows
        (1, 9, 128, 8, 128, 1.0),  # the widest tap count
    ],
)
def test_kda_conv_matches_eager(shape, dtype):
    B, T, C, K, D, scale = shape
    x, state, w = _make(B, T, C, K, dtype, seed=T * 7 + C)
    y, tail = kq.kda_conv(x, state, w, D, scale, EPS)
    y_ref, tail_ref = _ref(x, state, w, D, scale)
    mx.eval(y, tail, y_ref, tail_ref)
    assert y.shape == (B, T, C) and y.dtype == dtype
    assert tail.shape == (B, K - 1, C) and tail.dtype == dtype
    assert _rel(y, y_ref) < REL_BOUND[dtype], f"{shape} y rel {_rel(y, y_ref):.3e}"
    assert _rel(tail, tail_ref) == 0.0


def test_kda_conv_matches_conv1d_chain():
    # Against the stock mlx chain (concat, conv1d, silu, rms_norm, scale).
    B, T, C, K, D, scale = 1, 48, 1024, 4, 128, 0.25
    x, state, w = _make(B, T, C, K, mx.bfloat16, seed=3)
    y, _ = kq.kda_conv(x, state, w, D, scale, EPS)
    xin = mx.concatenate([state, x], axis=1)
    yc = mx.conv1d(xin, w, 1, 0, 1, C)
    yc = yc * mx.sigmoid(yc)
    yc = scale * mx.fast.rms_norm(yc.reshape(B, T, C // D, D), None, EPS)
    yc = yc.reshape(B, T, C)
    mx.eval(y, yc)
    assert _rel(y, yc) < REL_BOUND[mx.bfloat16]


def test_kda_conv_chaining_is_transparent():
    # Two calls with the tail threaded through equal one call over the whole
    # sequence.
    B, C, K, D = 1, 256, 4, 128
    x, state, w = _make(B, 40, C, K, mx.bfloat16, seed=9)
    y_all, tail_all = kq.kda_conv(x, state, w, D, 1.0, EPS)
    y0, t0 = kq.kda_conv(x[:, :17], state, w, D, 1.0, EPS)
    y1, t1 = kq.kda_conv(x[:, 17:], t0, w, D, 1.0, EPS)
    y_split = mx.concatenate([y0, y1], axis=1)
    mx.eval(y_all, tail_all, y_split, t1)
    assert _rel(y_split, y_all) == 0.0
    assert _rel(t1, tail_all) == 0.0


def test_kda_conv_rejects_bad_args():
    x, state, w = _make(1, 8, 256, 4, mx.bfloat16, seed=1)
    with pytest.raises(ValueError):
        kq.kda_conv(x, state[:, :2], w, 128, 1.0, EPS)
    with pytest.raises(ValueError):
        kq.kda_conv(x, state, w, 96, 1.0, EPS)
    with pytest.raises(ValueError):
        kq.kda_conv(x, state, w[:128], 128, 1.0, EPS)
