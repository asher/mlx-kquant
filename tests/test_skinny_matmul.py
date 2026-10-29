"""skinny_matmul vs an f32 mx.matmul reference.

The kernel accumulates in f32 and rounds once at the write. The reference
runs on the CPU stream: the GPU f32 GEMM is TF32-by-default on M5-class
devices and would put ~1e-3 of error in the reference itself.
"""

import mlx.core as mx
import pytest

import mlx_kquant as kq

SHAPES = [
    # (M, K, N)
    (1, 4096, 256),  # router gate, decode width
    (4, 4096, 256),  # router gate, verify width
    (3, 4096, 64),  # indexer weights_proj, rows=2 verify
    (4, 16384, 24),  # hyper-connection mixes
    (16, 4096, 4),  # width cap, tiny N
    (2, 100, 7),  # K not a multiple of 128, odd N
]

COMBOS = [
    (mx.float16, mx.float16),
    (mx.bfloat16, mx.bfloat16),
    (mx.float32, mx.float32),
    (mx.float16, mx.float32),
    (mx.bfloat16, mx.float32),
    (mx.float32, mx.float16),
    (mx.float32, mx.bfloat16),
]


def _rel(got, ref):
    denom = mx.abs(ref).max().item() + 1e-6
    return mx.abs(got.astype(mx.float32) - ref).max().item() / denom


@pytest.mark.parametrize("xt,wt", COMBOS)
@pytest.mark.parametrize("M,K,N", SHAPES)
def test_skinny_matmul_parity(xt, wt, M, K, N):
    mx.random.seed(3)
    x = (mx.random.normal((2, M, K)) * 0.5).astype(xt)
    w = (mx.random.normal((N, K)) * 0.5).astype(wt)
    got = kq.skinny_matmul(x, w)
    with mx.stream(mx.cpu):
        ref = x.astype(mx.float32) @ w.astype(mx.float32).T
    mx.eval(got, ref)

    expect_dtype = mx.float32 if mx.float32 in (xt, wt) else xt
    assert got.dtype == expect_dtype
    assert got.shape == (2, M, N)
    tol = 3e-5 if expect_dtype == mx.float32 else 5e-3
    assert _rel(got, ref) < tol


@pytest.mark.parametrize("wt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("M", [1, 4, 16])
def test_skinny_matmul_f32_x_half_w_wide_n(wt, M):
    """float32 x against a float16 or bfloat16 weight at a vocab-wide N
    (an f32 residual against a half-precision head): float32 out."""
    K, N = 128, 131072
    mx.random.seed(11)
    x = mx.random.normal((M, K)) * 0.5
    w = (mx.random.normal((N, K)) * 0.5).astype(wt)
    got = kq.skinny_matmul(x, w)
    with mx.stream(mx.cpu):
        ref = x @ w.astype(mx.float32).T
    mx.eval(got, ref)
    assert got.dtype == mx.float32
    assert got.shape == (M, N)
    assert _rel(got, ref) < 3e-5


def test_skinny_matmul_noncontiguous_inputs():
    x = mx.random.normal((4, 4096)).astype(mx.float16).T.swapaxes(0, 1)
    w = mx.random.normal((4096, 64)).astype(mx.float16).T
    got = kq.skinny_matmul(x[None], w)
    with mx.stream(mx.cpu):
        ref = x[None].astype(mx.float32) @ w.astype(mx.float32).T
    mx.eval(got, ref)
    assert _rel(got, ref) < 5e-3


def test_skinny_matmul_rejects():
    x = mx.zeros((17, 4096), dtype=mx.float16)
    w = mx.zeros((8, 4096), dtype=mx.float16)
    with pytest.raises(ValueError):
        kq.skinny_matmul(x, w)  # M > 16
    with pytest.raises(ValueError):
        kq.skinny_matmul(
            mx.zeros((2, 4098), dtype=mx.float16),
            mx.zeros((8, 4098), dtype=mx.float16),
        )  # K % 4
    with pytest.raises(ValueError):
        kq.skinny_matmul(
            mx.zeros((2, 4096), dtype=mx.float16),
            mx.zeros((8, 4096), dtype=mx.bfloat16),
        )  # f16 x with bf16 w
    with pytest.raises(ValueError):
        kq.skinny_matmul(
            mx.zeros((2, 4096), dtype=mx.bfloat16),
            mx.zeros((8, 4096), dtype=mx.float16),
        )  # bf16 x with f16 w
    with pytest.raises(ValueError):
        kq.skinny_matmul(
            mx.zeros((2, 4096), dtype=mx.float32),
            mx.zeros((8, 4096), dtype=mx.int32),
        )  # f32 x with an integer w
