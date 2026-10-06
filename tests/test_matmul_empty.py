"""quantized_matmul and quantized_matmul_glu on empty inputs.

No rows or no output columns give an empty array of the product's shape,
and an inner dim of 0 gives zeros. The cases run on the default device, so
the suite covers the GPU and, under KQUANT_FORCE_CPU, the CPU stream, at a
few rows (the mat-vec routes) and at 1100 rows (the matmul routes).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import mlx_kquant as kq

N, K = 96, 768
SCALES = mx.zeros((1,), dtype=mx.uint8)


def _wire(codec):
    w, _ = kq.quantize(mx.random.normal((N, K), key=mx.random.key(1)) * 0.1, codec)
    mx.eval(w)
    return w


def _ops(codec):
    return {
        "matmul": lambda x, w: kq.quantized_matmul(x, w, SCALES, codec, True),
        "glu": lambda x, w: kq.quantized_matmul_glu(x, w, w, codec),
    }


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("op", ["matmul", "glu"])
@pytest.mark.parametrize("codec", ["q4_k", "q8_0", "q3_k"])
def test_empty_inputs(codec, op, dtype):
    f = _ops(codec)[op]
    w = _wire(codec)
    # No rows, alone and under a batch dim.
    for shape in ([0, K], [2, 0, K], [0, 3, K]):
        y = f(mx.zeros(shape, dtype=dtype), w)
        mx.eval(y)
        assert y.shape == (*shape[:-1], N) and y.dtype == dtype
    for m in (1, 4, 1100):
        x = (mx.random.normal((m, K), key=mx.random.key(m)) * 0.5).astype(dtype)
        # No output columns.
        y = f(x, w[:0])
        mx.eval(y)
        assert y.shape == (m, 0) and y.dtype == dtype
        # An inner dim of 0: every sum is empty.
        y = f(mx.zeros((m, 0), dtype=dtype), mx.zeros((N, 0), dtype=mx.uint8))
        mx.eval(y)
        assert y.shape == (m, N) and y.dtype == dtype
        assert not y.astype(mx.float32).any().item()
        # The stream still runs a full call.
        y = f(x, w)
        mx.eval(y)
        assert y.shape == (m, N) and mx.isfinite(y.astype(mx.float32)).all().item()
