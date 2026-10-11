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


@pytest.mark.parametrize(
    "dtype", [mx.float32, mx.bfloat16, mx.float16], ids=["f32", "bf16", "f16"]
)
@pytest.mark.parametrize("m", [4, 1100])
def test_inner_dim_of_zero_has_the_dtype_of_a_full_call(m, dtype):
    """The zeros for an inner dim of 0 have the dtype the op computes in,
    which for float32 x is not the dtype of x."""
    full = kq.quantized_matmul(
        mx.zeros((m, K), dtype=dtype), _wire("q4_k"), SCALES, "q4_k", True
    )
    y = kq.quantized_matmul(
        mx.zeros((m, 0), dtype=dtype),
        mx.zeros((N, 0), dtype=mx.uint8),
        SCALES,
        "q4_k",
        True,
    )
    mx.eval(y)
    assert y.shape == (m, N) and y.dtype == full.dtype
    assert not y.astype(mx.float32).any().item()
    if dtype == mx.float32:
        assert y.dtype == mx.bfloat16


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
def test_inner_dim_of_zero_checks_the_lora_arguments(dtype):
    """An inner dim of 0 returns zeros only after the LoRA arguments pass
    the checks of a full call."""
    m, r = 4, 8
    x = mx.zeros((m, 0), dtype=dtype)
    w = mx.zeros((N, 0), dtype=mx.uint8)
    a = mx.zeros((0, r), dtype=dtype)
    b = mx.zeros((r, N), dtype=dtype)
    rows = mx.ones((m,))

    def call(**lora):
        return kq.quantized_matmul(x, w, SCALES, "q4_k", True, **lora)

    y = call(lora_a=a, lora_b=b, lora_rows=rows)
    mx.eval(y)
    assert y.shape == (m, N) and y.dtype == dtype
    assert not y.astype(mx.float32).any().item()
    other = mx.float16 if dtype == mx.bfloat16 else mx.bfloat16
    for bad in (
        {"lora_a": a},
        {"lora_rows": rows},
        {"lora_a": a.astype(other), "lora_b": b},
        {"lora_a": mx.zeros((3, r), dtype=dtype), "lora_b": b},
        {"lora_a": a, "lora_b": mx.zeros((r, N + 1), dtype=dtype)},
        {"lora_a": a, "lora_b": mx.zeros((r + 1, N), dtype=dtype)},
        {"lora_a": a, "lora_b": b, "lora_rows": mx.ones((m + 1,))},
    ):
        with pytest.raises(ValueError):
            call(**bad)
    with pytest.raises(ValueError, match="2-D w"):
        kq.quantized_matmul(
            x,
            mx.zeros((2, N, 0), dtype=mx.uint8),
            SCALES,
            "q4_k",
            True,
            lora_a=a,
            lora_b=b,
        )
