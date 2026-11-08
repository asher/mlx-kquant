"""int8_quantize_rows and int8_matmul on a CPU stream. Both run on the GPU
only and refuse a CPU stream when the call is built, before any eval.
tests/test_int8_matmul.py runs on Metal and skips under KQUANT_FORCE_CPU;
this test runs on every device.
"""

import mlx.core as mx
import pytest

import mlx_kquant as kq


def test_quantize_refuses_a_cpu_stream():
    with pytest.raises(ValueError, match="GPU only"):
        kq.int8_quantize_rows(mx.zeros((4, 64), mx.bfloat16), stream=mx.cpu)


def test_matmul_refuses_a_cpu_stream():
    args = (
        mx.zeros((4, 64), mx.int8),
        mx.ones((4, 1), mx.float32),
        mx.zeros((8, 64), mx.int8),
        mx.ones((8,), mx.float32),
    )
    with pytest.raises(ValueError, match="GPU only"):
        kq.int8_matmul(*args, stream=mx.cpu)
