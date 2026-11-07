"""sdpa_qk8_inputs and sdpa_qk8 on a CPU stream. Both run on the GPU only
and refuse a CPU stream when the call is built, before any eval.
tests/test_sdpa_qk8.py runs on Metal and skips under KQUANT_FORCE_CPU; this
test runs on every device.
"""

import mlx.core as mx
import pytest

import mlx_kquant as kq

D = 128


def test_inputs_refuse_a_cpu_stream():
    q = mx.zeros((1, 4, 64, D), mx.bfloat16)
    k = mx.zeros((1, 2, 96, D), mx.bfloat16)
    with pytest.raises(ValueError, match="GPU only"):
        kq.sdpa_qk8_inputs(q, k, D**-0.5, stream=mx.cpu)


def test_call_refuses_a_cpu_stream():
    args = (
        mx.zeros((1, 4, 64, D), mx.int8),
        mx.ones((1, 4, 64), mx.float32),
        mx.zeros((1, 2, 96, D), mx.int8),
        mx.ones((1, 2, 96), mx.float32),
        mx.zeros((1, 2, 96, D), mx.bfloat16),
    )
    with pytest.raises(ValueError, match="GPU only"):
        kq.sdpa_qk8(*args, stream=mx.cpu)
