"""Shared pytest fixtures.

The op tests (dequant / matmul / gather / encode / codecs) and the GGUF
wire-byte reader test touch only small mx array ops and need no model files.
Set ``KQUANT_FORCE_CPU=1`` in environments without a usable Metal GPU (e.g. CI)
to keep them off the GPU path.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_mlx_cache():
    # MLX keeps freed buffers in its cache, and its default limit is near the
    # device memory. Clear it after each test, so the process footprint over a
    # session stays at the largest single test.
    yield
    import mlx.core as mx

    mx.clear_cache()


def _compile_switch_on():
    # While mx's compile switch is on, a compiled function traces once over
    # two calls. While it is off, the function runs on each call.
    import mlx.core as mx

    calls = []

    def f(t):
        calls.append(1)
        return t

    g = mx.compile(f)
    t = mx.zeros((1,))
    g(t)
    g(t)
    return len(calls) == 1


@pytest.fixture
def compile_on():
    # mx's compile switch on for a test of mx.compile, and back as found after
    # it. MLX_DISABLE_COMPILE sets the switch off for the process.
    import mlx.core as mx

    found = _compile_switch_on()
    mx.enable_compile()
    yield
    (mx.enable_compile if found else mx.disable_compile)()


def pytest_configure(config):
    # The logic tests touch only small mx array ops and never dispatch a kquant
    # kernel, so they can run on the CPU device. Set KQUANT_FORCE_CPU=1 in
    # environments without a usable Metal GPU (e.g. CI) to keep them off the GPU.
    if os.environ.get("KQUANT_FORCE_CPU"):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
