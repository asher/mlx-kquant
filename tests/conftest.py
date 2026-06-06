"""Shared pytest fixtures.

The op tests (dequant / matmul / gather / encode / codecs) and the GGUF
wire-byte reader test touch only small mx array ops and need no model files.
Set ``KQUANT_FORCE_CPU=1`` in environments without a usable Metal GPU (e.g. CI)
to keep them off the GPU path.
"""

from __future__ import annotations

import os


def pytest_configure(config):
    # The logic tests touch only small mx array ops and never dispatch a kquant
    # kernel, so they can run on the CPU device. Set KQUANT_FORCE_CPU=1 in
    # environments without a usable Metal GPU (e.g. CI) to keep them off the GPU.
    if os.environ.get("KQUANT_FORCE_CPU"):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
