"""get_cb_caps/set_cb_caps roundtrip against the live Metal device.

Metal-only (the device singleton): skipped under KQUANT_FORCE_CPU.

Usage: test_cb_caps.py
"""

from __future__ import annotations

import os

import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="cb caps live on the Metal device; no CPU path.",
)


def test_roundtrip():
    ops, mb = kq.get_cb_caps()
    assert ops > 0 and mb > 0
    prev = kq.set_cb_caps(ops + 7, mb + 13)
    assert prev == (ops, mb)
    assert kq.get_cb_caps() == (ops + 7, mb + 13)
    assert kq.set_cb_caps(ops, mb) == (ops + 7, mb + 13)
    assert kq.get_cb_caps() == (ops, mb)


def test_rejects_implausible():
    with pytest.raises(ValueError):
        kq.set_cb_caps(0, 40)
    with pytest.raises(ValueError):
        kq.set_cb_caps(400, -1)
