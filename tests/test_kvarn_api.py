"""KVarN python surface that needs no device."""

from __future__ import annotations

import mlx_kquant as kq


def test_record_version_exported():
    assert kq.KVARN_RECORD_VERSION == 1
    assert "KVARN_RECORD_VERSION" in kq.__all__
