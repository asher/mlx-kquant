"""The built package fits PyPI's per-file limit (scripts/check-wheel-size.py)."""

import importlib.util
from pathlib import Path

import pytest

import mlx_kquant as kq

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check-wheel-size.py"


def _script():
    spec = importlib.util.spec_from_file_location("check_wheel_size", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wheel_under_budget():
    m = _script()
    pkg = Path(kq.__file__).parent
    if not (pkg / m.METALLIB).is_file():
        pytest.skip("no metallib in this build")
    mb = m.wheel_bytes(pkg) / 1e6
    assert mb <= m.CHECK_MB, (
        f"wheel estimate {mb:.1f} MB is over {m.CHECK_MB:.0f} MB; see "
        "CONTRIBUTING.md, Kernels and the wheel size"
    )
