"""require_tools: absent vs broken [tools] install both raise actionable hints."""

import importlib
import sys

import pytest

from mlx_kquant._deps import require_tools


class _PoisonedFinder:
    """Meta-path finder that makes ``import mlx_lm`` raise a given exception."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    def find_spec(self, name, path=None, target=None):
        if name == "mlx_lm" or name.startswith("mlx_lm."):
            raise self._exc
        return None


@pytest.fixture
def _no_mlx_lm(monkeypatch):
    for mod in [m for m in sys.modules if m == "mlx_lm" or m.startswith("mlx_lm.")]:
        monkeypatch.delitem(sys.modules, mod)
    yield
    importlib.invalidate_caches()


def _install_finder(monkeypatch, exc):
    monkeypatch.setattr(sys, "meta_path", [_PoisonedFinder(exc)] + sys.meta_path)


def test_require_tools_missing(monkeypatch, _no_mlx_lm):
    _install_finder(monkeypatch, ModuleNotFoundError("No module named 'mlx_lm'"))
    with pytest.raises(ImportError, match=r"mlx-kquant\[tools\]"):
        require_tools()


def test_require_tools_broken(monkeypatch, _no_mlx_lm):
    # e.g. transformers 5.13 rejecting mlx-lm's tokenizer registration
    _install_finder(
        monkeypatch, AttributeError("'str' object has no attribute '__module__'")
    )
    with pytest.raises(ImportError, match="incompatible dependency"):
        require_tools()


def test_require_tools_present():
    pytest.importorskip("mlx_lm")
    require_tools()
