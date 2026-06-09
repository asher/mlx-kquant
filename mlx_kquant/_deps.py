"""Optional ``[tools]`` dependency guard, shared by the loader and the CLI.

The model-level features (loading / quantizing real checkpoints) need ``mlx-lm``
and friends, which live behind the ``[tools]`` extra. Keeping those imports out
of the raw ``kq.*`` op layer is deliberate; this helper turns a missing extra
into one clear, actionable message instead of a raw ``ImportError`` surfacing
deep in a traceback.
"""

from __future__ import annotations

_TOOLS_HINT = (
    "this needs the optional model-level dependencies (mlx-lm, transformers, "
    "gguf). Install them with:\n\n    pip install 'mlx-kquant[tools]'"
)


def require_tools() -> None:
    """Raise ``ImportError`` with an actionable hint if ``[tools]`` is absent."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError as e:
        raise ImportError(f"mlx-kquant: {_TOOLS_HINT}") from e
