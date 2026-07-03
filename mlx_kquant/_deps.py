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
    """Raise ``ImportError`` with an actionable hint if ``[tools]`` is absent
    or broken.

    A present-but-broken mlx-lm raises whatever its import died on (e.g.
    transformers 5.13 rejects mlx-lm <= 0.31.3's tokenizer registration with
    an ``AttributeError``); surface that as a version-conflict hint instead of
    a raw traceback from a dependency's internals.
    """
    try:
        import mlx_lm  # noqa: F401
    except ModuleNotFoundError as e:
        raise ImportError(f"mlx-kquant: {_TOOLS_HINT}") from e
    except Exception as e:
        raise ImportError(
            "mlx-kquant: mlx-lm is installed but failed to import, which "
            "usually means an incompatible dependency version (for example "
            "transformers 5.13 breaks mlx-lm <= 0.31.3; fix with: pip install "
            f"'transformers<5.13'). Underlying error: {e!r}"
        ) from e
