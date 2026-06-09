"""Load a kquant checkpoint onto a stock ``mlx`` wheel.

Stock ``mlx_lm.load_model`` cannot read a kquant checkpoint: it routes the
``quantization`` config into ``nn.quantize`` (which ``KeyError``s on the absent
``group_size`` and rejects ``mode="kquant"`` regardless). This loader builds the
``mlx-lm`` model class, swaps in ``KQuant*`` modules from the ``per_tensor`` map,
and loads the uint8 wire-byte weights — all against an unmodified ``mlx`` + this
extension.

On-disk format is the one :mod:`mlx_kquant.convert` writes: ``config.json``
carries ``{"quantization": {"mode": "kquant", "per_tensor": {bare_path: codec}}}``
and the safetensors hold ``<path>.weight`` / ``<path>.scales`` (+ optional
``<path>.bias``). The ``per_tensor`` keys are **bare** module paths; the install
seam keys on ``<path>.weight``, so the loader normalizes between the two.

``[tools]`` (mlx-lm) is required; decoding the wire bytes at forward time is
GPU-only until the CPU decode path lands.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

from ._deps import require_tools
from .nn import install_kquant_modules

if TYPE_CHECKING:
    import mlx.nn as nn

_HF_ALLOW = [
    "*.json",
    "model*.safetensors",
    "*.txt",
    "*.jsonl",
    "*.jinja",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
]


def _resolve_path(path_or_hf_repo: str | Path, revision: str | None) -> Path:
    """Return a local checkpoint dir, downloading from the HF Hub if needed."""
    p = Path(path_or_hf_repo)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(path_or_hf_repo),
            revision=revision,
            allow_patterns=_HF_ALLOW,
        )
    )


def _load_config(model_path: Path) -> dict:
    with open(model_path / "config.json") as f:
        return json.load(f)


def _get_classes(config: dict):
    """Resolve ``(Model, ModelArgs)`` for a config; guards the private mlx-lm import."""
    try:
        from mlx_lm.utils import _get_classes as _mlxlm_get_classes

        return _mlxlm_get_classes(config)
    except (ImportError, AttributeError):
        # Fallback that resolves the class here if that private helper moves.
        model_type = config["model_type"]
        try:
            from mlx_lm.utils import MODEL_REMAPPING

            model_type = MODEL_REMAPPING.get(model_type, model_type)
        except (ImportError, AttributeError):
            pass
        try:
            arch = importlib.import_module(f"mlx_lm.models.{model_type}")
        except ImportError as e:
            raise ValueError(f"Model type {model_type!r} not supported.") from e
        return arch.Model, arch.ModelArgs


def _kquant_block(config: dict) -> dict | None:
    """Pull the kquant quantization block from a config, if present and ours."""
    for key in ("quantization", "quantization_config"):
        block = config.get(key)
        if isinstance(block, dict) and block.get("mode") == "kquant":
            return block
    # Some multimodal configs nest the text quantization under text_config.
    text = config.get("text_config")
    if isinstance(text, dict):
        return _kquant_block(text)
    return None


def load(
    path_or_hf_repo: str | Path,
    *,
    lazy: bool = False,
    strict: bool = True,
    trust_remote_code: bool = False,
    model_config: dict | None = None,
    revision: str | None = None,
) -> tuple[nn.Module, dict]:
    """Load a kquant checkpoint and return ``(model, config)``.

    Args:
        path_or_hf_repo: local checkpoint dir or an HF repo id to download.
        lazy: if ``False`` (default), ``mx.eval`` the parameters before
            returning so they are materialized; otherwise defer.
        strict: passed to ``model.load_weights``; require an exact key match.
        trust_remote_code: allow a checkpoint's ``config["model_file"]`` to be
            imported and executed (arbitrary code). Off by default.
        model_config: optional overrides merged into the loaded config.
        revision: optional HF revision (branch / tag / commit).

    Returns:
        ``(model, config)`` with ``KQuant*`` modules installed and weights loaded.

    Raises:
        ValueError: the checkpoint is not a kquant checkpoint, or declares a
            custom ``model_file`` without ``trust_remote_code=True``.
        FileNotFoundError: no ``model*.safetensors`` in the checkpoint.
    """
    require_tools()

    model_path = _resolve_path(path_or_hf_repo, revision)
    config = _load_config(model_path)
    if model_config:
        config.update(model_config)

    quant = _kquant_block(config)
    if quant is None:
        raise ValueError(
            f"{path_or_hf_repo}: not a kquant checkpoint "
            f"(config.quantization.mode != 'kquant'). Load affine / other quants "
            f"with mlx_lm.load instead."
        )
    per_tensor = quant.get("per_tensor")
    if not per_tensor:
        raise ValueError(f"{path_or_hf_repo}: kquant config has no 'per_tensor' map")

    # Build the model class. A custom model_file imports arbitrary code from the
    # checkpoint, so it is gated behind an explicit opt-in.
    model_file = config.get("model_file")
    if model_file is not None and not trust_remote_code:
        raise ValueError(
            f"{path_or_hf_repo} declares a custom model_file ({model_file!r}), "
            f"which would import and execute code shipped in the checkpoint. "
            f"Pass trust_remote_code=True only if you trust its source."
        )
    if model_file is not None:
        spec = importlib.util.spec_from_file_location(
            "kq_custom_model", model_path / model_file
        )
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        model_class, args_class = arch.Model, arch.ModelArgs
    else:
        model_class, args_class = _get_classes(config)

    model = model_class(args_class.from_dict(config))

    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No model*.safetensors in {model_path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # sanitize first — model.sanitize may rename / drop keys.
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # The config stores bare module paths; the install seam keys on
    # "<path>.weight". Normalize before swapping modules in.
    weight_keyed = {f"{path}.weight": codec for path, codec in per_tensor.items()}
    install_kquant_modules(model, weight_keyed)

    model.eval()
    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())

    return model, config
