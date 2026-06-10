"""kquant encode driver + checkpoint serializer.

:func:`quantize_model` walks a constructed (unquantized) ``mlx-lm`` model,
resolves a per-tensor codec from a recipe preset (see :mod:`mlx_kquant.recipes`),
encodes each quantizable weight with the ``quantize`` op (GPU), and swaps in the
matching ``KQuant*`` module so the returned model runs immediately. :func:`save`
writes the result as an ``mlx-lm``-readable checkpoint (sharded safetensors +
``config.json``).

This module is named ``convert`` (not ``quantize``) on purpose: a ``quantize``
submodule would shadow the package-level ``mlx_kquant.quantize`` encode op.

On-disk format:

* ``config.json`` carries ``quantization`` **and** a mirrored
  ``quantization_config``, each ``{"mode": "kquant", "per_tensor": {path: codec}}``
  where ``path`` is the **bare module path** (no ``.weight`` suffix).
* safetensors hold ``<path>.weight`` (uint8 wire bytes), ``<path>.scales``
  (uint8 ``[1]`` placeholder - K-quant scales live inside the wire bytes), and an
  optional ``<path>.bias``; unquantized tensors keep their source dtype.

Encode is GPU-only in v0.1.0 (the ``quantize`` op has no CPU path yet).
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Bind the encode op by its stable extension location: the package attribute
# ``mlx_kquant.quantize`` is the op, but importing a ``quantize`` submodule would
# shadow it - hence this module is named ``convert`` and reaches the op via _ext.
from ._ext import quantize as _encode_op
from .codec_geometry import geometry
from .recipes import classify_tensors, resolve_codec_map

if TYPE_CHECKING:
    import numpy as np

# Tokenizer / processor / aux files copied unchanged from the source model on
# save (everything that isn't weights or the config we rewrite).
_AUX_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
)


def _encode_weight(w: mx.array, codec: str, imatrix_vec: np.ndarray | None) -> mx.array:
    """Encode a float weight tensor into uint8 kquant wire bytes (GPU)."""
    imatrix_arg = None
    if imatrix_vec is not None:
        in_dims = w.shape[-1]
        if imatrix_vec.shape == (in_dims,):
            imatrix_arg = mx.array(imatrix_vec, dtype=mx.float32)
    wq, _ = _encode_op(w, codec, imatrix=imatrix_arg)
    mx.eval(wq)
    return wq


def quantize_model(
    model: nn.Module,
    config: dict,
    *,
    preset: str | None = None,
    default_codec: str | None = None,
    overrides: dict[str, str] | None = None,
    imatrix_path: str | None = None,
) -> tuple[nn.Module, dict]:
    """Encode ``model`` in place with a kquant recipe and return the new config.

    Args:
        model: a constructed (unquantized) ``mlx-lm`` model.
        config: the model's config dict (copied, not mutated).
        preset: a recipe preset name (see ``recipes.KQUANT_PRESETS``).
        default_codec: codec for tensors with no preset entry (and the base when
            no preset is given, i.e. a uniform quant).
        overrides: ``{path: codec}`` taking precedence over the preset.
        imatrix_path: optional imatrix file to steer the K-quant codecs.

    Returns:
        ``(model, quantized_config)`` - the same model with ``KQuant*`` modules
        swapped in, and a config carrying the ``quantization`` map.
    """
    if preset is None and default_codec is None:
        raise ValueError("quantize_model() needs a preset or a default_codec (or both)")

    from mlx_lm.models.switch_layers import SwitchLinear

    from .nn import (
        KQuantEmbedding,
        KQuantLinear,
        KQuantMultiLinear,
        KQuantSwitchLinear,
    )

    try:  # absorbed-MLA module (deepseek_v3 family); absent on older mlx-lm.
        from mlx_lm.models.mla import MultiLinear
    except Exception:  # pragma: no cover - depends on installed mlx-lm version
        MultiLinear = None

    role_map = classify_tensors(model)
    codec_map = resolve_codec_map(
        role_map, preset=preset, default_codec=default_codec, overrides=overrides
    )

    imatrix_by_path: dict[str, np.ndarray] = {}
    if imatrix_path is not None:
        from .imatrix import load_imatrix, map_imatrix_to_hf

        raw_imatrix = load_imatrix(imatrix_path)
        quant_paths = [
            p for p, m in model.named_modules() if hasattr(m, "to_quantized")
        ]
        arch = (config.get("architectures") or [None])[0]
        imatrix_by_path = map_imatrix_to_hf(raw_imatrix, quant_paths, arch)
        print(
            f"[mlx-kquant] imatrix coverage: "
            f"{len(imatrix_by_path)}/{len(quant_paths)} tensors"
        )

    per_tensor: dict[str, str] = {}
    replacements = []
    skipped = 0
    for path, module in model.named_modules():
        codec = codec_map.get(path)
        if codec is None:
            continue
        w = module.weight
        in_dims = w.shape[-1]
        _, _, _, wpb = geometry(codec)
        if in_dims % wpb != 0:
            print(
                f"[mlx-kquant] skip {path}: row width {in_dims} not a multiple "
                f"of {wpb} for codec {codec!r}"
            )
            skipped += 1
            continue

        wq = _encode_weight(w, codec, imatrix_by_path.get(path))
        has_bias = "bias" in module

        if isinstance(module, nn.Embedding):
            num_emb, dims = w.shape
            repl = KQuantEmbedding(num_emb, dims, codec)
            repl.weight = wq
        elif isinstance(module, SwitchLinear):
            n_experts, out_dims, in_d = w.shape
            repl = KQuantSwitchLinear(n_experts, out_dims, in_d, has_bias, codec)
            repl.weight = wq
            if has_bias:
                repl.bias = module.bias
        elif MultiLinear is not None and isinstance(module, MultiLinear):
            num_heads, out_dims, in_d = w.shape
            repl = KQuantMultiLinear(in_d, out_dims, num_heads, codec)
            repl.weight = wq
        elif isinstance(module, nn.Linear):
            out_dims, in_d = w.shape
            repl = KQuantLinear(in_d, out_dims, has_bias, codec)
            repl.weight = wq
            if has_bias:
                repl.bias = module.bias
        else:
            # Codec resolved for a module type we don't encode - leave it.
            skipped += 1
            continue

        per_tensor[path] = codec
        replacements.append((path, repl))

    if replacements:
        model.update_modules(tree_unflatten(replacements))

    print(
        f"[mlx-kquant] encoded {len(per_tensor)} tensors"
        + (f" ({skipped} skipped)" if skipped else "")
    )

    quantized_config = copy.deepcopy(config)
    quant_block = {"mode": "kquant", "per_tensor": per_tensor}
    quantized_config["quantization"] = quant_block
    quantized_config["quantization_config"] = quant_block
    return model, quantized_config


def save(
    dst: str | Path,
    model: nn.Module,
    config: dict,
    *,
    hf_path: str | Path | None = None,
) -> None:
    """Write a quantized model to ``dst`` as an ``mlx-lm``-readable checkpoint.

    Writes sharded safetensors + ``model.safetensors.index.json`` +
    ``config.json``. If ``hf_path`` (the source repo dir) is given, tokenizer /
    processor files are copied alongside so the result is self-contained.
    """
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Weights (sharded safetensors + index) via mlx-lm's saver.
    from mlx_lm.utils import save_model

    save_model(dst, model)

    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if hf_path is not None:
        hf_path = Path(hf_path)
        for name in _AUX_FILES:
            src = hf_path / name
            if src.exists():
                shutil.copy2(src, dst / name)


def tree_weight_keys(model: nn.Module) -> list[str]:
    """Flattened parameter keys for ``model`` - handy for tests/debugging."""
    return [k for k, _ in tree_flatten(model.parameters())]
