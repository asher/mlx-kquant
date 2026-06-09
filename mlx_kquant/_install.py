"""The module-swap seam: turn a constructed mlx-lm model's quantizable leaves
into their ``KQuant*`` equivalents.

``install_kquant_modules`` walks the model's leaf modules and replaces each
``Linear`` / ``Embedding`` / ``SwitchLinear`` / ``MultiLinear`` whose
``<path>.weight`` carries a codec with the matching ``KQuant*`` module. It is
arch-generic — driven entirely by the ``{path: codec}`` map — so widening
architecture coverage never touches this file.

``mlx-lm`` is imported lazily (inside the function) so that ``import
mlx_kquant.nn`` needs only ``mlx`` + the built extension.
"""

from __future__ import annotations

import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .nn import (
    KQuantEmbedding,
    KQuantLinear,
    KQuantMultiLinear,
    KQuantSwitchLinear,
)


def install_kquant_modules(model: nn.Module, per_tensor: dict[str, str]) -> int:
    """Swap leaf modules for kquant equivalents.

    Args:
        model: a constructed (unquantized) mlx-lm model.
        per_tensor: maps ``<path>.weight`` -> codec name (e.g. ``"q4_k"``).

    Returns:
        The number of leaf modules replaced.
    """
    # Lazy imports: only needed to type-test the leaves, and only when actually
    # installing onto a model (keeps ``import mlx_kquant.nn`` mlx-lm-free).
    from mlx_lm.models.switch_layers import SwitchLinear

    try:  # absorbed-MLA module (deepseek_v3 family); absent on older mlx-lm.
        from mlx_lm.models.mla import MultiLinear
    except Exception:  # pragma: no cover - depends on installed mlx-lm version
        MultiLinear = None

    n_replaced = 0

    def _replace(path: str, module):
        nonlocal n_replaced
        codec = per_tensor.get(f"{path}.weight")
        if codec is None:
            return module
        if isinstance(module, nn.Linear):
            out_dims, in_dims = module.weight.shape
            n_replaced += 1
            return KQuantLinear(in_dims, out_dims, "bias" in module, codec)
        if isinstance(module, nn.Embedding):
            num_emb, dims = module.weight.shape
            n_replaced += 1
            return KQuantEmbedding(num_emb, dims, codec)
        if isinstance(module, SwitchLinear):
            n_experts, out_dims, in_dims = module.weight.shape
            n_replaced += 1
            return KQuantSwitchLinear(
                n_experts, out_dims, in_dims, "bias" in module, codec
            )
        if MultiLinear is not None and isinstance(module, MultiLinear):
            num_heads, out_dims, in_dims = module.weight.shape
            n_replaced += 1
            return KQuantMultiLinear(in_dims, out_dims, num_heads, codec)
        return module

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_replace, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return n_replaced
