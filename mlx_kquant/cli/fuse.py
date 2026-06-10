"""``mlx-kquant fuse`` - merge a trained LoRA adapter into a kquant checkpoint.

Two output modes:

* default (**keep-kquant**) - each adapted weight is decoded, the low-rank delta
  is added, and the merged weight is **re-encoded with that tensor's own codec**
  (read off the base layer, not a recipe), so the result is a kquant checkpoint
  byte-compatible with the original. Re-encoding runs on CPU or Metal, and adds a
  small per-codec re-quant rounding error. Pass ``--imatrix``
  (the same one used to quantize the base) to steer that re-encode so the merge
  preserves the base's calibration instead of rounding the adapted tensors blind.
* ``--dequantize`` - writes a **float** checkpoint: adapted layers fuse to float
  and the remaining (non-adapted) kquant layers are decoded too, so the whole
  model is float. No re-quant error; loads with stock ``mlx_lm.load``; runs
  without a GPU (decode has a CPU path).

The base must be a kquant checkpoint (see ``mlx-kquant quantize``) and the
adapter the standard mlx-lm layout (``adapter_config.json`` +
``adapters.safetensors``). DoRA adapters are not supported on a kquant base.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "fuse",
        help="merge a trained LoRA adapter back into a kquant checkpoint",
        description="Fuse LoRA adapter weights into a kquant base. By default the "
        "result stays kquant (merged weights re-encoded with each tensor's "
        "original codec); --dequantize writes a float checkpoint instead "
        "(no re-quant error). Both modes run on CPU or Metal.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="kquant checkpoint dir or HF repo id (the adapted base).",
    )
    p.add_argument(
        "--adapter-path",
        required=True,
        help="Trained adapter dir (adapter_config.json + adapters.safetensors).",
    )
    p.add_argument(
        "--save-path",
        required=True,
        help="Output directory for the fused checkpoint.",
    )
    p.add_argument(
        "--dequantize",
        action="store_true",
        help="Write a float (dequantized) checkpoint instead of re-encoding to "
        "kquant. Loads with stock mlx_lm.load and needs no GPU.",
    )
    p.add_argument(
        "--imatrix",
        help="Importance matrix (.dat / .gguf) to steer the keep-kquant re-encode. "
        "Pass the same imatrix the base was quantized with so the merge preserves "
        "that calibration. Ignored with --dequantize (no re-encode happens).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow the base checkpoint's custom model_file (arbitrary code) to load.",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    from .._deps import require_tools

    require_tools()

    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.utils import load_adapters

    from ..convert import save
    from ..loader import _resolve_path, load
    from ..mlx_lm_patch import patch_mlx_lm_lora

    # Teach mlx-lm's tuner to wrap + fuse kquant bases before applying adapters.
    patch_mlx_lm_lora()

    src = _resolve_path(args.model, None)
    model, config = load(args.model, trust_remote_code=args.trust_remote_code)
    load_adapters(model, args.adapter_path)

    imatrix_by_path = _resolve_imatrix(args, model, config)
    fused = [
        (
            name,
            module.fuse(dequantize=args.dequantize, imatrix=imatrix_by_path.get(name)),
        )
        for name, module in model.named_modules()
        if hasattr(module, "fuse")
    ]
    if not fused:
        raise ValueError(
            f"{args.adapter_path}: no fusable LoRA layers on the base - is "
            f"{args.model} a kquant checkpoint with a matching adapter?"
        )
    model.update_modules(tree_unflatten(fused))

    if args.dequantize:
        # Fusing made the adapted layers float; decode any kquant layers the
        # adapter didn't touch so the saved checkpoint is uniformly float.
        n_dq = _dequantize_remaining(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)
        note = f", {n_dq} base layers dequantized" if n_dq else ""
    else:
        note = ""

    save(args.save_path, model, config, hf_path=src)
    mode = "float" if args.dequantize else "kquant"
    print(
        f"[mlx-kquant] fused {len(fused)} adapter layers{note} "
        f"-> {args.save_path} ({mode})"
    )
    return 0


def _resolve_imatrix(args, model, config) -> dict:
    """Map ``--imatrix`` onto the fusable module paths (keep-kquant only).

    Returns ``{module_path: importance_vector}`` for the adapted (fusable) layers.
    Only those layers are re-encoded, so only they consult the imatrix; the
    untouched kquant base keeps its original bytes. Empty when no imatrix is given,
    under ``--dequantize`` (nothing is re-encoded), or when nothing resolves.
    """
    if not args.imatrix:
        return {}
    if args.dequantize:
        print("[mlx-kquant] note: --imatrix ignored with --dequantize (no re-encode)")
        return {}

    from ..imatrix import load_imatrix, map_imatrix_to_hf

    raw = load_imatrix(args.imatrix)
    fusable = [name for name, m in model.named_modules() if hasattr(m, "fuse")]
    arch = (config.get("architectures") or [None])[0]
    mapped = map_imatrix_to_hf(raw, fusable, arch)
    print(f"[mlx-kquant] imatrix coverage: {len(mapped)}/{len(fusable)} fused tensors")
    return mapped


def _dequantize_remaining(model) -> int:
    """Decode any KQuant* layers left after fusing into matching float layers.

    Returns the number of layers replaced. Used only by ``--dequantize`` so a
    checkpoint whose adapter covered a subset of tensors is still all-float.
    """
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    import mlx_kquant as kq

    from ..codec_geometry import in_features
    from ..nn import (
        KQuantEmbedding,
        KQuantLinear,
        KQuantMultiLinear,
        KQuantSwitchLinear,
    )

    try:
        from mlx_lm.models.switch_layers import SwitchLinear
    except Exception:  # pragma: no cover - depends on installed mlx-lm
        SwitchLinear = None
    try:
        from mlx_lm.models.mla import MultiLinear
    except Exception:  # pragma: no cover - present only on newer mlx-lm
        MultiLinear = None

    replacements = []
    for name, m in model.named_modules():
        bias = "bias" in m
        if isinstance(m, KQuantEmbedding):
            weight = kq.dequantize(m["weight"], m["scales"], m.kquant_type)
            num_emb, dims = weight.shape
            repl = nn.Embedding(num_emb, dims)
            repl.weight = weight
        elif isinstance(m, KQuantSwitchLinear) and SwitchLinear is not None:
            num_experts, out_dims, row_bytes = m["weight"].shape
            in_dims = in_features(m.kquant_type, row_bytes)
            flat = m["weight"].reshape(num_experts * out_dims, row_bytes)
            weight = kq.dequantize(flat, m["scales"], m.kquant_type).reshape(
                num_experts, out_dims, in_dims
            )
            repl = SwitchLinear(in_dims, out_dims, num_experts, bias=bias)
            repl.weight = weight
            if bias:
                repl.bias = m.bias
        elif MultiLinear is not None and isinstance(m, KQuantMultiLinear):
            num_heads, out_dims, row_bytes = m["weight"].shape
            in_dims = in_features(m.kquant_type, row_bytes)
            flat = m["weight"].reshape(num_heads * out_dims, row_bytes)
            weight = kq.dequantize(flat, m["scales"], m.kquant_type).reshape(
                num_heads, out_dims, in_dims
            )
            repl = MultiLinear(in_dims, out_dims, num_heads)
            repl.weight = weight
        elif isinstance(m, KQuantLinear):
            weight = kq.dequantize(m["weight"], m["scales"], m.kquant_type)
            out_dims, in_dims = weight.shape
            repl = nn.Linear(in_dims, out_dims, bias=bias)
            repl.weight = weight
            if bias:
                repl.bias = m.bias
        else:
            continue
        replacements.append((name, repl))

    if replacements:
        model.update_modules(tree_unflatten(replacements))
    return len(replacements)
