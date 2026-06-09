"""``mlx-kquant quantize`` — encode a model into a kquant checkpoint."""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "quantize",
        help="encode an mlx-lm / HF model into a kquant checkpoint",
        description="Encode a model's weights with a kquant recipe and write a "
        "checkpoint that loads on a stock mlx wheel.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="HF repo id or local path to a float (bf16/f16) source model.",
    )
    p.add_argument(
        "--mlx-path",
        required=True,
        help="Output directory for the kquant checkpoint.",
    )
    recipe = p.add_mutually_exclusive_group(required=True)
    recipe.add_argument(
        "--preset",
        help="Recipe preset (e.g. q4_k_m, q5_k_moe). See `verify --presets`.",
    )
    recipe.add_argument(
        "--kquant-type",
        help="Uniform codec for every quantizable tensor (e.g. q4_k, q8_0).",
    )
    p.add_argument(
        "--default-codec",
        help="Codec for tensors a preset does not map (defaults to the preset's "
        "own default).",
    )
    p.add_argument(
        "--imatrix",
        help="Optional importance-matrix file (.dat / .gguf) to steer K-quant codecs.",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    from .._deps import require_tools

    require_tools()

    from mlx_lm.utils import load_model

    from ..convert import quantize_model, save
    from ..loader import _resolve_path

    src = _resolve_path(args.model, None)
    model, config = load_model(src, lazy=False)

    # --kquant-type names a uniform codec; it maps to the same default_codec the
    # recipe resolver uses for unmapped tensors.
    default_codec = args.default_codec or args.kquant_type
    _, qconfig = quantize_model(
        model,
        config,
        preset=args.preset,
        default_codec=default_codec,
        imatrix_path=args.imatrix,
    )
    save(args.mlx_path, model, qconfig, hf_path=src)
    n = len(qconfig["quantization"]["per_tensor"])
    print(f"[mlx-kquant] wrote {args.mlx_path} ({n} tensors quantized)")
    return 0
