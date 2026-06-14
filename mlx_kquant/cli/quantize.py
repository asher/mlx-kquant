"""``mlx-kquant quantize`` - encode a model into a kquant checkpoint."""

from __future__ import annotations

import argparse


class _ListPresetsAction(argparse.Action):
    """``--list-presets``: print the recipe presets and exit.

    A print-and-exit action (like ``--version``) so it fires before argparse
    enforces the otherwise-required ``--model`` / ``--mlx-path`` / recipe args.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(
            option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        from ..recipes import format_presets

        print(format_presets())
        parser.exit()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    # Cheap, base-install-safe imports: codec/preset typos fail in argparse
    # (with the choices listed) instead of after a multi-GB model load.
    from ..codec_geometry import CODEC_GEOMETRY
    from ..recipes import KQUANT_PRESETS

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
    p.add_argument(
        "--list-presets",
        action=_ListPresetsAction,
        help="List the recipe presets (and what each maps) and exit.",
    )
    recipe = p.add_mutually_exclusive_group(required=True)
    recipe.add_argument(
        "--preset",
        choices=sorted(KQUANT_PRESETS),
        help="Recipe preset. See `--list-presets` for what each maps.",
    )
    recipe.add_argument(
        "--kquant-type",
        choices=sorted(CODEC_GEOMETRY),
        help="Uniform codec for every quantizable tensor.",
    )
    p.add_argument(
        "--default-codec",
        choices=sorted(CODEC_GEOMETRY),
        help="Codec for tensors a preset does not map (defaults to the preset's "
        "own default).",
    )
    p.add_argument(
        "--imatrix",
        help="Optional importance-matrix file (.dat / .gguf) to steer K-quant codecs.",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from .._deps import require_tools

    require_tools()

    from mlx_lm.utils import load_model

    from ..convert import quantize_model, save
    from ..loader import _resolve_path

    # Fail on bad side inputs before the (slow, large) model load.
    if args.imatrix and not Path(args.imatrix).is_file():
        raise FileNotFoundError(f"--imatrix {args.imatrix}: no such file")

    src = _resolve_path(args.model, None)
    try:
        src_config = json.loads((src / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        src_config = {}
    quant = src_config.get("quantization") or src_config.get("quantization_config")
    if quant:
        mode = "unknown"
        if isinstance(quant, dict):
            mode = quant.get("mode") or quant.get("quant_method") or "affine"
        raise ValueError(
            f"{args.model} is already quantized (mode={mode!r}); quantize needs "
            f"a float (bf16/f16) source model."
        )

    print(f"[mlx-kquant] loading {args.model}")
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
