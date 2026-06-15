"""``mlx-kquant lora`` - train or test a LoRA adapter on a kquant base.

A thin pass-through to mlx-lm's own LoRA trainer: it applies the kquant patch
(so mlx-lm can load, adapt, and save against a kquant checkpoint) and then hands
every remaining argument straight to ``mlx_lm.lora``. There is no bespoke trainer
here - all of mlx-lm's lora flags apply unchanged. Run ``mlx-kquant lora --help``
to see them.

    mlx-kquant lora --model my-q4-ckpt --train --data ./data --iters 200

The trained adapter (``adapters/adapter_config.json`` + ``adapters.safetensors``)
is then mergeable with ``mlx-kquant fuse``.

The dispatcher intercepts ``lora`` before argparse and calls :func:`passthrough`
directly, so mlx-lm's own flags (including ``--help``) reach it untouched;
:func:`add_parser` only registers the command for the top-level ``--help`` list.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser(
        "lora",
        help="train/test a LoRA adapter on a kquant base (mlx-lm trainer)",
        add_help=False,
        description="Pass-through to mlx-lm's LoRA trainer with the kquant patch "
        "applied, so --model may be a kquant checkpoint. All mlx-lm lora flags "
        "apply; run `mlx-kquant lora --help`.",
    )


def passthrough(rest: list[str]) -> int:
    """Apply the kquant patch and delegate ``rest`` to ``mlx_lm.lora``.

    One flag is intercepted rather than forwarded: ``--trust-remote-code``
    opts the kquant loader into a checkpoint's custom ``model_file``
    (arbitrary code); mlx-lm's trainer has no flag of its own for this.
    """
    import sys

    from .._deps import require_tools
    from ._args import has_opt

    wants_help = "-h" in rest or "--help" in rest
    if rest and not wants_help and not has_opt(rest, "-c", "--config"):
        missing = [
            flag
            for flag, present in (
                ("--model (a kquant checkpoint)", has_opt(rest, "--model")),
                ("--data", has_opt(rest, "--data")),
            )
            if not present
        ]
        if missing:
            raise ValueError(
                "lora needs " + " and ".join(missing) + " (or -c <config.yaml>); "
                "refusing mlx-lm's defaults (Qwen3-0.6b / WikiSQL). "
                "See `mlx-kquant lora --help`."
            )

    require_tools()

    from mlx_lm.lora import main as lora_main

    from ..mlx_lm_patch import patch_mlx_lm_lora

    # Bare `mlx-kquant lora` shows mlx-lm's lora help.
    if not rest:
        rest = ["--help"]

    trust = "--trust-remote-code" in rest
    if trust:
        rest = [a for a in rest if a != "--trust-remote-code"]
    patch_mlx_lm_lora(trust_remote_code=trust)

    saved_argv = sys.argv
    sys.argv = ["mlx_lm.lora", *rest]
    try:
        lora_main()
    finally:
        sys.argv = saved_argv
    return 0
