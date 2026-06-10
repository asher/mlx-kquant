"""``mlx-kquant chat`` - interactive chat REPL on a kquant checkpoint.

A thin pass-through to mlx-lm's chat REPL: it applies the kquant patch (so the
REPL can load a kquant checkpoint and attach a LoRA adapter via
``--adapter-path``) and then hands every argument straight to ``mlx_lm.chat``.
All of mlx-lm's chat flags apply unchanged (``--temp``, ``--top-p``, ``--seed``,
``--system-prompt``, ``--max-tokens``, ...); run ``mlx-kquant chat --help`` to
see them.

    mlx-kquant chat --model my-q4-ckpt --temp 0.7

``--trust-remote-code`` is forwarded (mlx-lm reads it for the tokenizer) and
additionally opts the kquant loader into a checkpoint's custom ``model_file``.

The dispatcher intercepts ``chat`` before argparse and calls
:func:`passthrough` directly (like ``lora``), so mlx-lm's own flags (including
``--help``) reach it untouched; :func:`add_parser` only registers the command
for the top-level ``--help`` list.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser(
        "chat",
        help="interactive chat REPL on a kquant checkpoint (mlx-lm chat)",
        add_help=False,
        description="Pass-through to mlx-lm's chat REPL with the kquant patch "
        "applied, so --model may be a kquant checkpoint. All mlx-lm chat flags "
        "apply; run `mlx-kquant chat --help`.",
    )


def passthrough(rest: list[str]) -> int:
    """Apply the kquant patch and delegate ``rest`` to ``mlx_lm.chat``.

    Unlike the ``lora`` pass-through, ``--trust-remote-code`` is *not* stripped:
    mlx-lm's chat parser has its own (tokenizer) meaning for it, and the same
    flag also grants the kquant loader's ``model_file`` opt-in here.
    """
    import sys

    from .._deps import require_tools

    require_tools()

    from mlx_lm.chat import main as chat_main

    from ..mlx_lm_patch import patch_mlx_lm_lora

    patch_mlx_lm_lora(trust_remote_code="--trust-remote-code" in rest)

    saved_argv = sys.argv
    sys.argv = ["mlx_lm.chat", *rest]
    try:
        chat_main()
    finally:
        sys.argv = saved_argv
    return 0
