"""mlx-kquant command-line interface.

A single dispatcher, exposed as the ``mlx-kquant`` and ``mlxkq`` console scripts
and as ``python -m mlx_kquant``. Subcommands:

  quantize           encode an mlx-lm / HF model into a kquant checkpoint
  calibrate-imatrix  build an importance matrix from a calibration corpus
  lora               train/test a LoRA adapter on a kquant base (mlx-lm trainer)
  fuse               merge a trained LoRA adapter into a kquant checkpoint
  verify             verify compatible codecs, or a built checkpoint
  run                load a checkpoint and generate text (one-shot)
  chat               interactive chat REPL on a kquant checkpoint (mlx-lm chat)
  inspect            print a checkpoint's per-tensor codec recipe (no GPU)

The model-level subcommands need the ``[tools]`` extra (mlx-lm); a missing extra
surfaces as one clear message via :func:`mlx_kquant._deps.require_tools` rather
than a raw ``ImportError``. The subcommand modules keep their heavy imports
(``mlx_lm``, ``numpy``) inside ``cmd`` so the dispatcher itself - and
``verify --codecs`` / ``inspect`` - work on a base install.
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__
from . import calibrate, chat, fuse, inspect, lora, quantize, run, verify

_SUBCOMMANDS = (quantize, calibrate, lora, fuse, verify, run, chat, inspect)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-kquant",
        description="Create, inspect, and run K-quant checkpoints on MLX.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mlx-kquant {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for mod in _SUBCOMMANDS:
        mod.add_parser(sub)
    return parser


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise RuntimeError("no subparsers registered")  # unreachable


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    # `lora` / `chat` are pass-throughs to mlx-lm CLIs - intercept them before
    # argparse so every mlx-lm flag (including --help) reaches them untouched
    if raw and raw[0] in ("lora", "chat"):
        mod = lora if raw[0] == "lora" else chat
        try:
            return mod.passthrough(raw[1:])
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130
        except (ImportError, ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    parser = _build_parser()
    # A bare subcommand (name only, no flags) prints that command's full help
    # rather than argparse's terse "required arguments" error.
    if len(raw) == 1:
        sub = _subparsers(parser)
        if raw[0] in sub.choices:
            sub.choices[raw[0]].print_help()
            return 0
    args = parser.parse_args(raw)
    if not getattr(args, "func", None):
        # No subcommand at all: show the top-level help rather than erroring out.
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ImportError, ValueError, OSError) as e:
        # OSError covers FileNotFoundError plus the huggingface_hub / requests
        # error family (RepositoryNotFoundError, network failures, ...), which
        # would otherwise surface as raw tracebacks.
        print(f"error: {e}", file=sys.stderr)
        return 1
