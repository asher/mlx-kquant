"""mlx-kquant command-line interface.

A single dispatcher, exposed as the ``mlx-kquant`` and ``mlxkq`` console scripts
and as ``python -m mlx_kquant``. Subcommands:

  quantize           encode an mlx-lm / HF model into a kquant checkpoint
  calibrate-imatrix  build an importance matrix from a calibration corpus
  lora               train/test a LoRA adapter on a kquant base (mlx-lm trainer)
  fuse               merge a trained LoRA adapter into a kquant checkpoint
  verify             smoke-check the codecs / presets, or a built checkpoint
  run                load a checkpoint and generate a few tokens

The model-level subcommands need the ``[tools]`` extra (mlx-lm); a missing extra
surfaces as one clear message via :func:`mlx_kquant._deps.require_tools` rather
than a raw ``ImportError``. The subcommand modules keep their heavy imports
(``mlx_lm``, ``numpy``) inside ``cmd`` so the dispatcher itself — and
``verify --codecs`` — work on a base install.
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__
from . import calibrate, fuse, lora, quantize, run, verify

_SUBCOMMANDS = (quantize, calibrate, lora, fuse, verify, run)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-kquant",
        description="Create, inspect, and run K-quant checkpoints on MLX.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mlx-kquant {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    for mod in _SUBCOMMANDS:
        mod.add_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    # `lora` is a pass-through to mlx-lm's trainer — intercept it before argparse
    # so every mlx-lm flag (including --help) reaches it untouched (REMAINDER
    # can't capture a leading option).
    if raw and raw[0] == "lora":
        try:
            return lora.passthrough(raw[1:])
        except (ImportError, ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    args = _build_parser().parse_args(raw)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ImportError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
