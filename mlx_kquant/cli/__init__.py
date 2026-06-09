"""mlx-kquant command-line interface.

A single dispatcher, exposed as the ``mlx-kquant`` and ``mlxkq`` console scripts
and as ``python -m mlx_kquant``. Subcommands:

  quantize           encode an mlx-lm / HF model into a kquant checkpoint
  calibrate-imatrix  build an importance matrix from a calibration corpus
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
from . import calibrate, quantize, run, verify

_SUBCOMMANDS = (quantize, calibrate, verify, run)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-kquant",
        description="Create, inspect, and run GGUF K-quant checkpoints on MLX.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mlx-kquant {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    for mod in _SUBCOMMANDS:
        mod.add_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ImportError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
