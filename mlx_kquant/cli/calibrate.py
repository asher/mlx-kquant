"""``mlx-kquant calibrate-imatrix`` — build an importance matrix from a corpus.

Runs the model forward over a plain-text calibration corpus, capturing per-input-
feature ``sum(x²) / n_tokens`` on every ``nn.Linear``. Output is the legacy
llama-imatrix binary ``.dat`` format, consumable by ``mlx-kquant quantize
--imatrix`` (and :mod:`mlx_kquant.imatrix`).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

_Sink = dict[str, dict[str, object]]


class _AccumLinear(nn.Module):
    """Wraps an ``nn.Linear`` to accumulate ``sum(x²)`` per input feature."""

    def __init__(self, base: nn.Module, key: str, sink: _Sink):
        super().__init__()
        self.base = base
        # Keep the bookkeeping out of MLX's parameter system.
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_sink", sink)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        flat = x.astype(mx.float32).reshape(-1, x.shape[-1])
        sq = (flat * flat).sum(axis=0)
        ntok = flat.shape[0]
        entry = self._sink.get(self._key)
        if entry is None:
            self._sink[self._key] = {"sum_x2": sq, "ncall": int(ntok)}
        else:
            entry["sum_x2"] = entry["sum_x2"] + sq
            entry["ncall"] += int(ntok)
        return self.base(x, *args, **kwargs)


def _install_hooks(model: nn.Module, sink: _Sink) -> int:
    """Wrap every 2-D ``nn.Linear`` in ``_AccumLinear``; return the count."""
    count = 0

    def _replace(path: str, module):
        nonlocal count
        if not isinstance(module, nn.Linear):
            return module
        w = getattr(module, "weight", None)
        if w is None or w.ndim != 2:
            return module
        count += 1
        return _AccumLinear(module, f"{path}.weight", sink)

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_replace, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return count


def _chunk_corpus(tokenizer, corpus_path: Path, ctx: int, max_chunks: int):
    """Tokenize a plain-text file into non-overlapping chunks of ``ctx`` tokens."""
    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[list[int]] = []
    for start in range(0, len(ids), ctx):
        chunk = ids[start : start + ctx]
        if len(chunk) < max(1, ctx // 2):
            break
        chunks.append(chunk)
        if 0 < max_chunks <= len(chunks):
            break
    return chunks


def _write_dat(out_path: Path, sink: _Sink) -> int:
    """Write the legacy llama-imatrix ``.dat`` binary format; return entry count."""
    import numpy as np

    items = []
    for name, entry in sink.items():
        ncall = int(entry["ncall"])
        if ncall == 0:
            continue
        avg = entry["sum_x2"] / float(ncall)
        mx.eval(avg)
        items.append((name, ncall, np.asarray(avg, dtype=np.float32).copy()))

    with out_path.open("wb") as f:
        f.write(struct.pack("<i", len(items)))
        for name, ncall, data in items:
            name_b = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<ii", ncall, data.size))
            f.write(data.tobytes())

    return len(items)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "calibrate-imatrix",
        help="build an importance matrix from a calibration corpus",
        description="Generate a llama-imatrix .dat file to steer K-quant encoding.",
    )
    p.add_argument(
        "--model", required=True, help="HF repo id or local path to a float model."
    )
    p.add_argument(
        "--corpus", required=True, help="Path to a plain-text calibration file."
    )
    p.add_argument(
        "--output", required=True, help="Output path for the .dat importance matrix."
    )
    p.add_argument(
        "--ctx",
        type=int,
        default=512,
        help="Context length (tokens per forward pass). Default: 512.",
    )
    p.add_argument(
        "--chunks",
        type=int,
        default=50,
        help="Max calibration chunks to process (0 = all). Default: 50.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the tokenizer.",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    from .._deps import require_tools

    require_tools()

    from mlx_lm.utils import load

    print(f"[mlx-kquant] loading {args.model}")
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
    )

    sink: _Sink = {}
    n_hooked = _install_hooks(model, sink)
    print(f"[mlx-kquant] hooked {n_hooked} Linear modules")
    if n_hooked == 0:
        print("error: no quantizable Linear modules found.", file=sys.stderr)
        return 1

    token_chunks = _chunk_corpus(tokenizer, Path(args.corpus), args.ctx, args.chunks)
    if not token_chunks:
        print("error: corpus produced no chunks (too short?).", file=sys.stderr)
        return 1
    print(f"[mlx-kquant] {len(token_chunks)} chunks x {args.ctx} tokens")

    for i, ids in enumerate(token_chunks):
        x = mx.array(ids, dtype=mx.int32).reshape(1, -1)
        logits = model(x)
        mx.eval(logits)
        mx.eval([v["sum_x2"] for v in sink.values()])
        if (i + 1) % 10 == 0 or i + 1 == len(token_chunks):
            print(f"[mlx-kquant] chunk {i + 1}/{len(token_chunks)}")

    out_path = Path(args.output)
    n_entries = _write_dat(out_path, sink)
    print(f"[mlx-kquant] wrote {out_path} ({n_entries} entries)")
    return 0
