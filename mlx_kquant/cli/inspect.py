"""``mlx-kquant inspect`` - print a kquant checkpoint's per-tensor codec recipe.

Reads only ``config.json`` and the safetensors headers (tensor shapes), so it
needs no GPU, no model build, and not even the ``[tools]`` extra - it works on a
base install and returns instantly.
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inspect",
        help="print the per-tensor codec recipe of a kquant checkpoint",
        description="Read a kquant checkpoint's config + safetensors headers and "
        "print each quantized tensor's codec, bit width, and packed/logical shape. "
        "No GPU or [tools] extra needed.",
    )
    p.add_argument(
        "--model", required=True, help="kquant checkpoint dir or HF repo id."
    )
    p.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit JSON, not a table."
    )
    p.set_defaults(func=cmd)


def _packed_shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape, read from the safetensors header(s) (no data load)."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))

    shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with open(model_dir / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for k, meta in header.items():
            if k != "__metadata__":
                shapes[k] = tuple(meta["shape"])
    return shapes


def cmd(args: argparse.Namespace) -> int:
    from ..codec_geometry import geometry, in_features
    from ..loader import _kquant_block, _load_config, _resolve_path

    model_dir = _resolve_path(args.model, None)
    config = _load_config(model_dir)
    block = _kquant_block(config)
    if block is None:
        print(f"error: {args.model} is not a kquant checkpoint", file=sys.stderr)
        return 1

    per_tensor = block.get("per_tensor", {})
    shapes = _packed_shapes(model_dir)

    rows, hist = [], collections.Counter()
    for path, codec in sorted(per_tensor.items()):
        gs, bits, _, _ = geometry(codec)
        packed = shapes.get(f"{path}.weight")
        kind, logical = "?", None
        if packed is not None and len(packed) == 3:
            kind = "moe"
            logical = (packed[0], packed[1], in_features(codec, packed[2]))
        elif packed is not None and len(packed) == 2:
            kind = "embedding" if "embed" in path else "linear"
            logical = (packed[0], in_features(codec, packed[1]))
        hist[codec] += 1
        rows.append((path, kind, codec, bits, gs, packed, logical))

    if args.as_json:
        print(
            json.dumps(
                {
                    "model_type": config.get("model_type"),
                    "num_hidden_layers": config.get("num_hidden_layers"),
                    "num_quantized": len(rows),
                    "codec_histogram": dict(hist),
                    "tensors": [
                        {
                            "path": p,
                            "kind": k,
                            "codec": c,
                            "bits": b,
                            "group_size": g,
                            "packed_shape": ps,
                            "logical_shape": ls,
                        }
                        for p, k, c, b, g, ps, ls in rows
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(
        f"model_type={config.get('model_type')}  "
        f"layers={config.get('num_hidden_layers')}  quantized tensors={len(rows)}"
    )
    print(f"codec histogram: {dict(hist)}\n")
    hdr = (
        f"{'module path':44} {'kind':9} {'codec':6} {'bits':>4} {'gsize':>5} "
        f"{'packed shape':>18} {'logical shape':>18}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p, k, c, b, g, ps, ls in rows:
        print(f"{p:44} {k:9} {c:6} {b:>4} {g:>5} {str(ps):>18} {str(ls):>18}")
    return 0
