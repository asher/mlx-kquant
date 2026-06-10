#!/usr/bin/env python3
"""Doc-lint: keep the codec facts in three places from drifting apart.

Compares, for all ten GGUF K-quant / legacy codecs:

  1. ``mlx_kquant/codec_geometry.py`` - the Python single source of truth;
  2. the README "Codec reference" table - what users read;
  3. ``kq.codecs()`` - the runtime list the built extension exposes (only when
     the extension is importable; skipped, with a note, otherwise).

This is a *documentation* lint (it catches geometry / naming drift), not a
numeric kernel test. ``--check`` exits non-zero on any mismatch so CI fails on
drift; without it the script just prints the comparison.

Loads ``codec_geometry.py`` directly by path so the lint runs even when the
compiled ``_ext`` has not been built (e.g. the lint-only CI job).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_geometry() -> dict[str, tuple[int, int, int, int]]:
    path = REPO / "mlx_kquant" / "codec_geometry.py"
    spec = importlib.util.spec_from_file_location("_kq_codec_geometry", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CODEC_GEOMETRY


def _parse_readme_table() -> dict[str, tuple[int, int, int]]:
    """Return ``codec -> (block, bits, bytes_per_block)`` from the README table."""
    text = (REPO / "README.md").read_text()
    out: dict[str, tuple[int, int, int]] = {}
    # Rows look like: | q2_k  | 256 | 2 |  84 | K-quant superblock |
    row = re.compile(
        r"^\|\s*(q\d[\w]*)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
        re.MULTILINE,
    )
    for m in row.finditer(text):
        codec, block, bits, bpb = m.group(1), *map(int, m.group(2, 3, 4))
        out[codec] = (block, bits, bpb)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero on any drift (for CI)",
    )
    args = ap.parse_args(argv)

    geom = _load_geometry()
    readme = _parse_readme_table()

    problems: list[str] = []

    # 1+2: geometry <-> README
    geom_codecs = set(geom)
    readme_codecs = set(readme)
    if geom_codecs != readme_codecs:
        only_geom = sorted(geom_codecs - readme_codecs)
        only_readme = sorted(readme_codecs - geom_codecs)
        if only_geom:
            problems.append(f"codecs in codec_geometry but not README: {only_geom}")
        if only_readme:
            problems.append(f"codecs in README but not codec_geometry: {only_readme}")

    print(f"{'codec':<6} {'block':>5} {'bits':>4} {'bytes/blk':>9}  geometry==README")
    for codec in sorted(geom_codecs & readme_codecs):
        gs, bits, bpb, wpb = geom[codec]
        r_block, r_bits, r_bpb = readme[codec]
        agree = (gs == r_block == wpb) and (bits == r_bits) and (bpb == r_bpb)
        if not agree:
            problems.append(
                f"{codec}: geometry (block={gs}/wpb={wpb}, bits={bits}, "
                f"bytes={bpb}) != README (block={r_block}, bits={r_bits}, "
                f"bytes={r_bpb})"
            )
        print(f"{codec:<6} {gs:>5} {bits:>4} {bpb:>9}  {'ok' if agree else 'DRIFT'}")

    # 3: runtime list from the built extension (optional).
    try:
        spec = importlib.util.find_spec("mlx_kquant")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        try:
            import mlx_kquant as kq

            runtime = set(kq.codecs())
            if runtime != geom_codecs:
                problems.append(
                    f"kq.codecs() {sorted(runtime)} != codec_geometry "
                    f"{sorted(geom_codecs)}"
                )
            else:
                print(f"\nkq.codecs() matches codec_geometry ({len(runtime)} codecs)")
        except Exception as e:  # ext present but unbuilt / unloadable
            print(f"\n(skipped kq.codecs() leg: {type(e).__name__}: {e})")
    else:
        print("\n(skipped kq.codecs() leg: mlx_kquant extension not importable)")

    if problems:
        print("\nDRIFT DETECTED:")
        for p in problems:
            print(f"  - {p}")
        return 1 if args.check else 0
    print("\nall codec facts agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
