#!/usr/bin/env python3
"""Compressed size of the built mlx_kquant package, as its wheel will carry
it, and the kernel families that take the most room in the metallib.

  python scripts/check-wheel-size.py              # report
  python scripts/check-wheel-size.py --check      # exit 1 past the budget
  python scripts/check-wheel-size.py --top 30     # longer family list

PyPI refuses any file of this project over 100 MB, and a wheel is a zip of
the package with each file deflated, so the estimate deflates each file of
the built package the way the wheel writer does. --check fails at 95 MB,
which leaves room for the estimate's error. Without a metallib (a CPU-only
build) it reports that and exits 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

LIMIT_MB = 100.0
CHECK_MB = 95.0
METALLIB = "mlx_kquant.metallib"


def package_dir() -> Path:
    spec = importlib.util.find_spec("mlx_kquant")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("mlx_kquant is not importable; build it first")
    return Path(next(iter(spec.submodule_search_locations)))


def wheel_bytes(pkg: Path) -> int:
    """Deflated size of every file the wheel would carry from pkg."""
    total = 0
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            c = zlib.compressobj(6, zlib.DEFLATED, -15)
            n = 0
            with open(p, "rb") as f:
                while chunk := f.read(1 << 24):
                    n += len(c.compress(chunk))
            total += n + len(c.flush())
    return total


def functions(path: Path) -> list[tuple[str, int]]:
    """(name, bitcode bytes) of each function in a metallib."""
    b = path.read_bytes()
    if b[:4] != b"MTLB":
        return []
    fl_off = struct.unpack_from("<Q", b, 24)[0]
    count = struct.unpack_from("<I", b, fl_off)[0]
    p, out = fl_off + 4, []
    for _ in range(count):
        end = p + struct.unpack_from("<I", b, p)[0]
        q, name, size = p + 4, None, 0
        while q < end and b[q : q + 4] != b"ENDT":
            n = struct.unpack_from("<H", b, q + 4)[0]
            data = b[q + 6 : q + 6 + n]
            if b[q : q + 4] == b"NAME":
                name = data.rstrip(b"\0").decode()
            elif b[q : q + 4] == b"MDSZ":
                size = struct.unpack_from("<Q", data)[0]
            q += 6 + n
        if name:
            out.append((name, size))
        p = end
    return out


def family(name: str, codecs: list[str]) -> str:
    """A kernel name with its codecs as C and its tile and dtype tail cut."""
    for c in codecs:
        name = re.sub(rf"(?<=_){re.escape(c)}(?=_)", "C", name)
    name = re.split(r"_(?:bfloat16_t|float16_t|float|kq_x16)(?:_|$)", name)[0]
    return re.sub(r"_(?:nx16|nx32|c32|g\d|fine)(?=_|$)", "", name)


def codec_names() -> list[str]:
    root = Path(__file__).resolve().parent.parent
    text = (root / "mlx_kquant" / "codec_geometry.py").read_text()
    names = set(re.findall(r'^\s+"([a-z0-9_]+)": \(', text, re.M))
    return sorted(names, key=len, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    ap.add_argument("--check", action="store_true", help="exit 1 past the budget")
    ap.add_argument("--top", type=int, default=15, help="families to list")
    ap.add_argument("--package", type=Path, help="package dir (default: import path)")
    args = ap.parse_args()

    pkg = args.package or package_dir()
    lib = pkg / METALLIB
    if not lib.is_file():
        print(f"no {METALLIB} in {pkg}: nothing to check")
        return 0
    mb = wheel_bytes(pkg) / 1e6
    print(f"wheel estimate {mb:.1f} MB", end=" ")
    print(f"(check at {CHECK_MB:.0f}, PyPI limit {LIMIT_MB:.0f})")
    print(f"metallib {lib.stat().st_size / 1e6:.1f} MB on disk")

    fns = functions(lib)
    if fns:
        codecs = codec_names()
        size, count = defaultdict(int), defaultdict(int)
        for name, n in fns:
            f = family(name, codecs)
            size[f] += n
            count[f] += 1
        total = sum(size.values())
        print(f"\n{len(fns)} functions, {total / 1e6:.1f} MB of bitcode.")
        print("Largest families:")
        print("| MB | functions | family |\n|---|---|---|")
        for f in sorted(size, key=size.get, reverse=True)[: args.top]:
            print(f"| {size[f] / 1e6:.1f} | {count[f]} | `{f}` |")

    if args.check and mb > CHECK_MB:
        print(f"\nFAIL: the wheel estimate is over {CHECK_MB:.0f} MB")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
