#!/usr/bin/env python3
"""Dequant validation: kq.dequantize (or the fork's mx.dequantize) vs the
`gguf.quants` numpy reference.

Backend-agnostic so the SAME script runs in two environments and the outputs can
be diffed to prove byte-parity between the extension and the fork:

  * stock mlx + mlx_kquant  -> uses kq.dequantize           (BACKEND=mlx_kquant)
  * the kquant fork wheel    -> uses mx.dequantize(mode=...)  (BACKEND=fork-mx)

For each quantized tensor of the requested codec(s) it compares:
  * default (float16) output  -> expect LOOSE vs the float32 reference
  * float32 output            -> tests the dequant math directly (may be bit-exact)

Usage:
    test_dequant.py <gguf> [--codecs q4_k,q8_0] [--limit N]

Exit non-zero on any hard failure (neither bit-exact nor within loose tol).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, quants

# (weights_per_block, bytes_per_block, group_size, bits, codec_name)
KQUANT_CODECS = {
    GGMLQuantizationType.Q4_0: (32, 18, 32, 4, "q4_0"),
    GGMLQuantizationType.Q4_1: (32, 20, 32, 4, "q4_1"),
    GGMLQuantizationType.Q5_0: (32, 22, 32, 5, "q5_0"),
    GGMLQuantizationType.Q5_1: (32, 24, 32, 5, "q5_1"),
    GGMLQuantizationType.Q8_0: (32, 34, 32, 8, "q8_0"),
    GGMLQuantizationType.Q2_K: (256, 84, 256, 2, "q2_k"),
    GGMLQuantizationType.Q3_K: (256, 110, 256, 3, "q3_k"),
    GGMLQuantizationType.Q4_K: (256, 144, 256, 4, "q4_k"),
    GGMLQuantizationType.Q5_K: (256, 176, 256, 5, "q5_k"),
    GGMLQuantizationType.Q6_K: (256, 210, 256, 6, "q6_k"),
}

ATOL_LOOSE = 1e-3
RTOL_LOOSE = 1e-3

# Resolve a backend: prefer the extension, fall back to the fork's core op.
try:
    import mlx_kquant as kq

    BACKEND = "mlx_kquant"

    def _dequant(w, scales, gs, bits, codec, dtype):
        return kq.dequantize(w, scales, codec, dtype=dtype)

except ImportError:
    BACKEND = "fork-mx"

    def _dequant(w, scales, gs, bits, codec, dtype):
        return mx.dequantize(
            w,
            scales,
            group_size=gs,
            bits=bits,
            mode="kquant",
            kquant_type=codec,
            dtype=dtype,
        )


def _pack(tensor, wpb, bpb):
    raw = np.ascontiguousarray(tensor.data, dtype=np.uint8)
    ref = quants.dequantize(raw, tensor.tensor_type).astype(np.float32)
    logical = [int(d) for d in tensor.shape][::-1]
    last = logical[-1]
    packed = list(logical)
    packed[-1] = (last // wpb) * bpb
    return raw.reshape(packed), ref


def _status(ref, got):
    if np.array_equal(ref, got):
        return "bit_exact", 0.0, 0.0
    diff = np.abs(ref - got)
    max_abs = float(diff.max())
    max_rel = float((diff / (np.abs(ref) + 1e-12)).max())
    if np.allclose(ref, got, atol=ATOL_LOOSE, rtol=RTOL_LOOSE):
        return "loose", max_abs, max_rel
    return "fail", max_abs, max_rel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--codecs", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    allow = {c.strip().lower() for c in args.codecs.split(",") if c.strip()}
    reader = GGUFReader(args.gguf, "r")
    # per codec: {f16: {status: n}, f32: {status: n}}
    counts = defaultdict(lambda: {"f16": defaultdict(int), "f32": defaultdict(int)})
    worst = defaultdict(lambda: {"f16": (0.0, 0.0), "f32": (0.0, 0.0)})
    n = 0
    hard_fail = 0

    print(f"=== test_dequant [{BACKEND}]: {args.gguf} ===")
    for t in reader.tensors:
        geom = KQUANT_CODECS.get(t.tensor_type)
        if geom is None:
            continue
        wpb, bpb, gs, bits, codec = geom
        if allow and codec not in allow:
            continue
        if args.limit and n >= args.limit:
            break

        w_np, ref = _pack(t, wpb, bpb)
        w = mx.array(w_np)
        scales = mx.zeros((1,), dtype=mx.uint8)

        for tag, dt in (("f16", None), ("f32", mx.float32)):
            out = _dequant(w, scales, gs, bits, codec, dt)
            mx.eval(out)
            got = np.array(out).astype(np.float32).reshape(ref.shape)
            st, ma, mr = _status(ref, got)
            counts[codec][tag][st] += 1
            if ma > worst[codec][tag][0]:
                worst[codec][tag] = (ma, mr)
            if st == "fail":
                hard_fail += 1
                print(
                    f"  FAIL {t.name} {codec} [{tag}] max_abs={ma:.3e}", file=sys.stderr
                )
            elif args.verbose:
                print(
                    f"  {t.name:<55} {codec} [{tag}] {st} "
                    f"max_abs={ma:.3e} max_rel={mr:.3e}"
                )
        n += 1

    print(f"\nvalidated {n} tensors")
    print(
        f"  {'codec':<6} {'out':>4} {'bit_exact':>10} {'loose':>6} "
        f"{'fail':>5} {'worst_abs':>11} {'worst_rel':>11}"
    )
    for codec in sorted(counts):
        for tag in ("f16", "f32"):
            c = counts[codec][tag]
            wa, wr = worst[codec][tag]
            print(
                f"  {codec:<6} {tag:>4} {c['bit_exact']:>10} {c['loose']:>6} "
                f"{c['fail']:>5} {wa:>11.3e} {wr:>11.3e}"
            )
    if n == 0:
        # No matching tensors validated nothing — a misspelled --codecs or a
        # GGUF without these codecs must not masquerade as a pass.
        print("ERROR: no tensors matched the requested codec(s)", file=sys.stderr)
        return 2
    return 1 if hard_fail else 0


def test_dequant():
    """pytest entry: runs against KQUANT_TEST_GGUF, else skips (never silent)."""
    import pytest

    gguf = os.environ.get("KQUANT_TEST_GGUF")
    if not gguf:
        pytest.skip("set KQUANT_TEST_GGUF=<path> to run the GGUF dequant test")
    assert main([gguf]) == 0


if __name__ == "__main__":
    sys.exit(main())
