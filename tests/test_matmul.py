#!/usr/bin/env python3
"""Quantized-matmul validation: kq.quantized_matmul vs an INDEPENDENT
dequant-then-matmul reference.

The reference dequant comes from gguf-py's numpy decoder, NOT the extension, so
a shared bug in the dequant math cannot cancel out of both sides of the
comparison (a circular check against kq.dequantize would hide exactly that).

For a real K-quant weight tensor pulled from a GGUF it checks, across several M
(to exercise qmv at small M and qmm / qmm_nax at large M):

    quantized_matmul(x, w, transpose=True)  ~=  x @ dequant(w).T

Under pytest the GGUF-based check runs only when KQUANT_TEST_GGUF=<path> is set
(otherwise it is reported as skipped, not silently absent).

Usage:
    test_matmul.py <gguf> [--codec q4_k] [--ms 1,2,64]
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, quants

import mlx_kquant as kq

# codec -> (gguf type, weights_per_block, bytes_per_block, bits)
CODEC_BY_NAME = {
    "q4_0": (GGMLQuantizationType.Q4_0, 32, 18, 4),
    "q4_1": (GGMLQuantizationType.Q4_1, 32, 20, 4),
    "q5_0": (GGMLQuantizationType.Q5_0, 32, 22, 5),
    "q5_1": (GGMLQuantizationType.Q5_1, 32, 24, 5),
    "q8_0": (GGMLQuantizationType.Q8_0, 32, 34, 8),
    "q2_k": (GGMLQuantizationType.Q2_K, 256, 84, 2),
    "q3_k": (GGMLQuantizationType.Q3_K, 256, 110, 3),
    "q4_k": (GGMLQuantizationType.Q4_K, 256, 144, 4),
    "q5_k": (GGMLQuantizationType.Q5_K, 256, 176, 5),
    "q6_k": (GGMLQuantizationType.Q6_K, 256, 210, 6),
}

BACKEND = "mlx_kquant"


def _qmm(x, w, scales, gs, bits, codec, transpose):
    return kq.quantized_matmul(x, w, scales, codec, transpose=transpose)


def _find_weight(reader, gtype, wpb, bpb, min_k=256):
    """First 2D tensor of this codec whose logical [N, K] has K % wpb == 0."""
    for t in reader.tensors:
        if t.tensor_type != gtype:
            continue
        logical = [int(d) for d in t.shape][::-1]
        if len(logical) != 2:
            continue
        n, k = logical
        if k % wpb == 0 and k >= min_k and n >= 8:
            raw = np.ascontiguousarray(t.data, dtype=np.uint8)
            packed = raw.reshape([n, (k // wpb) * bpb])
            return t.name, n, k, packed
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--codec", default="q4_k")
    ap.add_argument("--ms", default="1,2,8,64,256")
    args = ap.parse_args(argv)

    gtype, wpb, bpb, bits = CODEC_BY_NAME[args.codec]
    reader = GGUFReader(args.gguf, "r")
    found = _find_weight(reader, gtype, wpb, bpb)
    if found is None:
        print(f"no usable 2D {args.codec} tensor in {args.gguf}", file=sys.stderr)
        return 2
    name, N, K, packed = found
    print(f"=== test_matmul [{BACKEND}]: {args.codec} {name} [N={N}, K={K}] ===")

    w = mx.array(packed)
    scales = mx.zeros((1,), dtype=mx.uint8)
    # Independent reference: gguf-py's numpy decoder AND a numpy f32 matmul (NOT
    # kq.dequantize, and NOT mx.matmul) - so neither a shared dequant bug nor a
    # shared matmul bug can cancel out of both `got` and `ref`. numpy on CPU also
    # stays correct for a wide weight (>2GB, N~262144) where an f16 matmul
    # overflows to inf and even stock mx.matmul overflows int32 row offsets to 0;
    # either would mask or fake a kernel failure. CPU f32 is the trustworthy oracle.
    # [N, K]
    deq = quants.dequantize(np.ascontiguousarray(packed), gtype).astype(np.float32)

    rng = np.random.default_rng(0)
    worst = 0.0
    fail = 0
    print(f"  {'M':>5} {'shape':>14} {'max_abs':>11} {'max_rel':>11} {'status':>8}")
    for M in (int(m) for m in args.ms.split(",")):
        x_np = rng.standard_normal((M, K)).astype(np.float32) * 0.1
        x = mx.array(x_np).astype(mx.float16)
        got = _qmm(x, w, scales, wpb, bits, args.codec, True)  # [M, N]
        mx.eval(got)
        g = np.array(got).astype(np.float32)
        # reference matmul in numpy f32, using the f16-rounded x the kernel saw.
        r = np.array(x).astype(np.float32) @ deq.T
        diff = np.abs(g - r)
        max_abs = float(diff.max())
        denom = np.abs(r) + 1e-3
        max_rel = float((diff / denom).max())
        # f16 accumulation vs the on-the-fly kernel: ~1e-2 rel is expected.
        ok = max_rel < 5e-2 or max_abs < 5e-3
        worst = max(worst, max_rel)
        if not ok:
            fail += 1
        print(
            f"  {M:>5} {str(tuple(g.shape)):>14} {max_abs:>11.3e} "
            f"{max_rel:>11.3e} {'ok' if ok else 'FAIL':>8}"
        )

    print(f"\nworst max_rel = {worst:.3e}; {'PASS' if not fail else f'{fail} FAIL'}")
    return 1 if fail else 0


def _present_codecs(gguf) -> list:
    """K-quant codec names from CODEC_BY_NAME that actually appear in the file,
    in CODEC_BY_NAME order. Lets the pytest entry test whatever a given model
    happens to contain instead of assuming a fixed codec."""
    reader = GGUFReader(gguf, "r")
    present = {t.tensor_type for t in reader.tensors}
    return [name for name, (gt, *_r) in CODEC_BY_NAME.items() if gt in present]


def test_matmul():
    """pytest entry: validate quantized_matmul for EVERY K-quant codec present
    in KQUANT_TEST_GGUF (not just q4_k - UD/mixed quants vary), else skip.
    main() returns 0=pass, 1=numeric mismatch, 2=codec present but no usable 2D
    tensor. A 2 is tolerated (not every codec has a matmul-shaped tensor); only a
    1 fails. Requires at least one codec to actually run so a mis-pointed path
    can't pass vacuously."""
    import pytest

    gguf = os.environ.get("KQUANT_TEST_GGUF")
    if not gguf:
        pytest.skip("set KQUANT_TEST_GGUF=<path> to run the GGUF matmul test")

    codecs = _present_codecs(gguf)
    if not codecs:
        pytest.skip(f"no supported K-quant codec in {gguf}")

    ran = 0
    for codec in codecs:
        rc = main([gguf, "--codec", codec])
        assert rc != 1, f"{codec}: quantized_matmul mismatch vs dequant ref"
        ran += rc == 0
    assert ran > 0, f"no usable 2D tensor for any present codec in {gguf}"


if __name__ == "__main__":
    sys.exit(main())
