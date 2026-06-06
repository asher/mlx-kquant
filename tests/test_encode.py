#!/usr/bin/env python3
"""M5: validate kq.quantize (encode) for all 10 K-quant codecs.

For a fixed random weight tensor, per codec it checks:
  * scales placeholder shape is [1];
  * round-trip: gguf.quants.dequantize(encode(w)) ~= w within a sane bound
    (gguf-py decodes every codec, so this is a real reconstruction check);
  * the imatrix actually steers the encoder: K-quant wire bytes MUST change
    when an imatrix is supplied, flat codecs MUST NOT (they have no imatrix path);
  * a stable sha1 of the wire bytes, with and without an imatrix.

Backend-agnostic (kq.quantize or the fork's mx.quantize), so the SAME script
runs under stock-mlx+mlx_kquant and under the kquant fork. Run it in both venvs
and diff the sha1 columns: identical sha1 == byte-identical encoder output.

Exit non-zero on any local failure (shape / round-trip). Byte-parity across
backends is confirmed by comparing the printed sha1s between the two runs.

Usage: test_encode.py [--codecs q4_k,q8_0,...]
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType as GT, quants

# codec -> (gguf type, weights_per_block, bits, round-trip rel-Frobenius bound)
CODECS = {
    "q8_0": (GT.Q8_0, 32, 8, 0.02),
    "q4_0": (GT.Q4_0, 32, 4, 0.15),
    "q4_1": (GT.Q4_1, 32, 4, 0.15),
    "q5_0": (GT.Q5_0, 32, 5, 0.10),
    "q5_1": (GT.Q5_1, 32, 5, 0.10),
    "q6_k": (GT.Q6_K, 256, 6, 0.05),
    "q5_k": (GT.Q5_K, 256, 5, 0.08),
    "q4_k": (GT.Q4_K, 256, 4, 0.12),
    "q3_k": (GT.Q3_K, 256, 3, 0.25),
    "q2_k": (GT.Q2_K, 256, 2, 0.40),
}

N, K = 256, 512

try:
    import mlx_kquant as kq

    BACKEND = "mlx_kquant"

    def _encode(w, gs, bits, codec, imatrix):
        return kq.quantize(w, codec, imatrix=imatrix)

except ImportError:
    BACKEND = "fork-mx"

    def _encode(w, gs, bits, codec, imatrix):
        return mx.quantize(w, group_size=gs, bits=bits, mode="kquant",
                           kquant_type=codec, imatrix=imatrix)


def _sha(a) -> str:
    return hashlib.sha1(np.array(a).astype(np.uint8).tobytes()).hexdigest()[:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="")
    args = ap.parse_args(argv)
    allow = {c.strip() for c in args.codecs.split(",") if c.strip()}

    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)).astype(np.float32) * 0.1)
    w = mx.array(w_np)
    # A non-trivial importance vector (length K) to exercise the imatrix path.
    imat_np = (np.abs(rng.standard_normal(K)).astype(np.float32) + 0.1)
    imat = mx.array(imat_np)

    print(f"=== test_encode [{BACKEND}]  w[{N},{K}] ===")
    print(f"  {'codec':<6} {'scales':>7} {'rt_rel':>9} {'imat':>5} {'wq_sha1':>14} "
          f"{'wq_imat_sha1':>14} {'verdict':>8}")
    fails = 0
    for codec, (gtype, wpb, bits, bound) in CODECS.items():
        if allow and codec not in allow:
            continue
        wq, scales = _encode(w, wpb, bits, codec, None)
        wq_im, _ = _encode(w, wpb, bits, codec, imat)
        mx.eval(wq, scales, wq_im)

        scales_ok = tuple(np.array(scales).shape) == (1,)
        wire = np.array(wq).astype(np.uint8)
        w_rt = quants.dequantize(
            np.ascontiguousarray(wire), gtype).astype(np.float32)
        rel = float(np.linalg.norm(w_rt - w_np) / (np.linalg.norm(w_np) + 1e-6))

        # The imatrix steers importance-weighted rounding in the K-quant
        # encoders, so it MUST change their wire bytes; flat codecs (wpb==32)
        # have no imatrix path, so identical bytes are correct there. A K-quant
        # that ignores the imatrix (regression / unimplemented dispatch) yields
        # equal sha1s and would otherwise pass silently.
        is_kquant = wpb == 256
        imat_changed = _sha(wq) != _sha(wq_im)
        imat_ok = imat_changed if is_kquant else not imat_changed

        bad = (not scales_ok) or (rel >= bound) or (not imat_ok)
        fails += bad
        print(f"  {codec:<6} {str(scales_ok):>7} {rel:>9.3e} "
              f"{('chg' if imat_changed else 'same'):>5} {_sha(wq):>14} "
              f"{_sha(wq_im):>14} {'FAIL' if bad else 'ok':>8}")

    print(f"{'FAILURES: ' + str(fails) if fails else 'ALL OK'}")
    return 1 if fails else 0


def test_encode():
    """pytest entry: runs the full encode sweep (no external data needed)."""
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main())
