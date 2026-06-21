#!/usr/bin/env python3
"""Validate all 10 K-quant codecs (dequant + matmul) end to end.

Test data per codec:
  * flat codecs (q4_0/q4_1/q5_0/q5_1/q8_0): synthesized in-process via
    gguf.quants.quantize(random weights) - gguf-py can encode these.
  * K-quant codecs (q2_k..q6_k): loaded from tests/fixtures/<codec>.npz, minted
    by gen_fixtures.py via kq.quantize (gguf-py is decode-only for these).

For each codec it checks, against the gguf.quants numpy reference:
  * dequantize -> float32 : must be BIT-EXACT (enforced, not just "loose")
  * dequantize -> float16 : expect LOOSE (downcast)
  * quantized_matmul(x, w, transpose=True)  ~= x @ dequant(w).T : within f16 tol
  * quantized_matmul(x, w, transpose=False) ~= x @ dequant(w)   : within f16 tol
    (so both the qmm_t and qmm_n Metal kernels are exercised)

Exit non-zero on any failure.

Usage: test_codecs.py [--codecs q4_k,q8_0,...]
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

# codec -> (gguf type, weights_per_block, bytes_per_block, bits, is_kquant)
CODECS = {
    "q4_0": (GT.Q4_0, 32, 18, 4, False),
    "q4_1": (GT.Q4_1, 32, 20, 4, False),
    "q5_0": (GT.Q5_0, 32, 22, 5, False),
    "q5_1": (GT.Q5_1, 32, 24, 5, False),
    "q8_0": (GT.Q8_0, 32, 34, 8, False),
    "q2_k": (GT.Q2_K, 256, 84, 2, True),
    "q3_k": (GT.Q3_K, 256, 110, 3, True),
    "q4_k": (GT.Q4_K, 256, 144, 4, True),
    "q5_k": (GT.Q5_K, 256, 176, 5, True),
    "q6_k": (GT.Q6_K, 256, 210, 6, True),
    "iq4_nl": (GT.IQ4_NL, 32, 18, 4, False),
    "iq4_xs": (GT.IQ4_XS, 256, 136, 4, False),
    "iq3_s": (GT.IQ3_S, 256, 110, 3, False),
    "iq3_xxs": (GT.IQ3_XXS, 256, 98, 3, False),
    "iq2_xxs": (GT.IQ2_XXS, 256, 66, 2, False),
    "iq2_xs": (GT.IQ2_XS, 256, 74, 2, False),
    "iq2_s": (GT.IQ2_S, 256, 82, 2, False),
    "iq1_s": (GT.IQ1_S, 256, 50, 1, False),
    "iq1_m": (GT.IQ1_M, 256, 56, 1, False),
}

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
N, K = 256, 512

BACKEND = "mlx_kquant"


def _dequant(w, sc, gs, bits, codec, dt):
    return kq.dequantize(w, sc, codec, dtype=dt)


def _qmm(x, w, sc, gs, bits, codec, transpose=True):
    return kq.quantized_matmul(x, w, sc, codec, transpose=transpose)


def _synth_iq_wire(rng, bpb, n_blocks):
    """Structurally-valid random IQ wire (gguf-py is decode-only for IQ): random
    bytes with a sane fp16 d at block offset 0 so dequant can't hit Inf/NaN."""
    wire = rng.integers(0, 256, size=(n_blocks, bpb), dtype=np.uint8)
    d = rng.uniform(0.02, 0.08, n_blocks).astype(np.float16)
    if bpb == 56:
        # IQ1_M has no super-block d; its fp16 scale is rebuilt from the top
        # nibbles of the four uint16 scale words (bytes 49/51/53/55). Seed those
        # so the reconstructed scale is a sane (non-NaN) fp16.
        dbits = d.view(np.uint16)
        for k, byteidx in enumerate((49, 51, 53, 55)):
            nib = ((dbits >> (4 * k)) & 0xF).astype(np.uint8)
            wire[:, byteidx] = (wire[:, byteidx] & 0x0F) | (nib << 4)
    else:
        wire[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 2)
    return wire


def _wire_and_ref(codec, gtype, wpb, bpb, is_kquant):
    """Return (wire uint8[N, packed], ref float32[N, K])."""
    if codec.startswith("iq"):
        wire = _synth_iq_wire(np.random.default_rng(7), bpb, N * (K // wpb))
        wire = wire.reshape(N, (K // wpb) * bpb)
        ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
        return wire, ref
    if is_kquant:
        path = os.path.join(FIX, f"{codec}.npz")
        if not os.path.exists(path):
            return None, None
        z = np.load(path)
        wire = z["wire"].astype(np.uint8)
        ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
        return wire, ref
    # flat codec: synthesize
    rng = np.random.default_rng(7)
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    wire = quants.quantize(w, gtype).astype(np.uint8)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
    return wire, ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="")
    args = ap.parse_args(argv)
    allow = {c.strip() for c in args.codecs.split(",") if c.strip()}

    print(f"=== test_codecs [{BACKEND}] ===")
    print(
        f"  {'codec':<6} {'deq_f32':>9} {'deq_f16':>9} {'mm_rel_t':>10} "
        f"{'mm_rel_n':>10} {'verdict':>8}"
    )
    fails = 0
    missing = []
    for codec, (gtype, wpb, bpb, bits, is_kq) in CODECS.items():
        if allow and codec not in allow:
            continue
        wire, ref = _wire_and_ref(codec, gtype, wpb, bpb, is_kq)
        if wire is None:
            # A requested codec with no fixture is a HARD failure, not a silent
            # skip: otherwise a fresh clone / git-clean drops all K-quant
            # coverage and still prints ALL OK.
            missing.append(codec)
            fails += 1
            print(f"  {codec:<6} {'MISSING fixture - run gen_fixtures.py':>49}")
            continue

        rows, kk = ref.shape
        w = mx.array(wire)
        scales = mx.zeros((1,), dtype=mx.uint8)

        # dequant f32 (must be BIT-EXACT vs the gguf reference) and f16 (loose)
        out32 = _dequant(w, scales, wpb, bits, codec, mx.float32)
        out16 = _dequant(w, scales, wpb, bits, codec, mx.float16)
        mx.eval(out32, out16)
        a32 = np.array(out32).astype(np.float32).reshape(ref.shape)
        a16 = np.array(out16).astype(np.float32).reshape(ref.shape)
        deq_f32 = (
            "bit_exact"
            if np.array_equal(ref, a32)
            else ("loose" if np.allclose(ref, a32, atol=1e-3, rtol=1e-3) else "FAIL")
        )
        deq_f16 = "loose" if np.allclose(ref, a16, atol=1e-3, rtol=1e-3) else "FAIL"

        # matmul vs dequant-then-matmul. Use relative Frobenius norm
        # ||got-ref|| / ||ref||: robust to individual near-zero entries (which
        # make per-element max-rel blow up, esp. for coarse low-bit codecs).
        # Exercise BOTH transpose=True (qmm_t: x @ deq.T) and transpose=False
        # (qmm_n: x @ deq), so neither Metal kernel is left untested.
        deq16 = mx.array(ref).astype(mx.float16)  # [N, K]
        rng = np.random.default_rng(0)
        mm_t = 0.0
        mm_n = 0.0
        mm_fail = False
        for M in (1, 64):
            xt = mx.array(
                (rng.standard_normal((M, kk)) * 0.1).astype(np.float32)
            ).astype(mx.float16)
            got_t = _qmm(xt, w, scales, wpb, bits, codec, transpose=True)
            r_t = xt @ deq16.T  # [M, N]
            xn = mx.array(
                (rng.standard_normal((M, rows)) * 0.1).astype(np.float32)
            ).astype(mx.float16)
            got_n = _qmm(xn, w, scales, wpb, bits, codec, transpose=False)
            r_n = xn @ deq16  # [M, K]
            mx.eval(got_t, r_t, got_n, r_n)
            for got, r, slot in ((got_t, r_t, "t"), (got_n, r_n, "n")):
                g = np.array(got).astype(np.float32)
                rr = np.array(r).astype(np.float32)
                rel = float(np.linalg.norm(g - rr) / (np.linalg.norm(rr) + 1e-6))
                if slot == "t":
                    mm_t = max(mm_t, rel)
                else:
                    mm_n = max(mm_n, rel)
                if rel >= 2e-2 or g.shape != rr.shape:
                    mm_fail = True

        # f32 dequant must be bit-exact (loose silently masks a real math bug).
        bad = (deq_f32 != "bit_exact") or (deq_f16 == "FAIL") or mm_fail
        fails += bad
        print(
            f"  {codec:<6} {deq_f32:>9} {deq_f16:>9} {mm_t:>10.3e} "
            f"{mm_n:>10.3e} {'FAIL' if bad else 'ok':>8}"
        )

    if missing:
        print(f"\nMISSING fixtures (FAIL): {', '.join(missing)}")
    print(f"{'FAILURES: ' + str(fails) if fails else 'ALL OK'}")
    return 1 if fails else 0


def test_codecs():
    """pytest entry: runs the full codec sweep (no external data needed)."""
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main())
