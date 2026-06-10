#!/usr/bin/env python3
"""Validate kq.gather_qmm (mixture-of-experts gathered matmul).

For each codec, builds E expert weight matrices (flat codecs synthesized in
process via gguf.quants.quantize; K-quants loaded from tests/fixtures/
<codec>_moe.npz minted by gen_fixtures.py), then checks

    gather_qmm(x, w, lhs_indices=lhs, rhs_indices=experts, transpose=True)
      ~=  stack_b( x[lhs[b]] @ dequant(w[experts[b]]).T )

across several M to exercise both the decode path (M=1 -> gather_qmv) and the
prefill path (M large -> gather_qmm / gather_qmm_nax), and across two lhs_indices
patterns (contiguous arange and a reversed/repeated selection) so non-contiguous
lhs gather is covered, not just the identity case.

Exit non-zero on any failure.

Usage: test_gather.py [--codecs q4_k,q8_0,...]
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
}

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
E, N, K = 4, 128, 512  # experts, out_dims, in_features (K % 256 and % 64 == 0)

BACKEND = "mlx_kquant"


def _gather(x, w, sc, gs, bits, codec, lhs, rhs):
    return kq.gather_qmm(
        x, w, sc, codec, lhs_indices=lhs, rhs_indices=rhs, transpose=True
    )


def _wire_and_ref(codec, gtype, wpb, bpb, is_kquant):
    """Return (wire uint8[E, N, packed], ref float32[E, N, K])."""
    if is_kquant:
        path = os.path.join(FIX, f"{codec}_moe.npz")
        if not os.path.exists(path):
            return None, None
        z = np.load(path)
        wire = z["wire"].astype(np.uint8)  # [E, N, packed]
        refs = [
            quants.dequantize(np.ascontiguousarray(wire[e]), gtype)
            for e in range(wire.shape[0])
        ]
        return wire, np.stack(refs, axis=0).astype(np.float32)
    # flat codec: synthesize E experts
    rng = np.random.default_rng(11)
    wires, refs = [], []
    for _ in range(E):
        we = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        wq = quants.quantize(we, gtype).astype(np.uint8)
        wires.append(wq)
        refs.append(quants.dequantize(np.ascontiguousarray(wq), gtype))
    return np.stack(wires, 0), np.stack(refs, 0).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="")
    ap.add_argument("--ms", default="1,64")
    args = ap.parse_args(argv)
    allow = {c.strip() for c in args.codecs.split(",") if c.strip()}
    ms = [int(m) for m in args.ms.split(",")]

    print(f"=== test_gather [{BACKEND}]  E={E} N={N} K={K} ===")
    print(f"  {'codec':<6} {'M':>4} {'B':>3} {'lhs':>5} {'rel_fro':>10} {'verdict':>8}")
    fails = 0
    missing = []
    rng = np.random.default_rng(0)
    for codec, (gtype, wpb, bpb, bits, is_kq) in CODECS.items():
        if allow and codec not in allow:
            continue
        wire, ref = _wire_and_ref(codec, gtype, wpb, bpb, is_kq)
        if wire is None:
            # A requested K-quant codec with no fixture is a HARD failure, not a
            # silent skip - otherwise a fresh clone reports ALL OK with zero
            # K-quant coverage.
            missing.append(codec)
            fails += 1
            print(f"  {codec:<6} {'MISSING fixture - run gen_fixtures.py':>49}")
            continue
        ne, nn, _ = ref.shape
        w = mx.array(wire)
        scales = mx.zeros((1,), dtype=mx.uint8)
        deq = mx.array(ref).astype(mx.float16)  # [E, N, K]

        for M in ms:
            B = 8
            x_np = (rng.standard_normal((B, M, K)) * 0.1).astype(np.float32)
            x = mx.array(x_np).astype(mx.float16)
            experts = (rng.integers(0, ne, size=B)).astype(np.uint32)
            rhs = mx.array(experts)
            # Two lhs_indices patterns: contiguous arange (the fast/identity
            # path), and a reversed selection with a forced repeat (realistic
            # MoE routing - one x row consumed twice, one unused). The latter
            # exercises non-contiguous lhs gather, which arange never does.
            seq = np.arange(B, dtype=np.uint32)
            perm = seq[::-1].copy()
            perm[1] = perm[0]  # repeat -> x[perm[0]] used twice, x[seq[1]] unused
            for tag, lhs_np in (("seq", seq), ("perm", perm)):
                lhs = mx.array(lhs_np)
                got = _gather(x, w, scales, wpb, bits, codec, lhs, rhs)
                # reference: per-output x[lhs[b]] @ deq[experts[b]].T
                ref_rows = mx.stack(
                    [x[int(lhs_np[b])] @ deq[int(experts[b])].T for b in range(B)],
                    axis=0,
                )
                mx.eval(got, ref_rows)
                g = np.array(got).astype(np.float32)
                rr = np.array(ref_rows).astype(np.float32)
                rel = float(np.linalg.norm(g - rr) / (np.linalg.norm(rr) + 1e-6))
                bad = rel >= 2e-2 or g.shape != rr.shape
                fails += bad
                print(
                    f"  {codec:<6} {M:>4} {B:>3} {tag:>5} {rel:>10.3e} "
                    f"{'FAIL' if bad else 'ok':>8}"
                )

    if missing:
        print(f"\nMISSING fixtures (FAIL): {', '.join(missing)}")
    print(f"{'FAILURES: ' + str(fails) if fails else 'ALL OK'}")
    return 1 if fails else 0


def test_gather():
    """pytest entry: runs the full gather sweep (fixtures are committed)."""
    assert main([]) == 0


def test_rhs_gather_large_M_no_short_overflow():
    """Regression: the sorted single-shot rhs-gather (rhs_nax leaf) must stay
    correct when the total gathered-row count exceeds 32767.

    The tail-tile row-count computation narrowed (M - tile_origin) to `short`
    before clamping it to the per-simdgroup tile height, so once the row count
    crossed SHRT_MAX the leading tiles got a poisoned height and silently
    produced garbage rows (every tile whose remaining span > 32767). Only the
    sorted, B>=16, B/E>=4, M==1 path reaches that leaf, and only a single
    >32k-token forward (no chunking) makes M large enough - so it never showed
    up in the small-M sweep above. Assert the sorted leaf matches the unsorted
    per-row leaf (independent kernel) across all rows, and spot-check truth.
    """
    # q8_0: synthesizable in-process (gguf only quantizes flat codecs) and has a
    # rhs_nax kernel. The overflow lives in the shared tail-tile template, so the
    # codec is immaterial - q8_0 exercises the same poisoned path as q6_k.
    codec, gtype = "q8_0", GT.Q8_0
    Eg, Ng, Kg = 8, 64, 256  # K % 64 == 0
    rng = np.random.default_rng(7)
    wires, refs = [], []
    for _ in range(Eg):
        we = rng.standard_normal((Ng, Kg)).astype(np.float32) * 0.1
        wq = quants.quantize(we, gtype).astype(np.uint8)
        wires.append(wq)
        refs.append(quants.dequantize(np.ascontiguousarray(wq), gtype))
    w = mx.array(np.stack(wires, 0))
    deq = mx.array(np.stack(refs, 0).astype(np.float32)).astype(mx.float16)
    scales = mx.zeros((1,), dtype=mx.uint8)

    # B > 32767 with B % 64 != 0 (forces the partial-M / tail-tile path) and an
    # uneven sorted routing (long per-expert runs spanning many tiles).
    B = 33000
    assert B % 64 != 0 and B > 32767
    experts = np.sort(rng.integers(0, Eg, size=B).astype(np.uint32))
    rhs = mx.array(experts)
    x = mx.array((rng.standard_normal((B, 1, Kg)) * 0.1).astype(np.float32)).astype(
        mx.float16
    )

    fast = kq.gather_qmm(
        x, w, scales, codec, rhs_indices=rhs, transpose=True, sorted_indices=True
    ).reshape(B, Ng)
    ref = kq.gather_qmm(
        x, w, scales, codec, rhs_indices=rhs, transpose=True, sorted_indices=False
    ).reshape(B, Ng)
    mx.eval(fast, ref)
    max_abs = float(mx.abs(fast - ref).max().item())
    assert max_abs < 1e-2, f"sorted rhs-gather diverges from per-row leaf: {max_abs}"

    # spot-check a leading row (the overflow corrupted the FIRST tiles) vs truth
    b = 0
    truth = (x[b].astype(mx.float32) @ deq[int(experts[b])].T.astype(mx.float32))[0]
    mx.eval(truth)
    assert float(mx.abs(fast[b] - truth).max().item()) < 5e-2


if __name__ == "__main__":
    sys.exit(main())
