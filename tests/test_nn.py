#!/usr/bin/env python3
"""Validate the mlx_kquant.nn modules against an INDEPENDENT oracle.

KQuantLinear / KQuantEmbedding / KQuantSwitchLinear store GGUF wire bytes and
dispatch the kq.* ops. Here we feed them the exact same wire bytes used by
test_codecs / test_gather (committed K-quant fixtures + gguf-py-synthesized flat
codecs) and compare their forward output to a reference computed with **gguf-py's
numpy decoder** + numpy matmul - NOT kq.dequantize of the same bytes (that would
be circular and prove nothing).

Forward only (no kq.quantize), so this runs on the CPU decode path once it
lands; until then it needs an Apple-Silicon GPU like the other op tests.

Usage: test_nn.py [--codecs q4_k,q8_0,...]
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType as GT
from gguf import quants

from mlx_kquant.nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear

# codec -> (gguf type, is_kquant)
CODECS = {
    "q4_0": (GT.Q4_0, False),
    "q4_1": (GT.Q4_1, False),
    "q5_0": (GT.Q5_0, False),
    "q5_1": (GT.Q5_1, False),
    "q8_0": (GT.Q8_0, False),
    "q2_k": (GT.Q2_K, True),
    "q3_k": (GT.Q3_K, True),
    "q4_k": (GT.Q4_K, True),
    "q5_k": (GT.Q5_K, True),
    "q6_k": (GT.Q6_K, True),
}

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
N, K = 256, 512
MOE_E, MOE_N, MOE_K = 4, 128, 512
TOL = 2e-2  # relative Frobenius, f16 matmul vs f32 oracle


def _lin_wire_ref(codec, gtype, is_kquant):
    """(wire uint8[N, packed], ref float32[N, K]) for the Linear/Embedding case."""
    if is_kquant:
        path = os.path.join(FIX, f"{codec}.npz")
        if not os.path.exists(path):
            return None, None
        wire = np.load(path)["wire"].astype(np.uint8)
    else:
        rng = np.random.default_rng(7)
        w = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        wire = quants.quantize(w, gtype).astype(np.uint8)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
    return wire, ref


def _moe_wire_ref(codec, gtype, is_kquant):
    """(wire uint8[E, N, packed], ref float32[E, N, K]) for the SwitchLinear case."""
    if is_kquant:
        path = os.path.join(FIX, f"{codec}_moe.npz")
        if not os.path.exists(path):
            return None, None
        wire = np.load(path)["wire"].astype(np.uint8)
        refs = [
            quants.dequantize(np.ascontiguousarray(wire[e]), gtype)
            for e in range(wire.shape[0])
        ]
        return wire, np.stack(refs, 0).astype(np.float32)
    rng = np.random.default_rng(11)
    wires, refs = [], []
    for _ in range(MOE_E):
        we = rng.standard_normal((MOE_N, MOE_K)).astype(np.float32) * 0.1
        wq = quants.quantize(we, gtype).astype(np.uint8)
        wires.append(wq)
        refs.append(quants.dequantize(np.ascontiguousarray(wq), gtype))
    return np.stack(wires, 0), np.stack(refs, 0).astype(np.float32)


def _rel(got, ref) -> float:
    g = np.array(got).astype(np.float32)
    r = np.array(ref).astype(np.float32)
    return float(np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-6))


def _check_linear(codec, gtype, wire, ref) -> bool:
    rows, kk = ref.shape
    lin = KQuantLinear(in_dims=kk, out_dims=rows, bias=False, codec=codec)
    lin.weight = mx.array(wire)
    rng = np.random.default_rng(0)
    x = mx.array((rng.standard_normal((8, kk)) * 0.1).astype(np.float32)).astype(
        mx.float16
    )
    got = lin(x)
    oracle = x @ mx.array(ref).astype(mx.float16).T
    mx.eval(got, oracle)
    return _rel(got, oracle) < TOL and got.shape == oracle.shape


def _check_embedding(codec, gtype, wire, ref) -> bool:
    rows, kk = ref.shape
    emb = KQuantEmbedding(num_embeddings=rows, dims=kk, codec=codec)
    emb.weight = mx.array(wire)
    idx_np = np.array([[0, 3, 7], [rows - 1, 1, 5]], dtype=np.uint32)
    out = emb(mx.array(idx_np))
    mx.eval(out)
    # gather/dequant is loose (f16) vs the f32 oracle rows.
    gather_ok = np.allclose(
        np.array(out).astype(np.float32), ref[idx_np], atol=1e-3, rtol=1e-3
    )
    # tied as_linear == quantized_matmul against the same weight.
    rng = np.random.default_rng(1)
    x = mx.array((rng.standard_normal((4, kk)) * 0.1).astype(np.float32)).astype(
        mx.float16
    )
    proj = emb.as_linear(x)
    oracle = x @ mx.array(ref).astype(mx.float16).T
    mx.eval(proj, oracle)
    return gather_ok and _rel(proj, oracle) < TOL


def _check_switch(codec, gtype, wire, ref) -> bool:
    ne, nn_, kk = ref.shape
    sw = KQuantSwitchLinear(
        num_experts=ne, output_dims=nn_, input_dims=kk, bias=False, codec=codec
    )
    sw.weight = mx.array(wire)
    rng = np.random.default_rng(2)
    B, M = 6, 4
    x = mx.array((rng.standard_normal((B, M, kk)) * 0.1).astype(np.float32)).astype(
        mx.float16
    )
    experts = rng.integers(0, ne, size=B).astype(np.uint32)
    got = sw(x, mx.array(experts))
    deq = mx.array(ref).astype(mx.float16)
    oracle = mx.stack([x[b] @ deq[int(experts[b])].T for b in range(B)], axis=0)
    mx.eval(got, oracle)
    return _rel(got, oracle) < TOL and got.shape == oracle.shape


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="")
    args = ap.parse_args(argv)
    allow = {c.strip() for c in args.codecs.split(",") if c.strip()}

    print("=== test_nn [mlx_kquant] ===")
    print(f"  {'codec':<6} {'Linear':>8} {'Embed':>8} {'Switch':>8} {'verdict':>8}")
    fails = 0
    missing = []
    # Embedding/Switch are module-shape checks, not per-codec math; cover one
    # K-quant + one flat codec for each to keep the matrix small but real.
    emb_sw_codecs = {"q4_k", "q8_0"}
    for codec, (gtype, is_kq) in CODECS.items():
        if allow and codec not in allow:
            continue
        wire, ref = _lin_wire_ref(codec, gtype, is_kq)
        if wire is None:
            missing.append(codec)
            fails += 1
            print(f"  {codec:<6} {'MISSING fixture - run gen_fixtures.py':>40}")
            continue

        lin_ok = _check_linear(codec, gtype, wire, ref)
        emb_ok = sw_ok = True
        if codec in emb_sw_codecs:
            emb_ok = _check_embedding(codec, gtype, wire, ref)
            mwire, mref = _moe_wire_ref(codec, gtype, is_kq)
            if mwire is None:
                missing.append(f"{codec}_moe")
                sw_ok = False
            else:
                sw_ok = _check_switch(codec, gtype, mwire, mref)

        bad = not (lin_ok and emb_ok and sw_ok)
        fails += bad

        def _m(ok, tested):
            return ("ok" if ok else "FAIL") if tested else "-"

        print(
            f"  {codec:<6} {_m(lin_ok, True):>8} "
            f"{_m(emb_ok, codec in emb_sw_codecs):>8} "
            f"{_m(sw_ok, codec in emb_sw_codecs):>8} "
            f"{'FAIL' if bad else 'ok':>8}"
        )

    if missing:
        print(f"\nMISSING fixtures (FAIL): {', '.join(missing)}")
    print(f"{'FAILURES: ' + str(fails) if fails else 'ALL OK'}")
    return 1 if fails else 0


def test_nn():
    """pytest entry: runs the full nn-module sweep (fixtures are committed)."""
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main())
