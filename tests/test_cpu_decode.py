#!/usr/bin/env python3
"""Validate the CPU decode path (eval_cpu) for all 14 codecs.

Every kq call is pinned to ``stream=mx.cpu``, so this exercises the scalar CPU
decoders (dequantize / quantized_matmul / gather_qmm) and runs with no GPU - the
CI value of the CPU decode path. Checks, per codec, against the gguf.quants
numpy reference:

  * dequantize -> float32 : BIT-EXACT
  * dequantize -> float16 : loose (downcast)
  * quantized_matmul(transpose=True/False) ~= x @ dequant(w)[.T]
  * gather_qmm (MoE) ~= per-row x[lhs] @ dequant(w[expert]).T

When a Metal GPU is present, also A/Bs CPU vs GPU dequantize for bit-exactness.

K-quant wire bytes come from the committed fixtures (gguf-py is decode-only for
those); flat codecs are quantized in-process; IQ codecs (decode-only, no gguf-py
encode) use synthesized structurally-valid wire. Run directly for a sweep table.
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

CPU = mx.cpu

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
}

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
N, K = 256, 512  # K % 256 and % 64 == 0
E_MOE = 4  # experts for the gather sweep

gpu = pytest.mark.skipif(
    not kq.metallib_loads(), reason="no Metal GPU for the CPU-vs-GPU A/B"
)


def _synth_iq_wire(rng, bpb, n_blocks):
    """Structurally-valid random IQ wire: random bytes with a sane fp16 d
    (block offset 0 in every IQ4/IQ3 struct) so dequant can't hit Inf/NaN."""
    wire = rng.integers(0, 256, size=(n_blocks, bpb), dtype=np.uint8)
    d = rng.uniform(0.02, 0.08, n_blocks).astype(np.float16)
    wire[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 2)
    return wire


def _dense_wire_and_ref(codec, gtype, is_kquant):
    """(wire uint8[N, packed], ref float32[N, K]) or (None, None) if missing."""
    if codec.startswith("iq"):
        wpb, bpb = CODECS[codec][1], CODECS[codec][2]
        wire = _synth_iq_wire(np.random.default_rng(7), bpb, N * (K // wpb))
        wire = wire.reshape(N, (K // wpb) * bpb)
        ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
        return wire, ref
    if is_kquant:
        path = os.path.join(FIX, f"{codec}.npz")
        if not os.path.exists(path):
            return None, None
        wire = np.load(path)["wire"].astype(np.uint8)
    else:
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
        wire = quants.quantize(w, gtype).astype(np.uint8)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
    return wire, ref


def _moe_wire_and_ref(codec, gtype, is_kquant):
    """(wire uint8[E, N, packed], ref float32[E, N, K]) or (None, None)."""
    if codec.startswith("iq"):
        wpb, bpb = CODECS[codec][1], CODECS[codec][2]
        rng = np.random.default_rng(11)
        wires = [
            _synth_iq_wire(rng, bpb, N * (K // wpb)).reshape(N, (K // wpb) * bpb)
            for _ in range(E_MOE)
        ]
        refs = [quants.dequantize(np.ascontiguousarray(w), gtype) for w in wires]
        return np.stack(wires, 0), np.stack(refs, 0).astype(np.float32)
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
    for _ in range(E_MOE):
        we = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
        wq = quants.quantize(we, gtype).astype(np.uint8)
        wires.append(wq)
        refs.append(quants.dequantize(np.ascontiguousarray(wq), gtype))
    return np.stack(wires, 0), np.stack(refs, 0).astype(np.float32)


def _scales():
    return mx.array(np.zeros((1,), np.uint8))


def test_cpu_dequantize_matches_reference():
    for codec, (gtype, _wpb, _bpb, _bits, is_kq) in CODECS.items():
        wire, ref = _dense_wire_and_ref(codec, gtype, is_kq)
        assert wire is not None, f"{codec}: missing fixture - run gen_fixtures.py"
        w = mx.array(wire)
        o32 = kq.dequantize(w, _scales(), codec, dtype=mx.float32, stream=CPU)
        o16 = kq.dequantize(w, _scales(), codec, dtype=mx.float16, stream=CPU)
        mx.eval(o32, o16)
        a32 = np.array(o32).astype(np.float32).reshape(ref.shape)
        a16 = np.array(o16).astype(np.float32).reshape(ref.shape)
        assert np.array_equal(ref, a32), f"{codec}: CPU f32 dequant not bit-exact"
        assert np.allclose(ref, a16, atol=1e-3, rtol=1e-3), f"{codec}: f16 off"


def test_cpu_quantized_matmul_matches_reference():
    rng = np.random.default_rng(0)
    for codec, (gtype, _wpb, _bpb, _bits, is_kq) in CODECS.items():
        wire, ref = _dense_wire_and_ref(codec, gtype, is_kq)
        assert wire is not None, f"{codec}: missing fixture - run gen_fixtures.py"
        rows, kk = ref.shape  # [N, K]
        w = mx.array(wire)
        for M in (1, 64):
            xt = (rng.standard_normal((M, kk)) * 0.1).astype(np.float16)
            got_t = kq.quantized_matmul(
                mx.array(xt), w, _scales(), codec, transpose=True, stream=CPU
            )
            ref_t = xt.astype(np.float32) @ ref.T  # [M, N]
            xn = (rng.standard_normal((M, rows)) * 0.1).astype(np.float16)
            got_n = kq.quantized_matmul(
                mx.array(xn), w, _scales(), codec, transpose=False, stream=CPU
            )
            ref_n = xn.astype(np.float32) @ ref  # [M, K]
            mx.eval(got_t, got_n)
            for got, rr, slot in ((got_t, ref_t, "t"), (got_n, ref_n, "n")):
                g = np.array(got).astype(np.float32)
                assert g.shape == rr.shape, f"{codec} M={M} {slot}: shape"
                rel = float(np.linalg.norm(g - rr) / (np.linalg.norm(rr) + 1e-6))
                assert rel < 2e-2, f"{codec} M={M} {slot}: rel={rel:.3e}"


def test_cpu_gather_qmm_matches_reference():
    rng = np.random.default_rng(0)
    for codec, (gtype, _wpb, _bpb, _bits, is_kq) in CODECS.items():
        wire, ref = _moe_wire_and_ref(codec, gtype, is_kq)
        assert wire is not None, f"{codec}: missing _moe fixture - gen_fixtures.py"
        ne, nn, kk = ref.shape
        w = mx.array(wire)
        for M in (1, 64):
            B = 8
            x_np = (rng.standard_normal((B, M, kk)) * 0.1).astype(np.float16)
            x = mx.array(x_np)
            experts = rng.integers(0, ne, size=B).astype(np.uint32)
            # contiguous and a reversed-with-repeat lhs selection (non-identity)
            seq = np.arange(B, dtype=np.uint32)
            perm = seq[::-1].copy()
            perm[1] = perm[0]
            for lhs_np in (seq, perm):
                got = kq.gather_qmm(
                    x,
                    w,
                    _scales(),
                    codec,
                    lhs_indices=mx.array(lhs_np),
                    rhs_indices=mx.array(experts),
                    transpose=True,
                    stream=CPU,
                )
                mx.eval(got)
                g = np.array(got).astype(np.float32)  # [B, M, N]
                rr = np.stack(
                    [
                        x_np[int(lhs_np[b])].astype(np.float32) @ ref[experts[b]].T
                        for b in range(B)
                    ],
                    axis=0,
                )
                assert g.shape == rr.shape, f"{codec} M={M}: gather shape"
                rel = float(np.linalg.norm(g - rr) / (np.linalg.norm(rr) + 1e-6))
                assert rel < 2e-2, f"{codec} M={M}: gather rel={rel:.3e}"


@gpu
def test_cpu_vs_gpu_dequantize_bit_exact():
    """f32 dequant is a pure decode - CPU and GPU must produce identical bytes."""
    for codec, (gtype, _wpb, _bpb, _bits, is_kq) in CODECS.items():
        if codec.startswith("iq"):
            continue  # IQ Metal dequant lands in Phase D
        wire, _ref = _dense_wire_and_ref(codec, gtype, is_kq)
        assert wire is not None, f"{codec}: missing fixture - run gen_fixtures.py"
        w = mx.array(wire)
        cpu = kq.dequantize(w, _scales(), codec, dtype=mx.float32, stream=CPU)
        gpu_out = kq.dequantize(w, _scales(), codec, dtype=mx.float32, stream=mx.gpu)
        mx.eval(cpu, gpu_out)
        assert np.array_equal(np.array(cpu), np.array(gpu_out)), (
            f"{codec}: CPU vs GPU f32 dequant differ"
        )


def _main() -> int:
    print("=== test_cpu_decode (stream=cpu) ===")
    fns = (
        ("dequantize", test_cpu_dequantize_matches_reference),
        ("quantized_matmul", test_cpu_quantized_matmul_matches_reference),
        ("gather_qmm", test_cpu_gather_qmm_matches_reference),
    )
    fails = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  {name:<18} ok")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<18} FAIL: {e}")
    if kq.metallib_loads():
        try:
            test_cpu_vs_gpu_dequantize_bit_exact()
            print(f"  {'cpu-vs-gpu deq':<18} ok")
        except AssertionError as e:
            fails += 1
            print(f"  {'cpu-vs-gpu deq':<18} FAIL: {e}")
    print("ALL OK" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
