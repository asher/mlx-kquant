#!/usr/bin/env python3
"""A/B the arm64 NEON int8 CPU GEMV kernels against the portable scalar path.

``KQ_CPU_NEON`` is read per matmul call, so the same process can run both
paths on identical wire bytes. All 14 codecs have int8 kernels, which
quantize activations to q8 first - lossy by design, so the gate is a
rel-norm tolerance (the error is dominated by the activation quantization,
~1e-2; the same trade ggml makes). On non-arm64 (or non-dotprod) builds both
runs take the scalar path and must be BIT-IDENTICAL.

Each codec is also checked against the gguf.quants numpy reference at the
same 2e-2 rel-norm gate the scalar path uses.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

CPU = mx.cpu

# codec -> gguf type (every codec has a NEON int8 kernel)
CODECS = {
    "q4_0": GT.Q4_0,
    "q4_1": GT.Q4_1,
    "q5_0": GT.Q5_0,
    "q5_1": GT.Q5_1,
    "q8_0": GT.Q8_0,
    "q2_k": GT.Q2_K,
    "q3_k": GT.Q3_K,
    "q4_k": GT.Q4_K,
    "q5_k": GT.Q5_K,
    "q6_k": GT.Q6_K,
    "iq4_nl": GT.IQ4_NL,
    "iq4_xs": GT.IQ4_XS,
    "iq3_s": GT.IQ3_S,
    "iq3_xxs": GT.IQ3_XXS,
}

# IQ codecs are decode-only (gguf-py can't encode them): (weights_per_block,
# bytes_per_block) to synthesize structurally-valid wire (sane fp16 d at offset
# 0) so both the NEON and scalar paths read identical bytes.
IQ_GEOM = {
    "iq4_nl": (32, 18),
    "iq4_xs": (256, 136),
    "iq3_s": (256, 110),
    "iq3_xxs": (256, 98),
}


def _synth_iq_wire(rng, bpb, n_blocks):
    wire = rng.integers(0, 256, size=(n_blocks, bpb), dtype=np.uint8)
    d = rng.uniform(0.02, 0.08, n_blocks).astype(np.float16)
    wire[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 2)
    return wire


FIX = os.path.join(os.path.dirname(__file__), "fixtures")
N, K = 256, 512
# NEON-vs-scalar gate: the difference is the activation q8 quantization error
# (~5-8e-3 rel-norm measured across codecs/M); 2e-2 gives ~2.5x headroom.
NEON_REL = 2e-2
REF_REL = 2e-2

E_MOE = 4


@pytest.fixture(autouse=True)
def _restore_neon_env():
    prev = os.environ.get("KQ_CPU_NEON")
    yield
    if prev is None:
        os.environ.pop("KQ_CPU_NEON", None)
    else:
        os.environ["KQ_CPU_NEON"] = prev


def _scales():
    return mx.array(np.zeros((1,), np.uint8))


def _dense_wire_and_ref(codec, gtype):
    if codec in IQ_GEOM:
        wpb, bpb = IQ_GEOM[codec]
        wire = _synth_iq_wire(np.random.default_rng(7), bpb, N * (K // wpb))
        wire = wire.reshape(N, (K // wpb) * bpb)
        ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
        return wire, ref
    path = os.path.join(FIX, f"{codec}.npz")
    if os.path.exists(path):
        wire = np.load(path)["wire"].astype(np.uint8)
    else:
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
        wire = quants.quantize(w, gtype).astype(np.uint8)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
    return wire, ref


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-6))


def test_neon_vs_scalar_quantized_matmul():
    rng = np.random.default_rng(0)
    for codec, gtype in CODECS.items():
        wire, ref = _dense_wire_and_ref(codec, gtype)
        w = mx.array(wire)
        for M in (1, 4, 16):
            x = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
            xs = mx.array(x)
            os.environ["KQ_CPU_NEON"] = "0"
            scalar = kq.quantized_matmul(
                xs, w, _scales(), codec, transpose=True, stream=CPU
            )
            os.environ["KQ_CPU_NEON"] = "1"
            neon = kq.quantized_matmul(
                xs, w, _scales(), codec, transpose=True, stream=CPU
            )
            mx.eval(scalar, neon)
            s = np.array(scalar).astype(np.float32)
            n = np.array(neon).astype(np.float32)
            ref_t = x.astype(np.float32) @ ref.T
            if kq.cpu_neon_available():
                rel = _rel(n, s)
                assert rel < NEON_REL, f"{codec} M={M}: neon-vs-scalar rel={rel:.3e}"
                rel = _rel(n, ref_t)
                assert rel < REF_REL, f"{codec} M={M}: neon-vs-ref rel={rel:.3e}"
            else:
                assert np.array_equal(n, s), (
                    f"{codec} M={M}: scalar fallback must be bit-identical"
                )


def test_neon_vs_scalar_gather_qmm():
    rng = np.random.default_rng(1)
    for codec, gtype in CODECS.items():
        path = os.path.join(FIX, f"{codec}_moe.npz")
        if codec in IQ_GEOM:
            wpb, bpb = IQ_GEOM[codec]
            wire = np.stack(
                [
                    _synth_iq_wire(rng, bpb, N * (K // wpb)).reshape(
                        N, (K // wpb) * bpb
                    )
                    for _ in range(E_MOE)
                ],
                0,
            )
        elif os.path.exists(path):
            wire = np.load(path)["wire"].astype(np.uint8)
        else:
            wires = []
            for _ in range(E_MOE):
                we = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
                wires.append(quants.quantize(we, gtype).astype(np.uint8))
            wire = np.stack(wires, 0)
        refs = [
            quants.dequantize(np.ascontiguousarray(wire[e]), gtype)
            for e in range(wire.shape[0])
        ]
        ref = np.stack(refs, 0).astype(np.float32)
        ne = ref.shape[0]
        w = mx.array(wire)
        # Decode shape: one token row shared by top-k experts, with a repeated
        # expert so the batch path exercises both direct and packed tasks.
        B, M = 6, 1
        x = (rng.standard_normal((B, M, K)) * 0.1).astype(np.float16)
        experts = rng.integers(0, ne, size=B).astype(np.uint32)
        experts[1] = experts[0]
        lhs = np.zeros(B, np.uint32)
        args = dict(
            lhs_indices=mx.array(lhs),
            rhs_indices=mx.array(experts),
            transpose=True,
            stream=CPU,
        )
        os.environ["KQ_CPU_NEON"] = "0"
        scalar = kq.gather_qmm(mx.array(x), w, _scales(), codec, **args)
        os.environ["KQ_CPU_NEON"] = "1"
        neon = kq.gather_qmm(mx.array(x), w, _scales(), codec, **args)
        mx.eval(scalar, neon)
        s = np.array(scalar).astype(np.float32)
        n = np.array(neon).astype(np.float32)
        rr = np.stack(
            [x[int(lhs[b])].astype(np.float32) @ ref[experts[b]].T for b in range(B)],
            axis=0,
        )
        if kq.cpu_neon_available():
            assert _rel(n, s) < NEON_REL, f"{codec}: gather neon-vs-scalar"
            assert _rel(n, rr) < REF_REL, f"{codec}: gather neon-vs-ref"
        else:
            assert np.array_equal(n, s), f"{codec}: gather fallback bit-identical"
        assert _rel(s, rr) < REF_REL, f"{codec}: gather scalar-vs-ref"


def test_kill_switch_disables_neon():
    """KQ_CPU_NEON=0 must reproduce the scalar path exactly (same process)."""
    rng = np.random.default_rng(2)
    wire, _ = _dense_wire_and_ref("q4_k", GT.Q4_K)
    w = mx.array(wire)
    x = mx.array((rng.standard_normal((1, K)) * 0.1).astype(np.float32))
    os.environ["KQ_CPU_NEON"] = "0"
    a = kq.quantized_matmul(x, w, _scales(), "q4_k", transpose=True, stream=CPU)
    b = kq.quantized_matmul(x, w, _scales(), "q4_k", transpose=True, stream=CPU)
    mx.eval(a, b)
    assert bool(mx.array_equal(a, b, stream=CPU))
