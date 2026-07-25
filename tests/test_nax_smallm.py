#!/usr/bin/env python3
"""Small-M NAX qmm routing validation (all NAX codecs).

The batch-decode M range routes every NAX codec through three regimes: the
mv paths (up to a per-codec crossover at M 6-9), the double-buffered BM=32
NAX tile (crossover through 32), and the classic BM=64 NAX tile (M >= 33).
This sweeps M across every seam and bounds each result against a
dequantize-based float32 reference, on both an aligned and a ragged N. On
non-NAX GPUs the small-M route falls back to the mv paths and BM stays 64;
the numeric contract is identical, so the assertions hold on any Metal
device.

Encodable codecs quantize a fresh tensor; IQ codecs use the synthetic-wire
helpers from test_codecs (gguf-py dequantize as reference), since the
encoder requires an imatrix and grid search.

Run locally on GPU (per-phase NAX gate, not hosted CI).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx  # noqa: E402
from test_codecs import CODECS, _synth_iq_wire  # noqa: E402

import mlx_kquant as kq  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="NAX/GPU-only routing",
)

K = 1024
# Every routing seam: mv tail (2, 6), per-codec qmm crossover (7-10),
# NAX BM=32 body/edges (12, 13, 16, 24, 31, 32), BM=64 handoff (33, 64).
# 33/48/64 also cover the _db double-buffered BM=64 band variant where
# the policy enables it (q6_k, q8_0).
MS = [2, 6, 7, 8, 9, 10, 12, 13, 16, 24, 31, 32, 33, 48, 64]

ENCODABLE = [
    "q6_k",
    "q8_0",
    "q4_k",
    "q5_k",
    "q3_k",
    "q2_k",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
]
IQ = [c for c in CODECS if c.startswith("iq")]


def _sweep(codec, w, s, ref_w, n_out):
    for m in MS:
        x = (mx.random.normal((m, K)) * 0.5).astype(mx.bfloat16)
        y = kq.quantized_matmul(x, w, s, codec, transpose=True)
        y = y.astype(mx.float32)
        ref = x.astype(mx.float32) @ ref_w
        mx.eval(y, ref)
        err = float((mx.abs(y - ref)).max() / (mx.abs(ref).max() + 1e-6))
        assert err < 2e-2, f"{codec} N{n_out} M{m}: rel err {err:.3e}"


@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", ENCODABLE)
def test_smallm_routing(codec, n_out):
    mx.random.seed(11)
    wf = mx.random.normal((n_out, K)) * 0.1
    w, s = kq.quantize(wf, codec)
    ref_w = kq.dequantize(w, s, codec).astype(mx.float32).T
    mx.eval(w, s, ref_w)
    _sweep(codec, w, s, ref_w, n_out)


@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", IQ)
def test_smallm_routing_iq(codec, n_out):
    from gguf import quants

    gtype, wpb, bpb, _, _ = CODECS[codec]
    rng = np.random.default_rng(7)
    wire = _synth_iq_wire(rng, bpb, n_out * (K // wpb))
    wire = wire.reshape(n_out, (K // wpb) * bpb)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype)
    ref_w = mx.array(ref.astype(np.float32)).T
    w = mx.array(wire)
    s = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(w, s, ref_w)
    mx.random.seed(11)
    _sweep(codec, w, s, ref_w, n_out)
