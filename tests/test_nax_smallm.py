#!/usr/bin/env python3
"""Small-M NAX qmm routing validation (all NAX codecs).

The batch-decode M range routes every NAX codec through three regimes: the
mv paths (up to a per-codec crossover at M 6-9), the double-buffered BM=32
NAX tile (crossover through 32), and the classic BM=64 NAX tile (M >= 33).
This sweeps M across every seam and bounds each result against a
dequantize-based float32 reference, on both an aligned and a ragged N. The
BM=128 band tests extend the sweep so every codec dispatches the BM=128
tile on at least one tier cell. On non-NAX GPUs the small-M route falls
back to the mv paths and BM stays 64; the numeric contract is identical,
so the assertions hold on any Metal device.

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
# The _db double-buffered variant of the 33-64 band is N-gated far above
# these widths; test_db64_band_dispatch covers it at the policy floors.
MS = [2, 6, 7, 8, 9, 10, 12, 13, 16, 24, 31, 32, 33, 48, 64]

# BM=128 band: every entry has even ceil(M/64). M224/256 dispatch the
# tile for the 193 tier only (q6_k plus the IQ grid codecs); M512 adds
# the 449 tier and M1024 the 961 tier, so every codec takes BM=128 in at
# least one cell and stays on BM=64 in at least one other. The numeric
# contract is identical on both routings, and the padded M224 entry
# exercises the 32-dead-row tile edge.
BM128_MS = [224, 256, 512, 1024]

# Per-codec db64_min_n floors (kq_smallbm_policy). At these N the M33-64
# band dispatches the name-suffixed _db kernels on the default route; the
# small-N matrix below never reaches them, and env forcing cannot stand
# in because the mode reads are process-static.
DB64_N = {"q6_k": 16384, "q8_0": 8192, "q4_1": 8192, "q5_1": 8192, "q5_0": 8192}

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
IQ = [c for c in CODECS if c.startswith("iq") or c == "stq1_0"]


def _sweep(codec, w, s, ref_w, n_out, ms=MS):
    for m in ms:
        x = (mx.random.normal((m, K)) * 0.5).astype(mx.bfloat16)
        y = kq.quantized_matmul(x, w, s, codec, transpose=True)
        y = y.astype(mx.float32)
        ref = x.astype(mx.float32) @ ref_w
        mx.eval(y, ref)
        err = float((mx.abs(y - ref)).max() / (mx.abs(ref).max() + 1e-6))
        assert err < 2e-2, f"{codec} N{n_out} M{m}: rel err {err:.3e}"


def _encodable_setup(codec, n_out):
    mx.random.seed(11)
    wf = mx.random.normal((n_out, K)) * 0.1
    w, s = kq.quantize(wf, codec)
    ref_w = kq.dequantize(w, s, codec).astype(mx.float32).T
    mx.eval(w, s, ref_w)
    return w, s, ref_w


def _iq_setup(codec, n_out):
    from kqref import quants

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
    return w, s, ref_w


@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", ENCODABLE)
def test_smallm_routing(codec, n_out):
    w, s, ref_w = _encodable_setup(codec, n_out)
    _sweep(codec, w, s, ref_w, n_out)


@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", ENCODABLE)
def test_bm128_band(codec, n_out):
    w, s, ref_w = _encodable_setup(codec, n_out)
    _sweep(codec, w, s, ref_w, n_out, ms=BM128_MS)


@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", IQ)
def test_bm128_band_iq(codec, n_out):
    w, s, ref_w = _iq_setup(codec, n_out)
    _sweep(codec, w, s, ref_w, n_out, ms=BM128_MS)


@pytest.mark.parametrize("codec", sorted(DB64_N))
def test_db64_band_dispatch(codec):
    n_out = DB64_N[codec]
    w, s, ref_w = _encodable_setup(codec, n_out)
    _sweep(codec, w, s, ref_w, n_out, ms=[33, 48, 64])


@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", IQ)
def test_smallm_routing_iq(codec, n_out):
    w, s, ref_w = _iq_setup(codec, n_out)
    _sweep(codec, w, s, ref_w, n_out)


# Non-NAX split-K band, which the rest of this file cannot reach on NAX
# silicon: M 2 sits below every codec entry, 6-8 cover the bm8 tile for
# the codecs whose entry is at or under them, 10-16 the bm16 tile, 17-32
# the BM32 tile up to its ceiling, and 33 is the handoff back to plain
# qmm. KQ_DISABLE_NAX is read live, so toggling it re-routes in-process.
ALU_SPLITK_MS = [2, 6, 7, 8, 10, 12, 16, 17, 24, 32, 33]


@pytest.fixture
def nax_off(monkeypatch):
    monkeypatch.setenv("KQ_DISABLE_NAX", "1")


@pytest.mark.parametrize("codec", ENCODABLE)
def test_alu_splitk_band(codec, nax_off):
    w, s, ref_w = _encodable_setup(codec, 1000)
    _sweep(codec, w, s, ref_w, 1000, ms=ALU_SPLITK_MS)


@pytest.mark.parametrize("codec", IQ)
def test_alu_splitk_band_iq(codec, nax_off):
    w, s, ref_w = _iq_setup(codec, 1000)
    _sweep(codec, w, s, ref_w, 1000, ms=ALU_SPLITK_MS)
