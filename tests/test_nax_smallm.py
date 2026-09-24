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
from kqref import synth_wire  # noqa: E402
from test_codecs import CODECS  # noqa: E402

import mlx_kquant as kq  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="NAX/GPU-only routing",
)

K = 1024
# Every routing seam: mv tail (2, 6), verify_mma entries (3, 5) and
# their mat-vec neighbour (4), per-codec qmm crossover (7-10), NAX BM=32
# body/edges (12, 13, 16, 24, 31, 32), BM=64 handoff (33, 64). The _db
# double-buffered variant of the 33-64 band is N-gated far above these
# widths; test_db64_band_dispatch covers it at the policy floors.
MS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 16, 24, 31, 32, 33, 48, 64]

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
IQ = [c for c in CODECS if c.startswith("iq") or c in ("stq1_0", "pq2_0", "ptq1_0")]


def _sweep(codec, w, s, ref_w, n_out, ms=MS, dtype=mx.bfloat16, k=K):
    for m in ms:
        x = (mx.random.normal((m, k)) * 0.5).astype(dtype)
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
    wire = synth_wire(rng, codec, bpb, n_out * (K // wpb))
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


# Register-resident MMA verify band (kq_verify_mma.h): KQ_VERIFY_MMA=2
# forces the route at every M in 2..8 on any GPU, so the kernel is checked
# at every row count and both activation dtypes it is instantiated for,
# on an aligned and a ragged N (the row clamp past N). K=1000 blocks the
# route (the codecs need a whole number of wire blocks), so K stays 1024.
VERIFY_MMA_CODECS = ["pq2_0", "ptq1_0", "q4_0"]
VERIFY_MMA_MS = [2, 3, 4, 5, 6, 7, 8]


@pytest.fixture
def verify_mma_forced(monkeypatch):
    monkeypatch.setenv("KQ_VERIFY_MMA", "2")


def _verify_mma_setup(codec, n_out, k=K):
    if codec in ENCODABLE:
        mx.random.seed(11)
        wf = mx.random.normal((n_out, k)) * 0.1
        w, s = kq.quantize(wf, codec)
        ref_w = kq.dequantize(w, s, codec).astype(mx.float32).T
        mx.eval(w, s, ref_w)
        return w, s, ref_w
    from kqref import quants

    gtype, wpb, bpb, _, _ = CODECS[codec]
    rng = np.random.default_rng(7)
    wire = synth_wire(rng, codec, bpb, n_out * (k // wpb))
    wire = wire.reshape(n_out, (k // wpb) * bpb)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype)
    ref_w = mx.array(ref.astype(np.float32)).T
    w = mx.array(wire)
    s = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(w, s, ref_w)
    mx.random.seed(11)
    return w, s, ref_w


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", VERIFY_MMA_CODECS)
def test_verify_mma_band(codec, n_out, dtype, verify_mma_forced):
    w, s, ref_w = _verify_mma_setup(codec, n_out)
    _sweep(codec, w, s, ref_w, n_out, ms=VERIFY_MMA_MS, dtype=dtype)


@pytest.mark.parametrize("codec", VERIFY_MMA_CODECS)
def test_verify_mma_ragged_chunk(codec, verify_mma_forced):
    # 2176 = 17 blocks of 128: one split of 4 full 512-wide chunks plus a
    # 128-wide tail for the Prism codecs, and 4 splits of 512 + 32 for
    # q4_0, so the partial-chunk path runs on both block widths.
    k = 2176
    w, s, ref_w = _verify_mma_setup(codec, 1000, k=k)
    _sweep(codec, w, s, ref_w, 1000, ms=[2, 8], k=k)


@pytest.mark.parametrize("codec", VERIFY_MMA_CODECS)
def test_verify_mma_default_matches_forced(codec, monkeypatch):
    # The default route at M8 and the forced route agree to the bf16
    # output rounding, whichever kernel the default picks.
    w, s, ref_w = _verify_mma_setup(codec, 1024)
    x = (mx.random.normal((8, K)) * 0.5).astype(mx.bfloat16)
    monkeypatch.setenv("KQ_VERIFY_MMA", "2")
    y_forced = kq.quantized_matmul(x, w, s, codec, transpose=True)
    monkeypatch.setenv("KQ_VERIFY_MMA", "0")
    y_off = kq.quantized_matmul(x, w, s, codec, transpose=True)
    mx.eval(y_forced, y_off)
    err = float(
        mx.abs(y_forced.astype(mx.float32) - y_off.astype(mx.float32)).max()
        / (mx.abs(y_off.astype(mx.float32)).max() + 1e-6)
    )
    assert err < 2e-2, f"{codec}: forced vs off rel err {err:.3e}"


# KQ_QMM_ROUTE forces one small-M route per call (the probe behind
# benchmarks/bench_verify_routes.py). Every route either serves the call
# or declines it to the default routing, so each must hold the numeric
# contract at every width it can see, including the widths it declines.
QMM_ROUTES = [
    "qmv",
    "verify_qmv",
    "mv_ext",
    "verify_mma",
    "splitk",
    "nax",
    "nax_splitk",
]


@pytest.mark.parametrize("route", QMM_ROUTES)
@pytest.mark.parametrize("codec", ["pq2_0", "ptq1_0", "q4_0", "q4_k"])
def test_qmm_route_probe(codec, route, monkeypatch):
    if codec in ENCODABLE:
        w, s, ref_w = _encodable_setup(codec, 1000)
    else:
        w, s, ref_w = _iq_setup(codec, 1000)
    monkeypatch.setenv("KQ_QMM_ROUTE", route)
    _sweep(codec, w, s, ref_w, 1000, ms=[1, 2, 5, 8, 12, 16, 33])


VMMA_CODECS = ["pq2_0", "ptq1_0", "q4_0"]


def _setup(codec, n_out):
    if codec in ENCODABLE:
        return _encodable_setup(codec, n_out)
    return _iq_setup(codec, n_out)


def _route_serves(route, codec, m):
    """The KQ_QMM_ROUTE guards in kq_forced_route for N 1000, K 1024."""
    if m > 32:
        return False
    if route == "qmv":
        return True
    if route == "verify_qmv":
        return 2 <= m <= 8
    if route == "mv_ext":
        return 2 <= m <= 12
    if route == "verify_mma":
        return codec in VMMA_CODECS and m <= 8
    if route == "splitk":
        return True
    return kq.nax_available()


# Under KQ_QMM_ROUTE_STRICT=1 a declined route raises instead of running
# the default routing, so a served width proves the route itself ran.
@pytest.mark.parametrize("route", QMM_ROUTES)
@pytest.mark.parametrize("codec", ["pq2_0", "ptq1_0", "q4_0", "q4_k"])
def test_qmm_route_probe_strict(codec, route, monkeypatch):
    w, s, ref_w = _setup(codec, 1000)
    monkeypatch.setenv("KQ_QMM_ROUTE", route)
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    for m in [1, 2, 5, 8, 9, 12, 16, 33]:
        if _route_serves(route, codec, m):
            _sweep(codec, w, s, ref_w, 1000, ms=[m])
        else:
            x = mx.zeros((m, K), dtype=mx.bfloat16)
            with pytest.raises(RuntimeError, match="does not serve"):
                mx.eval(kq.quantized_matmul(x, w, s, codec, transpose=True))
    # A declined call leaves the stream usable for the next one.
    monkeypatch.delenv("KQ_QMM_ROUTE")
    _sweep(codec, w, s, ref_w, 1000, ms=[8])


# A NAX matmul whose rows fit one SG-row (16 rows on the BM=32 tile, 32 on
# the BM=64 tile) splits its K walk across both SG-rows and sums the halves
# through threadgroup memory. Forced through the split-K tile and the
# un-split tile at every width to 32, both activation dtypes, aligned and
# ragged N. The codecs without a BM=32 policy run the un-split route on the
# BM=64 tile.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX tile only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("route", ["nax_splitk", "nax"])
@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", ENCODABLE + IQ)
def test_nax_short_tile_ksplit(codec, n_out, route, dtype, monkeypatch):
    w, s, ref_w = _setup(codec, n_out)
    monkeypatch.setenv("KQ_QMM_ROUTE", route)
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    _sweep(codec, w, s, ref_w, n_out, ms=range(1, 33), dtype=dtype)


# An odd count of BK=64 steps (K 960 is 15) still gives each SG-row one
# substep per step. Only the 32-block codecs reach such a K.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX tile only")
@pytest.mark.parametrize("route", ["nax_splitk", "nax"])
@pytest.mark.parametrize("codec", ["q8_0", "q4_0", "q5_1"])
def test_nax_short_tile_ksplit_odd_k_steps(codec, route, monkeypatch):
    k = 960
    w, s, ref_w = _verify_mma_setup(codec, 1000, k=k)
    monkeypatch.setenv("KQ_QMM_ROUTE", route)
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    _sweep(codec, w, s, ref_w, 1000, ms=[1, 5, 8, 16], k=k)


# The verify_mma entries on float16 activations (the bfloat16 sweeps above
# cover the same widths).
@pytest.mark.parametrize("codec", VMMA_CODECS)
def test_verify_mma_entries_f16(codec):
    w, s, ref_w = _setup(codec, 1000)
    _sweep(codec, w, s, ref_w, 1000, ms=[2, 3, 4, 5, 8, 9], dtype=mx.float16)


# verify_mma sums each block in a half accumulator. One activation channel
# near the top of the half range must not overflow it; q4_0 multiplies it
# by up to 8 before the kernel's 1/16 prescale.
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("codec", VMMA_CODECS)
def test_verify_mma_outlier_channel(codec, dtype, monkeypatch):
    w, s, ref_w = _setup(codec, 1000)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_mma")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    for m in (2, 5, 8):
        x = mx.random.normal((m, K)) * 0.5
        x[:, 7] = 3.0e4
        x = x.astype(dtype)
        y = kq.quantized_matmul(x, w, s, codec, transpose=True).astype(mx.float32)
        ref = x.astype(mx.float32) @ ref_w
        mx.eval(y, ref)
        assert bool(mx.isfinite(y).all()), f"{codec} M{m}: non-finite output"
        err = float((mx.abs(y - ref)).max() / (mx.abs(ref).max() + 1e-6))
        assert err < 2e-2, f"{codec} M{m}: rel err {err:.3e}"
