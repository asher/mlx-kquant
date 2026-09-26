#!/usr/bin/env python3
"""Small-M NAX qmm routing validation (all NAX codecs).

The batch-decode M range routes every NAX codec through four regimes: per-row
qmv (M 2 up to a per-codec limit that falls with N), the mv paths, the
double-buffered BM=32 NAX tile with split-K (from a per-codec entry through
32), and the classic BM=64 NAX tile (M >= 33). This sweeps M across every
seam and bounds each result against a dequantize-based float32 reference,
on both an aligned and a ragged N, with the qmv route on and off. The
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

import json
import os
import subprocess
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
# Every routing seam: mv tail (2, 6), verify_mma entries (3, 5), verify_nax
# entries (3 to 6), per-codec qmm crossover (7-10),
# NAX BM=32 body/edges (12, 13, 16, 24, 31, 32), BM=64 handoff (33, 64). The _db
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


@pytest.fixture(params=["", "0"], ids=["qmv", "noqmv"])
def nax_qmv(request, monkeypatch):
    """Default routing, and with the NAX per-row qmv route off so the
    mat-vec and split-K bands stay covered at the N the qmv route claims."""
    if request.param:
        monkeypatch.setenv("KQ_NAX_QMV", request.param)
    else:
        monkeypatch.delenv("KQ_NAX_QMV", raising=False)


@pytest.mark.parametrize("n_out", [1024, 1000])
@pytest.mark.parametrize("codec", ENCODABLE)
def test_smallm_routing(codec, n_out, nax_qmv):
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
def test_smallm_routing_iq(codec, n_out, nax_qmv):
    w, s, ref_w = _iq_setup(codec, n_out)
    _sweep(codec, w, s, ref_w, n_out)


# N 500 sits in the narrowest qmv bucket (N <= 512), where most codecs
# take per-row qmv through M 7-8 before split-K.
@pytest.mark.parametrize("codec", ENCODABLE + IQ)
def test_smallm_routing_small_n(codec):
    w, s, ref_w = _setup(codec, 500)
    _sweep(codec, w, s, ref_w, 500, ms=range(2, 11))


# The mat-vec kernel at a ragged N on every codec, forced. On NAX GPUs the
# default routing reaches it at N 1000 only between the qmv limit and the
# split-K entry, which is empty for most codecs.
@pytest.mark.parametrize("codec", ENCODABLE + IQ)
def test_mv_ext_ragged_n(codec, monkeypatch):
    w, s, ref_w = _setup(codec, 1000)
    monkeypatch.setenv("KQ_QMM_ROUTE", "mv_ext")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    _sweep(codec, w, s, ref_w, 1000, ms=range(2, 13))


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
VERIFY_MMA_CODECS = ["pq2_0", "ptq1_0", "q4_0", "q8_0"]
VERIFY_MMA_MS = [2, 3, 4, 5, 6, 7, 8]


@pytest.fixture
def verify_mma_forced(monkeypatch):
    monkeypatch.setenv("KQ_VERIFY_MMA", "2")
    monkeypatch.setenv("KQ_VERIFY_NAX", "0")


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
    # q4_0 and q8_0, so the partial-chunk path runs on both block widths.
    k = 2176
    w, s, ref_w = _verify_mma_setup(codec, 1000, k=k)
    _sweep(codec, w, s, ref_w, 1000, ms=[2, 8], k=k)


@pytest.mark.parametrize("codec", VERIFY_MMA_CODECS)
def test_verify_mma_default_matches_forced(codec, monkeypatch):
    # The default route at M8 and the forced route agree to the bf16
    # output rounding, whichever kernel the default picks. The route is
    # chosen when the graph evaluates, so each output evaluates under its
    # own environment.
    w, s, ref_w = _verify_mma_setup(codec, 1024)
    x = (mx.random.normal((8, K)) * 0.5).astype(mx.bfloat16)
    monkeypatch.setenv("KQ_VERIFY_NAX", "0")
    monkeypatch.setenv("KQ_VERIFY_MMA", "2")
    y_forced = kq.quantized_matmul(x, w, s, codec, transpose=True)
    mx.eval(y_forced)
    monkeypatch.setenv("KQ_VERIFY_MMA", "0")
    y_off = kq.quantized_matmul(x, w, s, codec, transpose=True)
    mx.eval(y_off)
    assert not bool(mx.array_equal(y_forced, y_off).item())
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
    "verify_nax",
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


VMMA_CODECS = ["pq2_0", "ptq1_0", "q4_0", "q8_0"]
# verify_nax entries on NAX GPUs below N 100000 (kq_verify_nax_min_m).
# q2_k enters one row earlier above N 4096.
VNAX_ENTRY = {
    "pq2_0": 3,
    "q4_0": 3,
    "q8_0": 6,
    "q4_k": 3,
    "q5_k": 3,
    "q6_k": 5,
    "q3_k": 4,
    "q2_k": 5,
}
VNAX_WIDE_N = {"q2_k": 4096}
VNAX_CODECS = list(VNAX_ENTRY)


def _vnax_entry(codec, n):
    e = VNAX_ENTRY.get(codec, 9)
    return e - 1 if n > VNAX_WIDE_N.get(codec, n) else e


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
    if route == "verify_nax":
        return kq.nax_available() and codec in VNAX_CODECS and m <= 8
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


# NAX split-K slices K in units of max(block, 64) weights. A prime count
# of 19 units has no divisor to split on, so the route cuts 9 slices of 2
# units and a last slice of 1. The default routing, with the route forced
# on through KQ_QMM_SPLITK_NAX, runs the same slices bit for bit. With
# KQ_SPLITK_RAGGED=0 the divisor count is 1 and the route declines.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX tile only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("codec", ENCODABLE + IQ)
def test_nax_splitk_ragged(codec, dtype, monkeypatch):
    monkeypatch.delenv("KQ_SPLITK_RAGGED", raising=False)
    k = 19 * max(CODECS[codec][1], 64)
    w, s, ref_w = _verify_mma_setup(codec, 1000, k=k)
    monkeypatch.setenv("KQ_QMM_ROUTE", "nax_splitk")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    _sweep(codec, w, s, ref_w, 1000, ms=[1, 5, 8, 16, 32], dtype=dtype, k=k)
    x = (mx.random.normal((16, k)) * 0.5).astype(dtype)
    y_forced = kq.quantized_matmul(x, w, s, codec, transpose=True)
    monkeypatch.delenv("KQ_QMM_ROUTE")
    monkeypatch.setenv("KQ_QMM_SPLITK_NAX", "1")
    y_default = kq.quantized_matmul(x, w, s, codec, transpose=True)
    assert _bits_equal(y_forced, y_default)
    monkeypatch.delenv("KQ_QMM_SPLITK_NAX")
    monkeypatch.setenv("KQ_QMM_ROUTE", "nax_splitk")
    monkeypatch.setenv("KQ_SPLITK_RAGGED", "0")
    x = mx.zeros((8, k), dtype=dtype)
    with pytest.raises(RuntimeError, match="does not serve"):
        mx.eval(kq.quantized_matmul(x, w, s, codec, transpose=True))


# Ragged slices replace the divisor count only when they more than double
# it. 68 units divide into 4 and run as 14 ragged slices. 35 units divide
# into 7, and 12 ragged slices would not double that.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX tile only")
@pytest.mark.parametrize("units, ragged", [(68, True), (35, False)])
def test_nax_splitk_ragged_rule(units, ragged, monkeypatch):
    monkeypatch.delenv("KQ_SPLITK_RAGGED", raising=False)
    k = 256 * units
    w, s, ref_w = _verify_mma_setup("q4_k", 256, k=k)
    monkeypatch.setenv("KQ_QMM_ROUTE", "nax_splitk")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    x = (mx.random.normal((8, k)) * 0.5).astype(mx.bfloat16)
    ys = {}
    for v in ("0", "1"):
        monkeypatch.setenv("KQ_SPLITK_RAGGED", v)
        ys[v] = kq.quantized_matmul(x, w, s, "q4_k", transpose=True)
        mx.eval(ys[v])
    assert _bits_equal(ys["0"], ys["1"]) != ragged
    _sweep("q4_k", w, s, ref_w, 256, ms=[8], k=k)


# The verify_mma entries on float16 activations (the bfloat16 sweeps above
# cover the same widths). Per-row qmv claims the entries at N 1000 by
# default, so it is off here.
@pytest.mark.parametrize("codec", VMMA_CODECS)
def test_verify_mma_entries_f16(codec, monkeypatch):
    monkeypatch.setenv("KQ_NAX_QMV", "0")
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


# The register-fed NAX verify route (verify_nax) serves pq2_0, q4_0, q8_0
# and q2_k to q6_k through M 8 on NAX GPUs. Forced at every width it
# serves, both activation dtypes, aligned and ragged N (the row clamp in
# the last simdgroup's 32 rows). q4_0 runs its eight-block kernel at K 1024, and
# the "0" arm (KQ_VERIFY_NAX_Q4_0_SB=0) its two-block kernel.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("n_out", [1024, 1000, 40, 20])
@pytest.mark.parametrize(
    "codec, sb", [(c, None) for c in VNAX_CODECS] + [("q4_0", "0")]
)
def test_verify_nax_band(codec, sb, n_out, dtype, monkeypatch):
    w, s, ref_w = _setup(codec, n_out)
    if sb is not None:
        monkeypatch.setenv("KQ_VERIFY_NAX_Q4_0_SB", sb)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    _sweep(codec, w, s, ref_w, n_out, ms=range(1, 9), dtype=dtype)


# Split counts across the K walk. K 2176 is 17 steps of 128 for pq2_0 and
# q8_0 (no split divides it), K 1920 is 15 (odd counts 3 and 5), K 4096 is
# 32. The superblock codecs take K 4352 (17 q4_k steps of 256, 34 steps
# of 128 for the others), K 3840 (15 and 30) and K 4096. q4_0 runs its two-block
# kernel at K 2176 and 1920 (34 and 30 steps of 64) and its eight-block
# kernel at K 4096, 3840 and 4352 (16, 15 and 17 steps of 256). The forced
# counts cover the partial fold wherever they divide the steps.
VNAX_SB = ("q4_k", "q5_k", "q6_k", "q3_k", "q2_k")
VNAX_SPLIT_K = {
    c: [4352, 3840, 4096] if c in VNAX_SB else [2176, 1920, 4096] for c in VNAX_CODECS
}
VNAX_SPLIT_K["q4_0"] += [3840, 4352]


@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("splits", ["", "2", "3", "4", "5", "8"])
@pytest.mark.parametrize(
    "codec, k", [(c, k) for c in VNAX_CODECS for k in VNAX_SPLIT_K[c]]
)
def test_verify_nax_splits(codec, k, splits, dtype, monkeypatch):
    w, s, ref_w = _verify_mma_setup(codec, 1000, k=k)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", splits)
    _sweep(codec, w, s, ref_w, 1000, ms=[1, 5, 8], dtype=dtype, k=k)


# A K that is not a whole number of the kernel's steps declines the route:
# a q4_0 K with an odd count of 32-wide blocks (64-wide steps), and a q8_0
# K that is not a multiple of 128, with K % 64 of 0 and 32. The default
# routing then runs per-row qmv at M 5 and verify_mma at M 8 for q4_0, and
# NAX split-K from M 6 for q8_0.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("codec, k", [("q4_0", 1056), ("q8_0", 1088), ("q8_0", 1056)])
def test_verify_nax_k_declines(codec, k, monkeypatch):
    w, s, ref_w = _verify_mma_setup(codec, 1000, k=k)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    x = mx.zeros((8, k), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match="does not serve"):
        mx.eval(kq.quantized_matmul(x, w, s, codec, transpose=True))
    monkeypatch.delenv("KQ_QMM_ROUTE")
    monkeypatch.delenv("KQ_QMM_ROUTE_STRICT")
    _sweep(codec, w, s, ref_w, 1000, ms=[5, 6, 7, 8], k=k)


# The q8_0, eight-block q4_0 and q2_k kernels read their weights as 4-byte
# words, so a q8_0 or q4_0 weight view that starts 2 mod 4 declines the
# route. A base 4 bytes in runs it. Both match the reference, and the
# default route at M 8 takes the displaced route on the misaligned base
# (NAX split-K for q8_0, verify_mma for q4_0). A q2_k base 2 mod 4 raises
# before any route runs (tests/test_weight_base.py).
VNAX_DISPLACED_M8 = {"q8_0": "nax_splitk", "q4_0": "verify_mma"}


@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("offset", [2, 4])
@pytest.mark.parametrize("codec", ["q8_0", "q4_0", "q2_k"])
def test_verify_nax_word_weight_alignment(codec, offset, monkeypatch):
    w, s, ref_w = _setup(codec, 1000)
    pad = mx.zeros((offset,), dtype=mx.uint8)
    wv = mx.concatenate([pad, w.reshape(-1)])[offset:].reshape(w.shape)
    mx.eval(wv)
    monkeypatch.setenv("KQ_NAX_QMV", "0")
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    x = (mx.random.normal((8, K)) * 0.5).astype(mx.bfloat16)
    if offset % 4 and codec == "q2_k":
        with pytest.raises(ValueError, match="4-byte boundary"):
            mx.eval(kq.quantized_matmul(x, wv, s, codec, transpose=True))
        return
    if offset % 4:
        with pytest.raises(RuntimeError, match="does not serve"):
            mx.eval(kq.quantized_matmul(x, wv, s, codec, transpose=True))
    else:
        y_nax = kq.quantized_matmul(x, wv, s, codec, transpose=True)
        y_ref = kq.quantized_matmul(x, w, s, codec, transpose=True)
        mx.eval(y_nax, y_ref)
        assert _bits_equal(y_nax, y_ref)
    monkeypatch.delenv("KQ_QMM_ROUTE")
    monkeypatch.delenv("KQ_QMM_ROUTE_STRICT")
    _sweep(codec, wv, s, ref_w, 1000, ms=[3, 6, 8])
    if offset % 4:
        y_def = _run_route(monkeypatch, x, wv, s, codec)
        y_alt = _run_route(monkeypatch, x, w, s, codec, VNAX_DISPLACED_M8[codec])
        assert _bits_equal(y_def, y_alt)


# KQ_VERIFY_NAX_Q4_0_SB=0 keeps q4_0 on the two-block kernel at a K that
# is a multiple of 256, where the eight-block kernel runs by default. At K
# 3840 a split target of 4 gives the eight-block kernel 3 splits of its 15
# steps and the two-block kernel 4 of its 60, so a lever that changed
# nothing would return the same bits. Both kernels match the reference.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_verify_nax_q4_0_two_block_lever(dtype, monkeypatch):
    k = 3840
    w, s, ref_w = _verify_mma_setup("q4_0", 1000, k=k)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", "4")
    x = (mx.random.normal((8, k)) * 0.5).astype(dtype)
    y_sb = kq.quantized_matmul(x, w, s, "q4_0", transpose=True)
    mx.eval(y_sb)
    _sweep("q4_0", w, s, ref_w, 1000, ms=[1, 3, 8], dtype=dtype, k=k)
    monkeypatch.setenv("KQ_VERIFY_NAX_Q4_0_SB", "0")
    y_two = kq.quantized_matmul(x, w, s, "q4_0", transpose=True)
    mx.eval(y_two)
    assert not _bits_equal(y_sb, y_two)
    _sweep("q4_0", w, s, ref_w, 1000, ms=[1, 3, 8], dtype=dtype, k=k)


# The q4_k and q5_k kernels read their weights as 16-byte words. A base
# 16 bytes in runs the route bit for bit like the aligned tensor, and a
# base 4 or 8 bytes in raises before any route runs
# (tests/test_weight_base.py).
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("offset", [4, 8, 16])
@pytest.mark.parametrize("codec", ["q4_k", "q5_k"])
def test_verify_nax_kquant_weight_alignment(codec, offset, monkeypatch):
    w, s, ref_w = _setup(codec, 1000)
    pad = mx.zeros((offset,), dtype=mx.uint8)
    wv = mx.concatenate([pad, w.reshape(-1)])[offset:].reshape(w.shape)
    mx.eval(wv)
    monkeypatch.setenv("KQ_NAX_QMV", "0")
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    x = (mx.random.normal((8, K)) * 0.5).astype(mx.bfloat16)
    if offset % 16:
        with pytest.raises(ValueError, match="16-byte boundary"):
            mx.eval(kq.quantized_matmul(x, wv, s, codec, transpose=True))
        return
    y_nax = kq.quantized_matmul(x, wv, s, codec, transpose=True)
    y_ref = kq.quantized_matmul(x, w, s, codec, transpose=True)
    mx.eval(y_nax, y_ref)
    assert _bits_equal(y_nax, y_ref)
    monkeypatch.delenv("KQ_QMM_ROUTE")
    monkeypatch.delenv("KQ_QMM_ROUTE_STRICT")
    _sweep(codec, wv, s, ref_w, 1000, ms=[3, 8])


# The q6_k and q3_k kernels read 2-byte words, since every other 210- or
# 110-byte superblock starts 2 mod 4. A base 2 or 16 bytes in runs the
# route bit for bit like the aligned tensor, and the default route matches
# the reference on it. An odd base raises before any route runs
# (tests/test_weight_base.py).
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("offset", [1, 2, 16])
@pytest.mark.parametrize("codec", ["q6_k", "q3_k"])
def test_verify_nax_half_word_weight_alignment(codec, offset, monkeypatch):
    w, s, ref_w = _setup(codec, 1000)
    pad = mx.zeros((offset,), dtype=mx.uint8)
    wv = mx.concatenate([pad, w.reshape(-1)])[offset:].reshape(w.shape)
    mx.eval(wv)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    x = (mx.random.normal((8, K)) * 0.5).astype(mx.bfloat16)
    if offset % 2:
        with pytest.raises(ValueError, match="2-byte boundary"):
            mx.eval(kq.quantized_matmul(x, wv, s, codec, transpose=True))
        return
    y_nax = kq.quantized_matmul(x, wv, s, codec, transpose=True)
    y_ref = kq.quantized_matmul(x, w, s, codec, transpose=True)
    mx.eval(y_nax, y_ref)
    assert _bits_equal(y_nax, y_ref)
    _sweep(codec, wv, s, ref_w, 1000, ms=[1, 5, 8])
    monkeypatch.delenv("KQ_QMM_ROUTE")
    monkeypatch.delenv("KQ_QMM_ROUTE_STRICT")
    _sweep(codec, wv, s, ref_w, 1000, ms=[1, 5, 8])


# The q6_k split count is the smallest divisor of the K steps (128 each) at
# or above max(ceil(1280 / simdgroups), ceil(K / 2560)), at most 16, else
# the largest divisor under it. N 3072, K 4096 targets 14 and takes 16 of
# 32 steps. N 1024, K 5120 targets 16, which does not divide 40 steps, and
# takes 10. N 20480, K 10240 targets 4 from the K walk. The default count
# runs bit for bit like the count forced through the lever, and unlike the
# neighbouring count.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize(
    "n_out, k, count, other",
    [(3072, 4096, 16, 8), (1024, 5120, 10, 8), (20480, 10240, 4, 2)],
)
def test_verify_nax_q6_k_split_rule(n_out, k, count, other, monkeypatch):
    wf = mx.random.normal((64, k), key=mx.random.key(4)) * 0.1
    w1, s = kq.quantize(wf, "q6_k")
    w = mx.contiguous(mx.tile(w1, (n_out // 64, 1)))
    ref_w = kq.dequantize(w1, s, "q6_k").astype(mx.float32).T
    mx.eval(w, s, ref_w)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    monkeypatch.delenv("KQ_VERIFY_NAX_SPLITS", raising=False)
    x = (mx.random.normal((8, k)) * 0.5).astype(mx.bfloat16)

    def run(splits=None):
        if splits is not None:
            monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", str(splits))
        y = kq.quantized_matmul(x, w, s, "q6_k", transpose=True)
        mx.eval(y)
        monkeypatch.delenv("KQ_VERIFY_NAX_SPLITS", raising=False)
        return y

    y_def = run()
    assert _bits_equal(y_def, run(count))
    assert not _bits_equal(y_def, run(other))
    ref = x.astype(mx.float32) @ ref_w
    err = float(mx.abs(y_def[:, :64].astype(mx.float32) - ref).max())
    assert err < 2e-2 * float(mx.abs(ref).max())


# q2_k, q3_k, q4_k, q5_k and the eight-block q4_0 kernel take the cheapest
# power-of-two split count by GPU waves of 640 simdgroups, or another
# divisor that costs at most 3/4 as much, with a partial charge of 640 for
# q4_k and q4_0 and 2560 for the others. At 2560, N 12288, K 5120 takes 8,
# since 5 costs less in the model but not 3/4 as much, and N 1536, K 8960
# takes 35 over 10. At 640 the same N 1536, K 8960 in 256-weight steps
# takes 7 over 35. N 2560, K 9728 takes 38 at 2560 and 19 at 640, where 2
# splits ran up to 1.65x slower than 19. N 6144, K 5120 in 256-weight steps
# takes 2 over 10. N 1024, K 5120 takes 20, above 16, which on q4_k is
# every K step. The default count runs bit for bit like the count forced
# through the lever, and unlike the neighbouring count.
WAVE_SPLITS = [
    *[
        (codec, n_out, k, count, other)
        for codec in ("q5_k", "q3_k", "q2_k")
        for n_out, k, count, other in [
            (12288, 5120, 8, 5),
            (1536, 8960, 35, 10),
            (2560, 9728, 38, 19),
            (1024, 5120, 20, 10),
        ]
    ],
    ("q4_k", 6144, 5120, 2, 10),
    ("q4_k", 1536, 8960, 7, 35),
    ("q4_k", 2560, 9728, 19, 38),
    ("q4_k", 1024, 5120, 20, 10),
    ("q4_0", 6144, 5120, 2, 10),
    ("q4_0", 2560, 9728, 19, 38),
]


@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("codec, n_out, k, count, other", WAVE_SPLITS)
def test_verify_nax_wave_split_rule(codec, n_out, k, count, other, monkeypatch):
    wf = mx.random.normal((64, k), key=mx.random.key(4)) * 0.1
    w1, s = kq.quantize(wf, codec)
    w = mx.contiguous(mx.tile(w1, (n_out // 64, 1)))
    ref_w = kq.dequantize(w1, s, codec).astype(mx.float32).T
    mx.eval(w, s, ref_w)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    monkeypatch.delenv("KQ_VERIFY_NAX_SPLITS", raising=False)
    x = (mx.random.normal((8, k)) * 0.5).astype(mx.bfloat16)

    def run(splits=None):
        if splits is not None:
            monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", str(splits))
        y = kq.quantized_matmul(x, w, s, codec, transpose=True)
        mx.eval(y)
        monkeypatch.delenv("KQ_VERIFY_NAX_SPLITS", raising=False)
        return y

    y_def = run()
    assert _bits_equal(y_def, run(count))
    assert not _bits_equal(y_def, run(other))
    ref = x.astype(mx.float32) @ ref_w
    err = float(mx.abs(y_def[:, :64].astype(mx.float32) - ref).max())
    assert err < 2e-2 * float(mx.abs(ref).max())


# Random q4_k and q5_k wire bytes cover every scale, min and fifth-bit
# pattern, against gguf-py's dequant. d and dmin are drawn in a sane
# half range, the other bytes uniform.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("splits", ["", "3"])
@pytest.mark.parametrize("n_out, k", [(136, 1024), (1000, 4096)])
@pytest.mark.parametrize("codec", ["q4_k", "q5_k"])
def test_verify_nax_kquant_random_wire(codec, n_out, k, splits, dtype, monkeypatch):
    from kqref import quants

    gtype, wpb, bpb, _, _ = CODECS[codec]
    rng = np.random.default_rng(5)
    wire = synth_wire(rng, codec, bpb, n_out * (k // wpb))
    dmin = rng.uniform(0.0, 0.08, wire.shape[0]).astype(np.float16)
    wire[:, 2:4] = dmin.view(np.uint8).reshape(-1, 2)
    wire = wire.reshape(n_out, (k // wpb) * bpb)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype)
    ref_w = mx.array(ref.astype(np.float32)).T
    w = mx.array(wire)
    s = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(w, s, ref_w)
    mx.random.seed(11)
    if splits:
        monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", splits)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    _sweep(codec, w, s, ref_w, n_out, ms=range(1, 9), dtype=dtype, k=k)


# Random q6_k, q3_k and q2_k wire bytes cover every code, scale and high
# bit, with d (and the q2_k dmin) of both signs written at its offset,
# against gguf-py's dequant. The second d range is half-subnormal, as in
# real tensors with small weights. For q6_k the default and "5" counts
# split N 136, K 1024 into 8 and 4 and N 1000, K 3328 into 13 and 2, so
# splits also start on the second half of a superblock.
WIRE_D = {"q6_k": (208,), "q3_k": (108,), "q2_k": (80, 82)}


@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("splits", ["", "5"])
@pytest.mark.parametrize("d_lo, d_hi", [(0.002, 0.01), (1e-6, 6e-5)])
@pytest.mark.parametrize("n_out, k", [(136, 1024), (1000, 3328), (40, 5120)])
@pytest.mark.parametrize("codec", list(WIRE_D))
def test_verify_nax_signed_d_random_wire(
    codec, n_out, k, d_lo, d_hi, splits, dtype, monkeypatch
):
    from kqref import quants

    gtype, wpb, bpb, _, _ = CODECS[codec]
    rng = np.random.default_rng(5)
    wire = synth_wire(rng, codec, bpb, n_out * (k // wpb))
    for off in WIRE_D[codec]:
        d = rng.uniform(d_lo, d_hi, wire.shape[0])
        d = d * rng.choice([-1, 1], wire.shape[0])
        wire[:, off : off + 2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    wire = wire.reshape(n_out, (k // wpb) * bpb)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype)
    assert np.isfinite(ref).all()
    ref_w = mx.array(ref.astype(np.float32)).T
    w = mx.array(wire)
    s = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(w, s, ref_w)
    mx.random.seed(11)
    if splits:
        monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", splits)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    _sweep(codec, w, s, ref_w, n_out, ms=range(1, 9), dtype=dtype, k=k)


# Random q4_0 wire bytes cover every nibble, 0 included, and scales of
# both signs, against gguf-py's dequant. K 1024 and 4096 run the
# eight-block kernel and K 2176 the two-block kernel.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("splits", ["", "3"])
@pytest.mark.parametrize("n_out, k", [(136, 1024), (1000, 4096), (1000, 2176)])
def test_verify_nax_q4_0_random_wire(n_out, k, splits, dtype, monkeypatch):
    from kqref import quants

    gtype, wpb, bpb, _, _ = CODECS["q4_0"]
    rng = np.random.default_rng(5)
    wire = synth_wire(rng, "q4_0", bpb, n_out * (k // wpb))
    wire[:, 1] |= (rng.integers(0, 2, wire.shape[0]) << 7).astype(np.uint8)
    wire = wire.reshape(n_out, (k // wpb) * bpb)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype)
    ref_w = mx.array(ref.astype(np.float32)).T
    w = mx.array(wire)
    s = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(w, s, ref_w)
    mx.random.seed(11)
    if splits:
        monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", splits)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    _sweep("q4_0", w, s, ref_w, n_out, ms=range(1, 9), dtype=dtype, k=k)


# KQ_DISABLE_NAX turns the route off with the other NAX routes.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("codec", VNAX_CODECS)
def test_verify_nax_disable_nax(codec, monkeypatch):
    w, s, ref_w = _setup(codec, 1000)
    monkeypatch.setenv("KQ_DISABLE_NAX", "1")
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    x = mx.zeros((8, K), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match="does not serve"):
        mx.eval(kq.quantized_matmul(x, w, s, codec, transpose=True))
    monkeypatch.delenv("KQ_QMM_ROUTE")
    monkeypatch.delenv("KQ_QMM_ROUTE_STRICT")
    _sweep(codec, w, s, ref_w, 1000, ms=[3, 5, 8])


# verify_nax folds the block scale into half weights and accumulates in
# float32, so one activation channel near the top of the half range stays
# finite and within the contract.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("codec", VNAX_CODECS)
def test_verify_nax_outlier_channel(codec, dtype, monkeypatch):
    w, s, ref_w = _setup(codec, 1000)
    monkeypatch.setenv("KQ_QMM_ROUTE", "verify_nax")
    monkeypatch.setenv("KQ_QMM_ROUTE_STRICT", "1")
    for m in (1, 5, 8):
        x = mx.random.normal((m, K)) * 0.5
        x[:, 7] = 3.0e4
        x = x.astype(dtype)
        y = kq.quantized_matmul(x, w, s, codec, transpose=True).astype(mx.float32)
        ref = x.astype(mx.float32) @ ref_w
        mx.eval(y, ref)
        assert bool(mx.isfinite(y).all()), f"{codec} M{m}: non-finite output"
        err = float((mx.abs(y - ref)).max() / (mx.abs(ref).max() + 1e-6))
        assert err < 2e-2, f"{codec} M{m}: rel err {err:.3e}"


def _bits_equal(a, b):
    return bool(mx.array_equal(a, b).item())


# The default route dispatches verify_nax from its entry, and
# KQ_VERIFY_NAX moves the entry both ways: the default output is
# bit-identical to the route each case expects and differs from the other
# kernels timed, and verify_nax agrees with a second 16-bit kernel
# (verify_mma, or NAX split-K for the codecs without it) to the output
# rounding. Below the entry and with verify_nax off, the case expects the
# table's route: pq2_0 and q4_0 fall back to verify_mma at M 8, the others
# to NAX split-K.
# The route is chosen when the graph evaluates, so each output evaluates
# under its own environment. Per-row qmv claims these widths at N 1000 by
# default, so it is off here.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("codec", VNAX_CODECS)
def test_verify_nax_default_dispatch(codec, dtype, monkeypatch):
    monkeypatch.setenv("KQ_NAX_QMV", "0")
    w, s, _ = _setup(codec, 1000)

    def run(x, route=None):
        if route is None:
            monkeypatch.delenv("KQ_QMM_ROUTE", raising=False)
        else:
            monkeypatch.setenv("KQ_QMM_ROUTE", route)
        y = kq.quantized_matmul(x, w, s, codec, transpose=True)
        mx.eval(y)
        monkeypatch.delenv("KQ_QMM_ROUTE", raising=False)
        return y

    e = VNAX_ENTRY[codec]
    cases = [
        (e - 1, None, _nax_default_route(codec, 1000, e - 1, qmv=False)),
        (e, None, "verify_nax"),
        ((e + 8) // 2, None, "verify_nax"),
        (8, None, "verify_nax"),
        (2, "2", "verify_nax"),
        (8, "0", _nax_default_route(codec, 1000, 8, qmv=False, vnax=False)),
    ]
    for m, entry, expect in cases:
        if entry is None:
            monkeypatch.delenv("KQ_VERIFY_NAX", raising=False)
        else:
            monkeypatch.setenv("KQ_VERIFY_NAX", entry)
        x = (mx.random.normal((m, K)) * 0.5).astype(dtype)
        y_def = run(x)
        alt = "verify_mma" if codec in VMMA_CODECS else "nax_splitk"
        ys = {r: run(x, r) for r in ("verify_nax", alt, expect)}
        tag = f"{codec} M{m} KQ_VERIFY_NAX={entry}"
        for r, y in ys.items():
            assert _bits_equal(y_def, y) == (r == expect), f"{tag} vs {r}"
        y_nax, y_mma = ys["verify_nax"], ys[alt]
        err = float(
            mx.abs(y_nax.astype(mx.float32) - y_mma.astype(mx.float32)).max()
            / (mx.abs(y_mma.astype(mx.float32)).max() + 1e-6)
        )
        assert err < 2e-2, f"{tag}: verify_nax vs {alt} rel err {err:.3e}"


# A forced KQ_VERIFY_MMA or KQ_QMM_SPLITK_NAX takes precedence over the
# default verify_nax entry, so an A/B arm times the route it names. An
# explicit KQ_VERIFY_NAX wins back. On a codec verify_mma does not serve,
# a forced KQ_VERIFY_MMA leaves verify_nax in place. Per-row qmv is off,
# as above.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("codec", VNAX_CODECS)
def test_verify_nax_yields_to_forced_levers(codec, monkeypatch):
    monkeypatch.setenv("KQ_NAX_QMV", "0")
    w, s, _ = _setup(codec, 1000)
    m = max(6, VNAX_ENTRY[codec])
    x = (mx.random.normal((m, K)) * 0.5).astype(mx.bfloat16)

    def run(route=None, **env):
        for k in ("KQ_VERIFY_MMA", "KQ_QMM_SPLITK_NAX", "KQ_VERIFY_NAX"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        if route is None:
            monkeypatch.delenv("KQ_QMM_ROUTE", raising=False)
        else:
            monkeypatch.setenv("KQ_QMM_ROUTE", route)
        y = kq.quantized_matmul(x, w, s, codec, transpose=True)
        mx.eval(y)
        return y

    y_nax = run("verify_nax")
    y_sk = run("nax_splitk", KQ_QMM_SPLITK_NAX="8")
    assert not _bits_equal(y_nax, y_sk)
    if codec in VMMA_CODECS:
        y_mma = run("verify_mma")
        assert not _bits_equal(y_nax, y_mma)
        assert _bits_equal(run(KQ_VERIFY_MMA="2"), y_mma)
    else:
        assert _bits_equal(run(KQ_VERIFY_MMA="2"), y_nax)
    assert _bits_equal(run(KQ_VERIFY_MMA=str(m + 1)), y_nax)
    assert _bits_equal(run(KQ_QMM_SPLITK_NAX="8"), y_sk)
    assert _bits_equal(run(KQ_QMM_SPLITK_NAX="8", KQ_VERIFY_NAX="3"), y_nax)


# q6_k, q3_k and q2_k enter verify_nax at M 3 on a vocab head (N >=
# 100000). One row below the boundary they take their entry for wide N,
# with mv_ext below it.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("codec", ["q6_k", "q3_k", "q2_k"])
def test_verify_nax_head_entry(codec, monkeypatch):
    _clear_levers(monkeypatch)
    k = 256
    wf = mx.random.normal((64, k), key=mx.random.key(3)) * 0.1
    w1, s = kq.quantize(wf, codec)
    w = mx.contiguous(mx.tile(w1, (100000 // 64 + 1, 1))[:100000])
    mx.eval(w, s)
    for n, vnax_m in ((100000, 3), (99999, _vnax_entry(codec, 99999))):
        wn = w[:n]
        for m in (3, 4, 5):
            x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
            expect = "verify_nax" if m >= vnax_m else "mv_ext"
            other = "mv_ext" if expect == "verify_nax" else "verify_nax"
            y_def = _run_route(monkeypatch, x, wn, s, codec)
            y_exp = _run_route(monkeypatch, x, wn, s, codec, expect)
            y_oth = _run_route(monkeypatch, x, wn, s, codec, other)
            tag = f"{codec} N{n} M{m} expect {expect}"
            assert _bits_equal(y_def, y_exp), tag
            assert not _bits_equal(y_def, y_oth), tag


# kq_nax_small_m for the codecs the dispatch tests pin: the per-row qmv
# limit per N bucket (N <= 512, 1024, 2048, larger), then the split-K
# entries for N <= 1024 and above.
NAX_SMALL_M = {
    "q3_k": ((7, 3, 1, 1), 5, 5),
    "q2_k": ((8, 5, 3, 2), 2, 5),
    "pq2_0": ((8, 5, 3, 2), 9, 9),
    "ptq1_0": ((8, 5, 3, 2), 9, 9),
    "q4_0": ((8, 6, 3, 2), 8, 8),
    "q4_k": ((8, 5, 4, 2), 2, 5),
    "q5_k": ((7, 3, 2, 1), 2, 5),
    "q6_k": ((7, 2, 1, 1), 8, 9),
    "q8_0": ((7, 5, 2, 1), 2, 7),
    "iq2_xs": ((1, 1, 1, 1), 2, 4),
    "iq2_xxs": ((8, 5, 2, 1), 9, 13),
}
# verify_mma entries on NAX GPUs (kq_verify_mma_min_m_nax), which
# verify_nax shadows on the codecs it serves, and the qmv rows per
# threadgroup of the two codecs whose M 2 mat-vec route is verify_qmv.
# The Prism codecs take verify_qmv through M 4 (kq_prism_verify_max_m).
VMMA_NAX_ENTRY = {"ptq1_0": 3, "pq2_0": 5, "q4_0": 5}
QMV_BN = {"q4_k": 4, "q8_0": 8}
PRISM_VERIFY_MAX_M = {"pq2_0": 4, "ptq1_0": 4}
# Per-row qmv, mv_ext and verify_qmv dot the exact weights in float32 and
# agree bit for bit on most codecs, so a pin cannot tell them apart. The
# NAX tiles and the verify kernels round the weights to a 16-bit type, so
# the pins check every seam against those, whose results differ.
BITSAME = {"qmv", "mv_ext", "verify_qmv"}
# Read once per process, so the default routing tests skip when they are set.
STATIC_LEVERS = ("KQ_VERIFY_EXT", "KQ_QMM_SPLITK")


# The table below N 100000. Vocab-head widths move the q6_k, q3_k and q2_k
# entries, which test_verify_nax_head_entry covers.
def _nax_default_route(codec, n, m, qmv=True, vnax=True):
    assert n < 100000, n
    qmv_m, splitk_small, splitk = NAX_SMALL_M[codec]
    b = 0 if n <= 512 else 1 if n <= 1024 else 2 if n <= 2048 else 3
    if qmv and m <= qmv_m[b]:
        return "qmv"
    if vnax and _vnax_entry(codec, n) <= m <= 8:
        return "verify_nax"
    if VMMA_NAX_ENTRY.get(codec, 9) <= m <= 8:
        return "verify_mma"
    if m >= (splitk_small if n <= 1024 else splitk):
        return "nax_splitk"
    if m <= PRISM_VERIFY_MAX_M.get(codec, 0):
        return "verify_qmv"
    if m == 2 and codec in QMV_BN:
        return "verify_qmv" if n % QMV_BN[codec] == 0 else "qmv"
    return "mv_ext"


def _run_route(monkeypatch, x, w, s, codec, route=None):
    if route is None:
        monkeypatch.delenv("KQ_QMM_ROUTE", raising=False)
    else:
        monkeypatch.setenv("KQ_QMM_ROUTE", route)
    y = kq.quantized_matmul(x, w, s, codec, transpose=True)
    mx.eval(y)
    monkeypatch.delenv("KQ_QMM_ROUTE", raising=False)
    return y


def _clear_levers(monkeypatch):
    if any(os.environ.get(k) is not None for k in STATIC_LEVERS):
        pytest.skip("a process-static routing lever is set")
    for k in (
        "KQ_NAX_QMV",
        "KQ_VERIFY_NAX",
        "KQ_VERIFY_MMA",
        "KQ_QMM_SPLITK_NAX",
        "KQ_DISABLE_NAX",
        "KQ_QMM_ROUTE",
        "KQ_QMM_ROUTE_STRICT",
    ):
        monkeypatch.delenv(k, raising=False)


# The default small-M routing on NAX GPUs runs the kernel the per-codec
# table names, bit for bit, on both sides of every N bucket edge and of
# the q2_k verify_nax width, and across each seam: the qmv limit, the
# verify routes, the split-K entry and the mat-vec band between. Each pin
# is also checked against the route a limit one step off would give,
# unless both are in BITSAME.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX routing only")
@pytest.mark.parametrize("n_out", [512, 513, 1024, 1025, 2048, 2049, 4096, 4097])
@pytest.mark.parametrize("codec", sorted(NAX_SMALL_M))
def test_nax_small_m_default_dispatch(codec, n_out, monkeypatch):
    _clear_levers(monkeypatch)
    w, s, _ = _setup(codec, n_out)
    ms = [2, 3, 4, 5, 6, 7, 8, 9]
    if codec == "iq2_xxs":
        ms += [12, 13]
    for m in ms:
        x = (mx.random.normal((m, K)) * 0.5).astype(mx.bfloat16)
        expect = _nax_default_route(codec, n_out, m)
        y_def = _run_route(monkeypatch, x, w, s, codec)
        y_exp = _run_route(monkeypatch, x, w, s, codec, expect)
        tag = f"{codec} N{n_out} M{m} expect {expect}"
        assert _bits_equal(y_def, y_exp), tag
        if expect == "qmv":
            alt = _nax_default_route(codec, n_out, m, qmv=False)
        else:
            alt = "qmv"
        if not {expect, alt} <= BITSAME:
            y_alt = _run_route(monkeypatch, x, w, s, codec, alt)
            assert not _bits_equal(y_def, y_alt), f"{tag} vs {alt}"


# KQ_NAX_QMV=0 turns the route off and a value sets the limit at every N.
# The default yields to a forced KQ_VERIFY_NAX or KQ_VERIFY_MMA on a codec
# they serve and to a forced KQ_QMM_SPLITK_NAX, and an explicit
# KQ_NAX_QMV wins back. KQ_DISABLE_NAX turns it off with the NAX routes.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX routing only")
def test_nax_qmv_levers(monkeypatch):
    _clear_levers(monkeypatch)

    def run(codec, w, s, x, route=None, **env):
        _clear_levers(monkeypatch)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        y = _run_route(monkeypatch, x, w, s, codec, route)
        _clear_levers(monkeypatch)
        return y

    w, s, _ = _setup("q4_0", 500)
    x = (mx.random.normal((4, K)) * 0.5).astype(mx.bfloat16)
    y_qmv = run("q4_0", w, s, x, "qmv")
    y_nax = run("q4_0", w, s, x, "verify_nax")
    y_mma = run("q4_0", w, s, x, "verify_mma")
    y_sk = run("q4_0", w, s, x, "nax_splitk", KQ_QMM_SPLITK_NAX="8")
    for y in (y_nax, y_mma, y_sk):
        assert not _bits_equal(y, y_qmv)
    assert _bits_equal(run("q4_0", w, s, x), y_qmv)
    assert _bits_equal(run("q4_0", w, s, x, KQ_NAX_QMV="0"), y_nax)
    assert _bits_equal(run("q4_0", w, s, x, KQ_VERIFY_NAX="3"), y_nax)
    assert _bits_equal(run("q4_0", w, s, x, KQ_VERIFY_NAX="5"), y_qmv)
    y = run("q4_0", w, s, x, KQ_VERIFY_MMA="2", KQ_VERIFY_NAX="0")
    assert _bits_equal(y, y_mma)
    assert _bits_equal(run("q4_0", w, s, x, KQ_QMM_SPLITK_NAX="8"), y_sk)
    y = run("q4_0", w, s, x, KQ_QMM_SPLITK_NAX="8", KQ_NAX_QMV="4")
    assert _bits_equal(y, y_qmv)

    # A verify lever on a codec it does not serve leaves qmv in place: at
    # N 500 the q4_1 limit is 8 and split-K enters at 2.
    w, s, _ = _setup("q4_1", 500)
    x = (mx.random.normal((4, K)) * 0.5).astype(mx.bfloat16)
    y_qmv = run("q4_1", w, s, x, "qmv")
    assert not _bits_equal(y_qmv, run("q4_1", w, s, x, "nax_splitk"))
    assert _bits_equal(run("q4_1", w, s, x, KQ_VERIFY_NAX="2"), y_qmv)
    assert _bits_equal(run("q4_1", w, s, x, KQ_VERIFY_MMA="2"), y_qmv)

    # KQ_NAX_QMV widens the route past the table at a large N.
    w, s, _ = _setup("q4_1", 4096)
    x = (mx.random.normal((6, K)) * 0.5).astype(mx.bfloat16)
    y_qmv = run("q4_1", w, s, x, "qmv")
    y_sk = run("q4_1", w, s, x, "nax_splitk")
    assert not _bits_equal(y_qmv, y_sk)
    assert _bits_equal(run("q4_1", w, s, x), y_sk)
    assert _bits_equal(run("q4_1", w, s, x, KQ_NAX_QMV="6"), y_qmv)

    # KQ_DISABLE_NAX turns the route off even when KQ_NAX_QMV asks for it.
    # pq2_0 at M 5 is the check: its non-NAX routes there (mv_ext,
    # verify_mma) differ from qmv bit for bit.
    w, s, _ = _setup("pq2_0", 500)
    x = (mx.random.normal((5, K)) * 0.5).astype(mx.bfloat16)
    y_off = run("pq2_0", w, s, x, KQ_DISABLE_NAX="1")
    assert not _bits_equal(y_off, run("pq2_0", w, s, x, "qmv"))
    y = run("pq2_0", w, s, x, KQ_DISABLE_NAX="1", KQ_NAX_QMV="8")
    assert _bits_equal(y, y_off)


# Probe for the process-static levers: prints, per named route, whether the
# default output equals that route's output bit for bit.
_STATIC_PROBE = """
import json, os, sys
import mlx.core as mx
import mlx_kquant as kq
codec, m = sys.argv[1], int(sys.argv[2])
mx.random.seed(11)
w, s = kq.quantize(mx.random.normal((1000, 1024)) * 0.1, codec)
x = (mx.random.normal((m, 1024)) * 0.5).astype(mx.bfloat16)
def run(route=None):
    os.environ.pop("KQ_QMM_ROUTE", None)
    if route:
        os.environ["KQ_QMM_ROUTE"] = route
    y = kq.quantized_matmul(x, w, s, codec, transpose=True)
    mx.eval(y)
    return y
y = run()
print(json.dumps({r: bool(mx.array_equal(y, run(r)).item()) for r in sys.argv[3:]}))
"""


# KQ_VERIFY_EXT and KQ_QMM_SPLITK are read once per process, so each case
# runs in a fresh interpreter. A set KQ_VERIFY_EXT keeps per-row qmv and
# NAX split-K off the mat-vec band, and a forced KQ_QMM_SPLITK keeps them
# off its band. At N 1000 the q4_1 qmv limit is 8 and split-K enters at 2,
# and pq2_0 runs verify_nax from M 3, which neither lever moves.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX routing only")
@pytest.mark.parametrize(
    "lever, codec, m, expect, other",
    [
        ("KQ_VERIFY_EXT=1", "q4_1", 3, "mv_ext", "nax_splitk"),
        ("KQ_VERIFY_EXT=1", "pq2_0", 3, "verify_nax", "qmv"),
        ("KQ_VERIFY_EXT=0", "q4_1", 3, "verify_qmv", "nax_splitk"),
        ("KQ_VERIFY_EXT=0", "pq2_0", 3, "verify_nax", "qmv"),
        ("KQ_QMM_SPLITK=8", "q4_1", 3, "splitk", "qmv,nax_splitk"),
    ],
)
def test_nax_small_m_yields_to_static_levers(lever, codec, m, expect, other):
    env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    name, value = lever.split("=")
    env[name] = value
    others = other.split(",")
    out = subprocess.run(
        [sys.executable, "-c", _STATIC_PROBE, codec, str(m), expect, *others],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res[expect] and not any(res[o] for o in others), (lever, codec, m, res)
