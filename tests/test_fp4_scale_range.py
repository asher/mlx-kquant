"""Every fp4 decode path on wires that cover the whole scale byte range.

synth_wire draws nvfp4 scales from a narrow band, so nothing in the suite
reaches scale 0x00, the UE4M3 subnormals 0x01 to 0x07, the NaN code 0x7F or
a byte with bit 7 set, and nothing drives the mxfp4 exponent far from 127.
These wires cycle every scale byte and carry every E2M1 code. float32
dequantize must match the gguf reference exactly (a four-bit scale times
a two-bit code is exact in float32; only the sign of zero may differ).
The matmul families are held against float64 on the same exact weights
with a tolerance for their accumulation order.

mxfp4 runs e8m0 bytes 8 to 200: a scale under 2^-119 times a code lands in
the float32 subnormal range, where the GPU may flush, and above 2^73 a row
sum passes the float32 range.
The float16 arms use the nvfp4 wire whose scales stay under 0x48, since the
large scales overflow a float16 output.

Under KQUANT_FORCE_CPU only dequantize and the default matmul route run,
with KQ_CPU_NEON=0: the NEON GEMV rounds activations to int8 by design
(test_cpu_neon.py covers it), and this file tests the decode.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

N, K, E = 256, 512, 4
FORCE_CPU = bool(os.environ.get("KQUANT_FORCE_CPU"))
# Largest |err| / max |ref| by activation dtype (test_alu_dequantize_rows R1).
REL = {"bfloat16": 6e-3, "float16": 1e-3}
ROUTES = ("qmv", "mv_ext", "verify_nax", "splitk", "nax", "nax_splitk")


def _wire(name):
    rng = np.random.default_rng(7)
    if name.startswith("nvfp4"):
        nb = N * (K // 64)
        w = rng.integers(0, 256, (nb, 36), dtype=np.uint8)
        if name == "nvfp4_all":
            sc = np.arange(nb * 4) % 256
        elif name == "nvfp4_low":  # 0x00, the subnormals and small normals
            sc = np.arange(nb * 4) % 0x18
        else:  # nvfp4_mid: the band real files use, float16 safe
            sc = 0x08 + np.arange(nb * 4) % 0x40
        w[:, :4] = sc.reshape(nb, 4).astype(np.uint8)
        return "nvfp4", GT.NVFP4, w.reshape(N, -1)
    nb = N * (K // 32)
    w = rng.integers(0, 256, (nb, 17), dtype=np.uint8)
    w[:, 0] = (8 + np.arange(nb) % 193).astype(np.uint8)
    return "mxfp4", GT.MXFP4, w.reshape(N, -1)


def _ref(wire, gtype):
    return quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)


WIRES = ("nvfp4_all", "nvfp4_low", "nvfp4_mid", "mxfp4_all")


@pytest.mark.parametrize("name", WIRES)
def test_dequantize_matches_reference_bit_for_bit(name):
    codec, gtype, wire = _wire(name)
    ref = _ref(wire, gtype)
    got = kq.dequantize(
        mx.array(wire), mx.zeros((1,), mx.uint8), codec, dtype=mx.float32
    )
    got = np.asarray(got)
    assert got.shape == ref.shape
    # Exact values. The code for negative zero decodes to -0.0 here and to
    # +0.0 in gguf-py, whose table is integers, so the bits are compared only
    # where the value is not zero.
    assert np.array_equal(got, ref)
    nz = got != 0
    assert np.array_equal(got.view(np.uint32)[nz], ref.view(np.uint32)[nz])
    # Every code and every scale byte is present, so the wire tests what it
    # claims to.
    bpb, off = (36, 4) if codec == "nvfp4" else (17, 1)
    blocks = wire.reshape(-1, bpb)
    if name == "nvfp4_all":
        assert len(np.unique(blocks[:, :4])) == 256
    assert len(np.unique(blocks[:, off:] & 0xF)) == 16


def _check(out, x64, w64, dtn):
    ref = x64 @ w64.T
    got = np.asarray(out.astype(mx.float32)).astype(np.float64)
    assert got.shape == ref.shape
    assert np.isfinite(got).all()
    err = np.abs(got - ref).max()
    assert err <= REL[dtn] * np.abs(ref).max(), (err, np.abs(ref).max())


def _arms(name):
    dts = ("bfloat16", "float16") if name == "nvfp4_mid" else ("bfloat16",)
    return [(name, dtn) for dtn in dts]


@pytest.mark.parametrize("name,dtn", [a for n in WIRES for a in _arms(n)])
@pytest.mark.parametrize("M", (1, 4, 8, 16, 256))
def test_matmul_routes(name, dtn, M):
    codec, gtype, wire = _wire(name)
    w64 = _ref(wire, gtype).astype(np.float64)
    w = mx.array(wire)
    s = mx.zeros((1,), mx.uint8)
    x = mx.random.normal((M, K), key=mx.random.key(M)).astype(getattr(mx, dtn))
    mx.eval(x)
    x64 = np.asarray(x.astype(mx.float32)).astype(np.float64)
    if FORCE_CPU:
        prev = os.environ.get("KQ_CPU_NEON")
        os.environ["KQ_CPU_NEON"] = "0"
        try:
            out = kq.quantized_matmul(x, w, s, codec, transpose=True)
            mx.eval(out)
        finally:
            if prev is None:
                os.environ.pop("KQ_CPU_NEON", None)
            else:
                os.environ["KQ_CPU_NEON"] = prev
        _check(out, x64, w64, dtn)
        return
    _check(kq.quantized_matmul(x, w, s, codec, transpose=True), x64, w64, dtn)
    ran = []
    old = os.environ.get("KQ_QMM_ROUTE")
    os.environ["KQ_QMM_ROUTE_STRICT"] = "1"
    try:
        for r in ROUTES:
            os.environ["KQ_QMM_ROUTE"] = r
            try:
                out = kq.quantized_matmul(x, w, s, codec, transpose=True)
                mx.eval(out)
            except (RuntimeError, ValueError):
                continue  # strict mode: the route does not serve this cell
            _check(out, x64, w64, dtn)
            ran.append(r)
    finally:
        os.environ.pop("KQ_QMM_ROUTE_STRICT", None)
        if old is None:
            os.environ.pop("KQ_QMM_ROUTE", None)
        else:
            os.environ["KQ_QMM_ROUTE"] = old
    # The forced routes are the small-M band; at M 256 only the default
    # route (the NAX GEMM) serves this shape.
    assert ran or M >= 256, "no forced route served the cell"


@pytest.mark.skipif(FORCE_CPU, reason="gather and MoE kernels are Metal paths")
@pytest.mark.parametrize("name,dtn", [a for n in WIRES for a in _arms(n)])
@pytest.mark.parametrize("M", (8, 256))
def test_gather_and_moe_families(name, dtn, M):
    codec, gtype, wire = _wire(name)
    dt = getattr(mx, dtn)
    stack = np.stack([np.roll(wire, i, axis=0) for i in range(E)])
    w64 = np.stack([_ref(stack[e], gtype) for e in range(E)]).astype(np.float64)
    we = mx.array(stack)
    s = mx.zeros((1,), mx.uint8)
    x = mx.random.normal((M, K), key=mx.random.key(M)).astype(dt)
    mx.eval(x)
    x64 = np.asarray(x.astype(mx.float32)).astype(np.float64)
    ind = np.sort(np.arange(M) % E).astype(np.uint32)
    ref = np.stack([x64[i] @ w64[ind[i]].T for i in range(M)])

    def check(out):
        got = np.asarray(out.astype(mx.float32)).astype(np.float64).reshape(ref.shape)
        assert np.isfinite(got).all()
        assert np.abs(got - ref).max() <= REL[dtn] * np.abs(ref).max()

    check(
        kq.gather_qmm(
            x.reshape(M, 1, K),
            we,
            s,
            codec,
            rhs_indices=mx.array(ind),
            transpose=True,
            sorted_indices=True,
        )
    )
    tmap, counts = kq.expert_tile_map(mx.array(ind), E)
    check(kq.gather_qmm_seg(x, we, s, codec, tmap, counts))
    # The GLU product squares the weight range: mxfp4's 2^73 scales pass
    # float32, and the float16 arm scales x by 2^-5 to stay in range.
    if M <= 16 and name != "mxfp4_all":
        if dtn == "float16":
            x = x * 0.03125
            x64 = x64 * 0.03125
        ind2 = mx.array((np.arange(M * 2) % E).astype(np.uint32)).reshape(M, 2)
        out = kq.moe_glu_gather_kq(x, we, we, codec, ind2, act="silu")
        got = np.asarray(out.astype(mx.float32)).astype(np.float64)
        i2 = np.asarray(ind2).reshape(-1)
        g = np.stack([x64[t // 2] @ w64[i2[t]].T for t in range(M * 2)])
        want = (g / (1 + np.exp(-g))) * g
        assert np.isfinite(got).all()
        assert (
            np.abs(got.reshape(want.shape) - want).max()
            <= 4 * REL[dtn] * np.abs(want).max()
        )
