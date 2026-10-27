"""fp4 decode in the matmul, gather and MoE kernels over every scale byte.

synth_wire draws nvfp4 scales from a narrow band, so nothing else in the
suite reaches scale 0x00, the UE4M3 subnormals 0x01 to 0x07, the NaN code
0x7F or a byte with bit 7 set, or an mxfp4 exponent far from 127. Here each
weight row has one scale exponent, and its sub-blocks draw the mantissa
(nvfp4) or the exponent's low bit (mxfp4) at random. Together the rows carry
every scale byte and every E2M1 code, and each output is checked against its
own row's magnitude:

    |got - ref| <= TOL * (|x| @ |w|^T) + the dtype's smallest normal

per element. A float32 sum rounded once at the store and once per split-K
partial (bf16 partials by default, fp16 ones under KQ_SPLITK_F32=0) is off
by at most two unit roundoffs of |x| @ |w|^T; the floor admits a GPU that
flushes subnormal outputs. One bound on max |ref| over the whole output
cannot see rows far below the top ones. Only transpose=True runs here.

float32 dequantize must match the gguf reference exactly (a four-bit scale
times a two-bit code is exact in float32; only the sign of zero may differ).

mxfp4 exponents run 8 to 239: below that a weight times an activation can be
a float32 subnormal, which the GPU may flush, and above it a row sum leaves
the float32 range. The GLU arms square the range and use 64 to 180. The
float16 arms use nvfp4 scales 0x08 to 0x47, except the one-hot test, which
reads every nvfp4 weight back exactly in float16 on all 256 scale bytes.

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
from kqref import mxfp4_deinterleave

import mlx_kquant as kq

N, K, E = 256, 512, 4
FORCE_CPU = bool(os.environ.get("KQUANT_FORCE_CPU"))
# Two unit roundoffs: 2 * 2^-8 for bfloat16, 2 * 2^-11 for float16.
TOL = {"bfloat16": 2.0**-7, "float16": 2.0**-10}
FLOOR = {"bfloat16": 2.0**-126, "float16": 2.0**-14}
ROUTES = ("qmv", "mv_ext", "verify_mma", "verify_nax", "splitk", "nax", "nax_splitk")
WIRES = ("nvfp4_all", "nvfp4_low", "nvfp4_mid", "mxfp4_all", "mxfp4_mid")


def _wire(name):
    rng = np.random.default_rng(7)
    n = np.arange(N)[:, None]
    if name.startswith("nvfp4"):
        base = {"nvfp4_all": n, "nvfp4_low": n % 0x18, "nvfp4_mid": 0x08 + n % 0x40}
        w = rng.integers(0, 256, (N, K // 64, 36), dtype=np.uint8)
        sc = (base[name] & 0xF8) | rng.integers(0, 8, (N, K // 16))
        w[:, :, :4] = sc.reshape(N, K // 64, 4).astype(np.uint8)
        return "nvfp4", GT.NVFP4, w.reshape(N, -1)
    lo, span = {"mxfp4_all": (8, 231), "mxfp4_mid": (64, 116)}[name]
    w = rng.integers(0, 256, (N, K // 32, 17), dtype=np.uint8)
    w[:, :, 0] = (lo + n % span + rng.integers(0, 2, (N, K // 32))).astype(np.uint8)
    return "mxfp4", GT.MXFP4, w.reshape(N, -1)


def _ref(wire, gtype):
    return quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)


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


def _check(out, ref, bound, dtn):
    got = np.asarray(out.astype(mx.float32)).astype(np.float64).reshape(ref.shape)
    assert np.isfinite(got).all()
    bad = np.abs(got - ref) > TOL[dtn] * bound + FLOOR[dtn]
    assert not bad.any(), f"{int(bad.sum())} outputs past the bound"


def _x(M, dtn):
    x = mx.random.normal((M, K), key=mx.random.key(M)).astype(getattr(mx, dtn))
    mx.eval(x)
    return x, np.asarray(x.astype(mx.float32)).astype(np.float64)


def _arms(name, one_hot=False):
    both = name == "nvfp4_mid" or (one_hot and name == "nvfp4_all")
    dts = ("bfloat16", "float16") if both else ("bfloat16",)
    return [(name, dtn) for dtn in dts]


@pytest.mark.parametrize("name,dtn", [a for n in WIRES for a in _arms(n)])
@pytest.mark.parametrize("M", (1, 4, 8, 16, 256))
def test_matmul_routes(name, dtn, M):
    codec, gtype, wire = _wire(name)
    w64 = _ref(wire, gtype).astype(np.float64)
    w = mx.array(wire)
    s = mx.zeros((1,), mx.uint8)
    x, x64 = _x(M, dtn)
    ref, bound = x64 @ w64.T, np.abs(x64) @ np.abs(w64).T
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
        _check(out, ref, bound, dtn)
        return
    _check(kq.quantized_matmul(x, w, s, codec, transpose=True), ref, bound, dtn)
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
            _check(out, ref, bound, dtn)
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


# One-hot activations make each output one weight times 1, so every
# kernel must reproduce every weight of the wire exactly, whatever its sum
# order. Row m of x is one-hot at column M j + m. nvfp4 runs verify_mma
# and both verify_nax kernels (KQ_VERIFY_NAX_NVFP4_SB=0 and 1), and every
# wire runs each forced route that serves M 8.
@pytest.mark.skipif(FORCE_CPU, reason="the forced routes are Metal paths")
@pytest.mark.parametrize("name,dtn", [a for n in WIRES for a in _arms(n, one_hot=True)])
def test_one_hot_rows_reproduce_every_weight(name, dtn):
    codec, gtype, wire = _wire(name)
    ref = _ref(wire, gtype).T
    want = np.asarray(mx.array(ref).astype(getattr(mx, dtn)).astype(mx.float32))
    w = mx.array(wire)
    s = mx.zeros((1,), mx.uint8)
    M = 8
    xs = []
    for j in range(K // M):
        x = np.zeros((M, K), np.float32)
        x[np.arange(M), M * j + np.arange(M)] = 1.0
        xs.append(mx.array(x).astype(getattr(mx, dtn)))
    arms = [(r, None) for r in ROUTES]
    if codec == "nvfp4":
        arms += [("verify_nax", "0"), ("verify_nax", "1")]
    keys = ("KQ_QMM_ROUTE", "KQ_QMM_ROUTE_STRICT", "KQ_VERIFY_NAX_NVFP4_SB")
    old = {k: os.environ.get(k) for k in keys}
    ran = []
    try:
        os.environ["KQ_QMM_ROUTE_STRICT"] = "1"
        for route, sb in arms:
            os.environ["KQ_QMM_ROUTE"] = route
            os.environ.pop("KQ_VERIFY_NAX_NVFP4_SB", None)
            if sb is not None:
                os.environ["KQ_VERIFY_NAX_NVFP4_SB"] = sb
            try:
                ys = [kq.quantized_matmul(x, w, s, codec, transpose=True) for x in xs]
                mx.eval(ys)
            except (RuntimeError, ValueError):
                continue  # strict mode: the route does not serve this cell
            got = np.concatenate([np.asarray(y.astype(mx.float32)) for y in ys])
            nz = want != 0
            assert np.array_equal(got, want), (route, sb)
            assert np.array_equal(got.view(np.uint32)[nz], want.view(np.uint32)[nz])
            ran.append((route, sb))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert ran, "no forced route served the cell"
    if codec == "nvfp4":
        want_ran = {("verify_mma", None)}
        if kq.nax_available():
            want_ran |= {("verify_nax", None), ("verify_nax", "0"), ("verify_nax", "1")}
        assert want_ran <= set(ran), ran


def _experts(name):
    codec, gtype, wire = _wire(name)
    stack = np.stack([np.roll(wire, i, axis=0) for i in range(E)])
    w64 = np.stack([_ref(stack[e], gtype) for e in range(E)]).astype(np.float64)
    return codec, stack, w64


def _per_row(x64, w64, ind):
    """ref and bound for token i against expert ind[i]."""
    ref = np.stack([x64[i] @ w64[e].T for i, e in enumerate(ind)])
    bound = np.stack([np.abs(x64[i]) @ np.abs(w64[e]).T for i, e in enumerate(ind)])
    return ref, bound


def _silu_glu(g):
    with np.errstate(over="ignore"):
        return g / (1 + np.exp(-g)) * g


@pytest.mark.skipif(FORCE_CPU, reason="gather and MoE kernels are Metal paths")
@pytest.mark.parametrize("name,dtn", [a for n in WIRES for a in _arms(n)])
@pytest.mark.parametrize("M", (8, 256))
def test_gather_and_moe_families(name, dtn, M):
    codec, stack, w64 = _experts(name)
    we = mx.array(stack)
    s = mx.zeros((1,), mx.uint8)
    x, x64 = _x(M, dtn)
    ind = np.sort(np.arange(M) % E).astype(np.uint32)
    ref, bound = _per_row(x64, w64, ind)
    out = kq.gather_qmm(
        x.reshape(M, 1, K),
        we,
        s,
        codec,
        rhs_indices=mx.array(ind),
        transpose=True,
        sorted_indices=True,
    )
    _check(out, ref, bound, dtn)
    tmap, counts = kq.expert_tile_map(mx.array(ind), E)
    _check(kq.gather_qmm_seg(x, we, s, codec, tmap, counts), ref, bound, dtn)
    if M > 16:
        return
    # Decode: two expert slots per token.
    ind2 = (np.arange(M * 2) % E).astype(np.uint32)
    x2 = np.repeat(x64, 2, axis=0)
    ref, bound = _per_row(x2, w64, ind2)
    h = mx.repeat(x[:, None], 2, axis=1)
    out = kq.gather_qmv_kq(h, we, codec, mx.array(ind2).reshape(M, 2))
    _check(out, ref, bound, dtn)
    if name == "mxfp4_all":
        return  # the GLU squares the range past float32
    if dtn == "float16":  # keep the float16 product in range
        x = x * 0.03125
        x64 = np.asarray(x.astype(mx.float32)).astype(np.float64)
        ref, bound = _per_row(np.repeat(x64, 2, axis=0), w64, ind2)
    out = kq.moe_glu_gather_kq(
        x, we, we, codec, mx.array(ind2).reshape(M, 2), act="silu"
    )
    _check(out, _silu_glu(ref), bound * np.abs(ref), dtn)


@pytest.mark.skipif(FORCE_CPU, reason="gather and MoE kernels are Metal paths")
@pytest.mark.parametrize("name", ("mxfp4_all", "mxfp4_mid"))
def test_packed_mxfp4_moe(name):
    """The MLX-layout mxfp4 MoE kernels (the gpt-oss path) on the same wires."""
    M, dtn = 8, "bfloat16"
    _, stack, w64 = _experts(name)
    pw, ps = (mx.array(a) for a in mxfp4_deinterleave(stack))
    zb = mx.zeros((E, N), mx.float32)
    x, x64 = _x(M, dtn)
    ind2 = (np.arange(M * 2) % E).astype(np.uint32)
    inds = mx.array(ind2).reshape(M, 2)
    ref, bound = _per_row(np.repeat(x64, 2, axis=0), w64, ind2)
    h = mx.repeat(x[:, None], 2, axis=1)
    _check(kq.gather_qmv_bias(h, pw, ps, zb, inds), ref, bound, dtn)
    sc = np.linspace(0.25, 1.0, M * 2).astype(np.float32)
    mix = (sc[:, None] * ref).reshape(M, 2, N).sum(1)
    mix_bound = (sc[:, None] * bound).reshape(M, 2, N).sum(1)
    out = kq.gather_qmv_mix_bias(h, pw, ps, zb, inds, mx.array(sc.reshape(M, 2)))
    _check(out, mix, mix_bound, dtn)
    if name == "mxfp4_all":
        return  # the GLU squares the range past float32
    # gpt-oss clamped SwiGLU with the clamp out of reach: g sig(1.702 g) (u + 1).
    out = kq.moe_glu_gather(x, pw, ps, zb, pw, ps, zb, inds, 1.702, 1e30)
    with np.errstate(over="ignore"):
        want = ref / (1 + np.exp(-1.702 * ref)) * (ref + 1)
    _check(out, want, bound * (np.abs(ref) + 1), dtn)
