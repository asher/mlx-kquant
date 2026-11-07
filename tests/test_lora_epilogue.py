#!/usr/bin/env python3
"""LoRA epilogue validation: the optional lora_a / lora_b / lora_rows and
lora_ids / lora_table operands of quantized_matmul,
quantized_matmul_qmv_bias, gather_qmv_kq and gather_qmv_mix_ns_kq, checked
against an independent f32 reference (gguf-py dequant + numpy) across the
codec matrix, the base routes (M=1 qmv, M in [2, 8] verify, M=64 qmm) and
both activation dtypes. The epilogue is codec independent; the sweep proves
it composes with every base dispatch and that the delta it adds matches
rows * (x @ A) @ B to f16 rounding.

The dense ops run on CPU too (KQUANT_FORCE_CPU exercises eval_cpu's
epilogue); the gathered ops are Metal-only and skip there.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq
from mlx_kquant.nn import KQuantLinear

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
CPU = bool(os.environ.get("KQUANT_FORCE_CPU"))

# codec -> (gguf type, is_kquant fixture)
CODECS = {
    "q4_0": (GT.Q4_0, False),
    "q5_1": (GT.Q5_1, False),
    "q8_0": (GT.Q8_0, False),
    "q2_k": (GT.Q2_K, True),
    "q4_k": (GT.Q4_K, True),
    "q6_k": (GT.Q6_K, True),
}
N, K = 256, 512
MOE_E, MOE_N, MOE_K = 4, 128, 512
RANK = 4
DTYPES = [mx.float16, mx.bfloat16]
DELTA_TOL = 4e-2  # relative Frobenius on (adapted - base) vs the f32 delta


def _lin_wire_ref(codec, gtype, is_kquant):
    if is_kquant:
        path = os.path.join(FIX, f"{codec}.npz")
        if not os.path.exists(path):
            pytest.skip(f"no fixture {path}")
        wire = np.load(path)["wire"].astype(np.uint8)
    else:
        rng = np.random.default_rng(7)
        w = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        wire = quants.quantize(w, gtype).astype(np.uint8)
    ref = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
    return wire, ref


def _moe_wire_ref(codec, gtype, is_kquant):
    if is_kquant:
        path = os.path.join(FIX, f"{codec}_moe.npz")
        if not os.path.exists(path):
            pytest.skip(f"no fixture {path}")
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
    g = _f32(got) if isinstance(got, mx.array) else np.asarray(got, np.float32)
    r = np.asarray(ref).astype(np.float32)
    return float(np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-6))


def _f32(a):
    return np.array(a.astype(mx.float32))


def _lora(rng, shape_a, shape_b, dtype, gain=1.0):
    a = mx.array((rng.standard_normal(shape_a) * 0.2 * gain).astype(np.float32))
    b = mx.array((rng.standard_normal(shape_b) * 0.2 * gain).astype(np.float32))
    a, b = a.astype(dtype), b.astype(dtype)
    return a, b, _f32(a), _f32(b)


@pytest.mark.parametrize("codec", list(CODECS))
@pytest.mark.parametrize("m", [1, 2, 8, 64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_dense_epilogue_matches_reference(codec, m, dtype):
    gtype, is_kq = CODECS[codec]
    wire, ref = _lin_wire_ref(codec, gtype, is_kq)
    rows_n, kk = ref.shape
    lin = KQuantLinear(in_dims=kk, out_dims=rows_n, bias=False, codec=codec)
    lin.weight = mx.array(wire)
    rng = np.random.default_rng(3)
    x = mx.array((rng.standard_normal((m, kk)) * 0.1).astype(np.float32)).astype(dtype)
    a, b, a32, b32 = _lora(rng, (kk, RANK), (RANK, rows_n), dtype)
    rows = np.ones(m, dtype=np.float32)
    if m > 1:
        rows[0] = 0.0  # a bare row: the epilogue must skip it
        rows[1] = 0.5
    base = lin(x)
    got = lin(x, lora=(a, b, mx.array(rows)))
    got_norows = lin(x, lora=(a, b))
    mx.eval(base, got, got_norows)
    x32 = _f32(x)
    delta = rows[:, None] * ((x32 @ a32) @ b32)
    assert got.shape == base.shape and got.dtype == base.dtype
    assert _rel(_f32(got) - _f32(base), delta) < DELTA_TOL
    assert _rel(_f32(got_norows) - _f32(base), (x32 @ a32) @ b32) < DELTA_TOL
    # The base half is untouched: adapted output vs the f32 oracle.
    oracle = x32 @ ref.T + delta
    assert _rel(got, oracle) < 3e-2


# KQ_LORA_SPLIT settings: the default route, the one-dispatch epilogue and
# the split route.
LEVERS = [None, "0", "1"]


def _set_lever(monkeypatch, lever):
    if lever is None:
        monkeypatch.delenv("KQ_LORA_SPLIT", raising=False)
    else:
        monkeypatch.setenv("KQ_LORA_SPLIT", lever)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lever", LEVERS)
def test_qmv_bias_route_carries_the_epilogue(dtype, lever, monkeypatch):
    # q8_0 + bias at M=1 routes KQuantLinear to quantized_matmul_qmv_bias.
    _set_lever(monkeypatch, lever)
    wire, ref = _lin_wire_ref("q8_0", GT.Q8_0, False)
    rows_n, kk = ref.shape
    lin = KQuantLinear(in_dims=kk, out_dims=rows_n, bias=True, codec="q8_0")
    lin.weight = mx.array(wire)
    rng = np.random.default_rng(5)
    bias = (rng.standard_normal(rows_n) * 0.1).astype(np.float32)
    lin.bias = mx.array(bias)
    x = mx.array((rng.standard_normal((1, kk)) * 0.1).astype(np.float32)).astype(dtype)
    a, b, a32, b32 = _lora(rng, (kk, RANK), (RANK, rows_n), dtype)
    got = lin(x, lora=(a, b, mx.array([0.5], dtype=mx.float32)))
    base = lin(x)
    mx.eval(got, base)
    x32 = _f32(x)
    delta = 0.5 * ((x32 @ a32) @ b32)
    assert _rel(_f32(got) - _f32(base), delta) < DELTA_TOL
    assert _rel(got, x32 @ ref.T + bias + delta) < 3e-2


def test_dense_operand_validation():
    wire, ref = _lin_wire_ref("q8_0", GT.Q8_0, False)
    rows_n, kk = ref.shape
    w = mx.array(wire)
    scales = mx.zeros((1,), dtype=mx.float16)
    x = mx.zeros((2, kk), dtype=mx.float16)
    a = mx.zeros((kk, RANK), dtype=mx.float16)
    b = mx.zeros((RANK, rows_n), dtype=mx.float16)
    with pytest.raises(ValueError, match="go together"):
        kq.quantized_matmul(x, w, scales, "q8_0", lora_a=a)
    with pytest.raises(ValueError, match="activation dtype"):
        kq.quantized_matmul(
            x, w, scales, "q8_0", lora_a=a.astype(mx.bfloat16), lora_b=b
        )
    with pytest.raises(ValueError, match="do not compose"):
        kq.quantized_matmul(x, w, scales, "q8_0", lora_a=a, lora_b=b[:, : rows_n // 2])
    with pytest.raises(ValueError, match="lora_rows must be shaped"):
        kq.quantized_matmul(
            x,
            w,
            scales,
            "q8_0",
            lora_a=a,
            lora_b=b,
            lora_rows=mx.ones((3,), dtype=mx.float32),
        )
    with pytest.raises(ValueError, match="need lora_a"):
        kq.quantized_matmul(
            x, w, scales, "q8_0", lora_rows=mx.ones((2,), dtype=mx.float32)
        )
    with pytest.raises(ValueError, match="exceeds"):
        kq.quantized_matmul(
            x,
            w,
            scales,
            "q8_0",
            lora_a=mx.zeros((kk, 600), dtype=mx.float16),
            lora_b=mx.zeros((600, rows_n), dtype=mx.float16),
        )


@pytest.mark.parametrize("codec", ["q4_k", "q6_k", "q8_0"])
@pytest.mark.parametrize("m", [1, 2, 8, 64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_dense_stacked_epilogue_picks_each_rows_adapter(codec, m, dtype):
    """A stack of adapters on one dense op: lora_ids picks each row's
    entry, lora_table remaps it (below 0 skips the row) and lora_rows
    scales it, on every base route."""
    gtype, is_kq = CODECS[codec]
    wire, ref = _lin_wire_ref(codec, gtype, is_kq)
    rows_n, kk = ref.shape
    lin = KQuantLinear(in_dims=kk, out_dims=rows_n, bias=False, codec=codec)
    lin.weight = mx.array(wire)
    rng = np.random.default_rng(13)
    s = 3
    x = mx.array((rng.standard_normal((m, kk)) * 0.1).astype(np.float32)).astype(dtype)
    a, b, a32, b32 = _lora(rng, (s, kk, RANK), (s, RANK, rows_n), dtype)
    ids = rng.integers(0, s, m).astype(np.uint32)
    table = np.array([-1, 2, 0], dtype=np.int32)  # entry 0 skips its rows
    fac = rng.uniform(0.25, 1.0, m).astype(np.float32)
    base = lin(x)
    got = lin(x, lora=(a, b, None, mx.array(ids)))
    got_tab = lin(x, lora=(a, b, mx.array(fac), mx.array(ids), mx.array(table)))
    mx.eval(base, got, got_tab)
    x32 = _f32(x)
    want = np.stack([(x32[i] @ a32[e]) @ b32[e] for i, e in enumerate(ids)])
    assert _rel(_f32(got) - _f32(base), want) < DELTA_TOL
    want_tab = np.stack(
        [
            np.zeros(rows_n, np.float32)
            if table[e] < 0
            else fac[i] * ((x32[i] @ a32[table[e]]) @ b32[table[e]])
            for i, e in enumerate(ids)
        ]
    )
    dt = _f32(got_tab) - _f32(base)
    skipped = table[ids] < 0
    assert np.array_equal(_f32(got_tab)[skipped], _f32(base)[skipped])
    if (~skipped).any():
        assert _rel(dt[~skipped], want_tab[~skipped]) < DELTA_TOL


SPLIT_CODECS = {
    **CODECS,
    "q4_1": (GT.Q4_1, False),
    "q5_0": (GT.Q5_0, False),
    "q3_k": (GT.Q3_K, True),
    "q5_k": (GT.Q5_K, True),
}


@pytest.mark.parametrize("codec", list(SPLIT_CODECS))
@pytest.mark.parametrize("m", [1, 4, 5, 32, 33])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rank", [64, 192])
def test_dense_stacked_split_route_mixed_rows(codec, m, dtype, rank, monkeypatch):
    """Up to 32 rows run the split route (z before the base dispatch, the
    apply after it), 33 rows the one-dispatch epilogue. Mixed batches of
    skipped and live rows on every base route: skipped rows keep the base
    output bit for bit, live rows match the f32 delta and the one-dispatch
    epilogue."""
    gtype, is_kq = SPLIT_CODECS[codec]
    wire, ref = _lin_wire_ref(codec, gtype, is_kq)
    rows_n, kk = ref.shape
    lin = KQuantLinear(in_dims=kk, out_dims=rows_n, bias=False, codec=codec)
    lin.weight = mx.array(wire)
    rng = np.random.default_rng(29 + m)
    s = 3
    x = mx.array((rng.standard_normal((m, kk)) * 0.1).astype(np.float32)).astype(dtype)
    a, b, a32, b32 = _lora(rng, (s, kk, rank), (s, rank, rows_n), dtype)
    a, b = a * 0.5, b * 0.5
    a32, b32 = _f32(a), _f32(b)
    table = np.array([-1, 0, 1, 2], dtype=np.int32)
    ids = rng.integers(0, 4, m).astype(np.uint32)
    if m > 1:
        ids[0], ids[1] = 0, 3  # one skipped and one live row at least
    fac = rng.uniform(0.25, 1.0, m).astype(np.float32)
    if m > 2:
        fac[2] = 0.0
    lora = (a, b, mx.array(fac), mx.array(ids), mx.array(table))
    base = lin(x)
    got = lin(x, lora=lora)
    mx.eval(base, got)
    monkeypatch.setenv("KQ_LORA_SPLIT", "0")
    one = lin(x, lora=lora)
    mx.eval(one)
    monkeypatch.delenv("KQ_LORA_SPLIT")
    x32 = _f32(x)
    ent = table[ids]
    live = (ent >= 0) & (fac != 0)
    assert np.array_equal(_f32(got)[~live], _f32(base)[~live])
    if live.any():
        want = np.stack(
            [
                fac[i] * ((x32[i] @ a32[ent[i]]) @ b32[ent[i]])
                for i in np.flatnonzero(live)
            ]
        )
        dt = (_f32(got) - _f32(base))[live]
        assert _rel(dt, want) < DELTA_TOL
        assert _rel(dt, (_f32(one) - _f32(base))[live]) < DELTA_TOL


@pytest.mark.parametrize("rank", [1, 5, 320])
def test_dense_split_route_odd_ranks_and_widths(rank, monkeypatch):
    """Ranks off the 4- and 8-wide vector loads, a rank past 256 and an
    output width off the 4-wide vector loads. KQ_LORA_SPLIT=1 keeps the
    small ranks on the split route."""
    monkeypatch.setenv("KQ_LORA_SPLIT", "1")
    rng = np.random.default_rng(31)
    rows_n, kk, m, s, dtype = 100, 512, 6, 2, mx.float16
    w = (rng.standard_normal((rows_n, kk)) * 0.1).astype(np.float32)
    wire = quants.quantize(w, GT.Q8_0).astype(np.uint8)
    lin = KQuantLinear(in_dims=kk, out_dims=rows_n, bias=False, codec="q8_0")
    lin.weight = mx.array(wire)
    x = mx.array((rng.standard_normal((m, kk)) * 0.1).astype(np.float32)).astype(dtype)
    a, b, a32, b32 = _lora(rng, (s, kk, rank), (s, rank, rows_n), dtype)
    ids = np.array([0, 1, 2, 1, 0, 2], dtype=np.uint32)
    table = np.array([1, -1, 0], dtype=np.int32)
    base = lin(x)
    got = lin(x, lora=(a, b, None, mx.array(ids), mx.array(table)))
    mx.eval(base, got)
    x32 = _f32(x)
    ent = table[ids]
    live = ent >= 0
    assert np.array_equal(_f32(got)[~live], _f32(base)[~live])
    want = np.stack(
        [(x32[i] @ a32[ent[i]]) @ b32[ent[i]] for i in np.flatnonzero(live)]
    )
    assert _rel((_f32(got) - _f32(base))[live], want) < DELTA_TOL


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lever", LEVERS)
def test_qmv_bias_route_carries_the_stacked_epilogue(dtype, lever, monkeypatch):
    _set_lever(monkeypatch, lever)
    wire, ref = _lin_wire_ref("q8_0", GT.Q8_0, False)
    rows_n, kk = ref.shape
    lin = KQuantLinear(in_dims=kk, out_dims=rows_n, bias=True, codec="q8_0")
    lin.weight = mx.array(wire)
    rng = np.random.default_rng(17)
    lin.bias = mx.array((rng.standard_normal(rows_n) * 0.1).astype(np.float32))
    x = mx.array((rng.standard_normal((1, kk)) * 0.1).astype(np.float32)).astype(dtype)
    a, b, a32, b32 = _lora(rng, (2, kk, RANK), (2, RANK, rows_n), dtype)
    got = lin(x, lora=(a, b, None, mx.array([1], dtype=mx.uint32)))
    base = lin(x)
    mx.eval(got, base)
    x32 = _f32(x)
    assert _rel(_f32(got) - _f32(base), (x32 @ a32[1]) @ b32[1]) < DELTA_TOL


def test_dense_stacked_operand_validation():
    wire, ref = _lin_wire_ref("q8_0", GT.Q8_0, False)
    rows_n, kk = ref.shape
    w = mx.array(wire)
    scales = mx.zeros((1,), dtype=mx.float16)
    x = mx.zeros((2, kk), dtype=mx.float16)
    a3 = mx.zeros((2, kk, RANK), dtype=mx.float16)
    b3 = mx.zeros((2, RANK, rows_n), dtype=mx.float16)
    ids = mx.zeros((2,), dtype=mx.uint32)
    with pytest.raises(ValueError, match="need lora_ids"):
        kq.quantized_matmul(x, w, scales, "q8_0", lora_a=a3, lora_b=b3)
    with pytest.raises(ValueError, match="need stacked"):
        kq.quantized_matmul(
            x, w, scales, "q8_0", lora_a=a3[0], lora_b=b3[0], lora_ids=ids
        )
    with pytest.raises(ValueError, match="lora_table needs lora_ids"):
        kq.quantized_matmul(
            x,
            w,
            scales,
            "q8_0",
            lora_a=a3,
            lora_b=b3,
            lora_table=mx.zeros((2,), dtype=mx.int32),
        )
    with pytest.raises(ValueError, match="lora_ids must match"):
        kq.quantized_matmul(
            x,
            w,
            scales,
            "q8_0",
            lora_a=a3,
            lora_b=b3,
            lora_ids=mx.zeros((3,), dtype=mx.uint32),
        )
    with pytest.raises(ValueError, match="leading dim"):
        kq.quantized_matmul(
            x, w, scales, "q8_0", lora_a=a3, lora_b=b3[:1], lora_ids=ids
        )


def test_vjp_rejects_the_epilogue():
    wire, ref = _lin_wire_ref("q8_0", GT.Q8_0, False)
    rows_n, kk = ref.shape
    w = mx.array(wire)
    scales = mx.zeros((1,), dtype=mx.float16)
    a = mx.zeros((kk, RANK), dtype=mx.float16)
    b = mx.zeros((RANK, rows_n), dtype=mx.float16)

    def f(x):
        return kq.quantized_matmul(x, w, scales, "q8_0", lora_a=a, lora_b=b).sum()

    x = mx.zeros((2, kk), dtype=mx.float16)
    with pytest.raises(ValueError, match="inference-only"):
        mx.eval(mx.grad(f)(x))


# gathered ops (Metal-only)

gathered = pytest.mark.skipif(CPU, reason="gathered kq ops are Metal-only")

MOE_CODECS = [c for c in CODECS if kq.codec_has_moe_glu(c)]


def _moe_setup(codec, dtype, seed=9, t=3, s=2):
    gtype, is_kq = CODECS[codec]
    wire, ref = _moe_wire_ref(codec, gtype, is_kq)
    e, n, kk = ref.shape
    rng = np.random.default_rng(seed)
    h = mx.array((rng.standard_normal((t, s, kk)) * 0.1).astype(np.float32)).astype(
        dtype
    )
    idx = rng.integers(0, e, size=(t, s)).astype(np.uint32)
    a, b, a32, b32 = _lora(rng, (e, kk, RANK), (e, RANK, n), dtype)
    return wire, ref, h, idx, a, b, a32, b32


def _ref_rows_delta(h32, ids, a32, b32, fac=None):
    t, s, _ = h32.shape
    out = np.zeros((t, s, b32.shape[-1]), dtype=np.float32)
    for i in range(t):
        for j in range(s):
            e = int(ids[i, j])
            if e < 0:
                continue
            f = 1.0 if fac is None else float(fac[i, j])
            out[i, j] = f * ((h32[i, j] @ a32[e]) @ b32[e])
    return out


@gathered
@pytest.mark.parametrize("codec", MOE_CODECS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_qmv_epilogue_matches_reference(codec, dtype):
    wire, ref, h, idx, a, b, a32, b32 = _moe_setup(codec, dtype)
    w = mx.array(wire)
    base = kq.gather_qmv_kq(h, w, codec, mx.array(idx))
    got = kq.gather_qmv_kq(h, w, codec, mx.array(idx), lora_a=a, lora_b=b)
    mx.eval(base, got)
    delta = _ref_rows_delta(_f32(h), idx, a32, b32)
    assert got.shape == base.shape
    assert _rel(_f32(got) - _f32(base), delta) < DELTA_TOL


@gathered
@pytest.mark.parametrize("codec", MOE_CODECS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_mix_ns_epilogue_matches_reference(codec, dtype):
    wire, ref, h, idx, a, b, a32, b32 = _moe_setup(codec, dtype, t=5, s=3)
    w = mx.array(wire)
    rng = np.random.default_rng(13)
    sc = rng.random((5, 3)).astype(np.float32)
    base = kq.gather_qmv_mix_ns_kq(h, w, codec, mx.array(idx), mx.array(sc))
    got = kq.gather_qmv_mix_ns_kq(
        h, w, codec, mx.array(idx), mx.array(sc), lora_a=a, lora_b=b
    )
    mx.eval(base, got)
    delta = (sc[..., None] * _ref_rows_delta(_f32(h), idx, a32, b32)).sum(1)
    assert got.shape == base.shape
    assert _rel(_f32(got) - _f32(base), delta) < DELTA_TOL


@gathered
@pytest.mark.parametrize("op", ["rows", "mix"])
def test_gather_ids_table_and_rows(op):
    # Routing indices are arena slots; lora_table maps slot -> adapter
    # expert (-1 = dead slot, skipped); lora_ids overrides the routing ids;
    # lora_rows scales per (token, slot) and 0 skips.
    codec, dtype = "q8_0", mx.float16
    wire, ref, h, idx, a, b, a32, b32 = _moe_setup(codec, dtype, t=4, s=2)
    e = ref.shape[0]
    w = mx.array(wire)
    rng = np.random.default_rng(17)
    slots = 6
    table = np.full(slots, -1, dtype=np.int32)
    table[:e] = rng.permutation(e).astype(np.int32)
    slot_idx = rng.integers(0, slots, size=(4, 2)).astype(np.uint32)
    slot_idx[0, 0] = e  # a dead slot (table[e] == -1)
    fac = rng.random((4, 2)).astype(np.float32)
    fac[1, 1] = 0.0
    sc = rng.random((4, 2)).astype(np.float32)
    # Base gathers by the routing ids as given (they index the stack here).
    routing = idx
    eff = table[slot_idx]
    common = dict(
        lora_a=a,
        lora_b=b,
        lora_ids=mx.array(slot_idx),
        lora_table=mx.array(table),
        lora_rows=mx.array(fac),
    )
    if op == "rows":
        base = kq.gather_qmv_kq(h, w, codec, mx.array(routing))
        got = kq.gather_qmv_kq(h, w, codec, mx.array(routing), **common)
        delta = _ref_rows_delta(_f32(h), eff, a32, b32, fac)
    else:
        base = kq.gather_qmv_mix_ns_kq(h, w, codec, mx.array(routing), mx.array(sc))
        got = kq.gather_qmv_mix_ns_kq(
            h, w, codec, mx.array(routing), mx.array(sc), **common
        )
        delta = (sc[..., None] * _ref_rows_delta(_f32(h), eff, a32, b32, fac)).sum(1)
    mx.eval(base, got)
    assert _rel(_f32(got) - _f32(base), delta) < DELTA_TOL
    # The dead slot and the zero-factor row contributed nothing.
    d = _f32(got) - _f32(base)
    if op == "rows":
        assert np.abs(d[0, 0]).max() < 1e-2
        assert np.abs(d[1, 1]).max() < 1e-2


@gathered
def test_gather_operand_validation():
    codec, dtype = "q8_0", mx.float16
    wire, ref, h, idx, a, b, a32, b32 = _moe_setup(codec, dtype)
    w = mx.array(wire)
    with pytest.raises(ValueError, match="expert count"):
        kq.gather_qmv_kq(h, w, codec, mx.array(idx), lora_a=a[:2], lora_b=b[:2])
    with pytest.raises(ValueError, match="routing indices"):
        kq.gather_qmv_kq(
            h,
            w,
            codec,
            mx.array(idx),
            lora_a=a,
            lora_b=b,
            lora_ids=mx.zeros((3, 4), dtype=mx.uint32),
        )
    with pytest.raises(ValueError, match="1-D"):
        kq.gather_qmv_mix_ns_kq(
            h,
            w,
            codec,
            mx.array(idx),
            mx.ones((3, 2), dtype=mx.float32),
            lora_a=a,
            lora_b=b,
            lora_table=mx.zeros((2, 2), dtype=mx.int32),
        )


# Strided / lazy operands: an unevaluated array reports the default (dense)
# layout, so build-time contiguity checks cannot see that mx.repeat of one
# value evaluates to a stride-0 broadcast view. The epilogue densifies such
# operands at eval; every route must match the dense-operand result exactly.
def _lazy_variants(dense_rows):
    m = dense_rows.shape[0] if dense_rows.ndim == 1 else dense_rows.shape
    n = int(np.prod(m)) if isinstance(m, tuple) else m
    v = float(np.asarray(dense_rows).reshape(-1)[0])
    out = {
        "repeat": mx.repeat(mx.array([v], dtype=mx.float32), n).reshape(
            dense_rows.shape
        ),
        "broadcast": mx.broadcast_to(mx.array(v, dtype=mx.float32), dense_rows.shape),
    }
    wide = np.zeros(dense_rows.shape[:-1] + (dense_rows.shape[-1] * 2,), np.float32)
    wide[..., ::2] = np.asarray(dense_rows)
    out["slice"] = mx.array(wide)[..., ::2]
    return out


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_dense_lazy_rows_match_dense(m, bias):
    codec, dtype = "q8_0", mx.float16
    wire, ref = _lin_wire_ref(codec, GT.Q8_0, False)
    rows_n, kk = ref.shape
    lin = KQuantLinear(in_dims=kk, out_dims=rows_n, bias=bias, codec=codec)
    lin.weight = mx.array(wire)
    rng = np.random.default_rng(11)
    if bias:
        lin.bias = mx.array((rng.standard_normal(rows_n) * 0.1).astype(np.float32))
    x = mx.array((rng.standard_normal((m, kk)) * 0.1).astype(np.float32)).astype(dtype)
    a, b, _, _ = _lora(rng, (kk, RANK), (RANK, rows_n), dtype)
    rows = np.full(m, 0.75, dtype=np.float32)
    want = lin(x, lora=(a, b, mx.array(rows)))
    mx.eval(want)
    for name, lazy in _lazy_variants(rows).items():
        got = lin(x, lora=(a, b, lazy))
        mx.eval(got)
        assert np.array_equal(_f32(got), _f32(want)), name
    # A strided a/b (column slice of a wider matrix) densifies too.
    a_wide = mx.concatenate([a, a * 0 + 1], axis=1)[:, :RANK]
    b_wide = mx.concatenate([b, b * 0 + 1], axis=0)[:RANK]
    got = lin(x, lora=(a_wide, b_wide, mx.array(rows)))
    mx.eval(got)
    assert np.array_equal(_f32(got), _f32(want))


@gathered
@pytest.mark.parametrize("op", ["rows", "mix"])
def test_gather_lazy_rows_and_ids_match_dense(op):
    codec, dtype = "q8_0", mx.float16
    wire, ref, h, idx, a, b, a32, b32 = _moe_setup(codec, dtype, t=4, s=2)
    w = mx.array(wire)
    rng = np.random.default_rng(23)
    sc = mx.array(rng.random((4, 2)).astype(np.float32))
    fac = np.full((4, 2), 0.6, dtype=np.float32)
    ids = mx.array(idx)

    def run(rows, lora_ids):
        kw = dict(lora_a=a, lora_b=b, lora_ids=lora_ids, lora_rows=rows)
        if op == "rows":
            y = kq.gather_qmv_kq(h, w, codec, ids, **kw)
        else:
            y = kq.gather_qmv_mix_ns_kq(h, w, codec, ids, sc, **kw)
        mx.eval(y)
        return _f32(y)

    want = run(mx.array(fac), ids)
    for name, lazy in _lazy_variants(fac).items():
        assert np.array_equal(run(lazy, ids), want), name
    # lazy ids: a strided view over a wider index matrix, and a stride-0
    # broadcast when every slot routes to one expert.
    wide = np.zeros((4, 4), np.uint32)
    wide[:, ::2] = idx
    assert np.array_equal(run(mx.array(fac), mx.array(wide)[:, ::2]), want)
    one = np.full((4, 2), int(idx[0, 0]), np.uint32)
    want_one = run(mx.array(fac), mx.array(one))
    got_one = run(
        mx.array(fac),
        mx.broadcast_to(mx.array(int(idx[0, 0]), dtype=mx.uint32), (4, 2)),
    )
    assert np.array_equal(got_one, want_one)


# Small ranks on the split route, forced with KQ_LORA_SPLIT=1, against the
# one-dispatch epilogue (KQ_LORA_SPLIT=0) and the default route. The z kernel
# folds the k lanes of each rank group with simd shuffles up to 16 groups and
# pads the group count to a power of two; more groups take the threadgroup
# pass. At 16 groups the rows meet in two passes. These ranks cover one to 16
# groups at every vector width, padded and not, and the first ranks past the
# shuffle path. 33 rows run the one-dispatch epilogue on every setting.
SMALL_RANKS = [1, 2, 3, 4, 8, 12, 13, 16, 20, 24, 32, 64, 72, 128, 17, 68]
SPLIT_ROWS = [1, 2, 4, 5, 8, 16, 32, 33]
# A bf16 output rounds a small delta away, so the bf16 cases use larger
# factors.
GAIN = {mx.float16: 1.0, mx.bfloat16: 4.0}


def _one_route(rank, m, kk, n):
    """Whether the default route of m rows is the one-dispatch epilogue
    (kq_lora_one_shape in src/kquant_lora_epilogue.h)."""
    if m > 32 or rank == 1:
        return True
    groups = m * -(-n // 1024)
    return kk <= 2048 and kk * rank <= 8192 and groups <= 64


def _split_vs_one(run, monkeypatch):
    got = run()
    monkeypatch.setenv("KQ_LORA_SPLIT", "1")
    split = run()
    monkeypatch.setenv("KQ_LORA_SPLIT", "0")
    one = run()
    monkeypatch.delenv("KQ_LORA_SPLIT")
    return got, split, one


def _assert_default_route(got, split, one, rank, m, kk, n):
    # The default route is one of the two, bit for bit.
    assert np.array_equal(got, one if _one_route(rank, m, kk, n) else split)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rank", SMALL_RANKS)
@pytest.mark.parametrize("m", SPLIT_ROWS)
def test_dense_split_route_small_ranks(rank, m, dtype, monkeypatch):
    codec = "q6_k"
    gtype, is_kq = CODECS[codec]
    wire, ref = _lin_wire_ref(codec, gtype, is_kq)
    rows_n, kk = ref.shape
    lin = KQuantLinear(in_dims=kk, out_dims=rows_n, bias=False, codec=codec)
    lin.weight = mx.array(wire)
    rng = np.random.default_rng(41 + rank * 64 + m)
    x = mx.array((rng.standard_normal((m, kk)) * 0.1).astype(np.float32)).astype(dtype)
    a, b, a32, b32 = _lora(rng, (kk, rank), (rank, rows_n), dtype, GAIN[dtype])
    fac = rng.uniform(0.25, 1.0, m).astype(np.float32)
    if m > 1:
        fac[m // 2] = 0.0
    base = lin(x)

    def run():
        y = lin(x, lora=(a, b, mx.array(fac)))
        mx.eval(y)
        return _f32(y)

    got, split, one = _split_vs_one(run, monkeypatch)
    _assert_default_route(got, split, one, rank, m, kk, rows_n)
    mx.eval(base)
    b0 = _f32(base)
    live = fac != 0
    want = fac[:, None] * ((_f32(x) @ a32) @ b32)
    for y in (got, split, one):
        assert np.array_equal(y[~live], b0[~live])
        assert _rel((y - b0)[live], want[live]) < DELTA_TOL
    assert _rel((got - b0)[live], (one - b0)[live]) < DELTA_TOL


@gathered
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", ["rows", "mix"])
@pytest.mark.parametrize("rank", SMALL_RANKS)
@pytest.mark.parametrize("m", SPLIT_ROWS)
def test_gather_split_route_small_ranks(op, rank, m, dtype, monkeypatch):
    """Gathered rows at the split route's ranks and row counts, with a dead
    arena slot and a zero-factor row. The score-mixed gather runs its own
    z and apply kernels at every row count."""
    codec = "q6_k"
    gtype, is_kq = CODECS[codec]
    wire, ref = _moe_wire_ref(codec, gtype, is_kq)
    e, n, kk = ref.shape
    s = min(m, 4) if m % 4 == 0 or m < 4 else 1
    t = m // s
    rng = np.random.default_rng(53 + rank * 64 + m)
    h = mx.array((rng.standard_normal((t, s, kk)) * 0.1).astype(np.float32)).astype(
        dtype
    )
    idx = rng.integers(0, e, size=(t, s)).astype(np.uint32)
    a, b, a32, b32 = _lora(rng, (e, kk, rank), (e, rank, n), dtype, GAIN[dtype])
    table = np.array([2, -1, 0, 3, 1], dtype=np.int32)
    slot = rng.integers(0, len(table), size=(t, s)).astype(np.uint32)
    if m > 1:
        slot.reshape(-1)[0] = 1  # a dead slot
    fac = rng.uniform(0.25, 1.0, (t, s)).astype(np.float32)
    if m > 2:
        fac.reshape(-1)[2] = 0.0
    sc = rng.random((t, s)).astype(np.float32)
    w = mx.array(wire)
    kw = dict(
        lora_a=a,
        lora_b=b,
        lora_ids=mx.array(slot),
        lora_table=mx.array(table),
        lora_rows=mx.array(fac),
    )
    if op == "rows":
        base = kq.gather_qmv_kq(h, w, codec, mx.array(idx))
    else:
        base = kq.gather_qmv_mix_ns_kq(h, w, codec, mx.array(idx), mx.array(sc))

    def run():
        if op == "rows":
            y = kq.gather_qmv_kq(h, w, codec, mx.array(idx), **kw)
        else:
            y = kq.gather_qmv_mix_ns_kq(h, w, codec, mx.array(idx), mx.array(sc), **kw)
        mx.eval(y)
        return _f32(y)

    got, split, one = _split_vs_one(run, monkeypatch)
    mx.eval(base)
    b0 = _f32(base)
    eff = table[slot]
    delta = _ref_rows_delta(_f32(h), eff, a32, b32, fac)
    if op == "mix":
        # The lever leaves the score-mixed gather alone.
        assert np.array_equal(split, got) and np.array_equal(one, got)
        delta = (sc[..., None] * delta).sum(1)
        assert _rel(got - b0, delta) < DELTA_TOL
        return
    _assert_default_route(got, split, one, rank, m, kk, n)
    live = (eff >= 0) & (fac != 0)
    for y in (got, split, one):
        assert np.array_equal(y[~live], b0[~live])
        assert _rel((y - b0)[live], delta[live]) < DELTA_TOL
    assert _rel((got - b0)[live], (one - b0)[live]) < DELTA_TOL


# (K, N, rank, rows): shapes on each side of the one-dispatch bounds. Rank 1
# takes one dispatch at any K; ranks 2 and up take it at K up to 2048, K x
# rank up to 8192 and rows x N tiles up to 64.
ROUTE_SHAPES = [
    (4096, 256, 1, 8),
    (4096, 256, 2, 1),
    (2048, 256, 4, 8),
    (2048, 256, 8, 8),
    (1024, 256, 8, 4),
    (1024, 256, 16, 4),
    (512, 4096, 8, 16),
    (512, 4096, 8, 17),
]


def _probe_cols(kk):
    # Every eighth input carries a large value.
    return np.arange(3, kk, 8)


def _probe_lora(rng, kk, n, rank, e, dtype):
    """Factors whose z the two routes round differently. At the probe
    columns x is 2^8 and A is +-2^8, as many of each sign, so the 2^16
    products cancel in the end, but a partial sum that holds some of them
    rounds the small products added to it to 2^-7 or coarser. The routes
    split K between threads and chunks and add the partial sums in different
    orders, so they round different small products."""
    a = rng.uniform(-1, 1, (e, kk, rank)).astype(np.float32)
    cols = _probe_cols(kk)
    sign = np.where(np.arange(len(cols)) % 2 == 0, 1.0, -1.0)
    for r in range(rank):
        a[:, cols, r] = rng.permutation(sign) * 2.0**8
    b = (rng.standard_normal((e, rank, n)) * 0.1).astype(np.float32)
    return mx.array(a).astype(dtype), mx.array(b).astype(dtype)


def _probe_x(rng, shape, dtype):
    x = rng.uniform(-1, 1, shape).astype(np.float32)
    x[..., _probe_cols(shape[-1])] = 2.0**8
    return mx.array(x).astype(dtype)


def _probe_wire(rng, n, kk):
    # Zero at the large inputs, so the base output stays small.
    w = (rng.standard_normal((n, kk)) * 0.1).astype(np.float32)
    w[:, _probe_cols(kk)] = 0.0
    return quants.quantize(w, GT.Q8_0).astype(np.uint8)


def _route_outputs(run, monkeypatch):
    got, split, one = _split_vs_one(run, monkeypatch)
    # The probe tells the routes apart.
    assert not np.array_equal(split, one)
    return got, split, one


@pytest.mark.skipif(CPU, reason="the routes are Metal kernels")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kk,n,rank,m", ROUTE_SHAPES)
def test_default_route_follows_the_shape(kk, n, rank, m, dtype, monkeypatch):
    rng = np.random.default_rng(61 + kk + rank + m)
    lin = KQuantLinear(in_dims=kk, out_dims=n, bias=False, codec="q8_0")
    lin.weight = mx.array(_probe_wire(rng, n, kk))
    x = _probe_x(rng, (m, kk), dtype)
    a, b = _probe_lora(rng, kk, n, rank, 1, dtype)

    def run():
        y = lin(x, lora=(a[0], b[0]))
        mx.eval(y)
        return _f32(y)

    got, split, one = _route_outputs(run, monkeypatch)
    _assert_default_route(got, split, one, rank, m, kk, n)


@pytest.mark.skipif(CPU, reason="the routes are Metal kernels")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kk,rank", [(512, 4), (4096, 2)])
def test_qmv_bias_default_route_follows_the_shape(kk, rank, dtype, monkeypatch):
    n = 256
    rng = np.random.default_rng(67 + kk + rank)
    lin = KQuantLinear(in_dims=kk, out_dims=n, bias=True, codec="q8_0")
    lin.weight = mx.array(_probe_wire(rng, n, kk))
    lin.bias = mx.array((rng.standard_normal(n) * 0.1).astype(np.float32))
    x = _probe_x(rng, (1, kk), dtype)
    a, b = _probe_lora(rng, kk, n, rank, 1, dtype)

    def run():
        y = lin(x, lora=(a[0], b[0], mx.array([0.5], dtype=mx.float32)))
        mx.eval(y)
        return _f32(y)

    got, split, one = _route_outputs(run, monkeypatch)
    _assert_default_route(got, split, one, rank, 1, kk, n)


@gathered
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kk,n,rank,m", [(512, 2048, 16, 8), (1536, 512, 8, 8)])
def test_gather_default_route_follows_the_shape(kk, n, rank, m, dtype, monkeypatch):
    e, s = 3, 4
    rng = np.random.default_rng(71 + kk + rank)
    w = mx.array(np.stack([_probe_wire(rng, n, kk) for _ in range(e)]))
    x = _probe_x(rng, (m // s, s, kk), dtype)
    idx = mx.array(rng.integers(0, e, (m // s, s)).astype(np.uint32))
    a, b = _probe_lora(rng, kk, n, rank, e, dtype)

    def run():
        y = kq.gather_qmv_kq(x, w, "q8_0", idx, lora_a=a, lora_b=b)
        mx.eval(y)
        return _f32(y)

    got, split, one = _route_outputs(run, monkeypatch)
    _assert_default_route(got, split, one, rank, m, kk, n)
