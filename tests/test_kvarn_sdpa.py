"""Fused KVarN decode attention vs the materialize reference.

sdpa_decode_gqa_kvarn stages the same fp16 values the kvarn_dequant kernel
produces (identical fp32 expression and rounding), and everything downstream
of the tile stage is the sdpa_decode_gqa machinery unchanged. The reference
therefore materializes the cache (sink + dequantized records + live) into a
plain fp16 KV and runs sdpa_decode_gqa with identical splits/tile_c: outputs
must match bit for bit, including lse.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="sdpa_decode_gqa_kvarn is a Metal-only kernel; no CPU path.",
)

D = 128
SCALE = D**-0.5


def quantize_head(x, bits, kind):
    """Quantize [B, H, T, d] into the flat slice-minor wire the fused
    kernel reads: codes [B, H, G, (d/128) * 512 * bits], axes
    [B, H, G, 3 * (d/128), 128] (one triplet per slice)."""
    b, h, t, d = x.shape
    sl = d // 128
    if sl == 1:
        return kq.kvarn_quantize(x, bits, kind)
    xs = x.reshape(b, h, t, sl, 128).transpose(0, 1, 3, 2, 4)
    c, a = kq.kvarn_quantize(xs, bits, kind)
    g = c.shape[3]
    c = c.transpose(0, 1, 3, 2, 4).reshape(b, h, g, sl * 512 * bits)
    a = a.transpose(0, 1, 3, 2, 4, 5).reshape(b, h, g, 3 * sl, 128)
    return c, a


def dequant_head(codes, axes, bits, kind, d, dtype):
    """Invert quantize_head's layout back to [B, H, G * 128, d]."""
    b, h, g = codes.shape[:3]
    sl = d // 128
    if sl == 1:
        return kq.kvarn_dequant(codes, axes, bits, kind, dtype=dtype)
    c = codes.reshape(b, h, g, sl, 512 * bits).transpose(0, 1, 3, 2, 4)
    a = axes.reshape(b, h, g, sl, 3, 128).transpose(0, 1, 3, 2, 4, 5)
    out = kq.kvarn_dequant(c, a, bits, kind, dtype=dtype)
    return out.transpose(0, 1, 3, 2, 4).reshape(b, h, g * 128, d)


def build_state(B, H, N, k_bits, v_bits, sink_cap=128, seed=0, d=D):
    """Synthesize a rotated-domain KVarN cache state plus its materialized
    fp16 twin. Keys [0, min(N, sink_cap)) live in stage sink rows, full
    128-groups after that are sealed records, the remainder is the live
    stage slot."""
    rng = np.random.default_rng(seed)
    sl = d // 128
    kx = rng.standard_normal((B, H, N, d)).astype(np.float16)
    vx = rng.standard_normal((B, H, N, d)).astype(np.float16)

    sink = min(N, sink_cap)
    n_recs = max(0, (N - sink_cap) // 128)
    live0 = sink + n_recs * 128
    s_rows = sink_cap + 128

    stage_k = np.zeros((B, H, s_rows, d), np.float16)
    stage_v = np.zeros((B, H, s_rows, d), np.float16)
    stage_k[:, :, :sink] = kx[:, :, :sink]
    stage_v[:, :, :sink] = vx[:, :, :sink]
    live_len = N - live0
    stage_k[:, :, sink_cap : sink_cap + live_len] = kx[:, :, live0:N]
    stage_v[:, :, sink_cap : sink_cap + live_len] = vx[:, :, live0:N]

    g_cap = max(n_recs, 1)
    codes_k = np.zeros((B, H, g_cap, sl * 512 * k_bits), np.uint32)
    codes_v = np.zeros((B, H, g_cap, sl * 512 * v_bits), np.uint32)
    axes_k = np.zeros((B, H, g_cap, 3 * sl, 128), np.float16)
    axes_v = np.zeros((B, H, g_cap, 3 * sl, 128), np.float16)

    if n_recs:
        ck, ak = quantize_head(mx.array(kx[:, :, sink:live0]), k_bits, "k")
        cv, av = quantize_head(mx.array(vx[:, :, sink:live0]), v_bits, "v")
        mx.eval(ck, ak, cv, av)
        codes_k[:, :, :n_recs] = np.array(ck)
        axes_k[:, :, :n_recs] = np.array(ak)
        codes_v[:, :, :n_recs] = np.array(cv)
        axes_v[:, :, :n_recs] = np.array(av)

    return {
        "codes_k": mx.array(codes_k),
        "axes_k": mx.array(axes_k),
        "codes_v": mx.array(codes_v),
        "axes_v": mx.array(axes_v),
        "stage_k": mx.array(stage_k),
        "stage_v": mx.array(stage_v),
        "kx": mx.array(kx),
        "vx": mx.array(vx),
        "sink": sink,
        "live0": live0,
        "n_recs": n_recs,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "d": d,
    }


def materialize(st, dtype):
    """Reference fp16/bf16 KV: raw sink and live rows plus records
    dequantized at the target dtype (matching the fused kernel's single
    fp32 -> T rounding)."""
    refs = []
    for side, raw in (("k", st["kx"]), ("v", st["vx"])):
        parts = [raw[:, :, : st["sink"]].astype(dtype)]
        if st["n_recs"]:
            n = st["n_recs"]
            parts.append(
                dequant_head(
                    st[f"codes_{side}"][:, :, :n],
                    st[f"axes_{side}"][:, :, :n],
                    st[f"{side}_bits"],
                    side,
                    st["d"],
                    dtype,
                )
            )
        parts.append(raw[:, :, st["live0"] :].astype(dtype))
        refs.append(mx.concatenate(parts, axis=2))
    return refs


def run_pair(st, q, N, k_bits, v_bits, dtype=mx.float16, **kw):
    qm = mx.array(q).astype(dtype)
    scale = st["d"] ** -0.5
    out = kq.sdpa_decode_gqa_kvarn(
        qm,
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
        N,
        scale,
        k_bits,
        v_bits,
        **kw,
    )
    k_ref, v_ref = materialize(st, dtype)
    ref = kq.sdpa_decode_gqa(qm, k_ref, v_ref, scale, **kw)
    return out, ref


def assert_bit_equal(a, b):
    if isinstance(a, tuple):
        for x, y in zip(a, b, strict=True):
            assert_bit_equal(x, y)
        return
    mx.eval(a, b)
    if a.dtype == mx.bfloat16:
        # numpy has no bfloat16; the fp32 embedding is exact.
        a, b = a.astype(mx.float32), b.astype(mx.float32)
    an, bn = np.array(a), np.array(b)
    assert an.dtype == bn.dtype and an.shape == bn.shape
    maxd = np.abs(an.astype(np.float64) - bn.astype(np.float64)).max()
    assert np.array_equal(an, bn), f"max|d|={maxd}"


def make_q(B, n_q_heads, qL, seed=1, d=D):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, n_q_heads, qL, d)).astype(np.float16)


@pytest.mark.parametrize("n", [100, 129, 256, 300, 1000, 4097])
def test_decode_matches_materialize_at_boundaries(n):
    st = build_state(1, 2, n, 6, 6, seed=n)
    q = make_q(1, 8, 1)
    out, ref = run_pair(st, q, n, 6, 6)
    assert_bit_equal(out, ref)


@pytest.mark.parametrize("kv_bits", [(2, 2), (3, 3), (4, 4), (5, 5), (8, 8), (6, 5)])
def test_decode_matches_materialize_across_widths(kv_bits):
    kb, vb = kv_bits
    st = build_state(1, 2, 300, kb, vb, seed=kb * 10 + vb)
    q = make_q(1, 8, 1)
    out, ref = run_pair(st, q, 300, kb, vb)
    assert_bit_equal(out, ref)


@pytest.mark.parametrize("ql", [2, 3, 4])
def test_verify_width_matches_materialize(ql):
    st = build_state(1, 2, 700, 6, 6, seed=ql)
    q = make_q(1, 8, ql)
    out, ref = run_pair(st, q, 700, 6, 6)
    assert_bit_equal(out, ref)


def test_starts_and_batch():
    st = build_state(3, 2, 300, 6, 6, seed=3)
    q = make_q(3, 8, 1)
    starts = mx.array([0, 150, 296], dtype=mx.int32)
    out, ref = run_pair(st, q, 300, 6, 6, starts=starts)
    assert_bit_equal(out, ref)


def test_sinks_ride_through():
    st = build_state(1, 2, 300, 6, 6, seed=4)
    q = make_q(1, 8, 1)
    sinks = mx.array(np.linspace(-1.0, 2.0, 8), dtype=mx.float32)
    out, ref = run_pair(st, q, 300, 6, 6, sinks=sinks)
    assert_bit_equal(out, ref)


def test_lse_matches():
    st = build_state(1, 2, 500, 6, 6, seed=5)
    q = make_q(1, 8, 1)
    out, ref = run_pair(st, q, 500, 6, 6, return_lse=True)
    assert_bit_equal(out, ref)


def test_bfloat16_query():
    st = build_state(1, 2, 300, 6, 6, seed=6)
    q = make_q(1, 8, 1)
    out, ref = run_pair(st, q, 300, 6, 6, dtype=mx.bfloat16)
    assert_bit_equal(out, ref)


def test_tile_c_16_and_explicit_splits():
    st = build_state(1, 2, 1000, 6, 6, seed=7)
    q = make_q(1, 8, 1)
    out, ref = run_pair(st, q, 1000, 6, 6, tile_c=16, splits=7)
    assert_bit_equal(out, ref)


def test_multi_group_sink():
    # quantized_kv_start extension: sink capacity of two groups.
    st = build_state(1, 2, 700, 6, 6, sink_cap=256, seed=8)
    q = make_q(1, 8, 1)
    out, ref = run_pair(st, q, 700, 6, 6)
    assert_bit_equal(out, ref)


def test_n_attend_matches_truncated_reference():
    # Body walk of a tail merge: first 800 of 1000 keys, region map still on
    # the full layout. The 800-key cut lands mid-record. At qL 1 the causal
    # clamp is inert, so full_visibility must not change the result.
    st = build_state(1, 2, 1000, 6, 6, seed=10)
    q = mx.array(make_q(1, 8, 1))
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    out = kq.sdpa_decode_gqa_kvarn(q, *args, 1000, SCALE, 6, 6, n_attend=800)
    out_fv = kq.sdpa_decode_gqa_kvarn(
        q, *args, 1000, SCALE, 6, 6, n_attend=800, full_visibility=True
    )
    k_ref, v_ref = materialize(st, mx.float16)
    ref = kq.sdpa_decode_gqa(q, k_ref[:, :, :800], v_ref[:, :, :800], SCALE)
    assert_bit_equal(out, ref)
    assert_bit_equal(out_fv, ref)


def lse_merge(ob, lb, ot, lt):
    """Numerically stable two-segment softmax merge in fp32."""
    ob, ot = ob.astype(mx.float32), ot.astype(mx.float32)
    m = mx.maximum(lb, lt)
    wb, wt = mx.exp(lb - m), mx.exp(lt - m)
    out = (ob * wb[..., None] + ot * wt[..., None]) / (wb + wt)[..., None]
    return out, m + mx.log(wb + wt)


@pytest.mark.parametrize("ql", [1, 2, 3, 4])
def test_tail_lse_merge_composition(ql):
    # Precision-tail shape: body = fused kvarn over keys [0, 200) with the
    # causal clamp lifted, tail = plain sdpa over raw rows [200, 260) where
    # offset causality is exact for trailing queries. The LSE merge must
    # reproduce a single attention over the concatenated key values. The
    # body cut keeps 72 of record 0's 128 keys, so a clamp that is not
    # lifted would hide body keys from early queries and fail loudly.
    N, A = 260, 200
    st = build_state(1, 2, N, 6, 6, seed=20 + ql)
    q = mx.array(make_q(1, 8, ql))
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    body, lse_b = kq.sdpa_decode_gqa_kvarn(
        q,
        *args,
        N,
        SCALE,
        6,
        6,
        n_attend=A,
        full_visibility=True,
        return_lse=True,
    )
    tail, lse_t = kq.sdpa_decode_gqa(
        q, st["kx"][:, :, A:], st["vx"][:, :, A:], SCALE, return_lse=True
    )
    merged, lse_m = lse_merge(body, lse_b, tail, lse_t)

    k_m, v_m = materialize(st, mx.float16)
    k_ref = mx.concatenate([k_m[:, :, :A], st["kx"][:, :, A:]], axis=2)
    v_ref = mx.concatenate([v_m[:, :, :A], st["vx"][:, :, A:]], axis=2)
    ref, lse_ref = kq.sdpa_decode_gqa(q, k_ref, v_ref, SCALE, return_lse=True)
    mx.eval(merged, lse_m, ref, lse_ref)

    # Not bit-exact: the merge re-rounds fp16 partial outputs and the split
    # structure differs from the single call.
    np.testing.assert_allclose(
        np.array(merged), np.array(ref.astype(mx.float32)), rtol=2e-3, atol=2e-3
    )
    np.testing.assert_allclose(np.array(lse_m), np.array(lse_ref), rtol=1e-5, atol=1e-4)


def test_rejects_malformed():
    st = build_state(1, 2, 300, 6, 6, seed=9)
    q = mx.array(make_q(1, 8, 1))
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_kvarn(q, *args, 300, SCALE, 7, 6)
    with pytest.raises(ValueError):
        # n beyond sink + record + live capacity
        kq.sdpa_decode_gqa_kvarn(q, *args, 500, SCALE, 6, 6)
    with pytest.raises(ValueError):
        # codes width inconsistent with bits
        kq.sdpa_decode_gqa_kvarn(q, *args, 300, SCALE, 8, 6)
    with pytest.raises(ValueError):
        # unsupported head dim
        kq.sdpa_decode_gqa_kvarn(q[..., :64], *args, 300, SCALE, 6, 6)
    with pytest.raises(ValueError):
        # n_attend beyond n
        kq.sdpa_decode_gqa_kvarn(q, *args, 300, SCALE, 6, 6, n_attend=400)
    with pytest.raises(ValueError):
        # full visibility over keys at or past a query's causal position
        q4 = mx.array(make_q(1, 8, 4))
        kq.sdpa_decode_gqa_kvarn(q4, *args, 300, SCALE, 6, 6, full_visibility=True)
    with pytest.raises(ValueError):
        # a truncated walk at verify width needs the clamp lifted
        q4 = mx.array(make_q(1, 8, 4))
        kq.sdpa_decode_gqa_kvarn(q4, *args, 300, SCALE, 6, 6, n_attend=200)


# -- head_dim 256/512 (multi-slice records) ----------------------------------

WIDE = pytest.mark.parametrize("d", [256, 512])


@WIDE
def test_dwide_slice_layout_contract(d):
    # The flat wire must be slice-minor: slice s of a group's codes at word
    # offset s * 512 * bits, axes triplet s at rows [3s, 3s + 3). Pinned
    # against independent per-slice quantization of the same tokens.
    rng = np.random.default_rng(30)
    x = mx.array(rng.standard_normal((1, 2, 256, d)).astype(np.float16))
    c, a = quantize_head(x, 6, "k")
    mx.eval(c, a)
    for s in range(d // 128):
        cs, as_ = kq.kvarn_quantize(x[..., s * 128 : (s + 1) * 128], 6, "k")
        mx.eval(cs, as_)
        assert np.array_equal(
            np.array(c[..., s * 512 * 6 : (s + 1) * 512 * 6]), np.array(cs)
        )
        assert np.array_equal(np.array(a[:, :, :, 3 * s : 3 * s + 3]), np.array(as_))


@WIDE
@pytest.mark.parametrize("n", [100, 129, 300, 1000, 4097])
def test_dwide_decode_matches_materialize_at_boundaries(d, n):
    st = build_state(1, 2, n, 6, 6, seed=40 + n, d=d)
    q = make_q(1, 8, 1, d=d)
    out, ref = run_pair(st, q, n, 6, 6)
    assert_bit_equal(out, ref)


@WIDE
@pytest.mark.parametrize("kv_bits", [(2, 2), (3, 3), (5, 4), (8, 8), (6, 5)])
def test_dwide_decode_matches_materialize_across_widths(d, kv_bits):
    kb, vb = kv_bits
    st = build_state(1, 2, 300, kb, vb, seed=50 + kb * 10 + vb, d=d)
    q = make_q(1, 8, 1, d=d)
    out, ref = run_pair(st, q, 300, kb, vb)
    assert_bit_equal(out, ref)


@WIDE
@pytest.mark.parametrize("ql", [2, 3, 4])
def test_dwide_verify_width_matches_materialize(d, ql):
    st = build_state(1, 2, 700, 6, 6, seed=60 + ql, d=d)
    q = make_q(1, 8, ql, d=d)
    out, ref = run_pair(st, q, 700, 6, 6)
    assert_bit_equal(out, ref)


@WIDE
def test_dwide_starts_and_batch(d):
    st = build_state(3, 2, 300, 6, 6, seed=61, d=d)
    q = make_q(3, 8, 1, d=d)
    starts = mx.array([0, 150, 296], dtype=mx.int32)
    out, ref = run_pair(st, q, 300, 6, 6, starts=starts)
    assert_bit_equal(out, ref)


@WIDE
def test_dwide_sinks_and_lse(d):
    st = build_state(1, 2, 500, 6, 6, seed=62, d=d)
    q = make_q(1, 8, 1, d=d)
    sinks = mx.array(np.linspace(-1.0, 2.0, 8), dtype=mx.float32)
    out, ref = run_pair(st, q, 500, 6, 6, sinks=sinks, return_lse=True)
    assert_bit_equal(out, ref)


@WIDE
def test_dwide_tile_c8_and_explicit_splits(d):
    st = build_state(1, 2, 1000, 6, 6, seed=63, d=d)
    q = make_q(1, 8, 1, d=d)
    out, ref = run_pair(st, q, 1000, 6, 6, tile_c=8, splits=7)
    assert_bit_equal(out, ref)


@WIDE
def test_dwide_multi_group_sink(d):
    st = build_state(1, 2, 700, 6, 6, sink_cap=256, seed=64, d=d)
    q = make_q(1, 8, 1, d=d)
    out, ref = run_pair(st, q, 700, 6, 6)
    assert_bit_equal(out, ref)


@WIDE
def test_dwide_bfloat16_query(d):
    st = build_state(1, 2, 300, 6, 6, seed=65, d=d)
    q = make_q(1, 8, 1, d=d)
    out, ref = run_pair(st, q, 300, 6, 6, dtype=mx.bfloat16)
    assert_bit_equal(out, ref)


@WIDE
def test_dwide_n_attend_matches_truncated_reference(d):
    st = build_state(1, 2, 1000, 6, 6, seed=66, d=d)
    q = mx.array(make_q(1, 8, 1, d=d))
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    scale = d**-0.5
    out = kq.sdpa_decode_gqa_kvarn(q, *args, 1000, scale, 6, 6, n_attend=800)
    k_ref, v_ref = materialize(st, mx.float16)
    ref = kq.sdpa_decode_gqa(q, k_ref[:, :, :800], v_ref[:, :, :800], scale)
    assert_bit_equal(out, ref)


@WIDE
@pytest.mark.parametrize("ql", [1, 4])
def test_dwide_tail_lse_merge_composition(d, ql):
    N, A = 260, 200
    st = build_state(1, 2, N, 6, 6, seed=70 + ql, d=d)
    q = mx.array(make_q(1, 8, ql, d=d))
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    scale = d**-0.5
    body, lse_b = kq.sdpa_decode_gqa_kvarn(
        q, *args, N, scale, 6, 6, n_attend=A, full_visibility=True, return_lse=True
    )
    tail, lse_t = kq.sdpa_decode_gqa(
        q, st["kx"][:, :, A:], st["vx"][:, :, A:], scale, return_lse=True
    )
    merged, lse_m = lse_merge(body, lse_b, tail, lse_t)

    k_m, v_m = materialize(st, mx.float16)
    k_ref = mx.concatenate([k_m[:, :, :A], st["kx"][:, :, A:]], axis=2)
    v_ref = mx.concatenate([v_m[:, :, :A], st["vx"][:, :, A:]], axis=2)
    ref, lse_ref = kq.sdpa_decode_gqa(q, k_ref, v_ref, scale, return_lse=True)
    mx.eval(merged, lse_m, ref, lse_ref)
    np.testing.assert_allclose(
        np.array(merged), np.array(ref.astype(mx.float32)), rtol=2e-3, atol=2e-3
    )
    np.testing.assert_allclose(np.array(lse_m), np.array(lse_ref), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("ql", [1, 3, 4])
def test_d512_gqa16_shipped_shape(ql):
    # gemma-4 global layers: 1 kv head, 16 q heads. gqa 16 at qL 3/4 sits on
    # the kernel's gqa_factor * ceil(qL/2) <= 32 ceiling (1024-thread TG).
    st = build_state(1, 1, 700, 6, 6, seed=80 + ql, d=512)
    q = make_q(1, 16, ql, d=512)
    out, ref = run_pair(st, q, 700, 6, 6)
    assert_bit_equal(out, ref)


@WIDE
def test_dwide_rejects_wrong_tile_c_and_stale_shapes(d):
    st = build_state(1, 2, 300, 6, 6, seed=67, d=d)
    q = mx.array(make_q(1, 8, 1, d=d))
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    scale = d**-0.5
    for bad_c in (32,) if d == 256 else (32, 16):
        with pytest.raises(ValueError):
            # no instantiation at this (D, tile_c)
            kq.sdpa_decode_gqa_kvarn(q, *args, 300, scale, 6, 6, tile_c=bad_c)
    with pytest.raises(ValueError):
        # single-slice codes with a wide query
        st1 = build_state(1, 2, 300, 6, 6, seed=68)
        kq.sdpa_decode_gqa_kvarn(
            q,
            st1["codes_k"],
            st1["axes_k"],
            st1["codes_v"],
            st1["axes_v"],
            st["stage_k"],
            st["stage_v"],
            300,
            scale,
            6,
            6,
        )


# -- FA verify (matrix-unit) route over kvarn records -------------------------
#
# sdpa_fa_verify_kvarn stages the same values as the decode kernel through
# the shared kvarn loaders, and everything downstream is sdpa_fa_verify
# unchanged, so it must match sdpa_fa_verify over the materialized twin bit
# for bit, lse included.


def fold(q, hkv, ql):
    """[1, Hq, qL, d] -> the kv-major GQA fold [1, Hkv, G * qL, d]."""
    q = mx.array(q)
    _, hq, _, d = q.shape
    return q.reshape(1, hkv, (hq // hkv) * ql, d)


def run_fa_pair(st, q, N, k_bits, v_bits, ql, dtype=mx.float16, **kw):
    hkv = st["codes_k"].shape[1]
    qf = fold(q, hkv, ql).astype(dtype)
    scale = st["d"] ** -0.5
    out = kq.sdpa_fa_verify_kvarn(
        qf,
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
        N,
        scale,
        k_bits,
        v_bits,
        ql,
        **kw,
    )
    k_ref, v_ref = materialize(st, dtype)
    ref_kw = {k: v for k, v in kw.items() if k in ("splits", "return_lse")}
    ref = kq.sdpa_fa_verify(qf, k_ref, v_ref, scale, ql, **ref_kw)
    return out, ref


@pytest.mark.parametrize("ql", [1, 2, 3, 4, 5, 6, 8])
def test_fa_verify_matches_materialize(ql):
    st = build_state(1, 2, 700, 6, 6, seed=100 + ql)
    q = make_q(1, 8, ql)
    out, ref = run_fa_pair(st, q, 700, 6, 6, ql)
    assert_bit_equal(out, ref)


@pytest.mark.parametrize("n", [100, 129, 256, 300, 1000, 4097])
def test_fa_verify_matches_materialize_at_boundaries(n):
    st = build_state(1, 2, n, 6, 6, seed=n + 1)
    q = make_q(1, 8, 4)
    out, ref = run_fa_pair(st, q, n, 6, 6, 4, return_lse=True)
    assert_bit_equal(out, ref)


@pytest.mark.parametrize("kv_bits", [(2, 2), (3, 3), (4, 4), (5, 5), (8, 8), (6, 5)])
def test_fa_verify_matches_materialize_across_widths(kv_bits):
    kb, vb = kv_bits
    st = build_state(1, 2, 300, kb, vb, seed=kb * 10 + vb + 1)
    q = make_q(1, 8, 4)
    out, ref = run_fa_pair(st, q, 300, kb, vb, 4)
    assert_bit_equal(out, ref)


@pytest.mark.parametrize("g,ql", [(16, 4), (8, 8), (12, 5)])
def test_fa_verify_wide_folds(g, ql):
    # 64-row tile: gqa16 x qL4 (qwen3.8-27b), gqa8 x qL8, and a 60-row
    # padded fold.
    st = build_state(1, 2, 900, 6, 6, seed=110 + g)
    q = make_q(1, 2 * g, ql)
    out, ref = run_fa_pair(st, q, 900, 6, 6, ql)
    assert_bit_equal(out, ref)


def test_fa_verify_bfloat16_query():
    st = build_state(1, 2, 300, 6, 6, seed=120)
    q = make_q(1, 8, 4)
    out, ref = run_fa_pair(st, q, 300, 6, 6, 4, dtype=mx.bfloat16)
    assert_bit_equal(out, ref)


def test_fa_verify_explicit_splits():
    st = build_state(1, 2, 1000, 6, 6, seed=121)
    q = make_q(1, 8, 4)
    out, ref = run_fa_pair(st, q, 1000, 6, 6, 4, splits=7, return_lse=True)
    assert_bit_equal(out, ref)


def test_fa_verify_n_attend_full_visibility():
    # Body walk of a tail merge at verify width: the first 800 of 1000 keys
    # with the clamp lifted equals a q_len-1 fold (every row sees every
    # key) over the truncated twin.
    st = build_state(1, 2, 1000, 6, 6, seed=122)
    q = mx.array(make_q(1, 8, 4))
    qf = fold(q, 2, 4)
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    out, lse = kq.sdpa_fa_verify_kvarn(
        qf,
        *args,
        1000,
        SCALE,
        6,
        6,
        4,
        n_attend=800,
        full_visibility=True,
        return_lse=True,
    )
    k_ref, v_ref = materialize(st, mx.float16)
    ref, lse_ref = kq.sdpa_fa_verify(
        qf, k_ref[:, :, :800], v_ref[:, :, :800], SCALE, 1, return_lse=True
    )
    assert_bit_equal((out, lse), (ref, lse_ref))


@pytest.mark.parametrize("ql", [2, 4, 8])
def test_fa_verify_tail_lse_merge_composition(ql):
    # The gmlx verify-width shape: body = FA kvarn over keys [0, 200) with
    # the clamp lifted, tail = sdpa_fa_verify over raw rows [200, 260) with
    # exact offset causality; the LSE merge reproduces one attention over
    # the concatenated keys.
    N, A = 260, 200
    st = build_state(1, 2, N, 6, 6, seed=130 + ql)
    q = mx.array(make_q(1, 8, ql))
    qf = fold(q, 2, ql)
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    body, lse_b = kq.sdpa_fa_verify_kvarn(
        q=qf,
        codes_k=args[0],
        axes_k=args[1],
        codes_v=args[2],
        axes_v=args[3],
        stage_k=args[4],
        stage_v=args[5],
        n=N,
        scale=SCALE,
        k_bits=6,
        v_bits=6,
        q_len=ql,
        n_attend=A,
        full_visibility=True,
        return_lse=True,
    )
    tail, lse_t = kq.sdpa_fa_verify(
        qf, st["kx"][:, :, A:], st["vx"][:, :, A:], SCALE, ql, return_lse=True
    )
    merged, lse_m = lse_merge(body, lse_b, tail, lse_t)

    k_m, v_m = materialize(st, mx.float16)
    k_ref = mx.concatenate([k_m[:, :, :A], st["kx"][:, :, A:]], axis=2)
    v_ref = mx.concatenate([v_m[:, :, :A], st["vx"][:, :, A:]], axis=2)
    ref, lse_ref = kq.sdpa_fa_verify(qf, k_ref, v_ref, SCALE, ql, return_lse=True)
    mx.eval(merged, lse_m, ref, lse_ref)
    np.testing.assert_allclose(
        np.array(merged), np.array(ref.astype(mx.float32)), rtol=2e-3, atol=2e-3
    )
    np.testing.assert_allclose(np.array(lse_m), np.array(lse_ref), rtol=1e-5, atol=1e-4)


def test_fa_verify_rejects_malformed():
    st = build_state(1, 2, 300, 6, 6, seed=140)
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    ok = fold(make_q(1, 8, 4), 2, 4)
    with pytest.raises(ValueError, match="GQA-folded"):
        kq.sdpa_fa_verify_kvarn(mx.array(make_q(1, 8, 4)), *args, 300, SCALE, 6, 6, 4)
    with pytest.raises(ValueError, match="folded rows"):
        kq.sdpa_fa_verify_kvarn(
            fold(make_q(1, 10, 1), 2, 1), *args, 300, SCALE, 6, 6, 4
        )
    with pytest.raises(ValueError, match="folded rows"):
        kq.sdpa_fa_verify_kvarn(
            fold(make_q(1, 34, 4), 2, 4), *args, 300, SCALE, 6, 6, 4
        )
    with pytest.raises(ValueError, match="q_len must be"):
        kq.sdpa_fa_verify_kvarn(ok, *args, 300, SCALE, 6, 6, 9)
    with pytest.raises(ValueError, match="requires full_visibility"):
        kq.sdpa_fa_verify_kvarn(ok, *args, 300, SCALE, 6, 6, 4, n_attend=200)
    with pytest.raises(ValueError, match="causal position"):
        kq.sdpa_fa_verify_kvarn(
            ok, *args, 300, SCALE, 6, 6, 4, n_attend=298, full_visibility=True
        )
    with pytest.raises(ValueError, match="batch size"):
        st2 = build_state(2, 2, 300, 6, 6, seed=141)
        kq.sdpa_fa_verify_kvarn(
            mx.array(make_q(2, 2, 32)),
            st2["codes_k"],
            st2["axes_k"],
            st2["codes_v"],
            st2["axes_v"],
            st2["stage_k"],
            st2["stage_v"],
            300,
            SCALE,
            6,
            6,
            4,
        )


@WIDE
@pytest.mark.parametrize("ql", [1, 4, 8])
def test_dwide_fa_verify_matches_materialize(d, ql):
    st = build_state(1, 2, 700, 6, 6, seed=150 + ql, d=d)
    q = make_q(1, 8, ql, d=d)
    out, ref = run_fa_pair(st, q, 700, 6, 6, ql, return_lse=True)
    assert_bit_equal(out, ref)


@WIDE
@pytest.mark.parametrize("kv_bits", [(3, 3), (5, 4), (6, 5), (8, 8)])
def test_dwide_fa_verify_across_widths(d, kv_bits):
    kb, vb = kv_bits
    st = build_state(1, 2, 300, kb, vb, seed=160 + kb, d=d)
    q = make_q(1, 8, 4, d=d)
    out, ref = run_fa_pair(st, q, 300, kb, vb, 4)
    assert_bit_equal(out, ref)


@WIDE
def test_dwide_fa_verify_n_attend_and_boundaries(d):
    for n in (129, 1000, 4097):
        st = build_state(1, 2, n, 6, 6, seed=170 + n, d=d)
        q = mx.array(make_q(1, 8, 4, d=d))
        qf = fold(q, 2, 4)
        scale = d**-0.5
        a = n - 3
        out = kq.sdpa_fa_verify_kvarn(
            qf,
            st["codes_k"],
            st["axes_k"],
            st["codes_v"],
            st["axes_v"],
            st["stage_k"],
            st["stage_v"],
            n,
            scale,
            6,
            6,
            4,
            n_attend=a,
            full_visibility=True,
        )
        k_ref, v_ref = materialize(st, mx.float16)
        ref = kq.sdpa_fa_verify(qf, k_ref[:, :, :a], v_ref[:, :, :a], scale, 1)
        assert_bit_equal(out, ref)


def test_d512_fa_verify_row_cap_and_gemma_fold():
    # gemma-4 global layers fold 16 q heads on 1 kv head: 64 rows at qL 4
    # exceeds the d-split kernel's 32-row tile, so the caller chunks the
    # group; a G8 x qL4 chunk runs whole.
    st = build_state(1, 1, 700, 6, 6, seed=180, d=512)
    args = (
        st["codes_k"],
        st["axes_k"],
        st["codes_v"],
        st["axes_v"],
        st["stage_k"],
        st["stage_v"],
    )
    scale = 512**-0.5
    with pytest.raises(ValueError, match="folded rows"):
        kq.sdpa_fa_verify_kvarn(
            fold(make_q(1, 16, 4, d=512), 1, 4), *args, 700, scale, 6, 6, 4
        )
    q = make_q(1, 8, 4, d=512)
    out, ref = run_fa_pair(st, q, 700, 6, 6, 4)
    assert_bit_equal(out, ref)
