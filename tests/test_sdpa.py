#!/usr/bin/env python3
"""Vector-SDPA validation: kq.sdpa_vector vs an INDEPENDENT f32 materialized
attention reference (matmul -> masked softmax -> matmul, accumulated in float32).

The reference shares no code with the kernel, so a bug in the two-pass online
softmax cannot cancel out of both sides. Covers the head dims stock MLX's fused
vector path excludes (256, 512), both float dtypes, GQA, the decode (qL=1) and
speculative-verify (qL>1, offset-causal) widths, and a strided KV-cache prefix
(head stride > kL*D, the RotatingKVCache layout) which the op must read in place
without a copy.

The kernel is Metal-only (eval_cpu throws), so the module is skipped under
KQUANT_FORCE_CPU.

Usage:
    test_sdpa.py [--d 512] [--ql 1,5] [--kl 2048,8192]
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq.sdpa_vector is a Metal-only kernel; no CPU path.",
)

# rel-norm bound per dtype: bf16 has ~8 mantissa bits, fp16 ~11.
REL_BOUND = {mx.bfloat16: 5e-3, mx.float16: 2e-3}


def _ref_sdpa(q, k, v, scale, causal):
    """f32 materialized attention. Offset-causal: query row i (of qL) attends
    keys <= kL - qL + i, matching the kernel's mask."""
    g = q.shape[1] // k.shape[1]
    kr = mx.repeat(k, g, axis=1).astype(mx.float32)
    vr = mx.repeat(v, g, axis=1).astype(mx.float32)
    s = (q.astype(mx.float32) @ kr.swapaxes(-1, -2)) * scale  # [B,Hq,qL,kL]
    if causal:
        qL, kL = q.shape[2], k.shape[2]
        rows = mx.arange(kL - qL, kL).reshape(qL, 1)
        cols = mx.arange(kL).reshape(1, kL)
        s = mx.where(cols <= rows, s, float("-inf"))
    w = mx.softmax(s, axis=-1)
    return (w @ vr).astype(q.dtype)


def _rel(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    return float(mx.linalg.norm(af - bf) / (mx.linalg.norm(bf) + 1e-9))


def _make(B, Hq, Hkv, qL, kL, D, dtype, seed, strided):
    key = mx.random.key(seed)
    k0, k1, k2 = mx.random.split(key, 3)
    q = mx.random.normal((B, Hq, qL, D), key=k0).astype(dtype)
    if strided:
        # Allocate a longer seq dim and slice so the head stride is maxL*D (>
        # kL*D) and the seq stride stays D -- the RotatingKVCache prefix layout.
        maxL = kL + 256
        kf = mx.random.normal((B, Hkv, maxL, D), key=k1).astype(dtype)
        vf = mx.random.normal((B, Hkv, maxL, D), key=k2).astype(dtype)
        mx.eval(kf, vf)
        k, v = kf[:, :, :kL, :], vf[:, :, :kL, :]
    else:
        k = mx.random.normal((B, Hkv, kL, D), key=k1).astype(dtype)
        v = mx.random.normal((B, Hkv, kL, D), key=k2).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _eval_or_skip(*arrays):
    # Materialize op + reference; a device whose pipeline caps the dispatch
    # width raises the informative eval_gpu guard error -> capability skip,
    # never a silent-garbage numerics failure.
    try:
        mx.eval(*arrays)
    except RuntimeError as e:
        if "pipeline limit" in str(e):
            pytest.skip(str(e))
        raise


def _check(D, qL, kL, dtype, Hq=32, Hkv=16, strided=False):
    causal = qL > 1  # qL==1 attends all keys; offset-causal is the verify regime
    scale = 1.0 / (D**0.5)
    q, k, v = _make(1, Hq, Hkv, qL, kL, D, dtype, seed=qL * 7 + kL + D, strided=strided)
    got = kq.sdpa_vector(q, k, v, scale, causal=causal)
    ref = _ref_sdpa(q, k, v, scale, causal)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = REL_BOUND[dtype]
    tag = "strided" if strided else "contig"
    print(
        f"  [sdpa] D={D} qL={qL} kL={kL} {str(dtype)[9:]:>9} {tag}: "
        f"rel={rel:.3e} (bound {bound:.0e})"
    )
    assert rel < bound, f"D={D} qL={qL} kL={kL} {dtype} rel {rel:.3e} >= {bound:.0e}"
    assert got.shape == q.shape


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [256, 512])
@pytest.mark.parametrize("qL", [1, 2, 5])
def test_sdpa_vector(D, qL, dtype):
    _check(D, qL, kL=8192, dtype=dtype)


@pytest.mark.parametrize("D", [256, 512])
def test_sdpa_vector_strided_kv(D):
    # The no-copy strided-prefix path at the verify width (offset-causal).
    _check(D, qL=5, kL=4096, dtype=mx.bfloat16, strided=True)


@pytest.mark.parametrize("Hq,Hkv", [(32, 32), (32, 8)])
def test_sdpa_vector_gqa(Hq, Hkv):
    _check(512, qL=4, kL=2048, dtype=mx.bfloat16, Hq=Hq, Hkv=Hkv)


@pytest.mark.parametrize("D", [256, 512])
def test_sdpa_vector_f32_partials_outlier_v(D):
    # Un-normalized pass-1 partials scale with keys-per-block times |v|.
    # A float16 partial store overflows 65504 under flat attention with
    # outlier V channels; f32 partials must stay finite and rounding-level.
    scale = 1.0 / (D**0.5)
    q, k, v = _make(1, 8, 2, 1, 16384, D, mx.float16, seed=3, strided=False)
    k = (0.05 * k.astype(mx.float32)).astype(mx.float16)  # flatten attention
    v[:, :, :, ::64] = 2048.0
    mx.eval(k, v)
    got = kq.sdpa_vector(q, k, v, scale, causal=False)
    ref = _ref_sdpa(q, k, v, scale, causal=False)
    _eval_or_skip(got, ref)
    assert bool(mx.all(mx.isfinite(got.astype(mx.float32))).item())
    rel = _rel(got, ref)
    assert rel < REL_BOUND[mx.float16], f"D={D} rel {rel:.3e}"


@pytest.mark.parametrize("D", [256, 512])
def test_sdpa_vector_bf16_partials_outlier_v(D):
    # bfloat16 keeps 16-bit pass-1 partials (range covers the outlier
    # magnitudes that overflow fp16); same stress must stay finite and
    # within the bf16 rounding bound.
    scale = 1.0 / (D**0.5)
    q, k, v = _make(1, 8, 2, 1, 16384, D, mx.bfloat16, seed=3, strided=False)
    k = (0.05 * k.astype(mx.float32)).astype(mx.bfloat16)
    v[:, :, :, ::64] = 2048.0
    mx.eval(k, v)
    got = kq.sdpa_vector(q, k, v, scale, causal=False)
    ref = _ref_sdpa(q, k, v, scale, causal=False)
    _eval_or_skip(got, ref)
    assert bool(mx.all(mx.isfinite(got.astype(mx.float32))).item())
    rel = _rel(got, ref)
    assert rel < REL_BOUND[mx.bfloat16], f"D={D} rel {rel:.3e}"


def _ref_sdpa_sinks(q, k, v, scale, sinks):
    """f32 reference with per-q-head sink logits: an extra softmax column
    with no value row (raises the max / adds to the denominator only).
    qL > 1 is offset-causal (query row i attends keys <= kL - qL + i)."""
    g = q.shape[1] // k.shape[1]
    kr = mx.repeat(k, g, axis=1).astype(mx.float32)
    vr = mx.repeat(v, g, axis=1).astype(mx.float32)
    s = (q.astype(mx.float32) @ kr.swapaxes(-1, -2)) * scale  # [B,Hq,qL,kL]
    qL, kL = q.shape[2], k.shape[2]
    if qL > 1:
        rows = mx.arange(kL - qL, kL).reshape(qL, 1)
        cols = mx.arange(kL).reshape(1, kL)
        s = mx.where(cols <= rows, s, float("-inf"))
    if sinks is not None:
        col = mx.broadcast_to(
            sinks.astype(mx.float32).reshape(1, -1, 1, 1),
            (*s.shape[:3], 1),
        )
        s = mx.concatenate([s, col], axis=-1)
    w = mx.softmax(s, axis=-1)
    if sinks is not None:
        w = w[..., :-1]
    return (w @ vr).astype(q.dtype)


def _check_gqa(
    D,
    kL,
    dtype,
    Hq=24,
    Hkv=4,
    tile_c=0,
    sinks=False,
    strided=False,
    splits=0,
    qL=1,
):
    scale = 1.0 / (D**0.5)
    q, k, v = _make(1, Hq, Hkv, qL, kL, D, dtype, seed=kL + D, strided=strided)
    sk = None
    if sinks:
        sk = mx.random.normal((Hq,), key=mx.random.key(D + 1)).astype(mx.float32)
        mx.eval(sk)
    got = kq.sdpa_decode_gqa(q, k, v, scale, sinks=sk, splits=splits, tile_c=tile_c)
    ref = _ref_sdpa_sinks(q, k, v, scale, sk)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = REL_BOUND[dtype]
    print(
        f"  [gqa] D={D} qL={qL} kL={kL} Hq/Hkv={Hq}/{Hkv} c={tile_c} "
        f"sinks={sinks} {str(dtype)[9:]:>9}: rel={rel:.3e}"
    )
    assert rel < bound, f"D={D} kL={kL} rel {rel:.3e} >= {bound:.0e}"
    assert got.shape == q.shape


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "D,tile_c",
    [(64, 32), (64, 16), (128, 32), (128, 16), (256, 16), (256, 8), (512, 8)],
)
def test_sdpa_decode_gqa(D, tile_c, dtype):
    _check_gqa(D, kL=4096, dtype=dtype, tile_c=tile_c)


@pytest.mark.parametrize("Hq,Hkv", [(24, 4), (16, 4), (32, 8), (8, 8)])
def test_sdpa_decode_gqa_factors(Hq, Hkv):
    _check_gqa(256, kL=2048, dtype=mx.bfloat16, Hq=Hq, Hkv=Hkv)


@pytest.mark.parametrize("Hq,Hkv,D", [(16, 1, 512), (32, 4, 512)])
def test_sdpa_decode_gqa_wide_factor(Hq, Hkv, D):
    # gemma-4 12b/31b global-layer geometry (gqa 16 / 8 at hd512)
    _check_gqa(D, kL=2048, dtype=mx.bfloat16, Hq=Hq, Hkv=Hkv)


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_decode_gqa_sinks(D):
    _check_gqa(D, kL=2048, dtype=mx.bfloat16, sinks=True)


@pytest.mark.parametrize("D", [64, 256, 512])
def test_sdpa_decode_gqa_strided_unaligned(D):
    # strided KV-cache prefix + a key length off every tile/split boundary
    _check_gqa(D, kL=3071, dtype=mx.bfloat16, strided=True, splits=16)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [64, 128, 256, 512])
@pytest.mark.parametrize("qL", [2, 3, 4])
def test_sdpa_gqa_verify(D, qL, dtype):
    # speculative-verify width: offset-causal queries share the staged tiles
    _check_gqa(D, kL=4096, dtype=dtype, qL=qL)


def test_sdpa_gqa_verify_gemma_geometry():
    # gemma-4-31b global layers at verify width (gqa 8 at hd512)
    _check_gqa(512, kL=8192, dtype=mx.bfloat16, Hq=32, Hkv=4, qL=4)


def test_sdpa_gqa_verify_sinks():
    _check_gqa(64, kL=2048, dtype=mx.bfloat16, sinks=True, qL=4)


@pytest.mark.parametrize("D", [64, 512])
def test_sdpa_gqa_verify_strided_unaligned(D):
    _check_gqa(D, kL=3071, dtype=mx.bfloat16, strided=True, splits=16, qL=4)


def test_sdpa_gqa_verify_short_kv():
    # kL small enough that most splits stage zero keys and whole tiles fall
    # beyond a query's causal limit: exercises the empty-split partials and
    # the fully-invalid-tile guard (finite_min max would otherwise poison
    # the sum with exp(0) terms).
    _check_gqa(64, kL=17, dtype=mx.bfloat16, splits=16, qL=4)


def _ref_sdpa_starts(q, k, v, scale, pads, qL):
    # per-row f32 reference on the visible tail [pads[b], kL)
    outs = []
    for b in range(q.shape[0]):
        p = int(pads[b])
        outs.append(
            _ref_sdpa(
                q[b : b + 1],
                k[b : b + 1, :, p:, :],
                v[b : b + 1, :, p:, :],
                scale,
                causal=qL > 1,
            )
        )
    return mx.concatenate(outs, axis=0)


def _check_gqa_starts(
    D,
    kL,
    dtype,
    B=4,
    Hq=24,
    Hkv=4,
    qL=1,
    pads=None,
    strided=False,
    splits=0,
):
    scale = 1.0 / (D**0.5)
    q, k, v = _make(B, Hq, Hkv, qL, kL, D, dtype, seed=kL + D + B, strided=strided)
    if pads is None:
        pads = [(b * (kL - qL)) // B for b in range(B)]
    starts = mx.array(pads, dtype=mx.int32)
    mx.eval(starts)
    got = kq.sdpa_decode_gqa(q, k, v, scale, splits=splits, starts=starts)
    ref = _ref_sdpa_starts(q, k, v, scale, pads, qL)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = REL_BOUND[dtype]
    print(
        f"  [gqa-starts] D={D} qL={qL} kL={kL} B={B} Hq/Hkv={Hq}/{Hkv} "
        f"pads={pads} {str(dtype)[9:]:>9}: rel={rel:.3e}"
    )
    assert rel < bound, f"D={D} kL={kL} pads={pads} rel {rel:.3e} >= {bound:.0e}"
    assert got.shape == q.shape


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_decode_gqa_starts(D, dtype):
    # left-padded batched rows: row b attends keys [pads[b], kL)
    _check_gqa_starts(D, kL=4096, dtype=dtype)


@pytest.mark.parametrize("D", [64, 512])
def test_sdpa_decode_gqa_batched_nostarts(D):
    # B > 1 without starts: the plain batched grid the ragged route
    # degenerates to at pad-0 (and a pin for batched use on its own)
    _check_gqa(D, kL=4096, dtype=mx.bfloat16)
    scale = 1.0 / (D**0.5)
    q, k, v = _make(8, 24, 4, 1, 4096, D, mx.bfloat16, seed=D, strided=False)
    got = kq.sdpa_decode_gqa(q, k, v, scale)
    ref = _ref_sdpa_sinks(q, k, v, scale, None)
    _eval_or_skip(got, ref)
    assert _rel(got, ref) < REL_BOUND[mx.bfloat16]


def test_sdpa_decode_gqa_starts_zero_matches_plain():
    # all-zero starts must match the no-starts call on the same inputs
    scale = 1.0 / (512**0.5)
    q, k, v = _make(4, 24, 4, 1, 2048, 512, mx.bfloat16, seed=3, strided=False)
    starts = mx.zeros((4,), dtype=mx.int32)
    a = kq.sdpa_decode_gqa(q, k, v, scale, starts=starts)
    b = kq.sdpa_decode_gqa(q, k, v, scale)
    _eval_or_skip(a, b)
    assert _rel(a, b) < 1e-6


@pytest.mark.parametrize("qL", [2, 4])
def test_sdpa_decode_gqa_starts_verify(qL):
    # verify width on left-padded rows: the block occupies the last qL
    # positions of every row regardless of its pad (end-aligned causal)
    _check_gqa_starts(512, kL=4096, dtype=mx.bfloat16, qL=qL)


def test_sdpa_decode_gqa_starts_edges():
    # pad 0, a pad on a tile boundary, a pad mid-tile, and the maximum
    # in-contract pad (one visible key at qL=1: output equals that value row)
    _check_gqa_starts(
        512,
        kL=3071,
        dtype=mx.bfloat16,
        B=4,
        pads=[0, 1024, 1543, 3070],
        strided=True,
        splits=16,
    )


@pytest.mark.parametrize("Hq,Hkv", [(32, 4), (16, 1)])
def test_sdpa_decode_gqa_starts_gemma_geometry(Hq, Hkv):
    # gemma-4 31b (gqa 8) and 12b (gqa 16, single kv head) global layers
    _check_gqa_starts(512, kL=8192, dtype=mx.bfloat16, B=8, Hq=Hq, Hkv=Hkv)


def test_sdpa_decode_gqa_starts_validation():
    q, k, v = _make(4, 24, 4, 1, 512, 64, mx.bfloat16, seed=5, strided=False)
    with pytest.raises(ValueError, match="one element per batch row"):
        kq.sdpa_decode_gqa(q, k, v, 0.125, starts=mx.zeros((3,), mx.int32))
    with pytest.raises(ValueError, match="int32"):
        kq.sdpa_decode_gqa(q, k, v, 0.125, starts=mx.zeros((4,), mx.int64))


def _ref_sdpa_fold(q, k, v, scale, q_len):
    """f32 reference for the GQA-folded verify layout: q [B, Hkv, G*qL, D]
    attends its own kv head directly; folded row r is causally clamped to
    key <= kL - qL + (r % qL)."""
    s = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * scale
    n_rows, kL = q.shape[2], k.shape[2]
    lims = (kL - q_len + mx.arange(n_rows) % q_len).reshape(n_rows, 1)
    cols = mx.arange(kL).reshape(1, kL)
    s = mx.where(cols <= lims, s, float("-inf"))
    w = mx.softmax(s, axis=-1)
    return (w @ v.astype(mx.float32)).astype(q.dtype)


def _check_fa(D, qL, kL, dtype, Hkv=4, G=6, strided=False, splits=0):
    n_rows = G * qL
    scale = 1.0 / (D**0.5)
    q, k, v = _make(
        1, Hkv, Hkv, n_rows, kL, D, dtype, seed=qL * 13 + kL + D, strided=strided
    )
    got = kq.sdpa_fa_verify(q, k, v, scale, q_len=qL, splits=splits)
    ref = _ref_sdpa_fold(q, k, v, scale, qL)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = REL_BOUND[dtype]
    tag = "strided" if strided else "contig"
    print(
        f"  [fa] D={D} qL={qL} G={G} kL={kL} Hkv={Hkv} {str(dtype)[9:]:>9} "
        f"{tag}: rel={rel:.3e}"
    )
    assert rel < bound, f"D={D} qL={qL} G={G} kL={kL} rel {rel:.3e} >= {bound:.0e}"
    assert got.shape == q.shape


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [64, 128, 256, 512])
@pytest.mark.parametrize("qL", [2, 3, 4, 5, 6])
def test_sdpa_fa_verify(D, qL, dtype):
    _check_fa(D, qL, kL=4096, dtype=dtype, G=4)


def test_sdpa_fa_verify_qwen_geometry():
    # qwen3.5/3.6 full-attn fold: 24 rows = G6 x qL4 at hd256
    _check_fa(256, 4, kL=8192, dtype=mx.bfloat16, Hkv=4, G=6)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_fa_verify_decode(D, dtype):
    # q_len == 1: plain GQA decode on the matrix units (every folded row
    # attends the full KV). 122b shape: G16 x qL1 at hd256, 2 kv heads;
    # same fold on the hd512 d-split kernel.
    _check_fa(D, 1, kL=8192, dtype=dtype, Hkv=2, G=16)


def test_sdpa_fa_verify_gemma_geometry():
    # gemma-4-31b global layers: 32 rows = G8 x qL4 at hd512 (d-split kernel)
    _check_fa(512, 4, kL=8192, dtype=mx.bfloat16, Hkv=4, G=8)


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_fa_verify_full_tile(D):
    # n_rows == 32 fills the tile exactly (no padding rows), qL at the cap
    _check_fa(D, 8, kL=4096, dtype=mx.bfloat16, Hkv=2, G=4)


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_fa_verify_partial_warp(D):
    # n_rows == 18: the third row strip covers rows 16..17 plus padding
    _check_fa(D, 6, kL=2048, dtype=mx.bfloat16, G=3)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_fa_verify_strided_unaligned(D, dtype):
    # strided KV-cache prefix + a key length off every tile/split boundary
    _check_fa(D, 4, kL=3071, dtype=dtype, strided=True, splits=16)


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_fa_verify_short_kv(D):
    # kL small enough that most splits stage zero keys: their empty partials
    # (max = finite_min, sum = 0) must merge as weight zero
    _check_fa(D, 4, kL=17, dtype=mx.bfloat16, splits=16, G=6)


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_fa_verify_min_kv(D):
    # kL == qL floor: row 0 attends exactly one key, every row masked hard
    _check_fa(D, 4, kL=4, dtype=mx.bfloat16, G=6)


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_fa_verify_causal_split_straddle(D):
    # the last qL keys straddle a split boundary (splits=128, kL=4098 puts
    # keys 4096..4097 alone in the final split): that split is entirely past
    # the low rows' causal limits, exercising the dead-row guard in a
    # non-empty split
    _check_fa(D, 4, kL=4098, dtype=mx.bfloat16, splits=128, G=6)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_sdpa_fa_verify_bq64_full_tile(dtype):
    # gqa16 x qL4 = 64 rows (qwen3.5-122b-a10b verify fold): fills the BQ=64
    # tile exactly
    _check_fa(256, 4, kL=4096, dtype=dtype, Hkv=2, G=16)


@pytest.mark.parametrize("G,qL", [(8, 5), (12, 4), (10, 6)])
def test_sdpa_fa_verify_bq64_padded(G, qL):
    # 33..63 rows: BQ=64 with padding rows in the upper simdgroups
    _check_fa(256, qL, kL=2048, dtype=mx.bfloat16, Hkv=2, G=G)


def test_sdpa_fa_verify_bq64_strided_kv():
    _check_fa(256, 4, kL=3071, dtype=mx.bfloat16, Hkv=2, G=16, strided=True, splits=16)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("G", [32, 64])
def test_sdpa_fa_verify_cascade_fold(D, G):
    # shared-prefix cascade: B batch rows folded kv-head-major into the row
    # axis at q_len 1 (every row attends the full prefix); G=64 fills the
    # BQ=64 tile at the register-resident head dims
    _check_fa(D, 1, kL=6144, dtype=mx.bfloat16, Hkv=2, G=G)


def test_sdpa_fa_verify_lazy_strided_q():
    # Regression: an UNEVALUATED strided q view (here a chunk of a folded
    # tile) must not be trusted as row-contiguous at op-build time; the op
    # wraps q in mx.contiguous unconditionally, which resolves at eval.
    D, qL, kL = 256, 4, 4096
    scale = 1.0 / (D**0.5)
    q, k, v = _make(1, 2, 2, 32, kL, D, mx.bfloat16, seed=7, strided=False)
    whole = kq.sdpa_fa_verify(q, k, v, scale, q_len=qL)
    qc = q.reshape(1, 2, 2, 16, D)
    lazy_chunks = [qc[:, :, i] for i in range(2)]  # NOT evaluated
    got = mx.concatenate(
        [kq.sdpa_fa_verify(c, k, v, scale, q_len=qL) for c in lazy_chunks],
        axis=2,
    )
    _eval_or_skip(whole, got)
    rel = _rel(got, whole)
    print(f"  [fa] lazy strided q chunks: rel={rel:.3e}")
    assert rel < 1e-6, f"lazy strided q gave rel {rel:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--ql", default="1,2,5")
    ap.add_argument("--kl", default="2048,8192")
    args = ap.parse_args()
    for dtype in (mx.bfloat16, mx.float16):
        for qL in (int(x) for x in args.ql.split(",")):
            for kL in (int(x) for x in args.kl.split(",")):
                _check(args.d, qL, kL, dtype)
    _check(args.d, qL=5, kL=4096, dtype=mx.bfloat16, strided=True)
    print("ok")


if __name__ == "__main__":
    sys.exit(main())


def test_sdpa_return_lse_matches_reference():
    B, Hq, Hkv, D, S = 2, 32, 8, 128, 3001
    q, k, v = _make(B, Hq, Hkv, 1, S, D, mx.bfloat16, seed=11, strided=False)
    scale = 1.0 / (D**0.5)
    o, lse = kq.sdpa_decode_gqa(q, k, v, scale, return_lse=True)
    o0 = kq.sdpa_decode_gqa(q, k, v, scale)
    kr = mx.repeat(k, Hq // Hkv, axis=1).astype(mx.float32)
    s_ref = (q.astype(mx.float32) * scale) @ kr.swapaxes(-1, -2)
    lse_ref = mx.logsumexp(s_ref[:, :, 0, :], axis=-1)[..., None]
    _eval_or_skip(o, lse, o0, lse_ref)
    assert lse.shape == (B, Hq, 1) and lse.dtype == mx.float32
    assert float(mx.abs(o - o0).max()) == 0.0
    assert float(mx.abs(lse - lse_ref).max()) < 2e-2


def test_sdpa_fa_verify_return_lse():
    q, k, v = _make(1, 8, 8, 32, 2048, 128, mx.bfloat16, seed=12, strided=False)
    scale = 1.0 / (128**0.5)
    o, lse = kq.sdpa_fa_verify(q, k, v, scale, 1, return_lse=True)
    o0 = kq.sdpa_fa_verify(q, k, v, scale, 1)
    s_ref = (q.astype(mx.float32) * scale) @ k.astype(mx.float32).swapaxes(-1, -2)
    lse_ref = mx.logsumexp(s_ref, axis=-1)
    _eval_or_skip(o, lse, o0, lse_ref)
    assert lse.shape == (1, 8, 32) and lse.dtype == mx.float32
    assert float(mx.abs(o - o0).max()) == 0.0
    assert float(mx.abs(lse - lse_ref).max()) < 2e-2


def test_sdpa_cascade_lse_merge():
    # shared-prefix cascade: fa fold over the shared block plus a per-row
    # private call, LSE-merged, must match one call over the concatenated KV
    B, Hq, Hkv, D = 4, 32, 8, 128
    gqa = Hq // Hkv
    P, Sp = 4096, 384
    scale = 1.0 / (D**0.5)
    q, k_sh, v_sh = _make(1, Hq, Hkv, 1, P, D, mx.bfloat16, seed=13, strided=False)
    q = mx.random.normal((B, Hq, 1, D)).astype(mx.bfloat16) * 0.5
    _, k_pr, v_pr = _make(B, Hq, Hkv, 1, Sp, D, mx.bfloat16, seed=14, strided=False)
    k_full = mx.concatenate([mx.broadcast_to(k_sh, (B, Hkv, P, D)), k_pr], axis=2)
    v_full = mx.concatenate([mx.broadcast_to(v_sh, (B, Hkv, P, D)), v_pr], axis=2)
    ref = kq.sdpa_decode_gqa(q, mx.contiguous(k_full), mx.contiguous(v_full), scale)
    qf = mx.contiguous(
        q[:, :, 0, :]
        .reshape(B, Hkv, gqa, D)
        .transpose(1, 0, 2, 3)
        .reshape(1, Hkv, B * gqa, D)
    )
    o_f, l_f = kq.sdpa_fa_verify(qf, k_sh, v_sh, scale, 1, return_lse=True)
    o_sh = o_f.reshape(1, Hkv, B, gqa, D)[0].transpose(1, 0, 2, 3).reshape(B, Hq, 1, D)
    l_sh = l_f.reshape(1, Hkv, B, gqa)[0].transpose(1, 0, 2).reshape(B, Hq, 1)
    o_pr, l_pr = kq.sdpa_decode_gqa(q, k_pr, v_pr, scale, return_lse=True)
    m = mx.maximum(l_sh, l_pr)
    w_sh = mx.exp(l_sh - m)[..., None]
    w_pr = mx.exp(l_pr - m)[..., None]
    merged = (
        (o_sh.astype(mx.float32) * w_sh + o_pr.astype(mx.float32) * w_pr)
        / (w_sh + w_pr)
    ).astype(q.dtype)
    _eval_or_skip(merged, ref)
    rel = _rel(merged, ref)
    assert rel < REL_BOUND[mx.bfloat16], f"cascade merge rel {rel:.3e}"


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_sdpa_cascade_fused_matches_concat(D, dtype):
    # fused cascade op == one sdpa_decode_gqa call over the concatenated KV
    B, Hq, Hkv = 4, 32, 8
    P, Sp = 3071, 257
    scale = 1.0 / (D**0.5)
    _, k_sh, v_sh = _make(1, Hq, Hkv, 1, P, D, dtype, seed=21, strided=False)
    q, k_pr, v_pr = _make(B, Hq, Hkv, 1, Sp, D, dtype, seed=22, strided=False)
    k_full = mx.contiguous(
        mx.concatenate([mx.broadcast_to(k_sh, (B, Hkv, P, D)), k_pr], axis=2)
    )
    v_full = mx.contiguous(
        mx.concatenate([mx.broadcast_to(v_sh, (B, Hkv, P, D)), v_pr], axis=2)
    )
    ref = kq.sdpa_decode_gqa(q, k_full, v_full, scale)
    got = kq.sdpa_decode_gqa_cascade(q, k_sh, v_sh, k_pr, v_pr, scale)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    print(f"  [cascade] fused D={D} {dtype}: rel={rel:.3e}")
    assert rel < REL_BOUND[dtype], f"fused cascade rel {rel:.3e}"


def test_sdpa_cascade_fused_starts():
    # per-row private left-pad: keys below starts[b] in the PRIVATE region
    # are excluded; the shared prefix is always fully attended
    B, Hq, Hkv, D = 4, 32, 8, 128
    P, Sp = 2048, 384
    scale = 1.0 / (D**0.5)
    _, k_sh, v_sh = _make(1, Hq, Hkv, 1, P, D, mx.bfloat16, seed=23, strided=False)
    q, k_pr, v_pr = _make(B, Hq, Hkv, 1, Sp, D, mx.bfloat16, seed=24, strided=False)
    starts = mx.array([0, 7, 133, Sp - 1], dtype=mx.int32)
    got = kq.sdpa_decode_gqa_cascade(q, k_sh, v_sh, k_pr, v_pr, scale, starts=starts)
    k_full = mx.concatenate([mx.broadcast_to(k_sh, (B, Hkv, P, D)), k_pr], axis=2)
    v_full = mx.concatenate([mx.broadcast_to(v_sh, (B, Hkv, P, D)), v_pr], axis=2)
    pos = mx.arange(P + Sp)[None, :]
    keep = pos >= (starts[:, None] + P)
    keep = mx.logical_or(pos < P, keep)
    bias = mx.where(keep, mx.zeros(keep.shape), mx.full(keep.shape, -mx.inf))
    kr = mx.repeat(k_full, Hq // Hkv, axis=1).astype(mx.float32)
    vr = mx.repeat(v_full, Hq // Hkv, axis=1).astype(mx.float32)
    s_ref = (q.astype(mx.float32) * scale) @ kr.swapaxes(-1, -2)
    s_ref = s_ref + bias[:, None, None, :]
    ref = mx.softmax(s_ref, axis=-1) @ vr
    _eval_or_skip(got, ref)
    rel = _rel(got.astype(mx.float32), ref)
    assert rel < REL_BOUND[mx.bfloat16], f"cascade starts rel {rel:.3e}"


def test_sdpa_cascade_fused_return_lse():
    B, Hq, Hkv, D = 2, 16, 8, 128
    P, Sp = 1023, 65
    scale = 1.0 / (D**0.5)
    _, k_sh, v_sh = _make(1, Hq, Hkv, 1, P, D, mx.bfloat16, seed=25, strided=False)
    q, k_pr, v_pr = _make(B, Hq, Hkv, 1, Sp, D, mx.bfloat16, seed=26, strided=False)
    o, lse = kq.sdpa_decode_gqa_cascade(
        q, k_sh, v_sh, k_pr, v_pr, scale, return_lse=True
    )
    o0 = kq.sdpa_decode_gqa_cascade(q, k_sh, v_sh, k_pr, v_pr, scale)
    k_full = mx.concatenate([mx.broadcast_to(k_sh, (B, Hkv, P, D)), k_pr], axis=2)
    kr = mx.repeat(k_full, Hq // Hkv, axis=1).astype(mx.float32)
    s_ref = (q.astype(mx.float32) * scale) @ kr.swapaxes(-1, -2)
    lse_ref = mx.logsumexp(s_ref[:, :, 0, :], axis=-1)[..., None]
    _eval_or_skip(o, lse, o0, lse_ref)
    assert lse.shape == (B, Hq, 1) and lse.dtype == mx.float32
    assert float(mx.abs(o - o0).max()) == 0.0
    assert float(mx.abs(lse - lse_ref).max()) < 2e-2


def test_sdpa_cascade_fused_validation():
    B, Hq, Hkv, D = 2, 16, 8, 128
    scale = 1.0 / (D**0.5)
    _, k_sh, v_sh = _make(1, Hq, Hkv, 1, 512, D, mx.bfloat16, seed=27, strided=False)
    q, k_pr, v_pr = _make(B, Hq, Hkv, 1, 64, D, mx.bfloat16, seed=28, strided=False)
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_cascade(q, k_pr, v_pr, k_pr, v_pr, scale)  # shared B != 1
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_cascade(
            q, k_sh, v_sh, k_pr[:, :, :0, :], v_pr[:, :, :0, :], scale
        )  # empty private region
    q9 = mx.concatenate([q] * 9, axis=2)
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_cascade(q9, k_sh, v_sh, k_pr, v_pr, scale)  # qL > 8
    # over the folded-row cap: B2 * gqa8 * qL5 = 80 > 64
    qw, k_w, v_w = _make(B, 64, Hkv, 5, 64, D, mx.bfloat16, seed=29, strided=False)
    _, ksw, vsw = _make(1, 64, Hkv, 1, 512, D, mx.bfloat16, seed=30, strided=False)
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_cascade(qw, ksw, vsw, k_w, v_w, scale)


def _cascade_verify_ref(q, k, v, pads, scale, qL):
    # f32 masked reference: per-row left pad + end-aligned causal block
    B, Hq, _, D = q.shape
    Hkv, L = k.shape[1], k.shape[2]
    kr = mx.repeat(k, Hq // Hkv, axis=1).astype(mx.float32)
    vr = mx.repeat(v, Hq // Hkv, axis=1).astype(mx.float32)
    s = (q.astype(mx.float32) * scale) @ kr.swapaxes(-1, -2)
    pos = mx.arange(L)[None, None, None, :]
    end = (L - qL) + mx.arange(qL)[None, None, :, None]
    pad = mx.array(pads)[:, None, None, None]
    keep = (pos >= pad) & (pos <= end)
    s = mx.where(keep, s, mx.array(-mx.inf))
    return mx.softmax(s, axis=-1) @ vr


@pytest.mark.parametrize(
    "B,Hq,Hkv,D,qL,pads",
    [
        (2, 4, 2, 256, 8, [0, 64]),  # gemma-31b assistant geometry
        (2, 12, 2, 256, 5, [0, 64]),  # qwen nextn geometry at cap-2
        (2, 8, 4, 512, 3, [0, 16]),  # hd512 d-split walk
        (4, 16, 8, 128, 8, [0, 3, 511, 64]),  # 64 folded rows exactly
        (1, 8, 8, 64, 5, [0]),
    ],
)
def test_sdpa_cascade_fused_verify_width(B, Hq, Hkv, D, qL, pads):
    # qL > 1: end-aligned causal on the private suffix, full shared
    # visibility, per-row starts honored
    P, sp = 2048, 96
    scale = 1.0 / (D**0.5)
    mx.random.seed(41)
    L = max(pads) + P + sp
    kb = mx.random.normal((B, Hkv, L, D)).astype(mx.float16)
    vb = mx.random.normal((B, Hkv, L, D)).astype(mx.float16)
    pk = kb[0:1, :, pads[0] : pads[0] + P]
    pv = vb[0:1, :, pads[0] : pads[0] + P]
    rk, rv = [], []
    for b in range(B):
        rk.append(
            mx.concatenate(
                [kb[b : b + 1, :, : pads[b]], pk, kb[b : b + 1, :, pads[b] + P :]],
                axis=2,
            )
        )
        rv.append(
            mx.concatenate(
                [vb[b : b + 1, :, : pads[b]], pv, vb[b : b + 1, :, pads[b] + P :]],
                axis=2,
            )
        )
    k = mx.concatenate(rk, axis=0)
    v = mx.concatenate(rv, axis=0)
    q = mx.random.normal((B, Hq, qL, D)).astype(mx.float16)
    c0 = min(pads) + P
    starts = None
    if any(pads):
        starts = mx.array([p + P - c0 for p in pads], dtype=mx.int32)
    got, lse = kq.sdpa_decode_gqa_cascade(
        q, pk, pv, k[:, :, c0:], v[:, :, c0:], scale, starts=starts, return_lse=True
    )
    ref = _cascade_verify_ref(q, k, v, pads, scale, qL)
    _eval_or_skip(got, ref)
    assert lse.shape == (B, Hq, qL)
    rel = _rel(got, ref)
    print(f"  [cascade] verify D={D} qL={qL} B={B}: rel={rel:.3e}")
    assert rel < REL_BOUND[mx.float16], f"cascade verify rel {rel:.3e}"


def _q8(a):
    # mlx affine wire the batch quantized caches produce (group 64, bits 8)
    return mx.quantize(a, group_size=64, bits=8)


@pytest.mark.parametrize(
    "B,Hq,Hkv,D,qL",
    [
        (4, 32, 8, 128, 1),  # plain batch decode
        (4, 16, 8, 256, 1),  # hd256 decode
        (2, 12, 2, 256, 5),  # qwen verify geometry (60 rows)
        (2, 8, 2, 128, 8),  # 64 folded rows exactly
    ],
)
def test_sdpa_cascade_fused_kv_q8(B, Hq, Hkv, D, qL):
    # q8 operands == the fp16 cascade run on the dequantized arrays,
    # bit-exact: both stage the same T values, the math after the stage
    # is identical
    P, Sp = 2047, 193 + qL
    scale = 1.0 / (D**0.5)
    _, k_sh, v_sh = _make(1, Hq, Hkv, 1, P, D, mx.float16, seed=51, strided=False)
    q, k_pr, v_pr = _make(B, Hq, Hkv, qL, Sp, D, mx.float16, seed=52, strided=False)
    starts = mx.array([(7 * b) % 64 for b in range(B)], dtype=mx.int32)

    ksh_w, ksh_s, ksh_b = _q8(k_sh)
    vsh_w, vsh_s, vsh_b = _q8(v_sh)
    kpr_w, kpr_s, kpr_b = _q8(k_pr)
    vpr_w, vpr_s, vpr_b = _q8(v_pr)

    got = kq.sdpa_decode_gqa_cascade(
        q,
        ksh_w,
        vsh_w,
        kpr_w,
        vpr_w,
        scale,
        starts=starts,
        k_shared_scales=ksh_s,
        k_shared_biases=ksh_b,
        v_shared_scales=vsh_s,
        v_shared_biases=vsh_b,
        k_priv_scales=kpr_s,
        k_priv_biases=kpr_b,
        v_priv_scales=vpr_s,
        v_priv_biases=vpr_b,
    )

    def dq(w, s, b):
        return mx.dequantize(w, s, b, group_size=64, bits=8)

    ref = kq.sdpa_decode_gqa_cascade(
        q,
        dq(ksh_w, ksh_s, ksh_b).astype(mx.float16),
        dq(vsh_w, vsh_s, vsh_b).astype(mx.float16),
        dq(kpr_w, kpr_s, kpr_b).astype(mx.float16),
        dq(vpr_w, vpr_s, vpr_b).astype(mx.float16),
        scale,
        starts=starts,
    )
    _eval_or_skip(got, ref)
    diff = float(mx.abs(got - ref).max())
    print(f"  [cascade] kv_q8 D={D} qL={qL}: max|d|={diff:.3e}")
    assert diff == 0.0, f"cascade kv_q8 not bit-exact: {diff:.3e}"


def test_sdpa_cascade_fused_kv_q8_validation():
    B, Hq, Hkv, D = 2, 16, 8, 128
    scale = 1.0 / (D**0.5)
    _, k_sh, v_sh = _make(1, Hq, Hkv, 1, 512, D, mx.float16, seed=53, strided=False)
    q, k_pr, v_pr = _make(B, Hq, Hkv, 1, 64, D, mx.float16, seed=54, strided=False)
    ksh_w, ksh_s, ksh_b = _q8(k_sh)
    with pytest.raises(ValueError):
        # partial q8 set (scales without biases)
        kq.sdpa_decode_gqa_cascade(
            q, ksh_w, v_sh, k_pr, v_pr, scale, k_shared_scales=ksh_s
        )
    # D=512 rejects q8
    _, k5, v5 = _make(1, 8, 4, 1, 512, 512, mx.float16, seed=55, strided=False)
    q5, kp5, vp5 = _make(2, 8, 4, 1, 64, 512, mx.float16, seed=56, strided=False)
    args = {}
    for name, arr in (
        ("k_shared", k5),
        ("v_shared", v5),
        ("k_priv", kp5),
        ("v_priv", vp5),
    ):
        w, s, b = _q8(arr)
        args[name] = w
        args[name + "_scales"] = s
        args[name + "_biases"] = b
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_cascade(
            q5,
            args["k_shared"],
            args["v_shared"],
            args["k_priv"],
            args["v_priv"],
            1.0 / (512**0.5),
            k_shared_scales=args["k_shared_scales"],
            k_shared_biases=args["k_shared_biases"],
            v_shared_scales=args["v_shared_scales"],
            v_shared_biases=args["v_shared_biases"],
            k_priv_scales=args["k_priv_scales"],
            k_priv_biases=args["k_priv_biases"],
            v_priv_scales=args["v_priv_scales"],
            v_priv_biases=args["v_priv_biases"],
        )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sdpa_paged_matches_selected_reference(D, dtype):
    # page-gather decode == f32 attention over exactly the selected pages
    import numpy as np

    np.random.seed(31)
    B, Hq, Hkv, S, npages = 2, 16, 8, 4093, 24
    page = 32 if D <= 128 else 16
    tot = (S + page - 1) // page
    scale = 1.0 / (D**0.5)
    q, k, v = _make(B, Hq, Hkv, 1, S, D, dtype, seed=32, strided=False)
    pg = np.stack(
        [
            np.stack(
                [
                    np.sort(np.random.choice(tot, size=npages, replace=False))
                    for _ in range(Hkv)
                ]
            )
            for _ in range(B)
        ]
    ).astype(np.int32)
    pages = mx.array(pg)
    got = kq.sdpa_decode_gqa_paged(q, k, v, scale, pages)
    kr = mx.repeat(k, Hq // Hkv, axis=1).astype(mx.float32)
    vr = mx.repeat(v, Hq // Hkv, axis=1).astype(mx.float32)
    sc = (q.astype(mx.float32) * scale) @ kr.swapaxes(-1, -2)
    mask = np.zeros((B, Hkv, S), dtype=bool)
    for b in range(B):
        for h in range(Hkv):
            for pp in pg[b, h]:
                mask[b, h, pp * page : min((pp + 1) * page, S)] = True
    mask = np.repeat(mask, Hq // Hkv, axis=1)[:, :, None, :]
    bias = mx.array(np.where(mask, 0.0, -np.inf).astype(np.float32))
    ref = mx.softmax(sc + bias, axis=-1) @ vr
    _eval_or_skip(got, ref)
    err = float(mx.abs(got.astype(mx.float32) - ref).max())
    assert err < 2e-2, f"paged vs selected ref err={err}"


def test_sdpa_paged_all_pages_is_dense():
    # selecting every page must reproduce the dense decode call
    import numpy as np

    B, Hq, Hkv, D, S = 2, 32, 8, 128, 2048
    scale = 1.0 / (D**0.5)
    q, k, v = _make(B, Hq, Hkv, 1, S, D, mx.bfloat16, seed=33, strided=False)
    tot = S // 32
    pg = np.broadcast_to(np.arange(tot, dtype=np.int32), (B, Hkv, tot)).copy()
    got = kq.sdpa_decode_gqa_paged(q, k, v, scale, mx.array(pg))
    ref = kq.sdpa_decode_gqa(q, k, v, scale)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    assert rel < REL_BOUND[mx.bfloat16], f"all-pages vs dense rel {rel:.3e}"


def test_sdpa_paged_starts():
    # left-padded rows: pad positions inside selected pages score -inf
    import numpy as np

    np.random.seed(41)
    B, Hq, Hkv, D, S, npages = 3, 16, 8, 128, 4096, 20
    page, scale = 32, 1.0 / (D**0.5)
    pads = [0, 37, 511]
    q, k, v = _make(B, Hq, Hkv, 1, S, D, mx.float16, seed=42, strided=False)
    pg = np.stack(
        [
            np.stack(
                [
                    np.sort(np.random.choice(S // page, size=npages, replace=False))
                    for _ in range(Hkv)
                ]
            )
            for _ in range(B)
        ]
    ).astype(np.int32)
    # force the pad-boundary page resident so masking inside it is exercised
    for b in range(B):
        pg[b, :, 0] = pads[b] // page
    starts = mx.array(pads, dtype=mx.int32)
    got = kq.sdpa_decode_gqa_paged(q, k, v, scale, mx.array(pg), starts=starts)
    kr = mx.repeat(k, Hq // Hkv, axis=1).astype(mx.float32)
    vr = mx.repeat(v, Hq // Hkv, axis=1).astype(mx.float32)
    sc = (q.astype(mx.float32) * scale) @ kr.swapaxes(-1, -2)
    mask = np.zeros((B, Hkv, S), dtype=bool)
    for b in range(B):
        for h in range(Hkv):
            for pp in pg[b, h]:
                mask[b, h, pp * page : (pp + 1) * page] = True
        mask[b, :, : pads[b]] = False
    mask = np.repeat(mask, Hq // Hkv, axis=1)[:, :, None, :]
    bias = mx.array(np.where(mask, 0.0, -np.inf).astype(np.float32))
    ref = mx.softmax(sc + bias, axis=-1) @ vr
    _eval_or_skip(got, ref)
    err = float(mx.abs(got.astype(mx.float32) - ref).max())
    assert err < 2e-2, f"paged+starts vs masked ref err={err}"


def test_sdpa_paged_validation():
    B, Hq, Hkv, D, S = 2, 16, 8, 128, 1024
    scale = 1.0 / (D**0.5)
    q, k, v = _make(B, Hq, Hkv, 1, S, D, mx.bfloat16, seed=34, strided=False)
    good = mx.zeros((B, Hkv, 4), dtype=mx.int32)
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_paged(q, k, v, scale, good.astype(mx.float32))
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_paged(
            q, k, v, scale, mx.zeros((B, Hkv + 1, 4), dtype=mx.int32)
        )
    q2 = mx.concatenate([q, q], axis=2)
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_paged(q2, k, v, scale, good)
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_paged(
            q, k, v, scale, good, starts=mx.zeros((B + 1,), dtype=mx.int32)
        )
    with pytest.raises(ValueError):
        kq.sdpa_decode_gqa_paged(
            q, k, v, scale, good, starts=mx.zeros((B,), dtype=mx.float32)
        )
