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
@pytest.mark.parametrize("qL", [2, 3, 4, 5, 6])
def test_sdpa_fa_verify(qL, dtype):
    _check_fa(256, qL, kL=4096, dtype=dtype, G=4)


def test_sdpa_fa_verify_qwen_geometry():
    # qwen3.5/3.6 full-attn fold: 24 rows = G6 x qL4 at hd256
    _check_fa(256, 4, kL=8192, dtype=mx.bfloat16, Hkv=4, G=6)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_sdpa_fa_verify_decode(dtype):
    # q_len == 1: plain GQA decode on the matrix units (every folded row
    # attends the full KV). 122b shape: G16 x qL1 at hd256, 2 kv heads.
    _check_fa(256, 1, kL=8192, dtype=dtype, Hkv=2, G=16)


def test_sdpa_fa_verify_full_tile():
    # n_rows == 32 fills the tile exactly (no padding rows), qL at the cap
    _check_fa(256, 8, kL=4096, dtype=mx.bfloat16, Hkv=2, G=4)


def test_sdpa_fa_verify_partial_warp():
    # n_rows == 18: the third simdgroup covers rows 16..17 plus padding
    _check_fa(256, 6, kL=2048, dtype=mx.bfloat16, G=3)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_sdpa_fa_verify_strided_unaligned(dtype):
    # strided KV-cache prefix + a key length off every tile/split boundary
    _check_fa(256, 4, kL=3071, dtype=dtype, strided=True, splits=16)


def test_sdpa_fa_verify_short_kv():
    # kL small enough that most splits stage zero keys: their empty partials
    # (max = finite_min, sum = 0) must merge as weight zero
    _check_fa(256, 4, kL=17, dtype=mx.bfloat16, splits=16, G=6)


def test_sdpa_fa_verify_min_kv():
    # kL == qL floor: row 0 attends exactly one key, every row masked hard
    _check_fa(256, 4, kL=4, dtype=mx.bfloat16, G=6)


def test_sdpa_fa_verify_causal_split_straddle():
    # the last qL keys straddle a split boundary (splits=128, kL=4098 puts
    # keys 4096..4097 alone in the final split): that split is entirely past
    # the low rows' causal limits, exercising the dead-row guard in a
    # non-empty split
    _check_fa(256, 4, kL=4098, dtype=mx.bfloat16, splits=128, G=6)


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
