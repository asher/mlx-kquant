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


def _check(D, qL, kL, dtype, Hq=32, Hkv=16, strided=False):
    causal = qL > 1  # qL==1 attends all keys; offset-causal is the verify regime
    scale = 1.0 / (D**0.5)
    q, k, v = _make(1, Hq, Hkv, qL, kL, D, dtype, seed=qL * 7 + kL + D, strided=strided)
    got = kq.sdpa_vector(q, k, v, scale, causal=causal)
    ref = _ref_sdpa(q, k, v, scale, causal)
    mx.eval(got, ref)
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
