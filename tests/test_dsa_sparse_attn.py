#!/usr/bin/env python3
"""kq.dsa_sparse_attention validation vs an independent f32 reference.

The reference materializes, per (batch, query position), the visible KV set
the kernel streams through tiles -- the causal sliding local window plus the
top-k-selected pooled rows clamped to the causal pooled horizon
(q_offset + pos + 1) / compress_ratio -- and computes a plain f32 softmax
with the per-head sink folded into the denominator. It shares no code with
the kernel's exp2-space online softmax, so a bug cannot cancel out of both
sides.

Covers decode (qL=1), MTP verify (qL=2) and prefill (qL=8/64) widths -- the
omlx original was gated to qL>1; the qL<=1 arms pin the ported kernel's
softmax seeding and truncated-tile masking -- both float dtypes, the
visibility clamp (pooled_valid < topk), out-of-horizon indices, and a
pooled row count that spans multiple BK=256 key tiles.

Metal-only kernel (eval_cpu throws): skipped under KQUANT_FORCE_CPU.

Usage: test_dsa_sparse_attn.py
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq.dsa_sparse_attention is a Metal-only kernel; no CPU path.",
)

REL_BOUND = {mx.bfloat16: 5e-3, mx.float16: 2e-3}

H, D = 64, 512  # kernel geometry (fixed by the instantiations)


def _ref(q, local_kv, pooled, topk, sinks, scale, q_offset, ratio, window):
    """f32 reference on the exact (already dtype-cast) input values."""
    qf = np.array(q.astype(mx.float32))
    lf = np.array(local_kv.astype(mx.float32))
    pf = np.array(pooled.astype(mx.float32))
    sf = np.array(sinks.astype(mx.float32))
    tk = np.array(topk)
    B, _, qL, _ = qf.shape
    localL = lf.shape[2]
    P = pf.shape[1]
    out = np.zeros_like(qf)
    for b in range(B):
        for p in range(qL):
            local_end = min(localL, localL - qL + p + 1)
            local_start = max(0, local_end - window)
            loc = lf[b, 0, local_start:local_end]  # (n_loc, D)
            pooled_valid = min(P, (q_offset + p + 1) // ratio)
            sel = [int(i) for i in tk[b, 0, p] if int(i) < pooled_valid]
            kv = np.concatenate([loc, pf[b, sel]], axis=0)  # (n, D)
            s = (qf[b, :, p] @ kv.T) * scale  # (H, n)
            m = np.maximum(s.max(axis=-1), sf)
            w = np.exp(s - m[:, None])
            denom = w.sum(axis=-1) + np.exp(sf - m)  # sink in denominator
            out[b, :, p] = (w / denom[:, None]) @ kv
    return out


def _rel(got, ref):
    g = np.array(got.astype(mx.float32))
    return float(np.linalg.norm(g - ref) / (np.linalg.norm(ref) + 1e-6))


def _case(qL, localL, P, topk_n, q_offset, ratio, window, dtype, seed):
    rng = np.random.default_rng(seed)
    B = 1
    q = mx.array(rng.standard_normal((B, H, qL, D)) * 0.3).astype(dtype)
    local_kv = mx.array(rng.standard_normal((B, 1, localL, D)) * 0.3).astype(
        dtype
    )
    pooled = mx.array(rng.standard_normal((B, P, D)) * 0.3).astype(dtype)
    sinks = mx.array(rng.standard_normal((H,)) * 0.5).astype(dtype)
    # Unique indices into [0, P): the horizon clamp, not the test data, is
    # what masks slots. Some rows deliberately land beyond the horizon.
    tk = np.stack(
        [
            np.stack(
                [
                    rng.choice(P, size=topk_n, replace=False).astype(np.uint32)
                    for _ in range(qL)
                ],
                0,
            )
        ],
        0,
    )[:, None]
    topk = mx.array(tk)
    scale = D**-0.5

    got = kq.dsa_sparse_attention(
        q, local_kv, pooled, topk, sinks, scale, q_offset, ratio, window
    )
    mx.eval(got)
    ref = _ref(q, local_kv, pooled, topk, sinks, scale, q_offset, ratio, window)
    return _rel(got, ref)


CASES = [
    # (name, qL, localL, P, topk_n, q_offset, ratio, window)
    ("decode", 1, 128, 640, 512, 2560, 4, 128),
    ("decode_clamped", 1, 128, 640, 512, 1024, 4, 128),  # horizon < many idx
    ("verify_qL2", 2, 128, 640, 512, 2559, 4, 128),
    ("prefill_qL8", 8, 136, 700, 512, 2792, 4, 128),
    ("prefill_qL64", 64, 192, 700, 512, 2736, 4, 128),
    ("short_local", 1, 40, 300, 100, 1199, 4, 128),  # local_count < window
    ("ratio128", 1, 128, 96, 64, 12288, 128, 128),  # compressed-pool layer
]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_dsa_sparse_attention(case, dtype):
    name, qL, localL, P, topk_n, q_offset, ratio, window = case
    rel = _case(qL, localL, P, topk_n, q_offset, ratio, window, dtype, seed=7)
    assert rel < REL_BOUND[dtype], f"{name} {dtype}: rel {rel:.3e}"


def test_dsa_sparse_attention_rejects_bad_shapes():
    q = mx.zeros((1, 64, 1, 512), dtype=mx.float16)
    kv = mx.zeros((1, 1, 8, 512), dtype=mx.float16)
    pooled = mx.zeros((1, 16, 512), dtype=mx.float16)
    sinks = mx.zeros((64,), dtype=mx.float16)
    good_topk = mx.zeros((1, 1, 1, 8), dtype=mx.uint32)
    with pytest.raises(ValueError):
        kq.dsa_sparse_attention(  # topk must be uint32
            q, kv, pooled, good_topk.astype(mx.int32), sinks, 1.0, 8, 4, 128
        )
    with pytest.raises(ValueError):
        kq.dsa_sparse_attention(  # H must be 64
            q[:, :32], kv, pooled, good_topk, sinks[:32], 1.0, 8, 4, 128
        )


def main() -> int:
    fails = 0
    for case in CASES:
        name, qL, localL, P, topk_n, q_offset, ratio, window = case
        for dtype in (mx.float16, mx.bfloat16):
            rel = _case(
                qL, localL, P, topk_n, q_offset, ratio, window, dtype, seed=7
            )
            ok = rel < REL_BOUND[dtype]
            fails += not ok
            print(
                f"  {name:<16} {str(dtype):<18} rel={rel:.3e} "
                f"{'ok' if ok else 'FAIL'}"
            )
    print("ALL OK" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
