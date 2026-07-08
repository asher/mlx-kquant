#!/usr/bin/env python3
"""kq.dsa_kv_qat vs the mx-graph KV QAT chain it replaces, bit-identity.

The reference replicates gguf-mlx's _kv_qat_roundtrip inline (ds4.c
dsv4_fp8_kv_quantize_row + f16_round semantics): split off the trailing
n_rot RoPE dims, per-64-block FP8-E4M3FN round-trip on the rest (exact
power-of-two scale table, ties-to-even mantissa rounding, 1e-4 amax floor,
2^-9 subnormal step floor), concat, then round the whole row through fp16.
The kernel uses precise::log2 / ldexp and re-rounds the fp8 result through
the storage dtype before the fp16 step (the graph's per-slice astype), so
outputs are asserted BIT-identical, not just close: any drift would move
KV-cache contents.

Metal-only kernel (eval_cpu throws): skipped under KQUANT_FORCE_CPU.

Usage: test_dsa_kv_qat.py
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
    reason="kq.dsa_kv_qat is a Metal-only kernel; no CPU path.",
)

_EXP2_TABLE = mx.array([2.0**i for i in range(-126, 128)], dtype=mx.float32)


def _exp2i(e):
    return mx.take(_EXP2_TABLE, e.astype(mx.int32) + 126)


def _e4m3_round(v):
    s = mx.sign(v)
    a = mx.abs(v)
    e = mx.floor(mx.log2(mx.maximum(a, 2.0**-9)))
    e = mx.clip(e, -6.0, 8.0)
    q = _exp2i(e - 3.0)
    return s * mx.round(a / q) * q


def _fp8_roundtrip(x, block=64):
    orig = x.dtype
    v = mx.unflatten(x.astype(mx.float32), -1, (-1, block))
    amax = mx.maximum(mx.max(mx.abs(v), axis=-1, keepdims=True), 1e-4)
    scale = _exp2i(mx.ceil(mx.log2(amax / 448.0)))
    v = _e4m3_round(mx.clip(v / scale, -448.0, 448.0)) * scale
    return mx.flatten(v, -2).astype(orig)


def _ref(kv, n_rot):
    orig = kv.dtype
    nope, rot = mx.split(kv, [kv.shape[-1] - n_rot], axis=-1)
    kv = mx.concatenate([_fp8_roundtrip(nope), rot], axis=-1)
    return kv.astype(mx.float16).astype(orig)


def _bits(a):
    bits_dtype = {2: mx.uint16, 4: mx.uint32}[a.itemsize]
    return np.array(a.view(bits_dtype))


# Magnitudes stay inside each dtype's finite range (same rationale as
# test_dsa_qat: NaN/inf inputs are outside the QAT domain). f16 saturation
# of large in-range bf16/f32 values is in-domain: both paths convert
# through IEEE RTNE and produce identical inf bits.
_BIG = {mx.float16: 1e3, mx.bfloat16: 1e20, mx.float32: 1e20}
_MIX_HI = {mx.float16: 4, mx.bfloat16: 30, mx.float32: 30}

CASES = [
    ("normal", lambda r, s, d: r.standard_normal(s)),
    ("small", lambda r, s, d: r.standard_normal(s) * 1e-20),
    ("amax_floor", lambda r, s, d: r.standard_normal(s) * 1e-5),
    ("large", lambda r, s, d: r.standard_normal(s) * _BIG[d]),
    ("subnormal", lambda r, s, d: r.standard_normal(s) * 1e-38),
    (
        "mixed_rows",
        lambda r, s, d: (
            r.standard_normal(s) * (10.0 ** r.integers(-12, _MIX_HI[d], size=(s[0], 1)))
        ),
    ),
    ("zeros", lambda r, s, d: np.zeros(s)),
    # Dyadic lattice inputs land on exact multiples of the e4m3 step after
    # the power-of-two block scale, hammering the ties-to-even rounding.
    ("dyadic_ties", lambda r, s, d: r.integers(-512, 513, size=s) * 0.0625),
]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_dsa_kv_qat_bit_identity(case, dtype):
    name, gen = case
    rng = np.random.default_rng(7)
    x = mx.array(gen(rng, (2048, 576), dtype)).astype(dtype)
    mx.eval(x)
    got = kq.dsa_kv_qat(x, 64)
    ref = _ref(x, 64)
    mx.eval(got, ref)
    gb, rb = _bits(got), _bits(ref)
    mismatch = int((gb != rb).sum())
    assert mismatch == 0, (
        f"{name} {dtype}: {mismatch}/{gb.size} words differ "
        f"(first at {np.argwhere(gb != rb)[:4].tolist()})"
    )


def test_dsa_kv_qat_shapes_and_rejects():
    # 4-D decode shape [B, 1, L, D] and a non-576 geometry (128 + 64)
    x4 = mx.random.normal((2, 1, 3, 576)).astype(mx.bfloat16)
    mx.eval(x4)
    got, ref = kq.dsa_kv_qat(x4, 64), _ref(x4, 64)
    mx.eval(got, ref)
    assert got.shape == x4.shape
    assert not int((_bits(got) != _bits(ref)).sum())

    xs = mx.random.normal((5, 192)).astype(mx.float16)
    mx.eval(xs)
    got, ref = kq.dsa_kv_qat(xs, 64), _ref(xs, 64)
    mx.eval(got, ref)
    assert not int((_bits(got) != _bits(ref)).sum())

    # rope slice must be exempt from fp8 (only f16-rounded): values above
    # 448 in the rope tail survive when the row scale is 1
    xt = mx.concatenate([mx.ones((1, 512)), mx.full((1, 64), 1000.0)], axis=-1).astype(
        mx.float32
    )
    mx.eval(xt)
    got = kq.dsa_kv_qat(xt, 64)
    mx.eval(got)
    assert float(got[0, -1]) == 1000.0

    with pytest.raises(ValueError):
        kq.dsa_kv_qat(mx.zeros((4, 570), dtype=mx.float16), 64)  # misaligned
    with pytest.raises(ValueError):
        kq.dsa_kv_qat(mx.zeros((4, 64), dtype=mx.float16), 64)  # no fp8 dims
    with pytest.raises(ValueError):
        kq.dsa_kv_qat(mx.zeros((4, 576), dtype=mx.int32), 64)


def main() -> int:
    fails = 0
    for name, gen in CASES:
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            rng = np.random.default_rng(7)
            x = mx.array(gen(rng, (2048, 576), dtype)).astype(dtype)
            mx.eval(x)
            got = kq.dsa_kv_qat(x, 64)
            ref = _ref(x, 64)
            mx.eval(got, ref)
            n = int((_bits(got) != _bits(ref)).sum())
            fails += n > 0
            print(
                f"  {name:<16} {str(dtype):<18} "
                f"{'bit-identical' if n == 0 else f'{n} words differ'}"
            )
    print("ALL OK" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
