#!/usr/bin/env python3
"""kq.dsa_indexer_qat vs the mx-graph QAT chain it replaces, bit-identity.

The reference replicates gguf-mlx's _indexer_qat_roundtrip inline (ds4.c
dsv4_indexer_qat_row semantics): fp32 Hadamard via mx.hadamard_transform,
then the per-32-block FP4-E2M1 round-trip (exact power-of-two scale table,
threshold-ladder rounding). The kernel replicates mlx's hadamard butterfly
order and precise::log2, so outputs are asserted BIT-identical, not just
close: any drift would move top-k selection ties.

Metal-only kernel (eval_cpu throws): skipped under KQUANT_FORCE_CPU.

Usage: test_dsa_qat.py
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
    reason="kq.dsa_indexer_qat is a Metal-only kernel; no CPU path.",
)

_EXP2_TABLE = mx.array([2.0**i for i in range(-126, 128)], dtype=mx.float32)


def _exp2i(e):
    return mx.take(_EXP2_TABLE, e.astype(mx.int32) + 126)


def _e2m1_round(v):
    s = mx.sign(v)
    a = mx.abs(v)
    q = mx.where(
        a <= 0.25,
        0.0,
        mx.where(
            a < 0.75,
            0.5,
            mx.where(
                a <= 1.25,
                1.0,
                mx.where(
                    a < 1.75,
                    1.5,
                    mx.where(
                        a <= 2.5,
                        2.0,
                        mx.where(a < 3.5, 3.0, mx.where(a <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return s * q


def _ref(x):
    orig = x.dtype
    v = mx.hadamard_transform(x.astype(mx.float32))
    v = mx.unflatten(v, -1, (-1, 32))
    amax = mx.maximum(mx.max(mx.abs(v), axis=-1, keepdims=True), 7.052966104933725e-38)
    scale = _exp2i(mx.ceil(mx.log2(amax / 6.0)))
    v = _e2m1_round(mx.clip(v / scale, -6.0, 6.0)) * scale
    return mx.flatten(v, -2).astype(orig)


def _bits(a):
    bits_dtype = {2: mx.uint16, 4: mx.uint32}[a.itemsize]
    return np.array(a.view(bits_dtype))


# Magnitudes stay inside each dtype's finite range: inf/NaN inputs are out
# of the QAT domain, and NaN max semantics differ between metal::max and
# mx.maximum, so overflowed inputs would diverge for reasons the real
# indexer path can never hit.
_BIG = {mx.float16: 1e3, mx.bfloat16: 1e20, mx.float32: 1e20}
_MIX_HI = {mx.float16: 4, mx.bfloat16: 30, mx.float32: 30}

CASES = [
    ("normal", lambda r, s, d: r.standard_normal(s)),
    ("small", lambda r, s, d: r.standard_normal(s) * 1e-20),
    ("large", lambda r, s, d: r.standard_normal(s) * _BIG[d]),
    ("subnormal_floor", lambda r, s, d: r.standard_normal(s) * 1e-38),
    (
        "mixed_rows",
        lambda r, s, d: (
            r.standard_normal(s) * (10.0 ** r.integers(-12, _MIX_HI[d], size=(s[0], 1)))
        ),
    ),
    ("zeros", lambda r, s, d: np.zeros(s)),
    # Dyadic lattice inputs: post-hadamard values land on exact multiples of
    # small powers of two, hammering the e2m1 tie thresholds after the
    # power-of-two block scale.
    ("dyadic_ties", lambda r, s, d: r.integers(-24, 25, size=s) * 0.25),
]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_dsa_indexer_qat_bit_identity(case, dtype):
    name, gen = case
    rng = np.random.default_rng(11)
    x = mx.array(gen(rng, (4096, 128), dtype)).astype(dtype)
    mx.eval(x)
    got = kq.dsa_indexer_qat(x)
    ref = _ref(x)
    mx.eval(got, ref)
    gb, rb = _bits(got), _bits(ref)
    mismatch = int((gb != rb).sum())
    assert mismatch == 0, (
        f"{name} {dtype}: {mismatch}/{gb.size} words differ "
        f"(first at {np.argwhere(gb != rb)[:4].tolist()})"
    )


def test_dsa_indexer_qat_shapes_and_rejects():
    x4 = mx.random.normal((2, 7, 64, 128)).astype(mx.bfloat16)
    mx.eval(x4)
    got = kq.dsa_indexer_qat(x4)
    ref = _ref(x4)
    mx.eval(got, ref)
    assert got.shape == x4.shape
    assert not int((_bits(got) != _bits(ref)).sum())

    with pytest.raises(ValueError):
        kq.dsa_indexer_qat(mx.zeros((4, 64), dtype=mx.float16))
    with pytest.raises(ValueError):
        kq.dsa_indexer_qat(mx.zeros((4, 128), dtype=mx.int32))


def main() -> int:
    fails = 0
    for name, gen in CASES:
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            rng = np.random.default_rng(11)
            x = mx.array(gen(rng, (4096, 128), dtype)).astype(dtype)
            mx.eval(x)
            got = kq.dsa_indexer_qat(x)
            ref = _ref(x)
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
