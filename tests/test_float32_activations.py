"""float32 activations run the bfloat16 kernels.

The ops cast float32 activations to bfloat16 before dispatch, so the
metallib has no quantized matmul or mat-vec kernel for float activations.
On every route of every op, a float32 call returns bfloat16, bit-identical
to the same call on the activations cast to bfloat16.
"""

from __future__ import annotations

import functools
import re

import mlx.core as mx
import numpy as np
import pytest
from qmm_probe import force_cpu, run_probe

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(force_cpu(), reason="GPU kernels")

K, N, E = 1024, 256, 4
NAX_ROUTES = {"nax", "nax_splitk", "verify_nax"}
# quantized_matmul, transpose=True: (codec, M, forced route or None for the
# default route of that shape).
ROUTES = [
    ("q4_k", 1, None),
    ("q6_k", 1, None),
    ("q4_k", 4, "qmv"),
    ("q4_0", 4, "verify_qmv"),
    ("q4_k", 4, "mv_ext"),
    ("pq2_0", 4, "verify_mma"),
    ("q4_0", 4, "verify_nax"),
    ("q4_k", 8, "splitk"),
    ("q4_k", 16, "nax"),
    ("q4_k", 16, "nax_splitk"),
    ("q4_k", 64, None),
]


# The float kernels left are the encode and dequantize kernels, which take
# weights.
def test_no_float_activation_kernels():
    lib = kq.metallib_kernels()
    floats = [
        n
        for n in lib
        if n.startswith("kquant_")
        and "_float_gs_" in n
        and not re.search(r"_(de)?quantize_", n)
    ]
    assert floats == [], (len(floats), floats[:4])
    assert "kquant_q4_k_dequantize_float_gs_256_b_4" in lib
    assert "kquant_q4_k_quantize_float_gs_256_b_4" in lib


@functools.cache
def _probe():
    calls = []
    for codec, m, route in ROUTES:
        env = {"KQ_QMM_ROUTE": route, "KQ_QMM_ROUTE_STRICT": "1" if route else None}
        for dtype in ("float32", "bfloat16"):
            calls.append(
                {
                    "codec": codec,
                    "shape": [m, K],
                    "n": N,
                    "dtype": dtype,
                    "seed": 7,
                    "env": env,
                }
            )
    res = run_probe(calls, env={"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"})
    return {r: (res[2 * i], res[2 * i + 1]) for i, r in enumerate(ROUTES)}


@pytest.mark.parametrize("codec,m,route", ROUTES)
def test_quantized_matmul_float32(codec, m, route):
    if route in NAX_ROUTES and not kq.nax_available():
        pytest.skip("NAX route")
    f32, bf16 = _probe()[(codec, m, route)]
    assert f32["error"] is None and bf16["error"] is None, (f32, bf16)
    assert f32["digest"] == bf16["digest"]
    assert f32["kernels"] == bf16["kernels"]
    for name, _ in f32["kernels"]:
        assert "_bfloat16_t_" in name, name


def _weights(codec, rows, cols, experts=None):
    shape = (rows, cols) if experts is None else (experts, rows, cols)
    wf = mx.random.normal(shape, key=mx.random.key(1)) * 0.1
    w, s = kq.quantize(wf, codec)
    mx.eval(w, s)
    return w, s


def _x(shape):
    return mx.random.normal(shape, key=mx.random.key(2)) * 0.5


def _check(f, x):
    got = f(x)
    want = f(x.astype(mx.bfloat16))
    mx.eval(got, want)
    assert x.dtype == mx.float32
    assert got.dtype == mx.bfloat16, got.dtype
    assert mx.array_equal(got, want).item()


# transpose=False: qvm at M=1, qmm_n above it.
@pytest.mark.parametrize("m", [1, 64])
def test_quantized_matmul_n_float32(m):
    w, s = _weights("q4_k", K, N)
    _check(lambda x: kq.quantized_matmul(x, w, s, "q4_k", transpose=False), _x((m, K)))


def test_qmv_bias_float32():
    w, s = _weights("q8_0", N, K)
    bias = mx.random.normal((N,), key=mx.random.key(3)).astype(mx.float32)
    _check(lambda x: kq.quantized_matmul_qmv_bias(x, w, s, bias, "q8_0"), _x((1, K)))


# gather_qmm: the mat-vec (one row per expert pick), the tiled gather, the
# sorted per-expert GEMM and the transpose=False gather.
@pytest.mark.parametrize(
    "rows,m,transpose,sort",
    [
        (8, 1, True, False),
        (4, 32, True, False),
        (256, 1, True, True),
        (8, 1, False, False),
    ],
)
def test_gather_qmm_float32(rows, m, transpose, sort):
    w, s = _weights("q4_k", N, K, E) if transpose else _weights("q4_k", K, N, E)
    idx = np.random.default_rng(4).integers(0, E, rows).astype(np.uint32)
    if sort:
        idx = np.sort(idx)
    rhs = mx.array(idx)
    _check(
        lambda x: kq.gather_qmm(
            x, w, s, "q4_k", rhs_indices=rhs, transpose=transpose, sorted_indices=sort
        ),
        _x((rows, m, K)),
    )


def test_gather_qmm_seg_float32():
    w, s = _weights("q4_k", N, K, E)
    idx = np.sort(np.random.default_rng(5).integers(0, E, 200).astype(np.uint32))
    maps = kq.expert_tile_map(mx.array(idx), E)
    _check(lambda x: kq.gather_qmm_seg(x, w, s, "q4_k", *maps), _x((len(idx), K)))
