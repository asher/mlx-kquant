"""quantized_matmul_multi against quantized_matmul per weight.

On a NAX GPU the op runs one verify_nax dispatch per four weights at 2 to
8 rows, where quantized_matmul would run verify_nax on the widest weight.
With the split count forced to one value, the verify_nax entry at M 2 and
the per-row qmv route off, quantized_matmul per weight runs the same
kernel body and fold, so the outputs match bit for bit. At the default
levers the joint grid can take another split count than each weight
alone, so the outputs are checked
against a float64 reference built from kq.dequantize weights. Every other
case runs quantized_matmul per weight, which the CPU stream checks.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from qmm_probe import force_cpu

import mlx_kquant as kq

DTYPES = [mx.bfloat16, mx.float16]
R1 = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
# Gate of quantized_matmul on the CPU stream, as in test_cpu_neon.
CPU_REL = 5e-2
# Codecs with a verify_nax kernel, and an inner dim each serves.
CODECS = {
    "q4_0": 512,
    "q8_0": 512,
    "iq4_nl": 512,
    "q4_k": 512,
    "q5_k": 512,
    "q6_k": 512,
    "q3_k": 512,
    "q2_k": 512,
    "iq4_xs": 512,
}
# Row counts of the joint grid's weights: tiny, partial row tiles, a
# 32-row remainder and a weight wider than the rest.
NS = [48, 160, 1000, 300]

_W = {}


def _weight(codec, n, k, seed):
    key = (codec, n, k, seed)
    if key not in _W:
        wf = mx.random.normal((n, k), key=mx.random.key(97 * seed + n + k)) * 0.1
        w, s = kq.quantize(wf, codec)
        mx.eval(w, s)
        wd = np.array(kq.dequantize(w, s, codec, dtype=mx.float32)).astype(np.float64)
        _W[key] = (w, wd)
    return _W[key]


def _weights(codec, ns, k):
    return [_weight(codec, n, k, i) for i, n in enumerate(ns)]


def _x(m, k, dtype, seed=0):
    return mx.random.normal((m, k), key=mx.random.key(seed + 7 * m)).astype(dtype)


def _per_weight(x, ws, codec):
    sc = mx.zeros((1,), mx.uint8)
    return [kq.quantized_matmul(x, w, sc, codec, transpose=True) for w in ws]


@pytest.fixture
def exact_levers(monkeypatch):
    """One split count, verify_nax from M 2 and no per-row qmv route, so
    quantized_matmul runs verify_nax on every weight the op joins."""

    def set_splits(n):
        monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", str(n))

    monkeypatch.setenv("KQ_NAX_QMV", "0")
    monkeypatch.setenv("KQ_VERIFY_NAX", "2")
    return set_splits


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", sorted(CODECS))
@pytest.mark.parametrize("splits", [1, 2])
def test_matches_per_weight_bits(codec, dtype, splits, exact_levers):
    exact_levers(splits)
    k = CODECS[codec]
    ws = [w for w, _ in _weights(codec, NS, k)]
    for m in range(1, 9):
        x = _x(m, k, dtype)
        got = kq.quantized_matmul_multi(x, ws, codec)
        want = _per_weight(x, ws, codec)
        mx.eval(got, want)
        assert len(got) == len(ws)
        for g, w in zip(got, want, strict=True):
            assert g.dtype == w.dtype and g.shape == w.shape
            assert np.array_equal(
                np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
            ), f"{codec} M={m} N={w.shape[-1]}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", ["q4_0", "q4_k", "q6_k"])
def test_reference_default_levers(codec, dtype):
    k = CODECS[codec]
    pairs = _weights(codec, NS, k)
    ws = [w for w, _ in pairs]
    for m in (1, 2, 3, 4, 6, 8, 9, 16):
        x = _x(m, k, dtype, seed=3)
        got = kq.quantized_matmul_multi(x, ws, codec)
        mx.eval(got)
        xd = np.array(x.astype(mx.float32)).astype(np.float64)
        for g, (_, wd) in zip(got, pairs, strict=True):
            ref = xd @ wd.T
            err = np.linalg.norm(np.array(g.astype(mx.float32)) - ref)
            rel = CPU_REL if force_cpu() else R1[dtype]
            assert err <= rel * np.linalg.norm(ref), f"{codec} M={m}"


def test_q4_0_two_block_inner_dim(exact_levers):
    # K not a multiple of 256 runs the two-block q4_0 kernel.
    exact_levers(1)
    ws = [w for w, _ in _weights("q4_0", [64, 200], 320)]
    for m in (3, 5, 8):
        x = _x(m, 320, mx.bfloat16)
        got = kq.quantized_matmul_multi(x, ws, "q4_0")
        want = _per_weight(x, ws, "q4_0")
        mx.eval(got, want)
        for g, w in zip(got, want, strict=True):
            assert np.array_equal(
                np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
            )


@pytest.mark.parametrize("count", [1, 4, 5, 9])
def test_weight_counts(count, exact_levers):
    # More than four weights run one dispatch per four.
    exact_levers(1)
    ns = [64 + 32 * i for i in range(count)]
    ws = [w for w, _ in _weights("q4_0", ns, 512)]
    x = _x(6, 512, mx.bfloat16)
    got = kq.quantized_matmul_multi(x, ws, "q4_0")
    want = _per_weight(x, ws, "q4_0")
    mx.eval(got, want)
    for g, w in zip(got, want, strict=True):
        assert np.array_equal(
            np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
        )


def test_batch_dims_views_and_float32(exact_levers):
    exact_levers(1)
    big, _ = _weight("q4_0", 512, 512, 11)
    ws = [big[100:228], big[300:348], _weight("q4_0", 96, 512, 12)[0]]
    x = _x(6, 512, mx.float32).reshape(2, 3, 512)
    got = kq.quantized_matmul_multi(x, ws, "q4_0")
    want = _per_weight(x, ws, "q4_0")
    mx.eval(got, want)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape and g.dtype == mx.bfloat16
        assert np.array_equal(
            np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
        )


def test_empty_weight(exact_levers):
    exact_levers(1)
    ws = [_weight("q4_0", 128, 512, 0)[0], mx.zeros((0, 288), mx.uint8)]
    x = _x(4, 512, mx.bfloat16)
    got = kq.quantized_matmul_multi(x, ws, "q4_0")
    mx.eval(got)
    assert got[1].shape == (4, 0)
    want = _per_weight(x, ws[:1], "q4_0")[0]
    assert np.array_equal(
        np.array(got[0].astype(mx.float32)), np.array(want.astype(mx.float32))
    )


def test_lever_off(monkeypatch):
    monkeypatch.setenv("KQ_MATMUL_MULTI", "0")
    ws = [w for w, _ in _weights("q4_0", NS, 512)]
    x = _x(6, 512, mx.bfloat16)
    got = kq.quantized_matmul_multi(x, ws, "q4_0")
    want = _per_weight(x, ws, "q4_0")
    mx.eval(got, want)
    for g, w in zip(got, want, strict=True):
        assert np.array_equal(
            np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
        )


def test_grad_matches_per_weight():
    ws = [w for w, _ in _weights("q4_0", NS, 512)]
    x = _x(6, 512, mx.float32)
    cot = [mx.random.normal((6, n), key=mx.random.key(n)) for n in NS]

    def f_multi(x):
        return sum(
            (y.astype(mx.float32) * c).sum()
            for y, c in zip(kq.quantized_matmul_multi(x, ws, "q4_0"), cot, strict=True)
        )

    def f_each(x):
        return sum(
            (y.astype(mx.float32) * c).sum()
            for y, c in zip(_per_weight(x, ws, "q4_0"), cot, strict=True)
        )

    g1 = mx.grad(f_multi)(x)
    g2 = mx.grad(f_each)(x)
    mx.eval(g1, g2)
    assert np.allclose(np.array(g1), np.array(g2), rtol=2e-2, atol=2e-2)


def test_argument_errors():
    w, _ = _weight("q4_0", 64, 512, 0)
    x = _x(4, 512, mx.bfloat16)
    with pytest.raises(ValueError):
        kq.quantized_matmul_multi(x, [], "q4_0")
    with pytest.raises(ValueError):
        kq.quantized_matmul_multi(x, [w, _weight("q4_0", 64, 256, 0)[0]], "q4_0")
    with pytest.raises(ValueError):
        kq.quantized_matmul_multi(_x(4, 256, mx.bfloat16), [w], "q4_0")
    with pytest.raises(ValueError):
        kq.quantized_matmul_multi(x, [w], "no_such_codec")


_LOG_CHILD = r"""
import os, sys
import mlx.core as mx
import mlx_kquant as kq
ns = (4096, 640, 48, 48)
w = [kq.quantize(mx.random.normal((n, 512)) * 0.1, "q4_0")[0] for n in ns]
x = mx.random.normal((6, 512)).astype(mx.bfloat16)
mx.eval(kq.quantized_matmul_multi(x, w, "q4_0"))
"""


@pytest.mark.skipif(force_cpu() or not kq.nax_available(), reason="NAX GPU route")
def test_one_dispatch_on_nax():
    env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    env["KQ_QMM_LOG"] = "1"
    env["KQ_VERIFY_NAX_SPLITS"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", _LOG_CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stderr.splitlines() if ln.startswith("[kq qmm]")]
    assert len(lines) == 1 and "verify_nax_multi" in lines[0], r.stderr
