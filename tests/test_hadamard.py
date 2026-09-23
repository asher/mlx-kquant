"""hadamard_rotate and its fused forms vs an explicit float64
Walsh-Hadamard matrix.

The oracle is the popcount-parity matrix H[i, j] = (-1)^popcount(i & j)
/ sqrt(n) applied per block in numpy, never mx.hadamard_transform, so a
shared butterfly-order defect cannot cancel. The kernel's f32 math with
one rounding at the store puts f32 within a few ulps of the oracle; the
16-bit dtypes sit at their own rounding. The fused glu_hadamard is
checked against the same oracle and against the unfused chain it
replaces.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

BLOCKS = (256, 512, 1024, 2048, 4096)
DTYPES = (mx.float32, mx.bfloat16, mx.float16)
TOL = {mx.float32: 2e-5, mx.float16: 5e-3, mx.bfloat16: 3e-2}


def _hadamard(n):
    idx = np.arange(n, dtype=np.uint16)
    both = (idx[:, None] & idx[None, :]).reshape(n, n, 1).view(np.uint8)
    parity = np.unpackbits(both, axis=-1).sum(axis=-1) & 1
    return np.where(parity == 1, -1.0, 1.0) / np.sqrt(n)


def _rng(seed=3):
    return np.random.default_rng(seed)


def _signs(rng, k):
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=k)


def _reference(x, block, signs=None, perm=None):
    xf = np.asarray(x, dtype=np.float64)
    if perm is not None:
        rep, nk, hd = perm
        lead = xf.shape[:-1]
        xf = xf.reshape(*lead, rep, nk, hd).swapaxes(-3, -2).reshape(*lead, -1)
    if signs is not None:
        xf = xf * signs.astype(np.float64)
    lead = xf.shape[:-1]
    out = xf.reshape(*lead, -1, block) @ _hadamard(block)
    return out.reshape(*lead, -1)


def _check(got, ref, dtype):
    g = np.array(got.astype(mx.float32)).astype(np.float64)
    scale = max(float(np.abs(ref).max()), 1e-6)
    assert float(np.abs(g - ref).max()) / scale < TOL[dtype]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block", BLOCKS)
def test_rotate_matches_matrix(block, dtype):
    rng = _rng()
    rows, k = 5, 3 * block
    x = mx.array(rng.standard_normal((rows, k)).astype(np.float32)).astype(dtype)
    signs = _signs(rng, k)
    got = kq.hadamard_rotate(x, mx.array(signs), block=block)
    mx.eval(got)
    assert got.shape == x.shape and got.dtype == dtype
    _check(got, _reference(np.array(x.astype(mx.float32)), block, signs), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rotate_without_signs_is_self_inverse(dtype):
    rng = _rng(5)
    block, k = 1024, 2048
    x = mx.array(rng.standard_normal((2, 3, k)).astype(np.float32)).astype(dtype)
    once = kq.hadamard_rotate(x, None, block=block)
    twice = kq.hadamard_rotate(once, block=block)
    mx.eval(once, twice)
    _check(once, _reference(np.array(x.astype(mx.float32)), block), dtype)
    _check(twice, np.array(x.astype(mx.float32)).astype(np.float64), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rotate_with_grouped_head_permute(dtype):
    rng = _rng(7)
    rep, nk, hd = 3, 16, 128
    k = rep * nk * hd
    x = mx.array(rng.standard_normal((4, k)).astype(np.float32)).astype(dtype)
    signs = _signs(rng, k)
    got = kq.hadamard_rotate(x, mx.array(signs), block=1024, perm=(rep, nk, hd))
    mx.eval(got)
    ref = _reference(np.array(x.astype(mx.float32)), 1024, signs, (rep, nk, hd))
    _check(got, ref, dtype)


def test_rotate_matches_mlx_ops_form():
    rng = _rng(9)
    block, k = 1024, 5120
    x = mx.array(rng.standard_normal((7, k)).astype(np.float32))
    signs = mx.array(_signs(rng, k))
    got = kq.hadamard_rotate(x, signs, block=block)
    ops = mx.hadamard_transform(
        (x * signs).reshape(7, -1, block), scale=block**-0.5
    ).reshape(7, k)
    mx.eval(got, ops)
    assert float(mx.abs(got - ops).max()) < 2e-5


def test_rotate_noncontiguous_input():
    rng = _rng(11)
    block = 256
    base = mx.array(rng.standard_normal((4, 2 * block + 8)).astype(np.float32))
    x = base[:, 8:]
    got = kq.hadamard_rotate(x, None, block=block)
    mx.eval(got)
    _check(got, _reference(np.array(x), block), mx.float32)


@pytest.mark.parametrize(
    "call",
    [
        lambda x: kq.hadamard_rotate(x, None, block=300),
        lambda x: kq.hadamard_rotate(x, None, block=4096),
        lambda x: kq.hadamard_rotate(x, mx.ones((256,)), block=1024),
        lambda x: kq.hadamard_rotate(x, mx.ones((2048,), dtype=mx.float16), block=1024),
        lambda x: kq.hadamard_rotate(x, None, block=1024, perm=(2, 2, 256)),
        lambda x: kq.hadamard_rotate(x.astype(mx.int32), None, block=1024),
    ],
)
def test_rotate_rejects_malformed(call):
    x = mx.zeros((2, 2048), dtype=mx.float32)
    with pytest.raises((ValueError, TypeError)):
        mx.eval(call(x))


# Row widths of the Hadamard-folded Qwen3.8-27B projections: hidden, the
# attention output, the MLP intermediate.
WIDTHS = (5120, 6144, 17408)
ROWS = (1, 7, 64)


def _glu_reference(x, g, activation):
    sig = 1.0 / (1.0 + np.exp(-g))
    return (g * sig if activation == "silu" else sig) * x


@pytest.mark.parametrize("activation", ("silu", "sigmoid"))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("k", WIDTHS)
def test_glu_hadamard_matches_matrix(k, rows, dtype, activation):
    rng = _rng(23)
    block = 1024
    x = mx.array(rng.standard_normal((rows, k)).astype(np.float32)).astype(dtype)
    g = mx.array(4 * rng.standard_normal((rows, k)).astype(np.float32)).astype(dtype)
    signs = _signs(rng, k)
    got = kq.glu_hadamard(x, g, mx.array(signs), block=block, activation=activation)
    mx.eval(got)
    assert got.shape == x.shape and got.dtype == dtype
    xf, gf = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (x, g))
    _check(got, _reference(_glu_reference(xf, gf, activation), block, signs), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_glu_hadamard_matches_the_unfused_chain(dtype):
    rng = _rng(29)
    block, k = 1024, 17408
    x = mx.array(rng.standard_normal((2, k)).astype(np.float32)).astype(dtype)
    g = mx.array(4 * rng.standard_normal((2, k)).astype(np.float32)).astype(dtype)
    signs = mx.array(_signs(rng, k))
    got = kq.glu_hadamard(x, g, signs, block=block)
    chain = kq.hadamard_rotate(g * mx.sigmoid(g) * x, signs, block=block)
    gate_got = kq.glu_hadamard(x, g, signs, block=block, activation="sigmoid")
    gate_chain = kq.hadamard_rotate(x * mx.sigmoid(g), signs, block=block)
    mx.eval(got, chain, gate_got, gate_chain)
    for a, b in ((got, chain), (gate_got, gate_chain)):
        _check(a, np.array(b.astype(mx.float32)).astype(np.float64), dtype)


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the GPU and the CPU")
def test_glu_hadamard_agrees_across_devices():
    rng = _rng(31)
    block, k = 1024, 5120
    x = mx.array(rng.standard_normal((3, k)).astype(np.float32)).astype(mx.bfloat16)
    g = mx.array(rng.standard_normal((3, k)).astype(np.float32)).astype(mx.bfloat16)
    signs = mx.array(_signs(rng, k))
    outs = {}
    for dev in (mx.gpu, mx.cpu):
        with mx.stream(dev):
            outs[dev] = kq.glu_hadamard(x, g, signs, block=block)
            mx.eval(outs[dev])
    a, b = outs[mx.gpu], outs[mx.cpu]
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
    assert diff <= 2**-6 * max(mx.abs(b.astype(mx.float32)).max().item(), 1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda x: kq.glu_hadamard(x, x, block=300),
        lambda x: kq.glu_hadamard(x, x, mx.ones((1024,)), block=1024),
        lambda x: kq.glu_hadamard(x, x.astype(mx.float16), block=1024),
        lambda x: kq.glu_hadamard(x, x[:, :1024], block=1024),
        lambda x: kq.glu_hadamard(x, x, block=1024, activation="gelu"),
    ],
)
def test_glu_hadamard_rejects_malformed(call):
    x = mx.zeros((2, 2048), dtype=mx.float32)
    with pytest.raises((ValueError, TypeError)):
        mx.eval(call(x))
