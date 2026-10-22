"""Split-K gathered mat-vec at one activation row.

kq.gather_qmm with transpose at M 1 runs each batch entry through the dense
split-K mat-vec of its expert when the codec is q8_0 or q6_k, the tile's row
count divides N (2 for q8_0, 4 for q6_k), 256 divides K, K is 2048 or more
and the call has fewer than 16384 output rows. KQ_GATHER_SPLITK=1 takes it
at any K and row count that fit, and 0 keeps gather_qmv. An entry then
matches kq.quantized_matmul on its expert bit for bit. Against gather_qmv the
outputs differ only in float32 summation order: by one unit in the last
place, or near zero where the sum cancels.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or mx.default_device() != mx.gpu,
    reason="Metal kernel routing",
)

LEVER = "KQ_GATHER_SPLITK"
TRIM = "KQ_QMV_SPLITK_TRIM"
RPS = {"q8_0": 2, "q6_k": 4}
MANTISSA = {mx.float32: 7, mx.float16: 10, mx.bfloat16: 7}
# Each case: N, K, x shape, lhs (None takes the default), rhs, experts. The
# first has a per-entry x as an MLA or grouped output projection runs it, the
# second one shared x and repeated experts as a SwitchGLU decode step runs
# it, the third two batch dimensions, a permuted lhs with a repeat and a K
# whose 256-weight steps do not divide evenly between the simdgroups.
CASES = {
    "per_entry": (256, 4096, (1, 8, 1, 4096), [list(range(8))], [list(range(8))], 8),
    "shared_x": (260, 2048, (1, 1, 1, 2048), None, [[3, 0, 3, 5, 1, 2]], 6),
    "batch_2d": (
        128,
        5376,
        (2, 3, 1, 5376),
        [[5, 0, 0], [2, 4, 1]],
        [[1, 2, 0], [2, 2, 1]],
        3,
    ),
}

_W = {}


def _experts(codec, e, n, k):
    key = (codec, e, n, k)
    if key not in _W:
        rng = np.random.default_rng(n + k)
        ws = []
        for _ in range(e):
            wf = (rng.standard_normal((n, k)) * 0.1).astype(np.float32)
            wq, _ = kq.quantize(mx.array(wf), codec)
            ws.append(wq)
        w = mx.stack(ws)
        mx.eval(w)
        _W[key] = w
    return _W[key]


def _gather(x, w, codec, lhs, rhs, lever=None, transpose=True):
    if lever is None:
        os.environ.pop(LEVER, None)
    else:
        os.environ[LEVER] = lever
    y = kq.gather_qmm(
        x,
        w,
        mx.zeros((1,), dtype=mx.uint8),
        codec,
        lhs_indices=lhs,
        rhs_indices=rhs,
        transpose=transpose,
    )
    mx.eval(y)
    return y


@pytest.fixture(autouse=True)
def _lever(monkeypatch):
    monkeypatch.delenv(LEVER, raising=False)
    monkeypatch.delenv(TRIM, raising=False)
    yield
    os.environ.pop(LEVER, None)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("codec", list(RPS))
def test_matches_quantized_matmul_bits(codec, case, dtype):
    n, k, xs, lhs_l, rhs_l, e = CASES[case]
    w = _experts(codec, e, n, k)
    rng = np.random.default_rng(3)
    x = mx.array((rng.standard_normal(xs) * 0.5).astype(np.float32)).astype(dtype)
    lhs = None if lhs_l is None else mx.array(np.array(lhs_l, np.uint32))
    rhs = mx.array(np.array(rhs_l, np.uint32))
    got = _gather(x, w, codec, lhs, rhs, lever="1")
    base = _gather(x, w, codec, lhs, rhs, lever="0")
    rows = x.reshape(-1, 1, k)
    rhs_f = np.array(rhs_l, np.int64).ravel()
    if lhs_l is None:
        lhs_f = np.broadcast_to(
            np.arange(rows.shape[0]).reshape(xs[:-2]), np.array(rhs_l).shape
        ).ravel()
    else:
        lhs_f = np.array(lhs_l, np.int64).ravel()
    sc = mx.zeros((1,), dtype=mx.uint8)
    ref = mx.stack(
        [
            kq.quantized_matmul(rows[int(i)], w[int(j)], sc, codec, transpose=True)
            for i, j in zip(lhs_f, rhs_f, strict=True)
        ]
    ).reshape(got.shape)
    assert got.dtype == ref.dtype
    assert bool(mx.array_equal(got, ref).item()), f"{codec} {case} {dtype}"
    a = np.array(got.astype(mx.float32))
    b = np.array(base.astype(mx.float32))
    ulp = np.maximum(np.abs(a), np.abs(b)) * 2.0 ** -MANTISSA[dtype]
    assert np.all(np.abs(a - b) <= ulp + 1e-5 * float(np.abs(b).max()))


def _q8_0_block(q, d):
    d16 = np.array([d], np.float16).view(np.uint8)
    return np.concatenate([d16, np.asarray(q, np.int8).view(np.uint8)])


def _q6_k_block(ql, qh, scale, d):
    return np.concatenate(
        [
            np.full(128, ql, np.uint8),
            np.full(64, qh, np.uint8),
            np.full(16, scale, np.int8).view(np.uint8),
            np.array([d], np.float16).view(np.uint8),
        ]
    )


# Rows whose reduction order shows which kernel ran (x all ones), as in the
# dense routing tests of test_matmul_synth. q8_0: lane 0 of the first
# 256-weight step gives +B (B = 32768 * 1016), lane 1 of the second -B and
# lane 0 of the third 0.5; gather_qmv adds 0.5 to +B inside lane 0 and
# returns 0, split-K gives each step its own simdgroup and returns 0.5.
# q6_k: superblocks 0 and 2 give +B and -B (B = 3937 * 2**23) and superblock
# 1 gives 256; gather_qmv cancels +B and -B inside a lane and returns 256,
# split-K sums superblocks 0 and 1 in one simdgroup and returns 0. The rest
# of the row is zero blocks.
def _row(codec, k):
    if codec == "q8_0":
        blocks = [_q8_0_block(np.zeros(32), 0.0)] * (k // 32)
        blocks[0] = _q8_0_block([127] * 8 + [0] * 24, 32768.0)
        blocks[8] = _q8_0_block([0] * 8 + [-127] * 8 + [0] * 16, 32768.0)
        blocks[16] = _q8_0_block([1] + [0] * 31, 0.5)
        return np.concatenate(blocks)
    blocks = [_q6_k_block(0x00, 0x00, 0, 0.0)] * (k // 256)
    blocks[0] = _q6_k_block(0xFF, 0xFF, 127, 32768.0)
    blocks[1] = _q6_k_block(0x11, 0xAA, 1, 1.0)
    blocks[2] = _q6_k_block(0x11, 0x00, 127, 32768.0)
    return np.concatenate(blocks)


SPLIT = {"q8_0": 0.5, "q6_k": 0.0}
PER_ROW = {"q8_0": 0.0, "q6_k": 256.0}


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("codec", list(RPS))
def test_routing(codec, dtype):
    def run(n, k, lever=None, m=1, b=2):
        w = mx.array(np.tile(_row(codec, k), (2, n, 1)))
        x = mx.ones((1, b, m, k), dtype=dtype)
        lhs = mx.array(np.arange(b, dtype=np.uint32).reshape(1, b))
        rhs = mx.array((np.arange(b, dtype=np.uint32) % 2).reshape(1, b))
        y = _gather(x, w, codec, lhs, rhs, lever=lever)
        return sorted(set(np.array(y.astype(mx.float32)).ravel().tolist()))

    n = 2 * RPS[codec]
    split, per_row = [SPLIT[codec]], [PER_ROW[codec]]
    assert run(n, 2048) == split
    assert run(n, 1792) == per_row
    assert run(n, 2048, lever="0") == per_row
    assert run(n, 1024, lever="1") == split
    # 15360 and 16384 output rows.
    assert run(1024, 2048, b=15) == split
    assert run(1024, 2048, b=16) == per_row
    assert run(1024, 4096, b=16) == per_row
    assert run(1024, 2048, b=16, lever="1") == split
    # N off the tile, and two activation rows, keep gather_qmv.
    assert run(n + RPS[codec] // 2, 2048, lever="1") == per_row
    assert run(n, 2048, lever="1", m=2) == per_row


# q8_0 at K 2080: 256 does not divide K, so the split-K tile would step past
# the row. Forced, the call keeps gather_qmv.
def test_q8_0_k_tail_keeps_gather_qmv():
    rng = np.random.default_rng(5)
    w = _experts("q8_0", 2, 8, 2080)
    x = mx.array(rng.standard_normal((1, 2, 1, 2080)).astype(np.float32)).astype(
        mx.bfloat16
    )
    lhs = mx.array(np.array([[0, 1]], np.uint32))
    rhs = mx.array(np.array([[1, 0]], np.uint32))
    a = _gather(x, w, "q8_0", lhs, rhs, lever="1")
    b = _gather(x, w, "q8_0", lhs, rhs, lever="0")
    assert bool(mx.array_equal(a, b).item())


# Below eight 256-weight steps (q8_0) or superblock pairs (q6_k) the tile
# dispatches one simdgroup per step or pair, as the dense mat-vec does, and
# the output matches the full threadgroup (KQ_QMV_SPLITK_TRIM=0) bit for bit.
@pytest.mark.parametrize("k", [1280, 1792, 2048])
@pytest.mark.parametrize("codec", list(RPS))
def test_trimmed_tile_matches_full(codec, k, monkeypatch):
    rng = np.random.default_rng(8)
    w = _experts(codec, 4, 64, k)
    x = mx.array(rng.standard_normal((1, 6, 1, k)).astype(np.float32)).astype(
        mx.bfloat16
    )
    lhs = mx.array(np.arange(6, dtype=np.uint32).reshape(1, 6))
    rhs = mx.array(np.array([[2, 0, 3, 3, 1, 0]], np.uint32))
    a = _gather(x, w, codec, lhs, rhs, lever="1")
    monkeypatch.setenv(TRIM, "0")
    b = _gather(x, w, codec, lhs, rhs, lever="1")
    assert bool(mx.array_equal(a, b).item()), f"{codec} K={k}"


# Codecs without a split-K mat-vec, and the transpose=False gather, keep
# their kernels under the forcing lever.
@pytest.mark.parametrize("codec", ["q4_k", "q5_k", "iq4_xs"])
def test_other_codecs_keep_gather_qmv(codec):
    rng = np.random.default_rng(6)
    w = _experts(codec, 2, 8, 2048)
    x = mx.array(rng.standard_normal((1, 2, 1, 2048)).astype(np.float32)).astype(
        mx.bfloat16
    )
    lhs = mx.array(np.array([[0, 1]], np.uint32))
    rhs = mx.array(np.array([[1, 0]], np.uint32))
    a = _gather(x, w, codec, lhs, rhs, lever="1")
    b = _gather(x, w, codec, lhs, rhs, lever="0")
    assert bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("codec", list(RPS))
def test_untransposed_keeps_its_kernel(codec):
    # x @ W[e] contracts over W's 256 rows of 2048 weights: as a split-K
    # mat-vec that would read 256-weight rows and return other values.
    rng = np.random.default_rng(7)
    w = _experts(codec, 2, 256, 2048)
    x = mx.array(rng.standard_normal((1, 2, 1, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    lhs = mx.array(np.array([[0, 1]], np.uint32))
    rhs = mx.array(np.array([[1, 0]], np.uint32))
    a = _gather(x, w, codec, lhs, rhs, lever="1", transpose=False)
    b = _gather(x, w, codec, lhs, rhs, lever="0", transpose=False)
    assert a.shape == (1, 2, 1, 2048)
    assert bool(mx.array_equal(a, b).item())
