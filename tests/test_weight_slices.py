"""Weights that are views of a larger buffer.

A weight can be a dense slice of a larger buffer: a band of experts from a
stack, or the first bytes of a buffer sized for a larger layer, reshaped. It
is row-major at an offset, so the ops read it in place, evaluated or not,
and allocate no copy of it. A weight that is not row-major, such as the gate
half of a fused gate and up tensor, is copied where the kernel takes no
strides, and gives the same result as the half in its own buffer.

The sorted gather and skinny_matmul have CPU evals, so the suite covers them
on the GPU and, under KQUANT_FORCE_CPU, on the CPU stream. gather_qmm_seg is
a Metal-only kernel, and its cases are skipped there.
"""

from __future__ import annotations

import os
import time

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

metal_only = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="Metal-only kernel; no CPU path.",
)

CODEC = "q4_k"
E, P, N, K = 12, 16, 256, 512  # E experts of a P-expert buffer
ROWS = 64


def _stack(experts, n=N, seed=0):
    rng = np.random.default_rng(seed)
    wf = mx.array((rng.standard_normal((experts * n, K)) * 0.05).astype(np.float32))
    w = kq.quantize(wf, CODEC)[0].reshape(experts, n, -1)
    mx.eval(w)
    return w


def _own(view):
    return mx.array(np.ascontiguousarray(np.array(view)))


def _views(big):
    """The two dense forms of the first E experts of big."""
    flat = big.reshape(-1)[: E * big.shape[1] * big.shape[2]]
    return {
        "axis0": big[:E],
        "prefix": flat.reshape(E, big.shape[1], big.shape[2]),
    }


def _settle():
    # The buffers of the last command buffer are freed by its completion
    # handler, after eval returns.
    mx.synchronize()
    time.sleep(0.05)
    mx.clear_cache()


def _eval_growth(fn):
    """fn() evaluated, and the peak memory its eval adds, in bytes."""
    _settle()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    y = fn()
    mx.eval(y)
    return y, mx.get_peak_memory() - before


def _routing(seed=1):
    rng = np.random.default_rng(seed)
    idx = mx.array(np.sort(rng.integers(0, E, ROWS)).astype(np.uint32))
    x = mx.array((rng.standard_normal((ROWS, K)) * 0.1).astype(np.float16))
    mx.eval(idx, x)
    return x, idx


def _gather(w, x, idx):
    return kq.gather_qmm(
        x[:, None, :],
        w,
        mx.zeros((1,), mx.uint8),
        CODEC,
        rhs_indices=idx,
        transpose=True,
        sorted_indices=True,
    )


def _seg(w, x, idx):
    tile_map, counts = kq.expert_tile_map(idx, E, tile_rows=64)
    return kq.gather_qmm_seg(
        x, w, mx.zeros((1,), mx.uint8), CODEC, tile_map, counts, tile_rows=64
    )


def _same(a, b):
    f32 = [np.array(t.astype(mx.float32)) for t in (a, b)]
    return np.array_equal(*f32)


OPS = {
    "gather_qmm": pytest.param(_gather, id="gather_qmm"),
    "gather_qmm_seg": pytest.param(_seg, id="gather_qmm_seg", marks=metal_only),
}


@pytest.mark.parametrize("op", OPS.values())
@pytest.mark.parametrize("form", ("axis0", "prefix"))
@pytest.mark.parametrize("state", ("lazy", "evaluated"))
def test_dense_slice_weight_is_read_in_place(op, form, state):
    """A sorted gather and the segment GEMM read a dense slice of a larger
    expert stack in place: the eval adds well under one copy of the slice,
    and the output is bit-equal to the slice in its own buffer."""
    big = _stack(P)
    x, idx = _routing()
    want = op(_own(big[:E]), x, idx)
    mx.eval(want)
    view = _views(big)[form]
    if state == "evaluated":
        mx.eval(view)
    got, grew = _eval_growth(lambda: op(view, x, idx))
    slice_bytes = view.nbytes
    assert grew < slice_bytes // 4, f"eval grew {grew} B for a {slice_bytes} B slice"
    assert _same(got, want)


@pytest.mark.parametrize("op", OPS.values())
@pytest.mark.parametrize("state", ("lazy", "evaluated"))
def test_strided_half_weight_is_copied(op, state):
    """The gate half of a fused gate and up stack is strided across experts.
    It is copied for the kernels that take no strides, and gives the result
    of the half in its own buffer."""
    fused = _stack(E, n=2 * N, seed=2)
    x, idx = _routing()
    want = op(_own(fused[:, :N]), x, idx)
    half = fused[:, :N]
    if state == "evaluated":
        mx.eval(half)
    assert _same(op(half, x, idx), want)


@pytest.mark.parametrize("state", ("lazy", "evaluated"))
def test_skinny_matmul_reads_a_row_band_in_place(state):
    """A band of rows of a larger weight is row-major at an offset:
    skinny_matmul reads it in place, with the result of the band in its
    own buffer."""
    rng = np.random.default_rng(3)
    big = mx.array((rng.standard_normal((4 * N, 4096)) * 0.05).astype(np.float16))
    x = mx.array((rng.standard_normal((4, 4096)) * 0.1).astype(np.float16))
    mx.eval(big, x)
    want = kq.skinny_matmul(x, _own(big[N : 2 * N]))
    mx.eval(want)
    band = big[N : 2 * N]
    if state == "evaluated":
        mx.eval(band)
    got, grew = _eval_growth(lambda: kq.skinny_matmul(x, band))
    assert grew < band.nbytes // 4, f"eval grew {grew} B for a {band.nbytes} B band"
    assert _same(got, want)
