"""Weights that are views of a larger buffer.

A weight can be a dense slice of a larger buffer: a band of experts from a
stack, the up half of one expert of a fused gate and up stack, or the first
bytes of a buffer sized for a larger layer, reshaped. It is row-major at an
offset, so the ops read it in place, evaluated or not, and allocate no copy
of it. A weight that is not row-major, such as the gate half of a fused gate
and up tensor, is copied where the kernel takes no strides, and gives the
same result as the half in its own buffer.

The sorted gather, skinny_matmul, add_rmsnorm and kda_conv have CPU evals,
so the suite covers them on the GPU and, under KQUANT_FORCE_CPU, on the CPU
stream. gather_qmm_seg and the hc ops are Metal-only, and their cases are
skipped there.
"""

from __future__ import annotations

import gc
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


def _views(big, fused):
    """Dense slices of a larger buffer: the first E experts of big as an
    axis-0 slice and as a reshaped prefix, a middle band of E experts, and
    the up half of one expert of a fused gate and up stack."""
    flat = big.reshape(-1)[: E * big.shape[1] * big.shape[2]]
    return {
        "axis0": big[:E],
        "prefix": flat.reshape(E, big.shape[1], big.shape[2]),
        "band": big[3 : 3 + E],
        "fused_up": fused[2:3, N:],
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


def _routing(experts=E, seed=1):
    rng = np.random.default_rng(seed)
    idx = mx.array(np.sort(rng.integers(0, experts, ROWS)).astype(np.uint32))
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
    tile_map, counts = kq.expert_tile_map(idx, w.shape[0], tile_rows=64)
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
@pytest.mark.parametrize("form", ("axis0", "prefix", "band", "fused_up"))
@pytest.mark.parametrize("state", ("lazy", "evaluated"))
def test_dense_slice_weight_is_read_in_place(op, form, state):
    """A sorted gather and the segment GEMM read a dense slice of a larger
    expert stack in place: the eval adds its output and well under one copy
    of the slice, and the output is bit-equal to the slice in its own
    buffer."""
    big = _stack(P)
    fused = _stack(4, n=2 * N, seed=2)
    # want comes from a separate slice, so a lazy view reaches the op
    # unevaluated.
    want_view = _views(big, fused)[form]
    x, idx = _routing(want_view.shape[0])
    want = op(_own(want_view), x, idx)
    mx.eval(want)
    view = _views(big, fused)[form]
    if state == "evaluated":
        mx.eval(view)
    got, grew = _eval_growth(lambda: op(view, x, idx))
    extra = grew - got.nbytes
    assert extra < view.nbytes // 4, f"eval added {extra} B for a {view.nbytes} B slice"
    assert _same(got, want)


@pytest.mark.parametrize("op", OPS.values())
@pytest.mark.parametrize("state", ("lazy", "evaluated"))
def test_slice_keeps_its_dropped_parent_alive(op, state):
    """With the parent and the slice dropped before the eval, and buffers of
    the parent's size filled with a constant, which take the parent's buffer
    if it was freed, the op still reads the slice's bytes."""
    x, idx = _routing()
    parent = _stack(P, seed=7)
    want = op(_own(parent[3 : 3 + E]), x, idx)
    mx.eval(want)
    view = parent[3 : 3 + E]
    if state == "evaluated":
        mx.eval(view)
    y = op(view, x, idx)
    size = parent.nbytes
    del parent, view
    gc.collect()
    junk = [mx.full((size,), 165, mx.uint8) for _ in range(4)]
    mx.eval(junk)
    del junk
    mx.eval(y)
    assert _same(y, want)


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


D = 1024
HC, MIX = 4, 24
SINKHORN = {"iters": 20, "hc_eps": 1e-6, "norm_eps": 1e-6}


def _rand(shape, dt, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    a = mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(dt)
    mx.eval(a)
    return a


def _hc_inputs():
    x = _rand((1, 1, HC, D), mx.float16, 1, 0.05)
    fn = _rand((MIX, HC * D), mx.float32, 2, 0.02)
    scale = mx.array([1.1, 0.9, 1.3], dtype=mx.float32)
    base = _rand((MIX,), mx.float32, 3, 0.1)
    mr, ssq = kq.hc_front_reduce(x, fn)
    mx.eval(scale, mr, ssq)
    return x, fn, scale, base, mr, ssq


def _hc_sinkhorn(w):
    x, _, scale, base, mr, ssq = _hc_inputs()
    return kq.hc_sinkhorn_collapse(x, mr, ssq, scale, base, w, **SINKHORN)


def _hc_front(w):
    x, fn, scale, base, mr, ssq = _hc_inputs()
    w0 = _rand((D,), mx.float16, 5, 0.1) + 1.0
    col, post, comb = kq.hc_sinkhorn_collapse(x, mr, ssq, scale, base, w0, **SINKHORN)
    mx.eval(col, post, comb)
    return kq.hc_front_expand_collapse(
        col, x, post, comb, fn, scale, base, w, **SINKHORN
    )


def _add_rmsnorm(w):
    h = _rand((8, D), mx.float16, 6)
    r = _rand((8, D), mx.float16, 7)
    return [kq.add_rmsnorm(h, r, w, 1e-6)]


def _kda_conv(w):
    x = _rand((1, 16, D), mx.float16, 8)
    state = _rand((1, 3, D), mx.float16, 9)
    return list(kq.kda_conv(x, state, w, 128, 0.5, 1e-6))


SMALL_OPS = {
    "hc_sinkhorn_collapse": pytest.param(
        _hc_sinkhorn, (3 * D,), id="hc_sinkhorn_collapse", marks=metal_only
    ),
    "hc_front_expand_collapse": pytest.param(
        _hc_front, (3 * D,), id="hc_front_expand_collapse", marks=metal_only
    ),
    "add_rmsnorm": pytest.param(_add_rmsnorm, (3 * D,), id="add_rmsnorm"),
    "kda_conv": pytest.param(_kda_conv, (3 * D, 4), id="kda_conv"),
}


@pytest.mark.parametrize(("op", "parent_shape"), SMALL_OPS.values())
@pytest.mark.parametrize("state", ("lazy", "evaluated"))
def test_weight_at_an_offset_is_read_from_its_offset(op, parent_shape, state):
    """The hc collapse ops, add_rmsnorm (whose weight path the other fused
    norms share) and kda_conv read a weight sliced D rows into a larger
    buffer from its offset: the result is bit-equal to the weight in its own
    buffer."""
    parent = _rand(parent_shape, mx.float16, 4, 0.5)
    want = op(_own(parent[D : 2 * D]))
    mx.eval(want)
    view = parent[D : 2 * D]
    if state == "evaluated":
        mx.eval(view)
    got = op(view)
    mx.eval(got)
    assert all(_same(g, w) for g, w in zip(got, want, strict=True))
