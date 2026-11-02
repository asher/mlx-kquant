"""vsa_attention: MiniMax-H3 video sparse attention over one packed sequence,
against its parts in FastVideo's order: MLX's fused attention for the prefix
query rows (every row at sparsity 0), vsa_block_sparse for the video query
rows, then the gate term per row, the coarse output of the row's tile times
the gate in float32, rounded to bfloat16 and added. The routing scores, pooled
q against pooled k, are float32 products on every device: where the GPU
multiplies float32 at TF32, the sum of three products of TF32-exact parts.

The module skips under KQUANT_FORCE_CPU and without Metal. The sparse calls
need tensor-op (NAX) hardware, and the dense call and the guards run on any
Metal GPU.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from mlx_kquant._ext import _vsa_gather_source
from test_operand_views import _nodes

import mlx_kquant as kq
from mlx_kquant import vsa

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="vsa_attention runs on Metal; no CPU path.",
)

NAX = (
    not os.environ.get("KQUANT_FORCE_CPU")
    and mx.metal.is_available()
    and kq.nax_available()
)

requires_nax = pytest.mark.skipif(
    not NAX, reason="a sparse call needs tensor-op (NAX) hardware."
)

big = pytest.mark.skipif(
    not os.environ.get("KQUANT_BIG_TESTS"),
    reason="holds a 4.3 GB array; set KQUANT_BIG_TESTS=1 to run",
)

TILE = 64
D = 128
SCALE = D**-0.5
# Text and audio rows, then a latent grid that leaves short edge tiles on
# every axis: 3 prefix tiles (56, 64, 10 rows) and 12 video tiles.
PREFIX = (56, 74)
GRID = (5, 8, 10)


def _fields(*segments):
    """The tile map of a packed sequence, as FastVideo builds it from its
    segments in packed order: the rows of the dense segments (ints) in
    64-row tiles, the last tile of each segment short, then the 4x4x4 tiles
    of the video grid (a tuple) in raster order, the edge tiles short."""
    dense, rows_of, row = [], [], 0
    for seg in segments:
        if isinstance(seg, tuple):
            t, h, w = seg
            video = np.arange(t * h * w).reshape(t, h, w) + row
            row += t * h * w
        else:
            dense.append(np.arange(row, row + seg))
            row += seg
    for rows in dense:
        rows_of += [rows[i : i + TILE] for i in range(0, len(rows), TILE)]
    n_prefix = len(rows_of)
    for a in range(0, t, 4):
        for b in range(0, h, 4):
            for c in range(0, w, 4):
                rows_of.append(video[a : a + 4, b : b + 4, c : c + 4].reshape(-1))
    gather = np.full(len(rows_of) * TILE, row, dtype=np.int32)
    for i, rows in enumerate(rows_of):
        gather[i * TILE : i * TILE + len(rows)] = rows
    slots = np.nonzero(gather < row)[0]
    untile = np.empty(row, dtype=np.int32)
    untile[gather[slots]] = slots
    # Each prefix tile slot holds the index of its row among the dense rows.
    dense_rows = np.concatenate(dense) if dense else np.zeros(0, dtype=np.int64)
    prefix_length = len(dense_rows)
    prefix_gather = np.full(n_prefix * TILE, prefix_length, dtype=np.int32)
    prefix_gather[untile[dense_rows]] = np.arange(prefix_length)
    return dict(
        variable_block_sizes=mx.array([len(r) for r in rows_of], dtype=mx.int32),
        num_prefix_tiles=n_prefix,
        prefix_length=prefix_length,
        tile_gather_index=mx.array(gather),
        prefix_gather_index=mx.array(prefix_gather),
        untile_index=mx.array(untile),
    )


def _geometry(*segments):
    return kq.VSAGeometry(**_fields(*segments))


def _packed(rng, seq, heads, n=4):
    return [
        mx.array(rng.standard_normal((seq, heads, D)).astype(np.float32)).astype(
            mx.bfloat16
        )
        for _ in range(n)
    ]


def _views(q, k, v):
    """q, k and v cut as a FastH3 forward passes them, the same values: q
    and k through the [1, heads, S, D] transpose of a row-major [1, S, heads,
    D] buffer (qk_norm_rope's output), v as the third part of a split of the
    fused [S, 3 heads D] qkv product, rows 3 heads D apart."""
    seq, heads, dim = q.shape

    def roped(x):
        return x[None].transpose(0, 2, 1, 3).squeeze(0).transpose(1, 0, 2)

    fused = mx.concatenate([x.reshape(seq, heads * dim) for x in (q, k, v)], axis=1)
    mx.eval(fused)
    return roped(q), roped(k), mx.split(fused, 3, axis=1)[2].reshape(seq, heads, dim)


def _dense(q, k, v):
    def heads_first(x):
        return x.transpose(1, 0, 2)[None]

    out = mx.fast.scaled_dot_product_attention(
        heads_first(q), heads_first(k), heads_first(v), scale=SCALE
    )
    return out[0].transpose(1, 0, 2)


_TF32_HIGH = np.uint32(0xFFFFE000)
_TF32 = []


def _tf32_device():
    """Whether the GPU multiplies float32 matrices at TF32: one 64x64
    product against the CPU's."""
    if not _TF32:
        rng = np.random.default_rng(31)
        a, b = (mx.array(rng.standard_normal((64, 64)), mx.float32) for _ in "ab")
        gap = mx.abs(mx.matmul(a, b, stream=mx.gpu) - mx.matmul(a, b, stream=mx.cpu))
        _TF32.append(mx.max(gap).item() >= 1e-3)
    return _TF32[0]


def _split_np(x):
    """float32 x as its TF32-exact part, the low 13 mantissa bits cleared,
    and the rest."""
    x = np.asarray(x, dtype=np.float32)
    high = (x.view(np.uint32) & _TF32_HIGH).view(np.float32)
    return high, x - high


def _routing_scores(q_pooled, k_pooled, split):
    """Pooled q against pooled k over sqrt(D). With split, as on a GPU that
    multiplies float32 at TF32: qh @ kh.T + (qh @ kl.T + ql @ kh.T)."""
    if not split:
        return (q_pooled @ k_pooled.transpose(0, 2, 1)) / D**0.5
    qh, ql = (mx.array(x) for x in _split_np(q_pooled))
    kh, kl = (mx.array(x).transpose(0, 2, 1) for x in _split_np(k_pooled))
    return (qh @ kh + (qh @ kl + ql @ kh)) / D**0.5


def _parts(q, k, v, gate, g, sparsity, split=None):
    """The packed output from its parts, and the routing of a sparse call.
    The scores are split where the GPU multiplies float32 at TF32, unless
    split says otherwise."""
    split = _tf32_device() if split is None else split
    heads = q.shape[1]
    n_tiles, n_prefix = g.variable_block_sizes.shape[0], g.num_prefix_tiles
    untile = np.array(g.untile_index)

    def tiled(x):
        zero = mx.zeros((1, heads, D), dtype=x.dtype)
        return mx.concatenate([x, zero])[g.tile_gather_index]

    def pooled(x):
        x = x.astype(mx.float32).reshape(n_tiles, TILE, heads, D).sum(axis=1)
        x = x / g.variable_block_sizes.astype(mx.float32)[:, None, None]
        return x.transpose(1, 0, 2)

    qt, kt, vt = (tiled(x) for x in (q, k, v))
    scores = _routing_scores(pooled(qt), pooled(kt), split)
    idx = None
    if sparsity <= 0:
        out = _dense(q, k, v)
    else:
        spans = ((n_prefix, n_tiles),)
        idx, num = vsa._block_indices_from_scores(
            scores, n_prefix, n_tiles - n_prefix, sparsity, True, spans, (sparsity,)
        )
        video = kq.vsa_block_sparse(
            qt, kt, vt, idx, num, g.variable_block_sizes, n_prefix, SCALE
        )
        rows = g.prefix_length
        out = mx.concatenate([_dense(q[:rows], k, v), video[mx.array(untile[rows:])]])
    if gate is not None:
        coarse = mx.softmax(scores, axis=-1) @ pooled(vt)
        coarse = coarse.transpose(1, 0, 2)[mx.array(untile // TILE)]
        out = out + (coarse * gate).astype(mx.bfloat16)
    return out, idx, scores


def _bits(x):
    return np.array(x.view(mx.uint16))


# One video query tile keeps ceil(0.2 * 12) = 3 video tiles.
@requires_nax
@pytest.mark.parametrize("gated", [True, False], ids=["gate", "no_gate"])
@pytest.mark.parametrize("views", [False, True], ids=["rows", "views"])
def test_sparse_call_matches_its_parts(gated, views):
    rng = np.random.default_rng(21)
    g = _geometry(*PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 3)
    gate = gate if gated else None
    args = _views(q, k, v) if views else (q, k, v)
    out = kq.vsa_attention(*args, g, 0.8, gate=gate)
    ref, idx, scores = _parts(q, k, v, gate, g, 0.8)
    np.testing.assert_array_equal(_bits(out), _bits(ref))
    # Every prefix tile, then the top 3 video tiles by pooled score.
    n_prefix = g.num_prefix_tiles
    idx, scores = np.array(idx), np.array(scores)[:, n_prefix:, n_prefix:]
    assert (idx[..., :n_prefix] == np.arange(n_prefix)).all()
    top = np.sort(np.argsort(-scores, axis=-1)[..., :3], axis=-1) + n_prefix
    np.testing.assert_array_equal(np.sort(idx[..., n_prefix:], axis=-1), top)


@pytest.mark.parametrize("views", [False, True], ids=["rows", "views"])
def test_dense_call_matches_its_parts(views):
    rng = np.random.default_rng(23)
    g = _geometry(*PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 3)
    args = _views(q, k, v) if views else (q, k, v)
    out = kq.vsa_attention(*args, g, 0.0, gate=gate)
    ref, _, _ = _parts(q, k, v, gate, g, 0.0)
    np.testing.assert_array_equal(_bits(out), _bits(ref))


# Many tiles, with short edge tiles on every axis: 3 prefix tiles and 168
# video tiles.
GRID_WIDE = (13, 22, 26)


@pytest.mark.parametrize("sparsity", [0.0, pytest.param(0.8, marks=requires_nax)])
@pytest.mark.parametrize("views", [False, True], ids=["rows", "views"])
def test_call_with_many_tiles_matches_its_parts(sparsity, views):
    rng = np.random.default_rng(27)
    g = _geometry(*PREFIX, GRID_WIDE)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 3)
    args = _views(q, k, v) if views else (q, k, v)
    out = kq.vsa_attention(*args, g, sparsity, gate=gate)
    ref, _, _ = _parts(q, k, v, gate, g, sparsity)
    np.testing.assert_array_equal(_bits(out), _bits(ref))


# The pools divide MLX's float32 sum of each tile, read from the bfloat16
# tiles: the bits of the float32 sum of the upcast tiles, at FastH3's 56 heads
# (columns 56 x 128 apart) and at 3, with +0.0 pad rows, -0.0 and subnormal
# values among the rows.
@pytest.mark.parametrize("heads", [56, 3])
def test_tile_sums_have_the_bits_of_mlx_float32_sums(heads):
    rng = np.random.default_rng(31)
    n = 21
    x = rng.standard_normal((n * TILE, heads, D)).astype(np.float32)
    x[5 * TILE + 40 : 6 * TILE] = 0.0
    x[7 * TILE : 7 * TILE + 3, :, :8] = -0.0
    x[9 * TILE + 3, :, 8:16] = 1e-40
    x = mx.array(x).astype(mx.bfloat16)
    sums = vsa._vsa_tile_sums(x)
    ref = x.astype(mx.float32).reshape(n, TILE, heads, D).sum(axis=1)
    np.testing.assert_array_equal(
        np.array(sums.view(mx.uint32)), np.array(ref.view(mx.uint32))
    )


# The pools read the bfloat16 tiles in place: a pool of 64 tiles at 56 heads
# holds less than a float32 copy of its tiles (117 MB) at any point.
def test_pool_holds_no_float32_copy_of_its_tiles():
    n, heads = 64, 56
    x = mx.random.normal((n * TILE, heads, D), mx.bfloat16, key=mx.random.key(47))
    vbs = mx.full((n,), TILE, dtype=mx.int32)
    mx.eval(x, vbs)
    limit = mx.set_cache_limit(0)
    try:
        # Buffers free when their command buffer completes.
        mx.synchronize()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        mx.eval(vsa._pool(x, vbs))
        grown = mx.get_peak_memory() - base
    finally:
        mx.set_cache_limit(limit)
    assert grown < 4 * x.size


# v as 2 heads of a [S, 32768] bfloat16 buffer, rows 32768 apart, so its
# 65670 rows span 2.15e9 elements. MLX's gather reads such a view with 32-bit
# offsets (docs/upstream-defects.md), so the call copies it first.
@big
@pytest.mark.parametrize("sparsity", [0.0, pytest.param(0.8, marks=requires_nax)])
def test_call_on_a_view_past_32_bit_offsets_matches_a_contiguous_call(sparsity):
    heads, width = 2, 32768
    g = _geometry(*PREFIX, (5, 113, 116))
    rows = g.untile_index.shape[0]
    assert rows * width > 2**31 - 1
    q, k, v, gate = _packed(np.random.default_rng(41), rows, heads)
    buf = mx.zeros((rows, width), dtype=mx.bfloat16)
    buf[:, : heads * D] = v.reshape(rows, heads * D)
    view = buf[:, : heads * D].reshape(rows, heads, D)
    out = kq.vsa_attention(q, k, view, g, sparsity, gate=gate)
    ref = kq.vsa_attention(q, k, v, g, sparsity, gate=gate)
    rows_off = int((_bits(out) != _bits(ref)).any(axis=(1, 2)).sum())
    # The guard's copy is its own output, so a gather in a later command
    # buffer reads all of it. Eager and compiled calls at cache limit 0.
    idx = mx.arange(rows)
    take = mx.compile(lambda x, i: mx.take(_vsa_gather_source(x), i, axis=0))
    limit = mx.set_cache_limit(0)
    calls_off = []
    try:
        for call in (lambda x, i: mx.take(_vsa_gather_source(x), i, axis=0), take):
            for _ in range(6):
                y = call(view, idx)
                mx.eval(y)
                calls_off.append(int((y.view(mx.uint16) != v.view(mx.uint16)).sum()))
                del y
    finally:
        mx.set_cache_limit(limit)
    # Free the buffer before the asserts: a failed test's frame outlives it.
    del buf, view, out
    mx.clear_cache()
    assert rows_off == 0
    assert calls_off == [0] * 12


# The call reads each tiled operand through the gather guard: q, k, v and the
# gate, and at sparsity 0 the dense output. The test above checks the guard's
# bits on a view past 32-bit offsets.
@pytest.mark.parametrize("sparsity", [0.0, pytest.param(0.8, marks=requires_nax)])
def test_call_tiles_its_operands_through_the_gather_guard(sparsity):
    g = _geometry(*PREFIX, GRID)
    q, k, v, gate = _packed(np.random.default_rng(21), g.untile_index.shape[0], 3)
    nodes = _nodes(kq.vsa_attention(q, k, v, g, sparsity, gate=gate))
    assert nodes.count("KQuantVSAGatherSource") == (5 if sparsity == 0.0 else 4), nodes


# Past INT32_MAX elements the tile sums take 64-bit offsets, as MLX does:
# 4688 tiles of 56 heads, 2.15e9 elements. The first tiles, middle tiles and
# the last 8, past the 32-bit limit, against MLX's float32 sums of each part.
@big
def test_tile_sums_past_int32_max_elements_have_the_bits_of_mlx_float32_sums():
    heads, n = 56, 4688
    assert n * TILE * heads * D > 2**31 - 1
    rows = (mx.arange(n * TILE) % 1009).astype(mx.bfloat16) * 2.0**-7
    cols = mx.random.normal((heads, D), key=mx.random.key(43)).astype(mx.bfloat16)
    x = rows[:, None, None] + cols
    sums = vsa._vsa_tile_sums(x)
    off = {}
    for a in (0, n // 2, n - 8):
        part = x[a * TILE : (a + 8) * TILE].astype(mx.float32)
        ref = part.reshape(8, TILE, heads, D).sum(axis=1)
        off[a] = int((sums[a : a + 8].view(mx.uint32) != ref.view(mx.uint32)).sum())
    # Free x before the assert: a failed test's frame outlives it.
    del x, sums, part
    mx.clear_cache()
    assert off == {0: 0, n // 2: 0, n - 8: 0}


# Routing scores. FastVideo's PyTorch backend, the trained definition, scores
# pooled q against pooled k in float32, and so does vsa_attention on every
# device.


def _pooled(rng, heads, tiles):
    """Random pooled q or k: the mean of a 64-row tile of unit normal rows
    has standard deviation 1/8."""
    x = rng.standard_normal((heads, tiles, D)) / 8
    return mx.array(x.astype(np.float32))


def test_split_high_part_is_tf32_exact_and_the_parts_sum_to_the_input():
    rng = np.random.default_rng(33)
    x = np.concatenate(
        [
            rng.standard_normal(4096) / 8,
            rng.standard_normal(4096) * 10.0 ** rng.integers(-4, 5, 4096),
            [0.0],
        ]
    ).astype(np.float32)
    with mx.stream(mx.cpu):
        high, low = (np.array(a) for a in vsa._split(mx.array(x)))
    assert not (high.view(np.uint32) & 0x1FFF).any()
    np.testing.assert_array_equal(high + low, x)


# The bits a caller that mirrors the routing reproduces, at the 512x512x22
# tile counts.
def test_routing_scores_follow_the_documented_product():
    rng = np.random.default_rng(35)
    q_pooled, k_pooled = _pooled(rng, 56, 35), _pooled(rng, 56, 35)
    got = np.array(vsa._scores(q_pooled, k_pooled))
    want = np.array(_routing_scores(q_pooled, k_pooled, _tf32_device()))
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


# The routing against a top-k of the CPU's float32 scores, at the 768x768x39
# tile counts. A row whose k-th to next video score gap is under twice the
# largest score difference can flip with no bug behind it, and is not
# counted.
def test_routing_matches_a_cpu_fp32_top_k():
    rng = np.random.default_rng(37)
    heads, n_prefix, n_video = 56, 4, 108
    n_tiles = n_prefix + n_video
    q_pooled, k_pooled = _pooled(rng, heads, n_tiles), _pooled(rng, heads, n_tiles)
    got = vsa._scores(q_pooled, k_pooled)
    cpu = mx.matmul(q_pooled, k_pooled.transpose(0, 2, 1), stream=mx.cpu) / D**0.5
    noise = mx.max(mx.abs(got - cpu)).item()
    # float32 rounding of a 128-term product, far under TF32's.
    assert noise <= 1e-5 * mx.max(mx.abs(cpu)).item()

    def routed(scores):
        idx, _ = vsa._block_indices_from_scores(
            scores, n_prefix, n_video, 0.8, True, ((n_prefix, n_tiles),), (0.8,)
        )
        return np.sort(np.array(idx), axis=-1)

    differ = (routed(got) != routed(cpu)).any(axis=-1)
    video = -np.sort(-np.array(cpu)[:, n_prefix:, n_prefix:], axis=-1)
    top = vsa.compute_topk(0.8, n_video)
    near_tie = video[..., top - 1] - video[..., top] < 2 * noise
    assert near_tie.mean() < 0.01
    assert not (differ & ~near_tie).any()


# Guards. Each input fails only the named guard, on a dense call, so the
# tests run on any GPU and evaluate nothing.


def _guard(match, sparsity=0.0, **change):
    rng = np.random.default_rng(25)
    g = _geometry(*PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 2)
    args = dict(q=q, k=k, v=v, geometry=g, sparsity=sparsity, gate=gate)
    for name, fn in change.items():
        args[name] = fn(args[name]) if callable(fn) else fn
    with pytest.raises(ValueError, match=match):
        kq.vsa_attention(**args)


def _rows(x, n):
    return mx.zeros((n,) + x.shape[1:], dtype=x.dtype)


def _all(fn):
    return {name: fn for name in ("q", "k", "v", "gate")}


GUARDS = {
    "ndim": ("of one shape", _all(lambda x: x.reshape(x.shape[0], -1))),
    "k_shape": ("of one shape", {"k": lambda k: k[:, :1]}),
    "gate_shape": ("of one shape", {"gate": lambda x: x[:, :1]}),
    "head_dim": (
        "of one shape",
        _all(lambda x: mx.zeros(x.shape[:2] + (64,), x.dtype)),
    ),
    "dtype": ("of one shape", _all(lambda x: x.astype(mx.float16))),
    "rows": ("packed rows", _all(lambda x: _rows(x, x.shape[0] + 1))),
    "compete": ("compete mode", {"exempt": False}),
}


@pytest.mark.parametrize("name", sorted(GUARDS))
def test_guard_refuses(name):
    match, change = GUARDS[name]
    _guard(match, **change)


# Geometry guards, checked when the geometry is built, in the order of the
# checks. Each input fails only the named check, so with that check removed
# the geometry builds. The tile input, a 64-row map relabelled as 128-row
# tiles, cannot avoid failing the map checks after it as well.


def _swap_pad(index, old, new):
    return mx.where(index == old, new, index)


def _replace(**change):
    def build(g):
        return dataclasses.replace(g, **{k: fn(g) for k, fn in change.items()})

    return build


def _append(index, value, n):
    return mx.concatenate([index, mx.full((n,), value, dtype=index.dtype)])


def _row_in_a_pad_slot(g):
    """The last live row of the first short video tile moved into that
    tile's first pad slot, tile_gather_index moved with it."""
    vbs = np.array(g.variable_block_sizes)
    untile, gather = np.array(g.untile_index), np.array(g.tile_gather_index)
    t = g.num_prefix_tiles + int(np.argmax(vbs[g.num_prefix_tiles :] < TILE))
    live, pad = t * TILE + vbs[t] - 1, t * TILE + vbs[t]
    row = gather[live]
    untile[row], gather[pad], gather[live] = pad, row, untile.shape[0]
    return dataclasses.replace(
        g, untile_index=mx.array(untile), tile_gather_index=mx.array(gather)
    )


def _dense_after_video(g):
    """Dense rows after the video region, each prefix tile slot holding its
    packed row."""
    f = _fields(20, (4, 8, 8), 13)
    head = f["tile_gather_index"][: f["num_prefix_tiles"] * TILE]
    rows = f["untile_index"].shape[0]
    f["prefix_gather_index"] = _swap_pad(head, rows, f["prefix_length"])
    return kq.VSAGeometry(**f)


def _seq(g):
    return g.untile_index.shape[0]


GEOMETRY_GUARDS = {
    "tile": ("tile size 64", _replace(tile_size=lambda g: 128)),
    "regions": (
        "more than one video region",
        _replace(video_tile_spans=lambda g: ((3, 9), (9, 15))),
    ),
    "span": (
        "video_tile_spans must be",
        _replace(video_tile_spans=lambda g: ((0, g.variable_block_sizes.shape[0]),)),
    ),
    "empty_tile": (
        "variable_block_sizes must be",
        _replace(
            variable_block_sizes=lambda g: _append(g.variable_block_sizes, 0, 1),
            tile_gather_index=lambda g: _append(g.tile_gather_index, _seq(g), TILE),
        ),
    ),
    "oversized_tile": (
        "variable_block_sizes must be",
        _replace(
            variable_block_sizes=lambda g: mx.where(
                g.variable_block_sizes == TILE, TILE + 1, g.variable_block_sizes
            )
        ),
    ),
    "untile": ("untile_index must", _row_in_a_pad_slot),
    "tile_gather_pad": (
        "tile_gather_index must",
        _replace(
            tile_gather_index=lambda g: _swap_pad(
                g.tile_gather_index, _seq(g), g.prefix_length
            )
        ),
    ),
    "tile_gather_short": (
        "tile_gather_index must",
        _replace(tile_gather_index=lambda g: g.tile_gather_index[:-TILE]),
    ),
    "dense_after_video": ("dense rows must come before", _dense_after_video),
    "prefix_tiles": (
        "disagree",
        _replace(
            num_prefix_tiles=lambda g: g.num_prefix_tiles - 1,
            prefix_gather_index=lambda g: _swap_pad(
                g.tile_gather_index[: (g.num_prefix_tiles - 1) * TILE],
                _seq(g),
                g.prefix_length,
            ),
        ),
    ),
    "prefix_length": (
        "disagree",
        _replace(
            prefix_length=lambda g: g.prefix_length - 10,
            prefix_gather_index=lambda g: _swap_pad(
                g.prefix_gather_index, g.prefix_length, g.prefix_length - 10
            ),
        ),
    ),
    "prefix_gather_pad": (
        "prefix_gather_index must",
        _replace(
            prefix_gather_index=lambda g: _swap_pad(
                g.prefix_gather_index, g.prefix_length, _seq(g)
            )
        ),
    ),
    "prefix_gather_short": (
        "prefix_gather_index must",
        _replace(prefix_gather_index=lambda g: g.prefix_gather_index[:-TILE]),
    ),
}


@pytest.mark.parametrize("name", sorted(GEOMETRY_GUARDS))
def test_geometry_guard_refuses(name):
    match, build = GEOMETRY_GUARDS[name]
    g = _geometry(*PREFIX, GRID)
    with pytest.raises(ValueError, match=match):
        build(g)


# A caller can build the untile map with argsort, which returns uint32: the
# geometry takes it and the call gives the bits of the int32 map.
@pytest.mark.parametrize("sparsity", [0.0, pytest.param(0.8, marks=requires_nax)])
def test_untile_index_from_argsort_gives_the_same_bits(sparsity):
    rng = np.random.default_rng(29)
    g = _geometry(*PREFIX, GRID)
    sorted_slots = mx.argsort(g.tile_gather_index)[: _seq(g)]
    assert sorted_slots.dtype == mx.uint32
    g_sorted = dataclasses.replace(g, untile_index=sorted_slots)
    q, k, v, gate = _packed(rng, _seq(g), 2)
    out = kq.vsa_attention(q, k, v, g_sorted, sparsity, gate=gate)
    ref = kq.vsa_attention(q, k, v, g, sparsity, gate=gate)
    np.testing.assert_array_equal(_bits(out), _bits(ref))


# A compiled function holds the geometry it traced. A fixed-shape compile
# replays it only at the traced shapes, as the eager call computes it. A
# shapeless one would replay it at other shapes too, so the call refuses
# even its first, traced shape.
def _attend_dense(g):
    def attend(q, k, v, gate):
        return kq.vsa_attention(q, k, v, g, 0.0, gate=gate)

    return attend


def test_fixed_shape_compile_matches_the_eager_call():
    rng = np.random.default_rng(27)
    g = _geometry(*PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 2)
    attend = _attend_dense(g)
    compiled = mx.compile(attend)
    eager = _bits(attend(q, k, v, gate))
    np.testing.assert_array_equal(_bits(compiled(q, k, v, gate)), eager)
    np.testing.assert_array_equal(_bits(compiled(q, k, v, gate)), eager)


def test_guard_refuses_a_shapeless_compile():
    rng = np.random.default_rng(27)
    g = _geometry(*PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 2)
    with pytest.raises(ValueError, match="cannot hold this call"):
        mx.compile(_attend_dense(g), shapeless=True)(q, k, v, gate)


# MLX reads MLX_ENABLE_TF32 once a process, so these run in a child process.
_CHILD = r"""
import json
import os
import sys

spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["tests_dir"])
import mlx.core as mx
import numpy as np
import test_vsa_attention as t

import mlx_kquant as kq
from mlx_kquant import vsa

g = t._geometry(*t.PREFIX, t.GRID)
q, k, v, gate = t._packed(np.random.default_rng(39), t._seq(g), 2)
calls = [0.0] + ([0.8] if t.NAX else [])


def attend(q, k, v, gate, s):
    return kq.vsa_attention(q, k, v, g, s, gate=gate)


def same(a, b):
    return bool((t._bits(a) == t._bits(b)).all())


out = {}
if spec["mode"] == "geometry_then_off":
    os.environ["MLX_ENABLE_TF32"] = "0"
    out["exact"] = not t._tf32_device()
elif spec["mode"] == "compiled_first":
    out["measured"] = sorted(vsa._F32_EXACT)
    for s in calls:
        got = mx.compile(lambda *a, s=s: attend(*a, s))(q, k, v, gate)
        out[str(s)] = same(got, attend(q, k, v, gate, s))
else:
    out["exact"] = vsa._f32_products_exact()
    for s in calls:
        ref = t._parts(q, k, v, gate, g, s, split=False)[0]
        out[str(s)] = same(attend(q, k, v, gate, s), ref)
print(json.dumps(out))
"""


def _child(mode, env=None):
    spec = {"mode": mode, "tests_dir": os.path.dirname(os.path.abspath(__file__))}
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(spec)],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


# With TF32 off the GPU multiplies float32 at float32, and each call gives
# the bits of the direct product.
def test_tf32_off_keeps_the_direct_product():
    out = _child("tf32_off", {"MLX_ENABLE_TF32": "0"})
    assert out.pop("exact")
    assert out == {"0.0": True, **({"0.8": True} if NAX else {})}


# Building a geometry runs no product, so MLX reads MLX_ENABLE_TF32 at a
# later product, and a caller can still turn TF32 off after it.
def test_building_a_geometry_leaves_the_tf32_switch_unread():
    if not _tf32_device():
        pytest.skip("the GPU multiplies float32 at float32.")
    assert _child("geometry_then_off") == {"exact": True}


# A process whose first call is compiled measures the device inside the
# trace: the probe reads only constants, which MLX evaluates there.
def test_a_compiled_first_call_measures_the_device_inside_the_trace():
    out = _child("compiled_first")
    assert out.pop("measured") == []
    assert out == {"0.0": True, **({"0.8": True} if NAX else {})}
