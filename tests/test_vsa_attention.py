"""vsa_attention: MiniMax-H3 video sparse attention over one packed sequence,
against its parts in FastVideo's order: MLX's fused attention for the prefix
query rows (every row at sparsity 0), vsa_block_sparse for the video query
rows, then the gate term per row, the coarse output of the row's tile times
the gate in float32, rounded to bfloat16 and added.

The module skips under KQUANT_FORCE_CPU and without Metal. The sparse calls
need tensor-op (NAX) hardware, and the dense call and the guards run on any
Metal GPU.
"""

from __future__ import annotations

import dataclasses
import os

import mlx.core as mx
import numpy as np
import pytest

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

TILE = 64
D = 128
SCALE = D**-0.5
# Text and audio rows, then a latent grid that leaves short edge tiles on
# every axis: 3 prefix tiles (56, 64, 10 rows) and 12 video tiles.
PREFIX = (56, 74)
GRID = (5, 8, 10)


def _geometry(*segments):
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
    return kq.VSAGeometry(
        variable_block_sizes=mx.array([len(r) for r in rows_of], dtype=mx.int32),
        num_prefix_tiles=n_prefix,
        prefix_length=prefix_length,
        tile_gather_index=mx.array(gather),
        prefix_gather_index=mx.array(prefix_gather),
        untile_index=mx.array(untile),
    )


def _packed(rng, seq, heads, n=4):
    return [
        mx.array(rng.standard_normal((seq, heads, D)).astype(np.float32)).astype(
            mx.bfloat16
        )
        for _ in range(n)
    ]


def _dense(q, k, v):
    def heads_first(x):
        return x.transpose(1, 0, 2)[None]

    out = mx.fast.scaled_dot_product_attention(
        heads_first(q), heads_first(k), heads_first(v), scale=SCALE
    )
    return out[0].transpose(1, 0, 2)


def _parts(q, k, v, gate, g, sparsity):
    """The packed output from its parts, and the routing of a sparse call."""
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
    scores = (pooled(qt) @ pooled(kt).transpose(0, 2, 1)) / D**0.5
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
def test_sparse_call_matches_its_parts(gated):
    rng = np.random.default_rng(21)
    g = _geometry(*PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 3)
    gate = gate if gated else None
    out = kq.vsa_attention(q, k, v, g, 0.8, gate=gate)
    ref, idx, scores = _parts(q, k, v, gate, g, 0.8)
    np.testing.assert_array_equal(_bits(out), _bits(ref))
    # Every prefix tile, then the top 3 video tiles by pooled score.
    n_prefix = g.num_prefix_tiles
    idx, scores = np.array(idx), np.array(scores)[:, n_prefix:, n_prefix:]
    assert (idx[..., :n_prefix] == np.arange(n_prefix)).all()
    top = np.sort(np.argsort(-scores, axis=-1)[..., :3], axis=-1) + n_prefix
    np.testing.assert_array_equal(np.sort(idx[..., n_prefix:], axis=-1), top)


def test_dense_call_matches_its_parts():
    rng = np.random.default_rng(23)
    g = _geometry(*PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 3)
    out = kq.vsa_attention(q, k, v, g, 0.0, gate=gate)
    ref, _, _ = _parts(q, k, v, gate, g, 0.0)
    np.testing.assert_array_equal(_bits(out), _bits(ref))


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


# Geometry guards, checked when the geometry is built. Each input fails only
# the named check: a builder that cuts one empty tile too many or counts a
# full tile as 65 rows, an index map one tile off, the pad values of the two
# gathers swapped, the prefix fields off, and dense rows that follow the
# video region.


def _swap_pad(index, old, new):
    return mx.where(index == old, new, index)


def _replace(**change):
    def build(g):
        return dataclasses.replace(g, **{k: fn(g) for k, fn in change.items()})

    return build


def _append(index, value, n):
    return mx.concatenate([index, mx.full((n,), value, dtype=index.dtype)])


GEOMETRY_GUARDS = {
    "tile": ("tile size 64", _replace(tile_size=lambda g: 128)),
    "regions": (
        "more than one video region",
        _replace(video_tile_spans=lambda g: ((3, 9), (9, 15))),
    ),
    "empty_tile": (
        "variable_block_sizes must be",
        _replace(
            variable_block_sizes=lambda g: _append(g.variable_block_sizes, 0, 1),
            tile_gather_index=lambda g: _append(
                g.tile_gather_index, g.untile_index.shape[0], TILE
            ),
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
    "untile": (
        "untile_index must",
        _replace(untile_index=lambda g: g.untile_index + TILE),
    ),
    "tile_gather_pad": (
        "tile_gather_index must",
        _replace(
            tile_gather_index=lambda g: _swap_pad(
                g.tile_gather_index, g.untile_index.shape[0], g.prefix_length
            )
        ),
    ),
    "tile_gather_short": (
        "tile_gather_index must",
        _replace(tile_gather_index=lambda g: g.tile_gather_index[:-TILE]),
    ),
    "dense_after_video": (
        "dense rows must come before",
        lambda g: _geometry(20, (4, 8, 8), 13),
    ),
    "prefix_tiles": (
        "disagree",
        _replace(num_prefix_tiles=lambda g: g.num_prefix_tiles - 1),
    ),
    "prefix_length": (
        "disagree",
        _replace(prefix_length=lambda g: g.prefix_length - 10),
    ),
    "prefix_gather_pad": (
        "prefix_gather_index must",
        _replace(
            prefix_gather_index=lambda g: _swap_pad(
                g.prefix_gather_index, g.prefix_length, g.untile_index.shape[0]
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
