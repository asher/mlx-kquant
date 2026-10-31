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


def _geometry(prefix, grid):
    """The tile map of [prefix segments | video grid], packed in that order:
    each segment in 64-row tiles, its last one short, then 4x4x4 video tiles
    in raster order, the edge tiles short."""
    rows_of = []
    row = 0
    for n in prefix:
        rows_of += [
            np.arange(row + i, row + min(i + TILE, n)) for i in range(0, n, TILE)
        ]
        row += n
    n_prefix, prefix_length = len(rows_of), row
    t, h, w = grid
    video = np.arange(t * h * w).reshape(t, h, w) + row
    for a in range(0, t, 4):
        for b in range(0, h, 4):
            for c in range(0, w, 4):
                rows_of.append(video[a : a + 4, b : b + 4, c : c + 4].reshape(-1))
    seq = row + t * h * w
    gather = np.full(len(rows_of) * TILE, seq, dtype=np.int32)
    for i, rows in enumerate(rows_of):
        gather[i * TILE : i * TILE + len(rows)] = rows
    slots = np.nonzero(gather < seq)[0]
    untile = np.empty(seq, dtype=np.int32)
    untile[gather[slots]] = slots
    prefix_gather = gather[: n_prefix * TILE]
    prefix_gather = np.where(prefix_gather < seq, prefix_gather, prefix_length)
    return kq.VSAGeometry(
        variable_block_sizes=mx.array([len(r) for r in rows_of], dtype=mx.int32),
        num_prefix_tiles=n_prefix,
        prefix_length=prefix_length,
        tile_gather_index=mx.array(gather),
        prefix_gather_index=mx.array(prefix_gather.astype(np.int32)),
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
    g = _geometry(PREFIX, GRID)
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
    g = _geometry(PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 3)
    out = kq.vsa_attention(q, k, v, g, 0.0, gate=gate)
    ref, _, _ = _parts(q, k, v, gate, g, 0.0)
    np.testing.assert_array_equal(_bits(out), _bits(ref))


# Guards. Each input fails only the named guard, on a dense call, so the
# tests run on any GPU and evaluate nothing.


def _guard(match, sparsity=0.0, **change):
    rng = np.random.default_rng(25)
    g = _geometry(PREFIX, GRID)
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
    "k_shape": ("of one shape", {"k": lambda k: k[:, :1]}),
    "gate_shape": ("of one shape", {"gate": lambda x: x[:, :1]}),
    "head_dim": (
        "of one shape",
        _all(lambda x: mx.zeros(x.shape[:2] + (64,), x.dtype)),
    ),
    "dtype": ("of one shape", _all(lambda x: x.astype(mx.float16))),
    "tile": (
        "tile size 64",
        {"geometry": lambda g: dataclasses.replace(g, tile_size=128)},
    ),
    "rows": ("packed rows", _all(lambda x: _rows(x, x.shape[0] + 1))),
    "compete": ("compete mode", {"exempt": False}),
    "regions": (
        "more than one video region",
        {
            "geometry": lambda g: dataclasses.replace(
                g, video_tile_spans=((3, 9), (9, 15))
            )
        },
    ),
}


@pytest.mark.parametrize("name", sorted(GUARDS))
def test_guard_refuses(name):
    match, change = GUARDS[name]
    _guard(match, **change)


# A shapeless replay would reuse index arrays and shapes built for one
# sequence, so the call refuses even its first, traced shape.
def test_guard_refuses_a_shapeless_compile():
    rng = np.random.default_rng(27)
    g = _geometry(PREFIX, GRID)
    q, k, v, gate = _packed(rng, g.untile_index.shape[0], 2)

    def attend(q, k, v, gate):
        return kq.vsa_attention(q, k, v, g, 0.0, gate=gate)

    with pytest.raises(ValueError, match="cannot hold this call"):
        mx.compile(attend, shapeless=True)(q, k, v, gate)
