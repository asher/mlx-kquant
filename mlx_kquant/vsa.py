# SPDX-License-Identifier: Apache-2.0 AND MIT
"""MiniMax-H3 video sparse attention (VSA), the whole attention call.

``vsa_attention`` computes FastVideo's ``h3_vsa_attention`` with
``impl="nax"``, in exempt mode over one video region, bit for bit: the
packed q, k, v and gate in, the packed output out. The prefix query rows
attend every key through MLX's fused attention. The video query tiles attend
their routed key tiles through ``vsa_block_sparse``, which adds the gate term
in its epilogue. The pooling, the scores, the routing and the gate term of
the prefix tiles are MLX ops in FastVideo's order.

Ported from FastVideo (https://github.com/hao-ai-lab/FastVideo),
fastvideo/mlx_runtime/minimax_h3_vsa.py, licensed under the Apache License,
Version 2.0 (see licenses/FastVideo-LICENSE). The routing functions below
are copied from that file and reformatted, and vsa_attention follows its
h3_vsa_attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from ._ext import _vsa_fixed_shapes, vsa_block_sparse

_OP = "[mlx_kquant.vsa_attention] "
_TILE = 64
_HEAD_DIM = 128


def _maps_rows_onto_live_slots(untile, live):
    """untile_index sends each packed row to its own live tile slot, and
    every live slot gets one row. An entry out of range counts in one of two
    end bins, so no write leaves the counts. The map is read as int64, so an
    unsigned one clips as its values do."""
    n = live.shape[0]
    bins = mx.clip(untile.astype(mx.int64), -1, n) + 1
    hits = mx.zeros((n + 2,), dtype=mx.int32).at[bins].add(1)
    end = mx.zeros((1,), dtype=mx.int32)
    return mx.array_equal(
        hits, mx.concatenate([end, live.astype(mx.int32), end])
    ).item()


def _inverts(gather, untile, n_slots):
    """tile_gather_index holds the packed row of each live slot and S in
    each pad slot. An entry of untile_index out of range writes one of two
    end slots, so no write leaves the expected map."""
    rows = untile.shape[0]
    want = mx.full((n_slots + 2,), rows, dtype=mx.int32)
    want[mx.clip(untile.astype(mx.int64), -1, n_slots) + 1] = mx.arange(
        rows, dtype=mx.int32
    )
    return mx.array_equal(gather, want[1:-1]).item()


@dataclass(frozen=True, eq=False)
class VSAGeometry:
    """The tile map of one packed sequence of S rows: the dense prefix rows
    first, then one video region.

    The prefix rows are cut into tiles of at most tile_size rows, and the
    video rows into their 3D tiles. Each tile holds its live rows, then pad
    slots.

    Building one checks the index maps against each other on the host, which
    evaluates them once. It raises ValueError when they disagree, when the
    dense rows do not come first, for more than one video region (Ref2VA)
    and for a tile size other than 64. Build it outside a compiled function.

    Attributes:
        variable_block_sizes (array): int32 [tiles], the live rows of each
            tile, the prefix tiles first.
        num_prefix_tiles (int): the tiles that hold the prefix rows.
        prefix_length (int): the prefix rows, the first packed rows.
        tile_gather_index (array): int32 [tiles * tile_size], the packed row
            of each tile slot, or S for a pad slot.
        prefix_gather_index (array): int32 [num_prefix_tiles * tile_size],
            the prefix row of each prefix tile slot, or prefix_length for a
            pad slot.
        untile_index (array): int32 [S], the tile slot of each packed row.
        video_tile_spans (tuple): the [start, end) tiles of each video
            region: empty, or the one region ((num_prefix_tiles, tiles),).
        tile_size (int): rows per tile, 64.
    """

    variable_block_sizes: mx.array
    num_prefix_tiles: int
    prefix_length: int
    tile_gather_index: mx.array
    prefix_gather_index: mx.array
    untile_index: mx.array
    video_tile_spans: tuple[tuple[int, int], ...] = ()
    tile_size: int = _TILE

    def __post_init__(self):
        tile, n_prefix = self.tile_size, self.num_prefix_tiles
        if tile != _TILE:
            raise ValueError(_OP + "only tile size 64 is supported.")
        if len(self.video_tile_spans) > 1:
            raise ValueError(
                _OP + "more than one video region (Ref2VA) is not supported."
            )
        vbs, untile = self.variable_block_sizes, self.untile_index
        spans = tuple(tuple(span) for span in self.video_tile_spans)
        if len(spans) == 1 and spans[0] != (n_prefix, vbs.shape[0]):
            raise ValueError(
                _OP + "video_tile_spans must be empty or ((num_prefix_tiles, tiles),)."
            )
        if mx.any((vbs < 1) | (vbs > tile)).item():
            raise ValueError(_OP + "variable_block_sizes must be in [1, 64].")
        rows = untile.shape[0]
        live = (mx.arange(tile)[None, :] < vbs[:, None]).reshape(-1)
        if not _maps_rows_onto_live_slots(untile, live):
            raise ValueError(
                _OP + "untile_index must map the packed rows one to one onto the "
                "live tile slots of variable_block_sizes."
            )
        if not _inverts(self.tile_gather_index, untile, live.shape[0]):
            raise ValueError(
                _OP + "tile_gather_index must hold the packed row of each live "
                "slot and S in each pad slot."
            )
        in_prefix = (untile // tile < n_prefix).astype(mx.int32)
        if mx.any(in_prefix[1:] > in_prefix[:-1]).item():
            raise ValueError(_OP + "the dense rows must come before the video region.")
        held = mx.sum(in_prefix).item()
        if held != self.prefix_length:
            raise ValueError(
                _OP + f"num_prefix_tiles and prefix_length disagree: the first "
                f"{n_prefix} tiles hold {held} rows, not {self.prefix_length}."
            )
        head = self.tile_gather_index[: n_prefix * tile]
        want = mx.where(head == rows, self.prefix_length, head)
        if not mx.array_equal(self.prefix_gather_index, want).item():
            raise ValueError(
                _OP + "prefix_gather_index must hold the packed row of each "
                "prefix tile slot and prefix_length in each pad slot."
            )


# Copied from FastVideo, fastvideo/mlx_runtime/minimax_h3_vsa.py
# (Apache-2.0): compute_topk, region_sparsities, _region_plan and
# _block_indices_from_scores, reformatted. VSAGeometry refuses more than one
# region and vsa_attention refuses compete mode before it routes.


def compute_topk(sparsity: float, num_blocks: int) -> int:
    """Blocks to keep for a sparsity level, clamped to [1, num_blocks]."""
    if num_blocks <= 0:
        return 0
    return max(1, min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks))


def region_sparsities(
    num_regions: int, sparsity: float, ref_keep_rate: float | None
) -> tuple[float, ...]:
    """Per-region sparsity: reference videos keep ``ref_keep_rate``, the target
    follows ``sparsity``.

    Mirrors ``MiniMaxH3VSAMetadataBuilder.build``: a dense step (sparsity <= 0)
    keeps every region dense, whatever the reference keep rate.
    """
    if num_regions < 1:
        raise ValueError("VSA-H3 needs at least one video region.")
    if sparsity <= 0.0:
        return (0.0,) * num_regions
    if num_regions == 1:
        return (float(sparsity),)
    if ref_keep_rate is None:
        raise ValueError(
            f"A VSA-H3 build with {num_regions - 1} reference-video region(s) "
            "needs ref_keep_rate."
        )
    return (1.0 - float(ref_keep_rate),) * (num_regions - 1) + (float(sparsity),)


def _region_plan(
    num_prefix_tiles: int,
    num_video_tiles: int,
    sparsity: float,
    exempt: bool,
    spans: tuple[tuple[int, int], ...] | None,
    span_sparsities: tuple[float, ...] | None,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """(spans, per-span top-k); a missing plan is one region over every video tile."""
    if spans is None:
        spans = ((num_prefix_tiles, num_prefix_tiles + num_video_tiles),)
    if span_sparsities is None:
        span_sparsities = region_sparsities(len(spans), sparsity, None)
    if len(span_sparsities) != len(spans):
        raise ValueError(
            f"{len(span_sparsities)} span sparsities for {len(spans)} video spans."
        )
    if len(spans) > 1 and not exempt and num_prefix_tiles:
        raise ValueError(
            "vsa prefix_mode='compete' supports only the generated-video region; "
            "use prefix_mode='exempt' with reference-video regions."
        )
    topks = tuple(
        compute_topk(sp, end - start)
        for (start, end), sp in zip(spans, span_sparsities, strict=True)
    )
    return spans, topks


def _block_indices_from_scores(
    scores,
    num_prefix_tiles: int,
    num_video_tiles: int,
    sparsity: float,
    exempt: bool,
    spans: tuple[tuple[int, int], ...] | None = None,
    span_sparsities: tuple[float, ...] | None = None,
    query_scores_only: bool = False,
):
    """Return (block_idx [H, n_video_tiles, k_sel], block_num [H, n_video_tiles]).

    ``k_sel`` is the same for every query: all prefix tiles plus each video
    region's own top-k, so the gather path keeps one shape. With
    ``query_scores_only`` the rows of ``scores`` are already the video query
    tiles to route (a subset of the video tiles), not the full tile grid.
    """
    import mlx.core as mx

    n_tiles = scores.shape[-1]
    heads = scores.shape[0]
    spans, topks = _region_plan(
        num_prefix_tiles, num_video_tiles, sparsity, exempt, spans, span_sparsities
    )
    video_scores = scores if query_scores_only else scores[:, num_prefix_tiles:, :]
    num_video_tiles = int(video_scores.shape[1])
    if all(k == end - start for k, (start, end) in zip(topks, spans, strict=True)):
        idx = mx.broadcast_to(
            mx.arange(n_tiles, dtype=mx.int32)[None, None, :],
            (heads, num_video_tiles, n_tiles),
        )
        counts = mx.full((heads, num_video_tiles), n_tiles, dtype=mx.int32)
        return idx, counts

    if exempt or num_prefix_tiles == 0:
        parts = []
        if num_prefix_tiles:
            parts.append(
                mx.broadcast_to(
                    mx.arange(num_prefix_tiles, dtype=mx.int32)[None, None, :],
                    (heads, num_video_tiles, num_prefix_tiles),
                )
            )
        for (start, end), k_span in zip(spans, topks, strict=True):
            if k_span == end - start:
                parts.append(
                    mx.broadcast_to(
                        mx.arange(start, end, dtype=mx.int32)[None, None, :],
                        (heads, num_video_tiles, end - start),
                    )
                )
                continue
            cols = video_scores[:, :, start:end]
            parts.append(
                mx.argpartition(-cols, kth=k_span - 1, axis=-1)[:, :, :k_span].astype(
                    mx.int32
                )
                + start
            )
        idx = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
    else:
        k_sel = min(topks[0] + num_prefix_tiles, n_tiles)
        idx = mx.argpartition(-video_scores, kth=k_sel - 1, axis=-1)[
            :, :, :k_sel
        ].astype(mx.int32)
    counts = mx.full((heads, num_video_tiles), idx.shape[-1], dtype=mx.int32)
    return idx, counts


def _tile(x, geometry):
    """Packed [S, heads, D] rows into tiles, every pad slot reading one zero
    row."""
    zero = mx.zeros((1,) + x.shape[1:], dtype=x.dtype)
    return mx.concatenate([x, zero])[geometry.tile_gather_index]


def _pool(x, variable_block_sizes, tile):
    """float32 mean of each tile: [tiles * tile, heads, D] -> [heads, tiles, D]."""
    n_tiles = x.shape[0] // tile
    pooled = x.astype(mx.float32).reshape(n_tiles, tile, *x.shape[1:]).sum(axis=1)
    pooled = pooled / variable_block_sizes.astype(mx.float32)[:, None, None]
    return pooled.transpose(1, 0, 2)


def _dense(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(
        q.transpose(1, 0, 2)[None],
        k.transpose(1, 0, 2)[None],
        v.transpose(1, 0, 2)[None],
        scale=scale,
    )[0].transpose(1, 0, 2)


def _gate_term(coarse, gate_tiled, tiles, tile):
    """bf16(coarse * gate) of the first tiles tiles: [tiles * tile, heads, D]."""
    heads, dim = gate_tiled.shape[1:]
    out_c = coarse[:, :tiles].transpose(1, 0, 2)
    gate_view = gate_tiled[: tiles * tile].reshape(tiles, tile, heads, dim)
    term = (out_c[:, None, :, :] * gate_view).reshape(tiles * tile, heads, dim)
    return term.astype(gate_tiled.dtype)


def vsa_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    geometry: VSAGeometry,
    sparsity: float,
    *,
    gate: mx.array | None = None,
    exempt: bool = True,
    query_rows: int | None = None,
) -> mx.array:
    """MiniMax-H3 video sparse attention over one packed sequence.

    The result equals FastVideo's ``h3_vsa_attention`` with ``impl="nax"``
    in exempt mode, bit for bit. Each video query tile attends every prefix
    tile and its top-k video tiles by pooled score, through
    ``vsa_block_sparse``. The prefix query rows attend every key. With a
    gate, each row also adds the pooled-compression term. At sparsity 0 the
    call is dense attention, plus the gate term when a gate is given.

    Args:
        q (array): packed queries [S, heads, 128], bfloat16.
        k (array): packed keys, the shape and dtype of q.
        v (array): packed values, the shape and dtype of q.
        geometry (VSAGeometry): the tile map of the sequence.
        sparsity (float): the fraction of video tiles a video query tile
            skips. 0 or less is dense.
        gate (array, optional): the to_gate_compress projection, the shape
            and dtype of q.
        exempt (bool): prefix key tiles are always kept. Compete mode
            (False) raises.
        query_rows (int, optional): passed to ``vsa_block_sparse``.

    Returns:
        array: packed [S, heads, 128] bfloat16.

    Compete mode and a head dim other than 128 raise ValueError. Building
    the geometry refuses dense rows after the video region, more than one
    video region (Ref2VA) and a tile size other than 64. Build the geometry
    outside a compiled function. A sparse call needs a NAX GPU (M5 and
    later), as ``vsa_block_sparse`` does. A dense call runs on any Metal
    GPU.

    The call is for inference only: mx.grad through it raises. A compiled
    function holds the geometry it traced, and MLX reuses a trace for every
    call at the same input shapes, so two geometries of one length, such as
    a landscape and a portrait grid, would share a trace. Compile one
    function per geometry. A function compiled with shapeless=True would
    replay the traced geometry at other shapes too, so it cannot hold the
    call: it raises ValueError.
    """
    arrays = (q, k, v) if gate is None else (q, k, v, gate)
    if (
        q.ndim != 3
        or q.shape[2] != _HEAD_DIM
        or any(a.shape != q.shape or a.dtype != mx.bfloat16 for a in arrays)
    ):
        raise ValueError(
            _OP + "q, k, v and gate must be bfloat16 [S, heads, 128] of one shape."
        )
    tile = geometry.tile_size
    rows = geometry.untile_index.shape[0]
    if q.shape[0] != rows:
        raise ValueError(
            _OP + f"the geometry is for {rows} packed rows, got {q.shape[0]}."
        )
    if not exempt:
        raise ValueError(_OP + "compete mode (exempt=False) is not supported.")
    q = _vsa_fixed_shapes(q)
    heads, dim = q.shape[1:]
    scale = dim**-0.5
    if sparsity <= 0.0 and gate is None:
        return _dense(q, k, v, scale)

    vbs = geometry.variable_block_sizes
    n_tiles = vbs.shape[0]
    n_prefix = geometry.num_prefix_tiles
    q_tiled, k_tiled, v_tiled = (_tile(x, geometry) for x in (q, k, v))
    q_pooled = _pool(q_tiled, vbs, tile)
    k_pooled = _pool(k_tiled, vbs, tile)
    scores = (q_pooled @ k_pooled.transpose(0, 2, 1)) / (dim**0.5)
    coarse = gate_tiled = None
    if gate is not None:
        gate_tiled = _tile(gate, geometry)
        coarse = mx.softmax(scores, axis=-1) @ _pool(v_tiled, vbs, tile)
    if sparsity <= 0.0:
        out = _tile(_dense(q, k, v, scale), geometry)
        out = out + _gate_term(coarse, gate_tiled, n_tiles, tile)
        return out[geometry.untile_index]

    spans = geometry.video_tile_spans or ((n_prefix, n_tiles),)
    span_sparsities = region_sparsities(len(spans), sparsity, None)
    block_idx, block_num = _block_indices_from_scores(
        scores, n_prefix, n_tiles - n_prefix, sparsity, exempt, spans, span_sparsities
    )
    prefix_rows = geometry.prefix_length
    prefix_out = _dense(q[:prefix_rows], k, v, scale) if prefix_rows else q[:0]
    video = vsa_block_sparse(
        q_tiled,
        k_tiled,
        v_tiled,
        block_idx,
        block_num,
        vbs,
        n_prefix,
        scale,
        tile_size=tile,
        query_rows=query_rows,
        coarse=coarse,
        gate=gate_tiled,
    )[n_prefix * tile :]
    zero = mx.zeros((1, heads, dim), dtype=q.dtype)
    prefix = mx.concatenate([prefix_out, zero])[geometry.prefix_gather_index]
    if gate is not None:
        prefix = prefix + _gate_term(coarse, gate_tiled, n_prefix, tile)
    return mx.concatenate([prefix, video], axis=0)[geometry.untile_index]
