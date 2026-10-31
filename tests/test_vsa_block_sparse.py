"""vsa_block_sparse: MiniMax-H3 VSA block-sparse attention on the tensor-op
units, against MLX's fused attention over the gathered key tiles with a
key-valid mask (the gather reference of FastVideo's H3 VSA backends).

The kernel walks the same 32-key blocks in the same order with MLX's
fragment math and online softmax, and a fully dead block changes nothing
in MLX's kernel, so the live query rows match bit for bit. Pad query rows
are compared as zeros: the reference fills them, the op does not.

The kernel is Metal-only and needs tensor-op (NAX) hardware: the module
skips under KQUANT_FORCE_CPU and without Metal, and the numeric tests skip
on other GPUs, where the op raises (covered below).
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="vsa_block_sparse is a Metal kernel; no CPU path.",
)

NAX = (
    not os.environ.get("KQUANT_FORCE_CPU")
    and mx.metal.is_available()
    and kq.nax_available()
)

requires_nax = pytest.mark.skipif(
    not NAX, reason="vsa_block_sparse requires tensor-op (NAX) hardware."
)

TILE = 64
D = 128
SCALE = D**-0.5


def _sizes(prefix_rows, video_sizes):
    """(variable_block_sizes, prefix tiles): each prefix segment cut into
    64-row chunks, its last chunk short, then the video tiles."""
    sizes = []
    for rows in prefix_rows:
        full, rem = divmod(rows, TILE)
        sizes += [TILE] * full + ([rem] if rem else [])
    return np.array(sizes + list(video_sizes), dtype=np.int32), len(sizes)


def _live(vbs):
    return (np.arange(TILE)[None, :] < vbs[:, None]).reshape(-1)


def _tiled(rng, vbs, heads, pad=0.0):
    """[tiles * 64, heads, 128] bfloat16; pad slots hold pad."""
    x = rng.standard_normal((len(vbs), TILE, heads, D)).astype(np.float32)
    x[~_live(vbs).reshape(len(vbs), TILE)] = pad
    return mx.array(x.reshape(-1, heads, D)).astype(mx.bfloat16)


def _routing(rng, heads, n_prefix, n_video, k):
    """Every prefix tile, then k distinct video tiles in random order."""
    idx = np.empty((heads, n_video, n_prefix + k), dtype=np.int32)
    idx[..., :n_prefix] = np.arange(n_prefix)
    for h in range(heads):
        for j in range(n_video):
            idx[h, j, n_prefix:] = n_prefix + rng.permutation(n_video)[:k]
    return idx


def _reference(q, k, v, idx, vbs, n_prefix):
    """MLX's fused attention of each video query tile over its gathered key
    tiles, the dead key rows masked: [video tiles * 64, heads, 128]."""
    n, heads = len(vbs), q.shape[1]
    nv, ksel = idx.shape[1], idx.shape[2]

    def rows(x):
        return mx.contiguous(
            x.reshape(n, TILE, heads, D).transpose(2, 0, 1, 3)
        ).reshape(heads * n, TILE * D)

    flat = mx.array((idx + np.arange(heads)[:, None, None] * n).reshape(-1))
    gk = mx.take(rows(k), flat, axis=0).reshape(1, heads * nv, ksel * TILE, D)
    gv = mx.take(rows(v), flat, axis=0).reshape(1, heads * nv, ksel * TILE, D)
    valid = np.arange(TILE)[None, None, None, :] < vbs[idx][..., None]
    mask = mx.array(valid.reshape(1, heads * nv, 1, ksel * TILE))
    qv = q.reshape(n, TILE, heads, D).transpose(2, 0, 1, 3)[:, n_prefix:]
    qv = mx.contiguous(qv.reshape(1, heads * nv, TILE, D))
    out = mx.fast.scaled_dot_product_attention(
        qv, mx.contiguous(gk), mx.contiguous(gv), scale=SCALE, mask=mask
    )[0]
    return (
        out.reshape(heads, nv, TILE, D)
        .transpose(1, 2, 0, 3)
        .reshape(nv * TILE, heads, D)
    )


def _bits(x):
    return np.array(x.view(mx.uint16))


def _check(out, ref, vbs, n_prefix, ref_rows=None):
    """Live video query rows bit-identical to ref; every other row zero."""
    out = _bits(out)
    live = _live(vbs)
    vlive = live[n_prefix * TILE :]
    if ref_rows is None:
        ref_rows = vlive
    video = out[n_prefix * TILE :]
    np.testing.assert_array_equal(video[ref_rows], _bits(ref)[ref_rows])
    assert not out[: n_prefix * TILE].any(), "prefix rows must be zero"
    assert not video[~vlive].any(), "pad rows must be zero"


def _call(q, k, v, idx, num, vbs, n_prefix, **kw):
    return kq.vsa_block_sparse(
        q, k, v, mx.array(idx), mx.array(num), mx.array(vbs), n_prefix, SCALE, **kw
    )


# (prefix segment rows, video tile live rows, heads, video tiles kept,
# query rows per threadgroup). 74 audio rows end in a 10-row prefix tile;
# 40, 17, 1 and 33 are partly dead video tiles.
VIDEO = (64, 64, 40, 64, 17, 64, 1, 64, 33, 64)
CASES = [
    ((56, 74), VIDEO, 3, 4, 32),
    ((56, 74), VIDEO, 5, 1, 16),
    ((56, 74), VIDEO, 7, len(VIDEO), 64),
    ((), VIDEO, 2, 3, 32),
    ((23,), (48,), 1, 1, 32),
    ((130,), (64, 12, 64, 64, 50), 6, 2, 32),
]


@requires_nax
@pytest.mark.parametrize("prefix,video,heads,k_vid,query_rows", CASES)
def test_matches_gather_reference(prefix, video, heads, k_vid, query_rows):
    rng = np.random.default_rng(len(video) * 31 + heads * 7 + k_vid)
    vbs, n_prefix = _sizes(prefix, video)
    q, k, v = (_tiled(rng, vbs, heads) for _ in range(3))
    idx = _routing(rng, heads, n_prefix, len(video), k_vid)
    num = np.full(idx.shape[:2], idx.shape[2], dtype=np.int32)
    out = _call(q, k, v, idx, num, vbs, n_prefix, query_rows=query_rows)
    _check(out, _reference(q, k, v, idx, vbs, n_prefix), vbs, n_prefix)


@requires_nax
def test_pad_slots_are_never_read():
    # NaN in every pad slot of q, k and v: a pad key row read into a score
    # or into P V, or a pad query row stored, would show up.
    rng = np.random.default_rng(3)
    vbs, n_prefix = _sizes((56, 74), VIDEO)
    heads = 3
    clean = [_tiled(rng, vbs, heads) for _ in range(3)]
    nan = [mx.where(mx.array(_live(vbs))[:, None, None], x, mx.nan) for x in clean]
    idx = _routing(rng, heads, n_prefix, len(VIDEO), 5)
    num = np.full(idx.shape[:2], idx.shape[2], dtype=np.int32)
    out = _call(*nan, idx, num, vbs, n_prefix)
    _check(out, _reference(*clean, idx, vbs, n_prefix), vbs, n_prefix)


@requires_nax
def test_entries_past_block_num_are_not_read():
    # Per (head, tile) counts below k with out-of-range garbage after them;
    # a count of 0 leaves the tile's rows zero.
    rng = np.random.default_rng(5)
    vbs, n_prefix = _sizes((56, 74), VIDEO)
    heads, nv = 2, len(VIDEO)
    full = _routing(rng, heads, n_prefix, nv, nv)
    num = rng.integers(1, full.shape[2] + 1, size=full.shape[:2]).astype(np.int32)
    num[1, 4] = 0
    idx = full.copy()
    for h in range(heads):
        for j in range(nv):
            idx[h, j, num[h, j] :] = 10**6
    q, k, v = (_tiled(rng, vbs, heads) for _ in range(3))
    out = _bits(_call(q, k, v, idx, num, vbs, n_prefix))
    live = _live(vbs)[n_prefix * TILE :].reshape(nv, TILE)
    for h in range(heads):
        for j in range(nv):
            rows = slice((n_prefix + j) * TILE, (n_prefix + j + 1) * TILE)
            got = out[rows, h]
            if num[h, j] == 0:
                assert not got.any()
                continue
            sub = full[:, :, : num[h, j]]
            ref = _bits(_reference(q, k, v, sub, vbs, n_prefix))[
                j * TILE : (j + 1) * TILE, h
            ]
            np.testing.assert_array_equal(got[live[j]], ref[live[j]])
            assert not got[~live[j]].any()


@requires_nax
def test_out_of_range_tile_indices_are_skipped():
    rng = np.random.default_rng(7)
    vbs, n_prefix = _sizes((56, 74), VIDEO)
    heads = 3
    q, k, v = (_tiled(rng, vbs, heads) for _ in range(3))
    good = _routing(rng, heads, n_prefix, len(VIDEO), 3)
    bad = np.full(good.shape[:2] + (2,), [-1, len(vbs)], dtype=np.int32)
    # The bad entries sit inside the live count, between the good ones.
    idx = np.concatenate([good[..., :4], bad, good[..., 4:]], axis=-1)
    num = np.full(idx.shape[:2], idx.shape[2], dtype=np.int32)
    out = _call(q, k, v, idx, num, vbs, n_prefix)
    _check(out, _reference(q, k, v, good, vbs, n_prefix), vbs, n_prefix)


@requires_nax
def test_no_video_tile_or_no_key_tile_gives_zeros():
    rng = np.random.default_rng(9)
    vbs, n_prefix = _sizes((56, 74), ())
    q = _tiled(rng, vbs, 2)
    idx = np.zeros((2, 0, 1), dtype=np.int32)
    num = np.zeros((2, 0), dtype=np.int32)
    out = _call(q, q, q, idx, num, vbs, n_prefix)
    assert out.shape == q.shape and not _bits(out).any()
    vbs, n_prefix = _sizes((56,), VIDEO)
    q = _tiled(rng, vbs, 2)
    idx = np.zeros((2, len(VIDEO), 0), dtype=np.int32)
    num = np.zeros((2, len(VIDEO)), dtype=np.int32)
    out = _call(q, q, q, idx, num, vbs, n_prefix)
    assert out.shape == q.shape and not _bits(out).any()


@requires_nax
def test_strided_view_and_query_rows_give_the_same_bits():
    # A strided view of q is copied dense, and the query rows per
    # threadgroup only split the tile.
    rng = np.random.default_rng(11)
    vbs, n_prefix = _sizes((56, 74), VIDEO)
    heads = 4
    q, k, v = (_tiled(rng, vbs, heads) for _ in range(3))
    idx = _routing(rng, heads, n_prefix, len(VIDEO), 3)
    num = np.full(idx.shape[:2], idx.shape[2], dtype=np.int32)
    a = _call(q, k, v, idx, num, vbs, n_prefix)
    wide = mx.concatenate([q, q], axis=1)[:, :heads]
    b = _call(wide, k, v, idx, num, vbs, n_prefix, query_rows=64)
    np.testing.assert_array_equal(_bits(a), _bits(b))


# Host guards. Each input fails one guard, and the match names that guard,
# so a test stays green only while its own guard refuses the call.


def _small():
    rng = np.random.default_rng(13)
    vbs, n_prefix = _sizes((56,), (64, 40))
    q, k, v = (_tiled(rng, vbs, 2) for _ in range(3))
    idx = _routing(rng, 2, n_prefix, 2, 1)
    num = np.full(idx.shape[:2], idx.shape[2], dtype=np.int32)
    return q, k, v, mx.array(idx), mx.array(num), mx.array(vbs), n_prefix


def _guard(match, **change):
    q, k, v, idx, num, vbs, n_prefix = _small()
    args = dict(
        q=q,
        k=k,
        v=v,
        block_idx=idx,
        block_num=num,
        variable_block_sizes=vbs,
        num_prefix_tiles=n_prefix,
        scale=SCALE,
    )
    args.update(change)
    with pytest.raises(ValueError, match=match):
        kq.vsa_block_sparse(**args)


def _heads(x, n):
    return mx.zeros((x.shape[0], n, x.shape[2]), dtype=x.dtype)


GUARDS = {
    "kv_shape": ("share one shape", lambda q, idx, num, vbs: {"k": _heads(q, 3)}),
    "head_dim": (
        "head dim 128",
        lambda q, idx, num, vbs: {
            name: mx.zeros((q.shape[0], 2, 256), dtype=mx.bfloat16) for name in "qkv"
        },
    ),
    "dtype": (
        "must be bfloat16",
        lambda q, idx, num, vbs: {name: q.astype(mx.float16) for name in "qkv"},
    ),
    "tile": ("tile size 64", lambda q, idx, num, vbs: {"tile_size": 128}),
    "query_rows": ("query_rows must be", lambda q, idx, num, vbs: {"query_rows": 8}),
    "scale": ("scale must be finite", lambda q, idx, num, vbs: {"scale": float("inf")}),
    "sizes_dtype": (
        "1-D integer array",
        lambda q, idx, num, vbs: {"variable_block_sizes": vbs.astype(mx.float32)},
    ),
    "rows": (
        "tile_size rows",
        lambda q, idx, num, vbs: {"variable_block_sizes": vbs[1:]},
    ),
    "prefix": (
        "num_prefix_tiles must be",
        lambda q, idx, num, vbs: {"num_prefix_tiles": -1},
    ),
    "block_idx_shape": (
        "block_idx must be",
        lambda q, idx, num, vbs: {"block_idx": mx.concatenate([idx, idx])},
    ),
    "block_num_shape": (
        "block_num must be",
        lambda q, idx, num, vbs: {"block_num": mx.concatenate([num, num], axis=1)},
    ),
    "routing_dtype": (
        "must be integer arrays",
        lambda q, idx, num, vbs: {"block_idx": idx.astype(mx.float32)},
    ),
}


@requires_nax
@pytest.mark.parametrize("name", sorted(GUARDS))
def test_guard_refuses(name):
    match, change = GUARDS[name]
    q, _, _, idx, num, vbs, _ = _small()
    _guard(match, **change(q, idx, num, vbs))


@requires_nax
def test_guard_refuses_a_cpu_stream():
    with mx.stream(mx.cpu):
        _guard("runs on the GPU only")


@requires_nax
def test_guard_refuses_routing_past_32_bits():
    # Lazy inputs: the guard refuses at build time, nothing is evaluated.
    q, k, v, _, _, vbs, n_prefix = _small()
    idx = mx.zeros((2, 2, 2**30), dtype=mx.int32)
    num = mx.zeros((2, 2), dtype=mx.int32)
    with pytest.raises(ValueError, match="too large"):
        kq.vsa_block_sparse(q, k, v, idx, num, vbs, n_prefix, SCALE)


# 2**16 tiles of 512 heads pass 2**32 zero-fill words, and 2**19 heads pass
# 32-bit row offsets inside a key block.
@requires_nax
@pytest.mark.parametrize(
    "n_tiles,heads", [(2**16, 512), (1, 2**19)], ids=["fill_words", "row_offsets"]
)
def test_guard_refuses_rows_past_32_bits(n_tiles, heads):
    big = mx.zeros((n_tiles * TILE, heads, D), dtype=mx.bfloat16)
    vbs = mx.full((n_tiles,), TILE, dtype=mx.int32)
    idx = mx.zeros((heads, n_tiles, 1), dtype=mx.int32)
    num = mx.zeros((heads, n_tiles), dtype=mx.int32)
    with pytest.raises(ValueError, match="too large"):
        kq.vsa_block_sparse(big, big, big, idx, num, vbs, 0, SCALE)


# On an M5 this runs with MLX_METAL_GPU_ARCH=applegpu_g15s, which names an
# M3-class GPU to MLX.
@pytest.mark.skipif(NAX, reason="covers GPUs without tensor-op units")
def test_guard_refuses_a_gpu_without_nax():
    q, k, v, idx, num, vbs, n_prefix = _small()
    with pytest.raises(RuntimeError, match="M5 or later"):
        kq.vsa_block_sparse(q, k, v, idx, num, vbs, n_prefix, SCALE)
