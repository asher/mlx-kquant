"""kq.sdpa_sparse_decode at a 512 latent: both split kernels, and the routing.

A tensor-op GPU runs a 512 latent, with or without the 64-wide score-only
tail, on the NAX tile kernel (32-head strips, 16-key tiles), and
KQ_SDPA_SPARSE_NAX=0 keeps the simdgroup kernel. Both are held to the f32
reference of test_sdpa_sparse_decode at the shapes of an absorbed-MLA
decode step and at the edges of the NAX layout: heads short of a strip,
keys short of a tile, a split whose every slot is masked, padded slots,
several queries, two batch rows.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from test_sdpa_sparse_decode import _case, _ref

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or mx.default_device() != mx.gpu,
    reason="kq.sdpa_sparse_decode is a Metal-only kernel; no CPU path.",
)

LEVER = "KQ_SDPA_SPARSE_NAX"
TOL = {mx.float16: 3e-3, mx.bfloat16: 2e-2}


def _run(args, lever, **extra):
    q, win, pool, idx, scale, kw = args
    before = os.environ.get(LEVER)
    if lever is None:
        os.environ.pop(LEVER, None)
    else:
        os.environ[LEVER] = lever
    try:
        got = kq.sdpa_sparse_decode(q, win, pool, idx, scale, **kw, **extra)
        mx.eval(got)
    finally:
        if before is None:
            os.environ.pop(LEVER, None)
        else:
            os.environ[LEVER] = before
    return np.array(got.astype(mx.float32))


# B, H, L, D, W, P, N, sinks, masks, pad, splits. D 576 carries the tail.
SHAPES = {
    # A sparse layer's step: a few window rows, 2048 listed pool rows.
    "sparse_step": (1, 32, 1, 576, 3, 4096, 2048, True, False, False, 0),
    # Heads short of a strip in the second strip, keys short of a tile.
    "short_strip": (1, 40, 1, 576, 96, 700, 300, True, True, True, 0),
    "few_heads": (2, 12, 3, 512, 40, 96, 24, True, True, True, 7),
    "one_split": (1, 32, 2, 512, 96, 700, 300, False, True, True, 1),
    "most_splits": (2, 64, 1, 576, 33, 500, 200, True, False, True, 32),
    "no_window": (1, 16, 1, 512, 0, 64, 40, True, False, True, 0),
    "batch_tail": (2, 32, 3, 576, 40, 96, 24, True, True, True, 0),
    # Queries enough to fill the GPU: one split, which writes the output.
    "direct_block": (1, 32, 40, 576, 50, 300, 70, True, True, True, 0),
    "direct_short_strip": (1, 40, 20, 576, 33, 500, 200, True, False, True, 1),
    "direct_no_sinks": (2, 64, 2, 512, 96, 700, 300, False, True, True, 1),
}


@pytest.mark.parametrize("lever", [None, "0"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("idtype", [np.int32, np.uint32])
@pytest.mark.parametrize("name", list(SHAPES))
def test_matches_reference(name, idtype, dtype, lever):
    B, H, L, D, W, P, N, sinks, masks, pad, splits = SHAPES[name]
    args, ref = _case(B, H, L, D, W, P, N, dtype, idtype, sinks, masks, pad)
    extra = {"splits": splits}
    if D == 576:
        extra["score_tail"] = 64
    got = _run(args, lever, **extra)
    want = ref[..., :512]
    assert got.shape == want.shape
    err = float(np.abs(got - want).max())
    assert err < TOL[dtype], err


@pytest.mark.parametrize("lever", [None, "0"])
def test_window_step_with_no_listed_row(lever):
    # A sliding-window layer's step: 512 window rows and one padded slot.
    rng = np.random.default_rng(3)
    H, W, D = 64, 512, 576
    q = mx.array(rng.standard_normal((1, H, 1, D)), dtype=mx.bfloat16)
    win = mx.array(rng.standard_normal((1, 1, W, D)), dtype=mx.bfloat16)
    sinks = mx.array(rng.standard_normal(H), dtype=mx.bfloat16)
    idx = np.full((1, 1, 1), -1, np.int32)
    scale = 512**-0.5
    args = (q, win, win[:, 0, :1], mx.array(idx), scale, {"sinks": sinks})
    got = _run(args, lever, score_tail=64)
    f32 = [np.array(a.astype(mx.float32)) for a in (q, win, sinks)]
    ref = _ref(f32[0], f32[1], f32[1][:, 0, :1], idx, scale, f32[2], None, None)
    assert float(np.abs(got - ref[..., :512]).max()) < TOL[mx.bfloat16]


@pytest.mark.parametrize("lever", [None, "0"])
def test_a_split_with_every_slot_masked(lever):
    # The keys of whole splits drop out: their partials must weigh zero.
    B, H, L, D, W, P, N = 1, 32, 2, 512, 64, 400, 192
    args, _ = _case(B, H, L, D, W, P, N, mx.float16, np.int32, True, True, False)
    q, win, pool, idx, scale, kw = args
    wm = np.ones((L, W), bool)
    wm[0, :48] = False
    sm = np.ones((B, L, N), bool)
    sm[:, 1, 32:160] = False
    kw = dict(kw, win_mask=mx.array(wm), sel_mask=mx.array(sm))
    got = _run((q, win, pool, idx, scale, kw), lever, splits=16)
    f32 = [np.array(a.astype(mx.float32)) for a in (q, win, pool, kw["sinks"])]
    ref = _ref(f32[0], f32[1], f32[2], np.array(idx), scale, f32[3], wm[None], sm)
    assert float(np.abs(got - ref).max()) < TOL[mx.float16]


@pytest.mark.parametrize("splits", [1, 2, 5])
@pytest.mark.parametrize("lever", [None, "0"])
def test_a_masked_tile_between_live_tiles(lever, splits):
    # Whole 16-key tiles drop out inside a split: the first tile of one
    # query, a middle tile of another, a run of listed rows of both.
    B, H, L, D, W, P, N = 1, 32, 2, 576, 64, 400, 96
    args, _ = _case(B, H, L, D, W, P, N, mx.float16, np.int32, True, True, False)
    q, win, pool, idx, scale, kw = args
    wm = np.ones((L, W), bool)
    wm[0, :16] = False
    wm[1, 16:48] = False
    sm = np.ones((B, L, N), bool)
    sm[:, 0, 32:64] = False
    sm[:, 1, :16] = False
    kw = dict(kw, win_mask=mx.array(wm), sel_mask=mx.array(sm))
    got = _run((q, win, pool, idx, scale, kw), lever, splits=splits, score_tail=64)
    f32 = [np.array(a.astype(mx.float32)) for a in (q, win, pool, kw["sinks"])]
    ref = _ref(f32[0], f32[1], f32[2], np.array(idx), scale, f32[3], wm[None], sm)
    assert float(np.abs(got - ref[..., :512]).max()) < TOL[mx.float16]


@pytest.mark.parametrize("splits", [0, 1])
def test_a_dead_row_that_is_not_finite_never_counts(splits):
    # Window row 0 is masked for every query and holds inf and nan, and it
    # is the row a padded slot loads; a listed pool row is masked and nan.
    # The NAX kernel reads a dead row as zeros. The simdgroup kernel reads
    # its values at weight zero, so there they must be finite.
    if not kq.nax_available():
        pytest.skip("no tensor-op units on this GPU")
    lever = None
    B, H, L, D, W, P, N = 1, 32, 2, 512, 40, 96, 24
    args, _ = _case(B, H, L, D, W, P, N, mx.float16, np.int32, True, True, True)
    q, win, pool, idx, scale, kw = args
    wm = np.array(kw["win_mask"])
    wm[:, 0] = False
    idn = np.array(idx)
    bad = int(idn[0, 0, 5])
    sm = np.array(kw["sel_mask"]) & (idn != bad)
    winf = np.array(win.astype(mx.float32))
    poolf = np.array(pool.astype(mx.float32))
    win_bad, pool_bad = winf.copy(), poolf.copy()
    win_bad[:, :, 0, ::2] = np.inf
    win_bad[:, :, 0, 1::2] = np.nan
    pool_bad[:, bad] = np.nan
    kw = dict(kw, win_mask=mx.array(wm), sel_mask=mx.array(sm))
    got = _run(
        (
            q,
            mx.array(win_bad).astype(mx.float16),
            mx.array(pool_bad).astype(mx.float16),
            idx,
            scale,
            kw,
        ),
        lever,
        splits=splits,
    )
    f32 = [np.array(a.astype(mx.float32)) for a in (q, kw["sinks"])]
    ref = _ref(f32[0], winf, poolf, idn, scale, f32[1], wm[None], sm)
    assert np.isfinite(got).all()
    assert float(np.abs(got - ref).max()) < TOL[mx.float16]


def test_no_window_row_and_no_pool_row_is_refused():
    q = mx.zeros((1, 32, 1, 512), mx.float16)
    with pytest.raises(ValueError):
        kq.sdpa_sparse_decode(
            q,
            mx.zeros((1, 1, 0, 512), mx.float16),
            mx.zeros((1, 0, 512), mx.float16),
            mx.full((1, 1, 4), -1, dtype=mx.int32),
            0.05,
        )


@pytest.mark.parametrize("B", [1, 2])
def test_strided_query_window_and_pool(B):
    # Views of larger buffers: the window and pool pass by stride and
    # offset, the query is made dense before the kernel reads it.
    rng = np.random.default_rng(5)
    H, L, D, W, P, N = 32, 2, 576, 48, 128, 64
    qw = mx.array(rng.standard_normal((B, L, 2 * H, D)), dtype=mx.float16)
    q = qw[:, :, ::2].transpose(0, 2, 1, 3)
    winw = mx.array(rng.standard_normal((B, 1, W + 5, D)), dtype=mx.float16)
    win = winw[:, :, 5:]
    poolw = mx.array(rng.standard_normal((B, P + 3, D)), dtype=mx.float16)
    pool = poolw[:, 3:]
    idx = np.stack([rng.permutation(P)[:N] for _ in range(B * L)]).reshape(B, L, N)
    idx = mx.array(idx.astype(np.int32))
    scale = 512**-0.5
    dense = [mx.contiguous(a) for a in (q, win, pool)]
    mx.eval(dense)
    for lever in (None, "0"):
        got = _run((q, win, pool, idx, scale, {}), lever, score_tail=64)
        want = _run((*dense, idx, scale, {}), lever, score_tail=64)
        assert np.array_equal(got, want)


_CHILD = r"""
import json, os, sys
import mlx.core as mx
import mlx_kquant as kq
for i, (d, tail, packed, lever) in enumerate(json.loads(sys.argv[1])):
    if lever is None:
        os.environ.pop("KQ_SDPA_SPARSE_NAX", None)
    else:
        os.environ["KQ_SDPA_SPARSE_NAX"] = lever
    mx.random.seed(i)
    q = mx.random.normal((1, 32, 1, d)).astype(mx.float16)
    win = mx.random.normal((1, 1, 32, d)).astype(mx.float16)
    pool = mx.random.normal((1, 64, d)).astype(mx.float16)
    idx = mx.arange(48, dtype=mx.int32).reshape(1, 1, 48)
    kw = {"score_tail": tail} if tail else {}
    if packed:
        codes, scales = kq.latent_fp4_pack(pool)
        pool, kw = codes, {"pool_scales": scales}
    sys.stderr.write(f"[case] {i}\n")
    sys.stderr.flush()
    mx.eval(kq.sdpa_sparse_decode(q, win, pool, idx, 0.05, **kw))
"""
_LOG = re.compile(r"^\[kq qmm\] (\S+)")
# D, score_tail, packed pool, lever, takes the NAX kernel on a tensor-op GPU.
ROUTES = [
    (576, 64, False, None, True),
    (512, 0, False, None, True),
    (576, 64, False, "0", False),
    (512, 0, False, "0", False),
    (256, 0, False, None, False),
    (128, 0, False, None, False),
    (512, 0, True, None, False),
]


def test_routing():
    env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    env["KQ_QMM_LOG"] = "1"
    spec = json.dumps([list(r[:4]) for r in ROUTES])
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, spec],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    got, cur = {}, None
    for line in r.stderr.splitlines():
        if line.startswith("[case] "):
            cur = int(line.split()[1])
            continue
        m = _LOG.match(line)
        if m and cur is not None:
            got.setdefault(cur, []).append(m.group(1))
    nax = kq.nax_available()
    for i, (d, _tail, _packed, _lever, want) in enumerate(ROUTES):
        names = got.get(i, [])
        assert names, f"no dispatch logged for {ROUTES[i]}"
        took = any("sparse_decode_nax" in n for n in names)
        assert took == (want and nax), f"{ROUTES[i]}: {names}"
        if d == 576 and took:
            assert any(n.endswith("_t64") for n in names), names
