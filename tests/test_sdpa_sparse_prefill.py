#!/usr/bin/env python3
"""kq.sdpa_sparse_prefill against an independent f32 reference.

Query l sits at window row S - L + l and reads the band [pos - band + 1,
pos] plus its listed pool rows; the reference builds that key set per
query, drops masked and out-of-range pool entries, and runs a plain f32
softmax with the per-head sink in the denominator. Covers both float
dtypes, both index dtypes, head dims 128/256/512, a band the early queries
cannot fill, a batch with broadcast window and pool, a key count past one
row-table pass, and every instantiated configuration.

A 512 latent, with or without the 64-wide score-only tail, has two kernels:
the NAX tile kernel on a tensor-op GPU and the simdgroup kernel under
KQ_SDPA_SPARSE_NAX=0. The tests that reach a 512 latent run both.

Metal-only kernel (eval_cpu throws): skipped under KQUANT_FORCE_CPU.
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

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq.sdpa_sparse_prefill is a Metal-only kernel; no CPU path.",
)

CFGS = ["16,4,8", "32,4,8", "16,2,8"]
LEVER = "KQ_SDPA_SPARSE_NAX"


@pytest.fixture(params=["nax", "simd"])
def kernel(request, monkeypatch):
    """Each kernel of a 512 latent in turn."""
    if request.param == "simd":
        monkeypatch.setenv(LEVER, "0")
    elif not kq.nax_available():
        pytest.skip("no tensor-op units on this GPU")
    else:
        monkeypatch.delenv(LEVER, raising=False)
    return request.param


def _ref(q, win, pool, idx, scale, band, sinks, smask):
    """f32 numpy reference: q [B,H,L,D], win [B|1,1,S,D], pool [B|1,P,D],
    idx [B|1,L,N]; smask [B|1, L, N] bool or None."""
    B, H, L, D = q.shape
    S, P, N = win.shape[2], pool.shape[1], idx.shape[2]
    koff = S - L
    out = np.zeros((B, H, L, D), np.float32)
    for b in range(B):
        wb = win[b % win.shape[0], 0]
        pb = pool[b % pool.shape[0]]
        ib = idx[b % idx.shape[0]]
        for l in range(L):
            pos = koff + l
            rows = [wb[j] for j in range(max(0, pos - band + 1), pos + 1)]
            for n in range(N):
                if smask is not None and not smask[b % smask.shape[0], l, n]:
                    continue
                r = int(ib[l, n])
                if r < 0 or r >= P:
                    continue
                rows.append(pb[r])
            keys = np.stack(rows, 0)
            s = (q[b, :, l] * scale) @ keys.T
            if sinks is not None:
                s = np.concatenate([sinks[:, None], s], 1)
            s = s - s.max(1, keepdims=True)
            w = np.exp(s)
            w = w / w.sum(1, keepdims=True)
            if sinks is not None:
                w = w[:, 1:]
            out[b, :, l] = w @ keys
    return out


def _case(
    B, H, L, D, S, P, N, band, dtype, idtype, sinks, mask, pad, bcast=False, seed=0
):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, L, D)).astype(np.float32)
    wb = 1 if bcast else B
    win = rng.standard_normal((wb, 1, S, D)).astype(np.float32)
    pool = rng.standard_normal((wb, P, D)).astype(np.float32)
    idx = (
        np.stack([rng.permutation(P)[:N] for _ in range(wb * L)], 0)
        .reshape(wb, L, N)
        .astype(np.int64)
    )
    if pad and N >= 3:
        idx[..., -3:] = np.array([-1, P, P + 7]) if idtype == np.int32 else P
    idx = idx.astype(idtype)
    sk = rng.standard_normal(H).astype(np.float32) if sinks else None
    sm = rng.random((B, L, N)) > 0.3 if mask else None
    qm = mx.array(q).astype(dtype)
    winm = mx.array(win).astype(dtype)
    poolm = mx.array(pool).astype(dtype)
    # the kernel reads the rounded inputs; the reference must too
    qf = np.array(qm.astype(mx.float32))
    winf = np.array(winm.astype(mx.float32))
    poolf = np.array(poolm.astype(mx.float32))
    skm = mx.array(sk).astype(dtype) if sinks else None
    skf = np.array(skm.astype(mx.float32)) if sinks else None
    scale = D**-0.5
    ref = _ref(qf, winf, poolf, idx, scale, band, skf, sm)
    kw = {}
    if sinks:
        kw["sinks"] = skm
    if mask:
        kw["sel_mask"] = mx.array(sm)
    return (qm, winm, poolm, mx.array(idx), scale, band, kw), ref


def _check(args, ref, tol):
    q, win, pool, idx, scale, band, kw = args
    if q.shape[-1] == 576:
        # Rows 576 wide: all of a row scores, the first 512 values come out.
        kw = dict(kw, score_tail=64)
        ref = ref[..., :512]
    got = kq.sdpa_sparse_prefill(q, win, pool, idx, scale, band, **kw)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    assert g.shape == ref.shape
    err = float(np.abs(g - ref).max())
    assert err < tol, err


@pytest.mark.parametrize("dtype,tol", [(mx.float16, 3e-3), (mx.bfloat16, 2e-2)])
@pytest.mark.parametrize("idtype", [np.int32, np.uint32])
@pytest.mark.parametrize("D", [128, 256, 512])
def test_prefill_matches_reference(dtype, tol, idtype, D, kernel):
    # 40 history rows, band 128: the first queries read fewer window rows
    # than the band; the last read the full band.
    args, ref = _case(
        1,
        64,
        100,
        D,
        140,
        700,
        512,
        128,
        dtype,
        idtype,
        sinks=True,
        mask=False,
        pad=False,
    )
    _check(args, ref, tol)


@pytest.mark.parametrize("cfg", CFGS)
def test_every_configuration(monkeypatch, cfg):
    # The configurations are the simdgroup kernel's.
    monkeypatch.setenv(LEVER, "0")
    monkeypatch.setenv("KQ_SDPA_SPARSE_PREFILL_CFG", cfg)
    args, ref = _case(
        1,
        64,
        70,
        512,
        90,
        600,
        48,
        16,
        mx.float16,
        np.int32,
        sinks=True,
        mask=True,
        pad=True,
        seed=3,
    )
    _check(args, ref, 3e-3)


def test_batch_broadcast_masks_and_padding():
    args, ref = _case(
        2,
        12,
        9,
        128,
        30,
        96,
        24,
        8,
        mx.float16,
        np.int32,
        sinks=True,
        mask=True,
        pad=True,
        bcast=True,
        seed=5,
    )
    _check(args, ref, 3e-3)
    args, ref = _case(
        2,
        12,
        9,
        128,
        30,
        96,
        24,
        8,
        mx.float16,
        np.uint32,
        sinks=True,
        mask=True,
        pad=True,
        bcast=False,
        seed=6,
    )
    _check(args, ref, 3e-3)


def test_no_sinks_window_only_and_a_short_band():
    args, ref = _case(
        1,
        8,
        5,
        256,
        5,
        16,
        0,
        3,
        mx.float16,
        np.int32,
        sinks=False,
        mask=False,
        pad=False,
    )
    _check(args, ref, 3e-3)


def test_keys_past_one_table_pass():
    # 128 window rows + 1100 listed rows: two row-table passes.
    args, ref = _case(
        1,
        16,
        6,
        128,
        200,
        2048,
        1100,
        128,
        mx.float16,
        np.int32,
        sinks=True,
        mask=True,
        pad=False,
        seed=9,
    )
    _check(args, ref, 3e-3)


def test_validation():
    q = mx.zeros((1, 4, 3, 128), mx.float16)
    win = mx.zeros((1, 1, 8, 128), mx.float16)
    pool = mx.zeros((1, 8, 128), mx.float16)
    idx = mx.zeros((1, 3, 4), mx.int32)
    with pytest.raises(ValueError, match="head_dim"):
        kq.sdpa_sparse_prefill(
            mx.zeros((1, 4, 3, 64), mx.float16),
            mx.zeros((1, 1, 8, 64), mx.float16),
            mx.zeros((1, 8, 64), mx.float16),
            idx,
            0.1,
            4,
        )
    with pytest.raises(ValueError, match="at least L rows"):
        kq.sdpa_sparse_prefill(q, win[:, :, :2], pool, idx, 0.1, 4)
    with pytest.raises(ValueError, match="band"):
        kq.sdpa_sparse_prefill(q, win, pool, idx, 0.1, 0)
    with pytest.raises(ValueError, match="int32 or uint32"):
        kq.sdpa_sparse_prefill(q, win, pool, idx.astype(mx.int64), 0.1, 4)
    with pytest.raises(ValueError, match="sinks"):
        kq.sdpa_sparse_prefill(q, win, pool, idx, 0.1, 4, sinks=mx.zeros((3,)))
    with pytest.raises(ValueError, match="sel_mask"):
        kq.sdpa_sparse_prefill(
            q, win, pool, idx, 0.1, 4, sel_mask=mx.zeros((1, 7), mx.bool_)
        )


@pytest.mark.parametrize("D", [128, 512])
@pytest.mark.parametrize("B,bcast", [(1, False), (2, True)])
def test_packed_pool_matches_fp16_pool(D, B, bcast, monkeypatch):
    from test_sdpa_sparse_decode import _on_grid_pool

    # A packed pool runs the simdgroup kernel, so the fp16 pool it is held
    # to bit for bit runs it too.
    monkeypatch.setenv(LEVER, "0")
    args, _ = _case(
        B,
        16,
        70,
        D,
        90,
        600,
        48,
        16,
        mx.float16,
        np.int32,
        sinks=True,
        mask=True,
        pad=True,
        bcast=bcast,
        seed=D,
    )
    q, win, pool, idx, scale, band, kw = args
    pool = _on_grid_pool(pool)
    codes, scales = kq.latent_fp4_pack(pool)
    want = kq.sdpa_sparse_prefill(q, win, pool, idx, scale, band, **kw)
    got = kq.sdpa_sparse_prefill(
        q, win, codes, idx, scale, band, pool_scales=scales, **kw
    )
    assert mx.array_equal(got, want).item()
    P = pool.shape[1]
    nb = pool.shape[0]
    cbuf = mx.concatenate([codes, mx.zeros((nb, 50, D // 2), mx.uint8)], axis=1)
    sbuf = mx.concatenate([scales, mx.zeros((nb, 50, D // 16), mx.uint8)], axis=1)
    got = kq.sdpa_sparse_prefill(
        q, win, cbuf[:, :P], idx, scale, band, pool_scales=sbuf[:, :P], **kw
    )
    assert mx.array_equal(got, want).item()


@pytest.mark.parametrize("D", [128, 576])
@pytest.mark.parametrize("B", [1, 2])
def test_strided_window_and_pool_match_contiguous(B, D):
    """Prefix slices of larger buffers pass by stride; a transposed view
    with a non-unit feature stride is copied at eval. All match exactly."""
    args, _ = _case(
        B,
        64,
        100,
        D,
        140,
        700,
        512,
        128,
        mx.float16,
        np.int32,
        sinks=True,
        mask=True,
        pad=True,
    )
    q, win, pool, idx, scale, band, kw = args
    if D == 576:
        kw = dict(kw, score_tail=64)
    want = kq.sdpa_sparse_prefill(q, win, pool, idx, scale, band, **kw)
    S = win.shape[2]
    P = pool.shape[1]

    def z(*s):
        return mx.zeros(s, dtype=pool.dtype)

    wbuf = mx.concatenate([win, z(B, 1, 64, D)], axis=2)
    pbuf = mx.concatenate([z(B, 5, D), pool, z(B, 300, D)], axis=1)
    pt = mx.array(np.ascontiguousarray(np.swapaxes(np.array(pool), 1, 2)))
    for w, p in [
        (wbuf[:, :, :S], pbuf[:, 5 : 5 + P]),
        (win, mx.swapaxes(pt, 1, 2)),
    ]:
        got = kq.sdpa_sparse_prefill(q, w, p, idx, scale, band, **kw)
        assert mx.array_equal(got, want).item()


@pytest.mark.parametrize("dtype,tol", [(mx.float16, 3e-3), (mx.bfloat16, 2e-2)])
def test_score_tail_counts_in_the_score_and_has_no_output(dtype, tol, kernel):
    args, ref = _case(
        2,
        32,
        70,
        576,
        110,
        600,
        300,
        4,
        dtype,
        np.int32,
        sinks=True,
        mask=True,
        pad=True,
    )
    _check(args, ref, tol)


# B, H, L, D, S, P, N, band, sinks, mask, pad, bcast. D 576 carries the tail.
LATENT_512 = {
    # A sparse layer's block: the query's own row and 2051 listed rows.
    "sparse_block": (1, 32, 70, 576, 110, 4096, 2051, 1, True, False, True, False),
    # Heads short of a strip in the second strip, keys short of a tile.
    "short_strip": (1, 40, 9, 512, 30, 200, 37, 8, True, True, True, False),
    "window_only": (1, 32, 5, 512, 5, 16, 0, 3, False, False, False, False),
    # One query: the keys fan out over splits, as at decode.
    "one_query": (1, 64, 1, 576, 600, 700, 300, 512, True, True, False, False),
    "batch_broadcast": (2, 32, 20, 512, 50, 300, 100, 16, True, True, True, True),
    # Queries enough to fill the GPU: one split, the early ones short of
    # the band.
    "long_block": (1, 32, 200, 576, 200, 900, 130, 4, True, True, True, False),
}


@pytest.mark.parametrize("dtype,tol", [(mx.float16, 3e-3), (mx.bfloat16, 2e-2)])
@pytest.mark.parametrize("name", list(LATENT_512))
def test_a_512_latent_on_both_kernels(name, dtype, tol, kernel):
    B, H, L, D, S, P, N, band, sinks, mask, pad, bcast = LATENT_512[name]
    args, ref = _case(
        B,
        H,
        L,
        D,
        S,
        P,
        N,
        band,
        dtype,
        np.int32,
        sinks=sinks,
        mask=mask,
        pad=pad,
        bcast=bcast,
        seed=len(name),
    )
    _check(args, ref, tol)


def test_a_dead_row_that_is_not_finite_never_counts(kernel):
    # Window row 0 lies outside every query's band, holds inf and nan, and
    # is the row a padded slot loads; a pool row every query masks is nan.
    B, H, L, D, S, P, N, band = 1, 32, 6, 512, 40, 96, 24, 4
    args, _ = _case(
        B,
        H,
        L,
        D,
        S,
        P,
        N,
        band,
        mx.float16,
        np.int32,
        sinks=True,
        mask=True,
        pad=True,
    )
    q, win, pool, idx, scale, band, kw = args
    idn = np.array(idx)
    sm = np.array(kw["sel_mask"])
    bad = int(idn[0, 0, 5])
    sm &= idn != bad
    winf = np.array(win.astype(mx.float32))
    poolf = np.array(pool.astype(mx.float32))
    win_bad, pool_bad = winf.copy(), poolf.copy()
    win_bad[:, :, 0, ::2] = np.inf
    win_bad[:, :, 0, 1::2] = np.nan
    pool_bad[:, bad] = np.nan
    got = kq.sdpa_sparse_prefill(
        q,
        mx.array(win_bad).astype(mx.float16),
        mx.array(pool_bad).astype(mx.float16),
        idx,
        scale,
        band,
        sinks=kw["sinks"],
        sel_mask=mx.array(sm),
    )
    g = np.array(got.astype(mx.float32))
    ref = _ref(
        np.array(q.astype(mx.float32)),
        winf,
        poolf,
        idn,
        scale,
        band,
        np.array(kw["sinks"].astype(mx.float32)),
        sm,
    )
    assert np.isfinite(g).all()
    assert float(np.abs(g - ref).max()) < 3e-3


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
    q = mx.random.normal((1, 32, 8, d)).astype(mx.float16)
    win = mx.random.normal((1, 1, 40, d)).astype(mx.float16)
    pool = mx.random.normal((1, 64, d)).astype(mx.float16)
    idx = mx.broadcast_to(mx.arange(48, dtype=mx.int32), (1, 8, 48))
    kw = {"score_tail": tail} if tail else {}
    if packed:
        codes, scales = kq.latent_fp4_pack(pool)
        pool, kw = codes, {"pool_scales": scales}
    sys.stderr.write(f"[case] {i}\n")
    sys.stderr.flush()
    mx.eval(kq.sdpa_sparse_prefill(q, win, pool, idx, 0.05, 4, **kw))
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
            got[cur] = []
            continue
        m = _LOG.match(line)
        if m and cur is not None:
            got[cur].append(m.group(1))
    assert len(got) == len(ROUTES), r.stderr[-4000:]
    nax = kq.nax_available()
    for i, route in enumerate(ROUTES):
        took = any("sparse_decode_nax" in n for n in got[i])
        assert took == (route[4] and nax), f"{route}: {got[i]}"
        if route[0] == 576 and took:
            assert any(n.endswith("_t64") for n in got[i]), got[i]


def test_score_tail_is_refused_at_other_widths():
    args, _ = _case(
        1,
        16,
        8,
        512,
        16,
        32,
        8,
        4,
        mx.float16,
        np.int32,
        sinks=False,
        mask=False,
        pad=False,
    )
    q, win, pool, idx, scale, band, kw = args
    with pytest.raises(ValueError):
        kq.sdpa_sparse_prefill(q, win, pool, idx, scale, band, score_tail=64)
    with pytest.raises(ValueError):
        kq.sdpa_sparse_prefill(q, win, pool, idx, scale, band, score_tail=32)
