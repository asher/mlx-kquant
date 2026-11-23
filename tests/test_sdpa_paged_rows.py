#!/usr/bin/env python3
"""kq.sdpa_paged_rows against an independent float32 reference.

The reference gathers each query's listed pages row by row in numpy, drops
the rows at or past the query's end, and runs a float32 softmax. It shares no
code with the kernel.

The page lists are the ones a block-sparse attention model builds for a
verify block: every query selects its own complete pages and adds its own
partial tail page. Queries share some pages and differ in others, the
newest rows live in a tail segment, and a page may straddle the two.

The kernel is Metal-only, so the module is skipped under KQUANT_FORCE_CPU.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq.sdpa_paged_rows is a Metal-only kernel; no CPU path.",
)


@pytest.fixture(autouse=True, params=["tensor_ops", "simdgroup"])
def _shared_form(request, monkeypatch):
    """Both shared-tile forms: the tensor-op kernel where the GPU has the
    units (else the simdgroup kernel again), and the simdgroup kernel."""
    monkeypatch.setenv(
        "KQ_PAGED_ROWS_NAX", "1" if request.param == "tensor_ops" else "0"
    )


# rel-norm bound per dtype: bf16 has ~8 mantissa bits, fp16 ~11.
REL_BOUND = {mx.bfloat16: 5e-3, mx.float16: 2e-3}
TILE = {64: 32, 128: 32, 256: 16, 512: 8}


def _f32(a):
    return np.array(a.astype(mx.float32))


def _ref(q, k, v, scale, pages, ends, page_rows):
    """q [B, Hq, L, D], k/v [B, Hkv, N, D] float32 numpy over the whole
    logical key axis, pages [B, L, P] or [B, Hkv, L, P], ends [B, L]."""
    B, Hq, L, D = q.shape
    Hkv = k.shape[1]
    g = Hq // Hkv
    N = k.shape[2]
    out = np.zeros_like(q)
    for b in range(B):
        for h in range(Hq):
            kv = h // g
            for l in range(L):
                pl = pages[b, kv, l] if pages.ndim == 4 else pages[b, l]
                rows = [
                    p * page_rows + i for p in pl if p >= 0 for i in range(page_rows)
                ]
                rows = [r for r in rows if r < min(ends[b, l], N)]
                if not rows:
                    continue
                kk, vv = k[b, kv, rows], v[b, kv, rows]
                s = (kk @ q[b, h, l]) * scale
                w = np.exp(s - s.max())
                out[b, h, l] = (w / w.sum()) @ vv
    return out


def _rel(got, ref):
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12))


def _verify_pages(rng, L, n_sel, offset, page_rows, shared):
    """Page lists of a verify block at ``offset``: query l picks ``n_sel``
    complete pages, ``shared`` of them common to every query, and adds its
    own tail page. Returns (pages [L, n_sel + 1], ends [L])."""
    ends = offset + np.arange(L) + 1
    complete = ends // page_rows
    common = rng.choice(int(complete.min()), size=shared, replace=False)
    pages = np.zeros((L, n_sel + 1), np.int32)
    for l in range(L):
        rest = np.setdiff1d(np.arange(complete[l]), common)
        own = rng.choice(rest, size=n_sel - shared, replace=False)
        pages[l, :n_sel] = rng.permutation(np.concatenate([common, own]))
        pages[l, n_sel] = complete[l]
    return pages, ends.astype(np.int32)


def _make(rng, B, Hq, Hkv, L, N, D, dtype):
    q = mx.array(rng.standard_normal((B, Hq, L, D)).astype(np.float32)).astype(dtype)
    k = mx.array(rng.standard_normal((B, Hkv, N, D)).astype(np.float32)).astype(dtype)
    v = mx.array(rng.standard_normal((B, Hkv, N, D)).astype(np.float32)).astype(dtype)
    return q, k, v


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "D,page_rows,Hq,Hkv",
    [
        (64, 0, 8, 4),
        (128, 0, 16, 2),
        (256, 0, 24, 2),
        (256, 4, 24, 2),
        (512, 0, 8, 1),
        (256, 4, 16, 1),
        (128, 0, 4, 4),
    ],
)
@pytest.mark.parametrize("L", [2, 5, 16])
@pytest.mark.parametrize("window", [0, 3, 32])
def test_paged_rows_base_and_tail(D, page_rows, Hq, Hkv, L, dtype, window):
    # a base plus a tail segment whose first row is not on a page boundary,
    # so one page straddles the two and the verify rows sit in the tail
    rng = np.random.default_rng(D * 131 + L * 7 + Hq)
    pr = page_rows or TILE[D]
    B = 1
    base_len = 20 * pr + 3
    T = 3 * pr + 5
    N = base_len + T
    offset = N - L
    q, k, v = _make(rng, B, Hq, Hkv, L, N, D, dtype)
    pages, ends = _verify_pages(rng, L, 12, offset, pr, shared=6)
    scale = 1.0 / D**0.5
    spare = 7  # the base holds spare rows past base_len the kernel must skip
    junk = mx.zeros((B, Hkv, spare, D), dtype) + 9
    kb = mx.concatenate([k[:, :, :base_len], junk], axis=2)
    vb = mx.concatenate([v[:, :, :base_len], junk], axis=2)
    got = kq.sdpa_paged_rows(
        q,
        kb,
        vb,
        scale,
        mx.array(pages[None]),
        mx.array(ends[None]),
        k[:, :, base_len:],
        v[:, :, base_len:],
        base_len,
        page_rows=page_rows,
        window=window,
    )
    ref = _ref(_f32(q), _f32(k), _f32(v), scale, pages[None], ends[None], pr)
    rel = _rel(_f32(got), ref)
    assert rel < REL_BOUND[dtype], f"rel {rel:.3e}"
    # one buffer holds the same rows: the answer does not depend on the split
    one = kq.sdpa_paged_rows(
        q,
        k,
        v,
        scale,
        mx.array(pages[None]),
        mx.array(ends[None]),
        page_rows=page_rows,
        window=window,
    )
    assert mx.array_equal(one, got).item()


@pytest.mark.parametrize("splits", [0, 1, 3, 32])
@pytest.mark.parametrize("window", [0, 2])
def test_paged_rows_splits_and_padding(splits, window):
    # skipped pages, a list shorter than a tile, any split count
    rng = np.random.default_rng(5)
    B, Hq, Hkv, L, D, pr = 2, 24, 2, 3, 256, 4
    N = 400
    q, k, v = _make(rng, B, Hq, Hkv, L, N, D, mx.bfloat16)
    pages = np.stack(
        [_verify_pages(rng, L, 9, N - L, pr, shared=4)[0] for _ in range(B)]
    )
    ends = np.stack([N - L + np.arange(L) + 1] * B).astype(np.int32)
    pages[0, 1, 2:5] = -1
    pages[1, 0, 0] = -1
    scale = 1.0 / D**0.5
    got = kq.sdpa_paged_rows(
        q,
        k,
        v,
        scale,
        mx.array(pages),
        mx.array(ends),
        splits=splits,
        page_rows=pr,
        window=window,
    )
    ref = _ref(_f32(q), _f32(k), _f32(v), scale, pages, ends, pr)
    rel = _rel(_f32(got), ref)
    assert rel < REL_BOUND[mx.bfloat16], f"rel {rel:.3e}"


@pytest.mark.parametrize("window", [0, 4])
def test_paged_rows_per_head_lists_and_views(window):
    # a page list per kv head, K and V read in place through a strided view
    rng = np.random.default_rng(11)
    B, Hq, Hkv, L, D, pr = 2, 16, 4, 4, 128, 32
    N = 640
    q, k, v = _make(rng, B, Hq, Hkv, L, N, D, mx.float16)
    pages = np.stack(
        [
            [_verify_pages(rng, L, 5, N - L, pr, shared=2)[0] for _ in range(Hkv)]
            for _ in range(B)
        ]
    )
    ends = np.stack([N - L + np.arange(L) + 1] * B).astype(np.int32)
    scale = 1.0 / D**0.5
    wide = mx.concatenate([k, v], axis=2)  # one buffer, two row ranges
    mx.eval(wide)
    got = kq.sdpa_paged_rows(
        q,
        wide[:, :, :N],
        wide[:, :, N:],
        scale,
        mx.array(pages),
        mx.array(ends),
        window=window,
    )
    ref = _ref(_f32(q), _f32(k), _f32(v), scale, pages, ends, pr)
    rel = _rel(_f32(got), ref)
    assert rel < REL_BOUND[mx.float16], f"rel {rel:.3e}"


def test_paged_rows_one_query_matches_paged_decode():
    # a lone query with every row visible is the paged decode kernel's case
    rng = np.random.default_rng(3)
    B, Hq, Hkv, D, pr = 2, 24, 2, 256, 4
    N = 1024
    q, k, v = _make(rng, B, Hq, Hkv, 1, N, D, mx.bfloat16)
    pg = np.stack(
        [rng.choice(N // pr, size=64, replace=False) for _ in range(B)]
    ).astype(np.int32)
    scale = 1.0 / D**0.5
    got = kq.sdpa_paged_rows(q, k, v, scale, mx.array(pg[:, None]), page_rows=pr)
    dec = kq.sdpa_decode_gqa_paged(
        q, k, v, scale, mx.array(np.repeat(pg[:, None], Hkv, axis=1)), tile_c=pr
    )
    assert _rel(_f32(got), _f32(dec)) < REL_BOUND[mx.bfloat16]


def test_paged_rows_rejects_bad_operands():
    q = mx.zeros((1, 24, 2, 256), mx.bfloat16)
    k = mx.zeros((1, 2, 64, 256), mx.bfloat16)
    pages = mx.zeros((1, 2, 3), mx.int32)
    with pytest.raises(ValueError, match="page_rows"):
        kq.sdpa_paged_rows(q, k, k, 1.0, pages, page_rows=8)
    with pytest.raises(ValueError, match="pages must be int32"):
        kq.sdpa_paged_rows(q, k, k, 1.0, mx.zeros((1, 3, 3), mx.int32))
    with pytest.raises(ValueError, match="come together"):
        kq.sdpa_paged_rows(q, k, k, 1.0, pages, None, k)
    with pytest.raises(ValueError, match="base_len"):
        kq.sdpa_paged_rows(q, k, k, 1.0, pages, None, k, k, 65)
    with pytest.raises(ValueError, match="share a dtype"):
        kq.sdpa_paged_rows(q, k.astype(mx.float16), k, 1.0, pages)


@pytest.mark.parametrize("D,page_rows", [(64, 0), (256, 4), (256, 0), (512, 0)])
@pytest.mark.parametrize("L", [1, 3, 8])
@pytest.mark.parametrize("window", [0, 8])
def test_paged_rows_implied_tail_page(D, page_rows, L, window):
    # tail_page adds page ends // page_rows to every list: the same call as
    # listing it, including ends on a page boundary, where it holds no row
    rng = np.random.default_rng(D + L)
    pr = page_rows or TILE[D]
    B, Hq, Hkv = 2, 8, 2
    base_len = 9 * pr + 1
    N = base_len + 2 * pr + 3
    q, k, v = _make(rng, B, Hq, Hkv, L, N, D, mx.bfloat16)
    pages, ends = _verify_pages(rng, L, 6, N - L, pr, shared=3)
    pages = np.stack([pages, pages])
    ends = np.stack([ends, ends - (ends[-1] % pr)])
    pages[1, :, -1] = ends[1] // pr
    scale = 1.0 / D**0.5
    args = (k[:, :, :base_len], v[:, :, :base_len], scale)
    tails = (k[:, :, base_len:], v[:, :, base_len:], base_len)
    listed = kq.sdpa_paged_rows(
        q,
        *args,
        mx.array(pages),
        mx.array(ends),
        *tails,
        page_rows=page_rows,
        window=window,
    )
    implied = kq.sdpa_paged_rows(
        q,
        *args,
        mx.array(np.ascontiguousarray(pages[..., :-1])),
        mx.array(ends),
        *tails,
        page_rows=page_rows,
        tail_page=True,
        window=window,
    )
    assert mx.array_equal(listed, implied).item()


@pytest.mark.parametrize("splits", [1, 16])
def test_paged_rows_shared_form_covers_a_long_key_axis(splits):
    # a split walks its pages 512 at a time: one split covers a key axis of
    # 2100 pages in 5 rounds
    rng = np.random.default_rng(17)
    B, Hq, Hkv, L, D, pr = 1, 24, 2, 8, 256, 4
    N = 2100 * pr + 2
    q, k, v = _make(rng, B, Hq, Hkv, L, N, D, mx.bfloat16)
    pages, ends = _verify_pages(rng, L, 300, N - L, pr, shared=200)
    scale = 1.0 / D**0.5
    got = kq.sdpa_paged_rows(
        q,
        k,
        v,
        scale,
        mx.array(pages[None]),
        mx.array(ends[None]),
        splits=splits,
        page_rows=pr,
        window=8,
    )
    ref = _ref(_f32(q), _f32(k), _f32(v), scale, pages[None], ends[None], pr)
    rel = _rel(_f32(got), ref)
    assert rel < REL_BOUND[mx.bfloat16], f"rel {rel:.3e}"


@pytest.mark.parametrize("splits", [0, 2])
def test_paged_rows_prefill_chunk(splits):
    # a prefill chunk past the sparse boundary: 213 queries (ragged last
    # window), each with its own selected pages and its tail page implied
    rng = np.random.default_rng(23)
    B, Hq, Hkv, L, D, pr = 1, 24, 2, 213, 256, 4
    offset = 2100
    N = offset + L
    q, k, v = _make(rng, B, Hq, Hkv, L, N, D, mx.bfloat16)
    pages, ends = _verify_pages(rng, L, 64, offset, pr, shared=16)
    scale = 1.0 / D**0.5
    got = kq.sdpa_paged_rows(
        q,
        k,
        v,
        scale,
        mx.array(np.ascontiguousarray(pages[None, :, :-1])),
        mx.array(ends[None]),
        splits=splits,
        page_rows=pr,
        tail_page=True,
    )
    ref = _ref(_f32(q), _f32(k), _f32(v), scale, pages[None], ends[None], pr)
    rel = _rel(_f32(got), ref)
    assert rel < REL_BOUND[mx.bfloat16], f"rel {rel:.3e}"


@pytest.mark.parametrize("spare", ["stale", "nan"])
@pytest.mark.parametrize("window,splits", [(0, 0), (4, 0), (4, 4), (8, 1)])
@pytest.mark.parametrize("L", [2, 3, 4, 8])
def test_paged_rows_cache_segments_change_no_bit(window, splits, L, spare):
    # A decode cache read in place: the base ends off a page boundary and
    # both buffers hold spare rows past their live ones (rows a rollback
    # left, or NaN), the tail as many as its capacity leaves. Every bit
    # equals the one-buffer call over the live rows: no split depends on
    # the capacity and no spare row is read.
    rng = np.random.default_rng(L * 10 + window + splits)
    B, Hq, Hkv, D, pr = 1, 24, 2, 256, 4
    N, tl, cap = 8101, 38, 1024
    base_len = N - tl
    q, k, v = _make(rng, B, Hq, Hkv, L, N, D, mx.bfloat16)
    pages, ends = _verify_pages(rng, L, 512, N - L, pr, shared=400)
    scale = 1.0 / D**0.5
    fill = rng.standard_normal((B, Hkv, cap - tl, D)) if spare == "stale" else np.nan
    pad = mx.array(np.broadcast_to(fill, (B, Hkv, cap - tl, D)).astype(np.float32))
    pad = pad.astype(mx.bfloat16)
    kb = mx.concatenate([k[:, :, :base_len], pad[:, :, :61]], axis=2)
    vb = mx.concatenate([v[:, :, :base_len], pad[:, :, :61]], axis=2)
    kt = mx.concatenate([k[:, :, base_len:], pad], axis=2)
    vt = mx.concatenate([v[:, :, base_len:], pad], axis=2)
    lists = (mx.array(np.ascontiguousarray(pages[None, :, :-1])), mx.array(ends[None]))
    opts = dict(splits=splits, page_rows=pr, tail_page=True, window=window)
    seg = kq.sdpa_paged_rows(q, kb, vb, scale, *lists, kt, vt, base_len, **opts)
    one = kq.sdpa_paged_rows(q, k, v, scale, *lists, **opts)
    assert mx.array_equal(seg, one).item()


def test_paged_rows_window_bounds():
    q = mx.zeros((1, 24, 2, 256), mx.bfloat16)
    k = mx.zeros((1, 2, 64, 256), mx.bfloat16)
    with pytest.raises(ValueError, match="window"):
        kq.sdpa_paged_rows(q, k, k, 1.0, mx.zeros((1, 2, 3), mx.int32), window=33)


def _first_only(row):
    """The list with every page after its first listing skipped (-1)."""
    seen, out = set(), []
    for p in row:
        out.append(-1 if p >= 0 and p in seen else p)
        seen.add(p)
    return out


@pytest.mark.parametrize("D,page_rows", [(128, 0), (256, 4)])
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("window", [0, 3])
def test_paged_rows_count_a_repeated_page_once(D, page_rows, L, window):
    # A page listed three times, and the page each query's end falls in
    # listed while tail_page adds it too: every form counts each page once.
    rng = np.random.default_rng(D + 3 * L)
    pr = page_rows or TILE[D]
    B, Hq, Hkv = 1, 8, 2
    base_len = 9 * pr + 1
    N = base_len + 2 * pr + 3
    q, k, v = _make(rng, B, Hq, Hkv, L, N, D, mx.bfloat16)
    pages, ends = _verify_pages(rng, L, 6, N - L, pr, shared=3)
    lists = np.concatenate(
        [pages[:, :-1], pages[:, :1], pages[:, :1], pages[:, -1:]], axis=1
    ).astype(np.int32)
    scale = 1.0 / D**0.5
    got = kq.sdpa_paged_rows(
        q,
        k[:, :, :base_len],
        v[:, :, :base_len],
        scale,
        mx.array(lists[None]),
        mx.array(ends[None]),
        k[:, :, base_len:],
        v[:, :, base_len:],
        base_len,
        page_rows=page_rows,
        tail_page=True,
        window=window,
    )
    once = np.array([_first_only(list(lists[i]) + [ends[i] // pr]) for i in range(L)])
    ref = _ref(_f32(q), _f32(k), _f32(v), scale, once[None], ends[None], pr)
    rel = _rel(_f32(got), ref)
    assert rel < REL_BOUND[mx.bfloat16], f"rel {rel:.3e}"


def test_paged_rows_params_checks():
    rng = np.random.default_rng(5)
    D, pr, S, T = 256, 4, 64, 8
    q, k, v = _make(rng, 1, 4, 2, 2, S + T, D, mx.bfloat16)
    pages = mx.array(np.arange(4, dtype=np.int32).reshape(1, 1, 4).repeat(2, 1))
    ends = mx.array(np.array([[S + 1, S + 2]], np.int32))
    kb, vb, kt, vt = k[:, :, :S], v[:, :, :S], k[:, :, S:], v[:, :, S:]
    # base_len from the block past the S rows of the base reads past it.
    p = mx.array(np.array([0, S + 5], np.int32))
    mx.eval(p)
    out = kq.sdpa_paged_rows(
        q,
        kb,
        vb,
        0.0625,
        pages,
        ends,
        kt,
        vt,
        S,
        page_rows=pr,
        params=p,
        base_len_slot=1,
    )
    with pytest.raises(ValueError, match="base_len must be in"):
        mx.eval(out)
    # The default ends hold the key count of the base_len the call was
    # built with, which a block that moves it would leave behind.
    with pytest.raises(ValueError, match="needs ends"):
        kq.sdpa_paged_rows(
            q,
            kb,
            vb,
            0.0625,
            pages,
            None,
            kt,
            vt,
            S,
            page_rows=pr,
            params=p,
            base_len_slot=1,
        )
