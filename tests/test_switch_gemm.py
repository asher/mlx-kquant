#!/usr/bin/env python3
"""KQuantSwitchLinear sorted per-expert GEMM arm.

Prefill-shaped sorted calls (x [B,1,K], ascending 1-D indices) route to
_sorted_expert_gemm: one kq.quantized_matmul per expert segment instead of
gather_qmm's per-row matvec. Compared against the stock gather path
(KQ_SWITCH_GEMM_MIN_ROWS=0) on the codecs deepseek-v4 uses.

Usage: test_switch_gemm.py
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

from mlx_kquant.nn import KQuantSwitchLinear

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
E, N, K = 8, 128, 512


def _synth_iq2xxs_wire(rng, n_rows):
    bpb = 66
    n_blocks = n_rows * (K // 256)
    wire = rng.integers(0, 256, size=(n_blocks, bpb), dtype=np.uint8)
    d = rng.uniform(0.02, 0.08, n_blocks).astype(np.float16)
    wire[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 2)
    return wire.reshape(n_rows, (K // 256) * bpb)


def _synth_native_fp_wire(rng, codec, n_rows):
    bpb, wpb = (17, 32) if codec == "mxfp4" else (36, 64)
    n_blocks = n_rows * (K // wpb)
    wire = rng.integers(0, 256, size=(n_blocks, bpb), dtype=np.uint8)
    if codec == "mxfp4":
        wire[:, 0] = rng.integers(121, 132, n_blocks, dtype=np.uint8)
    else:
        wire[:, 0:4] = rng.integers(0x30, 0x41, (n_blocks, 4), dtype=np.uint8)
    return wire.reshape(n_rows, (K // wpb) * bpb)


def _make_switch(codec):
    if codec == "iq2_xxs":
        rng = np.random.default_rng(5)
        wire = np.stack([_synth_iq2xxs_wire(rng, N) for _ in range(E)], 0)
    elif codec in ("mxfp4", "nvfp4"):
        rng = np.random.default_rng(5)
        wire = np.stack([_synth_native_fp_wire(rng, codec, N) for _ in range(E)], 0)
    else:
        path = os.path.join(FIX, f"{codec}_moe.npz")
        if not os.path.exists(path):
            pytest.skip(f"missing fixture {path}")
        wire = np.load(path)["wire"].astype(np.uint8)
        reps = -(-E // wire.shape[0])
        wire = np.ascontiguousarray(np.tile(wire, (reps, 1, 1))[:E])
    ne, nn_, _ = wire.shape
    sw = KQuantSwitchLinear(
        num_experts=ne, output_dims=nn_, input_dims=K, bias=False, codec=codec
    )
    sw.weight = mx.array(wire)
    return sw


def _sorted_inputs(rng, rows, experts):
    idx = np.sort(rng.choice(experts, size=rows).astype(np.uint32))
    x = mx.array((rng.standard_normal((rows, 1, K)) * 0.1).astype(np.float32))
    return x.astype(mx.float16), mx.array(idx)


@pytest.mark.parametrize("codec", ["iq2_xxs", "q2_k", "mxfp4", "nvfp4"])
@pytest.mark.parametrize(
    "rows,experts",
    [(1024, list(range(E))), (600, [3]), (777, [0, 2, 7])],
    ids=["all_experts", "single_expert", "sparse_experts"],
)
def test_sorted_expert_gemm_matches_gather(codec, rows, experts, monkeypatch):
    sw = _make_switch(codec)
    rng = np.random.default_rng(rows)
    x, idx = _sorted_inputs(rng, rows, experts)

    monkeypatch.setenv("KQ_SWITCH_GEMM_MIN_ROWS", "512")
    got = sw(x, idx, sorted_indices=True)
    monkeypatch.setenv("KQ_SWITCH_GEMM_MIN_ROWS", "0")
    ref = sw(x, idx, sorted_indices=True)
    mx.eval(got, ref)

    assert got.shape == ref.shape == (rows, 1, N)
    assert got.dtype == ref.dtype
    g = np.array(got.astype(mx.float32))
    r = np.array(ref.astype(mx.float32))
    rel = np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-6)
    assert rel < 5e-3, f"{codec} rows={rows}: rel {rel:.3e}"


def test_threshold_and_shape_gating(monkeypatch):
    import mlx_kquant.nn as kqnn

    sw = _make_switch("iq2_xxs")
    rng = np.random.default_rng(0)
    x, idx = _sorted_inputs(rng, 64, list(range(E)))
    monkeypatch.setenv("KQ_SWITCH_GEMM_MIN_ROWS", "512")
    # On NAX GPUs big sorted batches defer to gather_qmm's tensor-core leaf;
    # pin that off so the row threshold itself is what routes.
    monkeypatch.setattr(
        kqnn.kq, "nax_gather_enabled", lambda codec: False, raising=False
    )
    calls = []
    orig = sw._sorted_expert_gemm
    monkeypatch.setattr(
        sw, "_sorted_expert_gemm", lambda *a: calls.append(1) or orig(*a)
    )
    mx.eval(sw(x, idx, sorted_indices=True))  # below threshold
    assert not calls
    x2, idx2 = _sorted_inputs(rng, 512, list(range(E)))
    mx.eval(sw(x2, idx2, sorted_indices=True))
    assert len(calls) == 1


# ------------------------------------------------------- gather_qmm_seg op


def _tile_maps(counts):
    from mlx_kquant.nn import _host_tile_maps

    return _host_tile_maps(counts)


def _counts_to_indices(counts):
    return mx.array(np.repeat(np.arange(len(counts), dtype=np.uint32), counts))


@pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="gather_qmm_seg is Metal-only.",
)
@pytest.mark.parametrize("codec", ["iq2_xxs", "q2_k", "q8_0"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gather_qmm_seg_matches_loop(codec, dtype):
    import mlx_kquant as kq

    if codec == "q8_0":
        rng = np.random.default_rng(7)
        wf = rng.standard_normal((E, N, K)).astype(np.float32) * 0.1
        from gguf import GGMLQuantizationType as GT
        from gguf import quants

        wire = np.stack(
            [quants.quantize(wf[e], GT.Q8_0).astype(np.uint8) for e in range(E)]
        )
        sw = KQuantSwitchLinear(
            num_experts=E, output_dims=N, input_dims=K, bias=False, codec=codec
        )
        sw.weight = mx.array(wire)
    else:
        sw = _make_switch(codec)
    w, s = sw["weight"], sw["scales"]

    rng = np.random.default_rng(23)
    # Ragged segments: absent experts and partial tiles at every fragment
    # boundary class (1, 17, 63, 96, 129, 200, 48 rows).
    counts = np.zeros(E, dtype=np.int64)
    counts[[0, 1, 2, 3, 4, 5, 7]] = [1, 17, 63, 96, 129, 200, 48]
    rows = int(counts.sum())
    x = mx.array((rng.standard_normal((rows, K)) * 0.1).astype(np.float32)).astype(
        dtype
    )

    refs, start = [], 0
    for e in np.flatnonzero(counts):
        c = int(counts[e])
        refs.append(
            kq.quantized_matmul(
                x[start : start + c], w[e], s, sw.kquant_type, transpose=True
            )
        )
        start += c
    ref = mx.concatenate(refs)

    # host-built maps and GPU-built maps must both match the loop
    for label, maps in (
        ("host", _tile_maps(counts)),
        ("gpu", kq.expert_tile_map(_counts_to_indices(counts), E)),
    ):
        got = kq.gather_qmm_seg(x, w, s, sw.kquant_type, *maps)
        mx.eval(got, ref)
        assert got.shape == (rows, N) and got.dtype == ref.dtype
        g = np.array(got.astype(mx.float32))
        r = np.array(ref.astype(mx.float32))
        rel = np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-6)
        assert rel < 2e-3, f"{codec} {dtype} {label}: rel {rel:.3e}"


@pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="gather_qmm_seg is Metal-only.",
)
def test_gather_qmm_seg_unaligned_n():
    import mlx_kquant as kq

    rng = np.random.default_rng(31)
    n_odd = 72  # not a multiple of the kernel's BN=64
    wire = np.stack([_synth_iq2xxs_wire(rng, n_odd) for _ in range(4)], 0)
    w = mx.array(wire)
    s = mx.zeros((1,), dtype=mx.uint8)
    counts = np.array([65, 0, 3, 64], dtype=np.int64)
    rows = int(counts.sum())
    x = mx.array((rng.standard_normal((rows, K)) * 0.1).astype(np.float32)).astype(
        mx.float16
    )
    got = kq.gather_qmm_seg(x, w, s, "iq2_xxs", *_tile_maps(counts))
    refs, start = [], 0
    for e in np.flatnonzero(counts):
        c = int(counts[e])
        refs.append(
            kq.quantized_matmul(
                x[start : start + c], w[e], s, "iq2_xxs", transpose=True
            )
        )
        start += c
    ref = mx.concatenate(refs)
    mx.eval(got, ref)
    g = np.array(got.astype(mx.float32))
    r = np.array(ref.astype(mx.float32))
    rel = np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-6)
    assert got.shape == (rows, n_odd) and rel < 2e-3, f"rel {rel:.3e}"


@pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="expert_tile_map is Metal-only.",
)
@pytest.mark.parametrize(
    "counts",
    [
        [1, 63, 64, 129, 200, 0, 0, 32],
        [512],
        [0, 0, 0, 5],
        [64] * 8,
        [65] * 8,
        [1, 7, 8, 9, 16, 17, 48, 96],
    ],
    ids=["ragged", "single", "tiny", "uniform64", "uniform65", "sub_frag"],
)
def test_expert_tile_map_matches_host(counts):
    import mlx_kquant as kq
    from mlx_kquant.nn import _host_tile_maps

    counts = np.asarray(counts, dtype=np.int64)
    mh, cnth = _host_tile_maps(counts)
    m, cnt = kq.expert_tile_map(_counts_to_indices(counts), len(counts))
    mx.eval(m, cnt, mh, cnth)
    cnt, cnth = np.array(cnt), np.array(cnth)
    np.testing.assert_array_equal(cnt, cnth)
    # tile order is unspecified: compare as sorted sets of valid rows
    g = np.array(m)[: cnt[0]]
    r = np.array(mh)[: cnt[0]]

    def order(a):
        return a[np.lexsort(a.T[::-1])]

    np.testing.assert_array_equal(order(g), order(r))


@pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="gather_qmm_seg is Metal-only.",
)
def test_gather_qmm_seg_rejects_bad_inputs():
    import mlx_kquant as kq

    x = mx.zeros((64, K), dtype=mx.float16)
    w = mx.zeros((2, N, 1056), dtype=mx.uint8)  # iq2_xxs bpr for K=4096
    s = mx.zeros((1,), dtype=mx.uint8)
    good = mx.zeros((1, 3), dtype=mx.uint32)
    cnt = mx.zeros((1,), dtype=mx.uint32)
    with pytest.raises(ValueError):  # K mismatch (bpr expands to 4096)
        kq.gather_qmm_seg(x, w, s, "iq2_xxs", good, cnt)
    x_ok = mx.zeros((64, 4096), dtype=mx.float16)
    with pytest.raises(ValueError):  # bad map shape
        kq.gather_qmm_seg(x_ok, w, s, "iq2_xxs", mx.zeros((3,), dtype=mx.uint32), cnt)
    with pytest.raises(ValueError):  # bad counts shape (old multi-map layout)
        kq.gather_qmm_seg(x_ok, w, s, "iq2_xxs", good, mx.zeros((2,), dtype=mx.uint32))
    with pytest.raises(ValueError):  # 3-D x
        kq.gather_qmm_seg(
            mx.zeros((64, 1, 4096), dtype=mx.float16),
            w,
            s,
            "iq2_xxs",
            good,
            cnt,
        )
    with pytest.raises(ValueError):  # expert_tile_map: non-uint32 indices
        kq.expert_tile_map(mx.zeros((8,), dtype=mx.int32), 4)


def test_seg_arm_defers_to_nax_gather(monkeypatch):
    import mlx_kquant.nn as kqnn

    sw = _make_switch("iq2_xxs")
    rng = np.random.default_rng(11)
    x, idx = _sorted_inputs(rng, 512, list(range(E)))
    monkeypatch.setenv("KQ_SWITCH_GEMM_MIN_ROWS", "512")
    calls = []
    orig = sw._sorted_expert_gemm
    monkeypatch.setattr(
        sw, "_sorted_expert_gemm", lambda *a: calls.append(1) or orig(*a)
    )
    # NAX leaf reachable -> the sorted arm must defer to gather_qmm
    monkeypatch.setattr(
        kqnn.kq, "nax_gather_enabled", lambda codec: True, raising=False
    )
    nax_out = sw(x, idx, sorted_indices=True)
    mx.eval(nax_out)
    assert not calls
    monkeypatch.setattr(
        kqnn.kq, "nax_gather_enabled", lambda codec: False, raising=False
    )
    seg_out = sw(x, idx, sorted_indices=True)
    mx.eval(seg_out)
    assert len(calls) == 1
    g = np.array(seg_out.astype(mx.float32))
    r = np.array(nax_out.astype(mx.float32))
    rel = np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-6)
    assert rel < 5e-3, f"rel {rel:.3e}"


def main() -> int:
    return int(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    sys.exit(main())
