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


def _make_switch(codec):
    if codec == "iq2_xxs":
        rng = np.random.default_rng(5)
        wire = np.stack([_synth_iq2xxs_wire(rng, N) for _ in range(E)], 0)
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


@pytest.mark.parametrize("codec", ["iq2_xxs", "q2_k"])
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
    sw = _make_switch("iq2_xxs")
    rng = np.random.default_rng(0)
    x, idx = _sorted_inputs(rng, 64, list(range(E)))
    monkeypatch.setenv("KQ_SWITCH_GEMM_MIN_ROWS", "512")
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


def main() -> int:
    return int(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    sys.exit(main())
