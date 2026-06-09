#!/usr/bin/env python3
"""Validate the training vjp (gradient wrt the float activation x) on the
quantized ops.

``kq.quantized_matmul`` and ``kq.gather_qmm`` define a vjp that backpropagates
only the gradient with respect to ``x``; the quantized weights/scales are a
frozen base and their gradient throws. That single branch is what lets a LoRA
adapter train on a frozen kquant base: a frozen quantized layer must still pass
gradient back to trainable parameters that sit upstream of it.

These run on the default device — Metal on Apple Silicon — so the vjp resolves
through the GPU kernels; set ``KQUANT_FORCE_CPU=1`` (the shared CI convention in
conftest.py) to run the same checks through the scalar CPU eval paths instead:

  * matmul vjp vs an independent oracle: d/dx (x @ Wq.T) == cotan @ dequant(Wq),
    with the dequant taken from gguf-py (not kq.dequantize) so a shared bug
    cannot cancel out of both sides.
  * a frozen dense layer trains a parameter placed upstream of it — the loss
    strictly decreases, the gradient is finite and non-zero, and the frozen wire
    bytes are byte-unchanged after the steps.
  * the MoE gather counterpart, with a non-identity ``lhs_indices`` selection so
    the scatter-add branch of the gather vjp is exercised (not the sorted fast
    path).

Run directly for a short report, or under pytest.
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq


def _scales():
    return mx.zeros((1,), dtype=mx.uint8)


def _q8_weight(n, k, seed):
    """Synthesize a q8_0 weight (gguf-py encodes flat codecs): wire + ref[N, K]."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k)) * 0.1).astype(np.float32)
    wire = quants.quantize(w, GT.Q8_0).astype(np.uint8)
    ref = quants.dequantize(np.ascontiguousarray(wire), GT.Q8_0).astype(np.float32)
    return mx.array(wire), ref


def _q8_experts(e, n, k, seed):
    """Synthesize E q8_0 expert weights: wire[E, N, packed] + ref[E, N, K]."""
    rng = np.random.default_rng(seed)
    wires, refs = [], []
    for _ in range(e):
        we = (rng.standard_normal((n, k)) * 0.1).astype(np.float32)
        wq = quants.quantize(we, GT.Q8_0).astype(np.uint8)
        wires.append(wq)
        refs.append(quants.dequantize(np.ascontiguousarray(wq), GT.Q8_0))
    return mx.array(np.stack(wires, 0)), np.stack(refs, 0).astype(np.float32)


def test_matmul_vjp_matches_oracle():
    """d/dx of (x @ dequant(w).T) is cotan @ dequant(w); compare to a gguf-py
    oracle so the kq dequant can't appear on both sides."""
    N, K, M = 32, 256, 8
    w, deq = _q8_weight(N, K, seed=7)
    sc = _scales()
    rng = np.random.default_rng(0)
    x = mx.array((rng.standard_normal((M, K)) * 0.1).astype(np.float16))
    cot = mx.array((rng.standard_normal((M, N)) * 0.1).astype(np.float16))

    def f(xx):
        return kq.quantized_matmul(xx, w, sc, "q8_0", transpose=True)

    _, vjps = mx.vjp(f, (x,), (cot,))
    mx.eval(vjps)
    grad = np.array(vjps[0]).astype(np.float32)  # [M, K]
    oracle = np.array(cot).astype(np.float32) @ deq  # [M, N] @ [N, K] = [M, K]

    assert grad.shape == oracle.shape
    assert np.all(np.isfinite(grad)), "matmul vjp produced non-finite grad"
    assert np.linalg.norm(grad) > 0, "matmul vjp grad is all-zero"
    rel = float(np.linalg.norm(grad - oracle) / (np.linalg.norm(oracle) + 1e-6))
    assert rel < 2e-2, f"matmul vjp diverges from oracle: rel={rel:.3e}"


def test_dense_frozen_layer_trains_upstream_param():
    """A trainable P sits upstream of a frozen quantized matmul; gradient must
    flow through the matmul's vjp to reach P. Assert real training, not just
    'did not crash'."""
    N, K, M = 16, 256, 8
    w, _deq = _q8_weight(N, K, seed=3)
    sc = _scales()
    rng = np.random.default_rng(1)
    x = mx.array((rng.standard_normal((M, K)) * 0.1).astype(np.float32))
    target = mx.array((rng.standard_normal((M, N)) * 0.1).astype(np.float32))
    P = mx.array((np.eye(K) + rng.standard_normal((K, K)) * 0.01).astype(np.float32))
    wire_before = np.array(w).copy()

    def loss_fn(p):
        h = mx.matmul(x, p).astype(mx.float16)
        y = kq.quantized_matmul(h, w, sc, "q8_0", transpose=True)
        return mx.mean((y.astype(mx.float32) - target) ** 2)

    lvg = mx.value_and_grad(loss_fn)
    losses, g0 = [], None
    for _ in range(20):
        loss, g = lvg(P)
        mx.eval(loss, g)
        if g0 is None:
            g0 = np.array(g)
        losses.append(float(loss))
        P = P - 1.0 * g
        mx.eval(P)

    assert np.all(np.isfinite(g0)), "grad wrt upstream param is non-finite"
    assert np.linalg.norm(g0) > 0, "grad wrt upstream param is all-zero"
    assert losses[-1] < 0.5 * losses[0], f"loss did not fall enough: {losses}"
    assert np.array_equal(np.array(w), wire_before), "frozen base wire bytes changed"


def test_moe_gather_frozen_experts_trains_upstream_param():
    """MoE counterpart: a non-identity lhs selection forces the scatter-add
    branch of the gather vjp. Gradient flows through it to a frozen-experts-
    upstream P; assert real training + frozen bytes unchanged."""
    E, N, K = 4, 16, 256
    B, M = 6, 2
    w, _deq = _q8_experts(E, N, K, seed=11)
    sc = _scales()
    rng = np.random.default_rng(2)
    x = mx.array((rng.standard_normal((B, M, K)) * 0.1).astype(np.float32))
    target = mx.array((rng.standard_normal((B, M, N)) * 0.1).astype(np.float32))
    experts = mx.array(rng.integers(0, E, size=B).astype(np.uint32))
    # reversed-with-repeat lhs: one x row used twice, one unused -> non-identity
    # gather, which routes the vjp through scatter_add rather than the fast path.
    seq = np.arange(B, dtype=np.uint32)
    perm = seq[::-1].copy()
    perm[1] = perm[0]
    lhs = mx.array(perm)
    P = mx.array((np.eye(K) + rng.standard_normal((K, K)) * 0.01).astype(np.float32))
    wire_before = np.array(w).copy()

    def loss_fn(p):
        h = mx.matmul(x, p).astype(mx.float16)
        y = kq.gather_qmm(
            h,
            w,
            sc,
            "q8_0",
            lhs_indices=lhs,
            rhs_indices=experts,
            transpose=True,
        )
        return mx.mean((y.astype(mx.float32) - target) ** 2)

    lvg = mx.value_and_grad(loss_fn)
    losses, g0 = [], None
    for _ in range(20):
        loss, g = lvg(P)
        mx.eval(loss, g)
        if g0 is None:
            g0 = np.array(g)
        losses.append(float(loss))
        P = P - 1.0 * g
        mx.eval(P)

    assert np.all(np.isfinite(g0)), "gather grad wrt upstream param is non-finite"
    assert np.linalg.norm(g0) > 0, "gather grad wrt upstream param is all-zero"
    assert losses[-1] < 0.5 * losses[0], f"gather loss did not fall enough: {losses}"
    assert np.array_equal(np.array(w), wire_before), "frozen experts wire bytes changed"


def _main() -> int:
    fns = [
        test_matmul_vjp_matches_oracle,
        test_dense_frozen_layer_trains_upstream_param,
        test_moe_gather_frozen_experts_trains_upstream_param,
    ]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"{'FAILURES: ' + str(fails) if fails else 'ALL OK'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
