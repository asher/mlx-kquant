#!/usr/bin/env python3
"""Validate mlx_kquant.mlx_lm_patch: LoRA on a kquant base (apply / fuse / train).

patch_mlx_lm_lora() teaches stock mlx-lm to wrap a kquant module in a LoRA layer
and to fuse the trained adapter back in. These checks cover:

  * apply numerics — lora(x) == base(x) + scale·(x@A)@B (the patched wrapper
    really dispatches through the kquant base plus the low-rank delta);
  * fuse(dequantize=True) — a float nn.Linear whose forward matches lora(x);
  * fuse(dequantize=False) — re-encodes the merged weight back to a KQuantLinear
    (GPU-only: kq.quantize has no CPU path yet, so this is GPU-gated);
  * training — a few SGD steps with a trainable projection UPSTREAM of the frozen
    kquant base, so the gradient flows through the base's vjp to reach it (the
    realistic multi-layer LoRA gradient path). Asserts the adapter + upstream
    grads are finite and non-zero, the loss strictly falls, and the frozen base
    wire bytes are byte-unchanged. Dense and MoE (KQuantSwitchLinear, which
    routes the gather_qmm vjp) variants.

q8_0 bases are minted with gguf-py (CPU encode) so everything but the kquant
round-trip runs under KQUANT_FORCE_CPU.

Run directly for a short report, or under pytest.
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq
from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora
from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

# fuse(dequantize=False) re-encodes via kq.quantize, which is GPU-only and would
# throw on the CPU device — gate on a real GPU *and* not the forced-CPU CI mode.
_gpu_encode = pytest.mark.skipif(
    not kq.metallib_loads() or bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kquant re-encode (kq.quantize) is GPU-only",
)


def _rel(a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-6))


def _kquant_linear(out_dims, in_dims, seed, bias=False):
    """A KQuantLinear with a real q8_0 base (gguf-py encodes q8_0 on CPU)."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((out_dims, in_dims)) * 0.1).astype(np.float32)
    wire = quants.quantize(w, GT.Q8_0).astype(np.uint8)
    lin = KQuantLinear(in_dims, out_dims, bias, "q8_0")
    lin.weight = mx.array(wire)
    if bias:
        lin.bias = mx.array((rng.standard_normal(out_dims) * 0.1).astype(np.float32))
    lin.freeze()
    return lin


def _kquant_switch(num_experts, out_dims, in_dims, seed):
    rng = np.random.default_rng(seed)
    wires = []
    for _ in range(num_experts):
        we = (rng.standard_normal((out_dims, in_dims)) * 0.1).astype(np.float32)
        wires.append(quants.quantize(we, GT.Q8_0).astype(np.uint8))
    sw = KQuantSwitchLinear(num_experts, out_dims, in_dims, False, "q8_0")
    sw.weight = mx.array(np.stack(wires, 0))
    sw.freeze()
    return sw


class _Stack(nn.Module):
    """A trainable projection upstream of a (frozen-base) LoRA layer, so the
    gradient must pass through the kquant base's vjp to reach the projection."""

    def __init__(self, in_dims, lora):
        super().__init__()
        self.proj = nn.Linear(in_dims, in_dims, bias=False)
        self.lora = lora

    def __call__(self, x, *rest):
        return self.lora(self.proj(x).astype(mx.float16), *rest)


def test_lora_apply_numerics():
    patch_mlx_lm_lora()
    in_dims, out_dims, M = 256, 16, 8
    base = _kquant_linear(out_dims, in_dims, seed=1)
    lora = base.to_lora(r=4, scale=2.0, dropout=0.0)
    # stock inits lora_b to zero (adapter == identity); perturb it so the delta
    # is actually exercised by the comparison.
    rng = np.random.default_rng(2)
    lora.lora_b = mx.array((rng.standard_normal(lora.lora_b.shape) * 0.1).astype("f"))
    x = mx.array((rng.standard_normal((M, in_dims)) * 0.1).astype(np.float16))

    got = lora(x)
    expect = base(x) + (lora.scale * ((x @ lora.lora_a) @ lora.lora_b)).astype(x.dtype)
    mx.eval(got, expect)
    assert _rel(got, expect) < 1e-2


def test_lora_fuse_dequantize_matches_apply():
    patch_mlx_lm_lora()
    in_dims, out_dims, M = 256, 16, 8
    base = _kquant_linear(out_dims, in_dims, seed=3)
    lora = base.to_lora(r=4, scale=2.0, dropout=0.0)
    rng = np.random.default_rng(4)
    lora.lora_b = mx.array((rng.standard_normal(lora.lora_b.shape) * 0.1).astype("f"))
    x = mx.array((rng.standard_normal((M, in_dims)) * 0.1).astype(np.float16))

    fused = lora.fuse(dequantize=True)
    assert isinstance(fused, nn.Linear)
    lo, fu = lora(x), fused(x)
    mx.eval(lo, fu)
    assert _rel(fu, lo) < 2e-2


@_gpu_encode
def test_lora_fuse_kquant_roundtrip():
    patch_mlx_lm_lora()
    in_dims, out_dims, M = 256, 16, 8
    base = _kquant_linear(out_dims, in_dims, seed=5)
    lora = base.to_lora(r=4, scale=2.0, dropout=0.0)
    rng = np.random.default_rng(6)
    lora.lora_b = mx.array((rng.standard_normal(lora.lora_b.shape) * 0.05).astype("f"))
    x = mx.array((rng.standard_normal((M, in_dims)) * 0.1).astype(np.float16))

    fused = lora.fuse(dequantize=False)
    assert isinstance(fused, KQuantLinear)
    lo, fu = lora(x), fused(x)
    mx.eval(lo, fu)
    # re-quantizing the merged weight adds a little error; q8_0 round-trip is tight.
    assert _rel(fu, lo) < 5e-2


def _train(stack, call_args, target, steps=40, lr=1.0):
    """Run SGD; return (losses, step0_grads). Grads are wrt trainable params only
    (the kquant base is frozen, so its bytes never enter the optimizer)."""

    def loss_fn(model, *args):
        return mx.mean((model(*args).astype(mx.float32) - target) ** 2)

    lvg = nn.value_and_grad(stack, loss_fn)
    opt = optim.SGD(learning_rate=lr)
    losses, g0 = [], None
    for _ in range(steps):
        loss, grads = lvg(stack, *call_args)
        if g0 is None:
            g0 = grads
        opt.update(stack, grads)
        mx.eval(stack.parameters(), opt.state, loss)
        losses.append(float(loss))
    return losses, g0


def test_lora_trains_dense():
    patch_mlx_lm_lora()
    in_dims, out_dims, M = 256, 8, 4  # M <= r so the rank-r adapter can fit
    base = _kquant_linear(out_dims, in_dims, seed=7)
    lora = base.to_lora(r=8, scale=2.0, dropout=0.0)
    rng = np.random.default_rng(8)
    lora.lora_b = mx.array((rng.standard_normal(lora.lora_b.shape) * 0.05).astype("f"))
    stack = _Stack(in_dims, lora)
    x = mx.array((rng.standard_normal((M, in_dims)) * 0.1).astype(np.float16))
    target = mx.array((rng.standard_normal((M, out_dims)) * 0.1).astype(np.float32))
    base_before = np.array(base.weight).copy()

    losses, g0 = _train(stack, (x,), target)

    ga, gb = np.array(g0["lora"]["lora_a"]), np.array(g0["lora"]["lora_b"])
    gp = np.array(g0["proj"]["weight"])  # non-zero proves the base vjp fired
    for name, g in (("lora_a", ga), ("lora_b", gb), ("proj", gp)):
        assert np.all(np.isfinite(g)), f"{name} grad non-finite"
        assert np.linalg.norm(g) > 0, f"{name} grad all-zero"
    fell = f"loss did not fall: {losses[0]:.4f}->{losses[-1]:.4f}"
    assert losses[-1] < 0.5 * losses[0], fell
    assert np.array_equal(np.array(base.weight), base_before), "frozen base changed"


def test_lora_trains_moe():
    patch_mlx_lm_lora()
    E, out_dims, in_dims = 4, 8, 256
    B, M = 6, 2
    base = _kquant_switch(E, out_dims, in_dims, seed=9)
    lora = base.to_lora(r=8, scale=2.0, dropout=0.0)
    rng = np.random.default_rng(10)
    lora.lora_b = mx.array((rng.standard_normal(lora.lora_b.shape) * 0.05).astype("f"))
    stack = _Stack(in_dims, lora)
    x = mx.array((rng.standard_normal((B, M, in_dims)) * 0.1).astype(np.float16))
    indices = mx.array(rng.integers(0, E, size=B).astype(np.uint32))
    target = mx.array((rng.standard_normal((B, M, out_dims)) * 0.1).astype(np.float32))
    base_before = np.array(base.weight).copy()

    losses, g0 = _train(stack, (x, indices), target)

    ga, gb = np.array(g0["lora"]["lora_a"]), np.array(g0["lora"]["lora_b"])
    gp = np.array(g0["proj"]["weight"])  # non-zero proves the gather_qmm vjp fired
    for name, g in (("lora_a", ga), ("lora_b", gb), ("proj", gp)):
        assert np.all(np.isfinite(g)), f"{name} grad non-finite"
        assert np.linalg.norm(g) > 0, f"{name} grad all-zero"
    fell = f"moe loss did not fall: {losses[0]:.4f}->{losses[-1]:.4f}"
    assert losses[-1] < 0.5 * losses[0], fell
    assert np.array_equal(np.array(base.weight), base_before), "frozen experts changed"


def _main() -> int:
    fns = [
        test_lora_apply_numerics,
        test_lora_fuse_dequantize_matches_apply,
        test_lora_trains_dense,
        test_lora_trains_moe,
    ]
    if kq.metallib_loads() and not os.environ.get("KQUANT_FORCE_CPU"):
        fns.insert(2, test_lora_fuse_kquant_roundtrip)
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
