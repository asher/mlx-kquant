"""Teach mlx-lm's LoRA tuner to adapt and fuse a kquant-quantized base.

mlx-lm's LoRA / fuse machinery recognises its own affine ``QuantizedLinear``
family but not the kquant modules in :mod:`mlx_kquant.nn`. :func:`patch_mlx_lm_lora`
closes that gap, in two ways:

* it gives ``KQuantLinear`` / ``KQuantSwitchLinear`` / ``KQuantEmbedding`` a
  ``to_lora`` method, which mlx-lm's tuner consults first (both when discovering
  adaptable layers and when wrapping them). Each wraps the kquant base in the
  matching ``LoRA*`` module with the right in/out dims, recovered from the
  wire-byte geometry (the kquant modules store only the packed row width).
* it replaces the ``LoRA*.fuse`` methods so fusing a trained adapter into a
  kquant base dequantizes through ``kq.dequantize`` and, when the quant format is
  kept, re-encodes the merged weight through ``kq.quantize``. A non-kquant base
  falls through to the original implementation unchanged.

Call it once before building LoRA layers or loading adapters; it is idempotent.
Training a kquant LoRA also relies on the gradient-wrt-x ``vjp`` the extension
defines on the matmul / gather ops (the frozen base passes gradient through to
the trainable adapter).

DoRA is not supported on a kquant base in this release: mlx-lm's DoRA dispatch
does not consult ``to_lora``, so ``--fine-tune-type dora`` on a kquant model
raises mlx-lm's own "Can't convert layer" error. Use LoRA.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

import mlx_kquant as kq

from .codec_geometry import in_features
from .nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear

_patched = False
_orig_fuse: dict = {}


# --- to_lora: wrap a kquant base in the matching LoRA module -----------------


def _linear_to_lora(self, r, scale, dropout):
    from mlx_lm.tuner.lora import LoRALinear

    out_dims, row_bytes = self.weight.shape
    lora = LoRALinear(
        input_dims=in_features(self.kquant_type, row_bytes),
        output_dims=out_dims,
        r=r,
        dropout=dropout,
        scale=scale,
    )
    lora.linear = self
    return lora


def _switch_to_lora(self, r, scale, dropout):
    from mlx_lm.tuner.lora import LoRASwitchLinear

    num_experts, out_dims, row_bytes = self.weight.shape
    lora = LoRASwitchLinear(
        input_dims=in_features(self.kquant_type, row_bytes),
        output_dims=out_dims,
        num_experts=num_experts,
        r=r,
        dropout=dropout,
        scale=scale,
    )
    lora.linear = self
    return lora


def _embedding_to_lora(self, r, scale, dropout):
    from mlx_lm.tuner.lora import LoRAEmbedding

    lora = LoRAEmbedding(
        num_embeddings=self.num_embeddings,
        dims=self.dims,
        r=r,
        dropout=dropout,
        scale=scale,
    )
    lora.embedding = self
    return lora


# --- fuse: bake the adapter into the base ------------------------------------
#
# dequantize=True  -> a float layer (loads anywhere; re-bloats to the float size)
# dequantize=False -> stays kquant by re-encoding the merged weight (GPU-only
#                     until the v0.2.0 CPU encoder; adds a small re-quant error).


def _linear_fuse(self, dequantize: bool = False):
    base = self.linear
    if not isinstance(base, KQuantLinear):
        return _orig_fuse["linear"](self, dequantize=dequantize)

    bias = "bias" in base
    weight = kq.dequantize(base["weight"], base["scales"], base.kquant_type)
    out_dims, in_dims = weight.shape
    delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
    merged = weight + delta

    if dequantize:
        fused = nn.Linear(in_dims, out_dims, bias=bias)
        fused.weight = merged
        if bias:
            fused.bias = base.bias
        return fused

    wire, scales = kq.quantize(merged.astype(mx.float32), base.kquant_type)
    fused = KQuantLinear(in_dims, out_dims, bias, base.kquant_type)
    fused.weight = wire
    fused.scales = scales
    if bias:
        fused.bias = base.bias
    return fused


def _switch_fuse(self, dequantize: bool = False):
    base = self.linear
    if not isinstance(base, KQuantSwitchLinear):
        return _orig_fuse["switch"](self, dequantize=dequantize)
    from mlx_lm.models.switch_layers import SwitchLinear

    bias = "bias" in base
    num_experts, out_dims, row_bytes = base["weight"].shape
    in_dims = in_features(base.kquant_type, row_bytes)
    # Decode every expert at once (flatten the expert axis into rows).
    flat = base["weight"].reshape(num_experts * out_dims, row_bytes)
    weight = kq.dequantize(flat, base["scales"], base.kquant_type).reshape(
        num_experts, out_dims, in_dims
    )
    lora_b = self.scale * self.lora_b  # [E, out, r]
    lora_a = self.lora_a.reshape(num_experts, -1, in_dims)  # [E, r, in]
    merged = weight + (lora_b @ lora_a).astype(weight.dtype)

    if dequantize:
        fused = SwitchLinear(in_dims, out_dims, num_experts, bias=bias)
        fused.weight = merged
        if bias:
            fused.bias = base.bias
        return fused

    wire, scales = kq.quantize(
        merged.reshape(num_experts * out_dims, in_dims).astype(mx.float32),
        base.kquant_type,
    )
    fused = KQuantSwitchLinear(num_experts, out_dims, in_dims, bias, base.kquant_type)
    fused.weight = wire.reshape(num_experts, out_dims, row_bytes)
    fused.scales = scales
    if bias:
        fused.bias = base.bias
    return fused


def _embedding_fuse(self, dequantize: bool = False):
    base = self.embedding
    if not isinstance(base, KQuantEmbedding):
        return _orig_fuse["embedding"](self, dequantize=dequantize)

    num_embeddings, dims = base.num_embeddings, base.dims
    weight = kq.dequantize(base["weight"], base["scales"], base.kquant_type)
    merged = (weight + (self.scale * self.lora_a) @ self.lora_b).astype(weight.dtype)

    if dequantize:
        fused = nn.Embedding(num_embeddings, dims)
        fused.weight = merged
        return fused

    wire, scales = kq.quantize(merged.astype(mx.float32), base.kquant_type)
    fused = KQuantEmbedding(num_embeddings, dims, base.kquant_type)
    fused.weight = wire
    fused.scales = scales
    return fused


def patch_mlx_lm_lora() -> None:
    """Make a stock mlx-lm install adapt and fuse kquant bases. Idempotent."""
    global _patched
    if _patched:
        return
    from mlx_lm.tuner import lora as _lora

    KQuantLinear.to_lora = _linear_to_lora
    KQuantSwitchLinear.to_lora = _switch_to_lora
    KQuantEmbedding.to_lora = _embedding_to_lora

    _orig_fuse["linear"] = _lora.LoRALinear.fuse
    _orig_fuse["switch"] = _lora.LoRASwitchLinear.fuse
    _orig_fuse["embedding"] = _lora.LoRAEmbedding.fuse
    _lora.LoRALinear.fuse = _linear_fuse
    _lora.LoRASwitchLinear.fuse = _switch_fuse
    _lora.LoRAEmbedding.fuse = _embedding_fuse

    _patched = True
