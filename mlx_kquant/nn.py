"""kquant-aware ``nn.Module`` subclasses.

Each module stores GGUF K-quant wire bytes directly as a ``uint8`` ``weight`` and
dispatches through the ``kq.dequantize`` / ``kq.quantized_matmul`` /
``kq.gather_qmm`` ops.

Importing this module needs only ``mlx`` + the built extension — no ``mlx-lm``.
The leaf-swap installer that wires these into a constructed mlx-lm model lives in
``mlx_kquant._install`` (which lazily imports ``mlx-lm``).

Each class carries ``mode = "kquant"`` plus ``kquant_type`` / ``group_size`` /
``bits`` / ``biases`` attributes so it duck-types with mlx-lm's affine
``QuantizedLinear`` family — that is what lets the LoRA tuner recognise and adapt
a kquant base (see ``mlx_kquant.mlx_lm_patch``).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

import mlx_kquant as kq

from .codec_geometry import CODEC_GEOMETRY, bytes_per_row


class KQuantEmbedding(nn.Module):
    """Embedding backed by GGUF kquant wire bytes.

    ``__call__`` gathers per-token wire-byte rows then dequantizes — small output
    sizes only, so it avoids dequantizing the full embedding table (which can
    exceed INT_MAX elements and overflow the dispatch grid). ``as_linear`` runs a
    full kquant matmul for tied lm_head projection (gemma / qwen3 etc. tie
    ``embed_tokens`` to ``lm_head``).
    """

    def __init__(self, num_embeddings: int, dims: int, codec: str):
        super().__init__()
        gs, bits, _, _ = CODEC_GEOMETRY[codec]
        self.mode = "kquant"
        self.group_size = gs
        self.bits = bits
        self.kquant_type = codec
        self.biases = None
        self.num_embeddings = num_embeddings
        self.dims = dims
        # Placeholders — overwritten by load_weights with the GGUF wire bytes.
        bpr = bytes_per_row(codec, dims)
        self.weight = mx.zeros((num_embeddings, bpr), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        self.freeze()

    def __call__(self, x):
        gathered = self["weight"][x]  # [*, bytes_per_row]
        flat = gathered.reshape(-1, gathered.shape[-1])
        deq = kq.dequantize(flat, self["scales"], self.kquant_type)
        return deq.reshape(*gathered.shape[:-1], self.dims)

    def as_linear(self, x):
        return kq.quantized_matmul(
            x, self["weight"], self["scales"], self.kquant_type, transpose=True
        )

    def _extra_repr(self):
        return f"{self.num_embeddings}, {self.dims}, kquant_type={self.kquant_type}"


class KQuantLinear(nn.Module):
    """Linear layer backed by GGUF kquant wire bytes.

    Stores ``weight`` as uint8 wire bytes shaped ``(output_dims, bytes_per_row)``
    and dispatches via ``kq.quantized_matmul(..., transpose=True)``.
    """

    def __init__(self, in_dims: int, out_dims: int, bias: bool, codec: str):
        super().__init__()
        gs, bits, _, _ = CODEC_GEOMETRY[codec]
        self.mode = "kquant"
        self.group_size = gs
        self.bits = bits
        self.kquant_type = codec
        self.biases = None
        # Placeholders — overwritten by load_weights with the GGUF wire bytes.
        bpr = bytes_per_row(codec, in_dims)
        self.weight = mx.zeros((out_dims, bpr), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        if bias:
            self.bias = mx.zeros((out_dims,))
        self.freeze()

    def __call__(self, x):
        y = kq.quantized_matmul(
            x, self["weight"], self["scales"], self.kquant_type, transpose=True
        )
        if "bias" in self:
            y = y + self["bias"]
        return y

    def _extra_repr(self):
        out_dims, bpr = self.weight.shape
        return (
            f"output_dims={out_dims}, bytes_per_row={bpr}, "
            f"kquant_type={self.kquant_type}"
        )


class KQuantSwitchLinear(nn.Module):
    """MoE expert linear layer backed by GGUF kquant wire bytes.

    Counterpart to mlx_lm's QuantizedSwitchLinear (mode="affine"), but for kquant
    codecs. Stores ``weight`` as uint8 wire bytes shaped
    ``(n_experts, output_dims, bytes_per_row)`` and dispatches via
    ``kq.gather_qmm(..., transpose=True)``.
    """

    def __init__(
        self,
        num_experts: int,
        output_dims: int,
        input_dims: int,
        bias: bool,
        codec: str,
    ):
        super().__init__()
        gs, bits, _, _ = CODEC_GEOMETRY[codec]
        self.mode = "kquant"
        self.group_size = gs
        self.bits = bits
        self.kquant_type = codec
        self.biases = None
        bpr = bytes_per_row(codec, input_dims)
        self.weight = mx.zeros((num_experts, output_dims, bpr), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        x = kq.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.kquant_type,
            rhs_indices=indices,
            transpose=True,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def _extra_repr(self):
        n, m, b = self.weight.shape
        return (
            f"num_experts={n}, output_dims={m}, bytes_per_row={b}, "
            f"kquant_type={self.kquant_type}"
        )


class KQuantMultiLinear(nn.Module):
    """Per-head batched linear backed by GGUF kquant wire bytes.

    Counterpart to mlx_lm's absorbed-MLA ``MultiLinear`` (and its affine
    ``QuantizedMultiLinear``), used by deepseek_v3-family attention for the
    ``embed_q`` / ``unembed_out`` up-projections. Each of ``num_heads`` heads has
    its own ``(output_dims, input_dims)`` weight; the GGUF stores these stacked,
    so the wire bytes arrive shaped ``(num_heads, output_dims, bytes_per_row)`` —
    identical to a SwitchLinear's expert stack, dispatched the same way through
    ``kq.gather_qmm``.

    Called in two modes (mlx_lm deepseek_v3 attention):
      * ``transpose=True``  — ``x[..., h, :, :] @ W[h].T`` (decode q_nope; and
        unembed_out always). Contracts over the quantized axis.
      * ``transpose=False`` — ``x @ W[h]`` (prefill ``embed_q(kv_latent)``),
        contracting over the *non-quantized* axis.
    In the ``transpose=False`` case ``x`` carries a singleton head axis
    (``kv_latent`` is shared across heads) which is broadcast to ``num_heads`` via
    the gather's ``lhs_indices``.
    """

    def __init__(self, input_dims: int, output_dims: int, num_heads: int, codec: str):
        super().__init__()
        gs, bits, _, _ = CODEC_GEOMETRY[codec]
        self.mode = "kquant"
        self.group_size = gs
        self.bits = bits
        self.kquant_type = codec
        self.biases = None
        self.num_heads = num_heads
        bpr = bytes_per_row(codec, input_dims)
        self.weight = mx.zeros((num_heads, output_dims, bpr), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        self.freeze()

    def __call__(self, x, transpose: bool = True):
        # x: (B, Hx, L, D) where Hx is num_heads (per-head activation) or 1
        # (a head-shared activation broadcast across heads, transpose=False).
        nh = self.num_heads
        B, Hx = x.shape[0], x.shape[1]
        rhs = mx.broadcast_to(mx.arange(nh, dtype=mx.uint32).reshape(1, nh), (B, nh))
        if Hx == nh:
            lhs = mx.arange(B * nh, dtype=mx.uint32).reshape(B, nh)
        elif Hx == 1:
            lhs = mx.broadcast_to(mx.arange(B, dtype=mx.uint32).reshape(B, 1), (B, nh))
        else:
            raise ValueError(
                f"KQuantMultiLinear: head axis {Hx} is neither 1 nor "
                f"num_heads={nh} (x.shape={x.shape})"
            )
        return kq.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.kquant_type,
            lhs_indices=lhs,
            rhs_indices=rhs,
            transpose=transpose,
        )

    def _extra_repr(self):
        nh, out_dims, bpr = self.weight.shape
        return (
            f"num_heads={nh}, output_dims={out_dims}, "
            f"bytes_per_row={bpr}, kquant_type={self.kquant_type}"
        )


# Re-exported here (after the classes are defined) so callers have a single
# public import site, e.g. ``from mlx_kquant.nn import install_kquant_modules``.
from ._install import install_kquant_modules  # noqa: E402

__all__ = [
    "KQuantEmbedding",
    "KQuantLinear",
    "KQuantMultiLinear",
    "KQuantSwitchLinear",
    "install_kquant_modules",
]
