"""kquant-aware ``nn.Module`` subclasses.

Each module stores GGUF K-quant wire bytes directly as a ``uint8`` ``weight`` and
dispatches through the ``kq.dequantize`` / ``kq.quantized_matmul`` /
``kq.gather_qmm`` ops.

Importing this module needs only ``mlx`` + the built extension - no ``mlx-lm``.
The leaf-swap installer that wires these into a constructed mlx-lm model lives in
``mlx_kquant._install`` (which lazily imports ``mlx-lm``).

Each class carries ``mode = "kquant"`` plus ``kquant_type`` / ``group_size`` /
``bits`` / ``biases`` attributes so it duck-types with mlx-lm's affine
``QuantizedLinear`` family - that is what lets the LoRA tuner recognise and adapt
a kquant base (see ``mlx_kquant.mlx_lm_patch``).
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_kquant as kq

from .codec_geometry import CODEC_GEOMETRY, bytes_per_row

# gather_qmm_seg tile height (the kernel's BM).
_SEG_TILE_ROWS = 64


def _host_tile_maps(counts):
    """Host-built equivalent of kq.expert_tile_map from per-expert counts:
    (map, counts) uint32 arrays. Segments tile into 64-row tiles; only the
    last tile of a segment can be partial (the kernel skips its dead row
    fragments)."""
    tiles = []
    row = 0
    for e in np.flatnonzero(counts):
        c = int(counts[e])
        for off in range(0, c, _SEG_TILE_ROWS):
            tiles.append((e, row + off, min(_SEG_TILE_ROWS, c - off)))
        row += c
    m = np.asarray(tiles or [(0, 0, 0)], dtype=np.uint32)
    cnt = np.asarray([len(tiles)], dtype=np.uint32)
    return mx.array(m), mx.array(cnt)


class KQuantEmbedding(nn.Module):
    """Embedding backed by GGUF kquant wire bytes.

    ``__call__`` gathers per-token wire-byte rows then dequantizes - small output
    sizes only, so it avoids dequantizing the full embedding table (which can
    exceed INT_MAX elements and overflow the dispatch grid). ``as_linear`` runs a
    full kquant matmul for tied lm_head projection (gemma / qwen3 etc. tie
    ``embed_tokens`` to ``lm_head``).
    """

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        codec: str,
        out_dtype: mx.Dtype = mx.bfloat16,
    ):
        super().__init__()
        gs, bits, _, _ = CODEC_GEOMETRY[codec]
        self.mode = "kquant"
        self.group_size = gs
        self.bits = bits
        self.kquant_type = codec
        self.biases = None
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.out_dtype = out_dtype
        # Placeholders - overwritten by load_weights with the GGUF wire bytes.
        bpr = bytes_per_row(codec, dims)
        self.weight = mx.zeros((num_embeddings, bpr), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        self.freeze()

    def __call__(self, x):
        gathered = self["weight"][x]  # [*, bytes_per_row]
        flat = gathered.reshape(-1, gathered.shape[-1])
        deq = kq.dequantize(flat, self["scales"], self.kquant_type)
        # emit compute dtype (bf16) so a bf16 stream isn't promoted to f32
        return deq.reshape(*gathered.shape[:-1], self.dims).astype(self.out_dtype)

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
        # Placeholders - overwritten by load_weights with the GGUF wire bytes.
        bpr = bytes_per_row(codec, in_dims)
        self.weight = mx.zeros((out_dims, bpr), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        if bias:
            self.bias = mx.zeros((out_dims,))
        self.freeze()

    def __call__(self, x):
        # Decode (M=1) with a real bias: fuse the add into the qmv/qmv_fast
        # dispatch instead of a separate elementwise op -- see
        # quantized_matmul_qmv_bias's docstring for the exact shape contract.
        # Only q8_0 is wired so far; every other codec/shape (prefill,
        # verify/MTP) falls through to the plain matmul-then-add below.
        # KQ_DISABLE_QMV_BIAS=1 forces the unfused path (A/B lever); read live
        # so a single process can toggle between calls.
        if (
            "bias" in self
            and self.kquant_type == "q8_0"
            and x.size // x.shape[-1] == 1
            and os.environ.get("KQ_DISABLE_QMV_BIAS") != "1"
        ):
            return kq.quantized_matmul_qmv_bias(
                x, self["weight"], self["scales"], self["bias"], self.kquant_type
            )
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
        # Prefill-shaped sorted calls (mlx-lm SwitchGLU's _gather_sort layout:
        # x [B,1,K], indices [B] ascending) run one GEMM per expert segment
        # instead of gather_qmm's per-row matvec fallback, which is much faster
        # where the sorted rhs GEMM kernel is unavailable. Costs one host sync
        # on the routing indices per MoE layer. KQ_SWITCH_GEMM_MIN_ROWS=0
        # disables; read live for A/B.
        if (
            sorted_indices
            and "bias" not in self
            and indices.ndim == 1
            and x.ndim == 3
            and x.shape[0] == indices.size
            and x.shape[1] == 1
        ):
            min_rows = int(os.environ.get("KQ_SWITCH_GEMM_MIN_ROWS", "512"))
            if (
                min_rows > 0
                and indices.size >= min_rows
                and not self._nax_gather_preferred(indices.size, x.shape[-1])
            ):
                return self._sorted_expert_gemm(x, indices)
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

    def _nax_gather_preferred(self, rows, k):
        """Mirror gather_qmm's sorted-rhs NAX GEMM gate: on NAX GPUs that
        tensor-core leaf beats the simdgroup-MMA seg kernel, so the sorted
        arm defers to gather_qmm whenever the leaf is reachable."""
        if not (
            hasattr(kq, "nax_gather_enabled")
            and kq.nax_gather_enabled(self.kquant_type)
        ):
            return False
        return k % 64 == 0 and rows >= 16 and rows >= 4 * self.weight.shape[0]

    def _sorted_expert_gemm(self, x, indices):
        xf = x.reshape(indices.size, -1)
        w, s, codec = self["weight"], self["scales"], self.kquant_type
        if hasattr(kq, "gather_qmm_seg") and mx.metal.is_available():
            # tile map built on GPU: no host sync, layers stay pipelined
            if indices.dtype != mx.uint32:
                indices = indices.astype(mx.uint32)
            maps = kq.expert_tile_map(indices, w.shape[0])
            return kq.gather_qmm_seg(xf, w, s, codec, *maps)[:, None, :]
        counts = np.bincount(np.array(indices), minlength=self.weight.shape[0])
        outs = []
        start = 0
        for e in np.flatnonzero(counts):
            c = int(counts[e])
            outs.append(
                kq.quantized_matmul(
                    xf[start : start + c], w[e], s, codec, transpose=True
                )
            )
            start += c
        y = mx.concatenate(outs) if len(outs) > 1 else outs[0]
        return y[:, None, :]

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
    so the wire bytes arrive shaped ``(num_heads, output_dims, bytes_per_row)`` -
    identical to a SwitchLinear's expert stack, dispatched the same way through
    ``kq.gather_qmm``.

    Called in two modes (mlx_lm deepseek_v3 attention):
      * ``transpose=True``  - ``x[..., h, :, :] @ W[h].T`` (decode q_nope; and
        unembed_out always). Contracts over the quantized axis.
      * ``transpose=False`` - ``x @ W[h]`` (prefill ``embed_q(kv_latent)``),
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
        # The gather index arrays depend only on (B, Hx); memoize them so a
        # decode loop reuses one constant pair instead of rebuilding
        # arange/broadcast graph nodes every layer every token (underscored
        # attr: excluded from Module parameters).
        nh = self.num_heads
        B, Hx = x.shape[0], x.shape[1]
        cached = getattr(self, "_gather_idx", None)
        if cached is not None and cached[0] == (B, Hx):
            lhs, rhs = cached[1], cached[2]
        else:
            rhs = mx.broadcast_to(
                mx.arange(nh, dtype=mx.uint32).reshape(1, nh), (B, nh)
            )
            if Hx == nh:
                lhs = mx.arange(B * nh, dtype=mx.uint32).reshape(B, nh)
            elif Hx == 1:
                lhs = mx.broadcast_to(
                    mx.arange(B, dtype=mx.uint32).reshape(B, 1), (B, nh)
                )
            else:
                raise ValueError(
                    f"KQuantMultiLinear: head axis {Hx} is neither 1 nor "
                    f"num_heads={nh} (x.shape={x.shape})"
                )
            mx.eval(lhs, rhs)
            self._gather_idx = ((B, Hx), lhs, rhs)
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
