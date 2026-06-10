"""Teach stock mlx-lm to understand kquant - across load, adapt, and fuse.

mlx-lm's loader and LoRA / fuse machinery recognise its own affine
``QuantizedLinear`` family but not the kquant modules in :mod:`mlx_kquant.nn`.
This module closes that gap by monkeypatching mlx-lm at stable entry points,
behind two idempotent calls:

:func:`patch_mlx_lm_load` - the **load** seam (enough for inference / eval / a
server). It makes ``mlx_lm.utils.load_model`` kquant-aware, so anything built on
it - ``mlx_lm.load`` and the ``mlx_lm.lora`` / ``mlx_lm.fuse`` / ``mlx_lm.generate``
CLIs - can open a kquant checkpoint (stock ``load_model`` routes the config into
``nn.quantize`` and fails). A non-kquant checkpoint falls through unchanged.

:func:`patch_mlx_lm_lora` - the **adapt + fuse** seam (calls the load seam for
you first), in two further ways:

* it gives ``KQuantLinear`` / ``KQuantSwitchLinear`` / ``KQuantEmbedding`` a
  ``to_lora`` method, which mlx-lm's tuner consults first (both when discovering
  adaptable layers and when wrapping them). Each wraps the kquant base in the
  matching ``LoRA*`` module with the right in/out dims, recovered from the
  wire-byte geometry (the kquant modules store only the packed row width).
* it replaces the ``LoRA*.fuse`` methods so fusing a trained adapter into a
  kquant base dequantizes through ``kq.dequantize`` and, when the quant format is
  kept, re-encodes the merged weight through ``kq.quantize`` (optionally steered by
  an importance matrix, so the merge can preserve the base's calibration). A
  non-kquant base falls through to the original implementation unchanged.

Call the relevant one once, before loading / building LoRA layers / loading
adapters; both are idempotent. Training a kquant LoRA also relies on the
gradient-wrt-x ``vjp`` the extension defines on the matmul / gather ops (the
frozen base passes gradient through to the trainable adapter).

DoRA is not supported on a kquant base in this release: mlx-lm's DoRA dispatch
does not consult ``to_lora``, so ``--fine-tune-type dora`` on a kquant model
raises mlx-lm's own "Can't convert layer" error. Use LoRA.

The patched seams (``mlx_lm.utils.load_model`` and the ``LoRA*.fuse`` methods)
are stable across the supported mlx-lm range (tested through 0.31). Both patch
functions check the seams exist before patching and raise a clear
``RuntimeError`` if a future mlx-lm moves them.

mlx-lm's entry points have no per-call ``trust_remote_code``, so a checkpoint
with a custom ``model_file`` (arbitrary code) is refused through the patched
seam by default. The opt-in is made at the patch site -
``patch_mlx_lm_load(trust_remote_code=True)`` - and is a process-wide,
one-way grant: once given it applies to every later load and is not revoked
by a repeat call with the default.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import mlx_kquant as kq

from .codec_geometry import in_features
from .nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear

_load_patched = False
_lora_patched = False
_orig_fuse: dict = {}
_orig_load_model = None
_trust_remote_code = False


# --- load_model: route a kquant checkpoint through the kquant loader ----------


def _patched_load_model(
    model_path,
    lazy: bool = False,
    strict: bool = True,
    model_config: dict | None = None,
    get_model_classes=None,
):
    from .loader import _kquant_block, _load_config
    from .loader import load as _kq_load

    config = None
    try:
        config = _load_config(Path(model_path))
    except (OSError, json.JSONDecodeError):
        # Missing / unreadable config.json: defer to stock load_model, which
        # raises its own (better) error for a broken checkpoint.
        pass

    if config is not None and _kquant_block(config) is not None:
        model_file = config.get("model_file")
        if model_file is not None and not _trust_remote_code:
            # The inner loader's gate would suggest a kwarg this caller cannot
            # pass (mlx-lm's load_model has no trust slot); say where the
            # opt-in actually lives instead.
            raise ValueError(
                f"{model_path} declares a custom model_file ({model_file!r}), "
                f"which would import and execute code shipped in the "
                f"checkpoint. mlx-lm's entry points carry no per-call flag "
                f"for this; opt in at the patch site with "
                f"patch_mlx_lm_load(trust_remote_code=True) (CLI: mlx-kquant "
                f"lora --trust-remote-code) only if you trust its source."
            )
        # Stock load_model can't read a kquant checkpoint; build it ourselves.
        return _kq_load(
            model_path,
            lazy=lazy,
            strict=strict,
            model_config=model_config,
            trust_remote_code=_trust_remote_code,
        )

    kwargs = {}
    if get_model_classes is not None:
        kwargs["get_model_classes"] = get_model_classes
    return _orig_load_model(
        model_path,
        lazy=lazy,
        strict=strict,
        model_config=model_config,
        **kwargs,
    )


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
# dequantize=False -> stays kquant by re-encoding the merged weight (runs on CPU
#                     or Metal; adds a small re-quant error).
#
# imatrix: an optional per-input-feature importance vector (keep-kquant only). The
# re-encode otherwise rounds uncalibrated, so pass the same imatrix the base was
# quantized with to preserve that calibration through the merge.


def _imatrix_arg(imatrix, in_dims: int):
    """The importance vector as a float32 ``mx.array``, or ``None`` if unusable.

    A vector whose length does not match the row width is ignored rather than
    raising, mirroring the encode driver - a partial-coverage imatrix simply
    leaves the unmatched tensors uncalibrated.
    """
    if imatrix is None or getattr(imatrix, "shape", None) != (in_dims,):
        return None
    return mx.array(imatrix, dtype=mx.float32)


def _linear_fuse(self, dequantize: bool = False, imatrix=None):
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

    wire, scales = kq.quantize(
        merged.astype(mx.float32),
        base.kquant_type,
        imatrix=_imatrix_arg(imatrix, in_dims),
    )
    fused = KQuantLinear(in_dims, out_dims, bias, base.kquant_type)
    fused.weight = wire
    fused.scales = scales
    if bias:
        fused.bias = base.bias
    return fused


def _switch_fuse(self, dequantize: bool = False, imatrix=None):
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

    # The importance vector is per-input-feature, so it applies to every expert.
    wire, scales = kq.quantize(
        merged.reshape(num_experts * out_dims, in_dims).astype(mx.float32),
        base.kquant_type,
        imatrix=_imatrix_arg(imatrix, in_dims),
    )
    fused = KQuantSwitchLinear(num_experts, out_dims, in_dims, bias, base.kquant_type)
    fused.weight = wire.reshape(num_experts, out_dims, row_bytes)
    fused.scales = scales
    if bias:
        fused.bias = base.bias
    return fused


def _embedding_fuse(self, dequantize: bool = False, imatrix=None):
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

    wire, scales = kq.quantize(
        merged.astype(mx.float32),
        base.kquant_type,
        imatrix=_imatrix_arg(imatrix, dims),
    )
    fused = KQuantEmbedding(num_embeddings, dims, base.kquant_type)
    fused.weight = wire
    fused.scales = scales
    return fused


def patch_mlx_lm_load(trust_remote_code: bool = False) -> None:
    """Make stock ``mlx_lm.load`` / ``load_model`` open a kquant checkpoint.

    Routes a kquant config through :func:`mlx_kquant.loader.load` and leaves every
    other checkpoint untouched. This is the *inference* seam - enough for a serving
    or eval consumer that never trains or fuses. Idempotent; called for you by
    :func:`patch_mlx_lm_lora`.

    Args:
        trust_remote_code: allow a checkpoint's custom ``model_file``
            (arbitrary code) to load through the patched seam. mlx-lm's entry
            points have no per-call flag, so the grant is made here, at the
            patch site, and applies process-wide to every later load. It is
            one-way: a repeat call with the default does not revoke it.
    """
    global _load_patched, _orig_load_model, _trust_remote_code
    _trust_remote_code = _trust_remote_code or trust_remote_code
    if _load_patched:
        return
    import mlx_lm.utils as _utils

    if not hasattr(_utils, "load_model"):
        raise RuntimeError(
            "mlx_lm.utils.load_model not found - mlx-lm's loader API has moved "
            "and this mlx-kquant release cannot patch it. Pin an mlx-lm this "
            "release supports (tested through 0.31), or update mlx-kquant."
        )
    _orig_load_model = _utils.load_model
    _utils.load_model = _patched_load_model
    _load_patched = True


def patch_mlx_lm_lora(trust_remote_code: bool = False) -> None:
    """Make a stock mlx-lm install load, adapt, and fuse kquant bases. Idempotent.

    ``trust_remote_code`` is forwarded to :func:`patch_mlx_lm_load` (same
    process-wide, one-way semantics).
    """
    global _lora_patched
    if _lora_patched:
        patch_mlx_lm_load(trust_remote_code)  # keep the trust grant current
        return
    from mlx_lm.tuner import lora as _lora

    for cls_name in ("LoRALinear", "LoRASwitchLinear", "LoRAEmbedding"):
        if not hasattr(getattr(_lora, cls_name, None), "fuse"):
            raise RuntimeError(
                f"mlx_lm.tuner.lora.{cls_name}.fuse not found - mlx-lm's LoRA "
                "API has moved and this mlx-kquant release cannot patch it. "
                "Pin an mlx-lm this release supports (tested through 0.31), "
                "or update mlx-kquant."
            )

    patch_mlx_lm_load(trust_remote_code)  # the loader seam is a prerequisite

    KQuantLinear.to_lora = _linear_to_lora
    KQuantSwitchLinear.to_lora = _switch_to_lora
    KQuantEmbedding.to_lora = _embedding_to_lora

    _orig_fuse["linear"] = _lora.LoRALinear.fuse
    _orig_fuse["switch"] = _lora.LoRASwitchLinear.fuse
    _orig_fuse["embedding"] = _lora.LoRAEmbedding.fuse
    _lora.LoRALinear.fuse = _linear_fuse
    _lora.LoRASwitchLinear.fuse = _switch_fuse
    _lora.LoRAEmbedding.fuse = _embedding_fuse

    _lora_patched = True
