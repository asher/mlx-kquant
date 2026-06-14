"""Attention-protected mixed-precision recipes for kquant encoding.

Pure string / regex / dict logic - no ops, no GPU. Given a model's module
paths, classify each quantizable tensor by *role* (attention q/k/v/o, MoE
router, routed vs shared expert, embedding, lm_head, ...) and resolve a
per-tensor codec from a named *preset* (``q4_k_m``, ``q5_k_moe``, ...). The
encoder in :mod:`mlx_kquant.convert` consumes the resulting ``{path: codec}``
map.

The recipe philosophy: spend bits where they matter (attention output, value
projections, the down-projection on a subset of layers) and stay frugal on the
bulk feed-forward weights, beating a uniform quant at the same byte budget.

Codec geometry comes from :mod:`mlx_kquant.codec_geometry` (the single source of
truth), so codec facts can never drift from the kernels.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum

import mlx.nn as nn

from .codec_geometry import CODEC_GEOMETRY

# ---------------------------------------------------------------------------
# Role classifier
# ---------------------------------------------------------------------------


class TensorRole(str, Enum):
    LINEAR_ATTN = "linear_attn"
    ATTENTION_OUTPUT = "attention_output"
    ATTENTION_QKVO = "attention_qkvo"
    LM_HEAD = "lm_head"
    EMBEDDING = "embedding"
    EMBEDDING_PER_LAYER = "embedding_per_layer"
    VLM_TOWER = "vlm_tower"
    MOE_ROUTER = "moe_router"
    FFN = "ffn"
    SHARED_EXPERT = "shared_expert"
    ROUTED_EXPERT = "routed_expert"


_LINEAR_ATTN = re.compile(r"\.linear_attn\.[^.]+$")
_ATTN_OUT_PROJ = re.compile(r"(?<!self_)\battn\.out_proj$")
_QKVO = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")
_LM_HEAD = re.compile(r"(^|\.)lm_head$")
_EMBED_TOKENS = re.compile(r"\b(embed_tokens|wte|word_embeddings)\b")
_EMBED_PER_LAYER = re.compile(r"\bembed_tokens_per_layer\b")
_VLM_TOWER = re.compile(
    r"(^|\.)("
    r"vision_tower|vision_model|visual"
    r"|audio_tower|audio_model"
    r"|multi_modal_projector|vl_connector"
    r"|merger|connector"
    r"|embed_vision|embed_audio"
    r")\."
)
_MOE_ROUTER = re.compile(r"(\.router\.proj" r"|\.mlp\.gate" r"|\.shared_expert_gate)$")
_FFN = re.compile(r"\.mlp\.(gate|up|down)_proj$")
_SHARED_EXPERT = re.compile(r"\.shared_expert\.(gate|up|down)_proj$")
_ROUTED_EXPERTS = re.compile(
    r"\.experts\.switch_glu\.(gate|up|down)_proj$"
    r"|\.mlp\.switch_mlp\.(gate|up|down)_proj$"
    r"|\.mlp\.experts\.\d+\.(gate|up|down)_proj$"
)


def classify_path(path: str) -> TensorRole | None:
    """Map a module path to its :class:`TensorRole`, or ``None`` if unrecognized."""
    if _LINEAR_ATTN.search(path):
        return TensorRole.LINEAR_ATTN
    if _ATTN_OUT_PROJ.search(path):
        return TensorRole.ATTENTION_OUTPUT
    if _ROUTED_EXPERTS.search(path):
        return TensorRole.ROUTED_EXPERT
    if _MOE_ROUTER.search(path):
        return TensorRole.MOE_ROUTER
    if _QKVO.search(path):
        return TensorRole.ATTENTION_QKVO
    if _LM_HEAD.search(path):
        return TensorRole.LM_HEAD
    if _EMBED_PER_LAYER.search(path):
        return TensorRole.EMBEDDING_PER_LAYER
    if _EMBED_TOKENS.search(path):
        return TensorRole.EMBEDDING
    if _SHARED_EXPERT.search(path):
        return TensorRole.SHARED_EXPERT
    if _FFN.search(path):
        return TensorRole.FFN
    if _VLM_TOWER.search(path):
        return TensorRole.VLM_TOWER
    return None


def classify_tensors(model: nn.Module) -> dict[str, TensorRole]:
    """Classify every quantizable (has ``to_quantized``) module in ``model``."""
    result: dict[str, TensorRole] = {}
    for path, module in model.named_modules():
        if not hasattr(module, "to_quantized"):
            continue
        role = classify_path(path)
        if role is not None:
            result[path] = role
    return result


def extract_layer_idx(path: str) -> int | None:
    """Pull the transformer layer index out of a path, or ``None``."""
    m = _LAYER_IDX_RE.search(path)
    if m is None:
        return None
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Preset system
# ---------------------------------------------------------------------------

# Routers and the multimodal towers are tiny and accuracy-sensitive; leave them
# in their source (bf16) precision rather than quantizing.
_BF16_PASSTHROUGH_ROLES = frozenset({TensorRole.MOE_ROUTER, TensorRole.VLM_TOWER})

_ATTN_V_BUMP = {".self_attn.v_proj": "q6_k"}
_MAMBA_OUT_BUMP_Q5K = {".linear_attn.out_proj": "q5_k"}
_MAMBA_OUT_BUMP_Q6K = {".linear_attn.out_proj": "q6_k"}
_MAMBA_LINEAR_ATTN_XL = {
    ".linear_attn.out_proj": "q8_0",
    ".linear_attn.in_proj_qkv": "q6_k",
    ".linear_attn.in_proj_z": "q5_k",
}
_MAMBA_LINEAR_ATTN_Q5_XL = {
    ".linear_attn.out_proj": "q8_0",
    ".linear_attn.in_proj_qkv": "q6_k",
    ".linear_attn.in_proj_z": "q6_k",
}

_PRESETS: dict[str, dict[str, str | None]] = {
    "q4_k_s": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_m": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_xl": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_moe": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
        TensorRole.SHARED_EXPERT.value: "q5_k",
        TensorRole.ROUTED_EXPERT.value: "q4_k",
    },
    "q5_k_s": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
    },
    "q5_k_m": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
    },
    "q5_k_xl": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
    },
    "q5_k_moe": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
        TensorRole.SHARED_EXPERT.value: "q6_k",
        TensorRole.ROUTED_EXPERT.value: "q5_k",
    },
    "q3_k_m": {
        "default": "q3_k",
        TensorRole.ATTENTION_QKVO.value: "q3_k",
        TensorRole.ATTENTION_OUTPUT.value: "q3_k",
        TensorRole.LM_HEAD.value: "q5_k",
        TensorRole.EMBEDDING.value: "q3_k",
    },
    "q6_k": {"default": "q6_k"},
    "q6_k_xl": {
        "default": "q6_k",
        TensorRole.ATTENTION_QKVO.value: "q8_0",
        TensorRole.ATTENTION_OUTPUT.value: "q6_k",
        TensorRole.LM_HEAD.value: "q8_0",
        TensorRole.EMBEDDING.value: "q8_0",
    },
    "q2_k": {
        "default": "q2_k",
        TensorRole.ATTENTION_QKVO.value: "q2_k",
        TensorRole.ATTENTION_OUTPUT.value: "q3_k",
        TensorRole.LM_HEAD.value: "q3_k",
        TensorRole.EMBEDDING.value: "q2_k",
    },
    "q8": {"default": "q8_0", TensorRole.EMBEDDING.value: "q8_0"},
}

KQUANT_PRESETS = _PRESETS

_PATH_BUMPS: dict[str, dict[str, str | None]] = {
    "q4_k_s": {**_ATTN_V_BUMP, **_MAMBA_OUT_BUMP_Q5K},
    "q4_k_m": {**_ATTN_V_BUMP, **_MAMBA_OUT_BUMP_Q5K},
    "q4_k_moe": {**_ATTN_V_BUMP, **_MAMBA_OUT_BUMP_Q5K},
    "q4_k_xl": {
        **_ATTN_V_BUMP,
        ".mlp.down_proj": "q6_k",
        **_MAMBA_LINEAR_ATTN_XL,
    },
    "q5_k_s": {**_ATTN_V_BUMP, **_MAMBA_OUT_BUMP_Q6K},
    "q5_k_m": {**_ATTN_V_BUMP, **_MAMBA_OUT_BUMP_Q6K},
    "q5_k_moe": {**_ATTN_V_BUMP, **_MAMBA_OUT_BUMP_Q6K},
    "q5_k_xl": {
        **_ATTN_V_BUMP,
        ".mlp.down_proj": "q6_k",
        **_MAMBA_LINEAR_ATTN_Q5_XL,
    },
    "q3_k_m": {".self_attn.v_proj": "q5_k", ".linear_attn.out_proj": "q5_k"},
    "q2_k": {".self_attn.v_proj": "q3_k", ".linear_attn.out_proj": "q3_k"},
    "q6_k_xl": {
        ".self_attn.o_proj": "q6_k",
        ".linear_attn.out_proj": "q8_0",
        ".linear_attn.in_proj_qkv": "q8_0",
        ".linear_attn.in_proj_z": "q8_0",
    },
}


def _use_more_bits(i_layer: int, n_layers: int) -> bool:
    """llama.cpp's "use more bits" layer schedule (first/last eighth + every 3rd)."""
    return (
        i_layer < n_layers // 8
        or i_layer >= 7 * n_layers // 8
        or (i_layer - n_layers // 8) % 3 == 2
    )


_LAYER_POSITION_RULES = {"use_more_bits": _use_more_bits}

_LAYER_POSITION_BUMPS: dict[str, dict[str, tuple[str, str]]] = {
    "q4_k_m": {".mlp.down_proj": ("q6_k", "use_more_bits")},
    "q4_k_moe": {".mlp.down_proj": ("q6_k", "use_more_bits")},
    "q5_k_m": {".mlp.down_proj": ("q6_k", "use_more_bits")},
    "q5_k_moe": {".mlp.down_proj": ("q6_k", "use_more_bits")},
    "q3_k_m": {".mlp.down_proj": ("q5_k", "use_more_bits")},
    "q6_k_xl": {".mlp.down_proj": ("q8_0", "use_more_bits")},
}


def resolve_codec_map(
    role_map: dict[str, TensorRole],
    *,
    preset: str | None,
    default_codec: str | None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str | None]:
    """Resolve ``{path: codec_or_None}`` for every classified path.

    Priority (highest to lowest):
      1. ``overrides[path]``
      2. Preset path-suffix bump
      3. Layer-position bump (use_more_bits)
      4. Role-specific preset entry
      5. BF16 passthrough roles (``None`` = leave unquantized)
      6. Preset default / ``default_codec``
    """
    overrides = overrides or {}
    if preset is not None:
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; choices: {sorted(_PRESETS)}")
        preset_map = _PRESETS[preset]
    else:
        preset_map = {}
    base_default = preset_map.get("default", default_codec)
    path_bumps = _PATH_BUMPS.get(preset, {}) if preset is not None else {}
    lpos_bumps = _LAYER_POSITION_BUMPS.get(preset, {}) if preset is not None else {}

    n_layers = 0
    if lpos_bumps:
        for path in role_map:
            idx = extract_layer_idx(path)
            if idx is not None and idx + 1 > n_layers:
                n_layers = idx + 1

    out: dict[str, str | None] = {}
    for path, role in role_map.items():
        if path in overrides:
            out[path] = overrides[path]
            continue

        matched_bump = False
        for suffix, bumped in path_bumps.items():
            if path.endswith(suffix):
                out[path] = bumped
                matched_bump = True
                break
        if matched_bump:
            continue

        matched_lpos = False
        for suffix, (lpos_codec, rule_name) in lpos_bumps.items():
            if path.endswith(suffix):
                i_layer = extract_layer_idx(path)
                if (
                    i_layer is not None
                    and n_layers > 0
                    and _LAYER_POSITION_RULES[rule_name](i_layer, n_layers)
                ):
                    out[path] = lpos_codec
                    matched_lpos = True
                break
        if matched_lpos:
            continue

        if role.value in preset_map:
            out[path] = preset_map[role.value]
            continue

        if role in _BF16_PASSTHROUGH_ROLES:
            out[path] = None
            continue

        out[path] = base_default
    return out


def format_presets() -> str:
    """Human-readable listing of every preset and what it maps, for the CLI.

    Renders each preset's default codec, its role-specific entries, and the
    path / layer-position bumps that distinguish the ``_s`` / ``_m`` / ``_xl``
    variants. The single source of truth is the dicts above, so the listing can
    never drift from what :func:`resolve_codec_map` actually does.
    """
    lines: list[str] = []
    for name in sorted(_PRESETS):
        entries = _PRESETS[name]
        default = entries.get("default")
        roles = [
            f"{role}={codec}"
            for role, codec in sorted(entries.items())
            if role != "default" and codec != default
        ]
        line = f"  {name:10} default={default}"
        if roles:
            line += "  " + "  ".join(roles)
        lines.append(line)
        bumps = _PATH_BUMPS.get(name)
        if bumps:
            btxt = "  ".join(f"{s}={c}" for s, c in sorted(bumps.items()))
            lines.append(f"  {'':10}   path bumps: {btxt}")
        for suffix, (codec, rule) in sorted(
            _LAYER_POSITION_BUMPS.get(name, {}).items()
        ):
            lines.append(f"  {'':10}   layer bumps: {suffix}={codec} on {rule} layers")
    lines.append(
        "\n  Roles not listed take the default; path/layer bumps match on "
        "module-path suffix. The _s/_m/_xl variants differ in these bumps."
    )
    return "\n".join(lines)


def kquant_predicate_builder(
    preset: str,
    model: nn.Module,
    default_codec: str | None = None,
) -> Callable[[str, nn.Module], bool | dict]:
    """Build a ``(path, module) -> dict | False`` quant predicate for a preset.

    The dict carries ``group_size`` / ``bits`` / ``mode`` / ``kquant_type``;
    ``False`` means "leave this tensor unquantized". Mirrors the per-tensor
    decisions :func:`resolve_codec_map` makes, exposed in predicate form.
    """
    role_map = classify_tensors(model)
    codec_map = resolve_codec_map(role_map, preset=preset, default_codec=default_codec)

    def predicate(path: str, module: nn.Module) -> bool | dict:
        codec = codec_map.get(path)
        if codec is None:
            return False
        gs, bits, _, _ = CODEC_GEOMETRY[codec]
        return {
            "group_size": gs,
            "bits": bits,
            "mode": "kquant",
            "kquant_type": codec,
        }

    return predicate
