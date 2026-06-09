"""Recipe resolution tests — pure logic, no GPU, no [tools], no model download.

Drives the classifier + preset resolver from fabricated path/role maps (and a
tiny fake model for the predicate) and asserts the *specific* per-tensor codec
decisions, not just "it returns a dict".
"""

from __future__ import annotations

import pytest

from mlx_kquant.recipes import (
    KQUANT_PRESETS,
    TensorRole,
    _use_more_bits,
    classify_path,
    classify_tensors,
    extract_layer_idx,
    kquant_predicate_builder,
    resolve_codec_map,
)

# --------------------------------------------------------------------------- #
# classify_path                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path,role",
    [
        ("model.layers.0.self_attn.q_proj", TensorRole.ATTENTION_QKVO),
        ("model.layers.3.self_attn.v_proj", TensorRole.ATTENTION_QKVO),
        ("model.layers.3.self_attn.o_proj", TensorRole.ATTENTION_QKVO),
        ("model.layers.0.attn.out_proj", TensorRole.ATTENTION_OUTPUT),
        ("model.layers.0.linear_attn.in_proj_qkv", TensorRole.LINEAR_ATTN),
        ("model.layers.0.mlp.gate_proj", TensorRole.FFN),
        ("model.layers.0.mlp.down_proj", TensorRole.FFN),
        ("model.layers.0.mlp.switch_mlp.gate_proj", TensorRole.ROUTED_EXPERT),
        ("model.layers.0.mlp.experts.7.up_proj", TensorRole.ROUTED_EXPERT),
        ("model.layers.0.mlp.shared_expert.down_proj", TensorRole.SHARED_EXPERT),
        ("model.layers.0.mlp.gate", TensorRole.MOE_ROUTER),
        ("model.embed_tokens", TensorRole.EMBEDDING),
        ("lm_head", TensorRole.LM_HEAD),
        ("model.vision_tower.embeddings.patch_embedding", TensorRole.VLM_TOWER),
        ("model.layers.0.mlp.experts.0.something_else", None),
    ],
)
def test_classify_path(path, role):
    assert classify_path(path) == role


def test_qkvo_beats_vlm_tower():
    # A vision-tower attention proj still classifies as attention (qkvo is
    # checked before the vlm-tower catch-all), so it is protected like attention.
    p = "model.vision_tower.encoder.layers.0.self_attn.q_proj"
    assert classify_path(p) == TensorRole.ATTENTION_QKVO


# --------------------------------------------------------------------------- #
# extract_layer_idx                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path,idx",
    [
        ("model.layers.7.mlp.down_proj", 7),
        ("model.layers.0.self_attn.q_proj", 0),
        ("model.layers.31.mlp.switch_mlp.gate_proj", 31),
        ("model.embed_tokens", None),
        ("lm_head", None),
    ],
)
def test_extract_layer_idx(path, idx):
    assert extract_layer_idx(path) == idx


# --------------------------------------------------------------------------- #
# resolve_codec_map: role-specific decisions                                  #
# --------------------------------------------------------------------------- #


def test_role_specific_codecs_q4_k_m():
    role_map = {
        "model.layers.0.self_attn.q_proj": TensorRole.ATTENTION_QKVO,
        "model.layers.0.self_attn.v_proj": TensorRole.ATTENTION_QKVO,
        "model.layers.0.self_attn.o_proj": TensorRole.ATTENTION_QKVO,
        "model.layers.0.mlp.gate_proj": TensorRole.FFN,
        "model.embed_tokens": TensorRole.EMBEDDING,
        "lm_head": TensorRole.LM_HEAD,
    }
    cm = resolve_codec_map(role_map, preset="q4_k_m", default_codec=None)
    assert cm["model.layers.0.self_attn.q_proj"] == "q4_k"
    assert cm["model.layers.0.self_attn.o_proj"] == "q4_k"
    # v_proj gets the path-suffix bump to q6_k.
    assert cm["model.layers.0.self_attn.v_proj"] == "q6_k"
    # FFN has no preset entry -> falls to the base default.
    assert cm["model.layers.0.mlp.gate_proj"] == "q4_k"
    assert cm["model.embed_tokens"] == "q4_k"
    assert cm["lm_head"] == "q6_k"


def test_moe_preset_expert_codecs():
    role_map = {
        "model.layers.0.mlp.switch_mlp.gate_proj": TensorRole.ROUTED_EXPERT,
        "model.layers.0.mlp.shared_expert.up_proj": TensorRole.SHARED_EXPERT,
        "model.layers.0.mlp.gate": TensorRole.MOE_ROUTER,
    }
    cm = resolve_codec_map(role_map, preset="q4_k_moe", default_codec=None)
    assert cm["model.layers.0.mlp.switch_mlp.gate_proj"] == "q4_k"
    assert cm["model.layers.0.mlp.shared_expert.up_proj"] == "q5_k"
    # Router is a bf16-passthrough role -> not quantized.
    assert cm["model.layers.0.mlp.gate"] is None


def test_bf16_passthrough_roles():
    role_map = {
        "model.vision_tower.x.fc1": TensorRole.VLM_TOWER,
        "model.layers.0.mlp.gate": TensorRole.MOE_ROUTER,
    }
    cm = resolve_codec_map(role_map, preset="q5_k_m", default_codec=None)
    assert cm["model.vision_tower.x.fc1"] is None
    assert cm["model.layers.0.mlp.gate"] is None


def test_overrides_win():
    role_map = {"model.layers.0.self_attn.q_proj": TensorRole.ATTENTION_QKVO}
    cm = resolve_codec_map(
        role_map,
        preset="q4_k_m",
        default_codec=None,
        overrides={"model.layers.0.self_attn.q_proj": "q8_0"},
    )
    assert cm["model.layers.0.self_attn.q_proj"] == "q8_0"


def test_uniform_default_no_preset():
    role_map = {
        "model.layers.0.self_attn.q_proj": TensorRole.ATTENTION_QKVO,
        "model.layers.0.mlp.gate_proj": TensorRole.FFN,
        "model.layers.0.mlp.gate": TensorRole.MOE_ROUTER,
    }
    cm = resolve_codec_map(role_map, preset=None, default_codec="q8_0")
    assert cm["model.layers.0.self_attn.q_proj"] == "q8_0"
    assert cm["model.layers.0.mlp.gate_proj"] == "q8_0"
    # passthrough roles stay None even with a uniform default.
    assert cm["model.layers.0.mlp.gate"] is None


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve_codec_map({}, preset="nope", default_codec=None)


# --------------------------------------------------------------------------- #
# Layer-position bumps (use_more_bits)                                         #
# --------------------------------------------------------------------------- #


def test_use_more_bits_boundaries():
    n = 16
    # first eighth (0,1) and last eighth (14,15) always bump.
    assert _use_more_bits(0, n)
    assert _use_more_bits(1, n)
    assert _use_more_bits(14, n)
    assert _use_more_bits(15, n)
    # a clearly-middle layer that the every-third rule misses.
    assert not _use_more_bits(3, n)


def test_layer_position_bump_fires_at_boundaries():
    n_layers = 16
    role_map = {
        f"model.layers.{i}.mlp.down_proj": TensorRole.FFN for i in range(n_layers)
    }
    cm = resolve_codec_map(role_map, preset="q4_k_m", default_codec=None)
    # boundary layer -> bumped to q6_k; middle non-bumped layer -> base q4_k.
    assert cm["model.layers.0.mlp.down_proj"] == "q6_k"
    assert cm["model.layers.15.mlp.down_proj"] == "q6_k"
    assert cm["model.layers.3.mlp.down_proj"] == "q4_k"
    # Every down_proj resolves to exactly the use_more_bits decision.
    for i in range(n_layers):
        expected = "q6_k" if _use_more_bits(i, n_layers) else "q4_k"
        assert cm[f"model.layers.{i}.mlp.down_proj"] == expected


# --------------------------------------------------------------------------- #
# predicate <-> codec_map agreement                                           #
# --------------------------------------------------------------------------- #


class _FakeQuantModule:
    """Stands in for a quantizable leaf — only needs a ``to_quantized`` attr."""

    def to_quantized(self):  # pragma: no cover - never called
        raise NotImplementedError


class _FakeModel:
    def __init__(self, paths):
        self._paths = paths

    def named_modules(self):
        # Root module first (path "", no to_quantized -> skipped), then leaves.
        yield "", self
        for p in self._paths:
            yield p, _FakeQuantModule()


def test_predicate_matches_resolve_codec_map():
    paths = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.gate",  # router -> passthrough
        "model.embed_tokens",
        "lm_head",
    ]
    model = _FakeModel(paths)
    role_map = classify_tensors(model)
    codec_map = resolve_codec_map(role_map, preset="q4_k_m", default_codec=None)
    predicate = kquant_predicate_builder("q4_k_m", model)

    for p in paths:
        result = predicate(p, _FakeQuantModule())
        codec = codec_map.get(p)
        if codec is None:
            assert result is False
        else:
            assert isinstance(result, dict)
            assert result["kquant_type"] == codec
            assert result["mode"] == "kquant"


def test_presets_are_self_consistent():
    # Every preset references only known codecs in its default + role entries.
    from mlx_kquant.codec_geometry import CODEC_GEOMETRY

    for name, mapping in KQUANT_PRESETS.items():
        for key, codec in mapping.items():
            if codec is not None:
                assert codec in CODEC_GEOMETRY, f"{name}:{key} -> {codec}"
