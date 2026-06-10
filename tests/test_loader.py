"""Round-trip tests for the kquant create-weights + load path.

The primary tests fabricate a tiny random-init mlx-lm model, ``quantize_model``
-> ``save`` it to a tmp dir, ``load`` it back, and assert the reload is faithful:
the per_tensor map survives, the reloaded leaves are ``KQuant*``, and the loaded
model's logits match the freshly-quantized model's bit-for-bit (same wire bytes
through disk). A dense (Llama) and an MoE (qwen3_moe -> SwitchLinear) variant
cover both the ``[out, bpr]`` and ``[E, out, bpr]`` weight layouts.

Encode and the forward ops both have a CPU path, so these round-trips run on CPU
(and in CI) as well as on a GPU. A real-HF-model round-trip runs only when
``KQUANT_TEST_MODEL`` names a (small) source id.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

# The whole module is model-level: needs the [tools] extra (mlx-lm).
pytest.importorskip("mlx_lm")

from mlx_kquant.loader import _get_classes, load  # noqa: E402


def _build(cfg: dict):
    model_class, args_class = _get_classes(cfg)
    model = model_class(args_class.from_dict(cfg))
    mx.eval(model.parameters())
    return model


def _tiny_llama_cfg() -> dict:
    # All quantizable row widths are multiples of 256 (q4_k weights_per_block).
    return dict(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=512,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def _tiny_qwen3_moe_cfg() -> dict:
    return dict(
        model_type="qwen3_moe",
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=512,
        num_attention_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=256,
        rms_norm_eps=1e-5,
        vocab_size=512,
        num_key_value_heads=2,
        head_dim=64,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        max_position_embeddings=512,
        norm_topk_prob=True,
    )


def _roundtrip_asserts(tmp_path, cfg: dict, preset: str):
    from mlx_kquant.convert import quantize_model, save

    model = _build(cfg)
    qmodel, qconfig = quantize_model(model, cfg, preset=preset)
    written_pt = dict(qconfig["quantization"]["per_tensor"])
    assert written_pt, "nothing got quantized"

    save(tmp_path, qmodel, qconfig)
    loaded, lconfig = load(tmp_path)

    # The per_tensor map survives the round-trip verbatim.
    assert lconfig["quantization"]["per_tensor"] == written_pt
    assert lconfig["quantization"]["mode"] == "kquant"

    # Every quantized leaf came back as a KQuant* module - one per per_tensor key.
    n_kq = sum(
        1 for _, m in loaded.named_modules() if type(m).__name__.startswith("KQuant")
    )
    assert n_kq == len(written_pt)

    # Forward is finite and bit-identical to the in-memory quantized model (the
    # reload restores the exact wire bytes, and the kernels are deterministic).
    x = mx.array([[1, 2, 3, 4, 5]])
    out_loaded = loaded(x)
    out_q = qmodel(x)
    mx.eval(out_loaded, out_q)
    assert bool(mx.all(mx.isfinite(out_loaded)).item())
    assert bool(mx.array_equal(out_loaded, out_q).item())


def test_dense_roundtrip(tmp_path):
    _roundtrip_asserts(tmp_path, _tiny_llama_cfg(), "q4_k_m")


def test_moe_roundtrip(tmp_path):
    _roundtrip_asserts(tmp_path, _tiny_qwen3_moe_cfg(), "q4_k_moe")


def test_load_rejects_non_kquant(tmp_path):
    # A checkpoint with no kquant block is rejected with a clear error.
    import json

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "model.safetensors").write_bytes(b"")  # presence not inspected
    with pytest.raises(ValueError, match="not a kquant checkpoint"):
        load(tmp_path)


def _model_file_cfg() -> dict:
    return {
        "model_type": "llama",
        "model_file": "custom.py",
        "quantization": {"mode": "kquant", "per_tensor": {"fc": "q4_k"}},
    }


def test_load_model_file_gate(tmp_path):
    # A custom model_file is refused without the opt-in, and the error names
    # both the kwarg and the CLI flag.
    import json

    (tmp_path / "config.json").write_text(json.dumps(_model_file_cfg()))
    with pytest.raises(ValueError, match="--trust-remote-code"):
        load(tmp_path)


def test_load_model_file_missing(tmp_path):
    # Opting in when the declared file is absent (the Hub-download case, where
    # _HF_ALLOW filters code files) raises a clear error, not FileNotFoundError.
    import json

    (tmp_path / "config.json").write_text(json.dumps(_model_file_cfg()))
    with pytest.raises(ValueError, match="not present"):
        load(tmp_path, trust_remote_code=True)


def test_install_kquant_modules_shape_math():
    """No GPU: install swaps a Linear leaf and sizes the uint8 weight correctly."""
    import mlx.nn as nn

    from mlx_kquant.codec_geometry import bytes_per_row
    from mlx_kquant.nn import KQuantLinear, install_kquant_modules

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 128, bias=False)
            self.keep = nn.Linear(256, 64, bias=False)

    m = Tiny()
    n = install_kquant_modules(m, {"fc.weight": "q4_k"})
    assert n == 1
    assert isinstance(m.fc, KQuantLinear)
    assert m.fc.weight.shape == (128, bytes_per_row("q4_k", 256))
    assert m.fc.weight.dtype == mx.uint8
    # An unlisted leaf is left untouched.
    assert isinstance(m.keep, nn.Linear)


@pytest.mark.skipif(
    not os.environ.get("KQUANT_TEST_MODEL"),
    reason="set KQUANT_TEST_MODEL=<small hf id or path> to run",
)
def test_real_model_roundtrip(tmp_path):
    from mlx_lm.utils import load_model

    from mlx_kquant.convert import quantize_model, save

    src = os.environ["KQUANT_TEST_MODEL"]
    from mlx_kquant.loader import _resolve_path

    src_path = _resolve_path(src, None)
    model, config = load_model(src_path, lazy=False)
    qmodel, qconfig = quantize_model(model, config, preset="q4_k_m")
    save(tmp_path, qmodel, qconfig, hf_path=src_path)

    loaded, lconfig = load(tmp_path)
    x = mx.array([[1, 2, 3, 4, 5]])
    out = loaded(x)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())
