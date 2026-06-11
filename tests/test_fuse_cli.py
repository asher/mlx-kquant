"""End-to-end tests for ``mlx-kquant fuse`` (the LoRA-merge subcommand).

Covers the plumbing on top of the numerics in ``test_lora_patch.py``:

  * ``_dequantize_remaining`` - decodes leftover KQuant* layers to float layers
    (runs on CPU, no GPU);
  * the full subcommand both ways - quantize a tiny model, mint a tiny LoRA
    adapter against it, ``fuse`` it, reload the result, and check the fused
    forward matches the LoRA-applied reference. ``--dequantize`` reloads with
    stock ``mlx_lm`` and matches tightly (lossless float merge); keep-kquant
    reloads with our loader and matches within the re-quant tolerance.

The end-to-end tests exercise the base quantize + the keep-kquant re-encode; both
the encode and decode paths run on CPU, so the whole suite runs on CPU (and in CI).
"""

from __future__ import annotations

import argparse
import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

# Model-level: needs the [tools] extra (mlx-lm).
pytest.importorskip("mlx_lm")

import mlx_kquant as kq  # noqa: E402
from mlx_kquant.nn import KQuantLinear  # noqa: E402


def _rel(a, b):
    a = np.array(mx.array(a).astype(mx.float32))
    b = np.array(mx.array(b).astype(mx.float32))
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-6))


def test_dequantize_remaining_decodes_leftover_layers():
    """A KQuant* layer the adapter never touched is decoded to a float nn.Linear,
    matching a dequantize-then-matmul reference; an unrelated float layer is left
    alone. No GPU: q8_0 minted by gguf-py, decoded by the CPU path."""
    from mlx_kquant.cli.fuse import _dequantize_remaining

    in_dims, out_dims = 256, 32
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((out_dims, in_dims)) * 0.1).astype(np.float32)
    wire = quants.quantize(w, GT.Q8_0).astype(np.uint8)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.kq = KQuantLinear(in_dims, out_dims, False, "q8_0")
            self.kq.weight = mx.array(wire)
            self.kq.freeze()
            self.keep = nn.Linear(in_dims, 8, bias=False)

    m = Tiny()
    ref = kq.dequantize(m.kq["weight"], m.kq["scales"], "q8_0")
    n = _dequantize_remaining(m)

    assert n == 1
    assert isinstance(m.kq, nn.Linear)
    assert isinstance(m.keep, nn.Linear)  # untouched
    x = mx.array((rng.standard_normal((4, in_dims)) * 0.1).astype(np.float32))
    got, expect = m.kq(x), x @ ref.T
    mx.eval(got, expect)
    assert _rel(got, expect) < 1e-5


def _write_imatrix_dat(path, vectors):
    """Write a legacy llama.cpp ``.dat`` imatrix: ``int32 n_entries``, then per
    entry ``int32 name_len, name bytes, int32 ncall, int32 nval, float32[nval]``.
    Matches what ``mlx_kquant.imatrix._load_imatrix_dat`` reads back."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(vectors)))
        for name, vec in vectors.items():
            raw = name.encode("utf-8")
            f.write(struct.pack("<i", len(raw)))
            f.write(raw)
            f.write(struct.pack("<i", 1))  # ncall
            f.write(struct.pack("<i", len(vec)))
            f.write(np.asarray(vec, dtype=np.float32).tobytes())


def test_imatrix_arg_only_accepts_a_matching_row_width():
    """``_imatrix_arg`` gates the per-tensor importance vector: None passes
    through, a wrong-width vector is dropped (partial coverage, no raise), and a
    matching vector becomes a float32 mx.array. No GPU."""
    from mlx_kquant.mlx_lm_patch import _imatrix_arg

    assert _imatrix_arg(None, 256) is None
    assert _imatrix_arg(np.ones(255, dtype=np.float32), 256) is None
    got = _imatrix_arg(np.ones(256, dtype=np.float32), 256)
    assert got is not None
    assert got.dtype == mx.float32
    assert tuple(got.shape) == (256,)


def test_resolve_imatrix_guards_short_circuit(capsys):
    """``_resolve_imatrix`` returns {} (without reading the model/config) when no
    imatrix is given, and when --dequantize is set it warns and skips. No GPU."""
    from mlx_kquant.cli.fuse import _resolve_imatrix

    none_args = argparse.Namespace(imatrix=None, dequantize=False)
    assert _resolve_imatrix(none_args, None, None) == {}

    deq_args = argparse.Namespace(imatrix="ignored.dat", dequantize=True)
    assert _resolve_imatrix(deq_args, None, None) == {}
    assert "ignored with --dequantize" in capsys.readouterr().out


def _tiny_llama_cfg() -> dict:
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


def _make_base(tmp_path):
    """Quantize a tiny random llama to uniform q8_0 and save it. Returns the dir."""
    from mlx_kquant.convert import quantize_model, save
    from mlx_kquant.loader import _get_classes

    cfg = _tiny_llama_cfg()
    cls, argcls = _get_classes(cfg)
    model = cls(argcls.from_dict(cfg))
    mx.eval(model.parameters())
    qmodel, qconfig = quantize_model(model, cfg, default_codec="q8_0")
    base = tmp_path / "base"
    save(base, qmodel, qconfig)
    return base


def _mint_adapter(base_dir, adapter_dir):
    """Apply LoRA to a fresh load of the base, perturb it, and save it as a
    standard mlx-lm adapter (adapter_config.json + adapters.safetensors)."""
    import json

    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from mlx_kquant.loader import load
    from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora

    patch_mlx_lm_lora()
    model, _ = load(base_dir)
    lora_params = {"rank": 8, "scale": 2.0, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers=2, config=lora_params)

    # Stock inits lora_b to zero (delta == 0); perturb every adapter so the merge
    # is actually exercised.
    rng = np.random.default_rng(1)
    updates = []
    for name, m in model.named_modules():
        if hasattr(m, "lora_b"):
            pert = (rng.standard_normal(m.lora_b.shape) * 0.05).astype("f")
            updates.append((f"{name}.lora_b", mx.array(pert)))
    model.update(tree_unflatten(updates))

    adapter_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), weights)
    cfg = {"fine_tune_type": "lora", "num_layers": 2, "lora_parameters": lora_params}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg))
    return lora_params


def _reference_lora_out(base_dir, adapter_dir, x):
    """Forward of the LoRA-applied (unfused) model - the merge target."""
    from mlx_lm.tuner.utils import load_adapters

    from mlx_kquant.loader import load
    from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora

    patch_mlx_lm_lora()
    model, _ = load(base_dir)
    load_adapters(model, str(adapter_dir))
    out = model(x)
    mx.eval(out)
    return out


def test_loader_patch_lets_mlx_lm_load_kquant(tmp_path):
    """After patching, stock `mlx_lm.utils.load_model` opens a kquant checkpoint
    and forwards finite logits - this is what makes `mlx_lm.load` (and the
    `mlx_lm.lora` CLI built on it) work on a kquant base.

    Uses the standalone `patch_mlx_lm_load` (the load-only seam, no LoRA) - a
    serving / eval consumer wants load without the adapt/fuse machinery."""
    import mlx_lm.utils as mlx_utils

    from mlx_kquant.mlx_lm_patch import patch_mlx_lm_load

    base = _make_base(tmp_path)
    patch_mlx_lm_load()
    # Resolve load_model through the module (as mlx_lm's own `load` does), so the
    # patched attribute is picked up - `from ... import load_model` would capture
    # the pre-patch binding.
    model, config = mlx_utils.load_model(base)

    assert config.get("quantization", {}).get("mode") == "kquant"
    assert any(type(m).__name__.startswith("KQuant") for _, m in model.named_modules())
    out = model(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_fuse_cli_keep_kquant(tmp_path):
    from mlx_kquant.cli import fuse as fuse_cli
    from mlx_kquant.loader import load

    base = _make_base(tmp_path)
    adapter = tmp_path / "adapter"
    _mint_adapter(base, adapter)
    out = tmp_path / "fused-kq"

    x = mx.array([[1, 2, 3, 4, 5]])
    ref = _reference_lora_out(base, adapter, x)

    args = argparse.Namespace(
        model=str(base),
        adapter_path=str(adapter),
        save_path=str(out),
        dequantize=False,
        imatrix=None,
        trust_remote_code=False,
    )
    assert fuse_cli.cmd(args) == 0

    fused, fconfig = load(out)
    # Still a kquant checkpoint with the same per-tensor map.
    assert fconfig["quantization"]["mode"] == "kquant"
    n_kq = sum(
        1 for _, m in fused.named_modules() if type(m).__name__.startswith("KQuant")
    )
    assert n_kq > 0
    fout = fused(x)
    mx.eval(fout)
    assert bool(mx.all(mx.isfinite(fout)).item())
    # Re-encoding the merged weight adds a small q8_0 re-quant error.
    assert _rel(fout, ref) < 5e-2


def test_fuse_cli_keep_kquant_imatrix(tmp_path, capsys):
    """A keep-kquant fuse with --imatrix: mint an imatrix keyed by the fusable
    layers' HF paths, fuse with it, and check it resolves (coverage) and produces
    a finite kquant checkpoint. The imatrix steers the re-encode; this asserts the
    plumbing (load -> map -> per-tensor steer -> save), not a quality delta."""
    from mlx_lm.tuner.utils import load_adapters

    from mlx_kquant.cli import fuse as fuse_cli
    from mlx_kquant.codec_geometry import in_features
    from mlx_kquant.loader import load
    from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora

    base = _make_base(tmp_path)
    adapter = tmp_path / "adapter"
    _mint_adapter(base, adapter)

    # Enumerate the fusable layers + their input widths the same way fuse does,
    # and mint an imatrix keyed by their HF paths (identity match, no remap).
    patch_mlx_lm_lora()
    probe, _ = load(base)
    load_adapters(probe, str(adapter))
    vectors = {}
    for name, m in probe.named_modules():
        if hasattr(m, "fuse"):
            in_dims = in_features(m.linear.kquant_type, m.linear["weight"].shape[1])
            vectors[f"{name}.weight"] = np.ones(in_dims, dtype=np.float32)
    assert vectors  # the adapter wrapped at least one kquant linear

    imat = tmp_path / "base.imatrix.dat"
    _write_imatrix_dat(imat, vectors)

    out = tmp_path / "fused-kq-imat"
    args = argparse.Namespace(
        model=str(base),
        adapter_path=str(adapter),
        save_path=str(out),
        dequantize=False,
        imatrix=str(imat),
        trust_remote_code=False,
    )
    assert fuse_cli.cmd(args) == 0
    log = capsys.readouterr().out
    assert f"imatrix coverage: {len(vectors)}/{len(vectors)}" in log

    fused, fconfig = load(out)
    assert fconfig["quantization"]["mode"] == "kquant"
    fout = fused(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(fout)
    assert bool(mx.all(mx.isfinite(fout)).item())


def test_fuse_cli_dequantize(tmp_path, monkeypatch):
    from mlx_lm.utils import load_model

    from mlx_kquant.cli import fuse as fuse_cli

    # Pin the reference forward to the scalar CPU path: this test gates the
    # float MERGE (bf16 arithmetic-order noise, < 1e-2), and on the CPU device
    # the NEON int8 path's activation-q8 error (~7e-3) on the KQuant reference
    # side would stack on top of that gate.
    monkeypatch.setenv("KQ_CPU_NEON", "0")

    base = _make_base(tmp_path)
    adapter = tmp_path / "adapter"
    _mint_adapter(base, adapter)
    out = tmp_path / "fused-f16"

    x = mx.array([[1, 2, 3, 4, 5]])
    ref = _reference_lora_out(base, adapter, x)

    args = argparse.Namespace(
        model=str(base),
        adapter_path=str(adapter),
        save_path=str(out),
        dequantize=True,
        imatrix=None,
        trust_remote_code=False,
    )
    assert fuse_cli.cmd(args) == 0

    # No quantization block - a plain float checkpoint stock mlx_lm can load.
    import json

    cfg = json.loads((out / "config.json").read_text())
    assert "quantization" not in cfg and "quantization_config" not in cfg
    model, _ = load_model(out, lazy=False)
    assert not any(
        type(m).__name__.startswith("KQuant") for _, m in model.named_modules()
    )
    fout = model(x)
    mx.eval(fout)
    assert bool(mx.all(mx.isfinite(fout)).item())
    # The float merge has no re-quant error; the small residual vs the LoRA-
    # applied model is bf16 arithmetic-order noise (fuse folds scale*B@A into the
    # weight, the reference applies it as x@A@B) - far tighter than re-encoding.
    assert _rel(fout, ref) < 1e-2
