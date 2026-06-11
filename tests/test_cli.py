"""CLI dispatcher tests - argument parsing + the no-[tools]/no-GPU verify legs.

Importing the CLI must not pull in mlx-lm (heavy imports stay inside ``cmd``), so
these run on a base install. ``verify --codecs`` / ``--presets`` dispatch no ops.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from mlx_kquant.cli import _build_parser, main
from mlx_kquant.codec_geometry import bytes_per_row


def test_version_exits_zero():
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args(["--version"])
    assert e.value.code == 0


def test_no_command_errors():
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args([])
    assert e.value.code == 2


@pytest.mark.parametrize(
    "cmd", ["quantize", "calibrate-imatrix", "verify", "run", "fuse", "inspect"]
)
def test_subcommand_help_exits_zero(cmd):
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args([cmd, "--help"])
    assert e.value.code == 0


def test_quantize_requires_a_recipe():
    # one of --preset / --kquant-type is required.
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["quantize", "--model", "m", "--mlx-path", "o"])


def test_quantize_recipe_is_mutually_exclusive():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "quantize",
                "--model",
                "m",
                "--mlx-path",
                "o",
                "--preset",
                "q4_k_m",
                "--kquant-type",
                "q4_k",
            ]
        )


def test_quantize_parses_ok():
    args = _build_parser().parse_args(
        ["quantize", "--model", "m", "--mlx-path", "o", "--preset", "q4_k_m"]
    )
    assert args.command == "quantize"
    assert args.preset == "q4_k_m"
    assert callable(args.func)


def test_quantize_rejects_unknown_preset():
    # choices= catches recipe typos in argparse, before any model load.
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args(
            ["quantize", "--model", "m", "--mlx-path", "o", "--preset", "q4km"]
        )
    assert e.value.code == 2


def test_quantize_rejects_unknown_codec():
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args(
            ["quantize", "--model", "m", "--mlx-path", "o", "--kquant-type", "q4k"]
        )
    assert e.value.code == 2


def test_run_sampling_and_template_flags_parse():
    args = _build_parser().parse_args(
        [
            "run",
            "--model",
            "m",
            "--top-p",
            "0.9",
            "--top-k",
            "40",
            "--min-p",
            "0.05",
            "--seed",
            "7",
            "--system-prompt",
            "be brief",
            "--no-chat-template",
            "--chat-template-config",
            '{"enable_thinking": false}',
        ]
    )
    assert args.top_p == 0.9
    assert args.top_k == 40
    assert args.min_p == 0.05
    assert args.seed == 7
    assert args.no_chat_template is True


def test_chat_interruptible_absorbs_ctrl_c_mid_stream(capsys):
    # Ctrl-C during generation lands inside the wrapped generator; absorbing
    # it ends the turn so the REPL survives instead of the session dying.
    from mlx_kquant.cli.chat import _interruptible

    def stream():
        yield "a"
        raise KeyboardInterrupt

    assert list(_interruptible(stream)()) == ["a"]
    assert "canceled" in capsys.readouterr().out


def test_chat_interruptible_is_transparent():
    from mlx_kquant.cli.chat import _interruptible

    def stream(n, step=1):
        yield from range(0, n, step)

    assert list(_interruptible(stream)(6, step=2)) == [0, 2, 4]


def test_passthrough_commands_registered():
    # lora / chat are pre-argparse pass-throughs but must still appear in the
    # registered command list for the top-level --help.
    import argparse

    parser = _build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert "lora" in sub.choices
    assert "chat" in sub.choices


def test_verify_requires_a_target():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["verify"])


def test_verify_codecs_runs():
    # No [tools], no GPU dispatch - lists codecs + metallib status.
    assert main(["verify", "--codecs"]) == 0


def test_verify_presets_runs():
    assert main(["verify", "--presets"]) == 0


def test_verify_presets_differentiates_variants(capsys):
    # The s/m/xl variants share role maps; the output must surface the path /
    # layer bumps that actually distinguish them.
    assert main(["verify", "--presets"]) == 0
    out = capsys.readouterr().out
    assert "path bumps:" in out
    assert ".mlp.down_proj=q6_k" in out  # q4_k_xl / q5_k_xl path bump
    assert "use_more_bits" in out  # the _m layer-position cadence


def _fake_kquant_checkpoint(d, codec="q8_0", path="model.layers.0.mlp.down_proj"):
    """Write a minimal kquant checkpoint inspect can read: a config with a
    per-tensor map plus a uint8 ``<path>.weight`` of the right packed shape. No
    GPU, no real encode - inspect only reads the config + safetensors headers."""
    d.mkdir(parents=True, exist_ok=True)
    out_dims, in_dims = 4, 32
    row = bytes_per_row(codec, in_dims)
    wire = mx.zeros((out_dims, row), dtype=mx.uint8)
    mx.save_safetensors(str(d / "model.safetensors"), {f"{path}.weight": wire})
    cfg = {
        "model_type": "llama",
        "num_hidden_layers": 1,
        "quantization": {"mode": "kquant", "per_tensor": {path: codec}},
    }
    (d / "config.json").write_text(json.dumps(cfg))
    return d, out_dims, in_dims


def test_inspect_parses_ok():
    args = _build_parser().parse_args(["inspect", "--model", "m"])
    assert args.command == "inspect"
    assert args.as_json is False
    assert callable(args.func)


def test_inspect_reads_kquant_checkpoint(tmp_path, capsys):
    d, _, in_dims = _fake_kquant_checkpoint(tmp_path / "ckpt")
    assert main(["inspect", "--model", str(d)]) == 0
    out = capsys.readouterr().out
    assert "quantized tensors=1" in out
    assert "q8_0" in out
    assert "down_proj" in out
    # logical shape recovers the in-features from the packed wire width.
    assert str(in_dims) in out


def test_inspect_json(tmp_path, capsys):
    d, out_dims, in_dims = _fake_kquant_checkpoint(tmp_path / "ckpt")
    assert main(["inspect", "--model", str(d), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["num_quantized"] == 1
    assert payload["codec_histogram"] == {"q8_0": 1}
    (tensor,) = payload["tensors"]
    assert tensor["codec"] == "q8_0"
    assert tensor["kind"] == "linear"
    assert tensor["path"].endswith("down_proj")
    assert tensor["logical_shape"] == [out_dims, in_dims]


def test_inspect_rejects_non_kquant(tmp_path, capsys):
    d = tmp_path / "plain"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert main(["inspect", "--model", str(d)]) == 1
    assert "not a kquant checkpoint" in capsys.readouterr().err


def test_quantize_rejects_quantized_source(tmp_path, capsys):
    # Pointing quantize at an already-quantized checkpoint (e.g. an
    # mlx-community 4-bit repo) must fail clearly, before the model load.
    pytest.importorskip("mlx_lm")
    d = tmp_path / "quantized-src"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {"model_type": "llama", "quantization": {"bits": 4, "group_size": 64}}
        )
    )
    rc = main(
        [
            "quantize",
            "--model",
            str(d),
            "--mlx-path",
            str(tmp_path / "out"),
            "--kquant-type",
            "q4_k",
        ]
    )
    assert rc == 1
    assert "already quantized" in capsys.readouterr().err


def test_quantize_then_verify_e2e(tmp_path):
    """End-to-end through the CLI: a float source -> quantize -> verify --model."""
    pytest.importorskip("mlx_lm")
    import json

    import mlx.core as mx
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.utils import save_model

    cfg = dict(
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
    src = tmp_path / "src"
    src.mkdir()
    model = Model(ModelArgs.from_dict(cfg))
    mx.eval(model.parameters())
    save_model(src, model)
    (src / "config.json").write_text(json.dumps(cfg))

    out = tmp_path / "out"
    rc = main(
        ["quantize", "--model", str(src), "--mlx-path", str(out), "--preset", "q4_k_m"]
    )
    assert rc == 0
    assert (out / "config.json").exists()
    assert main(["verify", "--model", str(out)]) == 0
