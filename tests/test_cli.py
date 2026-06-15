"""CLI dispatcher tests - argument parsing + the no-[tools]/no-GPU verify legs.

Importing the CLI must not pull in mlx-lm (heavy imports stay inside ``cmd``), so
these run on a base install. ``verify --codecs`` / ``quantize --list-presets``
dispatch no ops.
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


def test_no_command_prints_help(capsys):
    # Bare invocation prints help and exits 0, rather than erroring out.
    assert main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: mlx-kquant")
    assert "quantize" in out


@pytest.mark.parametrize(
    "cmd", ["quantize", "calibrate-imatrix", "verify", "run", "fuse", "inspect"]
)
def test_subcommand_help_exits_zero(cmd):
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args([cmd, "--help"])
    assert e.value.code == 0


@pytest.mark.parametrize(
    "cmd", ["quantize", "calibrate-imatrix", "verify", "run", "fuse", "inspect"]
)
def test_bare_subcommand_prints_help(cmd, capsys):
    # A bare subcommand prints its full help and exits 0, not a terse
    # "required arguments" error. (lora/chat are pass-throughs, covered
    # by their own help handling, and would pull in mlx-lm here.)
    assert main([cmd]) == 0
    assert capsys.readouterr().out.startswith(f"usage: mlx-kquant {cmd}")


def test_has_opt_recognizes_option_forms():
    from mlx_kquant.cli._args import has_opt

    assert has_opt(["--model", "x"], "--model")
    assert has_opt(["--model=x"], "--model")
    assert has_opt(["-m", "x"], "-m", "--model")
    assert has_opt(["-mx"], "-m", "--model")
    assert has_opt(["-c", "f.yaml"], "-c", "--config")
    assert has_opt(["-cf.yaml"], "-c", "--config")
    assert has_opt(["--config=f"], "-c", "--config")
    assert not has_opt(["--train"], "--model")
    assert not has_opt([], "--model")


def test_lora_guard_refuses_default_download(capsys):
    # `lora --train` with no model/data must error (naming both) rather than
    # download mlx-lm's Qwen3-0.6b + WikiSQL defaults. The guard fires before
    # mlx-lm is imported, so this stays a base-install test.
    assert main(["lora", "--train"]) == 1
    err = capsys.readouterr().err
    assert "--model" in err and "--data" in err


def test_chat_guard_refuses_default_download(capsys):
    # `chat` with flags but no --model must error rather than download
    # mlx-lm's default model.
    assert main(["chat", "--temp", "0.7"]) == 1
    assert "--model" in capsys.readouterr().err


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


def test_chat_command_filter_intercepts_slash_lines(capsys):
    # /-prefixed lines are consumed by the shim (with a re-prompt); only
    # ordinary lines reach mlx-lm's loop.
    from mlx_kquant.cli.chat import _command_filter

    lines = iter(["/bogus", "hello"])
    state = {"readline": None, "enabled": True, "loaded": False}
    filtered = _command_filter(lambda prompt="": next(lines), state)
    assert filtered(">> ") == "hello"
    assert "shim commands" in capsys.readouterr().out


def test_chat_sampling_slash_commands(capsys):
    from mlx_kquant.cli.chat import _handle_slash

    state = {
        "readline": None,
        "enabled": True,
        "loaded": False,
        "overridden": False,
        "sampling": {
            "temp": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "xtc_threshold": 0.0,
            "xtc_probability": 0.0,
            "max_tokens": 256,
        },
    }
    _handle_slash("/temp 0.8", state)
    assert state["sampling"]["temp"] == 0.8
    assert state["overridden"] is True
    _handle_slash("/top-k 40", state)
    assert state["sampling"]["top_k"] == 40
    _handle_slash("/presence-penalty 1.5", state)
    assert state["sampling"]["presence_penalty"] == 1.5
    _handle_slash("/temp", state)  # no value: prints current, no change
    _handle_slash("/temp abc", state)  # bad value: error, no change
    assert state["sampling"]["temp"] == 0.8
    _handle_slash("/sampling", state)
    out = capsys.readouterr().out
    assert "temp = 0.8" in out
    assert "needs a float" in out
    assert "max-tokens=256" in out


def test_chat_interruptible_applies_sampling_override():
    pytest.importorskip("mlx_lm")
    from mlx_kquant.cli.chat import _interruptible

    class FakeTokenizer:
        eos_token_ids = [2]

        def encode(self, s):
            return [1]

    captured = {}

    def stream(model, tokenizer, prompt, max_tokens=0, sampler=None, **kw):
        captured.update(max_tokens=max_tokens, sampler=sampler, **kw)
        yield "tok"

    state = {
        "overridden": True,
        "sampling": {
            "temp": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "xtc_threshold": 0.0,
            "xtc_probability": 0.0,
            "max_tokens": 99,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "frequency_penalty": 0.0,
        },
    }
    out = list(
        _interruptible(stream, state)(
            "model", FakeTokenizer(), "hi", max_tokens=256, sampler="orig"
        )
    )
    assert out == ["tok"]
    assert captured["max_tokens"] == 99
    assert callable(captured["sampler"])  # rebuilt, not the original string
    # presence_penalty=1.5 is non-neutral -> exactly one logits processor;
    # the neutral repetition (1.0) and frequency (0.0) values map to None.
    assert len(captured["logits_processors"]) == 1


class _FakeReadlineBuffer:
    """Just enough readline for completer / startup-hook tests."""

    def __init__(self, buffer=""):
        self.buffer = buffer
        self.hooks = []

    def get_line_buffer(self):
        return self.buffer

    def set_startup_hook(self, hook):
        self.hooks.append(hook)

    def insert_text(self, text):
        self.inserted = text


def test_chat_load_prefills_next_prompt(tmp_path, capsys):
    from mlx_kquant.cli.chat import _command_filter, _handle_slash

    f = tmp_path / "prompt.txt"
    f.write_text("review this diff\n")
    rl = _FakeReadlineBuffer()
    state = {"readline": rl, "enabled": True, "loaded": False}

    _handle_slash(f"/load {f}", state)
    assert state["pending_insert"] == "review this diff"
    assert "loaded" in capsys.readouterr().out

    # The next input() gets the startup hook (set, then cleared), and the
    # user-edited line is what reaches mlx-lm's loop.
    filtered = _command_filter(lambda prompt="": "review this diff please", state)
    assert filtered(">> ") == "review this diff please"
    assert len(rl.hooks) == 2 and rl.hooks[1] is None
    rl.hooks[0]()  # the installed hook inserts the loaded text
    assert rl.inserted == "review this diff"
    assert state["pending_insert"] is None


def test_chat_load_errors(tmp_path, capsys):
    from mlx_kquant.cli.chat import _handle_slash

    state = {"readline": None, "enabled": True, "loaded": False}
    _handle_slash("/load", state)
    _handle_slash(f"/load {tmp_path}/nope.txt", state)
    out = capsys.readouterr().out
    assert "usage: /load" in out
    assert "/load:" in out
    assert "pending_insert" not in state


def test_chat_completer():
    from mlx_kquant.cli.chat import _make_completer

    state: dict = {}
    rl = _FakeReadlineBuffer("/te")
    complete = _make_completer(rl, state)
    assert complete("/te", 0) == "/temp "
    assert complete("/te", 1) is None

    rl.buffer = "/history o"
    assert complete("o", 0) == "on"
    assert complete("o", 1) == "off"
    assert complete("o", 2) is None

    rl.buffer = "ordinary chat text"
    assert complete("text", 0) is None


def test_chat_completer_load_paths(tmp_path):
    from mlx_kquant.cli.chat import _make_completer

    (tmp_path / "prompt.txt").write_text("x")
    (tmp_path / "prompts").mkdir()
    rl = _FakeReadlineBuffer(f"/load {tmp_path}/pro")
    complete = _make_completer(rl, {})
    got = {complete(f"{tmp_path}/pro", i) for i in range(2)}
    assert got == {f"{tmp_path}/prompt.txt", f"{tmp_path}/prompts/"}


def test_chat_clear_forwards_reset(capsys):
    # /clear hands mlx-lm its own 'r' (conversation reset) after wiping the
    # screen - the reset semantics stay theirs.
    from mlx_kquant.cli.chat import _command_filter

    state = {"readline": None, "enabled": True, "loaded": False}
    filtered = _command_filter(lambda prompt="": "/clear", state)
    assert filtered(">> ") == "r"
    assert "conversation reset" in capsys.readouterr().out


def test_chat_h_command_also_shows_shim_help(capsys):
    # 'h' is mlx-lm's help command: it must still reach their loop, with the
    # shim's command list printed alongside.
    from mlx_kquant.cli.chat import _command_filter

    state = {"readline": None, "enabled": True, "loaded": False}
    filtered = _command_filter(lambda prompt="": "h", state)
    assert filtered(">> ") == "h"
    out = capsys.readouterr().out
    assert "/history" in out
    assert "/help" in out


def test_chat_history_toggle_and_clear(tmp_path, monkeypatch, capsys):
    from mlx_kquant.cli import chat as chat_mod

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    class FakeReadline:
        cleared = False

        def clear_history(self):
            self.cleared = True

        def read_history_file(self, path):
            pass

    rl = FakeReadline()
    state = {"readline": rl, "enabled": True, "loaded": True}

    chat_mod._handle_slash("/history off", state)
    assert state["enabled"] is False
    chat_mod._handle_slash("/history on", state)
    assert state["enabled"] is True

    hist = chat_mod._history_path()
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text("old\n")
    chat_mod._handle_slash("/history clear", state)
    assert rl.cleared
    assert not hist.exists()

    chat_mod._handle_slash("/history", state)
    assert "history is on" in capsys.readouterr().out


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


def test_quantize_list_presets_exits_zero():
    # A print-and-exit action (like --version); fires before the required args.
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args(["quantize", "--list-presets"])
    assert e.value.code == 0


def test_quantize_list_presets_differentiates_variants(capsys):
    # The s/m/xl variants share role maps; the output must surface the path /
    # layer bumps that actually distinguish them.
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["quantize", "--list-presets"])
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
