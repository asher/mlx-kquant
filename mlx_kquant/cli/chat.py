"""``mlx-kquant chat`` - interactive chat REPL on a kquant checkpoint.

A thin pass-through to mlx-lm's chat REPL: it applies the kquant patch (so the
REPL can load a kquant checkpoint and attach a LoRA adapter via
``--adapter-path``) and then hands every argument straight to ``mlx_lm.chat``.
All of mlx-lm's chat flags apply unchanged (``--temp``, ``--top-p``, ``--seed``,
``--system-prompt``, ``--max-tokens``, ...); run ``mlx-kquant chat --help`` to
see them.

    mlx-kquant chat --model my-q4-ckpt --temp 0.7

``--trust-remote-code`` is forwarded (mlx-lm reads it for the tokenizer) and
additionally opts the kquant loader into a checkpoint's custom ``model_file``.

On top of the unmodified mlx-lm loop, the shim upgrades the terminal it runs
in - no mlx-lm code is copied or re-implemented:

* importing :mod:`readline` makes the ``input()`` prompt line-editable (arrow
  keys, Ctrl-A/E) with up-arrow history, persisted across sessions
  (``$XDG_CACHE_HOME/mlx-kquant/chat_history``); ``--no-history`` keeps the
  session ephemeral (no file read or written; in-session recall still works);
* lines starting with ``/`` are shim commands, intercepted before mlx-lm's
  loop sees them: ``/history [on|off|clear]`` controls persistence at
  runtime, and ``/temp`` / ``/top-p`` / ``/top-k`` / ``/min-p`` /
  ``/max-tokens`` adjust sampling for subsequent responses (``/sampling``
  shows the current values; top-k and min-p go beyond mlx-lm's own chat
  flags, via the per-turn sampler the wrapped ``stream_generate`` rebuilds);
* **Ctrl-C during a response cancels it** and returns to the prompt (the
  ``stream_generate`` mlx-lm iterates is wrapped to absorb the interrupt);
  Ctrl-C at an idle prompt still exits the session;
* **Ctrl-D exits cleanly**, like the ``q`` command.

The dispatcher intercepts ``chat`` before argparse and calls
:func:`passthrough` directly (like ``lora``), so mlx-lm's own flags (including
``--help``) reach it untouched; :func:`add_parser` only registers the command
for the top-level ``--help`` list.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser(
        "chat",
        help="interactive chat REPL on a kquant checkpoint (mlx-lm chat)",
        add_help=False,
        description="Pass-through to mlx-lm's chat REPL with the kquant patch "
        "applied, so --model may be a kquant checkpoint. All mlx-lm chat flags "
        "apply; run `mlx-kquant chat --help`. Extra shim flag: --no-history "
        "(don't read or write the prompt-history file); in-chat, "
        "`/history [on|off|clear]` controls the same at runtime.",
    )


def _interruptible(stream_fn, state: dict | None = None):
    """Wrap a ``stream_generate``-shaped generator: absorb Ctrl-C, and apply
    any runtime sampling overrides from the shim's ``/`` commands.

    A SIGINT that lands while a token is being generated raises
    ``KeyboardInterrupt`` inside the wrapped generator's frame; absorbing it
    here simply ends the iteration, so mlx-lm's REPL loop finishes the turn
    and prompts again instead of the whole session dying. (The prompt cache
    keeps the partial reply; ``r`` resets the chat if that lingers.) A SIGINT
    at the idle ``input()`` prompt is raised in mlx-lm's frame, not here, and
    still exits the session.

    mlx-lm's loop builds a fresh ``sampler=`` for every turn from its parsed
    CLI args; once a ``/temp``-style command has run (``state["overridden"]``)
    this wrapper substitutes a sampler built from the shim's live values
    instead, mirroring the loop's own ``make_sampler`` call (including its
    xtc special tokens, rebuilt from the tokenizer it passes positionally).
    """

    def wrapped(*args, **kwargs):
        if state is not None and state.get("overridden"):
            from mlx_lm.sample_utils import make_sampler

            s = state["sampling"]
            tokenizer = args[1] if len(args) > 1 else kwargs["tokenizer"]
            kwargs["sampler"] = make_sampler(
                s["temp"],
                s["top_p"],
                min_p=s["min_p"],
                top_k=s["top_k"],
                xtc_threshold=s["xtc_threshold"],
                xtc_probability=s["xtc_probability"],
                xtc_special_tokens=(
                    tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                ),
            )
            kwargs["max_tokens"] = s["max_tokens"]
        try:
            yield from stream_fn(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n[mlx-kquant] response canceled ('r' resets the chat)")

    return wrapped


def _history_path():
    import os
    from pathlib import Path

    cache = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return cache / "mlx-kquant" / "chat_history"


def _wire_history(readline, enabled: bool) -> dict:
    """Set up prompt-history persistence; return its mutable control state.

    ``state["enabled"]`` is consulted at exit, so ``/history on|off`` can flip
    persistence mid-session. The file is only loaded once (``loaded``) - on
    startup when enabled, or lazily by ``/history on`` - so a later save
    merges prior history instead of overwriting it with just this session.
    """
    state = {"readline": readline, "enabled": enabled, "loaded": False}
    if readline is None:
        return state

    import atexit

    hist = _history_path()
    if enabled:
        try:
            if hist.exists():
                readline.read_history_file(hist)
            state["loaded"] = True
        except OSError:
            pass

    def _save() -> None:
        if not state["enabled"]:
            return
        try:
            hist.parent.mkdir(parents=True, exist_ok=True)
            readline.set_history_length(1000)
            readline.write_history_file(hist)
        except OSError:
            pass

    atexit.register(_save)
    return state


# Runtime-adjustable sampling knobs: /command -> (state key, parser).
_SAMPLING_COMMANDS = {
    "/temp": ("temp", float),
    "/top-p": ("top_p", float),
    "/top-k": ("top_k", int),
    "/min-p": ("min_p", float),
    "/max-tokens": ("max_tokens", int),
}


def _print_shim_help(state: dict) -> None:
    status = "on" if state["enabled"] else "off"
    print("[mlx-kquant] shim commands (the REPL's own are 'q' / 'r' / 'h'):")
    print(f"- '/history [on|off|clear]' control prompt-history saving ({status})")
    print("- '/temp /top-p /top-k /min-p /max-tokens <value>' adjust sampling")
    print("- '/sampling' to show the current sampling settings")
    print("- '/help' to display these commands")


def _print_sampling(state: dict) -> None:
    s = state["sampling"]
    knobs = "  ".join(f"{k.replace('_', '-')}={v}" for k, v in sorted(s.items()))
    print(f"[mlx-kquant] sampling: {knobs}")


def _handle_slash(line: str, state: dict) -> None:
    """Handle a shim ``/command`` line (never shown to mlx-lm's loop)."""
    readline = state["readline"]
    cmd, _, arg = line.strip().partition(" ")
    arg = arg.strip()
    if cmd in _SAMPLING_COMMANDS:
        key, cast = _SAMPLING_COMMANDS[cmd]
        if not arg:
            print(f"[mlx-kquant] {cmd[1:]} = {state['sampling'][key]}")
            return
        try:
            state["sampling"][key] = cast(arg)
        except ValueError:
            print(f"[mlx-kquant] {cmd} needs a {cast.__name__}, got {arg!r}")
            return
        state["overridden"] = True
        print(f"[mlx-kquant] {cmd[1:]} = {state['sampling'][key]} (next response)")
        return
    if cmd == "/sampling":
        _print_sampling(state)
        return
    if cmd != "/history":
        _print_shim_help(state)
        return
    if readline is None:
        print("[mlx-kquant] history unavailable (no readline on this build)")
        return
    arg = arg.strip()
    if arg == "off":
        state["enabled"] = False
        print("[mlx-kquant] history off (this session will not be saved)")
    elif arg == "on":
        if not state["loaded"]:
            try:
                if _history_path().exists():
                    readline.read_history_file(_history_path())
                state["loaded"] = True
            except OSError:
                pass
        state["enabled"] = True
        print("[mlx-kquant] history on")
    elif arg == "clear":
        readline.clear_history()
        try:
            _history_path().unlink(missing_ok=True)
        except OSError:
            pass
        print("[mlx-kquant] history cleared")
    else:
        status = "on" if state["enabled"] else "off"
        print(f"[mlx-kquant] history is {status} ({_history_path()})")


def _command_filter(real_input, state: dict):
    """An ``input()`` that consumes shim ``/commands`` and re-prompts.

    Installed as a module attribute on ``mlx_lm.chat``, where it shadows the
    builtin for that module only - mlx-lm's loop receives only the lines that
    are not shim commands.
    """

    def filtered(prompt: str = "") -> str:
        while True:
            line = real_input(prompt)
            if line.strip() == "h":
                # The REPL's own help follows; surface the shim commands with
                # it (its print_help is internal, so this prints just above).
                _print_shim_help(state)
                return line
            if not line.startswith("/"):
                return line
            _handle_slash(line, state)

    return filtered


def passthrough(rest: list[str]) -> int:
    """Apply the kquant patch and delegate ``rest`` to ``mlx_lm.chat``.

    Unlike the ``lora`` pass-through, ``--trust-remote-code`` is *not* stripped:
    mlx-lm's chat parser has its own (tokenizer) meaning for it, and the same
    flag also grants the kquant loader's ``model_file`` opt-in here.
    ``--no-history`` is the shim's own flag and is stripped before delegating.
    """
    import sys

    from .._deps import require_tools

    require_tools()

    no_history = "--no-history" in rest
    if no_history:
        rest = [a for a in rest if a != "--no-history"]

    # Imported for its side effect: input() becomes line-editable with
    # up-arrow history the moment the module exists.
    try:
        import readline
    except ImportError:  # pragma: no cover - absent on some builds
        readline = None
    state = _wire_history(readline, enabled=not no_history)

    import mlx_lm.chat as _chat

    from ..mlx_lm_patch import patch_mlx_lm_lora

    # Pre-parse with mlx-lm's own chat parser (so --help and arg errors exit
    # here, identically) to seed the runtime-adjustable sampling state from
    # the same values their loop will use.
    chat_args = _chat.setup_arg_parser().parse_args(rest)
    state["sampling"] = {
        "temp": chat_args.temp,
        "top_p": chat_args.top_p,
        "top_k": 0,
        "min_p": 0.0,
        "xtc_threshold": chat_args.xtc_threshold,
        "xtc_probability": chat_args.xtc_probability,
        "max_tokens": chat_args.max_tokens,
    }
    state["overridden"] = False

    patch_mlx_lm_lora(trust_remote_code="--trust-remote-code" in rest)
    _chat.stream_generate = _interruptible(_chat.stream_generate, state)
    _chat.input = _command_filter(input, state)

    print("[mlx-kquant] shim: '/history [on|off|clear]' - '/help' for commands")
    saved_argv = sys.argv
    sys.argv = ["mlx_lm.chat", *rest]
    try:
        _chat.main()
    except EOFError:
        print()  # Ctrl-D at the prompt: exit like the `q` command
    finally:
        sys.argv = saved_argv
    return 0
