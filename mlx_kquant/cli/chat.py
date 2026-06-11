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
  keys, Ctrl-A/E) with up-arrow history, persisted across sessions;
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
        "apply; run `mlx-kquant chat --help`.",
    )


def _interruptible(stream_fn):
    """Wrap a ``stream_generate``-shaped generator to absorb Ctrl-C.

    A SIGINT that lands while a token is being generated raises
    ``KeyboardInterrupt`` inside the wrapped generator's frame; absorbing it
    here simply ends the iteration, so mlx-lm's REPL loop finishes the turn
    and prompts again instead of the whole session dying. (The prompt cache
    keeps the partial reply; ``r`` resets the chat if that lingers.) A SIGINT
    at the idle ``input()`` prompt is raised in mlx-lm's frame, not here, and
    still exits the session.
    """

    def wrapped(*args, **kwargs):
        try:
            yield from stream_fn(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n[mlx-kquant] response canceled ('r' resets the chat)")

    return wrapped


def _wire_history(readline) -> None:
    """Persist the readline prompt history across chat sessions."""
    import atexit
    import os
    from pathlib import Path

    cache = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    hist = cache / "mlx-kquant" / "chat_history"
    try:
        hist.parent.mkdir(parents=True, exist_ok=True)
        if hist.exists():
            readline.read_history_file(hist)
    except OSError:
        return  # unwritable cache dir: line editing still works, just unsaved

    def _save() -> None:
        try:
            readline.set_history_length(1000)
            readline.write_history_file(hist)
        except OSError:
            pass

    atexit.register(_save)


def passthrough(rest: list[str]) -> int:
    """Apply the kquant patch and delegate ``rest`` to ``mlx_lm.chat``.

    Unlike the ``lora`` pass-through, ``--trust-remote-code`` is *not* stripped:
    mlx-lm's chat parser has its own (tokenizer) meaning for it, and the same
    flag also grants the kquant loader's ``model_file`` opt-in here.
    """
    import sys

    from .._deps import require_tools

    require_tools()

    # Imported for its side effect: input() becomes line-editable with
    # up-arrow history the moment the module exists.
    try:
        import readline
    except ImportError:  # pragma: no cover - absent on some builds
        readline = None
    if readline is not None:
        _wire_history(readline)

    import mlx_lm.chat as _chat

    from ..mlx_lm_patch import patch_mlx_lm_lora

    patch_mlx_lm_lora(trust_remote_code="--trust-remote-code" in rest)
    _chat.stream_generate = _interruptible(_chat.stream_generate)

    saved_argv = sys.argv
    sys.argv = ["mlx_lm.chat", *rest]
    try:
        _chat.main()
    except EOFError:
        print()  # Ctrl-D at the prompt: exit like the `q` command
    finally:
        sys.argv = saved_argv
    return 0
