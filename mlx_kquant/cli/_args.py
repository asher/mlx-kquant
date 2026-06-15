"""Shared argv helpers for the mlx-lm pass-through subcommands (lora / chat).

These pass-throughs hand their arguments to mlx-lm's own CLIs, which silently
fall back to a default model (and, for lora, the WikiSQL dataset) when --model
is omitted - downloading from the Hub before doing anything useful. The
pass-throughs use :func:`has_opt` to detect whether the user supplied an
explicit source and, if not, refuse rather than trigger that download.
"""

from __future__ import annotations


def has_opt(argv: list[str], *names: str) -> bool:
    """Return True if any of the given option ``names`` appears in ``argv``.

    Recognizes ``--long``, ``--long=val``, ``-s``, and ``-sval`` forms; a value
    in a separate token is the bare ``--long`` / ``-s`` case. It does not expand
    argparse's prefix abbreviations (``--mod`` for ``--model``) - the guard only
    decides whether to refuse mlx-lm's default model/dataset, and its error
    names the exact flag to pass.
    """
    for tok in argv:
        for name in names:
            if tok == name or tok.startswith(name + "="):
                return True
            # short option (-s) with an attached value: -mFOO, -cfile.yaml
            if (
                len(name) == 2
                and name[0] == "-"
                and name[1] != "-"
                and len(tok) > 2
                and tok.startswith(name)
            ):
                return True
    return False
