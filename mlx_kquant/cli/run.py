"""``mlx-kquant run`` — load a kquant checkpoint and generate a few tokens."""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="load a checkpoint and generate text (end-to-end sanity)",
        description="Load a kquant checkpoint, apply the chat template if present, "
        "and stream a short generation.",
    )
    p.add_argument(
        "--model", required=True, help="kquant checkpoint dir or HF repo id."
    )
    p.add_argument("--prompt", default="Hello!", help="Prompt text.")
    p.add_argument(
        "--max-tokens", type=int, default=64, help="Tokens to generate. Default: 64."
    )
    p.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy). Default: 0.0.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow a checkpoint's custom model_file (arbitrary code) to load.",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    from .._deps import require_tools

    require_tools()

    from mlx_lm.generate import generate
    from mlx_lm.utils import load_tokenizer

    from ..loader import _resolve_path, load

    model, _ = load(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer = load_tokenizer(_resolve_path(args.model, None))

    prompt = args.prompt
    if getattr(tokenizer, "chat_template", None) is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
        )

    kwargs = {"max_tokens": args.max_tokens}
    if args.temp > 0:
        from mlx_lm.sample_utils import make_sampler

        kwargs["sampler"] = make_sampler(temp=args.temp)

    generate(model, tokenizer, prompt, verbose=True, **kwargs)
    return 0
