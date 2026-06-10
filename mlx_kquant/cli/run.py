"""``mlx-kquant run`` - load a kquant checkpoint and generate a few tokens."""

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
        "--adapter-path",
        help="Optional LoRA adapter dir to attach at runtime (the base is not "
        "modified; use `mlx-kquant fuse` to merge it in permanently).",
    )
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
        "--top-p",
        type=float,
        default=0.0,
        help="Nucleus sampling cutoff (0 = off). Default: 0.0.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (0 = off). Default: 0.",
    )
    p.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p cutoff, scaled by the top token's probability (0 = off). "
        "Default: 0.0.",
    )
    p.add_argument("--seed", type=int, help="PRNG seed for reproducible sampling.")
    p.add_argument("--system-prompt", help="System message for the chat template.")
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Feed the prompt raw (skip the tokenizer's chat template).",
    )
    p.add_argument(
        "--chat-template-config",
        help="JSON of extra chat-template kwargs, e.g. '{\"enable_thinking\": false}'.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow a checkpoint's custom model_file (arbitrary code) to load.",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    import json

    from .._deps import require_tools

    require_tools()

    import mlx.core as mx
    from mlx_lm.generate import generate
    from mlx_lm.utils import load_tokenizer

    from ..loader import _resolve_path, load

    # Parse before the model load so a JSON typo fails fast.
    template_kwargs = {}
    if args.chat_template_config:
        try:
            template_kwargs = json.loads(args.chat_template_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"--chat-template-config is not valid JSON: {e}") from e

    if args.seed is not None:
        mx.random.seed(args.seed)

    model, _ = load(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer = load_tokenizer(_resolve_path(args.model, None))

    if args.adapter_path:
        from mlx_lm.tuner.utils import load_adapters

        from ..mlx_lm_patch import patch_mlx_lm_lora

        patch_mlx_lm_lora()  # install the to_lora seam so the adapter attaches
        load_adapters(model, args.adapter_path)

    prompt = args.prompt
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if has_template and not args.no_chat_template:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": args.prompt})
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **template_kwargs
        )

    kwargs = {"max_tokens": args.max_tokens}
    if args.temp > 0 or args.top_p > 0 or args.min_p > 0 or args.top_k > 0:
        from mlx_lm.sample_utils import make_sampler

        kwargs["sampler"] = make_sampler(
            temp=args.temp, top_p=args.top_p, min_p=args.min_p, top_k=args.top_k
        )

    generate(model, tokenizer, prompt, verbose=True, **kwargs)
    return 0
