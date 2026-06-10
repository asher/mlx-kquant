"""``mlx-kquant verify`` - smoke-check the codecs / presets, or a checkpoint."""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "verify",
        help="smoke-check the codecs / presets, or a built checkpoint",
        description="Quick self-checks: the supported codecs and metallib status, "
        "the recipe presets, or a built checkpoint's forward pass.",
    )
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument(
        "--codecs",
        action="store_true",
        help="print the supported codecs and whether the metallib loads.",
    )
    what.add_argument("--presets", action="store_true", help="list the recipe presets.")
    what.add_argument(
        "--model",
        help="load a checkpoint, forward a tiny input, and report finite logits.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="With --model: allow a checkpoint's custom model_file "
        "(arbitrary code) to load.",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    import mlx_kquant as kq

    if args.codecs:
        print("codecs:", ", ".join(kq.codecs()))
        print("metallib loads:", kq.metallib_loads())
        return 0

    if args.presets:
        from ..recipes import _LAYER_POSITION_BUMPS, _PATH_BUMPS, KQUANT_PRESETS

        for name in sorted(KQUANT_PRESETS):
            entries = KQUANT_PRESETS[name]
            default = entries.get("default")
            roles = [
                f"{role}={codec}"
                for role, codec in sorted(entries.items())
                if role != "default" and codec != default
            ]
            line = f"  {name:10} default={default}"
            if roles:
                line += "  " + "  ".join(roles)
            print(line)
            bumps = _PATH_BUMPS.get(name)
            if bumps:
                btxt = "  ".join(f"{s}={c}" for s, c in sorted(bumps.items()))
                print(f"  {'':10}   path bumps: {btxt}")
            for suffix, (codec, rule) in sorted(
                _LAYER_POSITION_BUMPS.get(name, {}).items()
            ):
                print(f"  {'':10}   layer bumps: {suffix}={codec} on {rule} layers")
        print(
            "\n  Roles not listed take the default; path/layer bumps match on "
            "module-path suffix. The _s/_m/_xl variants differ in these bumps."
        )
        return 0

    # --model: full load + forward (needs [tools] + a GPU to decode).
    from .._deps import require_tools

    require_tools()

    import mlx.core as mx

    from ..loader import load

    model, config = load(args.model, trust_remote_code=args.trust_remote_code)
    out = model(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(out)
    finite = bool(mx.all(mx.isfinite(out)).item())
    quant = config.get("quantization") or config.get("quantization_config") or {}
    n = len(quant.get("per_tensor", {}))
    print(f"loaded {args.model}: {n} kquant tensors; forward finite={finite}")
    return 0 if finite else 1
