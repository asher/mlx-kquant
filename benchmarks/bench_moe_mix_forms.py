"""Timing of the score-mixed down gather by kernel form, one process.

Source of kq_moe_mix_dd_first and the slot-parallel entry in
src/kquant_moe_glu.cpp.

Each arm forces one form through KQ_MOE_MIX_FORM, which the op reads live
per call: `loop` (every slot in one thread), `sp` (one simdgroup pair per
slot) and `dd` (the dedupe row-pair kernel, 2 rows and up). The `default`
arm leaves the form to the policy.

Weights are real codec output: --seed-rows float rows are quantized with
kq.quantize and tiled to the expert stack. Before timing, the output of
every arm is compared with the loop kernel's. The stack is sized past
--stream-mb, and every call of a chain routes to its own experts, so the
weights stream from DRAM. A sample is a chain of calls in one eval, each call's input
tied to the previous output through one small elementwise op. The `glue`
arm times the tie alone; reported times subtract it. Arm order reverses on
odd rounds.

Rows route in pairs (0 with 1, 2 with 3). --shared sets how many experts
the second row of a pair shares with the first.

Cells: codec x geometry x rows x shared experts. Markdown + JSON out.
"""

import argparse
import json
import os
import statistics
import time

ENV = "KQ_MOE_MIX_FORM"
os.environ.setdefault(ENV, "default")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import mlx_kquant as kq  # noqa: E402
from mlx_kquant.codec_geometry import CODEC_GEOMETRY  # noqa: E402

ARMS = ["default", "loop", "sp", "dd"]
# qwen4exp down, a GLM-5.3-Flash-like down and a wide down: experts per
# row, output rows N, input width K.
DEFAULT_GEOMS = "10x2560x640,8x4096x2048,8x4096x4096"


def make_stack(codec, N, K, stream_mb, seed_rows, rng):
    wpb = CODEC_GEOMETRY[codec][3]
    if K % wpb:
        raise ValueError(f"K {K} is not a multiple of the {wpb}-weight block")
    wf = mx.array((rng.standard_normal((seed_rows, K)) * 0.05).astype(np.float32))
    im = mx.ones((K,), dtype=mx.float32) if codec.startswith("iq") else None
    rows, _ = kq.quantize(wf, codec, im)
    row = rows.shape[-1]
    E = max(32, -(-stream_mb * (1 << 20) // (N * row)))
    one = mx.tile(rows, (-(-N // seed_rows), 1))[:N]
    w = mx.contiguous(mx.broadcast_to(one, (E, N, row)))
    sw = mx.contiguous(one)
    mx.eval(w, sw)
    return w, sw, E


def make_routes(rng, E, S, T, shared, calls):
    out = []
    for _ in range(calls):
        rows = []
        for t in range(T):
            if t % 2 == 1 and shared > 0:
                prev = rows[-1]
                keep = rng.choice(S, size=shared, replace=False)
                pool = np.setdiff1d(np.arange(E), prev)
                row = rng.choice(pool, size=S, replace=False)
                row[keep] = prev[keep]
            else:
                row = rng.choice(E, size=S, replace=False)
            rows.append(row)
        out.append(mx.array(np.stack(rows).astype(np.uint32)))
    return out


def call(codec, w, sw, h, idx, sc, shexp):
    if shexp:
        return kq.gather_qmv_mix_kq(h, w, sw, codec, idx, sc)
    return kq.gather_qmv_mix_ns_kq(h, w, codec, idx, sc)


def chain(codec, w, sw, h, routes, sc, shexp, run):
    tie = None
    for idx in routes:
        hh = h if tie is None else h + (tie * 0).astype(h.dtype)
        y = call(codec, w, sw, hh, idx, sc, shexp) if run else hh
        tie = y.reshape(-1)[:1]
    return tie


def bench_cell(codec, geom, T, shared, args, rng):
    S, N, K = geom
    w, sw, E = make_stack(codec, N, K, args.stream_mb, args.seed_rows, rng)
    slots = S + 1 if args.shexp else S
    dtype = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    h = (mx.random.normal((T, slots, K)) * 0.5).astype(dtype)
    sc = mx.full((T, slots), 1.0 / S, dtype=mx.float32)
    routes = make_routes(rng, E, S, T, shared, args.calls)
    mx.eval(h, sc, routes)
    arms = [a for a in ARMS if a != "dd" or T >= 2]
    outs = {}
    for arm in arms:
        os.environ[ENV] = arm
        y = call(codec, w, sw, h, routes[0], sc, args.shexp)
        mx.eval(y)
        outs[arm] = y.astype(mx.float32)
    for arm in arms:
        # forms on another lane count sum in another order: one ulp of the
        # 8-bit mantissa
        if not mx.allclose(outs[arm], outs["loop"], rtol=2**-6, atol=1e-5).item():
            raise SystemExit(f"{codec} {geom} T{T}: {arm} differs from loop")
    times = {a: [] for a in ["glue", *arms]}
    order = ["glue", *arms]
    for rnd in range(args.warm + args.rounds):
        for arm in order if rnd % 2 == 0 else order[::-1]:
            os.environ[ENV] = "default" if arm == "glue" else arm
            t = chain(codec, w, sw, h, routes, sc, args.shexp, arm != "glue")
            t0 = time.perf_counter()
            mx.eval(t)
            dt = 1e6 * (time.perf_counter() - t0) / args.calls
            if rnd >= args.warm:
                times[arm].append(dt)
    glue = statistics.median(times["glue"])
    return {a: statistics.median(times[a]) - glue for a in arms}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--codecs", nargs="+", default=["q2_0", "iq3_xxs", "q4_k"])
    ap.add_argument(
        "--geoms",
        default=DEFAULT_GEOMS,
        help="comma-separated SxNxK: experts per row, output rows, input width",
    )
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument(
        "--shared",
        type=int,
        nargs="+",
        default=[0, 3, 6],
        help="experts the second row of a pair shares with the first",
    )
    ap.add_argument(
        "--shexp",
        action="store_true",
        help="time gather_qmv_mix_kq with a shared expert in the same codec",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--calls", type=int, default=16, help="calls per sample")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--stream-mb", type=int, default=512)
    ap.add_argument(
        "--seed-rows",
        type=int,
        default=64,
        help="distinct float rows quantized and tiled to the stack",
    )
    ap.add_argument(
        "--cb-ops",
        type=int,
        default=0,
        help="ops per command buffer through kq.set_cb_caps; 0 keeps the MLX default",
    )
    ap.add_argument("--cb-mb", type=int, default=100000)
    ap.add_argument("--out", default="moe_mix_forms")
    args = ap.parse_args()

    if args.cb_ops:
        kq.set_cb_caps(args.cb_ops, args.cb_mb)
    geoms = [tuple(int(v) for v in g.split("x")) for g in args.geoms.split(",")]
    rng = np.random.default_rng(3)
    cells = []
    for codec in args.codecs:
        for geom in geoms:
            for T in args.rows:
                for shared in args.shared if T >= 2 else [0]:
                    if shared > geom[0]:
                        continue
                    try:
                        t = bench_cell(codec, geom, T, shared, args, rng)
                    except (RuntimeError, ValueError) as e:
                        print(f"{codec} {geom}: skipped, {str(e)[:100]}", flush=True)
                        break
                    forced = {a: v for a, v in t.items() if a != "default"}
                    best = min(forced, key=forced.get)
                    cells.append(
                        {
                            "codec": codec,
                            "geom": "x".join(str(v) for v in geom),
                            "rows": T,
                            "shared": shared,
                            "us": t,
                            "best": best,
                        }
                    )
                    print(
                        f"{codec} {cells[-1]['geom']} T{T} shared {shared} "
                        + " ".join(f"{a}:{v:.1f}" for a, v in t.items())
                        + f" best={best}",
                        flush=True,
                    )
    os.environ[ENV] = "default"

    with open(args.out + ".json", "w") as f:
        json.dump(
            {"shexp": args.shexp, "dtype": args.dtype, "cells": cells}, f, indent=1
        )
    lines = [
        "# Score-mixed down gather, microseconds per call",
        "",
        "| codec | S x N x K | rows | shared | " + " | ".join(ARMS) + " | best |",
        "|---|---|---|---|" + "---|" * (len(ARMS) + 1),
    ]
    for c in cells:
        lines.append(
            f"| {c['codec']} | {c['geom']} | {c['rows']} | {c['shared']} | "
            + " | ".join(f"{c['us'][a]:.1f}" if a in c["us"] else "-" for a in ARMS)
            + f" | {c['best']} |"
        )
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", args.out + ".md", args.out + ".json")


if __name__ == "__main__":
    main()
