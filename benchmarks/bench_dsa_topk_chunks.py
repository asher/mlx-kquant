"""Timing of dsa_topk_indices against the chunks one row is split over.

Source of kq_dsa_topk_chunks in src/kquant_dsa_indexer.cpp.

Each arm forces a chunk count through KQ_DSA_TOPK_CHUNKS, which the op
reads live per dispatch, so every arm shares one process. 1 is the
one-dispatch kernel. The `default` arm leaves the count to the policy.

A sample is a chain of calls in one eval, each call's scores tied to the
previous selection through one small elementwise op, so the calls
serialize the way the selects of a forward do. The `glue` arm times the
tie alone; reported times subtract it. Arm order reverses on odd rounds.
Before timing, every arm's selection is checked against the one-dispatch
kernel's, element for element.

Cells: row length K x rows x chunks. Markdown + JSON out.
"""

import argparse
import json
import os
import statistics
import time

import mlx.core as mx

import mlx_kquant as kq

ENV = "KQ_DSA_TOPK_CHUNKS"


def _set(chunks):
    if chunks == "default":
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = str(chunks)


def _chain(scores, topk, calls, select):
    s = scores
    for _ in range(calls):
        if select:
            o = kq.dsa_topk_indices(s, topk, True)
        else:
            o = s.reshape(-1)[:2].view(mx.uint32)
        s = scores + (o.reshape(-1)[:1] * 0).astype(scores.dtype)
    return s


def bench_cell(K, rows, topk, arms, calls, rounds, warm):
    mx.random.seed(K + rows)
    scores = mx.random.normal((1, 1, rows, K)).astype(mx.bfloat16)
    tied = (mx.round(mx.random.normal((1, 1, rows, K)) * 3.0)).astype(mx.bfloat16)
    mx.eval(scores, tied)
    for x in (scores, tied):
        _set(1)
        ref = kq.dsa_topk_indices(x, topk, True)
        mx.eval(ref)
        for arm in arms:
            _set(arm)
            got = kq.dsa_topk_indices(x, topk, True)
            mx.eval(got)
            if not mx.array_equal(got, ref).item():
                raise SystemExit(f"chunks={arm} K={K} rows={rows}: selection differs")
    times = {a: [] for a in ["glue", *arms]}
    order = ["glue", *arms]
    for rnd in range(warm + rounds):
        for arm in order if rnd % 2 == 0 else order[::-1]:
            _set("default" if arm == "glue" else arm)
            s = _chain(scores, topk, calls, arm != "glue")
            t0 = time.perf_counter()
            mx.eval(s)
            dt = 1e6 * (time.perf_counter() - t0) / calls
            if rnd >= warm:
                times[arm].append(dt)
    glue = statistics.median(times["glue"])
    return {str(a): statistics.median(times[a]) - glue for a in arms}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--k",
        default="1024,2048,4096,8192,16384,32768,65536,262144",
        help="row lengths, comma separated",
    )
    ap.add_argument("--rows", default="1,4,16,64,512", help="row counts")
    ap.add_argument("--chunks", default="1,2,4,8,16", help="chunk counts to force")
    ap.add_argument("--topk", type=int, default=512, choices=[512, 2048])
    ap.add_argument("--calls", type=int, default=32, help="calls per sample")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument(
        "--cb-ops",
        type=int,
        default=0,
        help="ops per command buffer through kq.set_cb_caps; 0 keeps the MLX default",
    )
    ap.add_argument("--cb-mb", type=int, default=100000)
    ap.add_argument("--out", default="dsa_topk_chunks")
    args = ap.parse_args()

    if args.cb_ops:
        kq.set_cb_caps(args.cb_ops, args.cb_mb)
    ks = [int(v) for v in args.k.split(",")]
    rows = [int(v) for v in args.rows.split(",")]
    arms = [int(v) for v in args.chunks.split(",")] + ["default"]
    cells = []
    for K in ks:
        if K < args.topk:
            continue
        for r in rows:
            calls = args.calls if r * K <= 1 << 22 else max(4, args.calls // 4)
            t = bench_cell(K, r, args.topk, arms, calls, args.rounds, args.warm)
            forced = {a: v for a, v in t.items() if a != "default"}
            best = min(forced, key=forced.get)
            cells.append({"K": K, "rows": r, "us": t, "best": int(best)})
            print(
                f"K={K} rows={r} "
                + " ".join(f"{a}:{v:.1f}" for a, v in t.items())
                + f" best={best}",
                flush=True,
            )
    os.environ.pop(ENV, None)

    with open(args.out + ".json", "w") as f:
        json.dump({"topk": args.topk, "cells": cells}, f, indent=1)
    head = [str(a) for a in arms]
    lines = [
        f"# dsa_topk_indices, microseconds per call, topk {args.topk}",
        "",
        "| K | rows | " + " | ".join(head) + " | best |",
        "|---|---|" + "---|" * (len(head) + 1),
    ]
    for c in cells:
        lines.append(
            f"| {c['K']} | {c['rows']} | "
            + " | ".join(f"{c['us'][a]:.1f}" for a in head)
            + f" | {c['best']} |"
        )
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", args.out + ".md", args.out + ".json")


if __name__ == "__main__":
    main()
