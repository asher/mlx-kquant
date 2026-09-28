"""Quantized matmul arms at DiT linear shapes, paired in one process.

Cells are (op, rows per item L, batch B, codec). A batch of two runs as
[2, L, K], which the op flattens to 2L rows, so the tile policy can pick a
different tile at B 2 than at B 1. Arms, switched by live levers
(KQ_TUNE_LIVE=1 is set on import of dit_common):

  dense     stock matmul on a dense weight in the activation dtype
  kq        kq.quantized_matmul at the current policy
  kq_bm64   kq with KQ_NAX_TILE=64,64,2,2 (KQ_QMM_TILE without NAX)
  kq_bm128  kq with KQ_NAX_TILE=128,64,2,2
  deq       kq.dequantize, then stock matmul, every call
  affine    mx.quantized_matmul, MLX affine, group 64, 4 bits (8 for q8_0)
  kq_aa     the kq arm a second time, for the A versus A noise floor

dense and affine do not depend on the codec, so they run once per shape
visit and pair with every codec in it.

Order: one discarded warmup pass, which also records the kernel each kq
arm ran, then rounds of Thue-Morse slots. Each slot visits every shape in
a fresh interleaved order. A visit opens with --visit-warmup discarded
calls, since the first call after a shape switch runs slow. Inside a visit
the arm list reverses on odd slots, so the two calls of every pair swap
order with the Thue-Morse bit and a round of 8 slots splits each pair 4/4.
A paired ratio compares calls made seconds apart. Round 0 is the
first-round column, later rounds the settled column.

Per cell, ratios carry a 95 percent CI of the paired median, from order
statistics; its half-width is the cell's noise floor. Per op, the pooled
median of all settled pairs is the aggregate, reported beside the spread
of the per-cell medians. Markdown and JSON out; the JSON keeps every
sample and pair.

A versus A: --arms kq,kq_aa. Plan only: --plan-only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import dit_common as dc


def parse_list(s, conv=str):
    return [conv(v) for v in s.split(",") if v != ""] if s else []


def build_groups(args, heavy_codecs):
    groups = []
    for fam in args.families:
        spec = dc.FAMILIES[fam]
        ops = [o for o in spec["ops"] if not args.ops or o in args.ops]
        for op in ops:
            for L in args.rows or spec["rows"]:
                for B in args.batches:
                    g = dc.cell(fam, op, L, B, args.shrink)
                    g["heavy"] = (fam, op, L, B) in dc.HEAVY
                    g["codecs"] = list(args.codecs)
                    if g["heavy"]:
                        g["codecs"] += [c for c in heavy_codecs if c not in args.codecs]
                    groups.append(g)
    return groups


def arm_instances(group, arms):
    """(arm, codec) pairs one visit runs; codec is None for shared arms."""
    out = []
    if "dense" in arms:
        out.append(("dense", None))
    if "affine" in arms:
        for bits in sorted({dc.affine_bits(c) for c in group["codecs"]}):
            out.append((f"affine{bits}", None))
    for c in group["codecs"]:
        for a in ("kq", "kq_bm64", "kq_bm128", "deq", "kq_aa"):
            if a in arms:
                out.append((a, c))
    return out


def plan_summary(groups, arms, rounds, slots, est_tflops):
    calls = 0
    flops = 0.0
    for g in groups:
        n = len(arm_instances(g, arms))
        calls += n
        flops += n * 2.0 * g["M"] * g["K"] * g["N"]
    per_round = {"calls": calls * slots, "tflop": flops * slots / 1e12}
    # Visit warmup calls are not counted.
    per_round["est_min"] = per_round["tflop"] / est_tflops / 60
    total_min = per_round["est_min"] * (rounds + 1.0 / slots)
    return per_round, total_min


def cell_key(arm, codec):
    return f"{arm}/{codec}" if codec else arm


def paired(samples, a, b):
    """Per-visit ratios a/b over the visits that ran both arms."""
    sb = {v: ms for (v, _r, _s, _t, ms) in samples.get(b, [])}
    return [(r, ms / sb[v]) for (v, r, _s, _t, ms) in samples.get(a, []) if v in sb]


def split_stats(rows, rounds, round_at, value_at):
    """First-round, settled (later rounds) and all-round summaries."""
    first = [p[value_at] for p in rows if p[round_at] == 0]
    settled = [p[value_at] for p in rows if p[round_at] > 0]
    allv = [p[value_at] for p in rows]
    return {
        "first": dc.summarize(first),
        "settled": dc.summarize(settled) if rounds > 1 else None,
        "all": dc.summarize(allv),
    }


def run_pass(dtype, groups, args, store, desc, t_start):
    import mlx.core as mx

    dt = dc.mx_dtype(dtype)
    nax = desc["nax"]
    arch = desc["architecture"]
    kernels = {}
    errors = {}
    samples = [dict() for _ in groups]

    # Warmup pass, discarded; records the kernel each kq arm dispatches.
    for gi, g in enumerate(groups):
        for arm, codec in arm_instances(g, args.arms):
            key = cell_key(arm, codec)
            fn = dc.make_fn(arm, codec, g, dt, store)
            dc.set_tile(dc.ARM_TILE.get(arm), nax)
            try:
                with dc.KernelLog() as log:
                    mx.eval(fn())
                kernels[(gi, key)] = log.kernels[-1] if log.kernels else None
            except ValueError as e:
                errors[(gi, key)] = str(e)
            dc.set_tile(None, nax)
    print(f"[{dtype}] warmup done at {time.perf_counter() - t_start:.0f} s", flush=True)

    visit = 0
    for r in range(args.rounds):
        for s in range(args.slots):
            tm = dc.THUE_MORSE[s % 8]
            order = list(range(len(groups)))
            random.Random(args.seed * 7919 + r * 131 + s).shuffle(order)
            for gi in order:
                g = groups[gi]
                inst = [
                    (a, c)
                    for a, c in arm_instances(g, args.arms)
                    if (gi, cell_key(a, c)) not in errors
                ]
                if tm:
                    inst.reverse()
                for arm, codec in inst[: args.visit_warmup]:
                    fn = dc.make_fn(arm, codec, g, dt, store)
                    dc.set_tile(dc.ARM_TILE.get(arm), nax)
                    dc.time_fn(fn, 1)
                for arm, codec in inst:
                    fn = dc.make_fn(arm, codec, g, dt, store)
                    dc.set_tile(dc.ARM_TILE.get(arm), nax)
                    ms = dc.time_fn(fn, args.inner)
                    t = time.perf_counter() - t_start
                    samples[gi].setdefault(cell_key(arm, codec), []).append(
                        (visit, r, s, t, ms)
                    )
                dc.set_tile(None, nax)
                visit += 1
            print(
                f"[{dtype}] round {r} slot {s} done at "
                f"{time.perf_counter() - t_start:.0f} s",
                flush=True,
            )

    out_groups = []
    for gi, g in enumerate(groups):
        sm = samples[gi]
        M, N, K = g["M"], g["N"], g["K"]

        def arm_record(arm, codec, gi=gi, sm=sm, M=M, N=N, K=K):
            key = cell_key(arm, codec)
            if arm == "dense":
                tile = dc.mlx_dense_tile(M, N, K, arch)
            elif arm.startswith("affine"):
                tile = dc.MLX_AFFINE_TILE
            elif arm == "deq":
                tile = dc.mlx_dense_tile(M, N, K, arch)
            else:
                tile = dc.tile_of(kernels.get((gi, key)))
            rec = {
                "kernel": kernels.get((gi, key)),
                "tile": tile,
                "error": errors.get((gi, key)),
                "samples": [
                    [r, s, round(t, 4), ms] for (_v, r, s, t, ms) in sm.get(key, [])
                ],
            }
            rec.update(split_stats(sm.get(key, []), args.rounds, 1, 4))
            return rec

        shared = {}
        for arm, codec in arm_instances(g, args.arms):
            if codec is None:
                shared[arm] = arm_record(arm, None)
        cells = []
        for codec in g["codecs"]:
            arms = {}
            for arm, c in arm_instances(g, args.arms):
                if c == codec:
                    arms[arm] = arm_record(arm, codec)
            ratios = {}
            aff = f"affine{dc.affine_bits(codec)}"
            for name, a, b in (
                ("kq/dense", cell_key("kq", codec), "dense"),
                ("kq/affine", cell_key("kq", codec), aff),
                ("bm64/kq", cell_key("kq_bm64", codec), cell_key("kq", codec)),
                ("bm128/kq", cell_key("kq_bm128", codec), cell_key("kq", codec)),
                (
                    "bm128/bm64",
                    cell_key("kq_bm128", codec),
                    cell_key("kq_bm64", codec),
                ),
                ("deq/kq", cell_key("deq", codec), cell_key("kq", codec)),
                ("aa/kq", cell_key("kq_aa", codec), cell_key("kq", codec)),
            ):
                pr = paired(sm, a, b)
                if pr:
                    ratios[name] = split_stats(pr, args.rounds, 0, 1)
                    ratios[name]["pairs"] = [[r, v] for r, v in pr]
            cells.append({"codec": codec, "arms": arms, "ratios": ratios})
        out_groups.append(dict(g, shared=shared, cells=cells))
    return out_groups


def aggregates(groups, rounds):
    """Per op and ratio: the pooled median of every settled pair across the
    op's cells, with its CI, beside the spread of per-cell medians and of
    per-cell CI half-widths."""
    by = {}
    for g in groups:
        op = f"{g['family']}.{g['op']}"
        for c in g["cells"]:
            for name, rt in c["ratios"].items():
                e = by.setdefault((op, name), {"pairs": [], "cells": [], "half": []})
                e["pairs"] += [v for r, v in rt["pairs"] if r > 0 or rounds == 1]
                st = rt.get("settled") or rt.get("all")
                if st:
                    e["cells"].append(st["median"])
                    if st["ci95_half"] is not None:
                        e["half"].append(st["ci95_half"])
    out = []
    for (op, name), e in by.items():
        out.append(
            {
                "op": op,
                "ratio": name,
                "pooled": dc.summarize(e["pairs"]),
                "cell_medians": dc.summarize(e["cells"]),
                "cell_ci95_half": dc.summarize(e["half"]),
                "cell_ci95_half_max": max(e["half"]) if e["half"] else None,
                "cells": len(e["cells"]),
            }
        )
    return out


def fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def pick(stats, key="median"):
    if not stats:
        return None
    s = stats.get("settled") or stats.get("all")
    return s[key] if s else None


def pick_first(stats):
    s = (stats or {}).get("first")
    return s["median"] if s else None


def markdown(result):
    lines = ["# DiT shape arms", "", dc.descriptor_md(result["machine"]), ""]
    a = result["args"]
    lines.append(
        f"Rounds {a['rounds']} x {a['slots']} Thue-Morse slots, inner {a['inner']}, "
        f"shrink {a['shrink']}. Times are settled medians in ms (all rounds when "
        "there is one); ratios are medians of per-visit paired ratios; "
        "first = round 0 median."
    )
    lines.append("")
    for p in result["passes"]:
        lines.append(f"## {p['dtype']}")
        lines.append("")
        lines.append(
            "| op | L x B | M | codec | kq tile | kq ms | kq IQR | dense | affine "
            "| bm64 | bm128 | deq |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for g in p["groups"]:
            for c in g["cells"]:
                ar = c["arms"]
                aff = g["shared"].get(f"affine{dc.affine_bits(c['codec'])}")
                kqa = ar.get("kq", {})
                lines.append(
                    f"| {g['family']}.{g['op']} | {g['L']} x {g['B']} | {g['M']} "
                    f"| {c['codec']} | {kqa.get('tile') or '-'} "
                    f"| {fmt(pick(kqa))} | {fmt(pick(kqa, 'rel_iqr'), '.1%')} "
                    f"| {fmt(pick(g['shared'].get('dense')))} | {fmt(pick(aff))} "
                    f"| {fmt(pick(ar.get('kq_bm64')))} "
                    f"| {fmt(pick(ar.get('kq_bm128')))} | {fmt(pick(ar.get('deq')))} |"
                )
        lines.append("")
        names = ["kq/dense", "kq/affine", "bm128/bm64", "deq/kq", "aa/kq"]
        lines.append(
            "| op | L x B | codec | "
            + " | ".join(names)
            + " | first kq/affine | first aa/kq |"
        )
        lines.append("|---|---|---|" + "---|" * (len(names) + 2))
        for g in p["groups"]:
            for c in g["cells"]:
                rt = c["ratios"]
                row = (
                    f"| {g['family']}.{g['op']} | {g['L']} x {g['B']} | {c['codec']} |"
                )
                row += "".join(f" {fmt(pick(rt.get(n)))} |" for n in names)
                row += f" {fmt(pick_first(rt.get('kq/affine')))} |"
                row += f" {fmt(pick_first(rt.get('aa/kq')))} |"
                lines.append(row)
        lines.append("")
        lines.append(
            "Aggregate per op: pooled median of settled pairs with its 95 percent "
            "CI; per-cell medians p10 to p90; per-cell CI half-width median and "
            "max."
        )
        lines.append("")
        lines.append(
            "| op | ratio | cells | pairs | pooled | CI | cell p10 | cell p90 "
            "| cell CI half med | cell CI half max |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for ag in p["aggregate"]:
            po, cm, ch = ag["pooled"], ag["cell_medians"], ag["cell_ci95_half"]
            ci = po["ci95"] if po else None
            lines.append(
                f"| {ag['op']} | {ag['ratio']} | {ag['cells']} "
                f"| {po['n'] if po else 0} | {fmt(po and po['median'], '.4f')} "
                f"| {f'{ci[0]:.4f} to {ci[1]:.4f}' if ci else '-'} "
                f"| {fmt(cm and cm['p10'])} | {fmt(cm and cm['p90'])} "
                f"| {fmt(ch and ch['median'], '.1%')} "
                f"| {fmt(ag['cell_ci95_half_max'], '.1%')} |"
            )
        lines.append("")
        errs = [
            (g, c["codec"], arm, rec["error"])
            for g in p["groups"]
            for c in g["cells"]
            for arm, rec in c["arms"].items()
            if rec["error"]
        ]
        if errs:
            lines.append("Arms not run:")
            lines.append("")
            for g, codec, arm, e in errs:
                lines.append(
                    f"- {g['family']}.{g['op']} {g['L']} x {g['B']} {codec} {arm}: {e}"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--families", default=",".join(dc.FAMILIES))
    ap.add_argument("--ops", default="", help="subset of op names, comma separated")
    ap.add_argument("--rows", default="", help="rows per item; default per family")
    ap.add_argument("--batches", default="1,2")
    ap.add_argument("--codecs", default=",".join(dc.MAIN_CODECS))
    ap.add_argument(
        "--heavy-codecs",
        default="nax",
        help="codecs added at the two heaviest cells: 'nax' (every codec "
        "with NAX kernels), 'none', or a list",
    )
    ap.add_argument(
        "--arms",
        default="dense,kq,kq_bm64,kq_bm128,deq,affine",
        help="from dense,kq,kq_bm64,kq_bm128,deq,affine,kq_aa",
    )
    ap.add_argument("--dtypes", default="bfloat16,float16")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--inner", type=int, default=3, help="calls per sample")
    ap.add_argument(
        "--visit-warmup",
        type=int,
        default=1,
        help="discarded calls at the start of each shape visit",
    )
    ap.add_argument("--shrink", type=int, default=1, help="divide rows, K, N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--est-tflops", type=float, default=25.0, help="rate for the time estimate"
    )
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    args = ap.parse_args()
    args.families = parse_list(args.families)
    args.ops = parse_list(args.ops)
    args.rows = parse_list(args.rows, int)
    args.batches = parse_list(args.batches, int)
    args.codecs = parse_list(args.codecs)
    args.arms = parse_list(args.arms)
    args.dtypes = parse_list(args.dtypes)

    import mlx.core as mx

    import mlx_kquant as kq

    if args.heavy_codecs == "nax":
        heavy_codecs = dc.nax_codecs()
    elif args.heavy_codecs == "none":
        heavy_codecs = []
    else:
        heavy_codecs = parse_list(args.heavy_codecs)
    groups = build_groups(args, heavy_codecs)
    per_round, total_min = plan_summary(
        groups, args.arms, args.rounds, args.slots, args.est_tflops
    )
    cells = sum(len(g["codecs"]) for g in groups)
    print(
        f"{len(groups)} shapes, {cells} codec cells per dtype; per round "
        f"{per_round['calls']} calls, {per_round['tflop']:.1f} TFLOP, "
        f"about {per_round['est_min']:.1f} min at {args.est_tflops} TFLOPS; "
        f"about {total_min * len(args.dtypes):.0f} min for {len(args.dtypes)} "
        f"dtype(s) x {args.rounds} rounds plus warmup",
        flush=True,
    )
    if args.plan_only:
        return

    for k in ("KQ_NAX_TILE", "KQ_QMM_TILE", "KQ_QMM_LOG"):
        os.environ.pop(k, None)
    desc = dc.machine_descriptor()
    store = dc.Store(args.seed)
    t_start = time.perf_counter()
    passes = []
    for dtype in args.dtypes:
        store.clear_dtype()
        mx.clear_cache()
        passes.append(
            {
                "dtype": dtype,
                "groups": run_pass(dtype, groups, args, store, desc, t_start),
            }
        )
        passes[-1]["aggregate"] = aggregates(passes[-1]["groups"], args.rounds)
    result = {
        "machine": desc,
        "args": vars(args),
        "heavy_codecs": heavy_codecs,
        "wall_s": time.perf_counter() - t_start,
        "peak_memory_bytes": mx.get_peak_memory(),
        "mlx_kquant_file": os.path.dirname(kq.__file__),
        "passes": passes,
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f)
    md = markdown(result)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    sys.exit(main())
