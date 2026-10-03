"""Quantized matmul arms at DiT linear shapes, paired in one process.

Cells are (op, rows per item L, batch B, codec). A batch of two runs as
[2, L, K], which the op flattens to 2L rows, so the tile policy can pick a
different tile at B 2 than at B 1. The modulation linears run at one row
per item, where the tile arms do not apply. Arms, switched by live levers
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
a fresh interleaved order. A visit opens with a discarded sample of the
first --visit-warmup arms, since the first call after a shape switch runs
slow; --visit-warmup-calls sets its call count. Inside a visit
the arm list reverses on odd slots, so the two calls of every pair swap
order with the Thue-Morse bit and a round of 8 slots splits each pair 4/4.
A paired ratio compares calls made seconds apart. Round 0 gets its own
column and later rounds the settled column. Round 0 is not a cold figure:
the GPU slows within seconds of load, and bench_dit_thermal.py gives the
cold rate per machine.

A sample is a chain of --inner calls in one evaluation, divided by the
count. Each call is fed a tie of the previous output (the activation
again, selected on the output's first element), so the GPU runs the calls
one after another, as in a model forward; calls evaluated together without
the tie overlap, which slows some kernels and speeds others. Every visit
also times the tie chain alone ("glue"), and each sample of the visit has
it subtracted. Shapes whose fastest arm runs under --min-sample-ms per call
take more calls, up to --max-inner, and every arm of a shape uses the same
count. Reports and raw logs carry "sample_mode": "chain"; a log without it
is from the earlier independent-call bench, and its figures do not compare
with chained ones.

Per cell, ratios carry a 95 percent CI of the paired median, from order
statistics; its half-width is the cell's noise floor. Per op, the pooled
median of all settled pairs is the aggregate, reported beside the spread
of the per-cell medians. Markdown and JSON out; the JSON keeps every
sample and pair.

Every sample also goes to a JSON Lines raw log (--raw-out, by default
beside --json-out) as each slot ends, so a run that stops or fails keeps
every finished slot. --reaggregate rebuilds the JSON and Markdown from a
raw log without running anything, and marks a partial run. --resume
continues a stopped run from its raw log: it keeps the finished slots and
the warmup's call counts, and runs the slots left, in the same order.

Memory: MLX keeps freed buffers in its cache for reuse, and every shape
leaves its own sizes there, so a sweep of many shapes would otherwise hold
the outputs of all of them. --cache-limit-gib bounds the cache, and the
visit warmup runs a full sample by default so the timed calls reuse
cached buffers. Each slot record carries the active, cache
and process footprint bytes after every visit; --mem-trace prints them
as they are taken.

A versus A: --arms kq,kq_aa. Plan only: --plan-only.

Lever arms: --env-arm NAME:VAR=VALUE[,VAR=VALUE] defines a kq arm that
runs with those environment levers set (read live), adds it to --arms and
pairs it with kq as NAME/kq, for example --env-arm sw1:KQ_NAX_SWIZZLE=1.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import math
import os
import random
import sys
import time

import bench_stats as bs
import dit_common as dc


def parse_list(s, conv=str):
    return [conv(v) for v in s.split(",") if v != ""] if s else []


def build_groups(args, heavy_codecs):
    groups = []
    for fam in args.families:
        spec = dc.FAMILIES[fam]
        ops = [o for o in spec["ops"] if not args.ops or o in args.ops]
        for op in ops:
            for L in dc.op_rows(fam, op, args.rows):
                for B in args.batches:
                    g = dc.cell(fam, op, L, B, args.shrink)
                    g["heavy"] = (fam, op, L, B) in dc.HEAVY
                    g["codecs"] = list(args.codecs)
                    if g["heavy"]:
                        g["codecs"] += [c for c in heavy_codecs if c not in args.codecs]
                    groups.append(g)
    return groups


CODEC_ARMS = ("kq", "kq_bm64", "kq_bm128", "deq", "kq_aa")
SHARED_ARMS = ("dense", "affine")


def lever_arms(arms):
    """Arms defined by --env-arm, in --arms order."""
    return [a for a in arms if a not in CODEC_ARMS and a not in SHARED_ARMS]


def parse_env_arms(specs):
    """{name: {VAR: VALUE}} from --env-arm NAME:VAR=VALUE[,VAR=VALUE]."""
    out = {}
    for spec in specs:
        name, sep, body = spec.partition(":")
        env = dict(kv.split("=", 1) for kv in body.split(",") if kv)
        if not sep or not name or not env or name in CODEC_ARMS + SHARED_ARMS:
            raise SystemExit(f"--env-arm {spec!r}: want NAME:VAR=VALUE[,VAR=VALUE]")
        out[name] = env
    return out


def set_arm_env(arm, env_arms):
    """Sets the levers of a lever arm and clears those of the others."""
    for env in env_arms.values():
        for k in env:
            os.environ.pop(k, None)
    os.environ.update(env_arms.get(arm, {}))


def arm_instances(group, arms):
    """(arm, codec) pairs one visit runs; codec is None for shared arms."""
    out = []
    if "dense" in arms:
        out.append(("dense", None))
    if "affine" in arms:
        for bits in sorted({dc.affine_bits(c) for c in group["codecs"]}):
            out.append((f"affine{bits}", None))
    for c in group["codecs"]:
        for a in CODEC_ARMS + tuple(lever_arms(arms)):
            # Tile and lever arms do not apply at one row per item, where the
            # mat-vec routes run.
            forced = a in dc.ARM_TILE_FORCED or a not in CODEC_ARMS
            if a in arms and not (group["vector"] and forced):
                out.append((a, c))
    return out


def plan_summary(groups, arms, rounds, slots, est_tflops, inner):
    calls = 0
    flops = 0.0
    for g in groups:
        n = len(arm_instances(g, arms)) * inner
        calls += n
        flops += n * 2.0 * g["M"] * g["K"] * g["N"]
    per_round = {"calls": calls * slots, "tflop": flops * slots / 1e12}
    # Visit warmup calls and the extra calls of fast shapes are not counted.
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
    """Round 0, settled (later rounds) and all-round summaries."""
    round0 = [p[value_at] for p in rows if p[round_at] == 0]
    settled = [p[value_at] for p in rows if p[round_at] > 0]
    allv = [p[value_at] for p in rows]
    return {
        "round0": bs.summarize(round0),
        "settled": bs.summarize(settled) if rounds > 1 else None,
        "all": bs.summarize(allv),
    }


_LIBC = None


def footprint_bytes():
    """The process physical footprint on macOS, which memory limits
    count, cache included; None elsewhere."""
    global _LIBC
    if sys.platform != "darwin":
        return None
    if _LIBC is None:
        _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
    buf = ctypes.create_string_buffer(512)
    # RUSAGE_INFO_V4; ri_phys_footprint is at byte 72.
    if _LIBC.proc_pid_rusage(os.getpid(), 4, buf) != 0:
        return None
    return int.from_bytes(buf.raw[72:80], "little")


def mem_row(mx, gi, trace=False):
    row = [gi, mx.get_active_memory(), mx.get_cache_memory(), footprint_bytes()]
    if trace:
        gib = [None if v is None else round(v / 2**30, 3) for v in row[1:]]
        print("mem", gi, *gib, flush=True)
    return row


def complete_lines(path):
    """Drops a last line cut short, so records can be appended."""
    with open(path) as f:
        text = f.read()
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
        with open(path, "w") as f:
            f.write(text)


class RawLog:
    """JSON Lines log of a run: a meta record, then per dtype a warmup
    record and one record per finished slot, then an end record. Each
    record is flushed and synced as it is written."""

    def __init__(self, path, mode="w"):
        self.f = open(path, mode) if path else None

    def write(self, rec):
        if self.f is None:
            return
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


def load_raw(path):
    """Meta record, per-dtype measurements and end record (empty when the
    run stopped early) from a raw log. A last line cut short is skipped."""
    with open(path) as f:
        lines = [line for line in f.read().split("\n") if line.strip()]
    recs = []
    for i, line in enumerate(lines):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
    if not recs or recs[0].get("kind") != "meta":
        raise ValueError(f"{path}: no meta record")
    meta, measured, end = recs[0], {}, {}
    ngroups = len(meta["groups"])
    for rec in recs[1:]:
        if rec["kind"] == "warmup":
            measured[rec["dtype"]] = {
                "inner": rec["inner"],
                "kernels": {(gi, k): v for gi, k, v in rec["kernels"]},
                "errors": {(gi, k): v for gi, k, v in rec["errors"]},
                "samples": [dict() for _ in range(ngroups)],
                "slots": 0,
            }
        elif rec["kind"] == "slot":
            m = measured[rec["dtype"]]
            for gi, key, visit, r, s, t, ms in rec["samples"]:
                m["samples"][gi].setdefault(key, []).append((visit, r, s, t, ms))
            m["slots"] += 1
        elif rec["kind"] == "end":
            end = rec
    return meta, measured, end


def measure_pass(dtype, groups, args, store, desc, t_start, raw, prior=None):
    """Warmup and timed rounds for one dtype. Samples go to `raw` as each
    slot ends; no statistics here. `prior` is the part of the pass a raw
    log holds: its slots are skipped, and its warmup's call counts and
    kernels are kept."""
    import mlx.core as mx

    dt = dc.mx_dtype(dtype)
    nax = desc["nax"]
    env_arms = parse_env_arms(args.env_arm)
    kernels = {}
    errors = {}
    calib = {}
    samples = [dict() for _ in groups]
    mem = []

    # Warmup pass, discarded; records the kernel each kq arm dispatches.
    for gi, g in enumerate(groups):
        for arm, codec in arm_instances(g, args.arms):
            key = cell_key(arm, codec)
            x0, f = dc.make_call(arm, codec, g, dt, store)
            dc.set_tile(dc.ARM_TILE.get(arm), nax)
            set_arm_env(arm, env_arms)
            try:
                with dc.KernelLog() as log:
                    mx.eval(f(x0))
                kernels[(gi, key)] = log.kernels[-1] if log.kernels else None
                calib.setdefault(gi, []).append(dc.chain_ms(f, x0, 1))
            except ValueError as e:
                errors[(gi, key)] = str(e)
            dc.set_tile(None, nax)
            set_arm_env(None, env_arms)
        mem.append(mem_row(mx, gi, args.mem_trace))
    inner = []
    for gi in range(len(groups)):
        fastest = min(calib.get(gi, [args.min_sample_ms]))
        n = math.ceil(args.min_sample_ms / max(fastest, 1e-3))
        inner.append(min(args.max_inner, max(args.inner, n)))
    done = set()
    visit = 0
    slots = 0
    if prior is None:
        raw.write(
            {
                "kind": "warmup",
                "dtype": dtype,
                "inner": inner,
                "kernels": [[gi, k, v] for (gi, k), v in kernels.items()],
                "errors": [[gi, k, v] for (gi, k), v in errors.items()],
                "mem": mem,
            }
        )
    else:
        inner = prior["inner"]
        kernels = prior["kernels"]
        errors = prior["errors"]
        samples = prior["samples"]
        slots = prior["slots"]
        for sm in samples:
            for rows in sm.values():
                for v, r, s, _, _ in rows:
                    done.add((r, s))
                    visit = max(visit, v + 1)
    print(f"[{dtype}] warmup done at {time.perf_counter() - t_start:.0f} s", flush=True)

    for r in range(args.rounds):
        for s in range(args.slots):
            if (r, s) in done:
                continue
            rows = []
            mem = []
            tm = dc.THUE_MORSE[s % 8]
            warm = args.visit_warmup_calls[s % len(args.visit_warmup_calls)]
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
                wcalls = inner[gi] if warm == "full" else int(warm)
                for arm, codec in inst[: args.visit_warmup]:
                    x0, f = dc.make_call(arm, codec, g, dt, store)
                    dc.set_tile(dc.ARM_TILE.get(arm), nax)
                    set_arm_env(arm, env_arms)
                    dc.chain_ms(f, x0, wcalls)
                x0 = store.act(g, dt)
                glue = dc.chain_ms(lambda a: a, x0, inner[gi])
                t = round(time.perf_counter() - t_start, 4)
                samples[gi].setdefault("glue", []).append((visit, r, s, t, glue))
                rows.append([gi, "glue", visit, r, s, t, glue])
                for arm, codec in inst:
                    x0, f = dc.make_call(arm, codec, g, dt, store)
                    dc.set_tile(dc.ARM_TILE.get(arm), nax)
                    set_arm_env(arm, env_arms)
                    ms = dc.chain_ms(f, x0, inner[gi]) - glue
                    t = round(time.perf_counter() - t_start, 4)
                    key = cell_key(arm, codec)
                    samples[gi].setdefault(key, []).append((visit, r, s, t, ms))
                    rows.append([gi, key, visit, r, s, t, ms])
                dc.set_tile(None, nax)
                set_arm_env(None, env_arms)
                mem.append(mem_row(mx, gi, args.mem_trace))
                visit += 1
            raw.write(
                {
                    "kind": "slot",
                    "dtype": dtype,
                    "round": r,
                    "slot": s,
                    "visit_warmup_calls": warm,
                    "samples": rows,
                    "mem": mem,
                }
            )
            slots += 1
            fp = [m[3] for m in mem if m[3] is not None]
            fp = f", footprint up to {max(fp) / 2**30:.1f} GiB" if fp else ""
            print(
                f"[{dtype}] round {r} slot {s} done at "
                f"{time.perf_counter() - t_start:.0f} s{fp}",
                flush=True,
            )
    return {
        "inner": inner,
        "kernels": kernels,
        "errors": errors,
        "samples": samples,
        "slots": slots,
    }


def summarize_pass(groups, m, rounds, arm_names, desc):
    """Per-cell records and paired ratios from one dtype's measurements."""
    nax = desc["nax"]
    arch = desc["architecture"]
    kernels, errors, inner = m["kernels"], m["errors"], m["inner"]
    out_groups = []
    for gi, g in enumerate(groups):
        sm = m["samples"][gi]
        M, N, K = g["M"], g["N"], g["K"]

        def arm_record(arm, codec, gi=gi, sm=sm, M=M, N=N, K=K):
            key = cell_key(arm, codec)
            if arm == "dense":
                tile = dc.mlx_dense_tile(M, N, K, arch, nax)
            elif arm.startswith("affine"):
                tile = dc.mlx_affine_tile(nax)
            elif arm == "deq":
                tile = dc.mlx_dense_tile(M, N, K, arch, nax)
            elif arm == "glue":
                tile = None
            else:
                tile = dc.tile_of(kernels.get((gi, key)))
            rec = {
                "kernel": kernels.get((gi, key)),
                "tile": tile,
                "error": errors.get((gi, key)),
                "samples": [[r, s, t, ms] for (_v, r, s, t, ms) in sm.get(key, [])],
            }
            rec.update(split_stats(sm.get(key, []), rounds, 1, 4))
            return rec

        shared = {}
        for arm, codec in arm_instances(g, arm_names):
            if codec is None:
                shared[arm] = arm_record(arm, None)
        if "glue" in sm:
            shared["glue"] = arm_record("glue", None)
        cells = []
        for codec in g["codecs"]:
            arms = {}
            for arm, c in arm_instances(g, arm_names):
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
                *(
                    (f"{a}/kq", cell_key(a, codec), cell_key("kq", codec))
                    for a in lever_arms(arm_names)
                ),
            ):
                pr = paired(sm, a, b)
                if pr:
                    ratios[name] = split_stats(pr, rounds, 0, 1)
                    ratios[name]["pairs"] = [[r, v] for r, v in pr]
            cells.append({"codec": codec, "arms": arms, "ratios": ratios})
        out_groups.append(dict(g, inner=inner[gi], shared=shared, cells=cells))
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
                "pooled": bs.summarize(e["pairs"]),
                "cell_medians": bs.summarize(e["cells"]),
                "cell_ci95_half": bs.summarize(e["half"]),
                "cell_ci95_half_max": max(e["half"]) if e["half"] else None,
                "cells": len(e["cells"]),
            }
        )
    return out


def build_result(meta, measured, end, raw_path):
    """The report from a meta record and per-dtype measurements, live or
    read back from a raw log."""
    a = meta["args"]
    groups = meta["groups"]
    planned = a["rounds"] * a["slots"]
    passes = []
    last_t = None
    for dtype, m in measured.items():
        pg = summarize_pass(groups, m, a["rounds"], a["arms"], meta["machine"])
        passes.append(
            {
                "dtype": dtype,
                "slots_done": m["slots"],
                "slots_planned": planned,
                "groups": pg,
                "aggregate": aggregates(pg, a["rounds"]),
            }
        )
        for sm in m["samples"]:
            for rows in sm.values():
                for row in rows:
                    last_t = row[3] if last_t is None else max(last_t, row[3])
    return {
        "machine": meta["machine"],
        "args": a,
        "heavy_codecs": meta["heavy_codecs"],
        "wall_s": end.get("wall_s", last_t),
        "peak_memory_bytes": end.get("peak_memory_bytes"),
        "mlx_kquant_file": meta["mlx_kquant_file"],
        "raw": raw_path,
        "sample_mode": meta.get("sample_mode", "independent"),
        "complete": bool(end),
        "passes": passes,
    }


def write_outputs(result, json_out, md_out):
    if json_out:
        with open(json_out, "w") as f:
            json.dump(result, f)
    md = markdown(result)
    if md_out:
        with open(md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)


def fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def pick(stats, key="median"):
    if not stats:
        return None
    s = stats.get("settled") or stats.get("all")
    return s[key] if s else None


def pick_round0(stats):
    s = (stats or {}).get("round0")
    return s["median"] if s else None


def markdown(result):
    lines = ["# DiT shape arms", "", dc.descriptor_md(result["machine"]), ""]
    a = result["args"]
    if result.get("sample_mode", "independent") == "chain":
        lines.append(
            "Samples are dependent call chains with the tie chain (glue) subtracted."
        )
    else:
        lines.append(
            "Samples are independent calls evaluated together (earlier "
            "bench): do not compare with chained runs."
        )
    lines.append("")
    lines.append(
        f"Rounds {a['rounds']} x {a['slots']} Thue-Morse slots, inner {a['inner']} "
        f"(up to {a['max_inner']} under {a['min_sample_ms']} ms per call), "
        f"shrink {a['shrink']}. Times are settled medians in ms (all rounds when "
        "there is one); ratios are medians of per-visit paired ratios; "
        "round 0 = median of round 0, which is not a cold figure (see "
        "bench_dit_thermal.py for the cold rate)."
    )
    lines.append("")
    if not result.get("complete", True):
        lines.append(
            "Partial run: the raw log has no end record, so the figures cover "
            "the slots it holds."
        )
        lines.append("")
    for p in result["passes"]:
        lines.append(f"## {p['dtype']}")
        lines.append("")
        if p.get("slots_done", 0) < p.get("slots_planned", 0):
            lines.append(f"{p['slots_done']} of {p['slots_planned']} slots ran.")
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
        names += [f"{arm}/kq" for arm in lever_arms(a["arms"])]
        lines.append(
            "| op | L x B | codec | "
            + " | ".join(names)
            + " | round 0 kq/affine | round 0 aa/kq |"
        )
        lines.append("|---|---|---|" + "---|" * (len(names) + 2))
        for g in p["groups"]:
            for c in g["cells"]:
                rt = c["ratios"]
                row = (
                    f"| {g['family']}.{g['op']} | {g['L']} x {g['B']} | {c['codec']} |"
                )
                row += "".join(f" {fmt(pick(rt.get(n)))} |" for n in names)
                row += f" {fmt(pick_round0(rt.get('kq/affine')))} |"
                row += f" {fmt(pick_round0(rt.get('aa/kq')))} |"
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--families", default=",".join(dc.FAMILIES))
    ap.add_argument("--ops", default="", help="subset of op names, comma separated")
    ap.add_argument(
        "--rows",
        default="",
        help="rows per item; default per family (modulation ops keep one row)",
    )
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
        "--min-sample-ms",
        type=float,
        default=10.0,
        help="shapes faster than this per call take more calls per sample",
    )
    ap.add_argument("--max-inner", type=int, default=64)
    ap.add_argument(
        "--visit-warmup",
        type=int,
        default=1,
        help="arms that run a discarded sample at the start of each shape visit",
    )
    ap.add_argument(
        "--env-arm",
        action="append",
        default=[],
        metavar="NAME:VAR=VALUE[,VAR=VALUE]",
        help="a kq arm run with these environment levers, paired with kq as "
        "NAME/kq (repeatable)",
    )
    ap.add_argument(
        "--visit-warmup-calls",
        default="full",
        help="calls in that discarded sample: 'full' (the sample's count) or "
        "a number; a comma list cycles over the slots (A versus B of the "
        "warmup in one process)",
    )
    ap.add_argument("--shrink", type=int, default=1, help="divide rows, K, N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--est-tflops", type=float, default=25.0, help="rate for the time estimate"
    )
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    ap.add_argument(
        "--raw-out",
        help="JSON Lines log of every sample, written as each slot ends "
        "(default: --json-out plus .raw.jsonl)",
    )
    ap.add_argument(
        "--cache-limit-gib",
        type=float,
        default=8.0,
        help="MLX buffer cache limit; it must hold the outputs of one sample "
        "of the largest shape, or the timed calls allocate. Negative keeps "
        "the MLX default",
    )
    ap.add_argument(
        "--mem-trace",
        action="store_true",
        help="print active, cache and footprint GiB after every visit",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue the run in the raw log (same arguments): keep its "
        "finished slots and run the rest",
    )
    ap.add_argument(
        "--reaggregate",
        metavar="RAW",
        help="rebuild the JSON and Markdown from a raw log without running "
        "anything; only --json-out and --md-out apply",
    )
    args = ap.parse_args(argv)
    if args.reaggregate:
        meta, measured, end = load_raw(args.reaggregate)
        result = build_result(meta, measured, end, args.reaggregate)
        write_outputs(result, args.json_out, args.md_out)
        return 0
    args.families = parse_list(args.families)
    args.ops = parse_list(args.ops)
    args.rows = parse_list(args.rows, int)
    args.batches = parse_list(args.batches, int)
    args.codecs = parse_list(args.codecs)
    args.arms = parse_list(args.arms)
    for name in parse_env_arms(args.env_arm):
        if name not in args.arms:
            args.arms.append(name)
    unknown = set(lever_arms(args.arms)) - set(parse_env_arms(args.env_arm))
    if unknown:
        ap.error(f"arms without a definition: {sorted(unknown)}")
    args.dtypes = parse_list(args.dtypes)
    args.visit_warmup_calls = parse_list(args.visit_warmup_calls)

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
        groups, args.arms, args.rounds, args.slots, args.est_tflops, args.inner
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
    if args.cache_limit_gib >= 0:
        mx.set_cache_limit(int(args.cache_limit_gib * 2**30))
    desc = dc.machine_descriptor()
    store = dc.Store(args.seed)
    raw_out = args.raw_out
    if raw_out is None and args.json_out:
        raw_out = args.json_out + ".raw.jsonl"
    meta = {
        "kind": "meta",
        "machine": desc,
        "args": vars(args),
        "heavy_codecs": heavy_codecs,
        "groups": groups,
        "mlx_kquant_file": os.path.dirname(kq.__file__),
        "sample_mode": "chain",
    }
    prior = {}
    if args.resume and raw_out and os.path.exists(raw_out):
        complete_lines(raw_out)
        meta0, prior, end0 = load_raw(raw_out)
        if (
            meta0.get("sample_mode") != "chain"
            or meta0["groups"] != groups
            or any(
                meta0["args"].get(k) != getattr(args, k)
                for k in (
                    "arms",
                    "env_arm",
                    "dtypes",
                    "rounds",
                    "slots",
                    "seed",
                    "shrink",
                )
            )
        ):
            sys.exit(f"{raw_out}: another run's raw log; --resume needs the same plan")
        if end0:
            result = build_result(meta0, prior, end0, raw_out)
            write_outputs(result, args.json_out, args.md_out)
            return 0
        meta = meta0
        raw = RawLog(raw_out, "a")
    else:
        raw = RawLog(raw_out)
        raw.write(meta)
    t_start = time.perf_counter()
    measured = {}
    for dtype in args.dtypes:
        store.clear_dtype()
        gc.collect()
        mx.clear_cache()
        p = prior.get(dtype)
        if p is not None and p["slots"] == args.rounds * args.slots:
            measured[dtype] = p
            continue
        measured[dtype] = measure_pass(
            dtype, groups, args, store, desc, t_start, raw, p
        )
    end = {
        "kind": "end",
        "wall_s": time.perf_counter() - t_start,
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    raw.write(end)
    raw.close()
    try:
        result = build_result(meta, measured, end, raw_out)
        write_outputs(result, args.json_out, args.md_out)
    except Exception:
        if raw_out:
            print(
                f"summaries failed; the samples are in {raw_out}, and "
                f"--reaggregate {raw_out} rebuilds the reports",
                file=sys.stderr,
                flush=True,
            )
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
