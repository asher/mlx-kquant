"""Thermal curve of one DiT cell: a timed loop, then idle recovery.

After --cold-wait seconds idle and one discarded call, the cell loops for
--seconds and every call is recorded. Reported:

  cold          the fastest median of --cold-calls consecutive calls in the
                first --cold-span seconds, which skips the clock ramp of
                the first calls after idle
  first slowdown  seconds until the rolling median of --window calls, after
                the cold window, first exceeds cold by --slowdown (default
                3 percent)
  settled ratio median of the last --tail fraction of calls over cold
  idle gap      after the loop the GPU idles along --idle-schedule
                (cumulative seconds); at each point --probe-calls calls run,
                and the gap is the first point whose fastest median of 3
                consecutive calls is within --recover of cold

These define one thermal round for bench_dit_shapes.py. Markdown and JSON
out; the JSON keeps every call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import dit_common as dc


def cold_window(ts, ms, n, span):
    """(median, end index) of the fastest n consecutive calls that end
    within `span` seconds."""
    best = (float("inf"), min(n, len(ms)))
    for i in range(min(n, len(ms)), len(ms) + 1):
        if ts[i - 1] > span and i > n:
            break
        best = min(best, (statistics.median(ms[i - n : i]), i))
    return best


def rolling_first_over(ts, ms, start, window, limit):
    for i in range(max(window, start + 1), len(ms) + 1):
        if statistics.median(ms[i - window : i]) > limit:
            return ts[i - 1]
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--family", default="qwen_image", choices=list(dc.FAMILIES))
    ap.add_argument("--op", default="gate_up")
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--codec", default="q4_k")
    ap.add_argument("--arm", default="kq", choices=["kq", "dense", "affine", "deq"])
    ap.add_argument("--dtype", default="bfloat16", choices=list(dc.DTYPES))
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--cold-wait", type=float, default=0.0)
    ap.add_argument("--cold-calls", type=int, default=5)
    ap.add_argument("--cold-span", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--slowdown", type=float, default=0.03)
    ap.add_argument("--tail", type=float, default=0.2)
    ap.add_argument("--idle-schedule", default="5,10,20,40,80,160,320")
    ap.add_argument("--probe-calls", type=int, default=8)
    ap.add_argument("--recover", type=float, default=0.01)
    ap.add_argument("--shrink", type=int, default=1)
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    args = ap.parse_args()

    import mlx.core as mx

    g = dc.cell(args.family, args.op, args.rows, args.batch, args.shrink)
    dt = dc.mx_dtype(args.dtype)
    store = dc.Store(0)
    arm = f"affine{dc.affine_bits(args.codec)}" if args.arm == "affine" else args.arm
    fn = dc.make_fn(arm, args.codec, g, dt, store)
    desc = dc.machine_descriptor()
    with dc.KernelLog() as log:
        mx.eval(fn())
    kernel = log.kernels[-1] if log.kernels else None

    if args.cold_wait > 0:
        time.sleep(args.cold_wait)
    dc.time_fn(fn, 1)
    t0 = time.perf_counter()
    ts, ms = [], []
    while time.perf_counter() - t0 < args.seconds:
        v = dc.time_fn(fn, 1)
        ts.append(time.perf_counter() - t0)
        ms.append(v)
    loop_end = time.perf_counter()
    cold, cold_end = cold_window(ts, ms, args.cold_calls, args.cold_span)
    limit = cold * (1 + args.slowdown)
    first = rolling_first_over(ts, ms, cold_end, args.window, limit)
    tail = ms[int(len(ms) * (1 - args.tail)) :] or ms[-1:]
    settled_ratio = statistics.median(tail) / cold

    probes = []
    gap = None
    for point in (float(v) for v in args.idle_schedule.split(",") if v):
        wait = loop_end + point - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        pms = [dc.time_fn(fn, 1) for _ in range(args.probe_calls)]
        pm = cold_window(list(range(len(pms))), pms, 3, float("inf"))[0]
        probes.append({"idle_s": point, "ms": pm, "ratio": pm / cold, "calls": pms})
        # Probes add a little heat, so the clock restarts after each one.
        loop_end = time.perf_counter() - point
        if pm <= cold * (1 + args.recover):
            gap = point
            break

    result = {
        "machine": desc,
        "args": vars(args),
        "cell": g,
        "arm": arm,
        "kernel": kernel,
        "tile": dc.tile_of(kernel),
        "calls": len(ms),
        "cold_ms": cold,
        "cold_window_end_s": ts[cold_end - 1],
        "first_slowdown_s": first,
        "settled_ratio": settled_ratio,
        "idle_gap_s": gap,
        "probes": probes,
        "series": [[round(t, 4), m] for t, m in zip(ts, ms, strict=True)],
    }
    lines = ["# DiT thermal curve", "", dc.descriptor_md(desc), ""]
    lines.append(
        f"Cell {g['family']}.{g['op']} {g['L']} x {g['B']} (M {g['M']}, K {g['K']}, "
        f"N {g['N']}), {arm} {args.codec if arm.startswith(('kq', 'deq')) else ''} "
        f"{args.dtype}, tile {result['tile'] or '-'}, {len(ms)} calls in "
        f"{args.seconds:.0f} s."
    )
    lines.append("")
    lines.append("| cold ms | first slowdown s | settled ratio | idle gap s |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| {cold:.3f} | {'-' if first is None else f'{first:.1f}'} "
        f"| {settled_ratio:.3f} | {'-' if gap is None else f'{gap:g}'} |"
    )
    lines.append("")
    lines.append("| t s (bin end) | median ms | over cold |")
    lines.append("|---|---|---|")
    step = max(args.seconds / 24, 1e-9)
    edge, bucket = step, []
    for t, m in zip(ts, ms, strict=True):
        if t > edge and bucket:
            med = statistics.median(bucket)
            lines.append(f"| {edge:.1f} | {med:.3f} | {med / cold:.3f} |")
            bucket = []
            while t > edge:
                edge += step
        bucket.append(m)
    if bucket:
        med = statistics.median(bucket)
        lines.append(f"| {ts[-1]:.1f} | {med:.3f} | {med / cold:.3f} |")
    lines.append("")
    lines.append("| idle s | probe ms | over cold |")
    lines.append("|---|---|---|")
    for p in probes:
        lines.append(f"| {p['idle_s']:g} | {p['ms']:.3f} | {p['ratio']:.3f} |")
    md = "\n".join(lines)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    sys.exit(main())
