"""int8_matmul against a Q4_K quantized_matmul at a video DiT's linear shapes.

Arms per shape (y = x @ W^T, x [M, K] bfloat16, W [N, K]):
  q4k     quantized_matmul(x, w, scales, "q4_k", transpose=True)
  int8    int8_quantize_rows on x, then int8_matmul, in one eval (the
          quantize pass is inside the timed call)
  int8_mm int8_matmul alone on codes and scales made beforehand

The int8 weight is int8_quantize_rows of the float32 dequantized Q4_K
weight, as a caller converting a Q4_K model at load makes it. The shapes
are MiniMax-H3's four block linears; mlp.fc2 runs with one activation scale
per row per 1024 columns.

Timing: mx.synchronize() fences around every sample; the arms alternate in
Thue-Morse order across rounds after a warm-up sample of each. Before the
first cell's timed rounds, its arms run untimed for --warmup-s seconds
(default 45), so every cell is timed at the sustained clock of a loaded
GPU rather than the higher clock of a cold one, at which the int8 arms
gain less on some shapes. Each warm-up sample prints a "warm" line with its
wall-clock start and end too. Each sample
prints one line with its wall-clock start and end (time.time()), so a
wrapper can match them to an external clock record. Reports the median,
the spread (max - min over the median), the relative IQR and the 95
percent CI of the median per arm, and the paired per-round ratio of each
int8 arm's time to q4k's. The relative L2 distance of int8 to q4k over the
first 256 rows is printed as a sanity check (both are lossy).

The MLX buffer cache is capped at --cache-limit-gib (default 6), enough to
hold each arm's output and temporaries at 38010 rows, so every arm reuses
its buffers from call to call; under a smaller cap an arm whose temporaries
push the cache past it allocates its output afresh on each call.

Protocol: idle machine; GPU microbench only (no model load). Needs a GPU
with tensor-op (NAX) units.
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
from bench_stats import summarize

import mlx_kquant as kq

# (K, N, activation group) per linear.
SHAPES = {
    "qkv_proj": (5376, 21504, None),
    "out_proj": (7168, 5376, None),
    "mlp.fc1": (5376, 28672, None),
    "mlp.fc2": (14336, 5376, 1024),
}
ARMS = ["q4k", "int8", "int8_mm"]


def tm_bit(n):
    return bin(n).count("1") & 1


def sample(fn):
    """Seconds of one fenced call, and its wall-clock start and end."""
    mx.synchronize()
    t0 = time.time()
    p0 = time.perf_counter()
    fn()
    mx.synchronize()
    return time.perf_counter() - p0, t0, time.time()


def run_cell(name, rows, rounds, warmup_s=0.0):
    K, N, gs = SHAPES[name]
    mx.clear_cache()
    wf = mx.random.normal((N, K), key=mx.random.key(1)) * 0.02
    wq, ws = kq.quantize(wf, "q4_k")
    del wf
    mx.eval(wq, ws)
    w8, w_scales = kq.int8_quantize_rows(kq.dequantize(wq, ws, "q4_k", mx.float32))
    w_scales = w_scales.reshape(N)
    mx.eval(w8, w_scales)
    x = mx.random.normal((rows, K), key=mx.random.key(2)).astype(mx.bfloat16)
    x8p, sxp = kq.int8_quantize_rows(x, group_size=gs)
    mx.eval(x, x8p, sxp)

    def q4k():
        mx.eval(kq.quantized_matmul(x, wq, ws, "q4_k", transpose=True))

    def int8():
        x8, sx = kq.int8_quantize_rows(x, group_size=gs)
        mx.eval(kq.int8_matmul(x8, sx, w8, w_scales, group_size=gs))

    def int8_mm():
        mx.eval(kq.int8_matmul(x8p, sxp, w8, w_scales, group_size=gs))

    arms = {"q4k": q4k, "int8": int8, "int8_mm": int8_mm}

    ref = kq.quantized_matmul(x[:256], wq, ws, "q4_k", transpose=True)
    got = kq.int8_matmul(x8p[:256], sxp[:256], w8, w_scales, group_size=gs)
    ref, got = ref.astype(mx.float32), got.astype(mx.float32)
    rel = (mx.linalg.norm((got - ref).flatten()) / mx.linalg.norm(ref.flatten())).item()
    del ref, got
    mx.clear_cache()
    mx.reset_peak_memory()

    for nm in ARMS:
        sample(arms[nm])
    t_end = time.time() + warmup_s
    while time.time() < t_end:
        for nm in ARMS:
            dt, t0, t1 = sample(arms[nm])
            print(
                f"warm {name} rows {rows} {nm} {dt * 1e3:.2f} ms "
                f"t0 {t0:.4f} t1 {t1:.4f}",
                flush=True,
            )
    times = {nm: [] for nm in ARMS}
    for rnd in range(rounds):
        order = ARMS[::-1] if tm_bit(rnd) else ARMS
        for nm in order:
            dt, t0, t1 = sample(arms[nm])
            times[nm].append(dt)
            print(
                f"sample {name} rows {rows} round {rnd} {nm} {dt * 1e3:.2f} ms "
                f"t0 {t0:.4f} t1 {t1:.4f}",
                flush=True,
            )
    flop = 2.0 * rows * K * N
    cell = {
        "shape": name,
        "rows": rows,
        "K": K,
        "N": N,
        "group_size": gs,
        "rel_l2_int8_vs_q4k_first_256": rel,
        "peak_active_gib": mx.get_peak_memory() / 2**30,
        "arms": {},
    }
    for nm in ARMS:
        ts = times[nm]
        s = summarize(ts)
        cell["arms"][nm] = {
            **s,
            "samples": ts,
            "spread": (max(ts) - min(ts)) / s["median"],
            "tops": flop / s["median"] / 1e12,
        }
    for nm in ("int8", "int8_mm"):
        ratios = [b / q for b, q in zip(times[nm], times["q4k"], strict=True)]
        cell["arms"][nm]["paired_ratio_vs_q4k"] = {
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
        }
    print(
        f"cell {name} rows {rows}: rel L2 int8 vs q4k {rel:.3e}, "
        f"peak active {cell['peak_active_gib']:.2f} GiB",
        flush=True,
    )
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", type=int, nargs="+", default=[38010, 9472])
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--cache-limit-gib", type=float, default=6.0)
    ap.add_argument(
        "--warmup-s",
        type=float,
        default=45.0,
        help="untimed load before the first cell's timed rounds",
    )
    ap.add_argument("--json-out", default=None, help="write results here")
    a = ap.parse_args()
    if not kq.nax_available():
        raise SystemExit("int8_matmul needs a GPU with tensor-op (NAX) units")

    mx.set_cache_limit(int(a.cache_limit_gib * 2**30))
    res = {
        "mlx_version": mx.__version__,
        "kq_version": kq.__version__,
        "rounds": a.rounds,
        "cache_limit_gib": a.cache_limit_gib,
        "warmup_s": a.warmup_s,
        "cells": [],
    }
    warm = a.warmup_s
    for rows in a.rows:
        for name in a.shapes:
            res["cells"].append(run_cell(name, rows, a.rounds, warm))
            warm = 0.0
            mx.clear_cache()

    print(
        "\n| rows | linear | arm | median ms | spread | rel IQR | TOPS | paired x q4k |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for c in res["cells"]:
        for nm in ARMS:
            s = c["arms"][nm]
            pr = s.get("paired_ratio_vs_q4k")
            prs = (
                "1"
                if pr is None
                else f"{pr['median']:.3f} ({pr['min']:.3f} to {pr['max']:.3f})"
            )
            print(
                f"| {c['rows']} | {c['shape']} | {nm} | {s['median'] * 1e3:.2f} "
                f"| {s['spread'] * 100:.1f}% | {s['rel_iqr'] * 100:.1f}% "
                f"| {s['tops']:.1f} | {prs} |"
            )
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
