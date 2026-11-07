"""Timing of the rows LoRA epilogue by route, one process.

Source of kq_lora_split_chunks in src/kquant_lora_epilogue.cpp and the
kq_lora_rows_z / kq_lora_rows_apply kernels.

Arms, through KQ_LORA_SPLIT, which the op reads live per call:
  base     the op with no adapter.
  default  the route policy (unset lever).
  split    the split route (z before the base dispatch, apply after it),
           forced with KQ_LORA_SPLIT=1 at every rank. Rows above 32 have no
           split route.
  one      the one-dispatch epilogue after the base route, KQ_LORA_SPLIT=0.
  plain    the base op, then the delta as plain MLX ops: x @ A and an addmm
           with B (dense), or two gather_mm calls and an add (gather). This
           is the path a caller takes when it leaves the epilogue off.
Reported times are the extra microseconds per call over base.

To compare with an earlier build, run the script again with that build
first on PYTHONPATH. Builds up to 0.5.9 read KQ_LORA_SPLIT=1 as unset and
took the split route at 32 rows or fewer at every rank, so there the split
and default arms both time the earlier split kernels.

Paths: `gather` is gather_qmv_kq with per-expert LoRA tables (the MoE down
projection; rows are tokens x routed slots), `dense` is quantized_matmul with
one adapter (rows are tokens), `mix` is gather_qmv_mix_ns_kq (its own z and
apply kernels; the lever does not change it, so it reports base and default).
The default geometry is the qwen35moe down projection: expert and
shared-expert width K 512, hidden N 2048, 256 experts.

Weights are real codec output: --seed-rows float rows are quantized with
kq.quantize and tiled to N rows and to the expert stack. A sample is a chain
of --calls calls in one eval, each call's input tied to the previous output
through one small elementwise op, so the time per call is latency, as in
decode. The `glue` arm times the tie alone; reported times subtract base,
which carries the same glue. Arm order reverses on odd rounds. Before timing,
the delta of every arm, each evaluated under its own lever, is compared with
an f32 reference, to one bf16 step of the output plus 2 percent of the
largest delta.

Cells: path x codec x rank x rows. Markdown + JSON out.
"""

import argparse
import json
import math
import os
import statistics
import time

ENV = "KQ_LORA_SPLIT"
os.environ.pop(ENV, None)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import mlx_kquant as kq  # noqa: E402
from mlx_kquant.codec_geometry import CODEC_GEOMETRY  # noqa: E402

ARMS = ["base", "default", "split", "one", "plain"]
LEVER = {"default": None, "split": "1", "one": "0"}
SPLIT_MAX_ROWS = 32


def set_lever(arm):
    v = LEVER.get(arm)
    if v is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = v


def quantized_rows(codec, N, K, seed_rows, rng):
    wpb = CODEC_GEOMETRY[codec][3]
    if K % wpb:
        raise ValueError(f"K {K} is not a multiple of the {wpb}-weight block")
    wf = mx.array((rng.standard_normal((seed_rows, K)) * 0.05).astype(np.float32))
    rows, _ = kq.quantize(wf, codec)
    return mx.contiguous(mx.tile(rows, (-(-N // seed_rows), 1))[:N])


def layout(rows):
    s = min(rows, 8)
    return rows // s, s


class Cell:
    def __init__(self, path, codec, rank, rows, w, E, args, rng):
        self.path, self.codec, self.w, self.rank = path, codec, w, rank
        dtype = mx.bfloat16
        K, N = args.k, args.n
        if path == "dense":
            self.x = (mx.random.normal((rows, K)) * 0.5).astype(dtype)
            self.scales = mx.zeros((1,), dtype=dtype)
            self.lora = dict(
                lora_a=(mx.random.normal((K, rank)) * 0.02).astype(dtype),
                lora_b=(mx.random.normal((rank, N)) * 0.02).astype(dtype),
                lora_rows=mx.ones((rows,), mx.float32),
            )
        else:
            t, s = layout(rows)
            self.x = (mx.random.normal((t, s, K)) * 0.5).astype(dtype)
            self.idx = mx.array(
                np.stack([rng.choice(E, s, replace=False) for _ in range(t)]).astype(
                    np.uint32
                )
            )
            self.sc = mx.full((t, s), 1.0 / s, dtype=mx.float32)
            self.lora = dict(
                lora_a=(mx.random.normal((E, K, rank)) * 0.02).astype(dtype),
                lora_b=(mx.random.normal((E, rank, N)) * 0.02).astype(dtype),
                lora_rows=mx.ones((t, s), mx.float32),
            )
        if path != "dense":
            t, s = self.idx.shape
            self.lhs = mx.arange(t * s, dtype=mx.uint32)
            self.rhs = self.idx.reshape(-1)
        mx.eval(self.x, self.lora, *([self.idx, self.sc] if path != "dense" else []))

    def plain(self, x):
        a, b = self.lora["lora_a"], self.lora["lora_b"]
        if self.path == "dense":
            y = kq.quantized_matmul(x, self.w, self.scales, self.codec)
            return mx.addmm(y, x @ a, b)
        y = kq.gather_qmv_kq(x, self.w, self.codec, self.idx)
        t, s, K = x.shape
        xr = x.reshape(t * s, 1, K)
        z = mx.gather_mm(xr, a, self.lhs, self.rhs)
        d = mx.gather_mm(z, b, self.lhs, self.rhs)
        return y + d.reshape(y.shape)

    def ref_delta(self):
        f = mx.float32
        x = self.x.astype(f)
        a, b = self.lora["lora_a"].astype(f), self.lora["lora_b"].astype(f)
        if self.path == "dense":
            return (x @ a) @ b
        t, s, K = x.shape
        z = mx.gather_mm(x.reshape(t * s, 1, K), a, self.lhs, self.rhs)
        d = mx.gather_mm(z, b, self.lhs, self.rhs).reshape(t, s, -1)
        if self.path == "gather":
            return d
        return (self.sc[..., None] * d).sum(1)

    def call(self, x, adapted):
        if adapted == "plain":
            return self.plain(x)
        kw = self.lora if adapted else {}
        if self.path == "dense":
            return kq.quantized_matmul(x, self.w, self.scales, self.codec, **kw)
        if self.path == "gather":
            return kq.gather_qmv_kq(x, self.w, self.codec, self.idx, **kw)
        return kq.gather_qmv_mix_ns_kq(x, self.w, self.codec, self.idx, self.sc, **kw)

    def chain(self, calls, arm):
        x = self.x
        tie = None
        for _ in range(calls):
            xx = x if tie is None else x + (tie * 0).astype(x.dtype)
            if arm == "glue":
                y = xx
            else:
                y = self.call(xx, "plain" if arm == "plain" else arm != "base")
            tie = y.reshape(-1)[:1]
        return tie


def bench_cell(cell, rows, args):
    arms = ["base", "default"]
    if cell.path != "mix":
        arms += ["split", "one"] if rows <= SPLIT_MAX_ROWS else ["one"]
        arms += ["plain"]
    set_lever("default")
    base = cell.call(cell.x, False).astype(mx.float32)
    mx.eval(base)
    outs = {}
    for arm in arms[1:]:
        # The op reads the lever when the graph runs.
        set_lever(arm)
        adapted = "plain" if arm == "plain" else True
        outs[arm] = cell.call(cell.x, adapted).astype(mx.float32) - base
        mx.eval(outs[arm])
    ref = cell.ref_delta().reshape(base.shape)
    scale = mx.abs(ref).max().item()
    step = 2.0 ** (math.floor(math.log2(mx.abs(base).max().item() + 1e-30)) - 7)
    for arm, d in outs.items():
        err = mx.abs(d - ref).max().item()
        if err > step + 2e-2 * scale:
            where = f"{cell.path} {cell.codec} rank {cell.rank} rows {rows}"
            raise SystemExit(f"{where}: {arm} off by {err} (step {step})")
    order = ["glue", *arms]
    times = {a: [] for a in order}
    for rnd in range(args.warm + args.rounds):
        for arm in order if rnd % 2 == 0 else order[::-1]:
            set_lever(arm)
            t = cell.chain(args.calls, arm)
            t0 = time.perf_counter()
            mx.eval(t)
            dt = 1e6 * (time.perf_counter() - t0) / args.calls
            if rnd >= args.warm:
                times[arm].append(dt)
    set_lever("default")
    stat = min if args.stat == "min" else statistics.median
    med = {a: stat(v) for a, v in times.items()}
    out = {"base": med["base"] - med["glue"]}
    for a in arms[1:]:
        out[a] = med[a] - med["base"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paths", nargs="+", default=["gather", "dense"])
    ap.add_argument("--codecs", nargs="+", default=["q6_k", "q4_k"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--k", type=int, default=512, help="input width")
    ap.add_argument("--n", type=int, default=2048, help="output width")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--calls", type=int, default=64, help="calls per sample")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--seed-rows", type=int, default=64)
    ap.add_argument(
        "--stat",
        default="median",
        choices=["median", "min"],
        help="per-arm statistic over rounds; min is steadier on a shared GPU",
    )
    ap.add_argument("--out", default="lora_epilogue")
    args = ap.parse_args()

    rng = np.random.default_rng(3)
    cells = []
    for codec in args.codecs:
        row = quantized_rows(codec, args.n, args.k, args.seed_rows, rng)
        dense_w = row
        stack = None
        if any(p != "dense" for p in args.paths):
            stack = mx.contiguous(mx.broadcast_to(row, (args.experts, *row.shape)))
        mx.eval(dense_w, *([stack] if stack is not None else []))
        for path in args.paths:
            w = dense_w if path == "dense" else stack
            for rank in args.ranks:
                for rows in args.rows:
                    cell = Cell(path, codec, rank, rows, w, args.experts, args, rng)
                    t = bench_cell(cell, rows, args)
                    cells.append(
                        {
                            "path": path,
                            "codec": codec,
                            "rank": rank,
                            "rows": rows,
                            "us": t,
                        }
                    )
                    print(
                        f"{path} {codec} rank {rank} rows {rows} "
                        + " ".join(f"{a}:{v:.1f}" for a, v in t.items()),
                        flush=True,
                    )

    meta = {"version": getattr(kq, "__version__", "?"), "k": args.k, "n": args.n}
    with open(args.out + ".json", "w") as f:
        json.dump({**meta, "cells": cells}, f, indent=1)
    lines = [
        f"# Rows LoRA epilogue, microseconds per call (mlx-kquant {meta['version']})",
        "",
        "base is the op alone; the other arms are the extra time over base.",
        "",
        "| path | codec | rank | rows | " + " | ".join(ARMS) + " |",
        "|---|---|---|---|" + "---|" * len(ARMS),
    ]
    for c in cells:
        lines.append(
            f"| {c['path']} | {c['codec']} | {c['rank']} | {c['rows']} | "
            + " | ".join(f"{c['us'][a]:.1f}" if a in c["us"] else "-" for a in ARMS)
            + " |"
        )
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", args.out + ".md", args.out + ".json")


if __name__ == "__main__":
    main()
