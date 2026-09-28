"""First-dispatch cost of the DiT kernel list, in fresh processes.

The parent starts --repeats child processes. Each child times the metallib
load on its own, then walks the DiT cells (shape matrix x rows x batches x
codecs at one dtype). Per cell it builds weights and input, primes the
allocator with a buffer of the output's size, and times the first and the
second call of the kq arm at the current policy. The first call of a new
kernel builds its pipeline state; a cell whose kernel an earlier cell built
is counted once. The report sums first minus second over the distinct
kernels, as the median over children.

--stock adds stock MLX ops a DiT block runs (dense matmul, attention at
head dim 128, RMS norm, RoPE, SiLU gate) for comparison.

The OS keeps compiled shaders in a cache that outlives the process, so a
child measures a first image with that cache warm; clearing it is outside
this script. Markdown and JSON out.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import dit_common as dc

CHILD = "--child"


def child(args):
    t0 = time.perf_counter()
    import mlx.core as mx

    t_import = time.perf_counter() - t0
    import mlx_kquant as kq

    t1 = time.perf_counter()
    kq.metallib_loads()
    t_metallib = time.perf_counter() - t1
    dt = dc.mx_dtype(args.dtype)
    store = dc.Store(0)
    seen = {}
    rows = []

    def timed(label, fn, out_shape, log_kernel):
        mx.eval(mx.zeros(out_shape, dtype=dt))
        if log_kernel:
            with dc.KernelLog() as log:
                ta = time.perf_counter()
                mx.eval(fn())
                first = time.perf_counter() - ta
            kname = log.kernels[-1] if log.kernels else label
        else:
            ta = time.perf_counter()
            mx.eval(fn())
            first = time.perf_counter() - ta
            kname = label
        if kname in seen:
            # An earlier cell built this kernel; one call is enough.
            return
        seen[kname] = label
        ta = time.perf_counter()
        mx.eval(fn())
        second = time.perf_counter() - ta
        rows.append(
            {
                "cell": label,
                "kernel": kname,
                "first_ms": first * 1e3,
                "second_ms": second * 1e3,
            }
        )

    for fam in args.families:
        for op in dc.FAMILIES[fam]["ops"]:
            for L in args.rows or dc.FAMILIES[fam]["rows"]:
                for B in args.batches:
                    g = dc.cell(fam, op, L, B, args.shrink)
                    for codec in args.codecs:
                        fn = dc.make_fn("kq", codec, g, dt, store)
                        mx.eval(store.act(g, dt), *store.kq(g, codec))
                        label = f"{fam}.{op} {L}x{B} {codec}"
                        timed(label, fn, (B, g["L"], g["N"]), True)
                    store.x.clear()
    if args.stock:
        g = dc.cell("zimage", "w13", 4608, 1, args.shrink)
        fn = dc.make_fn("dense", None, g, dt, store)
        mx.eval(store.act(g, dt), store.dense_w(g, dt))
        timed("matmul", fn, (1, g["L"], g["N"]), False)
        H, D, L = 4, 128, max(64, 4608 // args.shrink)
        q = (mx.random.normal((1, H, L, D)) * 0.1).astype(dt)
        w = mx.ones((D,), dtype=dt)
        mx.eval(q)
        timed(
            "sdpa",
            lambda: mx.fast.scaled_dot_product_attention(q, q, q, scale=D**-0.5),
            (1, H, L, D),
            False,
        )
        timed("rms_norm", lambda: mx.fast.rms_norm(q, w, 1e-6), q.shape, False)
        timed(
            "rope",
            lambda: mx.fast.rope(
                q, D, traditional=False, base=10000.0, scale=1.0, offset=0
            ),
            q.shape,
            False,
        )
        timed("silu_gate", lambda: mx.sigmoid(q) * q * q, q.shape, False)
    print(
        json.dumps(
            {"import_s": t_import, "metallib_s": t_metallib, "rows": rows},
        )
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--families", default=",".join(dc.FAMILIES))
    ap.add_argument("--rows", default="")
    ap.add_argument("--batches", default="1,2")
    ap.add_argument("--codecs", default=",".join(dc.MAIN_CODECS))
    ap.add_argument("--dtype", default="bfloat16", choices=list(dc.DTYPES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--shrink", type=int, default=1)
    ap.add_argument(CHILD, action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    args = ap.parse_args()
    args.families = [v for v in args.families.split(",") if v]
    args.rows = [int(v) for v in args.rows.split(",") if v]
    args.batches = [int(v) for v in args.batches.split(",") if v]
    args.codecs = [v for v in args.codecs.split(",") if v]
    if args.child:
        child(args)
        return

    argv = [a for a in sys.argv[1:] if a != CHILD]
    env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    runs = []
    for _ in range(args.repeats):
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), CHILD, *argv],
            env=env,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            sys.exit(r.stderr[-4000:])
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    desc = dc.machine_descriptor()

    per_kernel = {}
    for run in runs:
        for row in run["rows"]:
            per_kernel.setdefault(row["kernel"], []).append(
                (row["cell"], row["first_ms"] - row["second_ms"], row["second_ms"])
            )
    kernels = []
    for k, v in per_kernel.items():
        kernels.append(
            {
                "kernel": k,
                "cell": v[0][0],
                "tile": dc.tile_of(k),
                "extra_ms": statistics.median(x[1] for x in v),
                "second_ms": statistics.median(x[2] for x in v),
            }
        )
    sums = [sum(r["first_ms"] - r["second_ms"] for r in run["rows"]) for run in runs]
    kq_sums = [
        sum(
            r["first_ms"] - r["second_ms"]
            for r in run["rows"]
            if r["kernel"].startswith("kquant_")
        )
        for run in runs
    ]
    result = {
        "machine": desc,
        "args": vars(args),
        "metallib_ms": statistics.median(r["metallib_s"] * 1e3 for r in runs),
        "import_ms": statistics.median(r["import_s"] * 1e3 for r in runs),
        "sum_extra_ms": statistics.median(sums),
        "kq_sum_extra_ms": statistics.median(kq_sums),
        "kernels": kernels,
        "runs": runs,
    }
    lines = ["# DiT first dispatch", "", dc.descriptor_md(desc), ""]
    lines.append(
        f"{args.repeats} fresh processes, {args.dtype}. mlx import "
        f"{result['import_ms']:.0f} ms, metallib load {result['metallib_ms']:.1f} ms, "
        f"first-dispatch extra summed over {len(kernels)} kernels "
        f"{result['sum_extra_ms']:.1f} ms "
        f"(kq kernels {result['kq_sum_extra_ms']:.1f} ms)."
    )
    lines.append("")
    lines.append("| kernel | tile | first cell | extra ms | second ms |")
    lines.append("|---|---|---|---|---|")
    for k in kernels:
        lines.append(
            f"| {k['kernel']} | {k['tile'] or '-'} | {k['cell']} "
            f"| {k['extra_ms']:.2f} | {k['second_ms']:.3f} |"
        )
    md = "\n".join(lines)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=1)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    sys.exit(main())
