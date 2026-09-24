"""Per-route timing of the small-M transpose band, one process.

Source of the NAX-silicon small-M routing entries in src/kquant_matmul.cpp
(kq_splitk_nax_min_m, kq_verify_mma_min_m, kq_smallbm_policy route_min).

Each arm forces one route through KQ_QMM_ROUTE, which the op reads live per
call, so every arm shares one process and one resident copy of the
weights. The `default` arm leaves routing to the per-codec policy. An arm
is skipped where its route cannot serve the codec, shape or M.

A sample is a chain of --chain calls in one eval, each call's input tied to
the previous output through one small elementwise op, so the calls
serialize the way the projections of a forward do instead of overlapping
on the GPU. The `glue` arm times the tie alone; reported times subtract
it. Arm order rotates every round and reverses on odd rounds, so each arm
visits every slot position equally.

Weights are real codec output: --seed-rows float rows are quantized with
kq.quantize and tiled to N. The chain cycles through enough distinct
copies of the weights to exceed --stream-mb, so each call streams its
weights from DRAM the way a forward does instead of reading them from the
system cache. Before timing, every arm's output is checked against an
f32 reference on the first --check-rows output columns.

Cells: codec x (N,K) shape x M x route. Markdown + JSON out.
"""

import argparse
import json
import os
import statistics
import time

ROUTES = [
    "default",
    "qmv",
    "verify_qmv",
    "mv_ext",
    "verify_mma",
    "splitk",
    "nax",
    "nax_splitk",
]

# Ternary Bonsai 2 27B / Qwen3.8-27B projections: gate/up, down, GDN qkv,
# GDN z, GDN out and attention o, attention q with gate, attention k/v.
DEFAULT_SHAPES = (
    "17408x5120,5120x17408,10240x5120,6144x5120,5120x6144,12288x5120,1024x5120"
)
DEFAULT_MS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]

VERIFY_QMV = {
    "q6_k",
    "q8_0",
    "q4_k",
    "q5_k",
    "q5_1",
    "q3_k",
    "q2_k",
    "q4_0",
    "q4_1",
    "q5_0",
    "pq2_0",
    "ptq1_0",
}
VERIFY_MMA = {"pq2_0", "ptq1_0", "q4_0"}
NO_NAX = {"mxfp4", "nvfp4"}


def applies(route, codec, M, K):
    if route in ("default", "qmv"):
        return True
    if route == "verify_qmv":
        return codec in VERIFY_QMV and 2 <= M <= 8
    if route == "mv_ext":
        return 2 <= M <= 12
    if route == "verify_mma":
        return codec in VERIFY_MMA and M <= 8
    if route == "splitk":
        return codec not in NO_NAX
    if route in ("nax", "nax_splitk"):
        return codec not in NO_NAX and K % 64 == 0
    return route in extra_routes()


def extra_routes():
    return [r for r in os.environ.get("BENCH_EXTRA_ROUTES", "").split(",") if r]


def make_weights(codec, N, K, seed_rows):
    import mlx.core as mx

    import mlx_kquant as kq

    wf = mx.random.normal((seed_rows, K), key=mx.random.key(K)).astype(mx.float32)
    w, s = kq.quantize(wf, codec)
    mx.eval(w, s)
    reps = (N + seed_rows - 1) // seed_rows
    wt = mx.tile(w, (reps, 1))[:N] if w.ndim == 2 else w
    st = s if s.size <= 1 else mx.tile(s, (reps,) + (1,) * (s.ndim - 1))[:N]
    wt = mx.contiguous(wt)
    st = mx.contiguous(st)
    mx.eval(wt, st)
    return wt, st, wf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codecs", nargs="+", default=["pq2_0", "ptq1_0", "q4_0"])
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS)
    ap.add_argument("--routes", nargs="+", default=ROUTES)
    ap.add_argument("--chain", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed-rows", type=int, default=256)
    ap.add_argument("--stream-mb", type=int, default=512)
    ap.add_argument("--check-rows", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    args = ap.parse_args()

    import mlx.core as mx

    import mlx_kquant as kq

    dt = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    routes = list(args.routes) + [r for r in extra_routes() if r not in args.routes]
    dev = mx.device_info()["device_name"]

    def set_route(r):
        if r in ("default", "glue"):
            os.environ.pop("KQ_QMM_ROUTE", None)
        else:
            os.environ["KQ_QMM_ROUTE"] = r

    def chain(r, x0, ws, s, codec):
        set_route(r)
        x = x0
        for i in range(max(args.chain, len(ws))):
            if r == "glue":
                o = x
            else:
                o = kq.quantized_matmul(x, ws[i % len(ws)], s, codec, transpose=True)
            t = o[:, :1]
            x = mx.where(t > 1e30, t, x0)
        mx.eval(x)

    def sample(r, x0, ws, s, codec):
        t0 = time.perf_counter()
        chain(r, x0, ws, s, codec)
        return (time.perf_counter() - t0) * 1e3 / max(args.chain, len(ws))

    results = []
    for codec in args.codecs:
        for N, K in shapes:
            w, s, wf = make_weights(codec, N, K, args.seed_rows)
            wbytes = w.nbytes
            ncopy = max(1, -(-args.stream_mb * 2**20 // wbytes))
            ws = [w] + [mx.array(w) + 0 for _ in range(ncopy - 1)]
            mx.eval(ws)
            nchk = min(args.check_rows, args.seed_rows, N)
            wdeq = kq.dequantize(w[:nchk], s if s.size <= 1 else s[:nchk], codec)
            wdeq = wdeq.astype(mx.float32)
            mx.eval(wdeq)
            for M in args.ms:
                x0 = mx.random.normal((M, K), key=mx.random.key(M)).astype(dt)
                mx.eval(x0)
                ref = x0.astype(mx.float32) @ wdeq.T
                arms = [r for r in routes if applies(r, codec, M, K)]
                errs = {}
                for r in arms:
                    set_route(r)
                    o = kq.quantized_matmul(x0, w, s, codec, transpose=True)
                    d = (o[:, :nchk].astype(mx.float32) - ref).abs().max()
                    errs[r] = (d / (ref.abs().max() + 1e-6)).item()
                arms = arms + ["glue"]
                for _ in range(args.warmup):
                    for r in arms:
                        chain(r, x0, ws, s, codec)
                times = {r: [] for r in arms}
                for rnd in range(args.rounds):
                    k = rnd % len(arms)
                    order = arms[k:] + arms[:k]
                    if rnd % 2:
                        order = order[::-1]
                    for r in order:
                        times[r].append(sample(r, x0, ws, s, codec))
                glue = statistics.median(times["glue"])
                base = statistics.median(times["default"]) - glue
                row = {
                    "codec": codec,
                    "N": N,
                    "K": K,
                    "M": M,
                    "glue_ms": glue,
                    "routes": {},
                }
                for r in arms:
                    if r == "glue":
                        continue
                    ms = statistics.median(times[r]) - glue
                    row["routes"][r] = {
                        "ms": ms,
                        "vs_default": base / ms if ms > 0 else float("nan"),
                        "gbs": wbytes / (ms * 1e-3) / 1e9 if ms > 0 else float("nan"),
                        "rel_err": errs[r],
                    }
                results.append(row)
                best = min(row["routes"].items(), key=lambda kv: kv[1]["ms"])
                cells = " ".join(
                    f"{r}={v['ms'] * 1e3:.0f}" for r, v in row["routes"].items()
                )
                print(
                    f"{codec:7s} [{N}x{K}] M{M:<3d} us: {cells} | best={best[0]}",
                    flush=True,
                )
            del w, ws, s, wf, wdeq

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"device": dev, "dtype": args.dtype, "rows": results}, f)

    lines = ["# Small-M route sweep", ""]
    lines.append(
        f"Device: {dev}  dtype: {args.dtype}  chain: {args.chain}  "
        f"stream: {args.stream_mb} MB"
    )
    lines.append("")
    lines.append("Microseconds per call, glue subtracted. Fastest route marked *.")
    lines.append("")
    for codec in args.codecs:
        for N, K in shapes:
            sub = [
                r for r in results if r["codec"] == codec and (r["N"], r["K"]) == (N, K)
            ]
            if not sub:
                continue
            present = [r for r in routes if any(r in x["routes"] for x in sub)]
            lines.append(f"## {codec} [{N}x{K}]")
            lines.append("")
            lines.append("| M | " + " | ".join(present) + " |")
            lines.append("|---|" + "---|" * len(present))
            for x in sub:
                best = min(v["ms"] for v in x["routes"].values())
                cells = []
                for r in present:
                    v = x["routes"].get(r)
                    if v is None:
                        cells.append("-")
                    else:
                        mark = "*" if v["ms"] == best else ""
                        cells.append(f"{v['ms'] * 1e3:.0f}{mark}")
                lines.append(f"| {x['M']} | " + " | ".join(cells) + " |")
            lines.append("")
    md = "\n".join(lines)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    main()
