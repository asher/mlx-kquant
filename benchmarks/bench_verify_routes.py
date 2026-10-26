"""Per-route timing of the small-M transpose band, one process.

Source of the NAX-GPU small-M routing entries in src/kquant_matmul.cpp
(kq_nax_small_m, kq_verify_nax_min_m, kq_verify_mma_min_m_nax,
kq_smallbm_policy route_min).

Each arm forces one route through KQ_QMM_ROUTE, which the op reads live per
call, so every arm shares one process and one resident copy of the
weights. The `default` arm leaves routing to the per-codec policy. The
bench sets KQ_QMM_ROUTE_STRICT=1, so a route that cannot serve the codec,
shape or M raises instead of running the default routing, and that arm is
dropped for the cell.

A sample is a chain of calls in one eval, each call's input tied to the
previous output through one small elementwise op, so the calls serialize
the way the projections of a forward do instead of overlapping on the GPU.
The graph is built before the timer starts, so a sample times the
evaluation only. The `glue` arm times the tie alone; reported times
subtract it. Arm order rotates by one slot every round and reverses on
odd rounds.

Weights are real codec output: --seed-rows float rows are quantized with
kq.quantize and tiled to N. The decode-only codecs (mxfp4, nvfp4), which
kq.quantize cannot encode, take --seed-rows rows of synthetic wire from
tests/kqref.py synth_wire instead, with nvfp4 group scales of 1/64 to
1/32, the range of real weights. The chain cycles through enough distinct
copies of the weights to exceed --stream-mb, so each call streams its
weights from DRAM the way a forward does instead of reading them from the
system cache. Before timing, every arm's output is checked against an f32
reference on the first --check-rows output columns; an arm over 2e-2
relative error is reported and left out of the best-route pick.

Cells: codec x (N,K) shape x M x route. Markdown + JSON out.

--interleave-codecs times every codec of one (N,K) shape and M in the same
rounds, with all codecs' weights resident: the arms are (codec, route)
pairs, arm order rotates by one slot every round and reverses on the
Thue-Morse bit of the round. Over all arm pairs each codec leads about as
often as the other, but one given pair of routes can lead 2 to 6 times in
8 rounds, and some arms never open a round.
The Markdown then adds a parity table, the fastest route of each codec
over the fastest route of the first codec, per shape and M. Without the
flag codecs run one after another as before.
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
    "verify_nax",
    "splitk",
    "nax",
    "nax_splitk",
]

# Ternary Bonsai 2 27B / Qwen3.8-27B projections: gate/up, down, GDN qkv,
# GDN z, GDN out and attention o, attention q with gate, attention k/v,
# and the vocab head.
DEFAULT_SHAPES = (
    "17408x5120,5120x17408,10240x5120,6144x5120,5120x6144,12288x5120,1024x5120,"
    "248320x5120"
)
DEFAULT_MS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
MAX_REL_ERR = 2e-2

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
VERIFY_MMA = {"pq2_0", "ptq1_0", "q4_0", "q4_1", "q8_0"}
VERIFY_NAX = {
    "pq2_0": 128,
    "q4_0": 64,
    "q4_1": 64,
    "q8_0": 128,
    "q4_k": 256,
    "iq4_xs": 256,
    "iq4_nl": 256,
    "q5_k": 128,
    "q6_k": 128,
    "q3_k": 128,
    "q2_k": 128,
    "nvfp4": 128,
}
NO_NAX = {"mxfp4"}


def applies(route, codec, M, K, nax):
    """Cheap pre-filter; strict mode in the op has the final say."""
    if route in ("default", "qmv"):
        return True
    if route == "verify_qmv":
        return codec in VERIFY_QMV and 2 <= M <= 8
    if route == "mv_ext":
        return 2 <= M <= 12
    if route == "verify_mma":
        return codec in VERIFY_MMA and M <= 8
    if route == "verify_nax":
        return nax and codec in VERIFY_NAX and M <= 8 and K % VERIFY_NAX[codec] == 0
    if route == "splitk":
        return codec not in NO_NAX
    if route in ("nax", "nax_splitk"):
        return nax and codec not in NO_NAX and K % 64 == 0
    return False


def synth_weights(codec, rows, K):
    """Synthetic wire for a codec kq.quantize cannot encode."""
    import sys

    import mlx.core as mx
    import numpy as np

    from mlx_kquant.codec_geometry import geometry

    tests = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests")
    sys.path.insert(0, os.path.abspath(tests))
    from kqref import synth_wire

    _, _, bpb, wpb = geometry(codec)
    wire = synth_wire(
        np.random.default_rng(K),
        codec,
        bpb,
        rows * (K // wpb),
        nvfp4_scales=(0x08, 0x11),
    )
    return mx.array(wire.reshape(rows, (K // wpb) * bpb)), mx.zeros((1,), mx.uint8)


def make_weights(codec, N, K, seed_rows):
    import mlx.core as mx

    import mlx_kquant as kq
    from mlx_kquant.codec_geometry import DECODE_ONLY_CODECS

    if codec in DECODE_ONLY_CODECS:
        wf = None
        w, s = synth_weights(codec, seed_rows, K)
    else:
        wf = mx.random.normal((seed_rows, K), key=mx.random.key(K))
        wf = wf.astype(mx.float32)
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--codecs",
        nargs="+",
        default=["pq2_0", "ptq1_0", "q4_0"],
        help="codecs to sweep",
    )
    ap.add_argument(
        "--shapes",
        default=DEFAULT_SHAPES,
        help="comma-separated NxK weight shapes (default: the 27B projections "
        "and head)",
    )
    ap.add_argument(
        "--ms", type=int, nargs="+", default=DEFAULT_MS, help="activation rows M"
    )
    ap.add_argument(
        "--routes",
        nargs="+",
        default=ROUTES,
        help="routes to time; default is always timed as the baseline",
    )
    ap.add_argument(
        "--chain",
        type=int,
        default=8,
        help="minimum calls per sample (raised to the weight-copy count)",
    )
    ap.add_argument("--rounds", type=int, default=8, help="timed samples per arm")
    ap.add_argument(
        "--warmup", type=int, default=2, help="untimed samples per arm first"
    )
    ap.add_argument(
        "--seed-rows",
        type=int,
        default=256,
        help="distinct float rows quantized and tiled to N",
    )
    ap.add_argument(
        "--stream-mb",
        type=int,
        default=512,
        help="total size of the weight copies a chain cycles through",
    )
    ap.add_argument(
        "--check-rows",
        type=int,
        default=512,
        help="output columns checked against the f32 reference",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="activation dtype",
    )
    ap.add_argument(
        "--interleave-codecs",
        action="store_true",
        help="time all codecs of a shape and M in the same rounds (Thue-Morse "
        "order) and add a codec parity table",
    )
    ap.add_argument("--json-out", help="write every cell as JSON here")
    ap.add_argument("--md-out", help="write the Markdown tables here, not stdout")
    args = ap.parse_args()

    import mlx.core as mx

    import mlx_kquant as kq

    dt = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    routes = ["default"] + [r for r in args.routes if r != "default"]
    dev = mx.device_info()["device_name"]
    nax = kq.nax_available()
    os.environ["KQ_QMM_ROUTE_STRICT"] = "1"

    def set_route(r):
        if r in ("default", "glue"):
            os.environ.pop("KQ_QMM_ROUTE", None)
        else:
            os.environ["KQ_QMM_ROUTE"] = r

    def sample(r, x0, ws, s, codec):
        set_route(r)
        n = max(args.chain, len(ws))
        x = x0
        for i in range(n):
            if r == "glue":
                o = x
            else:
                o = kq.quantized_matmul(x, ws[i % len(ws)], s, codec, transpose=True)
            t = o[:, :1]
            x = mx.where(t > 1e30, t, x0)
        t0 = time.perf_counter()
        mx.eval(x)
        return (time.perf_counter() - t0) * 1e3 / n

    def tm_bit(n):
        return bin(n).count("1") & 1

    def build(codec, N, K):
        w, s, wf = make_weights(codec, N, K, args.seed_rows)
        wbytes = w.nbytes
        ncopy = max(1, -(-args.stream_mb * 2**20 // wbytes))
        ws = [w] + [mx.array(w) + 0 for _ in range(ncopy - 1)]
        mx.eval(ws)
        nchk = min(args.check_rows, args.seed_rows, N)
        wdeq = kq.dequantize(w[:nchk], s if s.size <= 1 else s[:nchk], codec)
        wdeq = wdeq.astype(mx.float32)
        mx.eval(wdeq)
        return (codec, w, s, ws, wdeq, nchk, wbytes)

    def time_cells(N, K, M, cells, thue_morse):
        """One (N, K, M) for every codec in cells, all arms in shared rounds.
        A glue arm per distinct chain length. Returns one row per codec."""
        x0 = mx.random.normal((M, K), key=mx.random.key(M)).astype(dt)
        mx.eval(x0)
        by_codec = {c[0]: c for c in cells}
        arms = []
        errs = {}
        for codec, w, s, _ws, wdeq, nchk, _wbytes in cells:
            ref = x0.astype(mx.float32) @ wdeq.T
            for r in routes:
                if not applies(r, codec, M, K, nax):
                    continue
                set_route(r)
                try:
                    o = kq.quantized_matmul(x0, w, s, codec, transpose=True)
                    d = (o[:, :nchk].astype(mx.float32) - ref).abs().max()
                    errs[(codec, r)] = (d / (ref.abs().max() + 1e-6)).item()
                    arms.append((codec, r))
                except RuntimeError:
                    pass
        glue_of = {}
        for codec, _w, s, ws, _wdeq, _nchk, _wbytes in cells:
            n = max(args.chain, len(ws))
            if n not in glue_of:
                glue_of[n] = (ws, s, codec)
                arms.append(("glue", n))

        def run(arm):
            codec, r = arm
            if codec == "glue":
                ws, s, c = glue_of[r]
                return sample("glue", x0, ws, s, c)
            _, w, s, ws, _, _, _ = by_codec[codec]
            return sample(r, x0, ws, s, codec)

        for _ in range(args.warmup):
            for a in arms:
                run(a)
        times = {a: [] for a in arms}
        for rnd in range(args.rounds):
            k = rnd % len(arms)
            order = arms[k:] + arms[:k]
            flip = tm_bit(rnd) if thue_morse else rnd % 2
            if flip:
                order = order[::-1]
            for a in order:
                times[a].append(run(a))
        rows = []
        for codec, _w, _s, ws, _wdeq, _nchk, wbytes in cells:
            glue = statistics.median(times[("glue", max(args.chain, len(ws)))])
            base = statistics.median(times[(codec, "default")]) - glue
            row = {
                "codec": codec,
                "N": N,
                "K": K,
                "M": M,
                "glue_ms": glue,
                "routes": {},
            }
            for c, r in arms:
                if c != codec:
                    continue
                ms = statistics.median(times[(c, r)]) - glue
                row["routes"][r] = {
                    "ms": ms,
                    "vs_default": base / ms if ms > 0 else float("nan"),
                    "gbs": wbytes / (ms * 1e-3) / 1e9 if ms > 0 else float("nan"),
                    "rel_err": errs[(c, r)],
                }
            rows.append(row)
            ok = {r: v for r, v in row["routes"].items() if v["rel_err"] <= MAX_REL_ERR}
            best = min(ok.items() or row["routes"].items(), key=lambda kv: kv[1]["ms"])
            cells_s = " ".join(
                f"{r}={v['ms'] * 1e3:.0f}"
                + ("" if r in ok else f"(err {v['rel_err']:.1e})")
                for r, v in row["routes"].items()
            )
            print(
                f"{codec:7s} [{N}x{K}] M{M:<3d} us: {cells_s} | best={best[0]}",
                flush=True,
            )
        return rows

    results = []
    if args.interleave_codecs:
        for N, K in shapes:
            cells = [build(codec, N, K) for codec in args.codecs]
            for M in args.ms:
                results.extend(time_cells(N, K, M, cells, True))
            del cells
    else:
        for codec in args.codecs:
            for N, K in shapes:
                cell = build(codec, N, K)
                for M in args.ms:
                    results.extend(time_cells(N, K, M, [cell], False))
                del cell

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "device": dev,
                    "dtype": args.dtype,
                    "interleaved": bool(args.interleave_codecs),
                    "rows": results,
                },
                f,
            )

    lines = ["# Small-M route sweep", ""]
    lines.append(
        f"Device: {dev}  dtype: {args.dtype}  chain: {args.chain}  "
        f"stream: {args.stream_mb} MB"
    )
    lines.append("")
    lines.append(
        "Microseconds per call, glue subtracted. Fastest route marked *; "
        "a route over the error bound is marked !."
    )
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
                best = min(
                    v["ms"] for v in x["routes"].values() if v["rel_err"] <= MAX_REL_ERR
                )
                cells = []
                for r in present:
                    v = x["routes"].get(r)
                    if v is None:
                        cells.append("-")
                    else:
                        mark = "*" if v["ms"] == best else ""
                        if v["rel_err"] > MAX_REL_ERR:
                            mark = "!"
                        cells.append(f"{v['ms'] * 1e3:.0f}{mark}")
                lines.append(f"| {x['M']} | " + " | ".join(cells) + " |")
            lines.append("")
    if args.interleave_codecs and len(args.codecs) > 1:
        base_c = args.codecs[0]

        def best_ms(codec, N, K, M):
            for x in results:
                if (x["codec"], x["N"], x["K"], x["M"]) == (codec, N, K, M):
                    ok = [
                        v["ms"]
                        for v in x["routes"].values()
                        if v["rel_err"] <= MAX_REL_ERR
                    ]
                    return min(ok) if ok else None
            return None

        lines.append(f"## Parity: fastest route over fastest {base_c} route")
        lines.append("")
        others = args.codecs[1:]
        lines.append(
            f"| Shape | M | {base_c} us | "
            + " | ".join(f"{c} us | {c} / {base_c}" for c in others)
            + " |"
        )
        lines.append("|---|---|---|" + "---|---|" * len(others))
        for N, K in shapes:
            for M in args.ms:
                b = best_ms(base_c, N, K, M)
                cells = []
                for c in others:
                    o = best_ms(c, N, K, M)
                    if b and o:
                        cells.append(f"{o * 1e3:.0f} | {o / b:.2f}")
                    else:
                        cells.append("- | -")
                bs = f"{b * 1e3:.0f}" if b else "-"
                lines.append(f"| {N}x{K} | {M} | {bs} | " + " | ".join(cells) + " |")
        lines.append("")
    md = "\n".join(lines)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    main()
