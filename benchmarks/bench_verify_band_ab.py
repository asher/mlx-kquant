"""Thermally-paired A/B of the NAX split-K tile (KQ_QMM_SPLITK_NAX) per codec.

Source of the kq_splitk_nax_min_m entries in src/kquant_matmul.cpp.
Measured on an M5 Max; re-run on new silicon before trusting them.

KQ_QMM_SPLITK_NAX is read live per dispatch, so all arms share one
process and one resident copy of the weights. Arms alternate in
Thue-Morse slot order, not ABBA: an ABBA contrast is the quadratic
contrast and aliases thermal curvature into the arm difference.

Weights are synthetic random codes in the quantized layout; every bit
pattern is valid, and it skips the minutes-long IQ encodes.

Cells: codec x (N,K) shape x M x split target. Markdown + JSON out.
"""

import argparse
import json
import os
import statistics
import sys
import time

DEFAULT_CODECS = [
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "iq4_nl",
    "iq4_xs",
    "iq3_s",
    "iq3_xxs",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq1_s",
    "iq1_m",
]

# Muse-Glimmer-30B MLP shapes: gate/up [19968x6656], down [6656x19968].
DEFAULT_SHAPES = "19968x6656,6656x19968"
DEFAULT_MS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
DEFAULT_TARGETS = [32, 16, 8]

# Thue-Morse: t[i] = parity of popcount(i). Balances linear drift without
# the quadratic aliasing an ABBA block introduces.
THUE_MORSE = [bin(i).count("1") & 1 for i in range(8)]


def group_size_of(codec):
    return 32 if codec in ("q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl") else 256


def effective_sp(target, K, gs):
    """Mirror the host-side split resolution: the target resolves down to
    a divisor of K / max(gs, BK). Different targets can be one kernel."""
    sliceq = max(gs, 64)
    nblk = K // sliceq
    sp = min(target, nblk)
    while sp > 1 and nblk % sp != 0:
        sp -= 1
    return sp


def probe_layout(codec, N, K):
    import mlx.core as mx
    import numpy as np

    import mlx_kquant as kq

    if codec.startswith("iq"):
        tests_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"
        )
        sys.path.insert(0, tests_dir)
        from test_codecs import CODECS, _synth_iq_wire

        _, wpb, bpb, _, _ = CODECS[codec]
        rng = np.random.default_rng(N + K)
        wire = _synth_iq_wire(rng, bpb, N * (K // wpb))
        return (
            mx.array(wire.reshape(N, (K // wpb) * bpb)),
            mx.array(np.zeros((1,), dtype=np.uint8)),
        )

    wf = mx.random.normal((8, K)).astype(mx.float32)
    w8, s8 = kq.quantize(wf, codec)
    mx.eval(w8, s8)
    rng = np.random.default_rng(N + K)

    def full(sample):
        a = np.asarray(sample)
        shape = (N,) + a.shape[1:]
        if np.issubdtype(a.dtype, np.floating):
            return (rng.standard_normal(shape) * 0.01).astype(a.dtype)
        info = np.iinfo(a.dtype)
        return rng.integers(
            info.min, info.max, size=shape, endpoint=True, dtype=a.dtype
        )

    return mx.array(full(w8)), mx.array(full(s8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codecs", nargs="+", default=DEFAULT_CODECS)
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS)
    ap.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    args = ap.parse_args()

    import mlx.core as mx

    import mlx_kquant as kq

    if not kq.nax_available():
        sys.exit("NAX not available on this device; nothing to measure.")

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    dt = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]

    def time_arm(target, x, w, s, codec):
        os.environ["KQ_QMM_SPLITK_NAX"] = str(target)

        def call(xx):
            return kq.quantized_matmul(xx, w, s, codec, transpose=True)

        o = call(x)
        mx.eval(o)
        t0 = time.perf_counter()
        o = call(x)
        mx.eval(o)
        return (time.perf_counter() - t0) * 1e3

    results = []
    for codec in args.codecs:
        gs = group_size_of(codec)
        for N, K in shapes:
            try:
                w, s = probe_layout(codec, N, K)
                mx.eval(w, s)
            except Exception as e:  # codec cannot synthesize at this shape
                print(f"skip {codec} [{N}x{K}]: {e}", file=sys.stderr)
                continue
            wbytes = w.nbytes + s.nbytes
            for M in args.ms:
                x = mx.random.normal((M, K), key=mx.random.key(M)).astype(dt)
                mx.eval(x)
                for target in args.targets:
                    sp = effective_sp(target, K, gs)
                    if sp <= 1:
                        continue
                    for _ in range(args.warmup):
                        time_arm(0, x, w, s, codec)
                        time_arm(target, x, w, s, codec)
                    off, on = [], []
                    for _rep in range(args.iters):
                        for slot in THUE_MORSE:
                            # slot 0 -> off first, slot 1 -> on first
                            if slot == 0:
                                off.append(time_arm(0, x, w, s, codec))
                                on.append(time_arm(target, x, w, s, codec))
                            else:
                                on.append(time_arm(target, x, w, s, codec))
                                off.append(time_arm(0, x, w, s, codec))
                    off_ms = statistics.median(off)
                    on_ms = statistics.median(on)
                    results.append(
                        {
                            "codec": codec,
                            "N": N,
                            "K": K,
                            "M": M,
                            "target": target,
                            "sp": sp,
                            "off_ms": off_ms,
                            "on_ms": on_ms,
                            "speedup": off_ms / on_ms,
                            "off_gbs": wbytes / (off_ms * 1e-3) / 1e9,
                            "on_gbs": wbytes / (on_ms * 1e-3) / 1e9,
                        }
                    )
                    print(
                        f"{codec:8s} [{N}x{K}] M{M:<3d} t{target:<3d} sp={sp:<3d} "
                        f"off={off_ms:.3f} on={on_ms:.3f} "
                        f"speedup={off_ms / on_ms:.3f}",
                        flush=True,
                    )
            del w, s

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"device": mx.device_info()["device_name"], "rows": results}, f)

    lines = ["# NAX split-K verify band A/B", ""]
    lines.append(f"Device: {mx.device_info()['device_name']}  dtype: {args.dtype}")
    lines.append("")
    lines.append("speedup = off / on; >1 means split-K is faster.")
    lines.append("")
    for codec in args.codecs:
        rows = [r for r in results if r["codec"] == codec]
        if not rows:
            continue
        lines.append(f"## {codec}")
        lines.append("")
        for N, K in shapes:
            sub = [r for r in rows if r["N"] == N and r["K"] == K]
            if not sub:
                continue
            targets = sorted({r["target"] for r in sub}, reverse=True)
            lines.append(f"### [{N}x{K}]")
            lines.append("")
            lines.append(
                "| M | off ms | " + " | ".join(f"t{t}" for t in targets) + " |"
            )
            lines.append("|---|---|" + "---|" * len(targets))
            for M in args.ms:
                cells = [r for r in sub if r["M"] == M]
                if not cells:
                    continue
                row = f"| {M} | {cells[0]['off_ms']:.3f} |"
                for t in targets:
                    c = [r for r in cells if r["target"] == t]
                    row += f" {c[0]['speedup']:.3f} |" if c else " - |"
                lines.append(row)
            lines.append("")
    md = "\n".join(lines)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    main()
