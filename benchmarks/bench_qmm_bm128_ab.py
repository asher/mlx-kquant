"""Thermally-paired A/B of the NAX qmm BM=128 tile (KQ_NAX_BM128) per codec.

Source of the per-codec `bm128_min_m` floors in `kq_smallbm_policy`
(src/kquant_matmul.cpp). The floors were measured on an M5 Max; re-run
this on new silicon before trusting them there.

The bm128 gate is a process-static env read, so the two arms run as
persistent child processes (on / off) and the parent alternates per-cell
timing requests between them in ABBA order: both arms of a cell execute
within seconds of each other on a warm GPU, defeating thermal drift.

Weights are synthetic random codes in the exact quantized layout (derived
from a probe quantize of an 8-row slab): every bit pattern is a valid
code, timing does not care about quality, and it skips the minutes-long
IQ encodes real weights would cost. Both arms load identical bytes from
a shared .npz cache.

Cells: codec x (N,K) shape x M, all M chosen inside the even-ceil(M/64)
dispatch window. GB/s = weight-bytes / call. Markdown + JSON out.

--codec-pair times two codecs against each other instead of one lever:
one child with the per-codec policy, the cell runs codec A, B, B, A (min
of the two), --rounds repeats every cell and reports each round beside
the min, and --cool-gate waits before each round until a bf16 4096^3
matmul in the child reaches that many TFLOPS (retry every 15 s, up to
--cool-wait s). A codec spelled mlx:<mode> is the stock MLX kernel on
mx.quantize(mode=<mode>) weights, for a reference ceiling. MLX picks its
tile by shape and does not expose the kernel name, so the cell records
the mlx version and the mode.

--tile-ab is the source of the large-M K ceilings (`kq_largem_max_k`): the
arms force KQ_NAX_TILE to 128,64,2,2 and 64,64,2,2, which bypasses the
current ceiling and the parity rule, so M may sit at either parity. Run it
at N 4096 and M 4128, 8256 and 16384 over K 4096, 10240, 12288 and 16384.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

CHILD = "__qmm_bm128_child__"

DEFAULT_CODECS = [
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "stq1_0",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
]
DEFAULT_MS = [224, 256, 512, 1024]


def child_main():
    import mlx.core as mx
    import numpy as np

    import mlx_kquant as kq

    cache = {}

    def get_weights(codec, N, K, path):
        key = (codec, N, K)
        if key not in cache:
            if len(cache) >= 2:  # keep residency bounded per child
                cache.pop(next(iter(cache)))
            z = np.load(path)
            w = mx.array(z["w"])
            s = mx.array(z["s"])
            mx.eval(w, s)
            cache[key] = (w, s)
        return cache[key]

    def mlx_weights(mode, N, K):
        key = ("mlx:" + mode, N, K)
        if key not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            wf = mx.random.normal((N, K), key=mx.random.key(N + K)).astype(mx.bfloat16)
            q = mx.quantize(wf, mode=mode)
            mx.eval(*q)
            cache[key] = tuple(q)
        return cache[key]

    def probe_tflops(seconds=1.0):
        n = 4096
        a = mx.random.normal((n, n)).astype(mx.bfloat16)
        b = mx.random.normal((n, n)).astype(mx.bfloat16)
        mx.eval(a, b)
        for _ in range(3):
            mx.eval(a @ b)
        t0 = time.perf_counter()
        k = 0
        while time.perf_counter() - t0 < seconds:
            mx.eval([a @ b for _ in range(8)])
            k += 8
        return 2 * n**3 * k / (time.perf_counter() - t0) / 1e12

    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        if req.get("cmd") == "probe":
            print(json.dumps({"tflops": probe_tflops()}), flush=True)
            continue
        codec, N, K, M = req["codec"], req["N"], req["K"], req["M"]
        x = mx.random.normal((M, K), key=mx.random.key(M)).astype(mx.bfloat16)
        mx.eval(x)
        if codec.startswith("mlx:"):
            mode = codec[4:]
            q = mlx_weights(mode, N, K)

            def call(xx, q=q, mode=mode):
                return mx.quantized_matmul(xx, *q, transpose=True, mode=mode)

        else:
            w, s = get_weights(codec, N, K, req["npz"])

            def call(xx, w=w, s=s, codec=codec):
                return kq.quantized_matmul(xx, w, s, codec, transpose=True)

        for _ in range(req["warmup"]):
            mx.eval(call(x))
        mx.synchronize()
        t0 = time.perf_counter()
        o = None
        for _ in range(req["iters"]):
            if o is not None:  # chain to serialize timing
                x = x + (o[0, 0] * mx.array(0.0, dtype=x.dtype))
            o = call(x)
            mx.eval(o)
        mx.synchronize()
        dt = (time.perf_counter() - t0) / req["iters"]
        print(json.dumps({"us": dt * 1e6}), flush=True)


def probe_layout(codec, N, K):
    """Synthesize (w, scales) in the codec's quantized layout: the IQ
    codecs, stq1_0 and the decode-only codecs via the tests' synthetic-wire
    helper (the IQ encoders demand an imatrix, and nvfp4 has no encoder;
    its group scales are 1/64 to 1/32), encodable codecs by random-filling
    the shape of a probe quantize."""
    import numpy as np

    from mlx_kquant.codec_geometry import DECODE_ONLY_CODECS, geometry

    if codec.startswith("iq") or codec == "stq1_0" or codec in DECODE_ONLY_CODECS:
        tests_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests",
        )
        sys.path.insert(0, tests_dir)
        from kqref import synth_wire

        _, _, bpb, wpb = geometry(codec)
        rng = np.random.default_rng(N + K)
        wire = synth_wire(rng, codec, bpb, N * (K // wpb), nvfp4_scales=(0x08, 0x11))
        return (
            wire.reshape(N, (K // wpb) * bpb),
            np.zeros((1,), dtype=np.uint8),
        )

    import mlx.core as mx

    import mlx_kquant as kq

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

    return full(w8), full(s8)


def codec_pair_main(args, shapes):
    import mlx.core as mx

    tmp = tempfile.mkdtemp(prefix="bm128ab-")
    env = dict(os.environ)
    env.pop("KQ_NAX_BM128", None)
    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), CHILD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )

    def ask(req):
        child.stdin.write(json.dumps(req) + "\n")
        child.stdin.flush()
        return json.loads(child.stdout.readline())

    def gate():
        if args.cool_gate <= 0:
            return None
        waited = 0.0
        while True:
            t = ask({"cmd": "probe"})["tflops"]
            if t >= args.cool_gate or waited >= args.cool_wait:
                return t
            print(f"[cool] {t:.1f} TFLOPS < {args.cool_gate:.1f}; waiting", flush=True)
            time.sleep(15)
            waited += 15

    a_c, b_c = args.codecs
    results = {"mlx_version": mx.__version__, "codecs": args.codecs, "cells": {}}
    try:
        for N, K in shapes:
            npz = {}
            wbytes = {}
            for codec in args.codecs:
                if codec.startswith("mlx:"):
                    npz[codec] = None
                    wbytes[codec] = None
                    continue
                import numpy as np

                path = os.path.join(tmp, f"{codec}-{N}x{K}.npz")
                if not os.path.exists(path):
                    w, s = probe_layout(codec, N, K)
                    np.savez(path, w=w, s=s)
                npz[codec] = path
                wbytes[codec] = np.load(path)["w"].nbytes
            for M in args.ms:

                def req(codec, N=N, K=K, M=M, npz=npz):
                    return {
                        "cmd": "cell",
                        "codec": codec,
                        "N": N,
                        "K": K,
                        "M": M,
                        "npz": npz[codec],
                        "iters": args.iters,
                        "warmup": args.warmup,
                    }

                ra, rb, gates = [], [], []
                for _ in range(args.rounds):
                    gates.append(gate())
                    a1 = ask(req(a_c))["us"]
                    b1 = ask(req(b_c))["us"]
                    b2 = ask(req(b_c))["us"]
                    a2 = ask(req(a_c))["us"]
                    ra.append(min(a1, a2))
                    rb.append(min(b1, b2))
                ua, ub = min(ra), min(rb)
                cell = {
                    "us_a": ua,
                    "us_b": ub,
                    "ratio_b_over_a": ub / ua,
                    "rounds_a": ra,
                    "rounds_b": rb,
                    "gate_tflops": gates,
                    "bytes_a": wbytes[a_c],
                    "bytes_b": wbytes[b_c],
                }
                results["cells"][f"{N}x{K}|M{M}"] = cell
                spread = ""
                if args.rounds > 1:
                    spread = (
                        f"  rounds a {min(ra):.0f}-{max(ra):.0f} "
                        f"b {min(rb):.0f}-{max(rb):.0f}"
                    )
                print(
                    f"{N}x{K} M{M:<5d} {a_c} {ua:8.1f}us  {b_c} {ub:8.1f}us  "
                    f"{b_c}/{a_c} {ub / ua:.3f}{spread}",
                    flush=True,
                )
    finally:
        try:
            child.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            child.stdin.flush()
        except Exception:
            pass
        child.wait(timeout=30)

    print(f"\n| shape | M | {a_c} us | {b_c} us | {b_c} / {a_c} |")
    print("|---|---|---|---|---|")
    for key, c in results["cells"].items():
        shape, m = key.split("|")
        print(
            f"| {shape} | {m[1:]} | {c['us_a']:.0f} | {c['us_b']:.0f} | "
            f"{c['ratio_b_over_a']:.3f} |"
        )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", nargs="+", default=DEFAULT_CODECS)
    ap.add_argument("--shapes", default="17408x5120,4096x14336")
    ap.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument(
        "--policy-arm",
        action="store_true",
        help="on-arm runs with KQ_NAX_BM128 unset (per-codec "
        "policy) instead of 1 (force floor 193)",
    )
    ap.add_argument(
        "--tile-ab",
        action="store_true",
        help="arms force KQ_NAX_TILE=128,64,2,2 and 64,64,2,2 (K ceiling "
        "sweep; M at either parity)",
    )
    ap.add_argument(
        "--codec-pair",
        action="store_true",
        help="time --codecs A B against each other in one policy child, "
        "A B B A per cell; mlx:<mode> is the stock MLX kernel",
    )
    ap.add_argument("--rounds", type=int, default=1, help="repeats per cell")
    ap.add_argument(
        "--cool-gate",
        type=float,
        default=0.0,
        help="TFLOPS a bf16 4096^3 matmul must reach before each round; 0 = off",
    )
    ap.add_argument("--cool-wait", type=float, default=600.0, help="max gate wait, s")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    for m in args.ms:
        assert args.tile_ab or args.codec_pair or ((m + 63) // 64) % 2 == 0, (
            f"M={m} outside even-ceil window"
        )
    if args.codec_pair:
        assert len(args.codecs) == 2, "--codec-pair takes exactly two --codecs"
        codec_pair_main(args, shapes)
        return

    tmp = tempfile.mkdtemp(prefix="bm128ab-")
    env_base = dict(os.environ)

    def spawn(bm128):
        env = dict(env_base)
        if args.tile_ab:
            env["KQ_NAX_TILE"] = "128,64,2,2" if bm128 != "0" else "64,64,2,2"
        elif bm128 is None:  # policy mode: leave the tri-state env unset
            env.pop("KQ_NAX_BM128", None)
        else:
            env["KQ_NAX_BM128"] = bm128
        return subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), CHILD],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )

    on = spawn(None if args.policy_arm else "1")
    off = spawn("0")

    def ask(child, req):
        child.stdin.write(json.dumps(req) + "\n")
        child.stdin.flush()
        return json.loads(child.stdout.readline())["us"]

    results = {}
    try:
        for codec in args.codecs:
            for N, K in shapes:
                npz = os.path.join(tmp, f"{codec}-{N}x{K}.npz")
                if not os.path.exists(npz):
                    import numpy as np

                    w, s = probe_layout(codec, N, K)
                    np.savez(npz, w=w, s=s)
                    wbytes = w.nbytes
                else:
                    import numpy as np

                    wbytes = np.load(npz)["w"].nbytes
                for M in args.ms:
                    req = {
                        "cmd": "cell",
                        "codec": codec,
                        "N": N,
                        "K": K,
                        "M": M,
                        "npz": npz,
                        "iters": args.iters,
                        "warmup": args.warmup,
                    }
                    a1 = ask(on, req)
                    b1 = ask(off, req)
                    b2 = ask(off, req)
                    a2 = ask(on, req)
                    us_on, us_off = min(a1, a2), min(b1, b2)
                    d = (us_off - us_on) / us_off * 100.0
                    results[f"{codec}|{N}x{K}|M{M}"] = {
                        "us_on": us_on,
                        "us_off": us_off,
                        "delta_pct": d,
                        "gbps_on": wbytes / (us_on * 1e-6) / 1e9,
                        "gbps_off": wbytes / (us_off * 1e-6) / 1e9,
                    }
                    print(
                        f"{codec:8s} {N}x{K} M{M:<5d} "
                        f"on {us_on:8.1f}us  off {us_off:8.1f}us  "
                        f"bm128 {d:+6.1f}%",
                        flush=True,
                    )
    finally:
        for c in (on, off):
            try:
                c.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                c.stdin.flush()
            except Exception:
                pass
            c.wait(timeout=30)

    print("\n| codec | shape | " + " | ".join(f"M{m}" for m in args.ms) + " |")
    print("|---|---|" + "---|" * len(args.ms))
    for codec in args.codecs:
        for N, K in shapes:
            cells = [results.get(f"{codec}|{N}x{K}|M{m}") for m in args.ms]
            row = " | ".join(f"{c['delta_pct']:+.1f}%" if c else "-" for c in cells)
            print(f"| {codec} | {N}x{K} | {row} |")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == CHILD:
        child_main()
    else:
        main()
