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

    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        codec, N, K, M = req["codec"], req["N"], req["K"], req["M"]
        w, s = get_weights(codec, N, K, req["npz"])
        x = mx.random.normal((M, K), key=mx.random.key(M)).astype(mx.bfloat16)
        mx.eval(x)

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
    """Synthesize (w, scales) in the codec's quantized layout: IQ codecs
    via the tests' synthetic-wire helper (their encoder demands an
    imatrix), encodable codecs by random-filling the shape of a probe
    quantize."""
    import numpy as np

    if codec.startswith("iq"):
        tests_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests",
        )
        sys.path.insert(0, tests_dir)
        from test_codecs import CODECS, _synth_iq_wire

        _, wpb, bpb, _, _ = CODECS[codec]
        rng = np.random.default_rng(N + K)
        wire = _synth_iq_wire(rng, bpb, N * (K // wpb))
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
    ap.add_argument("--json-out")
    args = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    for m in args.ms:
        assert ((m + 63) // 64) % 2 == 0, f"M={m} outside even-ceil window"

    tmp = tempfile.mkdtemp(prefix="bm128ab-")
    env_base = dict(os.environ)

    def spawn(bm128):
        env = dict(env_base)
        if bm128 is None:  # policy mode: leave the tri-state env unset
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
