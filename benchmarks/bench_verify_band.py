"""Verify-band row-scaling bench: matmul groups and full forward vs M.

Times one decoder layer's MLP, the lm_head, and (with --full) the full
forward against a warm cache, at row counts M = 1..32. A near-flat
curve means verify rows ride the weight read. A linear curve shows the
speculative verify defect.

Use one process per kernel-env config. Most KQ_* levers latch at first
dispatch. Label each run with CFG:

    CFG=default python benchmarks/bench_verify_band.py --model m.gguf
    CFG=splitk16 KQ_QMM_SPLITK=16 python benchmarks/bench_verify_band.py ...
    CFG=nax_splitk KQ_QMM_SPLITK_NAX=1 python benchmarks/bench_verify_band.py ...

Requires gmlx in the environment (loads the model through the gmlx
loader so weights come in as real kquant wire tensors).
"""

import argparse
import os
import time

import mlx.core as mx
from gmlx.loader import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="GGUF path or gmlx model name")
parser.add_argument("--rows", default="1,2,3,4,6,8,12,16,17,24,32")
parser.add_argument("--reps", type=int, default=10)
parser.add_argument(
    "--depth", type=int, default=320, help="cache depth for the full-forward sweep"
)
parser.add_argument(
    "--full", action="store_true", help="also run the full-forward sweep (slower)"
)
args = parser.parse_args()

ROWS = [int(r) for r in args.rows.split(",")]
label = os.environ.get("CFG", "default")

model, config, tok = load_model(args.model, verbose=False)
lm = getattr(model, "language_model", model)
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
inner = lm.model if hasattr(lm, "model") else lm
layer = inner.layers[0]
hidden = layer.input_layernorm.weight.shape[0]
head = getattr(lm, "lm_head", None) or inner.embed_tokens.as_linear


def sweep(name, fn):
    out = []
    for n in ROWS:
        x = mx.random.normal((1, n, hidden)).astype(mx.float16)
        for _ in range(3):
            mx.eval(fn(x))
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            mx.eval(fn(x))
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        out.append(f"M{n}={ts[len(ts) // 2]:.2f}")
    print(f"[{label}] {name:>12}: " + "  ".join(out))


sweep("mlp", lambda x: layer.mlp(x))
sweep("lm_head", lambda x: head(x))

if args.full:
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 60)
    ids = ids[: args.depth]

    def build_cache():
        cache = lm.make_cache() if hasattr(lm, "make_cache") else model.make_cache()
        for i in range(0, len(ids), 128):
            out = lm(mx.array([ids[i : i + 128]]), cache=cache)
            mx.eval(out if isinstance(out, mx.array) else out[0])
        return cache

    cache = build_cache()
    out = []
    for n in ROWS:
        x = mx.array([[ids[-1]] * n])
        for _ in range(2):
            o = lm(x, cache=cache)
            mx.eval(o if isinstance(o, mx.array) else o[0])
            for c in cache:
                c.trim(n)
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            o = lm(x, cache=cache)
            mx.eval(o if isinstance(o, mx.array) else o[0])
            ts.append((time.perf_counter() - t0) * 1e3)
            for c in cache:
                c.trim(n)
        ts.sort()
        out.append(f"M{n}={ts[len(ts) // 2]:.1f}")
    print(f"[{label}] full@{args.depth:>5}: " + "  ".join(out))
