"""DSA lightning-indexer score GEMM A/B: simdgroup steel kernel vs
tensor-op f16 (kq_dsa_indexer_score_mma) vs int8 MXFP4-exact
(kq.dsa_indexer_scores_q).

All arms run the same op semantics on the same QAT'd operands (every arm
consumes values on the E2M1 grid, as production does after
dsa_indexer_qat). causal=False so every tile does full work and achieved
TFLOPS are comparable across P.

Arms:
  simdgroup   kq.dsa_indexer_scores with KQ_DISABLE_INDEXER_MMA=1
  mma-f16     kq.dsa_indexer_scores (tensor-op dispatch, the default)
  i8mx        kq.dsa_indexer_scores_q on dsa_indexer_qat_quant codes+scales

Timing: SUBMIT=1 -- one op per mx.eval, mx.synchronize() fences around
every rep, median of REPS after WARMUP. Requires MTL_SHADER_VALIDATION
unset. Optionally pass --npz with captured operands (keys: q [1,H,L,128],
keys [1,1,P,128], weights [1,L,H]) to bench on real data.

Protocol: idle machine; GPU microbench only (no model load).
"""

import argparse
import json
import os
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

import mlx_kquant as kq

HEADS = 64
D = 128
DEFAULT_P = (9216, 17408, 32768, 65536, 131072)


def fence():
    mx.synchronize()


def time_op(fn, warmup, reps):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    fence()
    times = []
    for _ in range(reps):
        fence()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        fence()
        times.append((time.perf_counter() - t0) * 1e3)
        del out
    return statistics.median(times), times


def make_operands(L, P, dtype, seed):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((1, HEADS, L, D)) * 0.4).astype(dtype)
    k = mx.array(rng.standard_normal((1, 1, P, D)) * 0.4).astype(dtype)
    w = mx.array(np.abs(rng.standard_normal((1, L, HEADS))) * 0.2).astype(dtype)
    return q, k, w


def qat_arms(q, k):
    """QAT both operands once; return (deq fp16 pair, codes+scales pair)."""
    B, H, L, _ = q.shape
    P = k.shape[2]
    qd = kq.dsa_indexer_qat(q.reshape(-1, D)).reshape(q.shape)
    kd = kq.dsa_indexer_qat(k.reshape(-1, D)).reshape(k.shape)
    qc, qs = kq.dsa_indexer_qat_quant(q.reshape(-1, D))
    kc, ks = kq.dsa_indexer_qat_quant(k.reshape(-1, D))
    qc = qc.reshape(q.shape)
    qs = qs.reshape(B, H, L, 4)
    kc = kc.reshape(k.shape)
    ks = ks.reshape(B, 1, P, 4)
    mx.eval(qd, kd, qc, qs, kc, ks)
    return (qd, kd), (qc, qs, kc, ks)


def bench_point(L, P, q, k, w, warmup, reps):
    (qd, kd), (qc, qs, kc, ks) = qat_arms(q, k)
    flops = 2.0 * L * P * HEADS * D
    res = {}

    os.environ["KQ_DISABLE_INDEXER_MMA"] = "1"
    try:
        ms, ts = time_op(
            lambda: kq.dsa_indexer_scores(qd, kd, w, causal=False),
            warmup,
            reps,
        )
    finally:
        os.environ.pop("KQ_DISABLE_INDEXER_MMA", None)
    res["simdgroup"] = {"ms": ms, "tflops": flops / ms / 1e9, "iters": ts}

    ms, ts = time_op(
        lambda: kq.dsa_indexer_scores(qd, kd, w, causal=False), warmup, reps
    )
    res["mma-f16"] = {"ms": ms, "tflops": flops / ms / 1e9, "iters": ts}

    ms, ts = time_op(
        lambda: kq.dsa_indexer_scores_q(qc, qs, kc, ks, w, causal=False),
        warmup,
        reps,
    )
    res["i8mx"] = {"ms": ms, "tflops": flops / ms / 1e9, "iters": ts}
    return res


def main():
    ap = argparse.ArgumentParser(
        description="DSA indexer score GEMM A/B: simdgroup vs tensor-op f16 vs i8mx"
    )
    ap.add_argument("--L", type=int, default=4096, help="query rows")
    ap.add_argument(
        "--P",
        type=int,
        nargs="+",
        default=list(DEFAULT_P),
        help="pooled key counts",
    )
    ap.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="operand dtype",
    )
    ap.add_argument("--reps", type=int, default=5, help="timed reps")
    ap.add_argument("--warmup", type=int, default=2, help="warmup reps")
    ap.add_argument("--seed", type=int, default=7, help="rng seed")
    ap.add_argument("--npz", help="captured operands npz (overrides synth)")
    ap.add_argument("--json", help="write results json here")
    args = ap.parse_args()

    if os.environ.get("MTL_SHADER_VALIDATION"):
        sys.exit("MTL_SHADER_VALIDATION is set; unset it for timing runs")

    dtype = getattr(mx, args.dtype)
    rows = []
    print(f"L={args.L} dtype={args.dtype} reps={args.reps} (median)")
    print(f"{'P':>8} {'arm':<10} {'ms':>10} {'TFLOPS':>8} {'vs simdgroup':>13}")

    for P in args.P:
        if args.npz:
            d = np.load(args.npz)
            q = mx.array(d["q"]).astype(dtype)
            k = mx.array(d["keys"][:, :, :P]).astype(dtype)
            w = mx.array(d["weights"]).astype(dtype)
            L = q.shape[2]
            if k.shape[2] < P:
                print(f"{P:>8} skipped (npz has {k.shape[2]} keys)")
                continue
        else:
            L = args.L
            q, k, w = make_operands(L, P, dtype, args.seed)

        res = bench_point(L, P, q, k, w, args.warmup, args.reps)
        base = res["simdgroup"]["ms"]
        for arm in ("simdgroup", "mma-f16", "i8mx"):
            r = res[arm]
            print(
                f"{P:>8} {arm:<10} {r['ms']:>10.2f} {r['tflops']:>8.2f} "
                f"{base / r['ms']:>12.2f}x"
            )
        rows.append({"L": L, "P": P, "arms": res})
        del q, k, w
        mx.clear_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
