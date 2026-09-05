"""KVarN decode attention A/B: fused kvarn branch vs the q8 branch vs fp16.

All arms run sdpa_decode_gqa semantics at decode width (qL 1) on the same
synthetic cache depth. The kvarn arm reads ~204 bytes per key per kv-head at
6/6 vs ~272 for q8 and 512 for fp16, so the bandwidth floor is ~0.75x q8.
On ALU-bound GPUs the staging loops set the floor instead: M3 Max measures
kvarn6 at ~1.08-1.10x q8 and ~0.80x fp16 at 16-32k, flat across bit widths
(the extraction is fully unrolled against the function-constant widths, so
the residual is loader structure, not bit math). Treat a kvarn/q8 ratio
well above ~1.1 as a loader regression.

Arms:
  fp16     sdpa_decode_gqa on materialized fp16 KV
  q8       sdpa_decode_gqa on mlx affine wire (bits 8, group 64)
  kvarn    sdpa_decode_gqa_kvarn on records + stage

Timing: one op per mx.eval, mx.synchronize() fences around every rep,
median of REPS after WARMUP, arms interleaved ABBA per depth.

Protocol: idle machine; GPU microbench only (no model load).

--head-dim 256 runs the two-slice record path (qwen3.5/3.6 geometry); the
tile stages C=16 there, so compare against q8/fp16 at the same head dim.
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
import numpy as np

import mlx_kquant as kq


def quantize_head(x, bits, kind):
    """Flat slice-minor records for any supported head dim (kq wire)."""
    b, h, t, d = x.shape
    sl = d // 128
    if sl == 1:
        return kq.kvarn_quantize(x, bits, kind)
    xs = x.reshape(b, h, t, sl, 128).transpose(0, 1, 3, 2, 4)
    c, a = kq.kvarn_quantize(xs, bits, kind)
    g = c.shape[3]
    c = c.transpose(0, 1, 3, 2, 4).reshape(b, h, g, sl * 512 * bits)
    a = a.transpose(0, 1, 3, 2, 4, 5).reshape(b, h, g, 3 * sl, 128)
    return c, a


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
    return times


def affine_q8(x):
    """mlx affine wire (bits 8, group 64) for the q8 arm."""
    w, scales, biases = mx.quantize(x, group_size=64, bits=8)
    return w.view(mx.uint32), scales, biases


def build(n, h, seed, d):
    rng = np.random.default_rng(seed)
    kx = mx.array(rng.standard_normal((1, h, n, d)).astype(np.float16))
    vx = mx.array(rng.standard_normal((1, h, n, d)).astype(np.float16))
    mx.eval(kx, vx)
    return kx, vx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4096,16384,32768")
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--gqa", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128, choices=(128, 256, 512))
    ap.add_argument("--k-bits", type=int, default=6)
    ap.add_argument("--v-bits", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    h = args.kv_heads
    hq = h * args.gqa
    d = args.head_dim
    scale = d**-0.5
    rng = np.random.default_rng(1)
    q = mx.array(rng.standard_normal((1, hq, 1, d)).astype(np.float16))
    mx.eval(q)

    results = {}
    for n in [int(x) for x in args.depths.split(",")]:
        kx, vx = build(n, h, n, d)

        # kvarn cache state: sink group + sealed records + live remainder.
        sink = 128
        n_recs = max(0, (n - sink) // 128)
        live0 = sink + n_recs * 128
        ck, ak = quantize_head(kx[:, :, sink:live0], args.k_bits, "k")
        cv, av = quantize_head(vx[:, :, sink:live0], args.v_bits, "v")
        stage_k = mx.concatenate(
            [kx[:, :, :sink], mx.zeros((1, h, 128, d), dtype=mx.float16)], axis=2
        )
        stage_v = mx.concatenate(
            [vx[:, :, :sink], mx.zeros((1, h, 128, d), dtype=mx.float16)], axis=2
        )
        live_len = n - live0
        if live_len:
            stage_k[:, :, 128 : 128 + live_len] = kx[:, :, live0:]
            stage_v[:, :, 128 : 128 + live_len] = vx[:, :, live0:]
        kw, ks, kb = affine_q8(kx)
        vw, vs, vb = affine_q8(vx)
        mx.eval(ck, ak, cv, av, stage_k, stage_v, kw, ks, kb, vw, vs, vb)

        arms = {
            "fp16": lambda kx=kx, vx=vx: kq.sdpa_decode_gqa(q, kx, vx, scale),
            "q8": lambda kw=kw, vw=vw, ks=ks, kb=kb, vs=vs, vb=vb: kq.sdpa_decode_gqa(
                q,
                kw,
                vw,
                scale,
                k_scales=ks,
                k_biases=kb,
                v_scales=vs,
                v_biases=vb,
            ),
            "kvarn": lambda ck=ck, ak=ak, cv=cv, av=av, sk=stage_k, sv=stage_v, n=n: (
                kq.sdpa_decode_gqa_kvarn(
                    q,
                    ck,
                    ak,
                    cv,
                    av,
                    sk,
                    sv,
                    n,
                    scale,
                    args.k_bits,
                    args.v_bits,
                )
            ),
        }

        # ABBA: warm all arms once, then interleave timed halves.
        halves = {name: [] for name in arms}
        order = list(arms) + list(reversed(arms))
        for name in order:
            halves[name] += time_op(arms[name], args.warmup, max(1, args.reps // 2))
        row = {name: statistics.median(ts) for name, ts in halves.items()}
        row["kvarn_vs_q8"] = row["kvarn"] / row["q8"]
        row["kvarn_vs_fp16"] = row["kvarn"] / row["fp16"]
        results[n] = row
        if not args.json:
            print(
                f"n={n:6d}  fp16 {row['fp16']:7.3f} ms  q8 {row['q8']:7.3f} ms"
                f"  kvarn {row['kvarn']:7.3f} ms"
                f"  kvarn/q8 {row['kvarn_vs_q8']:.3f}"
                f"  kvarn/fp16 {row['kvarn_vs_fp16']:.3f}"
            )

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
