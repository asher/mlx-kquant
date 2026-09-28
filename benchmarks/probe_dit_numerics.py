"""Numerics probes against float64, with MLX_ENABLE_TF32 at 0 and at 1.

MLX reads MLX_ENABLE_TF32 once per process, so the parent runs one child
per setting and compares them.

  reduction  K 12288: each row of x holds one large term, then small terms.
             Arms: stock matmul in f32, bf16 and fp16; MLX affine 4-bit
             in f32 and bf16; kq q8_0 and q4_k in bf16 and fp16 (kq casts
             f32 activations to bf16, so it has no f32 arm).
  attention  16896 keys at head dim 128 whose scores are near uniform.
             Arms: mx.fast.scaled_dot_product_attention and the composed
             softmax(q k^T) v, in f32, bf16 and fp16.

References use float64 over the inputs as the arm sees them (rounded to
its dtype; kq and affine weights dequantized exactly). Each arm reports
relative Frobenius error, max abs error, the error of rounding the exact
result to the output dtype (the floor), and whether the output bits match
between the two settings. Markdown and JSON out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import dit_common as dc

CHILD = "--child"


def child(args):
    import mlx.core as mx
    import numpy as np

    import mlx_kquant as kq

    out = []

    def f64(a):
        return np.array(a.astype(mx.float32)).astype(np.float64)

    def record(probe, arm, dtype, y, ref):
        mx.eval(y)
        got = f64(y)
        err = got - ref
        rounded = f64(mx.array(ref.astype(np.float32)).astype(y.dtype))
        out.append(
            {
                "probe": probe,
                "arm": arm,
                "dtype": dtype,
                "rel": float(np.linalg.norm(err) / np.linalg.norm(ref)),
                "max_abs": float(np.abs(err).max()),
                "floor": float(np.linalg.norm(rounded - ref) / np.linalg.norm(ref)),
                "sha": hashlib.sha256(np.array(y.view(mx.uint8)).tobytes()).hexdigest(),
            }
        )

    # Reduction over K with one large leading term.
    M, K, N = args.m, args.k, args.n
    x = mx.random.normal((M, K), key=mx.random.key(1)) * 0.01
    x[:, 0] = 1024.0
    w = mx.random.normal((N, K), key=mx.random.key(2)) * 0.05
    mx.eval(x, w)
    for name in ("float32", "bfloat16", "float16"):
        dt = dc.mx_dtype(name)
        xd, wd = x.astype(dt), w.astype(dt)
        record("reduction", "matmul", name, xd @ wd.T, f64(xd) @ f64(wd).T)
    for name in ("float32", "bfloat16"):
        dt = dc.mx_dtype(name)
        xd, wd = x.astype(dt), w.astype(dt)
        wq, sc, bi = mx.quantize(wd, group_size=64, bits=4)
        deq = mx.dequantize(wq, sc, bi, group_size=64, bits=4).astype(mx.float32)
        y = mx.quantized_matmul(xd, wq, sc, bi, transpose=True, group_size=64, bits=4)
        record("reduction", "affine4", name, y, f64(xd) @ f64(deq).T)
    for codec in ("q8_0", "q4_k"):
        wq, s = kq.quantize(w, codec)
        deq = kq.dequantize(wq, s, codec, dtype=mx.float32)
        for name in ("bfloat16", "float16"):
            xd = x.astype(dc.mx_dtype(name))
            y = kq.quantized_matmul(xd, wq, s, codec, transpose=True)
            record("reduction", f"kq {codec}", name, y, f64(xd) @ f64(deq).T)

    # Attention over near-uniform keys.
    H, Lq, Lk, D = args.heads, args.lq, args.lk, 128
    q = mx.random.normal((1, H, Lq, D), key=mx.random.key(3))
    base = mx.random.normal((1, H, 1, D), key=mx.random.key(4))
    k = base + 1e-3 * mx.random.normal((1, H, Lk, D), key=mx.random.key(5))
    v = mx.random.normal((1, H, Lk, D), key=mx.random.key(6))
    mx.eval(q, k, v)
    scale = D**-0.5
    for name in ("float32", "bfloat16", "float16"):
        dt = dc.mx_dtype(name)
        qd, kd, vd = q.astype(dt), k.astype(dt), v.astype(dt)
        qn, kn, vn = f64(qd), f64(kd), f64(vd)
        sc = (qn @ np.swapaxes(kn, -1, -2)) * scale
        sc -= sc.max(axis=-1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(axis=-1, keepdims=True)
        ref = p @ vn
        y = mx.fast.scaled_dot_product_attention(qd, kd, vd, scale=scale)
        record("attention", "sdpa", name, y, ref)
        s2 = (qd * scale) @ mx.swapaxes(kd, -1, -2)
        y2 = mx.softmax(s2, axis=-1, precise=True) @ vd
        record("attention", "composed", name, y2, ref)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--k", type=int, default=12288)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--lq", type=int, default=128)
    ap.add_argument("--lk", type=int, default=16896)
    ap.add_argument(CHILD, action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    args = ap.parse_args()
    if args.child:
        child(args)
        return

    argv = [a for a in sys.argv[1:] if a != CHILD]
    runs = {}
    for tf32 in ("0", "1"):
        env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
        env["MLX_ENABLE_TF32"] = tf32
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), CHILD, *argv],
            env=env,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            sys.exit(r.stderr[-4000:])
        runs[tf32] = json.loads(r.stdout.strip().splitlines()[-1])
    desc = dc.machine_descriptor()
    rows = []
    for a, b in zip(runs["0"], runs["1"], strict=True):
        rows.append(
            {
                "probe": a["probe"],
                "arm": a["arm"],
                "dtype": a["dtype"],
                "tf32_0": {k: a[k] for k in ("rel", "max_abs")},
                "tf32_1": {k: b[k] for k in ("rel", "max_abs")},
                "floor": a["floor"],
                "bits_equal": a["sha"] == b["sha"],
            }
        )
    result = {"machine": desc, "args": vars(args), "rows": rows}
    lines = ["# DiT numerics probes", "", dc.descriptor_md(desc), ""]
    lines.append(
        f"Reduction M {args.m}, K {args.k}, N {args.n}; attention {args.heads} "
        f"heads, {args.lq} queries, {args.lk} keys, head dim 128. rel is the "
        "relative Frobenius error against float64; floor is the error of "
        "rounding the exact result to the output dtype."
    )
    lines.append("")
    lines.append(
        "| probe | arm | dtype | rel TF32 0 | rel TF32 1 | max abs TF32 0 "
        "| max abs TF32 1 | floor | bits equal |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['probe']} | {r['arm']} | {r['dtype']} | {r['tf32_0']['rel']:.2e} "
            f"| {r['tf32_1']['rel']:.2e} | {r['tf32_0']['max_abs']:.2e} "
            f"| {r['tf32_1']['max_abs']:.2e} | {r['floor']:.2e} "
            f"| {'yes' if r['bits_equal'] else 'no'} |"
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
