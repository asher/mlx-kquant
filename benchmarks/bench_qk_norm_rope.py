"""kq.qk_norm_rope against the unfused ops at DiT attention shapes.

One call normalizes and rotates q and k of one attention layer, q and k
sliced from one fused qkv output as a DiT block produces them. Arms, all in
one process:

  composed     mx.fast.rms_norm, then the rotation of interleaved pairs from
               primitive ops in f32, then the transpose to [B, H, L, D]
  stock_fused  mx.fast.rms_norm, then mx.fast.rope(traditional=True). Its
               angles follow the sequence index, not per-row tables, so it
               computes a different rotation; it prices stock fused ops
  kq           kq.qk_norm_rope
  copy         a transposed copy of q and k to [B, H, L, D], the same bytes
               as kq with no math, the practical bandwidth floor

Order: one discarded warmup call per arm and shape, then rounds of 8
Thue-Morse slots. Each slot visits every shape; a visit opens with one
discarded call per arm, then times each arm, in reversed order on odd
slots. Round 0 gets its own column and later rounds the settled column.

A sample is --inner calls queued back to back with one sync at the end;
shapes whose fastest arm runs under --min-sample-ms per sample take more
calls, up to --max-inner, and every arm of a shape uses the same count.

The bandwidth floor of a shape is the bytes kq must move, q and k read
once and written once plus the tables, at --gbs GB/s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_stats as bs  # noqa: E402
import dit_common as dc  # noqa: E402

ARMS = ("composed", "stock_fused", "kq", "copy")
# (rows per item L, batch B): 1024-pixel Z-Image rows (4096 image rows plus
# 32 caption rows), 1024x1152-class rows, and 2048-pixel rows.
SHAPES = [(4128, 1), (4608, 1), (4128, 2), (4608, 2), (16416, 1)]
EPS = 1e-6


def make_inputs(L, B, H, D, dt, seed=0):
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(seed)
    key = mx.random.key(seed)
    qkv = mx.random.normal((B, L, 3 * H * D), key=key).astype(dt)
    qkv = qkv.reshape(B, L, 3, H, D)
    ang = rng.uniform(0.0, 64.0, (L, D // 2)).astype(np.float32)
    cos, sin = mx.array(np.cos(ang)), mx.array(np.sin(ang))
    w = (1.0 + 0.1 * rng.standard_normal((2, D))).astype(np.float32)
    q_w, k_w = mx.array(w[0]).astype(dt), mx.array(w[1]).astype(dt)
    mx.eval(qkv, cos, sin, q_w, k_w)
    return {
        "q": qkv[:, :, 0],
        "k": qkv[:, :, 1],
        "q_w": q_w,
        "k_w": k_w,
        "cos": cos,
        "sin": sin,
    }


def composed(q, k, q_w, k_w, cos, sin):
    import mlx.core as mx

    def rope(x):
        b, n, h, d = x.shape
        x = x.astype(mx.float32).reshape(b, n, h, d // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        c = cos[None, :, None, :]
        s = sin[None, :, None, :]
        out = mx.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
        return out.reshape(b, n, h, d)

    dt = q.dtype
    qo = rope(mx.fast.rms_norm(q, q_w, EPS)).astype(dt)
    ko = rope(mx.fast.rms_norm(k, k_w, EPS)).astype(dt)
    return qo.transpose(0, 2, 1, 3), ko.transpose(0, 2, 1, 3)


def make_fn(arm, x):
    import mlx.core as mx

    import mlx_kquant as kq

    q, k = x["q"], x["k"]
    if arm == "composed":
        return lambda: composed(q, k, x["q_w"], x["k_w"], x["cos"], x["sin"])
    if arm == "stock_fused":
        d = q.shape[-1]

        def f():
            qn = mx.fast.rms_norm(q, x["q_w"], EPS).transpose(0, 2, 1, 3)
            kn = mx.fast.rms_norm(k, x["k_w"], EPS).transpose(0, 2, 1, 3)
            return (
                mx.fast.rope(qn, d, traditional=True, base=256.0, scale=1.0, offset=0),
                mx.fast.rope(kn, d, traditional=True, base=256.0, scale=1.0, offset=0),
            )

        return f
    if arm == "kq":
        return lambda: kq.qk_norm_rope(
            q, k, x["q_w"], x["k_w"], x["cos"], x["sin"], EPS
        )
    if arm == "copy":
        return lambda: (
            mx.contiguous(q.transpose(0, 2, 1, 3)),
            mx.contiguous(k.transpose(0, 2, 1, 3)),
        )
    raise ValueError(arm)


def time_fn(fn, inner):
    """ms per call over `inner` calls queued back to back, one sync at the
    end. Each call's outputs are released once it is queued, so memory
    stays at the calls in flight."""
    import mlx.core as mx

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(inner):
        mx.async_eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3 / inner


def floor_bytes(L, B, H, D, itemsize):
    return 4 * B * L * H * D * itemsize + 2 * L * (D // 2) * 4


def run_dtype(dtype, args):
    import mlx.core as mx

    dt = dc.mx_dtype(dtype)
    want = args.shapes
    shapes = [(L, B) for (L, B) in SHAPES if not want or f"{L}x{B}" in want]
    inputs = {s: make_inputs(s[0], s[1], args.heads, args.head_dim, dt) for s in shapes}
    fns = {(s, a): make_fn(a, inputs[s]) for s in shapes for a in args.arms}

    inner = {}
    for s in shapes:
        fastest = None
        for a in args.arms:
            time_fn(fns[(s, a)], 1)
            ms = time_fn(fns[(s, a)], 3)
            fastest = ms if fastest is None else min(fastest, ms)
        n = int(args.min_sample_ms / max(fastest, 1e-3)) + 1
        inner[s] = max(1, min(args.max_inner, n))
    counts = {f"{L}x{B}": n for (L, B), n in inner.items()}
    print(f"[{dtype}] inner {counts}", flush=True)

    samples = {(s, a): [] for s in shapes for a in args.arms}
    for r in range(args.rounds):
        for slot in range(8):
            arms = list(args.arms)
            if dc.THUE_MORSE[slot]:
                arms.reverse()
            for s in shapes:
                for a in arms:
                    time_fn(fns[(s, a)], 1)
                for a in arms:
                    samples[(s, a)].append((r, slot, time_fn(fns[(s, a)], inner[s])))
        print(f"[{dtype}] round {r} done", flush=True)

    cells = []
    for s in shapes:
        L, B = s
        fb = floor_bytes(L, B, args.heads, args.head_dim, mx.array(0, dt).itemsize)
        floor_ms = fb / (args.gbs * 1e9) * 1e3
        rec = {
            "L": L,
            "B": B,
            "rows": L * B,
            "inner": inner[s],
            "floor_ms": floor_ms,
            "bytes": fb,
            "arms": {},
            "ratios": {},
        }
        for a in args.arms:
            xs = samples[(s, a)]
            rec["arms"][a] = {
                "round0": bs.summarize([ms for r, _, ms in xs if r == 0]),
                "settled": bs.summarize([ms for r, _, ms in xs if r > 0]),
                "samples": xs,
            }
        if "kq" in args.arms:
            for base in args.arms:
                if base == "kq":
                    continue
                pairs = [
                    k[2] / b[2]
                    for k, b in zip(samples[(s, "kq")], samples[(s, base)], strict=True)
                    if k[0] > 0
                ]
                rec["ratios"][f"kq/{base}"] = bs.summarize(pairs)
        cells.append(rec)
    return cells


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def report_md(res):
    a = res["args"]
    lines = [
        "# qk_norm_rope at DiT attention shapes",
        "",
        dc.descriptor_md(res["machine"]),
        "",
        f"{a['heads']} heads of {a['head_dim']}, q and k sliced from one fused qkv "
        f"output. {a['rounds']} rounds of 8 Thue-Morse slots. Round 0 is the first "
        f"column and later rounds are settled. Times are median ms per call, and "
        f"spread is the relative IQR of the settled samples. The floor is q and k "
        f"read and written once plus the tables, at {a['gbs']} GB/s. The kq ratios "
        f"to other arms are medians of settled pairs.",
        "",
    ]
    for dtype, cells in res["dtypes"].items():
        lines += [f"## {dtype}", ""]
        hdr = "| L x B | floor |"
        sep = "|---|---|"
        for arm in a["arms"]:
            hdr += f" {arm} first | {arm} settled | {arm} spread |"
            sep += "---|---|---|"
        hdr += " kq / floor | kq / composed | kq / copy |"
        sep += "---|---|---|"
        lines += [hdr, sep]
        for c in cells:
            row = f"| {c['L']} x {c['B']} | {fmt(c['floor_ms'])} |"
            for arm in a["arms"]:
                st = c["arms"][arm]
                r0 = st["round0"]["median"] if st["round0"] else None
                s = st["settled"] or {}
                iqr = s.get("rel_iqr")
                spread = "n/a" if iqr is None else fmt(100 * iqr, 1) + "%"
                row += f" {fmt(r0)} | {fmt(s.get('median'))} | {spread} |"
            kq_s = (c["arms"].get("kq", {}).get("settled") or {}).get("median")
            row += f" {fmt(kq_s / c['floor_ms'] if kq_s else None, 2)} |"
            for base in ("composed", "copy"):
                r = c["ratios"].get(f"kq/{base}")
                row += f" {fmt(r['median'] if r else None, 3)} |"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dtypes", default="bfloat16,float16")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--shapes", default="", help="L x B cells, e.g. 4128x1,16416x1")
    ap.add_argument("--heads", type=int, default=30)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-sample-ms", type=float, default=10.0)
    ap.add_argument("--max-inner", type=int, default=64)
    ap.add_argument("--gbs", type=float, default=614.0, help="bandwidth for the floor")
    ap.add_argument("--json-out", default="bench_qk_norm_rope.json")
    ap.add_argument("--md-out", default="bench_qk_norm_rope.md")
    args = ap.parse_args()
    args.arms = [x for x in args.arms.split(",") if x]
    args.shapes = [x for x in args.shapes.split(",") if x]
    for arm in args.arms:
        if arm not in ARMS:
            ap.error(f"unknown arm {arm}")

    res = {
        "args": {k: v for k, v in vars(args).items()},
        "machine": dc.machine_descriptor(),
        "dtypes": {},
    }
    t0 = time.perf_counter()
    for dtype in args.dtypes.split(","):
        res["dtypes"][dtype] = run_dtype(dtype, args)
    res["seconds"] = time.perf_counter() - t0
    with open(args.json_out, "w") as f:
        json.dump(res, f, indent=1)
    md = report_md(res)
    with open(args.md_out, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
