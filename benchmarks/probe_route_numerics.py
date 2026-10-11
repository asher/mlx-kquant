"""Where the quantized matmul routes round, against float64, per M band.

Every route can be modeled one of two ways:

  float weights    decode to f32 in registers, accumulate in f32, round
                   the output once (qmv, mv_ext). Its reference R1 uses
                   the exact dequantized weights.
  dtype weights    decode into the activation dtype for the matrix units,
                   accumulate in f32, round the output once (the qmm
                   tiles). Its reference R2 uses the dequantized weights
                   rounded to the activation dtype.

A route that matches one model lands on that reference's floor, the error
of rounding the reference itself to the output dtype, and its excess over
the floor is near 1e-7. A positive excess on both models means a rounding
the models do not have, such as a K slice stored in the activation dtype
before the fold. The excess is sqrt(R^2 - floor^2).

Inputs are nonnegative, so partial sums only grow. In the lead design
each row of x opens with one large term (1024), so the first K slice holds
most of the output and a rounding of that slice shows fully; the flat
design has no large term. Arms per cell: stock matmul and MLX affine
4-bit in the activation dtype, kq per codec on the default route, and kq
per forced route (KQ_QMM_ROUTE through M 32, the plain tile through
KQ_QMM_MIDM=0 from M 33 to 128). A forced route that cannot serve the cell
is left out. Markdown and JSON out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import dit_common as dc

ROUTES = [
    "qmv",
    "verify_qmv",
    "mv_ext",
    "verify_mma",
    "verify_nax",
    "splitk",
    "nax_splitk",
    "nax",
]
DEFAULT_CELLS = (
    # K sweep at a mid-M split band row count and at a DiT prefill row count.
    "64:1024:256,64:3072:256,64:12288:256,"
    "4096:1024:256,4096:3072:256,4096:12288:256,"
    # The small-M bands at one K.
    "2:12288:256,8:12288:256,16:12288:256,32:12288:256,48:12288:256,"
    "96:12288:256,128:12288:256,129:12288:256,"
    # DiT modulation widths at one and two rows.
    "1:256:15360,2:256:15360,1:4096:16384,2:4096:16384"
)


def probe(args):
    import mlx.core as mx
    import numpy as np

    import mlx_kquant as kq

    dt = dc.mx_dtype(args.dtype)
    os.environ["KQ_QMM_ROUTE_STRICT"] = "1"

    def f64(a):
        return np.array(a.astype(mx.float32)).astype(np.float64)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    def rounded(ref):
        return f64(mx.array(ref.astype(np.float32)).astype(dt))

    def run(fn, env):
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            with dc.KernelLog() as log:
                y = fn()
                mx.eval(y)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return y, (log.kernels[-1] if log.kernels else None)

    out = []
    for design in args.designs.split(","):
        for spec in args.cells.split(","):
            M, K, N = (int(v) for v in spec.split(":"))
            w = mx.abs(mx.random.normal((N, K), key=mx.random.key(2))) * 0.05
            x = mx.abs(mx.random.normal((M, K), key=mx.random.key(1))) * 0.01
            if design == "lead":
                x[:, 0] = 1024.0
            xd = x.astype(dt)
            mx.eval(xd, w)
            xn = f64(xd)
            wd = w.astype(dt)
            arms = [
                ("matmul", "-", lambda xd=xd, wd=wd: xd @ wd.T, f64(wd), f64(wd), {})
            ]
            wq, sc, bi = mx.quantize(wd, group_size=64, bits=4)
            deq = mx.dequantize(wq, sc, bi, group_size=64, bits=4)
            arms.append(
                (
                    "affine4",
                    "-",
                    lambda xd=xd, wq=wq, sc=sc, bi=bi: mx.quantized_matmul(
                        xd, wq, sc, bi, transpose=True, group_size=64, bits=4
                    ),
                    f64(deq.astype(mx.float32)),
                    f64(deq.astype(dt)),
                    {},
                )
            )
            for codec in args.codecs.split(","):
                kw, ks = kq.quantize(w, codec)
                kdeq = kq.dequantize(kw, ks, codec, dtype=mx.float32)
                w1, w2 = f64(kdeq), f64(kdeq.astype(dt))

                def fn(xd=xd, kw=kw, ks=ks, codec=codec):
                    return kq.quantized_matmul(xd, kw, ks, codec, transpose=True)

                arms.append((f"kq {codec}", "default", fn, w1, w2, {}))
                if M <= 32:
                    for route in ROUTES:
                        arms.append(
                            (f"kq {codec}", route, fn, w1, w2, {"KQ_QMM_ROUTE": route})
                        )
                elif M <= 128:
                    arms.append(
                        (f"kq {codec}", "tile", fn, w1, w2, {"KQ_QMM_MIDM": "0"})
                    )
            refs = {}
            for arm, route, fn, w1, w2, env in arms:
                try:
                    y, kname = run(fn, env)
                except RuntimeError as e:
                    # A forced route that cannot serve the cell raises under
                    # KQ_QMM_ROUTE_STRICT; any other error, such as a
                    # missing kernel, is a failure.
                    if "does not serve" not in str(e):
                        raise
                    continue
                got = f64(y)
                row = {
                    "design": design,
                    "M": M,
                    "K": K,
                    "N": N,
                    "arm": arm,
                    "route": route,
                    "kernel": kname,
                }
                for tag, wref in (("1", w1), ("2", w2)):
                    key = (id(wref), tag)
                    if key not in refs:
                        ref = xn @ wref.T
                        refs[key] = (ref, rel(rounded(ref), ref))
                    ref, floor = refs[key]
                    r = rel(got, ref)
                    row[f"r{tag}"] = r
                    row[f"floor{tag}"] = floor
                    row[f"excess{tag}"] = float(np.sqrt(max(0.0, r * r - floor**2)))
                out.append(row)
    return out


def markdown(rows, args):
    lines = [
        "# Route numerics",
        "",
        f"Activation dtype {args.dtype}. R1 and R2 are relative Frobenius errors "
        "against float64 with exact and with dtype-rounded dequantized weights; "
        "floor is the error of rounding that reference to the output dtype; "
        "excess is sqrt(R^2 - floor^2). The route column is `-` for the stock "
        "arms and names the forced route for kq.",
        "",
        "| design | M | K | N | arm | route | R1 | excess 1 | R2 | excess 2 | kernel |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        kname = r["kernel"] or "-"
        lines.append(
            f"| {r['design']} | {r['M']} | {r['K']} | {r['N']} | {r['arm']} "
            f"| {r['route']} | {r['r1']:.2e} | {r['excess1']:.1e} "
            f"| {r['r2']:.2e} | {r['excess2']:.1e} | {kname} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cells", default=DEFAULT_CELLS, help="M:K:N list")
    ap.add_argument("--designs", default="lead,flat")
    ap.add_argument("--codecs", default="q8_0,q4_k")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    args = ap.parse_args()
    rows = probe(args)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"machine": dc.machine_descriptor(), "rows": rows}, f)
    md = dc.descriptor_md(dc.machine_descriptor()) + "\n\n" + markdown(rows, args)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
