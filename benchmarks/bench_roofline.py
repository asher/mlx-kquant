"""Three floors per DiT cell from stock-op rates, against measured times.

Reads bench_dit_shapes.py JSON and, per cell and arm, computes:

  NAX floor    2*M*K*N over the stock matmul rate at a cube of --cube
               (per activation dtype)
  ALU floor    decoded weights N*K*ceil(M/BM), times the codec's ALU ops
               per weight, over the rate of an fp32 multiply-add chain
               written with mx.fast.metal_kernel
  bytes floor  compulsory traffic: x, w and out each moved once, over the
               copy bandwidth of a large elementwise op

and reports measured time over the largest floor, and which floor binds.
Two columns sit beside the floors without entering them:

  loads        scalar byte loads per decoded weight (Q6_K: its 210-byte
               blocks rule out vector loads), counted as issue slots at the
               ALU rate
  no reuse     x_bytes*ceil(N/BN) + w_bytes*ceil(M/BM) + out_bytes, the
               traffic with no cache reuse across tiles, an upper bound

BM and BN come from the kernel each kq arm logged; the dense and affine
arms use MLX's dispatch rule. Ops per weight are hand counts of the NAX
loaders (kq_quantized_nax.h); where a count is a range the floor takes the
low end. The chain rate counts a multiply-add as one op, as the counts do,
so the ALU floor is an estimate; --ops-per-weight overrides a count. Rates
are measured here unless --rates names a JSON a previous run wrote.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import dit_common as dc

# Per decoded weight in load_unsafe of each NAX loader: integer ops,
# conversions, float ops, table reads, plus per-call scale setup spread
# over the 32 weights a thread decodes.
OPS_PER_WEIGHT = {
    "q8_0": 4.09,
    "q5_1": 7.16,
    "q4_0": 5.09,
    "q4_1": 4.13,
    "q5_0": 8.19,
    "q4_k": 5.22,
    "q5_k": 8.3,
    "q6_k": 8.4,
    "q3_k": 8.44,
    "q2_k": 4.81,
    "iq4_nl": 5.09,
    "iq4_xs": 5.63,
    "iq3_xxs": 5.76,
    "iq3_s": 6.41,
    "iq2_xxs": 7.44,
    "iq2_xs": 5.97,
    "iq2_s": 5.85,
    "iq1_s": 6.01,
    "iq1_m": 6.95,
    "stq1_0": 9.21,
    "pq2_0": 5.25,
    "ptq1_0": 9.50,
}
# High end where a count is a range.
OPS_PER_WEIGHT_HIGH = {"q5_k": 9.3}
# Scalar byte loads per decoded weight, outside the ALU count.
SCALAR_LOADS_PER_WEIGHT = {"q6_k": 2.0}
# MLX affine: nibble or byte extract, two conversions, one multiply-add.
AFFINE_OPS = 4.0


def _median_ms(fn, reps, warmup=3):
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def measure_rates(dtypes, cube, alu_elems, alu_chain, bw_mib, reps):
    import mlx.core as mx

    rates = {"matmul_flops": {}, "cube": cube}
    for name in dtypes:
        dt = dc.mx_dtype(name)
        a = mx.random.normal((cube, cube)).astype(dt)
        b = mx.random.normal((cube, cube)).astype(dt)
        mx.eval(a, b)
        ms = _median_ms(lambda a=a, b=b: a @ b, reps)
        rates["matmul_flops"][name] = 2.0 * cube**3 / (ms * 1e-3)

    # Four independent multiply-add chains per thread in registers; the
    # rate counts one op per multiply-add.
    kernel = mx.fast.metal_kernel(
        name="roofline_alu_chain",
        input_names=["inp"],
        output_names=["out"],
        source="""
            uint i = thread_position_in_grid.x;
            float a = inp[i];
            float b = a * 0.5f;
            float c = a + 1.0f;
            float d = a - 1.0f;
            for (int k = 0; k < CHAIN; ++k) {
              a = fma(a, 0.9990234375f, 0.0009765625f);
              b = fma(b, 0.9990234375f, 0.0009765625f);
              c = fma(c, 0.9990234375f, 0.0009765625f);
              d = fma(d, 0.9990234375f, 0.0009765625f);
            }
            out[i] = a + b + c + d;
        """,
    )
    x = mx.random.uniform(shape=(alu_elems,))
    mx.eval(x)

    def run():
        return kernel(
            inputs=[x],
            template=[("CHAIN", alu_chain)],
            grid=(alu_elems, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[mx.float32],
        )[0]

    ms = _median_ms(run, reps)
    alu_chain = 4 * alu_chain
    rates["alu_ops"] = alu_elems * alu_chain / (ms * 1e-3)
    rates["alu_chain"] = alu_chain
    rates["alu_elems"] = alu_elems

    n = bw_mib * (1 << 20) // 2
    y = mx.random.normal((n,)).astype(mx.bfloat16)
    mx.eval(y)
    ms = _median_ms(lambda: mx.abs(y), reps)
    rates["copy_bytes"] = 2.0 * y.nbytes / (ms * 1e-3)
    rates["bw_mib"] = bw_mib
    return rates


def floors(M, K, N, wbits, bm, bn, ops, loads, dtype, rates):
    """Floors in ms, the binding one, and the two side columns."""
    nax = 2.0 * M * K * N / rates["matmul_flops"][dtype]
    decoded = N * K * math.ceil(M / bm)
    alu = decoded * ops / rates["alu_ops"]
    x_bytes, w_bytes, out_bytes = M * K * 2, N * K * wbits / 8, M * N * 2
    compulsory = (x_bytes + w_bytes + out_bytes) / rates["copy_bytes"]
    no_reuse = x_bytes * math.ceil(N / bn) + w_bytes * math.ceil(M / bm) + out_bytes
    fl = {"nax": nax * 1e3, "alu": alu * 1e3, "bytes": compulsory * 1e3}
    bind = max(fl, key=fl.get)
    side = {
        "loads": decoded * loads / rates["alu_ops"] * 1e3,
        "no_reuse_bytes": no_reuse / rates["copy_bytes"] * 1e3,
    }
    return fl, bind, side


def settled(rec):
    s = rec.get("settled") or rec.get("all")
    return s["median"] if s else None


def cell_rows(result, rates, ops_over, arms):
    out = []
    for p in result["passes"]:
        dtype = p["dtype"]
        for g in p["groups"]:
            M, K, N = g["M"], g["K"], g["N"]
            ref = []
            if "dense" in arms and "dense" in g["shared"]:
                ref.append(("dense", None, g["shared"]["dense"], 16.0, 0.0, 0.0))
            for c in g["cells"]:
                codec = c["codec"]
                bits = dc.affine_bits(codec)
                aff = g["shared"].get(f"affine{bits}")
                if (
                    "affine" in arms
                    and aff
                    and not any(r[0] == f"affine{bits}" for r in ref)
                ):
                    ref.append(
                        (f"affine{bits}", None, aff, bits + 0.5, AFFINE_OPS, 0.0)
                    )
                ops = ops_over.get(codec, OPS_PER_WEIGHT.get(codec))
                loads = SCALAR_LOADS_PER_WEIGHT.get(codec, 0.0)
                for arm in ("kq", "kq_bm64", "kq_bm128"):
                    rec = c["arms"].get(arm)
                    if arm in arms and rec and not rec.get("error"):
                        wb = dc.weight_bits(codec)
                        ref.append((arm, codec, rec, wb, ops, loads))
            for arm, codec, rec, wbits, ops, loads in ref:
                ms = settled(rec)
                bm, bn = dc.tile_bm_bn(rec.get("tile"))
                if ms is None or bm is None:
                    continue
                fl, bind, side = floors(
                    M, K, N, wbits, bm, bn, ops or 0.0, loads, dtype, rates
                )
                out.append(
                    {
                        "dtype": dtype,
                        "family": g["family"],
                        "op": g["op"],
                        "L": g["L"],
                        "B": g["B"],
                        "M": M,
                        "K": K,
                        "N": N,
                        "arm": arm,
                        "codec": codec,
                        "tile": rec.get("tile"),
                        "ms": ms,
                        "floors_ms": fl,
                        "side_ms": side,
                        "binding": bind,
                        "over_floor": ms / fl[bind],
                        "ops_per_weight": ops,
                        "ops_per_weight_high": OPS_PER_WEIGHT_HIGH.get(codec),
                        "scalar_loads_per_weight": loads,
                    }
                )
    return out


def markdown(rows, rates, desc):
    lines = ["# DiT roofline", "", dc.descriptor_md(desc), ""]
    mm = ", ".join(
        f"{k} {v / 1e12:.1f} TFLOPS" for k, v in rates["matmul_flops"].items()
    )
    lines.append(
        f"Rates: matmul at {rates['cube']} cube {mm}; ALU chain "
        f"{rates['alu_ops'] / 1e12:.2f} Tops/s; copy "
        f"{rates['copy_bytes'] / 1e9:.0f} GB/s."
    )
    lines.append("")
    for dtype in dict.fromkeys(r["dtype"] for r in rows):
        lines.append(f"## {dtype}")
        lines.append("")
        lines.append(
            "| op | L x B | M | arm | codec | tile | ms | NAX | ALU | bytes "
            "| binds | ms / floor | loads | no reuse |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["dtype"] != dtype:
                continue
            f, sd = r["floors_ms"], r["side_ms"]
            lines.append(
                f"| {r['family']}.{r['op']} | {r['L']} x {r['B']} | {r['M']} "
                f"| {r['arm']} | {r['codec'] or '-'} | {r['tile']} | {r['ms']:.3f} "
                f"| {f['nax']:.3f} | {f['alu']:.3f} | {f['bytes']:.3f} "
                f"| {r['binding']} | {r['over_floor']:.2f} "
                f"| {sd['loads']:.3f} | {sd['no_reuse_bytes']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="bench_dit_shapes.py JSON files")
    ap.add_argument("--rates", help="rates JSON from an earlier run")
    ap.add_argument("--rates-out")
    ap.add_argument("--cube", type=int, default=4096)
    ap.add_argument("--alu-elems", type=int, default=1 << 24)
    ap.add_argument("--alu-chain", type=int, default=1024)
    ap.add_argument("--bw-mib", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--arms", default="kq,kq_bm64,kq_bm128,affine,dense")
    ap.add_argument(
        "--ops-per-weight", default="", help="overrides, e.g. q6_k=6,q5_k=7.5"
    )
    ap.add_argument("--json-out")
    ap.add_argument("--md-out")
    args = ap.parse_args()

    results = []
    for path in args.inputs:
        with open(path) as f:
            results.append(json.load(f))
    ops_over = {
        k: float(v)
        for k, v in (kv.split("=") for kv in args.ops_per_weight.split(",") if kv)
    }
    dtypes = sorted({p["dtype"] for r in results for p in r["passes"]})
    desc = dc.machine_descriptor()
    if args.rates:
        with open(args.rates) as f:
            rates = json.load(f)["rates"]
    else:
        rates = measure_rates(
            dtypes, args.cube, args.alu_elems, args.alu_chain, args.bw_mib, args.reps
        )
        if args.rates_out:
            with open(args.rates_out, "w") as f:
                json.dump({"machine": desc, "rates": rates}, f, indent=1)
    arms = set(args.arms.split(","))
    rows = []
    for r in results:
        if r["machine"].get("metallib_sha256") != desc["metallib_sha256"]:
            print("warning: input was measured on another build", file=sys.stderr)
        rows += cell_rows(r, rates, ops_over, arms)
    out = {"machine": desc, "rates": rates, "ops_per_weight": OPS_PER_WEIGHT}
    out["ops_per_weight_high"] = OPS_PER_WEIGHT_HIGH
    out["scalar_loads_per_weight"] = SCALAR_LOADS_PER_WEIGHT
    out["ops_overrides"] = ops_over
    out["rows"] = rows
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=1)
    md = markdown(rows, rates, desc)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    sys.exit(main())
