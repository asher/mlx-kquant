"""kq.quantized_matmul_glu against the unfused ops at DiT feed-forward shapes.

One call computes silu(x @ Wg^T) * (x @ Wu^T) for x [1, M, K] and two
packed weights [N, K] of one codec. Arms, all in one process:

  composed     two kq.quantized_matmul calls, then stock silu and multiply
  composed_aa  the same again (A versus A)
  two_mm       the two matmuls alone, both evaluated, with no activation
               and no product: the part a fused kernel cannot remove
  glu          kq.quantized_matmul_glu with no lever set
  glu:BMxBK    kq.quantized_matmul_glu on one tile, in a build with
               KQ_PROBE_KERNELS=ON only (--tiles). BK takes a suffix: w for
               the two-walk kernel, d for the one-walk kernel on two
               unpadded buffers

Where the op has no fused kernel for the codec, the dtype, the shape or
the GPU, glu is the composition, and composed against two_mm prices what a
fused kernel could save there.

--cells is a list of DTYPE:K:N:ROWS sets separated by semicolons, ROWS a
comma list of M. The default is the sweep that decides whether a codec's
fused kernel is the op's default: it is, where glu is at or below composed
in every cell.

Order: every arm of a cell runs once to build its pipelines and is compared
with composed. Then the MLX peak of one call of each arm is taken, over the
memory that is live once the buffers of the calls before it are released.
Then each round visits every arm once, in an order that rotates with the
round and reverses on odd cycles. A visit is one discarded call and --calls
timed calls, each evaluated alone; the sample is their median. Ratios pair
the arms inside a round. The JSON is rewritten after every round, with the
process footprint of that round.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time

# The tile levers are read per call.
os.environ["KQ_TUNE_LIVE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dit_common as dc  # noqa: E402

DEFAULT_CELLS = (
    "bfloat16:3840:10240:1056,2080,4128,4608,8208,16416;"
    "float16:3840:10240:1056,4128,16416;"
    "bfloat16:4096:12288:4128,16416"
)
LEVERS = ("KQ_GLU_BM", "KQ_GLU_BK", "KQ_GLU_DUAL128", "KQ_GLU_WALKS")


# struct rusage_info_v2 of <sys/resource.h>, whole: the call fills it all.
class _RUsageV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


def phys_footprint():
    """Physical footprint of this process in bytes, 0 where unknown."""
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        r = _RUsageV2()
        if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(r)) != 0:
            return 0
        return int(r.ri_phys_footprint)
    except Exception:
        return 0


def med(v):
    return statistics.median(v)


def settle(mx):
    """Active MLX memory once the buffers of finished calls are released.
    A buffer is returned when its command buffer completes, which can be
    after the call that dropped the array, so read until five readings
    10 ms apart agree."""
    mx.synchronize()
    last, same = -1, 0
    for _ in range(500):
        mx.clear_cache()
        cur = mx.get_active_memory()
        same = same + 1 if cur == last else 0
        last = cur
        if same >= 5:
            break
        time.sleep(0.01)
    return last


def tile_env(tile):
    bm, bk = tile.split("x")
    return {
        "KQ_GLU_BM": bm,
        "KQ_GLU_BK": bk.rstrip("dw"),
        "KQ_GLU_DUAL128": "1" if bk.endswith("d") else "0",
        "KQ_GLU_WALKS": "1" if bk.endswith("w") else "0",
    }


def build_arms(a, dt, K, N):
    import mlx.core as mx
    import mlx.nn as nn

    import mlx_kquant as kq

    s0 = mx.zeros((1,), dtype=mx.uint8)
    ws = []
    for i in range(2):
        wf = mx.random.normal((N, K), key=mx.random.key(11 + i)) * K**-0.5
        w, _ = kq.quantize(wf, a.codec)
        mx.eval(w)
        ws.append(w)
    wg, wu = ws

    def mm(x, w):
        return kq.quantized_matmul(x, w, s0, a.codec, True)

    def composed(x):
        y = nn.silu(mm(x, wg)) * mm(x, wu)
        mx.eval(y)
        return y

    def two_mm(x):
        g, u = mm(x, wg), mm(x, wu)
        mx.eval(g, u)
        return g

    def glu(env):
        def f(x):
            for name in LEVERS:
                os.environ.pop(name, None)
            os.environ.update(env)
            y = kq.quantized_matmul_glu(x, wg, wu, a.codec)
            mx.eval(y)
            return y

        return f

    arms = {
        "composed": composed,
        "composed_aa": composed,
        "two_mm": two_mm,
        "glu": glu({}),
    }
    for tile in a.tiles:
        arms["glu:" + tile] = glu(tile_env(tile))
    for name in a.skip:
        arms.pop(name, None)
    return arms


def run(a):
    import mlx.core as mx

    import mlx_kquant as kq

    mx.set_cache_limit(a.cache_gib << 30)
    res = {
        "machine": dc.machine_descriptor(),
        "args": {k: v for k, v in vars(a).items() if k != "out"},
        "kq_file": kq.__file__,
        "cells": [],
    }
    print(dc.descriptor_md(res["machine"]), flush=True)
    for spec in a.cells:
        dtype, K, N, rows = spec.split(":")
        K, N = int(K), int(N)
        dt = dc.mx_dtype(dtype)
        mx.clear_cache()
        arms = build_arms(a, dt, K, N)
        names = list(arms)
        for M in (int(r) for r in rows.split(",")):
            x = mx.random.normal((1, M, K), key=mx.random.key(M)) * 0.5
            x = x.astype(dt)
            mx.eval(x)
            cell = {
                "dtype": dtype,
                "K": K,
                "N": N,
                "M": M,
                "arms": names,
                "check": {},
                "peak_outputs": {},
                "samples": [],
                "rounds": [],
            }
            res["cells"].append(cell)
            ref = None
            for n in names:
                y = arms[n](x)
                if n == "composed":
                    ref = y.astype(mx.float32)
                elif n != "two_mm" and ref is not None:
                    d = mx.linalg.norm(y.astype(mx.float32) - ref)
                    cell["check"][n] = (d / mx.linalg.norm(ref)).item()
                del y
            ref = None
            # MLX peak of one call over the memory live before it, in
            # outputs. The baseline waits for the arm before to let go.
            for n in names:
                act0 = settle(mx)
                mx.reset_peak_memory()
                y = arms[n](x)
                pk = mx.get_peak_memory() - act0
                cell["peak_outputs"][n] = pk / (M * N * 2)
                del y
            mx.clear_cache()
            for rnd in range(a.rounds):
                r = rnd % len(names)
                order = names[r:] + names[:r]
                if (rnd // len(names)) % 2:
                    order = order[::-1]
                for pos, n in enumerate(order):
                    f = arms[n]
                    f(x)
                    mx.synchronize()
                    ts = []
                    for _ in range(a.calls):
                        t0 = time.perf_counter()
                        f(x)
                        ts.append(time.perf_counter() - t0)
                    cell["samples"].append(
                        {"round": rnd, "arm": n, "pos": pos, "s": med(ts)}
                    )
                fp = phys_footprint() / 2**30
                cell["rounds"].append({"round": rnd, "footprint_gib": fp})
                with open(a.out + ".json", "w") as fh:
                    json.dump(res, fh, indent=1)
                print(
                    f"[glu] {a.codec} {dtype} K {K} N {N} M {M} round {rnd} "
                    f"footprint {fp:.2f} GiB, MLX cache "
                    f"{mx.get_cache_memory() / 2**30:.2f}",
                    flush=True,
                )
            del x
            mx.clear_cache()
        del arms
    report(res, a.out + ".md")


def _ratios(cell, arm, base):
    b = {s["round"]: s["s"] for s in cell["samples"] if s["arm"] == base}
    return sorted(
        s["s"] / b[s["round"]]
        for s in cell["samples"]
        if s["arm"] == arm and s["round"] in b
    )


def report(res, path):
    a = res["args"]
    lines = [
        f"# kq.quantized_matmul_glu against the unfused ops, {a['codec']}",
        "",
        dc.descriptor_md(res["machine"]),
        "",
        f"{a['rounds']} rounds, {a['calls']} calls per visit. Ratios are "
        "medians of per-round pairs against composed; the range is the "
        "smallest and largest round. Peak is the MLX peak of one call over "
        "the memory live before it, in output buffers.",
        "",
        "| dtype | K | N | M | Arm | ms per call | Ratio to composed | Range | "
        "Rounds below 1 | Peak, outputs | Difference to composed |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    verdict = []
    for cell in res["cells"]:
        head = f"| {cell['dtype']} | {cell['K']} | {cell['N']} | {cell['M']} | "
        for n in cell["arms"]:
            t = [s["s"] for s in cell["samples"] if s["arm"] == n]
            if not t:
                continue
            row = head + f"{n} | {1e3 * med(t):.3f} | "
            r = _ratios(cell, n, "composed")
            if r and n != "composed":
                below = sum(x < 1 for x in r)
                row += f"{med(r):.3f} | {r[0]:.3f} to {r[-1]:.3f} | "
                row += f"{below}/{len(r)} | "
                if n == "glu":
                    verdict.append(med(r))
            else:
                row += " | | | "
            row += f"{cell['peak_outputs'].get(n, 0):.2f} | "
            d = cell["check"].get(n)
            row += (f"{d:.2e}" if d is not None else "") + " |"
            lines.append(row)
    lines.append("")
    aa = [med(r) for c in res["cells"] if (r := _ratios(c, "composed_aa", "composed"))]
    if verdict:
        lines.append(
            f"- glu over composed, by cell: smallest {min(verdict):.3f}, "
            f"largest {max(verdict):.3f}, {sum(v <= 1 for v in verdict)} of "
            f"{len(verdict)} cells at or below 1."
        )
    if aa:
        lines.append(
            f"- A versus A (composed_aa over composed), by cell: "
            f"{min(aa):.3f} to {max(aa):.3f}."
        )
    fp = max(
        (r["footprint_gib"] for c in res["cells"] for r in c["rounds"]),
        default=0.0,
    )
    lines.append(f"- Largest process footprint {fp:.2f} GiB.")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="path stem of the .json and .md")
    ap.add_argument("--codec", default="q4_k")
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument(
        "--tiles",
        default="",
        help="comma list of BMxBK tiles, such as 64x64,128x128d; each adds an "
        "arm glu:BMxBK",
    )
    ap.add_argument("--skip", default="", help="comma list of arms to leave out")
    ap.add_argument("--cache-gib", type=int, default=2, help="MLX cache limit")
    ap.add_argument("--report", action="store_true", help="rebuild the .md")
    a = ap.parse_args()
    if a.report:
        with open(a.out + ".json") as fh:
            report(json.load(fh), a.out + ".md")
        return
    a.cells = [c for c in a.cells.split(";") if c]
    a.tiles = [t for t in a.tiles.split(",") if t]
    if a.tiles:
        from mlx_kquant._ext import _probe_kernels

        if not _probe_kernels():
            ap.error(
                "--tiles needs a build with KQ_PROBE_KERNELS=ON: this build "
                "has one tile per codec and reads no tile lever"
            )
    a.skip = [s for s in a.skip.split(",") if s]
    run(a)


if __name__ == "__main__":
    main()
