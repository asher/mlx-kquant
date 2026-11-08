"""sdpa_qk8 against MLX's attention at a video model's self-attention size.

Arms, each over the whole sequence with the query rows in chunks, every
chunk evaluated on its own as a caller that pauses between chunks runs it:
  mlx       mx.fast.scaled_dot_product_attention on bfloat16 q, k, v
  qk8       sdpa_qk8_inputs on the whole q and k, then sdpa_qk8 per chunk
  qk8_attn  sdpa_qk8 per chunk on inputs quantized beforehand (not timed)

Timing: mx.synchronize() fences around every sample; the arms alternate in
Thue-Morse order across rounds after a warm-up round. Reports the median,
the spread (max - min over the median), the relative IQR and the 95 percent
CI of the median, and the speedup of each arm's median over mlx's. The
first chunk's relative L2 distance to mlx is printed as a sanity check.

Protocol: idle machine; GPU microbench only (no model load). Needs a GPU
with tensor-op (NAX) units.
"""

import argparse
import json
import time

import mlx.core as mx
from bench_stats import summarize

import mlx_kquant as kq

D = 128


def tm_bit(n):
    return bin(n).count("1") & 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", type=int, default=37888)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--json-out", default=None, help="write results here")
    a = ap.parse_args()
    if not kq.nax_available():
        raise SystemExit("sdpa_qk8 needs a GPU with tensor-op (NAX) units")

    mx.set_cache_limit(2 << 30)
    n, h = a.rows, a.heads
    scale = D**-0.5
    keys = mx.random.split(mx.random.key(1), 3)
    q, k, v = (mx.random.normal((1, h, n, D), key=x).astype(mx.bfloat16) for x in keys)
    mx.eval(q, k, v)
    chunks = [(r, min(r + a.chunk, n)) for r in range(0, n, a.chunk)]

    def run_mlx():
        for r0, r1 in chunks:
            mx.eval(
                mx.fast.scaled_dot_product_attention(q[:, :, r0:r1], k, v, scale=scale)
            )

    def run_qk8_chunks(ins):
        q8, qs, k8, ks = ins
        for r0, r1 in chunks:
            mx.eval(kq.sdpa_qk8(q8[:, :, r0:r1], qs[:, :, r0:r1], k8, ks, v))

    def run_qk8():
        ins = kq.sdpa_qk8_inputs(q, k, scale)
        mx.eval(*ins)
        run_qk8_chunks(ins)

    pre = kq.sdpa_qk8_inputs(q, k, scale)
    mx.eval(*pre)
    arms = {
        "mlx": run_mlx,
        "qk8": run_qk8,
        "qk8_attn": lambda: run_qk8_chunks(pre),
    }

    r0, r1 = chunks[0]
    ref = mx.fast.scaled_dot_product_attention(q[:, :, r0:r1], k, v, scale=scale)
    got = kq.sdpa_qk8(pre[0][:, :, r0:r1], pre[1][:, :, r0:r1], pre[2], pre[3], v)
    ref, got = ref.astype(mx.float32), got.astype(mx.float32)
    rel = (mx.linalg.norm((got - ref).flatten()) / mx.linalg.norm(ref.flatten())).item()
    del ref, got
    mx.clear_cache()

    def sample(fn):
        mx.synchronize()
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        return time.perf_counter() - t0

    names = list(arms)
    for nm in names:
        sample(arms[nm])
    times = {nm: [] for nm in names}
    for rnd in range(a.rounds):
        order = names[::-1] if tm_bit(rnd) else names
        for nm in order:
            times[nm].append(sample(arms[nm]))
        print(
            f"round {rnd} "
            + " ".join(f"{nm} {times[nm][-1] * 1e3:.1f} ms" for nm in names)
            + f" footprint {mx.get_active_memory() / 2**30:.2f} GiB",
            flush=True,
        )

    flop = 4.0 * n * n * D * h
    stats = {nm: summarize(ts) for nm, ts in times.items()}
    base = stats["mlx"]["median"]
    res = {
        "rows": n,
        "heads": h,
        "chunk": a.chunk,
        "rounds": a.rounds,
        "mlx_version": mx.__version__,
        "kq_version": kq.__version__,
        "first_chunk_rel_l2_vs_mlx": rel,
        "peak_active_gib": mx.get_peak_memory() / 2**30,
        "arms": {},
    }
    print(f"\nrows {n}, heads {h}, head dim {D}, chunks of {a.chunk}")
    print(f"first chunk relative L2 against mlx: {rel:.2e}\n")
    print("| arm | median ms | spread | rel IQR | CI95 half | TFLOPS | x mlx |")
    print("|---|---|---|---|---|---|---|")
    for nm in names:
        s, ts = stats[nm], times[nm]
        med = s["median"]
        spread = (max(ts) - min(ts)) / med
        res["arms"][nm] = {
            **s,
            "samples": ts,
            "spread": spread,
            "tflops": flop / med / 1e12,
            "speedup_vs_mlx": base / med,
        }
        ci = s["ci95_half"]
        print(
            f"| {nm} | {med * 1e3:.1f} | {spread * 100:.1f}% "
            f"| {s['rel_iqr'] * 100:.1f}% "
            f"| {'n/a' if ci is None else f'{ci * 100:.1f}%'} "
            f"| {flop / med / 1e12:.1f} | {base / med:.2f} |"
        )
    print(f"\npeak active {res['peak_active_gib']:.2f} GiB")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
