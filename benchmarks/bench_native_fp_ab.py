"""Native-fp MoE A/B: stock packed-layout kernels vs kq wire-layout kernels.

Decode unit is the fused-path decomposition gguf-mlx actually runs:
  gate+up   two gather matmuls + SiLU*mul (stock / wire), or one
            kq.moe_glu_gather dispatch (packed fused, mxfp4 only)
  down      one gathered matvec (stock / wire), or kq.gather_qmv_bias
            (packed fused)
Prefill unit is a single expert-sorted gather GEMM (gate proj shape),
matching SwitchGLU's >=64-token sorted path.

Arms:
  gpu-packed        mx.gather_qmm(mode=<codec>) on the MLX packed layout
  gpu-packed-fused  kq.moe_glu_gather / kq.gather_qmv_bias (decode, mxfp4)
  gpu-wire          kq.gather_qmm(<codec>) on GGUF wire bytes
  gpu-wire-fused    kq.moe_glu_gather_kq / kq.gather_qmv_kq on wire bytes
                    (-bias rows: swiglu_clamp + expert biases, the gpt-oss
                    decode unit; needs a kq build with codec_has_moe_glu)
  cpu-wire          kq.gather_qmm(<codec>) under mx.stream(mx.cpu)
  gpu-wire-q4_k     same-geometry k-quant wire reference

Wire arms for mxfp4/nvfp4 skip automatically on builds where the codec is
not in the kq registry. Weights are resident (kernel bench, not SSD
streaming). Protocol: idle machine, median of REPS after WARMUP.
"""

import argparse
import platform
import statistics
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

from mlx_kquant import _ext as kq

GPTOSS_GGUF = (
    "/Users/asher/llm/gguf/lmstudio-community__gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
)

DECODE_M = (1, 2, 4)
PREFILL_M = (512, 2048, 8192)
CPU_PREFILL_CAP = 512
WARMUP = 3
DECODE_REPS = 50
PREFILL_REPS = 5
CPU_DECODE_REPS = 20

# codec -> (group_size, bits, wire bytes/block, values/block)
GEOMETRY = {"mxfp4": (32, 4, 17, 32), "nvfp4": (16, 4, 36, 64)}


def synth_wire(rng, codec, e, n, k):
    """Random wire bytes with scale bytes pinned near 1.0 so dequantized
    magnitudes stay sane (e8m0 / ue4m3 exponents are wide)."""
    _, _, bpb, vpb = GEOMETRY[codec]
    blocks = k // vpb
    raw = rng.integers(0, 256, (e, n, blocks, bpb), dtype=np.uint8)
    if codec == "mxfp4":
        raw[..., 0] = rng.integers(121, 130, (e, n, blocks), dtype=np.uint8)
    else:
        raw[..., :4] = rng.integers(48, 64, (e, n, blocks, 4), dtype=np.uint8)
    return raw.reshape(e, n, blocks * bpb)


def _repack_groups_sequential(data):
    # mirrors gguf_mlx.native_fp._repack_groups_sequential
    *lead, ngrp, half = data.shape
    v = np.concatenate([data & 0x0F, data >> 4], axis=-1)
    out = np.ascontiguousarray((v[..., 0::2] | (v[..., 1::2] << 4)).astype(np.uint8))
    return out.reshape(-1, half).view(np.uint32).reshape(*lead, ngrp * (half // 4))


def deinterleave(codec, wire):
    """GGUF wire bytes -> (packed uint32, scales uint8), the MLX native
    layout; mirrors gguf_mlx.native_fp.{mxfp4,nvfp4}_deinterleave."""
    *lead, last = wire.shape
    if codec == "mxfp4":
        blocks = wire.reshape(*lead, last // 17, 17)
        return _repack_groups_sequential(blocks[..., 1:]), np.ascontiguousarray(
            blocks[..., 0]
        )
    blocks = wire.reshape(*lead, last // 36, 36)
    scales = np.ascontiguousarray(blocks[..., :4]).reshape(*lead, -1)
    data = blocks[..., 4:].reshape(*lead, blocks.shape[-2] * 4, 8)
    return _repack_groups_sequential(data), scales


def wire_bytes_per_row(codec, k):
    _, _, bpb, vpb = GEOMETRY[codec]
    return k // vpb * bpb


def codec_supported(codec, device="gpu"):
    """Probe in a subprocess: a missing Metal kernel aborts the process
    (kernel-load errors throw on the Metal worker thread and terminate), so
    an in-process try/except cannot catch the GPU case."""
    wpr = wire_bytes_per_row(codec, 64)
    code = (
        "import mlx.core as mx, numpy as np\n"
        "from mlx_kquant import _ext as kq\n"
        f"with mx.stream(mx.{device}):\n"
        f"    w = mx.array(np.zeros((2, 32, {wpr}), np.uint8))\n"
        "    x = mx.ones((1, 1, 1, 64), dtype=mx.float16)\n"
        "    ids = mx.array(np.zeros((1, 1), np.uint32))\n"
        f"    mx.eval(kq.gather_qmm(x, w, mx.zeros((1,), dtype=mx.uint8),"
        f" {codec!r}, rhs_indices=ids))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    return r.returncode == 0


def timed(fn, reps, warmup=WARMUP):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def report(label, secs, bytes_per_call):
    print(
        f"  {label:34s} {secs * 1e6:10.0f} us  {bytes_per_call / secs / 1e9:8.1f} GB/s"
    )


def gptoss_shape():
    try:
        from gguf import GGUFReader

        r = GGUFReader(GPTOSS_GGUF)
        tensors = {t.name: t for t in r.tensors}
        fields = {f.name: f for f in r.fields.values()}
        topk = (
            int(fields["gpt-oss.expert_used_count"].parts[-1][0])
            if "gpt-oss.expert_used_count" in fields
            else 4
        )
        t = (
            tensors.get("blk.0.ffn_gate_exps.weight")
            or tensors["blk.0.ffn_gate_up_exps.weight"]
        )
        shp = t.shape.tolist()  # gguf order: [in, out(*2 if fused), experts]
        e, n, k = shp[2], shp[1], shp[0]
        if "gate_up" in t.name:
            n //= 2
        return e, n, k, topk
    except Exception as exc:
        print(f"[gpt-oss] header read failed ({exc}); using 32x2880x2880 top-4")
        return 32, 2880, 2880, 4


def bench_shape(tag, codec, e, n_gateup, k, topk, ref_codec):
    print(f"\n== {tag}: {codec} E={e} gate/up {n_gateup}x{k} top-{topk} ==")
    rng = np.random.default_rng(0)
    n_down, k_down = k, n_gateup

    ref_geom = {
        "q4_k": (256, 144),
        "q8_0": (32, 34),
        "q6_k": (256, 210),
        "iq4_nl": (32, 18),
    }
    ref_vpb, ref_bpb = ref_geom[ref_codec]
    ref_ok = k % ref_vpb == 0 and k_down % ref_vpb == 0
    if not ref_ok:
        print(f"  [skip] ref arm: K={k} not divisible by {ref_codec} block")

    arms = {}
    for name, (nn_, kk) in {
        "gate": (n_gateup, k),
        "up": (n_gateup, k),
        "down": (n_down, k_down),
    }.items():
        wire = synth_wire(rng, codec, e, nn_, kk)
        packed, scales = deinterleave(codec, wire)
        arms[name] = {
            "wire": mx.array(wire),
            "packed": mx.array(packed),
            "scales": mx.array(scales),
            "bias": mx.zeros((e, nn_), dtype=mx.float16),
            "ref_wire": mx.array(
                rng.integers(0, 256, (e, nn_, kk // ref_vpb * ref_bpb), dtype=np.uint8)
            )
            if kk % ref_vpb == 0
            else None,
        }
    placeholder = mx.zeros((1,), dtype=mx.uint8)
    mx.eval([v for a in arms.values() for v in a.values() if v is not None])
    gs, bits, _, _ = GEOMETRY[codec]
    wire_gpu = codec_supported(codec, "gpu")
    wire_cpu = codec_supported(codec, "cpu")
    if not (wire_gpu and wire_cpu):
        print(f"  [skip] wire arms: {codec!r} gpu={wire_gpu} cpu={wire_cpu}")
    wire_fused = wire_gpu and getattr(kq, "codec_has_moe_glu", lambda c: False)(codec)

    gu_bytes_wire = 2 * n_gateup * wire_bytes_per_row(codec, k)
    gu_bytes_packed = 2 * n_gateup * (k // 8 * 4 + k // gs)
    down_bytes_wire = n_down * wire_bytes_per_row(codec, k_down)
    down_bytes_packed = n_down * (k_down // 8 * 4 + k_down // gs)

    for m in DECODE_M:
        rows = m * topk
        print(f"-- decode M={m} (gate+up unit, then down unit) --")
        x = mx.array(rng.standard_normal((m, 1, 1, k)).astype(np.float16))
        xd = mx.array(rng.standard_normal((m, topk, 1, k_down)).astype(np.float16))
        x2d = x.reshape(m, k)
        xqv = xd.reshape(m, topk, k_down)
        ids = mx.array(rng.integers(0, e, (m, topk), dtype=np.uint32))
        mx.eval(x, xd, x2d, xqv, ids)

        def stock(a, xx, kk_, ids=ids):
            return mx.gather_qmm(
                xx,
                a["packed"],
                a["scales"],
                rhs_indices=ids,
                transpose=True,
                group_size=gs,
                bits=bits,
                mode=codec,
            )

        def stock_gateup(x=x):
            g = stock(arms["gate"], x, k)
            u = stock(arms["up"], x, k)
            return nn_silu_mul(g, u)

        report(
            "gpu-packed gate+up",
            timed(stock_gateup, DECODE_REPS),
            rows * gu_bytes_packed,
        )
        if codec == "mxfp4":

            def fused_gateup(x2d=x2d, ids=ids):
                return kq.moe_glu_gather(
                    x2d,
                    arms["gate"]["packed"],
                    arms["gate"]["scales"],
                    arms["gate"]["bias"],
                    arms["up"]["packed"],
                    arms["up"]["scales"],
                    arms["up"]["bias"],
                    ids,
                )

            report(
                "gpu-packed-fused gate+up",
                timed(fused_gateup, DECODE_REPS),
                rows * gu_bytes_packed,
            )
        if wire_fused:

            def wf_gateup(x2d=x2d, ids=ids):
                return kq.moe_glu_gather_kq(
                    x2d, arms["gate"]["wire"], arms["up"]["wire"], codec, ids
                )

            report(
                "gpu-wire-fused gate+up",
                timed(wf_gateup, DECODE_REPS),
                rows * gu_bytes_wire,
            )

            def wf_gateup_bias(x2d=x2d, ids=ids):
                return kq.moe_glu_gather_kq(
                    x2d,
                    arms["gate"]["wire"],
                    arms["up"]["wire"],
                    codec,
                    ids,
                    act="swiglu_clamp",
                    limit=7.0,
                    gate_bias=arms["gate"]["bias"],
                    up_bias=arms["up"]["bias"],
                    alpha=1.702,
                )

            report(
                "gpu-wire-fused-bias gate+up",
                timed(wf_gateup_bias, DECODE_REPS),
                rows * gu_bytes_wire,
            )
        if wire_gpu or wire_cpu:

            def wire_gateup(stream=mx.gpu, x=x, ids=ids):
                with mx.stream(stream):
                    g = kq.gather_qmm(
                        x, arms["gate"]["wire"], placeholder, codec, rhs_indices=ids
                    )
                    u = kq.gather_qmm(
                        x, arms["up"]["wire"], placeholder, codec, rhs_indices=ids
                    )
                    return nn_silu_mul(g, u)

            if wire_gpu:
                report(
                    "gpu-wire gate+up",
                    timed(wire_gateup, DECODE_REPS),
                    rows * gu_bytes_wire,
                )
            if wire_cpu:
                report(
                    "cpu-wire gate+up",
                    timed(lambda f=wire_gateup: f(mx.cpu), CPU_DECODE_REPS),
                    rows * gu_bytes_wire,
                )

        def ref_gateup(x=x, ids=ids):
            g = kq.gather_qmm(
                x, arms["gate"]["ref_wire"], placeholder, ref_codec, rhs_indices=ids
            )
            u = kq.gather_qmm(
                x, arms["up"]["ref_wire"], placeholder, ref_codec, rhs_indices=ids
            )
            return nn_silu_mul(g, u)

        if ref_ok:
            report(
                f"gpu-wire-{ref_codec} gate+up",
                timed(ref_gateup, DECODE_REPS),
                rows * 2 * n_gateup * (k // ref_vpb * ref_bpb),
            )
            if k % 256 == 0:  # fused wire ops require K % 256 == 0

                def ref_fused_gateup(x2d=x2d, ids=ids):
                    return kq.moe_glu_gather_kq(
                        x2d,
                        arms["gate"]["ref_wire"],
                        arms["up"]["ref_wire"],
                        ref_codec,
                        ids,
                    )

                report(
                    f"gpu-wire-fused-{ref_codec} gate+up",
                    timed(ref_fused_gateup, DECODE_REPS),
                    rows * 2 * n_gateup * (k // ref_vpb * ref_bpb),
                )
            if k_down % 256 == 0:

                def ref_fused_down(xqv=xqv, ids=ids):
                    return kq.gather_qmv_kq(
                        xqv, arms["down"]["ref_wire"], ref_codec, ids
                    )

                report(
                    f"gpu-wire-fused-{ref_codec} down",
                    timed(ref_fused_down, DECODE_REPS),
                    rows * n_down * (k_down // ref_vpb * ref_bpb),
                )

        report(
            "gpu-packed down",
            timed(lambda s_=stock, xd=xd: s_(arms["down"], xd, k_down), DECODE_REPS),
            rows * down_bytes_packed,
        )
        if codec == "mxfp4":

            def fused_down(xqv=xqv, ids=ids):
                return kq.gather_qmv_bias(
                    xqv,
                    arms["down"]["packed"],
                    arms["down"]["scales"],
                    arms["down"]["bias"],
                    ids,
                )

            report(
                "gpu-packed-fused down",
                timed(fused_down, DECODE_REPS),
                rows * down_bytes_packed,
            )
        if wire_fused:

            def wf_down(xqv=xqv, ids=ids):
                return kq.gather_qmv_kq(xqv, arms["down"]["wire"], codec, ids)

            report(
                "gpu-wire-fused down",
                timed(wf_down, DECODE_REPS),
                rows * down_bytes_wire,
            )

            def wf_down_bias(xqv=xqv, ids=ids):
                return kq.gather_qmv_kq(
                    xqv,
                    arms["down"]["wire"],
                    codec,
                    ids,
                    bias=arms["down"]["bias"],
                )

            report(
                "gpu-wire-fused-bias down",
                timed(wf_down_bias, DECODE_REPS),
                rows * down_bytes_wire,
            )
        if wire_gpu or wire_cpu:

            def wire_down(stream=mx.gpu, xd=xd, ids=ids):
                with mx.stream(stream):
                    return kq.gather_qmm(
                        xd, arms["down"]["wire"], placeholder, codec, rhs_indices=ids
                    )

            if wire_gpu:
                report(
                    "gpu-wire down",
                    timed(wire_down, DECODE_REPS),
                    rows * down_bytes_wire,
                )
            if wire_cpu:
                report(
                    "cpu-wire down",
                    timed(lambda f=wire_down: f(mx.cpu), CPU_DECODE_REPS),
                    rows * down_bytes_wire,
                )

    for m in PREFILL_M:
        print(f"-- prefill M={m} (sorted gate proj) --")
        rows = m * topk
        flat = np.sort(rng.integers(0, e, rows).astype(np.uint32))
        ids_s = mx.array(flat.reshape(rows, 1))
        xs = mx.array(rng.standard_normal((rows, 1, 1, k)).astype(np.float16))
        mx.eval(ids_s, xs)

        def stock_pre(xs=xs, ids_s=ids_s):
            return mx.gather_qmm(
                xs,
                arms["gate"]["packed"],
                arms["gate"]["scales"],
                rhs_indices=ids_s,
                transpose=True,
                group_size=gs,
                bits=bits,
                mode=codec,
                sorted_indices=True,
            )

        report(
            "gpu-packed prefill",
            timed(stock_pre, PREFILL_REPS),
            rows * gu_bytes_packed // 2,
        )
        if wire_gpu:

            def wire_pre(stream=mx.gpu, xs=xs, ids_s=ids_s):
                with mx.stream(stream):
                    return kq.gather_qmm(
                        xs,
                        arms["gate"]["wire"],
                        placeholder,
                        codec,
                        rhs_indices=ids_s,
                        sorted_indices=True,
                    )

            report(
                "gpu-wire prefill",
                timed(wire_pre, PREFILL_REPS),
                rows * gu_bytes_wire // 2,
            )
        if wire_cpu and m <= CPU_PREFILL_CAP:

            def wire_pre_cpu(xs=xs, ids_s=ids_s):
                with mx.stream(mx.cpu):
                    return kq.gather_qmm(
                        xs,
                        arms["gate"]["wire"],
                        placeholder,
                        codec,
                        rhs_indices=ids_s,
                        sorted_indices=True,
                    )

            report(
                "cpu-wire prefill",
                timed(wire_pre_cpu, PREFILL_REPS),
                rows * gu_bytes_wire // 2,
            )
        if ref_ok:

            def ref_pre(xs=xs, ids_s=ids_s):
                return kq.gather_qmm(
                    xs,
                    arms["gate"]["ref_wire"],
                    placeholder,
                    ref_codec,
                    rhs_indices=ids_s,
                    sorted_indices=True,
                )

            report(
                f"gpu-wire-{ref_codec} prefill",
                timed(ref_pre, PREFILL_REPS),
                rows * n_gateup * (k // ref_vpb * ref_bpb),
            )


def nn_silu_mul(g, u):
    return (g * mx.sigmoid(g)) * u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="mxfp4", choices=("mxfp4", "nvfp4"))
    ap.add_argument("--ref-codec", default="iq4_nl")
    ap.add_argument("--shapes", default="ds4,gptoss")
    args = ap.parse_args()

    print(f"mlx {mx.__version__}  macOS {platform.mac_ver()[0]}  {platform.machine()}")
    shapes = args.shapes.split(",")
    if "ds4" in shapes:
        bench_shape("DS-V4-Flash", args.codec, 256, 4096, 2048, 8, args.ref_codec)
    if "gptoss" in shapes:
        e, n, k, topk = gptoss_shape()
        bench_shape("gpt-oss-20b", args.codec, e, n, k, topk, args.ref_codec)


if __name__ == "__main__":
    main()
