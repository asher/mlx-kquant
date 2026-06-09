#!/usr/bin/env python3
"""Generate K-quant wire-byte fixtures using the kquant FORK's encoder.

K-quant codecs (q2_k..q6_k) cannot be encoded by gguf-py (decode-only) and we
have no local q2_k/q3_k GGUFs. The fork's mx.quantize(mode="kquant") emits
GGML-standard wire bytes that gguf.quants.dequantize CAN read — so we use it to
mint fixtures from a fixed random tensor, then validate kq.dequantize against
the gguf.quants reference on those bytes (in test_codecs.py, dev venv).

RUN WITH A KQUANT-FORK MLX BUILD (one that exposes mx.quantize(mode="kquant")):
    python tests/gen_fixtures.py

Writes tests/fixtures/<codec>.npz  {wire: uint8[N, K/wpb*bpb], N, K}.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

# codec -> (group_size, bits)
KCODECS = {
    "q2_k": (256, 2),
    "q3_k": (256, 3),
    "q4_k": (256, 4),
    "q5_k": (256, 5),
    "q6_k": (256, 6),
}

N, K = 256, 512  # K % 256 == 0 (K-quant block) and % 64 == 0 (nax matmul)
OUT = os.path.join(os.path.dirname(__file__), "fixtures")


E, MOE_N = 4, 128  # experts, out_dims for the MoE fixtures


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1234)
    w_np = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    w = mx.array(w_np)
    for codec, (gs, bits) in KCODECS.items():
        wq, _ = mx.quantize(
            w, group_size=gs, bits=bits, mode="kquant", kquant_type=codec
        )
        mx.eval(wq)
        wire = np.array(wq).astype(np.uint8)
        path = os.path.join(OUT, f"{codec}.npz")
        np.savez(path, wire=wire, N=N, K=K)
        print(f"  {codec:6} wire {wire.shape} -> {path}")

    # 3D MoE expert fixtures: E distinct [MOE_N, K] experts per codec, stacked
    # into [E, MOE_N, packed]. Exercises kq.gather_qmm (decode + prefill paths).
    mrng = np.random.default_rng(4321)
    for codec, (gs, bits) in KCODECS.items():
        wires = []
        for _ in range(E):
            we = mx.array((mrng.standard_normal((MOE_N, K)) * 0.1).astype(np.float32))
            wqe, _ = mx.quantize(
                we, group_size=gs, bits=bits, mode="kquant", kquant_type=codec
            )
            mx.eval(wqe)
            wires.append(np.array(wqe).astype(np.uint8))
        wire = np.stack(wires, axis=0)  # [E, MOE_N, packed]
        path = os.path.join(OUT, f"{codec}_moe.npz")
        np.savez(path, wire=wire, E=E, N=MOE_N, K=K)
        print(f"  {codec:6} moe  {wire.shape} -> {path}")


if __name__ == "__main__":
    main()
