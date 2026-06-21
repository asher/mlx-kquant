#!/usr/bin/env python3
"""Generate K-quant wire-byte fixtures with the kquant encoder (``kq.quantize``).

The K-quant superblock codecs (q2_k..q6_k) cannot be encoded by gguf-py (it is
decode-only) and we keep no q2_k/q3_k GGUFs in-tree. ``kq.quantize`` emits
GGML-standard wire bytes that ``gguf.quants.dequantize`` reads back, so we mint
fixtures from fixed random tensors here and validate ``kq.dequantize`` against
the gguf-py reference on those bytes (``tests/test_codecs.py`` and
``tests/test_gather.py``).

Requires a built extension (encode runs on CPU or Metal):

    python tests/gen_fixtures.py
    shasum -a 256 -c tests/fixtures/SHA256SUMS   # from the repo root

Writes
    tests/fixtures/<codec>.npz       {wire: uint8[N, packed], N, K}
    tests/fixtures/<codec>_moe.npz   {wire: uint8[E, N, packed], E, N, K}
    tests/fixtures/SHA256SUMS        sha256 manifest for the above
"""

from __future__ import annotations

import hashlib
import os

import mlx.core as mx
import numpy as np

import mlx_kquant as kq
from mlx_kquant.codec_geometry import IMATRIX_CODECS

# The 5 K-quant superblock codecs whose committed fixtures are minted with an
# imatrix. IMATRIX_CODECS is an explicit literal in codec_geometry (NOT every
# wpb==256 codec -- that would wrongly pull in the CPU-only IQ superblocks, which
# have no GPU encoder and no committed fixtures).
KCODECS = sorted(IMATRIX_CODECS)

N, K = 256, 512  # K % 256 == 0 (K-quant block) and % 64 == 0 (nax matmul)
E, MOE_N = 4, 128  # experts, out_dims for the MoE fixtures

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")
REPO_ROOT = os.path.dirname(HERE)


def _write_manifest(paths: list[str]) -> None:
    """Write a ``shasum -a 256 -c``-compatible manifest (repo-relative paths)."""
    lines = []
    for p in sorted(paths):
        with open(p, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        lines.append(f"{digest}  {os.path.relpath(p, REPO_ROOT)}\n")
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
        f.writelines(lines)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    written = []

    rng = np.random.default_rng(1234)
    w = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))
    for codec in KCODECS:
        wq, _ = kq.quantize(w, codec)
        mx.eval(wq)
        wire = np.array(wq).astype(np.uint8)
        path = os.path.join(OUT, f"{codec}.npz")
        np.savez(path, wire=wire, N=N, K=K)
        written.append(path)
        print(f"  {codec:6} wire {wire.shape} -> {path}")

    # 3D MoE expert fixtures: E distinct [MOE_N, K] experts per codec, stacked
    # into [E, MOE_N, packed]. Exercises kq.gather_qmm (decode + prefill paths).
    mrng = np.random.default_rng(4321)
    for codec in KCODECS:
        wires = []
        for _ in range(E):
            we = mx.array((mrng.standard_normal((MOE_N, K)) * 0.1).astype(np.float32))
            wqe, _ = kq.quantize(we, codec)
            mx.eval(wqe)
            wires.append(np.array(wqe).astype(np.uint8))
        wire = np.stack(wires, axis=0)  # [E, MOE_N, packed]
        path = os.path.join(OUT, f"{codec}_moe.npz")
        np.savez(path, wire=wire, E=E, N=MOE_N, K=K)
        written.append(path)
        print(f"  {codec:6} moe  {wire.shape} -> {path}")

    _write_manifest(written)
    print(f"  wrote {os.path.join(OUT, 'SHA256SUMS')} ({len(written)} files)")


if __name__ == "__main__":
    main()
