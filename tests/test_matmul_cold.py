#!/usr/bin/env python3
"""Cold-dispatch quantized_matmul correctness - the guard a warm test can't be.

A miscompiled qmm / qmm_nax kernel can return correct results once the GPU is
"warm" (after any prior matmul) yet garbage on the very FIRST matmul dispatch in
a process - which is exactly what a model hits on its first prefill. A normal
pytest run shares one process, so an earlier test's matmul warms the GPU and
masks the bug. So each codec is checked in a FRESH SUBPROCESS, making the
quantized_matmul the first GPU dispatch.

The reference is gguf-py's numpy dequant of the same bytes, then a numpy f32
matmul - never the extension's GPU matmul - so a shared bug can't cancel out of
both sides. The weight is synthesized in-process (no GGUF asset needed) at an
aligned, NAX-eligible shape and M >= 32 so the qmm_nax GEMM path is exercised.

Usage:
    test_matmul_cold.py                 # run every codec, each in a subprocess
    test_matmul_cold.py --cold-check q4_k   # one cold check (internal entrypoint)
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# Every codec. K is a multiple of 256 so each codec's block size divides it;
# N, K aligned and M >= 32 so the NAX GEMM path (not the per-row qmv) runs. The
# nine IQ codecs cover the IQ NAX cold dispatch; the three ggml imatrix-required
# ones are minted with an importance matrix (else kq.quantize rejects them).
CODECS = [
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq4_nl",
    "iq4_xs",
    "iq3_xxs",
    "iq3_s",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq1_s",
    "iq1_m",
]
REQUIRED_IMATRIX = {"iq2_xxs", "iq2_xs", "iq1_s"}
N, K, M = 1024, 1024, 64


def _cold_check(codec: str) -> int:
    """One cold matmul: 0 = ok, 1 = mismatch. The qmm is the first GPU dispatch.

    Everything before it (quantize, the reference dequant + matmul) runs on the
    CPU / in numpy so it cannot warm the GPU matmul path and mask a cold bug.
    """
    import mlx.core as mx
    import numpy as np
    from gguf import GGMLQuantizationType as GT
    from gguf import quants

    import mlx_kquant as kq

    gtype = getattr(GT, codec.upper())
    rng = np.random.default_rng(0)
    w_np = rng.standard_normal((N, K)).astype(np.float32)

    # Quantize on the CPU stream so no GPU matmul runs before the one under test.
    # The imatrix-required IQ codecs need an importance vector to encode at all.
    imatrix = None
    if codec in REQUIRED_IMATRIX:
        imatrix = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    wq, scales = kq.quantize(mx.array(w_np), codec, imatrix=imatrix, stream=mx.cpu)
    mx.eval(wq, scales)
    wq_np = np.ascontiguousarray(np.array(wq))

    # Independent oracle: gguf-py numpy dequant of the SAME bytes, then numpy f32
    # matmul. Never kq.dequantize and never the GPU matmul, so neither a shared
    # dequant nor a shared matmul bug can cancel out of both got and ref.
    deq = quants.dequantize(wq_np, gtype).astype(np.float32)  # [N, K]
    x_np = (rng.standard_normal((M, K)).astype(np.float32)) * 0.1
    ref = x_np @ deq.T  # [M, N]

    # FIRST GPU dispatch: the quantized matmul under test.
    got = kq.quantized_matmul(
        mx.array(x_np).astype(mx.float16),
        mx.array(wq_np),
        mx.array(scales),
        codec,
        transpose=True,
    )
    mx.eval(got)
    g = np.array(got).astype(np.float32)

    # Compare against the SAME dequantized weights, so quantization error cancels
    # and only the matmul accumulation differs (f16 GPU kernel vs f32 numpy). Use
    # the error relative to the output's largest magnitude: a correct kernel sits
    # at ~1e-4, a miscompiled one at ~1 (garbage), so 1e-2 separates them with a
    # wide margin and is insensitive to per-element near-zero reference values.
    max_abs = float(np.abs(g - ref).max())
    scale = max(float(np.abs(ref).max()), 1e-6)
    rel = max_abs / scale
    ok = rel < 1e-2
    print(
        f"{codec}: M={M} N={N} K={K} max_abs={max_abs:.3e} "
        f"rel={rel:.3e} {'ok' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


@pytest.mark.parametrize("codec", CODECS)
def test_quantized_matmul_cold(codec):
    """Each codec's matmul must be correct on the FIRST (cold) GPU dispatch.

    Runs in a fresh subprocess so the qmm is genuinely the process's first GPU
    matmul; a warmup in this (shared) pytest process would otherwise hide a
    cold-only miscompile. GPU-only: the bug lives in the metallib, so this skips
    honestly without a Metal GPU rather than passing vacuously on the CPU path.
    """
    import mlx_kquant as kq

    if os.environ.get("KQUANT_FORCE_CPU") or not kq.metallib_loads():
        pytest.skip("cold-dispatch matmul check needs a Metal GPU")

    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--cold-check", codec],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (
        f"{codec}: cold quantized_matmul mismatch vs dequant reference "
        f"(first GPU dispatch)\n{r.stdout}\n{r.stderr}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cold-check":
        raise SystemExit(_cold_check(sys.argv[2]))
    rc = 0
    for c in CODECS:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--cold-check", c]
        )
        rc |= r.returncode
    raise SystemExit(rc)
