#!/usr/bin/env python3
"""NAX (tensor-core) matmul validation for the IQ codecs.

The NAX prefill path engages only on NAX-capable Metal GPUs and only for
transpose=True matmuls with K % 64 == 0 and non-f32 activations (see
kquant_matmul.cpp). On the dev hardware the shape below routes deterministically
to the NAX qmm kernel; under KQUANT_FORCE_CPU there is no GPU/NAX, so the module
is skipped.

For each codec this:
  * bounds the NAX qmm output against an *operand-matched* float64 reference --
    operands rounded to the kernel's bf16 operand/tile dtype, then accumulated in
    float64. That isolates the kernel algorithm from bf16 input-rounding noise,
    so the bound is meaningfully tighter and more diagnostic than test_codecs.py's
    fp16 2e-2 Frobenius check;
  * A/Bs the NAX leaf against the proven ALU leaf via the KQ_DISABLE_NAX
    kill-switch (read live per dispatch in codec_has_nax), so both code paths are
    exercised in one process and shown to agree.

Run locally on GPU (it is part of the per-phase NAX gate, not hosted CI).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx  # noqa: E402
from test_codecs import CODECS, _wire_and_ref  # noqa: E402

import mlx_kquant as kq  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="NAX is a GPU-only path",
)

# NAX-eligible prefill shape: transpose=True, K % 64 == 0, bf16 x, M well into
# the qmm (not qmv/verify_qmv) regime.
M = 256

# Per-codec relative-Frobenius bound vs the operand-matched f64 reference.
# Measured nax_vs_f64 is ~1.65e-3 for ALL nine codecs (iq4_nl..iq1_m) -- the floor
# is the f32->bf16 output cast, not the quant precision, so it is bit-width
# independent (1-bit is no worse than 4-bit). The uniform 1e-2 bound is a ~6x
# regression guard, matching the repo's fp16 matmul check in test_codecs.py.
NAX_BOUND = {
    "iq4_nl": 1.0e-2,
    "iq4_xs": 1.0e-2,
    "iq3_xxs": 1.0e-2,
    "iq3_s": 1.0e-2,
    "iq2_xxs": 1.0e-2,
    "iq2_xs": 1.0e-2,
    "iq2_s": 1.0e-2,
    "iq1_s": 1.0e-2,
    "iq1_m": 1.0e-2,
}


def _bf16(a: np.ndarray) -> np.ndarray:
    """Round a float array through bf16 (the NAX operand/tile dtype), as f64."""
    return np.array(
        mx.array(a.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
    ).astype(np.float64)


def _qmm(codec, wire, x_bf16, *, disable_nax: bool) -> np.ndarray:
    """Run qmm (transpose=True) with NAX on/off via KQ_DISABLE_NAX; return f64."""
    if disable_nax:
        os.environ["KQ_DISABLE_NAX"] = "1"
    else:
        os.environ.pop("KQ_DISABLE_NAX", None)
    try:
        w = mx.array(wire)
        scales = mx.zeros((1,), dtype=mx.uint8)
        out = kq.quantized_matmul(x_bf16, w, scales, codec, transpose=True)
        # qmm follows the activation dtype (bf16 here); cast to f32 before numpy
        # since np.array can't consume MLX bfloat16 (PEP-3118 itemsize mismatch).
        out = out.astype(mx.float32)
        mx.eval(out)
    finally:
        os.environ.pop("KQ_DISABLE_NAX", None)
    return np.array(out).astype(np.float64)


def _check(codec: str) -> None:
    gtype, wpb, bpb, bits, is_kq = CODECS[codec]
    wire, ref = _wire_and_ref(codec, gtype, wpb, bpb, is_kq)  # ref [N, K] f32
    assert wire is not None, f"no wire/ref for {codec}"
    n_out, kk = ref.shape
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, kk)) * 0.1).astype(np.float32)
    x_bf16 = mx.array(x).astype(mx.bfloat16)

    # Operand-matched f64 reference: bf16 operands (what the kernel sees),
    # accumulated in float64 (~ the NAX float accumulator).
    ref_mm = _bf16(x) @ _bf16(ref).T  # [M, N]

    got_nax = _qmm(codec, wire, x_bf16, disable_nax=False)
    got_alu = _qmm(codec, wire, x_bf16, disable_nax=True)

    def rel(g: np.ndarray, r: np.ndarray) -> float:
        return float(np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-9))

    bound = NAX_BOUND[codec]
    rel_nax = rel(got_nax, ref_mm)
    rel_alu = rel(got_alu, ref_mm)
    rel_ab = rel(got_nax, got_alu)
    print(
        f"  [nax] {codec}: nax_vs_f64={rel_nax:.3e} alu_vs_f64={rel_alu:.3e} "
        f"nax_vs_alu={rel_ab:.3e} (bound {bound:.1e})"
    )
    assert rel_nax < bound, f"{codec}: NAX vs f64 rel {rel_nax:.3e} >= {bound:.1e}"
    assert rel_alu < bound, f"{codec}: ALU vs f64 rel {rel_alu:.3e} >= {bound:.1e}"
    # NAX (float accumulate) and ALU (T accumulate) must agree within the same
    # envelope -- both paths produce the same answer for the same operands.
    assert rel_ab < bound, f"{codec}: NAX vs ALU rel {rel_ab:.3e} >= {bound:.1e}"


@pytest.mark.parametrize("codec", list(NAX_BOUND))
def test_nax(codec):
    _check(codec)
