#!/usr/bin/env python3
"""Validate kq.quantize (encode) for the IQ codecs -- the CPU-only encode path.

gguf-py decodes every IQ codec but cannot encode them, so the oracle here is a
round-trip: our CPU encoder -> gguf-py reference decoder -> compare to the
original within a per-codec bound. (Byte-exactness vs llama-quantize is a
separate lab-side check, not run here.) Also asserts encode determinism and that
the imatrix steers the encoder -- every IQ codec is importance-weighted,
including the flat-32 iq4_nl, so the wpb==256 heuristic the K-quants use is wrong
here and the expectation is registry-driven instead.

Only codecs whose CPU encoder has landed are listed; the map grows per phase.
kq.quantize forces IQ onto a CPU stream internally, so these run on CI with no
GPU. Required-imatrix rejection is asserted once those codecs (iq2_xxs/iq2_xs/
iq1_s) gain an encoder.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

# codec -> (gguf type, weights_per_block, bits, round-trip rel-Frobenius bound).
# Measured plain round-trip is ~0.076 for both (non-linear 4-bit, q4_k-class);
# the 0.12 bound matches q4_k and leaves ~1.6x headroom.
IQ_ENCODE_CODECS = {
    "iq4_nl": (GT.IQ4_NL, 32, 4, 0.12),
    "iq4_xs": (GT.IQ4_XS, 256, 4, 0.12),
}

N, K = 256, 512


def _encode_wire(codec, w_np, imatrix=None):
    wq, scales = kq.quantize(mx.array(w_np), codec, imatrix=imatrix)
    mx.eval(wq, scales)
    assert tuple(np.array(scales).shape) == (1,), "scales placeholder must be [1]"
    return np.ascontiguousarray(np.array(wq).astype(np.uint8))


def _roundtrip_rel(codec, gtype, w_np, imatrix=None):
    wire = _encode_wire(codec, w_np, imatrix=imatrix)
    w_rt = quants.dequantize(wire, gtype).astype(np.float32)
    return float(np.linalg.norm(w_rt - w_np) / (np.linalg.norm(w_np) + 1e-6))


@pytest.mark.parametrize("codec", list(IQ_ENCODE_CODECS))
def test_iq_encode_roundtrip(codec):
    """Encode -> gguf-py reference decode -> within the codec's bound."""
    gtype, _wpb, _bits, bound = IQ_ENCODE_CODECS[codec]
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    rel = _roundtrip_rel(codec, gtype, w_np)
    assert rel < bound, f"{codec}: round-trip rel {rel:.3e} >= bound {bound}"


@pytest.mark.parametrize("codec", list(IQ_ENCODE_CODECS))
def test_iq_encode_imatrix_roundtrip(codec):
    """Encode with an importance matrix; reconstruction stays within a relaxed
    bound (imatrix minimizes importance-weighted, not plain Frobenius, error)."""
    gtype, _wpb, _bits, bound = IQ_ENCODE_CODECS[codec]
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    rel = _roundtrip_rel(codec, gtype, w_np, imatrix=imat)
    assert rel < bound * 1.5, f"{codec}: imatrix round-trip rel {rel:.3e}"


@pytest.mark.parametrize("codec", list(IQ_ENCODE_CODECS))
def test_iq_encode_deterministic(codec):
    """The CPU encoder is deterministic: the same weights encode byte-identically."""
    rng = np.random.default_rng(1)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    a = _encode_wire(codec, w_np)
    b = _encode_wire(codec, w_np)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("codec", list(IQ_ENCODE_CODECS))
def test_iq_encode_imatrix_steers(codec):
    """Every IQ codec is importance-weighted (including flat-32 iq4_nl), so an
    imatrix MUST change the wire bytes -- registry-driven, not the wpb heuristic."""
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    plain = _encode_wire(codec, w_np)
    steered = _encode_wire(codec, w_np, imatrix=imat)
    assert not np.array_equal(plain, steered), f"{codec}: imatrix did not steer"
