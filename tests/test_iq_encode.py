#!/usr/bin/env python3
"""Validate kq.quantize (encode) for the IQ codecs -- the CPU-only encode path.

gguf-py decodes every IQ codec but cannot encode them, so the oracle here is a
round-trip: our CPU encoder -> gguf-py reference decoder -> compare to the
original within a per-codec bound. (Byte-exactness vs llama-quantize is a
separate lab-side check, not run here.) Also asserts encode determinism, that
the imatrix steers the encoder, and that the ggml imatrix-required codecs reject
a missing imatrix -- every IQ codec is importance-weighted, including the flat-32
iq4_nl, so the wpb==256 heuristic the K-quants use is wrong here and the
expectation is registry-driven instead.

Only codecs whose CPU encoder has landed are listed; the map grows per phase.
kq.quantize forces IQ onto a CPU stream internally, so these run on CI with no
GPU.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

# codec -> (gguf type, weights_per_block, bits, round-trip rel-Frobenius bound).
# 4-bit (iq4_*) round-trips to ~0.076 on random gaussian data (q4_k-class); the
# 2-bit iq2_xxs to ~0.36 (random weights are worst-case for an importance-
# weighted codec; real model weights round-trip far tighter). Bounds carry
# ~1.3x headroom over the measured value.
IQ_ENCODE_CODECS = {
    "iq4_nl": (GT.IQ4_NL, 32, 4, 0.12),
    "iq4_xs": (GT.IQ4_XS, 256, 4, 0.12),
    "iq2_xxs": (GT.IQ2_XXS, 256, 2, 0.45),
    "iq2_xs": (GT.IQ2_XS, 256, 2, 0.45),
    "iq2_s": (GT.IQ2_S, 256, 2, 0.45),
    "iq3_xxs": (GT.IQ3_XXS, 256, 3, 0.30),
    "iq3_s": (GT.IQ3_S, 256, 3, 0.30),
    "iq1_s": (GT.IQ1_S, 256, 1, 0.65),
    "iq1_m": (GT.IQ1_M, 256, 1, 0.65),
}

# Codecs ggml marks imatrix-required: kq.quantize rejects them without an
# imatrix (mirrors ggml_quantize_requires_imatrix). The rest fall back gracefully
# to an importance-free weighting.
REQUIRED_IMATRIX = {"iq2_xxs", "iq2_xs", "iq1_s"}

N, K = 256, 512


def _imatrix(rng, scale=1.0):
    return mx.array((np.abs(rng.standard_normal(K)) * scale + 0.1).astype(np.float32))


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
    """Encode -> gguf-py reference decode -> within the codec's bound. Required
    codecs supply an imatrix (their natural call); graceful codecs do not (this
    exercises the importance-free fallback)."""
    gtype, _wpb, _bits, bound = IQ_ENCODE_CODECS[codec]
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    imat = _imatrix(rng) if codec in REQUIRED_IMATRIX else None
    rel = _roundtrip_rel(codec, gtype, w_np, imatrix=imat)
    assert rel < bound, f"{codec}: round-trip rel {rel:.3e} >= bound {bound}"


@pytest.mark.parametrize("codec", list(IQ_ENCODE_CODECS))
def test_iq_encode_imatrix_roundtrip(codec):
    """Encode with an importance matrix; reconstruction stays within a relaxed
    bound (imatrix minimizes importance-weighted, not plain Frobenius, error)."""
    gtype, _wpb, _bits, bound = IQ_ENCODE_CODECS[codec]
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    rel = _roundtrip_rel(codec, gtype, w_np, imatrix=_imatrix(rng))
    assert rel < bound * 1.5, f"{codec}: imatrix round-trip rel {rel:.3e}"


@pytest.mark.parametrize("codec", list(IQ_ENCODE_CODECS))
def test_iq_encode_deterministic(codec):
    """The CPU encoder is deterministic: the same weights (and imatrix, for the
    required codecs) encode byte-identically."""
    rng = np.random.default_rng(1)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    imat = _imatrix(np.random.default_rng(2)) if codec in REQUIRED_IMATRIX else None
    a = _encode_wire(codec, w_np, imatrix=imat)
    b = _encode_wire(codec, w_np, imatrix=imat)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("codec", list(IQ_ENCODE_CODECS))
def test_iq_encode_imatrix_steers(codec):
    """Every IQ codec is importance-weighted (including flat-32 iq4_nl), so the
    imatrix MUST change the wire bytes. Graceful codecs compare no-imatrix vs
    imatrix; required codecs (which reject no-imatrix) compare two imatrices."""
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    steered = _encode_wire(codec, w_np, imatrix=_imatrix(rng, scale=1.0))
    if codec in REQUIRED_IMATRIX:
        other = _encode_wire(codec, w_np, imatrix=_imatrix(rng, scale=4.0))
    else:
        other = _encode_wire(codec, w_np)
    assert not np.array_equal(steered, other), f"{codec}: imatrix did not steer"


@pytest.mark.parametrize("codec", sorted(REQUIRED_IMATRIX))
def test_iq_encode_requires_imatrix(codec):
    """The ggml imatrix-required codecs reject a missing imatrix through the
    public kq.quantize API (op-level gate, before the encoder runs)."""
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    with pytest.raises((ValueError, RuntimeError)):
        wq, _ = kq.quantize(mx.array(w_np), codec)
        mx.eval(wq)


@pytest.mark.parametrize("codec", list(IQ_ENCODE_CODECS))
def test_iq_encode_convert_wrapper(codec):
    """The convert._encode_weight wrapper (the CLI ``quantize`` encode path)
    produces wire bytes for every landed IQ codec -- graceful codecs with no
    imatrix, required codecs with one."""
    from mlx_kquant.convert import _encode_weight

    rng = np.random.default_rng(3)
    w = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))
    vec = None
    if codec in REQUIRED_IMATRIX:
        vec = (np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32)
    wq = _encode_weight(w, codec, vec)
    mx.eval(wq)
    assert wq.shape[0] == N
