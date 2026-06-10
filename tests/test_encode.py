#!/usr/bin/env python3
"""Validate kq.quantize (encode) for all 10 K-quant codecs.

For a fixed random weight tensor, per codec it checks:
  * scales placeholder shape is [1];
  * round-trip: gguf.quants.dequantize(encode(w)) ~= w within a sane bound
    (gguf-py decodes every codec, so this is a real reconstruction check);
  * the imatrix actually steers the encoder: K-quant wire bytes MUST change
    when an imatrix is supplied, flat codecs MUST NOT (they have no imatrix path);
  * a stable sha1 of the wire bytes, with and without an imatrix.

Exit non-zero on any local failure (shape / round-trip). The printed sha1s are a
stable fingerprint of the encoder output.

Usage: test_encode.py [--codecs q4_k,q8_0,...]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

# codec -> (gguf type, weights_per_block, bits, round-trip rel-Frobenius bound)
CODECS = {
    "q8_0": (GT.Q8_0, 32, 8, 0.02),
    "q4_0": (GT.Q4_0, 32, 4, 0.15),
    "q4_1": (GT.Q4_1, 32, 4, 0.15),
    "q5_0": (GT.Q5_0, 32, 5, 0.10),
    "q5_1": (GT.Q5_1, 32, 5, 0.10),
    "q6_k": (GT.Q6_K, 256, 6, 0.05),
    "q5_k": (GT.Q5_K, 256, 5, 0.08),
    "q4_k": (GT.Q4_K, 256, 4, 0.12),
    "q3_k": (GT.Q3_K, 256, 3, 0.25),
    "q2_k": (GT.Q2_K, 256, 2, 0.40),
}

N, K = 256, 512

BACKEND = "mlx_kquant"


def _encode(w, gs, bits, codec, imatrix):
    return kq.quantize(w, codec, imatrix=imatrix)


def _sha(a) -> str:
    return hashlib.sha1(np.array(a).astype(np.uint8).tobytes()).hexdigest()[:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="")
    args = ap.parse_args(argv)
    allow = {c.strip() for c in args.codecs.split(",") if c.strip()}

    rng = np.random.default_rng(7)
    w_np = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    w = mx.array(w_np)
    # A non-trivial importance vector (length K) to exercise the imatrix path.
    imat_np = np.abs(rng.standard_normal(K)).astype(np.float32) + 0.1
    imat = mx.array(imat_np)

    print(f"=== test_encode [{BACKEND}]  w[{N},{K}] ===")
    print(
        f"  {'codec':<6} {'scales':>7} {'rt_rel':>9} {'imat':>5} {'wq_sha1':>14} "
        f"{'wq_imat_sha1':>14} {'verdict':>8}"
    )
    fails = 0
    for codec, (gtype, wpb, bits, bound) in CODECS.items():
        if allow and codec not in allow:
            continue
        wq, scales = _encode(w, wpb, bits, codec, None)
        wq_im, _ = _encode(w, wpb, bits, codec, imat)
        mx.eval(wq, scales, wq_im)

        scales_ok = tuple(np.array(scales).shape) == (1,)
        wire = np.array(wq).astype(np.uint8)
        w_rt = quants.dequantize(np.ascontiguousarray(wire), gtype).astype(np.float32)
        rel = float(np.linalg.norm(w_rt - w_np) / (np.linalg.norm(w_np) + 1e-6))

        # The imatrix steers importance-weighted rounding in the K-quant
        # encoders, so it MUST change their wire bytes; flat codecs (wpb==32)
        # have no imatrix path, so identical bytes are correct there. A K-quant
        # that ignores the imatrix (regression / unimplemented dispatch) yields
        # equal sha1s and would otherwise pass silently.
        is_kquant = wpb == 256
        imat_changed = _sha(wq) != _sha(wq_im)
        imat_ok = imat_changed if is_kquant else not imat_changed

        bad = (not scales_ok) or (rel >= bound) or (not imat_ok)
        fails += bad
        print(
            f"  {codec:<6} {str(scales_ok):>7} {rel:>9.3e} "
            f"{('chg' if imat_changed else 'same'):>5} {_sha(wq):>14} "
            f"{_sha(wq_im):>14} {'FAIL' if bad else 'ok':>8}"
        )

    print(f"{'FAILURES: ' + str(fails) if fails else 'ALL OK'}")
    return 1 if fails else 0


@pytest.mark.skipif(
    not kq.metallib_loads() or bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kquant encode is GPU-only (no Metal GPU / forced CPU)",
)
def test_encode():
    """pytest entry: runs the full encode sweep (no external data needed)."""
    assert main([]) == 0


@pytest.mark.parametrize("codec", [c for c, v in CODECS.items() if v[1] == 32])
def test_encode_cpu_flat_roundtrip(codec):
    """CPU encode path (kquant_cpu_encode.cpp) for the flat codecs: encode on the
    CPU stream, dequantize with the gguf-py reference, and check the round-trip is
    within the codec's bound. Needs no GPU - this is the CI-runnable gate that the
    encode (and LoRA quantize/fuse-to-kquant) paths actually execute on hosted
    runners. The K-quant codecs stay GPU-only until their CPU encoders land."""
    gtype, _wpb, _bits, bound = CODECS[codec]
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    wq, scales = kq.quantize(mx.array(w_np), codec, stream=mx.cpu)
    mx.eval(wq, scales)

    assert tuple(np.array(scales).shape) == (1,)
    wire = np.ascontiguousarray(np.array(wq).astype(np.uint8))
    w_rt = quants.dequantize(wire, gtype).astype(np.float32)
    rel = float(np.linalg.norm(w_rt - w_np) / (np.linalg.norm(w_np) + 1e-6))
    assert rel < bound, f"{codec}: round-trip rel {rel:.3e} >= bound {bound}"


@pytest.mark.skipif(
    not kq.metallib_loads() or bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="needs the GPU encoder to A/B against (no Metal GPU / forced CPU)",
)
@pytest.mark.parametrize("codec", [c for c, v in CODECS.items() if v[1] == 32])
def test_encode_cpu_matches_gpu_flat(codec):
    """The CPU encoder is a byte-exact port of the GPU kernel: for the flat codecs
    the two paths must produce identical wire bytes (same scale derivation, same
    fp16 store, same round-half-away-from-zero). This is the strong oracle the
    CPU round-trip can only approximate."""
    rng = np.random.default_rng(7)
    w = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))
    wq_gpu, _ = kq.quantize(w, codec, stream=mx.gpu)
    wq_cpu, _ = kq.quantize(w, codec, stream=mx.cpu)
    mx.eval(wq_gpu, wq_cpu)
    assert np.array_equal(
        np.array(wq_gpu).astype(np.uint8), np.array(wq_cpu).astype(np.uint8)
    )


@pytest.mark.parametrize("codec", [c for c, v in CODECS.items() if v[1] == 256])
def test_encode_cpu_kquant_roundtrip(codec):
    """CPU encode path (kquant_cpu_encode.cpp) for the five K-quant super-block
    codecs: encode on the CPU stream, dequantize with the gguf-py reference, and
    check the round-trip is within the codec's bound. This is the CI-runnable gate
    that the encode (and LoRA quantize / fuse-to-kquant) paths execute on hosted
    runners with no GPU."""
    gtype, _wpb, _bits, bound = CODECS[codec]
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    wq, scales = kq.quantize(mx.array(w_np), codec, stream=mx.cpu)
    mx.eval(wq, scales)

    assert tuple(np.array(scales).shape) == (1,)
    wire = np.ascontiguousarray(np.array(wq).astype(np.uint8))
    w_rt = quants.dequantize(wire, gtype).astype(np.float32)
    rel = float(np.linalg.norm(w_rt - w_np) / (np.linalg.norm(w_np) + 1e-6))
    assert rel < bound, f"{codec}: round-trip rel {rel:.3e} >= bound {bound}"


@pytest.mark.parametrize("codec", [c for c, v in CODECS.items() if v[1] == 256])
def test_encode_cpu_kquant_imatrix_roundtrip(codec):
    """CPU encode with an importance matrix: the imatrix steers the per-weight
    rounding of the K-quant encoders (it is a no-op on the flat codecs), so this
    exercises the CPU imatrix branch. The reconstruction stays within a modestly
    relaxed bound - imatrix encode minimizes importance-weighted error, not plain
    Frobenius, so it can trade a little raw round-trip for weighted accuracy."""
    gtype, _wpb, _bits, bound = CODECS[codec]
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    imat_np = (np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32)
    wq, _ = kq.quantize(mx.array(w_np), codec, imatrix=mx.array(imat_np), stream=mx.cpu)
    mx.eval(wq)

    wire = np.ascontiguousarray(np.array(wq).astype(np.uint8))
    w_rt = quants.dequantize(wire, gtype).astype(np.float32)
    rel = float(np.linalg.norm(w_rt - w_np) / (np.linalg.norm(w_np) + 1e-6))
    assert rel < bound * 1.5, f"{codec}: imatrix round-trip rel {rel:.3e}"


@pytest.mark.skipif(
    not kq.metallib_loads() or bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="needs the GPU encoder to A/B against (no Metal GPU / forced CPU)",
)
@pytest.mark.parametrize("codec", [c for c, v in CODECS.items() if v[1] == 256])
def test_encode_cpu_matches_gpu_kquant(codec):
    """The K-quant CPU encoders are serial ports of the 256-thread GPU kernels.
    The numeric core (per-sub-block fit, fp16 super-scale round-trip, packing) is
    identical, so the wire bytes match the GPU exactly - with one caveat: the four
    codecs that consume `sigma2` (q2_k/q4_k/q5_k, and q3_k under an imatrix) reduce
    sum(x^2) over the super-block, and the GPU's simd_sum tree order can differ
    from the serial CPU sum by an ULP. That can round a boundary-tied quant level
    the other way in a rare block, so for those codecs we assert the two encoders
    are numerically *equivalent* rather than byte-identical: nearly all wire bytes
    match, and the CPU reconstruction is exactly as faithful as the GPU's (a
    flipped tie is, by definition, a level the search judged equally good). q6_k
    has no such reduction and must be byte-exact."""
    rng = np.random.default_rng(7)
    w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    w = mx.array(w_np)
    wq_gpu, _ = kq.quantize(w, codec, stream=mx.gpu)
    wq_cpu, _ = kq.quantize(w, codec, stream=mx.cpu)
    mx.eval(wq_gpu, wq_cpu)
    bg = np.array(wq_gpu).astype(np.uint8)
    bc = np.array(wq_cpu).astype(np.uint8)

    if codec == "q6_k":
        assert np.array_equal(bg, bc)
        return

    # Reduction-sensitive codecs: almost every wire byte matches (the rare
    # differences are boundary-tied levels), and the round-trip quality is equal.
    gtype = CODECS[codec][0]
    mismatch = float(np.mean(bg != bc))
    assert mismatch < 2e-3, f"{codec}: {mismatch:.2e} of bytes differ"
    nrm = float(np.linalg.norm(w_np)) + 1e-6
    rel_gpu = float(
        np.linalg.norm(
            quants.dequantize(np.ascontiguousarray(bg), gtype).astype(np.float32) - w_np
        )
        / nrm
    )
    rel_cpu = float(
        np.linalg.norm(
            quants.dequantize(np.ascontiguousarray(bc), gtype).astype(np.float32) - w_np
        )
        / nrm
    )
    assert abs(rel_cpu - rel_gpu) < 1e-3, (
        f"{codec}: round-trip quality differs (cpu {rel_cpu:.5f} vs gpu {rel_gpu:.5f})"
    )


if __name__ == "__main__":
    sys.exit(main())
