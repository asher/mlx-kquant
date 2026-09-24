"""Prism codec (PQ2_0, PTQ1_0) oracles and real-file decode checks.

The element map of PTQ1_0 is the one place a port can go wrong without any
kernel being involved, so it is pinned three ways: the stage-derived order in
kqref against the fork's literal three-branch map, the kqref decoder against a
per-element transcription of the fork's trit accessor, and both codecs' CPU
decoders against the numpy oracles on rows of the Ternary Bonsai files when
they are present (``KQUANT_PRISM_GGUF_DIR``, default the local model dir).
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from kqref import PTQ1_0_ORDER, quants, synth_wire

import mlx_kquant as kq

CPU = mx.cpu

PRISM_DIR = os.environ.get(
    "KQUANT_PRISM_GGUF_DIR",
    os.path.expanduser("~/llm/gguf/prism-ml__Ternary-Bonsai-2-27B-gguf"),
)
PRISM_FILES = {
    "ptq1_0": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "pq2_0": "Ternary-Bonsai-2-27B-PQ2_0.gguf",
}


def _fork_elem_map(e: int) -> tuple[int, int]:
    """PrismML/llama.cpp ``ptq1_0_elem`` (ggml-metal/kernels/dequantize.h):
    payload byte and trit of element e, transcribed literally."""
    if e < 80:
        return e & 15, e >> 4
    if e < 120:
        t = e - 80
        return 16 + (t & 7), t >> 3
    t = e - 120
    return 24 + (t & 1), t >> 1


def _fork_trit(b: int, n: int) -> int:
    """The reference integer recurrence ``((b * 3^n) & 0xFF) * 3 >> 8``."""
    return (((b * 3**n) & 0xFF) * 3) >> 8


def test_ptq1_0_order_matches_fork_map():
    fork = np.array([_fork_elem_map(e) for e in range(128)], dtype=np.int64)
    assert np.array_equal(PTQ1_0_ORDER, fork)


def test_ptq1_0_numpy_decoder_matches_fork_accessor():
    rng = np.random.default_rng(3)
    wire = synth_wire(rng, "ptq1_0", 28, 64)
    got = quants.dequantize(wire, "PTQ1_0").reshape(64, 128)
    for b in range(64):
        d = float(wire[b, 26:28].copy().view(np.float16)[0])
        for e in range(128):
            byte, n = _fork_elem_map(e)
            want = np.float32((_fork_trit(int(wire[b, byte]), n) - 1) * d)
            assert got[b, e] == want, f"block {b} element {e}"


def test_ptq1_0_every_byte_decodes_to_a_trit():
    for b in range(256):
        for n in range(5):
            assert _fork_trit(b, n) in (0, 1, 2)


def test_ptq1_0_quantize_round_trips_ternary_values():
    rng = np.random.default_rng(5)
    x = rng.integers(-1, 2, size=(16, 512)).astype(np.float32) * 0.05
    x[:, 0] = 0.05  # every block has a nonzero amax
    wire = quants.quantize(x, "PTQ1_0")
    assert wire.shape == (16, 512 // 128 * 28)
    back = quants.dequantize(wire, "PTQ1_0")
    assert np.array_equal(back, x.astype(np.float16).astype(np.float32))
    assert np.array_equal(quants.quantize(back, "PTQ1_0"), wire)


def test_pq2_0_code_three_decodes_to_plus_two():
    wire = np.zeros((1, 34), dtype=np.uint8)
    wire[0, 0:2] = np.array([1.0], dtype=np.float16).view(np.uint8)
    wire[0, 2] = 0b11_10_01_00  # elements 0..3 = codes 0, 1, 2, 3
    got = quants.dequantize(wire, "PQ2_0").reshape(-1)
    assert got[:4].tolist() == [-1.0, 0.0, 1.0, 2.0]
    out = kq.dequantize(
        mx.array(wire), mx.zeros((1,), mx.uint8), "pq2_0", dtype=mx.float32, stream=CPU
    )
    assert np.array(out).reshape(-1)[:4].tolist() == [-1.0, 0.0, 1.0, 2.0]


def test_pq2_0_quantize_round_trips_three_level_values():
    rng = np.random.default_rng(9)
    x = rng.integers(-1, 2, size=(16, 512)).astype(np.float32) * 0.25
    x[:, 0] = 0.25
    wire = quants.quantize(x, "PQ2_0")
    assert wire.shape == (16, 512 // 128 * 34)
    back = quants.dequantize(wire, "PQ2_0")
    assert np.array_equal(back, x)
    assert np.array_equal(quants.quantize(back, "PQ2_0"), wire)


@pytest.mark.parametrize("codec", sorted(PRISM_FILES))
def test_real_file_rows_decode_bit_exact(codec):
    path = os.path.join(PRISM_DIR, PRISM_FILES[codec])
    if not os.path.exists(path):
        pytest.skip(f"no {PRISM_FILES[codec]} under {PRISM_DIR}")
    from gguf import GGUFReader

    reader = GGUFReader(path, "r")
    want_name = codec.upper()
    checked = 0
    for t in reader.tensors:
        if getattr(t.tensor_type, "name", None) != want_name:
            continue
        rows = min(8, int(t.shape[1]))
        raw = np.ascontiguousarray(t.data[:rows], dtype=np.uint8)
        ref = quants.dequantize(raw, t.tensor_type).astype(np.float32)
        out = kq.dequantize(
            mx.array(raw), mx.zeros((1,), mx.uint8), codec, dtype=mx.float32, stream=CPU
        )
        got = np.array(out).reshape(ref.shape)
        assert np.array_equal(ref, got), f"{t.name}: CPU decode not bit-exact"
        checked += 1
        if checked == 6:
            break
    assert checked == 6, f"only {checked} {want_name} tensors in {path}"


# The PTQ1_0 M=1 mat-vec runs qmv_fast (four rows per simdgroup) when N is a
# multiple of 8 and K of 256, and plain qmv (two rows per simdgroup)
# otherwise, including the partial last threadgroup. Every shape and
# activation dtype must match the numpy oracle.
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "n,k", [(1024, 1024), (1004, 1024), (1002, 1024), (256, 640), (8, 128)]
)
def test_ptq1_0_matvec_shapes(n, k, dtype):
    rng = np.random.default_rng(3)
    wire = synth_wire(rng, "ptq1_0", 28, n * (k // 128)).reshape(n, k // 128 * 28)
    deq = quants.dequantize(np.ascontiguousarray(wire), "PTQ1_0")
    scales = mx.zeros((1,), dtype=mx.uint8)
    x = mx.array((rng.standard_normal((1, k)) * 0.5).astype(np.float32)).astype(dtype)
    ref = np.array(x.astype(mx.float32)) @ deq.T
    out = kq.quantized_matmul(x, mx.array(wire), scales, "ptq1_0", transpose=True)
    got = np.array(out.astype(mx.float32))
    err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-6))
    assert err < 2e-2, f"N{n} K{k} {dtype}: rel err {err:.3e}"
