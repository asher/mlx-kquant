#!/usr/bin/env python3
"""Smoke test for kq.load_gguf — the only coverage of kquant_gguf.cpp (gguflib
mmap + KV-metadata decode + tensor packing + shape reversal).

Mint a tiny GGUF with gguf-py (which can ENCODE the flat codecs q8_0/q4_0 and
F32 — all we need to exercise the loader's quantized and native-dtype paths),
load it through the C++ loader, and verify tensors, codecs, shapes and metadata
round-trip.

Runs under pytest; skipped automatically when the extension is unavailable
(load_gguf is mlx_kquant-specific; the kquant fork exposes mx.load instead).
"""

from __future__ import annotations

import numpy as np
import pytest

kq = pytest.importorskip("mlx_kquant")
import mlx.core as mx  # noqa: E402
from gguf import GGUFWriter, GGMLQuantizationType as GT, quants  # noqa: E402


def _mint(path) -> tuple:
    """Write plain.f32 + two flat-codec tensors + scalar metadata; return refs."""
    rng = np.random.default_rng(0)
    w = GGUFWriter(str(path), "smoke")
    w.add_uint32("smoke.answer", 42)
    w.add_string("smoke.name", "hello")
    f32 = (np.arange(64, dtype=np.float32).reshape(4, 16) * 0.01)
    w.add_tensor("plain.f32", f32, raw_dtype=GT.F32)
    src = {}
    for name, gt in (("layer.q8", GT.Q8_0), ("layer.q4", GT.Q4_0)):
        s = (rng.standard_normal((8, 64)).astype(np.float32) * 0.1)
        w.add_tensor(name, quants.quantize(s, gt), raw_dtype=gt)
        src[name] = (gt, s)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return f32, src


def test_load_gguf(tmp_path):
    f32, src = _mint(tmp_path / "smoke.gguf")
    arrays, codecs, metadata, shapes = kq.load_gguf(str(tmp_path / "smoke.gguf"))

    # metadata round-trips (uint, string) + the writer's architecture key.
    assert metadata["smoke.answer"] == 42
    assert metadata["smoke.name"] == "hello"
    assert metadata["general.architecture"] == "smoke"

    # F32 tensor: native passthrough, bit-exact, MLX axis order; the shapes dict
    # reports GGUF-native innermost-first order (reversed), per the loader's API.
    assert np.array_equal(np.array(arrays["plain.f32"]).astype(np.float32), f32)
    assert list(shapes["plain.f32"]) == [16, 4]

    # Quantized tensors: correct codec name, a [1] scales placeholder, wire bytes
    # that dequantize (gguf reference) back to the encoder input within tol, and
    # the reversed logical shape.
    for name, (gt, s) in src.items():
        assert codecs[name] == gt.name.lower()
        assert tuple(np.array(arrays[f"{name}.scales"]).shape) == (1,)
        wire = np.array(arrays[name]).astype(np.uint8)
        back = quants.dequantize(np.ascontiguousarray(wire), gt).astype(np.float32)
        rel = float(np.linalg.norm(back - s) / (np.linalg.norm(s) + 1e-6))
        assert rel < 0.2, f"{name} dequant rel={rel:.3e}"
        assert list(shapes[name]) == [64, 8]


def test_zero_copy_matches_copy(tmp_path):
    """zero_copy=True (no-copy mmap views) must be byte-identical to zero_copy=
    False (eager memcpy). Fork-independent: mints its own GGUF, compares kq-vs-kq.
    The small tensors here exercise the real no-copy path (each sits in a single
    page-aligned window), not just the >INT32 / unaligned fallback."""
    _mint(tmp_path / "zc.gguf")
    path = str(tmp_path / "zc.gguf")
    cp_arr, cp_cod, cp_meta, cp_shp = kq.load_gguf(path, zero_copy=False)
    zc_arr, zc_cod, zc_meta, zc_shp = kq.load_gguf(path, zero_copy=True)

    # Everything but the arrays is plain Python -> compare directly.
    assert zc_cod == cp_cod
    assert zc_meta == cp_meta
    assert zc_shp == cp_shp
    assert set(zc_arr) == set(cp_arr)

    for name in cp_arr:
        a, b = cp_arr[name], zc_arr[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        # raw-byte compare (reinterpret non-uint8 as bytes; no flatten so a >2GB
        # tensor wouldn't overflow the int32 shape — moot at this size).
        au = a if a.dtype == mx.uint8 else mx.view(a, mx.uint8)
        bu = b if b.dtype == mx.uint8 else mx.view(b, mx.uint8)
        assert bool(mx.all(au == bu)), f"byte mismatch in {name}"
