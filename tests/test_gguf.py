#!/usr/bin/env python3
"""Smoke test for kq.load_gguf - the only coverage of kquant_gguf.cpp (gguflib
mmap + KV-metadata decode + tensor packing + shape reversal).

Mint a tiny GGUF with gguf-py (which can ENCODE the flat codecs q8_0/q4_0 and
F32 - all we need to exercise the loader's quantized and native-dtype paths),
load it through the C++ loader, and verify tensors, codecs, shapes and metadata
round-trip.

Runs under pytest; skipped automatically when the extension is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

kq = pytest.importorskip("mlx_kquant")
import mlx.core as mx  # noqa: E402
from gguf import GGMLQuantizationType as GT  # noqa: E402
from gguf import GGUFWriter, quants  # noqa: E402


def _mint(path) -> tuple:
    """Write plain.f32 + two flat-codec tensors + scalar metadata; return refs."""
    rng = np.random.default_rng(0)
    w = GGUFWriter(str(path), "smoke")
    w.add_uint32("smoke.answer", 42)
    w.add_string("smoke.name", "hello")
    f32 = np.arange(64, dtype=np.float32).reshape(4, 16) * 0.01
    w.add_tensor("plain.f32", f32, raw_dtype=GT.F32)
    src = {}
    for name, gt in (("layer.q8", GT.Q8_0), ("layer.q4", GT.Q4_0)):
        s = rng.standard_normal((8, 64)).astype(np.float32) * 0.1
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


@pytest.mark.parametrize("name", ["STQ1_0", "PQ2_0", "PTQ1_0", "Q2_0"])
def test_load_gguf_codecs_newer_than_gguf_py(tmp_path, name):
    """A tensor of a ggml type gguf-py has no member for loads with its
    codec name, its wire bytes and its logical shape, as a zero-copy view
    and as a copy. The parser sizes the tensor from its own type table, so a
    type missing there fails the load."""
    from kqref import SENTINELS, synth_wire

    ((sent, (wpb, bpb, _, _)),) = [
        (s, v) for s, v in SENTINELS.items() if s.name == name
    ]
    codec = name.lower()
    rows, k = 6, 4 * 256
    wire = synth_wire(np.random.default_rng(1), codec, bpb, rows * k // wpb)
    wire = wire.reshape(rows, k // wpb * bpb)
    path = str(tmp_path / f"{codec}.gguf")
    w = GGUFWriter(path, "smoke")
    w.add_tensor("pad.f32", np.zeros((3,), dtype=np.float32), raw_dtype=GT.F32)
    w.add_tensor("layer.w", wire, raw_dtype=sent)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    for zero_copy in (True, False):
        arrays, codecs, _, shapes = kq.load_gguf(path, zero_copy=zero_copy)
        assert codecs["layer.w"] == codec
        assert list(shapes["layer.w"]) == [k, rows]
        got = arrays["layer.w"]
        assert got.dtype == mx.uint8
        assert np.array_equal(np.array(got), wire)
        aliases = kq.verify_zero_copy_views([("layer.w", got)], ["layer.w"])
        # a build without Metal has no zero-copy view: every load is a copy
        assert bool(aliases) == (zero_copy and mx.metal.is_available())
        del arrays, got


def _write_gguf(path, tensors, alignment=None):
    w = GGUFWriter(str(path), "smoke")
    if alignment is not None:
        w.add_custom_alignment(alignment)
    for name, arr, dtype in tensors:
        w.add_tensor(name, arr, raw_dtype=dtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _type_43_tensors(stq_rows=6, mx8_rows=12):
    """An STQ1_0 weight (rows of 1024) and an MXFP8 table (rows of 256), the
    two formats that share ggml type 43, as (name, wire, sentinel, codec)."""
    from kqref import MXFP8, STQ1_0, synth_wire

    rng = np.random.default_rng(2)
    stq = synth_wire(rng, "stq1_0", 42, stq_rows * 4).reshape(stq_rows, 4 * 42)
    mx8 = synth_wire(rng, "mxfp8", 33, mx8_rows * 8).reshape(mx8_rows, 8 * 33)
    return {
        "blk.0.ffn_up.weight": (stq, STQ1_0, "stq1_0", 1024),
        "per_layer_token_embd.weight": (mx8, MXFP8, "mxfp8", 256),
    }


def _check_type_43_load(path, want):
    from kqref import quants

    for zero_copy in (True, False):
        arrays, codecs, _, shapes = kq.load_gguf(str(path), zero_copy=zero_copy)
        for name, (wire, sent, codec, k) in want.items():
            assert codecs[name] == codec, name
            assert list(shapes[name]) == [k, wire.shape[0]]
            got = arrays[name]
            assert got.dtype == mx.uint8 and np.array_equal(np.array(got), wire)
            scales = arrays[name.removesuffix(".weight") + ".scales"]
            out = kq.dequantize(got, scales, codec, dtype=mx.float32)
            assert np.array_equal(np.array(out), quants.dequantize(wire, sent.name))
        del arrays, got, out


@pytest.mark.parametrize("layout", ["stq1_0_first", "mxfp8_first", "mxfp8_only"])
def test_load_gguf_type_43_resolves_by_span(tmp_path, layout):
    """Type 43 is STQ1_0 in ternary QAT GGUFs and MXFP8 in DeepSeek-V4.1
    engram tables, and one file can hold both. Each tensor resolves to its
    format by the bytes it spans, up to the next tensor or, for the last
    one, the end of the file (a standalone table file holds the table only).
    A skipped tensor reports its codec too: the caller reads it itself."""
    t = _type_43_tensors()
    stq, mx8 = "blk.0.ffn_up.weight", "per_layer_token_embd.weight"
    pad = ("pad.f32", np.arange(5, dtype=np.float32), GT.F32)
    order = {
        "stq1_0_first": [stq, "pad.f32", mx8],
        "mxfp8_first": [mx8, "pad.f32", stq],
        "mxfp8_only": [mx8],
    }[layout]
    tensors = [pad if n == "pad.f32" else (n, t[n][0], t[n][1]) for n in order]
    path = tmp_path / "type43.gguf"
    _write_gguf(path, tensors)

    _check_type_43_load(path, {n: t[n] for n in order if n in t})

    arrays, codecs, _, _ = kq.load_gguf(str(path), True, [mx8])
    assert mx8 not in arrays and codecs[mx8] == "mxfp8"


def test_load_gguf_type_43_span_uses_file_alignment(tmp_path):
    """At general.alignment 64 a tensor carries up to 63 bytes of padding.
    Both tensors here carry more than 32, the default alignment."""
    t = _type_43_tensors(stq_rows=2, mx8_rows=3)
    stq, mx8 = "blk.0.ffn_up.weight", "per_layer_token_embd.weight"
    assert -t[stq][0].nbytes % 64 > 32 and -t[mx8][0].nbytes % 64 > 32
    path = tmp_path / "align64.gguf"
    _write_gguf(
        path,
        [
            (stq, t[stq][0], t[stq][1]),
            (mx8, t[mx8][0], t[mx8][1]),
            ("pad.f32", np.arange(5, dtype=np.float32), GT.F32),
        ],
        alignment=64,
    )
    _check_type_43_load(path, t)


def test_load_gguf_type_43_stq1_0_with_trailing_bytes(tmp_path):
    """Bytes after the last tensor (here 32 zeros) leave an STQ1_0 tensor
    STQ1_0: its span holds its size and is far short of the MXFP8 size."""
    t = _type_43_tensors()
    stq = "blk.0.ffn_up.weight"
    path = tmp_path / "stq_tail.gguf"
    _write_gguf(path, [(stq, t[stq][0], t[stq][1])])
    with open(path, "ab") as f:
        f.write(bytes(32))

    _check_type_43_load(path, {stq: t[stq]})
    arrays, codecs, _, _ = kq.load_gguf(str(path), True, [stq])
    assert stq not in arrays and codecs[stq] == "stq1_0"


def test_load_gguf_type_43_small_mxfp8_at_page_alignment(tmp_path):
    """At a page-sized alignment a one-row table's padding has room for
    either format's size. It is MXFP8, the larger size that fits."""
    from kqref import MXFP8, synth_wire

    wire = synth_wire(np.random.default_rng(4), "mxfp8", 33, 8).reshape(1, 8 * 33)
    pad = ("pad.f32", np.arange(5, dtype=np.float32), GT.F32)
    path = tmp_path / "small43.gguf"
    _write_gguf(path, [("small.weight", wire, MXFP8), pad], alignment=4096)
    _check_type_43_load(path, {"small.weight": (wire, MXFP8, "mxfp8", 256)})


@pytest.mark.parametrize(
    "rows, cols, cut",
    [
        # 2^21 values: the span lands between the STQ1_0 size plus 1 MiB and
        # the MXFP8 size, where an STQ1_0 reading would load silently.
        pytest.param(8192, 256, 2**21 // 32 * 33 // 4, id="past-stq1_0-slack"),
        # 288 values are not whole STQ1_0 blocks: any short span is cut.
        pytest.param(3, 96, 100, id="not-stq1_0-blocks"),
    ],
)
def test_load_gguf_type_43_cut_short_mxfp8_raises(tmp_path, rows, cols, cut):
    """A file cut inside its last tensor, an MXFP8 table, fails the load,
    skipped or not, rather than read the table as STQ1_0 weights."""
    from kqref import MXFP8

    wire = np.zeros((rows, cols // 32 * 33), dtype=np.uint8)
    pad = ("pad.f32", np.arange(5, dtype=np.float32), GT.F32)
    path = tmp_path / "cut43.gguf"
    _write_gguf(path, [pad, ("table.weight", wire, MXFP8)])
    size = path.stat().st_size
    os.truncate(path, size - cut)

    n, mx8 = rows * cols, wire.nbytes
    span = -(-mx8 // 32) * 32 - cut
    assert span < mx8
    if n % 256 == 0:
        assert span >= n // 256 * 42 + 2**20
    msg = (
        rf"table\.weight has type 43 and is the last tensor in the file\. It "
        rf"reads as mxfp8, which needs {mx8} bytes for its {n} elements, but "
        rf"its byte span to the end of the file is {span}: the file looks cut"
    )
    with pytest.raises(RuntimeError, match=msg):
        kq.load_gguf(str(path))
    with pytest.raises(RuntimeError, match=msg):
        kq.load_gguf(str(path), True, ["table.weight"])


@pytest.mark.parametrize(
    "block, rows, cols, span, last",
    [
        pytest.param((256, 32), 4, 1024, 512, False, id="short"),
        pytest.param((256, 32), 4, 1024, 512, True, id="short-last"),
        pytest.param((16, 40), 3, 48, 384, False, id="partial-block"),
        pytest.param((32, 20), 3, 96, 192, False, id="short-not-stq1_0-blocks"),
    ],
)
def test_load_gguf_type_43_fitting_neither_raises(
    tmp_path, monkeypatch, block, rows, cols, span, last
):
    """A type-43 tensor that is neither format fails the load, skipped or
    not, naming the tensor, its span and both sizes: a span short of the
    STQ1_0 size (32-byte blocks of 256 weights), a count that is not whole
    blocks of either (40-byte blocks of 16 weights), or, before the last
    tensor, a span short of the MXFP8 size on a count that is not whole
    STQ1_0 blocks (20-byte blocks of 32 weights)."""
    from gguf.constants import GGML_QUANT_SIZES
    from kqref import MXFP8

    odd = type(MXFP8)("TYPE43_ODD", 43)
    monkeypatch.setitem(GGML_QUANT_SIZES, odd, block)
    wpb, bpb = block
    shape = (rows, cols // wpb * bpb)
    wire = np.random.default_rng(3).integers(0, 256, shape, dtype=np.uint8)
    tensors = [("odd.weight", wire, odd)]
    pad = ("pad.f32", np.arange(5, dtype=np.float32), GT.F32)
    tensors = [pad, *tensors] if last else [*tensors, pad]
    path = tmp_path / "odd43.gguf"
    _write_gguf(path, tensors)

    n = rows * cols
    msg = (
        rf"odd\.weight has type 43.* fits neither: its {n} elements span "
        rf"{span} bytes, where stq1_0 needs whole 256-element blocks "
        rf"\({-(-n // 256) * 42} bytes\) and mxfp8 whole 32-element blocks "
        rf"\({-(-n // 32) * 33} bytes\)"
    )
    with pytest.raises(RuntimeError, match=msg):
        kq.load_gguf(str(path))
    with pytest.raises(RuntimeError, match=msg):
        kq.load_gguf(str(path), True, ["odd.weight"])


_LOAD_CUT_FILE = """
import sys
import mlx_kquant as kq
path, skip = sys.argv[1], sys.argv[2:]
for zero_copy in (True, False):
    try:
        kq.load_gguf(path, zero_copy, skip)
    except RuntimeError as e:
        print("raised", e)
    else:
        print("loaded")
"""


def _pad32(n):
    return -(-n // 32) * 32


@pytest.mark.parametrize(
    "layout, skip",
    [
        pytest.param("table-then-f32", [], id="table-then-f32"),
        pytest.param("table-then-f32", ["table.weight"], id="table-then-f32-skip"),
        pytest.param("f32-last", [], id="f32-last"),
    ],
)
def test_load_gguf_cut_inside_a_tensor_raises(tmp_path, layout, skip):
    """A file cut inside a tensor fails the load in both zero_copy modes,
    rather than read past the end of the file (a crash, or garbage values).
    Here an MXFP8 table followed by an f32 tensor is cut 600 KB into the
    table, whose span in the header still holds the full table, or an f32
    last tensor loses its final 1000 bytes. The loads run in a child
    process, so a read past the end fails the test rather than the run."""
    from kqref import MXFP8

    rng = np.random.default_rng(8)
    table = np.zeros((4096, 8 * 33), dtype=np.uint8)
    f32 = rng.standard_normal(16384).astype(np.float32)
    if layout == "table-then-f32":
        tensors = [("table.weight", table, MXFP8), ("after.f32", f32, GT.F32)]
        name, nbytes, cut = "table.weight", table.nbytes, 600_000
        tail = _pad32(table.nbytes) + _pad32(f32.nbytes)
    else:
        tensors = [("pad.f32", np.arange(5, dtype=np.float32), GT.F32)]
        tensors.append(("last.f32", f32, GT.F32))
        name, nbytes, cut = "last.f32", f32.nbytes, 1000
        tail = _pad32(f32.nbytes)
    path = tmp_path / "cut.gguf"
    _write_gguf(path, tensors)
    size = path.stat().st_size
    os.truncate(path, size - cut)

    r = subprocess.run(
        [sys.executable, "-c", _LOAD_CUT_FILE, str(path), *skip],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    end = size - tail + nbytes
    msg = (
        f"raised [load_gguf] tensor {name} ends at byte {end}, past the end of "
        f"the file at byte {size - cut}: the file looks cut short."
    )
    assert r.stdout.splitlines() == [msg, msg]


def test_zero_copy_matches_copy(tmp_path):
    """zero_copy=True (no-copy mmap views) must be byte-identical to zero_copy=
    False (eager memcpy). Mints its own GGUF, compares kq-vs-kq.
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
        # tensor wouldn't overflow the int32 shape - moot at this size).
        au = a if a.dtype == mx.uint8 else mx.view(a, mx.uint8)
        bu = b if b.dtype == mx.uint8 else mx.view(b, mx.uint8)
        assert bool(mx.all(au == bu)), f"byte mismatch in {name}"


def test_verify_zero_copy_views(tmp_path):
    """Donation-tripwire red/green: mapped views must keep their wire dtype,
    no_alias names must not alias the mapping at all, owned arrays are exempt.
    """
    path = str(tmp_path / "views.gguf")
    w = GGUFWriter(path, "smoke")
    f16 = np.arange(64, dtype=np.float16).reshape(4, 16) * np.float16(0.25)
    f32 = np.arange(32, dtype=np.float32).reshape(2, 16)
    w.add_tensor("t.f16", f16, raw_dtype=GT.F16)
    w.add_tensor("t.f32", f32, raw_dtype=GT.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    arrays, _, _, _ = kq.load_gguf(path)
    a16, a32 = arrays["t.f16"], arrays["t.f32"]
    mx.eval(a16, a32)
    if not mx.metal.is_available():
        # CPU-only builds have no no-copy Metal wrap: every tensor falls back
        # to the memcpy path, so the checker must be inert by construction.
        assert kq.zero_copy_view_count() == 0
        assert kq.verify_zero_copy_views([("a", a16)], no_alias=["a"]) == []
        pytest.skip("zero-copy mmap views need the Metal allocator")
    assert kq.zero_copy_view_count() >= 2

    owned = mx.array(np.ones(8, dtype=np.float32))
    mx.eval(owned)

    # Green: untouched views carry their wire dtype; owned buffers are skipped.
    assert kq.verify_zero_copy_views([("a", a16), ("b", a32), ("c", owned)]) == []

    # Red: a same-itemsize reinterpret aliasing the mapping is exactly the
    # footprint a donated dtype-changing cast leaves behind.
    fake = mx.view(a16, mx.bfloat16)
    mx.eval(fake)
    (prob,) = kq.verify_zero_copy_views([("bad", fake)])
    assert "bad" in prob and "float16" in prob and "bfloat16" in prob

    # Red: names in no_alias must own their buffers even at matching dtype.
    (prob,) = kq.verify_zero_copy_views([("t", a32)], no_alias=["t"])
    assert "aliases" in prob
    assert kq.verify_zero_copy_views([("t", owned)], no_alias=["t"]) == []

    # Unevaluated arrays are reported, not silently skipped.
    (prob,) = kq.verify_zero_copy_views([("z", a32 + 0.0)])
    assert "unevaluated" in prob


@pytest.mark.skipif(
    not __import__("os").environ.get("KQUANT_BIG_TESTS"),
    reason="writes a >2 GiB GGUF; set KQUANT_BIG_TESTS=1 to run",
)
def test_load_gguf_wire_over_int32(tmp_path):
    """A quantized tensor whose wire bytes exceed INT32_MAX must still load
    zero-copy (wide-dtype window slice + view back), not fall back to the
    eager per-tensor memcpy - which OOMs a many-hundred-expert MoE at load."""
    import resource
    import sys

    rows, k = 247_000, 8192  # 34-byte q8_0 blocks -> 8704 B/row, ~2.15 GB
    row_bytes = k // 32 * 34
    nbytes = rows * row_bytes
    assert nbytes > 2**31

    rng = np.random.default_rng(0)
    wire = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    path = str(tmp_path / "big.gguf")
    w = GGUFWriter(path, "smoke")
    # add_tensor takes the byte shape; the writer derives the logical shape.
    w.add_tensor("big.q8", wire.reshape(rows, row_bytes), raw_dtype=GT.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss units
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    arrays, codecs, _meta, _shapes = kq.load_gguf(path, True)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale

    a = arrays["big.q8"]
    assert a.dtype == mx.uint8 and a.shape == (rows, row_bytes)
    assert dict(codecs)["big.q8"] == "q8_0"
    # The regression signature: the pre-fix path memcpy'd the wire eagerly.
    assert rss1 - rss0 < 512 * 1024 * 1024, "load copied the wire bytes"

    for i in (0, 1, rows // 2, rows - 1):
        got = np.array(a[i])
        assert np.array_equal(got, wire[i * row_bytes : (i + 1) * row_bytes]), i


@pytest.mark.skipif(
    not __import__("os").environ.get("KQUANT_BIG_TESTS"),
    reason="writes a >4 GiB GGUF; set KQUANT_BIG_TESTS=1 to run",
)
def test_load_gguf_wire_over_widest_window(tmp_path):
    """A tensor too large for a 1-D window at any width must still load
    zero-copy through a 2-D window. 34-byte q8_0 rows divide by 2 only, so the
    widest 1-D window is uint16 and the ceiling is INT32_MAX * 2 bytes. An odd
    row count leaves the tensor no power-of-two stride either, and the filler
    ahead of it puts the data section off a 34-byte boundary, so only the
    window base walking back page by page can align the rows."""
    import resource
    import sys

    rows, row_bytes = 127_000_001, 34  # k=32 q8_0 blocks, ~4.32 GB
    nbytes = rows * row_bytes
    assert nbytes > 2 * 2**31 and rows % 2 == 1

    # Zero fill keeps the pages untouched; a few marked rows carry the check.
    wire = np.zeros((rows, row_bytes), dtype=np.uint8)
    marks = (0, 1, rows // 2, rows - 1)
    for i in marks:
        wire[i] = np.arange(row_bytes, dtype=np.uint8) + (i % 251)

    path = str(tmp_path / "wide.gguf")
    w = GGUFWriter(path, "smoke")
    w.add_tensor(
        "filler.q8", np.zeros((32_768, row_bytes), dtype=np.uint8), raw_dtype=GT.Q8_0
    )
    w.add_tensor("wide.q8", wire, raw_dtype=GT.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss units
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    arrays, codecs, _meta, _shapes = kq.load_gguf(path, True)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale

    a = arrays["wide.q8"]
    assert a.dtype == mx.uint8 and a.shape == (rows, row_bytes)
    assert dict(codecs)["wide.q8"] == "q8_0"
    assert rss1 - rss0 < 512 * 1024 * 1024, "load copied the wire bytes"

    for i in marks:
        assert np.array_equal(np.array(a[i]), wire[i]), i


@pytest.mark.skipif(
    not __import__("os").environ.get("KQUANT_BIG_TESTS"),
    reason="writes a >4 GiB GGUF; set KQUANT_BIG_TESTS=1 to run",
)
def test_load_gguf_wire_over_widest_window_at_file_head(tmp_path):
    """The same oversize tensor as the first one in its file. Its data starts
    in the first page, off a 34-byte boundary, so the window base has no pages
    to walk back over and the rows cannot align. The row count divides by 64,
    so a power-of-two stride carries the window instead."""
    import resource
    import sys

    from gguf import GGUFReader

    rows, row_bytes = 127_000_064, 34  # k=32 q8_0 blocks, ~4.32 GB
    nbytes = rows * row_bytes
    assert nbytes > 2 * 2**31 and rows % 64 == 0

    wire = np.zeros((rows, row_bytes), dtype=np.uint8)
    marks = (0, 1, rows // 2, rows - 1)
    for i in marks:
        wire[i] = np.arange(row_bytes, dtype=np.uint8) + (i % 251)

    path = str(tmp_path / "head.gguf")
    w = GGUFWriter(path, "smoke")
    w.add_tensor("head.q8", wire, raw_dtype=GT.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    del wire

    start = int(GGUFReader(path).tensors[0].data_offset)
    assert start < 16384 and start % row_bytes != 0

    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss units
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    arrays, codecs, _meta, _shapes = kq.load_gguf(path, True)
    a = arrays["head.q8"]
    assert a.dtype == mx.uint8 and a.shape == (rows, row_bytes)
    assert dict(codecs)["head.q8"] == "q8_0"

    for i in marks:
        want = np.arange(row_bytes, dtype=np.uint8) + (i % 251)
        assert np.array_equal(np.array(a[i]), want), i
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    assert rss1 - rss0 < 512 * 1024 * 1024, "load copied the wire bytes"


def test_load_gguf_skip(tmp_path):
    """A skipped tensor gets no array but keeps its shape and codec."""
    _f32, _src = _mint(tmp_path / "smoke.gguf")
    arrays, codecs, _meta, shapes = kq.load_gguf(
        str(tmp_path / "smoke.gguf"), True, ["layer.q8", "plain.f32"]
    )

    assert "layer.q8" not in arrays and "layer.q8.scales" not in arrays
    assert "plain.f32" not in arrays
    assert codecs["layer.q8"] == "q8_0"
    assert list(shapes["layer.q8"]) == [64, 8]
    assert list(shapes["plain.f32"]) == [16, 4]

    # Everything else loads as usual.
    assert arrays["layer.q4"].shape == (8, 64 // 2 + 2 * 2)
    assert codecs["layer.q4"] == "q4_0"

    # Default: nothing skipped.
    arrays, _c, _m, _s = kq.load_gguf(str(tmp_path / "smoke.gguf"))
    assert "layer.q8" in arrays and "plain.f32" in arrays
