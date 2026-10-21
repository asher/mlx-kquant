"""mxfp8: decode, row gather and the matmul ops' refusal.

mxfp8 holds lookup tables (DeepSeek-V4.1 engram n-gram tables): each 33-byte
block is one E8M0 scale byte then 32 E4M3 "fn" codes, value = e4m3(q) *
2^(e - 127). The tables are read by row, so the codec has a CPU and Metal
dequantize and no matmul. The oracle is the reference's numpy decode
(kqref), itself checked against literal OCP values here.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from kqref import mxfp8_e4m3_table, mxfp8_e8m0_to_fp32, quants, synth_wire

import mlx_kquant as kq
from mlx_kquant.codec_geometry import GATHER_ONLY_CODECS
from mlx_kquant.nn import KQuantEmbedding, KQuantLinear

GPU = (
    not os.environ.get("KQUANT_FORCE_CPU")
    and mx.metal.is_available()
    and kq.metallib_loads()
)
STREAMS = [
    pytest.param(mx.cpu, id="cpu"),
    pytest.param(
        mx.gpu, id="gpu", marks=pytest.mark.skipif(not GPU, reason="no Metal GPU")
    ),
]
PLACEHOLDER = mx.zeros((1,), dtype=mx.uint8)

# Every scale class: subnormal products (e near 0), the bf16-exact range,
# the f16-exact range around 127, overflow to inf (e near 254), and NaN.
SCALES = (0, 1, 2, 3, 9, 10, 60, 120, 122, 127, 134, 200, 246, 247, 254, 255)


def _all_codes_wire(scales) -> np.ndarray:
    """One row per scale: the 256 E4M3 codes in order, 8 blocks of 32."""
    codes = np.arange(256, dtype=np.uint8).reshape(8, 32)
    rows = []
    for e in scales:
        blocks = np.concatenate([np.full((8, 1), e, np.uint8), codes], axis=1)
        rows.append(blocks.reshape(-1))
    return np.stack(rows)


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _assert_bits_equal(got: np.ndarray, want: np.ndarray):
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(got), nan)
    assert np.array_equal(_bits(got)[~nan], _bits(want)[~nan])


def test_oracle_matches_ocp_values():
    t = mxfp8_e4m3_table()
    assert t[0x00] == 0.0 and not np.signbit(t[0x00])
    assert t[0x80] == 0.0 and np.signbit(t[0x80])
    assert t[0x01] == 2.0**-9  # smallest subnormal
    assert t[0x07] == 7 * 2.0**-9  # largest subnormal
    assert t[0x08] == 2.0**-6  # smallest normal
    assert t[0x38] == 1.0
    assert t[0x7E] == 448.0 and t[0xFE] == -448.0  # no infinity: max finite
    assert np.isnan(t[0x7F]) and np.isnan(t[0xFF])
    assert np.isfinite(np.delete(t, [0x7F, 0xFF])).all()
    e = mxfp8_e8m0_to_fp32(np.array([0, 1, 127, 254, 255], np.uint8))
    assert e[0] == np.float32(2.0**-127) and e[1] == np.float32(2.0**-126)
    assert e[2] == 1.0 and e[3] == np.float32(2.0**127) and np.isnan(e[4])


def test_oracle_round_trips_e4m3_values():
    t = mxfp8_e4m3_table()
    x = np.where(np.isnan(t), 0.0, t).astype(np.float32).reshape(8, 32)
    x[:, 0] = 448.0  # every block's amax picks e == 127, so d == 1
    wire = quants.quantize(x, "MXFP8")
    assert wire.shape == (8, 33) and (wire[:, 0] == 127).all()
    assert np.array_equal(quants.dequantize(wire, "MXFP8"), x)


@pytest.mark.parametrize("stream", STREAMS)
def test_dequantize_float32_bit_exact(stream):
    """All 256 codes at every scale class: subnormal products, overflow to
    inf, signed zeros and the NaN codes and scale."""
    wire = _all_codes_wire(SCALES)
    want = quants.dequantize(wire, "MXFP8")
    assert np.isinf(want).any() and (np.abs(want[0]) < 2.0**-126).any()
    out = kq.dequantize(
        mx.array(wire), PLACEHOLDER, "mxfp8", dtype=mx.float32, stream=stream
    )
    assert out.shape == (len(SCALES), 256) and out.dtype == mx.float32
    _assert_bits_equal(np.array(out), want)


@pytest.mark.parametrize("stream", STREAMS)
def test_dequantize_half_outputs_exact_in_range(stream):
    """Where the products are normal values of the output type (e in [122,
    134] for float16, [10, 246] for bfloat16) the half outputs are exact.
    Below that the GPU's float to half conversions flush subnormals, as for
    every codec."""
    f16_scales = tuple(range(122, 135))
    wire = _all_codes_wire(f16_scales)
    want = quants.dequantize(wire, "MXFP8")
    out = kq.dequantize(
        mx.array(wire), PLACEHOLDER, "mxfp8", dtype=mx.float16, stream=stream
    )
    got = np.array(out.astype(mx.float32))
    _assert_bits_equal(got, want.astype(np.float16).astype(np.float32))

    bf16_scales = (10, 11, 60, 100, 127, 180, 245, 246)
    wire = _all_codes_wire(bf16_scales)
    want = quants.dequantize(wire, "MXFP8")
    out = kq.dequantize(
        mx.array(wire), PLACEHOLDER, "mxfp8", dtype=mx.bfloat16, stream=stream
    )
    got = np.array(out.astype(mx.float32))
    _assert_bits_equal(got, want)


def test_dequantize_default_dtype_is_float16():
    wire = synth_wire(np.random.default_rng(0), "mxfp8", 33, 16)
    out = kq.dequantize(mx.array(wire.reshape(2, 8 * 33)), PLACEHOLDER, "mxfp8")
    assert out.dtype == mx.float16 and out.shape == (2, 256)


@pytest.mark.parametrize("stream", STREAMS)
def test_dequantize_rejects_partial_blocks(stream):
    wire = mx.zeros((2, 8 * 33 + 1), dtype=mx.uint8)
    with pytest.raises(ValueError, match="bytes_per_block"):
        kq.dequantize(wire, PLACEHOLDER, "mxfp8", stream=stream)


def _table(rng, rows, dims, scales):
    wire = synth_wire(rng, "mxfp8", 33, rows * dims // 32)
    wire[:, 0] = rng.choice(np.asarray(scales, np.uint8), size=wire.shape[0])
    return wire.reshape(rows, dims // 32 * 33)


def test_embedding_gathers_and_decodes_rows():
    """A table's rows gathered by token id, the way an engram lookup reads
    them, at the module's bfloat16 output dtype."""
    rng = np.random.default_rng(4)
    rows, dims = 96, 256
    wire = _table(rng, rows, dims, range(110, 140))
    emb = KQuantEmbedding(num_embeddings=rows, dims=dims, codec="mxfp8")
    emb.weight = mx.array(wire)
    ids = np.array([[3, 0, 95, 41], [7, 7, 1, 60]], dtype=np.int32)
    out = emb(mx.array(ids))
    assert out.shape == (2, 4, dims) and out.dtype == mx.bfloat16
    want = quants.dequantize(wire, "MXFP8")[ids]
    _assert_bits_equal(np.array(out.astype(mx.float32)), want)


def test_embedding_keeps_values_past_float16_range():
    """Values past the float16 range (scales above 2^15) come through a
    float32 module exactly, not as float16 inf."""
    rng = np.random.default_rng(5)
    rows, dims = 8, 256
    wire = _table(rng, rows, dims, (150, 200, 240))
    emb = KQuantEmbedding(num_embeddings=rows, dims=dims, codec="mxfp8")
    emb.weight = mx.array(wire)
    emb.out_dtype = mx.float32
    ids = np.array([5, 2, 7], dtype=np.int32)
    want = quants.dequantize(wire, "MXFP8")[ids]
    assert np.abs(want).max() > 65504.0
    _assert_bits_equal(np.array(emb(mx.array(ids))), want)


def test_float_fuse_keeps_values_past_float16_range():
    """The fuse paths that decode a table to a float embedding (the fuse
    CLI's --dequantize and a LoRA embedding's fuse(dequantize=True)) keep
    values past the float16 range, as the module's own lookup does."""
    pytest.importorskip("mlx_lm")
    from mlx_kquant.cli.fuse import _dequantize_remaining
    from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora

    rows, dims = 8, 256
    wire = _table(np.random.default_rng(7), rows, dims, (150, 200, 240))
    want = quants.dequantize(wire, "MXFP8")
    assert np.abs(want).max() > 65504.0

    def table():
        emb = KQuantEmbedding(num_embeddings=rows, dims=dims, codec="mxfp8")
        emb.weight = mx.array(wire)
        return emb

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = table()

    model = Model()
    assert _dequantize_remaining(model) == 1
    assert isinstance(model.embed, nn.Embedding)
    _assert_bits_equal(np.array(model.embed.weight.astype(mx.float32)), want)

    patch_mlx_lm_lora()
    lora = table().to_lora(r=4, scale=1.0, dropout=0.0)  # lora_b is zero
    fused = lora.fuse(dequantize=True)
    assert isinstance(fused, nn.Embedding)
    # Equal values: adding the zero delta turns -0.0 into +0.0.
    assert np.array_equal(np.array(fused.weight.astype(mx.float32)), want)


def test_codec_registry():
    assert "mxfp8" in kq.codecs()
    assert not kq.codec_has_matmul("mxfp8")
    assert not kq.codec_has_moe_glu("mxfp8")
    assert {c for c in kq.codecs() if not kq.codec_has_matmul(c)} == set(
        GATHER_ONLY_CODECS
    )


def _matmul_calls():
    k, n = 256, 16
    w = mx.array(_table(np.random.default_rng(6), n, k, (127,)))
    x = mx.ones((2, k), dtype=mx.bfloat16)
    ex = mx.stack([w, w])  # two experts
    idx = mx.array(np.array([[0], [1]], dtype=np.uint32))
    lora_a = mx.zeros((k, 4), dtype=mx.bfloat16)
    lora_b = mx.zeros((4, n), dtype=mx.bfloat16)
    lin = KQuantLinear(in_dims=k, out_dims=n, bias=False, codec="mxfp8")
    lin.weight = w
    emb = KQuantEmbedding(num_embeddings=n, dims=k, codec="mxfp8")
    emb.weight = w
    return {
        "quantized_matmul": lambda: kq.quantized_matmul(x, w, PLACEHOLDER, "mxfp8"),
        "quantized_matmul_n": lambda: kq.quantized_matmul(
            mx.ones((2, n), dtype=mx.bfloat16), w, PLACEHOLDER, "mxfp8", transpose=False
        ),
        "gather_qmm": lambda: kq.gather_qmm(
            x[:, None, :], ex, PLACEHOLDER, "mxfp8", rhs_indices=idx
        ),
        "gather_qmm_seg": lambda: kq.gather_qmm_seg(
            x,
            ex,
            PLACEHOLDER,
            "mxfp8",
            mx.zeros((1, 3), dtype=mx.uint32),
            mx.ones((1,), dtype=mx.uint32),
        ),
        "quantized_matmul_multi": lambda: kq.quantized_matmul_multi(x, [w, w], "mxfp8"),
        "quantized_matmul_glu": lambda: kq.quantized_matmul_glu(x, w, w, "mxfp8"),
        "quantized_matmul_lora": lambda: kq.quantized_matmul_lora(
            x, w, "mxfp8", lora_a, lora_b
        ),
        "quantized_matmul_residual": lambda: kq.quantized_matmul_residual(
            x, w, "mxfp8", mx.zeros((2, n), dtype=mx.bfloat16)
        ),
        "KQuantLinear": lambda: lin(x),
        "KQuantEmbedding.as_linear": lambda: emb.as_linear(x),
    }


@pytest.mark.parametrize("stream", STREAMS)
@pytest.mark.parametrize("op", sorted(_matmul_calls()))
def test_matmul_ops_refuse(op, stream):
    with mx.stream(stream):
        call = _matmul_calls()[op]
        with pytest.raises(ValueError, match="gather-only"):
            mx.eval(call())


def test_quantize_refuses():
    with pytest.raises(ValueError, match="mxfp8"):
        kq.quantize(mx.ones((4, 256)), "mxfp8")
