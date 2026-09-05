"""KVarN oracle self-tests plus comparisons against the BeeLlama CPU fixtures.

The fixtures (tests/fixtures/kvarn/kvarn_cpu.npz) come from BeeLlama's CPU
reference, which accumulates Sinkhorn stds in double; the oracle pins the fp32
GPU order. Code and scale comparisons are therefore tolerance-based. Byte-level
layout (packing, record parsing, dequant of fixed bytes) is asserted exactly.
"""

from __future__ import annotations

import os

import kvarn_ref as ref
import numpy as np
import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "kvarn", "kvarn_cpu.npz")
TILES = range(4)
SIDES = ("k", "v")


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _rng(seed=7):
    return np.random.default_rng(seed)


def _ulp16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """fp16 ulp distance via an order-preserving uint16 key."""

    def key(x):
        u = np.asarray(x, np.float16).view(np.uint16).astype(np.int64)
        return np.where(u & 0x8000, 0xFFFF - u, u + 0x8000)

    return np.abs(key(a) - key(b))


def test_pack_unpack_roundtrip():
    rng = _rng()
    for bits in ref.BITS_OK:
        codes = rng.integers(0, 1 << bits, size=(ref.GROUP, ref.GROUP), dtype=np.uint8)
        payload = ref.pack_codes(codes, bits)
        assert payload.size == ref.payload_bytes(bits)
        assert np.array_equal(ref.unpack_codes(payload, bits), codes)


def test_pack_layout_lsb_first():
    codes = np.zeros((ref.GROUP, ref.GROUP), np.uint8)
    codes[0, 0] = 0b101010
    codes[0, 1] = 0b010101
    codes[1, 0] = 0b111111
    payload = ref.pack_codes(codes, 6)
    # Row 0: value 0 at bits [0,6), value 1 at bits [6,12).
    assert payload[0] == (0b101010 | ((0b010101 & 0x3) << 6))
    assert payload[1] == 0b010101 >> 2
    # Row 1 starts at byte 96.
    assert payload[96] == 0b111111


def test_half_boundary_rounding_is_half_away():
    # scale is exactly 1 when a row spans [0, qmax], so k + 0.5 sits exactly
    # on a code boundary. Half-away rounds every tie up; RNE would not.
    bits = 4
    qmax = (1 << bits) - 1
    row = np.array([0.0, float(qmax)] + [k + 0.5 for k in range(qmax)], np.float32)
    balanced = np.tile(row, (ref.GROUP, ref.GROUP // row.size + 1))[:, : ref.GROUP]
    ones = np.ones(ref.GROUP, np.float32)
    codes, _, _, _ = ref.rtn_rows(balanced, ones, ones, bits)
    got = codes[0, 2 : 2 + qmax]
    assert np.array_equal(got, np.arange(1, qmax + 1, dtype=np.uint8))


@pytest.mark.parametrize("head_dim", [128, 256, 512])
def test_wht_self_inverse(head_dim):
    rng = _rng(head_dim)
    x = rng.standard_normal((16, head_dim)).astype(np.float32)
    y = ref.wht_head(ref.wht_head(x))
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    assert rmse < 2e-5


@pytest.mark.parametrize("tix", TILES)
@pytest.mark.parametrize("head_dim", [128, 256, 512])
def test_wht_matches_fixture(fx, tix, head_dim):
    x = fx[f"t{tix}_input"].astype(np.float32).reshape(-1, head_dim)
    want = fx[f"t{tix}_wht{head_dim}"].reshape(-1, head_dim)
    got = ref.wht_head(x)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("tix", TILES)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("bits", ref.BITS_OK)
def test_codes_and_axes_match_fixture(fx, tix, side, bits):
    rec = ref.quantize_tile(fx[f"t{tix}_input"], bits, side)
    want = ref.parse_record(fx[f"t{tix}_{side}_b{bits}_record"], bits)

    diff = rec["codes"].astype(np.int16) - want["codes"].astype(np.int16)
    mismatched = np.count_nonzero(diff)
    assert np.abs(diff).max(initial=0) <= 1
    assert mismatched <= 0.001 * diff.size, f"{mismatched} code cells differ"

    for axis in ("scale_axis", "zp_axis", "other_axis"):
        assert _ulp16(rec[axis], want[axis]).max() <= 1, axis


@pytest.mark.parametrize("tix", TILES)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("bits", (2, 6))
def test_dequant_of_fixture_record_is_exact(fx, tix, side, bits):
    # Same record bytes, same fp32 dequant expression: must agree bit-for-bit.
    want = fx[f"t{tix}_{side}_b{bits}_recon"]
    got = ref.dequant_tile(
        ref.parse_record(fx[f"t{tix}_{side}_b{bits}_record"], bits), side
    )
    assert np.array_equal(got, want)


@pytest.mark.parametrize("tix", TILES)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("bits", (2, 6))
def test_full_chain_recon_close(fx, tix, side, bits):
    x = fx[f"t{tix}_input"].astype(np.float32)
    got = ref.dequant_tile(ref.quantize_tile(fx[f"t{tix}_input"], bits, side), side)
    want = fx[f"t{tix}_{side}_b{bits}_recon"]
    bound = 1e-3 * (1.0 + float(np.abs(x).max()))
    rmse = float(np.sqrt(np.mean((got - want) ** 2)))
    assert rmse < bound


@pytest.mark.parametrize("tix", TILES)
@pytest.mark.parametrize("side", SIDES)
def test_sinkhorn_scales_close_to_fixture(fx, tix, side):
    rec = ref.quantize_tile(fx[f"t{tix}_input"], 6, side)
    for name, key in (("s_col", "scol"), ("s_row", "srow")):
        want = fx[f"t{tix}_{side}_{key}"]
        np.testing.assert_allclose(rec[name], want, rtol=1e-3, atol=1e-6, err_msg=name)


def test_reconstruction_error_shrinks_with_bits(fx):
    x = fx["t1_input"].astype(np.float32)
    errs = []
    for bits in (2, 4, 6, 8):
        got = ref.dequant_tile(ref.quantize_tile(fx["t1_input"], bits, "k"), "k")
        errs.append(float(np.sqrt(np.mean((got - x) ** 2))))
    assert errs == sorted(errs, reverse=True)


def test_degenerate_tiles_finite():
    for tile in (
        np.zeros((128, 128)),
        np.full((128, 128), 3.0),
        np.full((128, 128), -1e4),
    ):
        rec = ref.quantize_tile(tile.astype(np.float16), 6, "k")
        out = ref.dequant_tile(rec, "k")
        assert np.isfinite(out).all()


def test_record_roundtrip():
    rng = _rng(3)
    x = rng.standard_normal((128, 128)).astype(np.float16)
    for bits in ref.BITS_OK:
        rec = ref.quantize_tile(x, bits, "v")
        parsed = ref.parse_record(ref.record_bytes(rec), bits)
        assert np.array_equal(parsed["codes"], rec["codes"])
        for axis in ("scale_axis", "zp_axis", "other_axis"):
            assert np.array_equal(parsed[axis], rec[axis])
        words = ref.codes_words(rec["payload"])
        assert words.dtype == np.uint32 and words.size == ref.payload_bytes(bits) // 4
