"""KVarN Metal kernels vs the numpy oracle (tests/kvarn_ref.py).

At iterations=0 the pipeline has no transcendentals and the kernel must match
the oracle bit for bit (codes, packing and fp16 axes). The full 16-iteration
Sinkhorn goes through precise::exp/log, which drift from numpy's float32
exp/log in last ulps; near-tie best-imbalance selection can then pick an
adjacent iteration's scales. Codes are nearly invariant to that (per-row
rescaling cancels out of the RTN), so full-pipeline comparisons are codes
>= 99.9% exact with |diff| <= 1 and axes within 2% relative. BeeLlama pins
its own CPU/CUDA backends with the same class of tolerance.

Dequant is pure fixed-point arithmetic on record bytes and is asserted
bit-exact against the oracle in both directions.
"""

from __future__ import annotations

import os

import kvarn_ref as ref
import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kvarn_quantize/kvarn_dequant are Metal-only kernels; no CPU path.",
)

SIDES = ("k", "v")


def _rng(seed=11):
    return np.random.default_rng(seed)


def _oracle_tiles(x, bits, kind, iterations=ref.SINKHORN_ITERS):
    """Yield (index, record) for every 128x128 tile of x [..., T, 128]."""
    flat = x.reshape(-1, ref.GROUP, ref.GROUP)
    for i in range(flat.shape[0]):
        yield i, ref.quantize_tile(flat[i], bits, kind, iterations=iterations)


def _kernel_arrays(x, bits, kind, iterations=ref.SINKHORN_ITERS):
    codes, axes = kq.kvarn_quantize(mx.array(x), bits, kind, iterations=iterations)
    mx.eval(codes, axes)
    return np.array(codes), np.array(axes)


def _oracle_record_arrays(x, bits, kind, iterations=ref.SINKHORN_ITERS):
    """Pack oracle records into the kernel's codes/axes array layout."""
    tiles = list(_oracle_tiles(x, bits, kind, iterations=iterations))
    words = np.stack([ref.codes_words(rec["payload"]) for _, rec in tiles])
    axes = np.stack(
        [
            np.stack([rec["scale_axis"], rec["zp_axis"], rec["other_axis"]])
            for _, rec in tiles
        ]
    )
    lead = x.shape[:-2] + (x.shape[-2] // ref.GROUP,)
    return words.reshape(lead + (-1,)), axes.reshape(lead + (3, ref.GROUP))


@pytest.mark.parametrize("kind", SIDES)
@pytest.mark.parametrize("bits", ref.BITS_OK)
def test_quantize_iters0_bit_exact(kind, bits):
    # No Sinkhorn: minmax, RTN, rounding, packing and fp16 axes all pin exactly.
    x = _rng(bits).standard_normal((2, 2, 256, 128)).astype(np.float16)
    codes, axes = _kernel_arrays(x, bits, kind, iterations=0)
    want_codes, want_axes = _oracle_record_arrays(x, bits, kind, iterations=0)
    assert np.array_equal(codes, want_codes)
    assert np.array_equal(axes, want_axes)


@pytest.mark.parametrize("kind", SIDES)
@pytest.mark.parametrize("bits", ref.BITS_OK)
def test_quantize_matches_oracle(kind, bits):
    x = _rng(bits + 100).standard_normal((2, 3, 256, 128)).astype(np.float16)
    codes, axes = _kernel_arrays(x, bits, kind)

    mismatched = 0
    total = 0
    for i, rec in _oracle_tiles(x, bits, kind):
        got = ref.unpack_codes(
            codes.reshape(-1, codes.shape[-1])[i].view(np.uint8), bits
        )
        diff = got.astype(np.int16) - rec["codes"].astype(np.int16)
        assert np.abs(diff).max(initial=0) <= 1
        mismatched += int(np.count_nonzero(diff))
        total += diff.size

        got_axes = axes.reshape(-1, 3, ref.GROUP)[i].astype(np.float32)
        want_axes = np.stack(
            [rec["scale_axis"], rec["zp_axis"], rec["other_axis"]]
        ).astype(np.float32)
        np.testing.assert_allclose(got_axes, want_axes, rtol=0.02, atol=2e-4)
    assert mismatched <= 0.001 * total, f"{mismatched}/{total} code cells differ"


@pytest.mark.parametrize("kind", SIDES)
@pytest.mark.parametrize("bits", (2, 6, 8))
def test_dequant_of_oracle_records_bit_exact(kind, bits):
    # Same record bytes, same fp32 dequant expression, RTNE to fp16: exact.
    x = _rng(bits + 200).standard_normal((2, 2, 256, 128)).astype(np.float16)
    words, axes = _oracle_record_arrays(x, bits, kind)
    out = kq.kvarn_dequant(mx.array(words), mx.array(axes), bits, kind)
    mx.eval(out)
    got = np.array(out).reshape(-1, ref.GROUP, ref.GROUP)

    for i, rec in _oracle_tiles(x, bits, kind):
        want = ref.dequant_tile(rec, kind).astype(np.float16)
        assert np.array_equal(got[i], want)


@pytest.mark.parametrize("kind", SIDES)
@pytest.mark.parametrize("bits", ref.BITS_OK)
def test_kernel_roundtrip_bit_exact_vs_oracle_dequant(kind, bits):
    # Kernel dequant of the kernel's own records == oracle dequant of the
    # parsed bytes. Pins the two kernels to one wire interpretation.
    x = _rng(bits + 300).standard_normal((1, 2, 384, 128)).astype(np.float16)
    codes, axes = _kernel_arrays(x, bits, kind)
    out = kq.kvarn_dequant(mx.array(codes), mx.array(axes), bits, kind)
    mx.eval(out)
    got = np.array(out).reshape(-1, ref.GROUP, ref.GROUP)

    flat_codes = codes.reshape(-1, codes.shape[-1])
    flat_axes = axes.reshape(-1, 3, ref.GROUP)
    for i in range(flat_codes.shape[0]):
        rec = {
            "codes": ref.unpack_codes(flat_codes[i].view(np.uint8), bits),
            "scale_axis": flat_axes[i, 0],
            "zp_axis": flat_axes[i, 1],
            "other_axis": flat_axes[i, 2],
        }
        want = ref.dequant_tile(rec, kind).astype(np.float16)
        assert np.array_equal(got[i], want)


def test_dequant_bfloat16_output():
    x = _rng(4).standard_normal((1, 1, 128, 128)).astype(np.float16)
    words, axes = _oracle_record_arrays(x, 6, "v")
    out = kq.kvarn_dequant(mx.array(words), mx.array(axes), 6, "v", dtype=mx.bfloat16)
    mx.eval(out)
    assert out.dtype == mx.bfloat16
    rec = next(_oracle_tiles(x, 6, "v"))[1]
    want = ref.dequant_tile(rec, "v")
    got = np.array(out.astype(mx.float32)).reshape(ref.GROUP, ref.GROUP)
    # bf16 keeps ~8 mantissa bits of the same fp32 value.
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-3)


def test_quantize_noncontiguous_input():
    x = _rng(5).standard_normal((2, 2, 256, 128)).astype(np.float16)
    xt = mx.array(np.swapaxes(x, 0, 1))
    codes_t, axes_t = kq.kvarn_quantize(mx.swapaxes(xt, 0, 1), 6, "k")
    codes, axes = kq.kvarn_quantize(mx.array(x), 6, "k")
    mx.eval(codes, axes, codes_t, axes_t)
    assert np.array_equal(np.array(codes), np.array(codes_t))
    assert np.array_equal(np.array(axes), np.array(axes_t))


def test_reconstruction_quality_matches_oracle():
    # Near-tie Sinkhorn selection may diverge; both picks must reconstruct
    # equally well.
    x = _rng(6).standard_normal((1, 4, 256, 128)).astype(np.float16)
    codes, axes = _kernel_arrays(x, 6, "k")
    out = kq.kvarn_dequant(mx.array(codes), mx.array(axes), 6, "k")
    mx.eval(out)
    got = np.array(out).astype(np.float32)

    oracle = np.stack(
        [ref.dequant_tile(rec, "k") for _, rec in _oracle_tiles(x, 6, "k")]
    ).reshape(x.shape)
    xf = x.astype(np.float32)
    rmse_kernel = float(np.sqrt(np.mean((got - xf) ** 2)))
    rmse_oracle = float(np.sqrt(np.mean((oracle - xf) ** 2)))
    assert rmse_kernel <= rmse_oracle * 1.02


@pytest.mark.parametrize(
    "call",
    [
        lambda x: kq.kvarn_quantize(x, 7, "k"),
        lambda x: kq.kvarn_quantize(x, 6, "q"),
        lambda x: kq.kvarn_quantize(x, 6, "k", iterations=-1),
        lambda x: kq.kvarn_quantize(x[..., :64], 6, "k"),
        lambda x: kq.kvarn_quantize(x[..., :100, :], 6, "k"),
        lambda x: kq.kvarn_quantize(x.astype(mx.float32), 6, "k"),
    ],
)
def test_quantize_rejects_malformed(call):
    x = mx.zeros((1, 1, 128, 128), dtype=mx.float16)
    with pytest.raises(ValueError):
        call(x)


def _ulp16(a, b):
    def key(x):
        u = np.asarray(x, np.float16).view(np.uint16).astype(np.int64)
        return np.where(u & 0x8000, 0xFFFF - u, u + 0x8000)

    return np.abs(key(a) - key(b))


@pytest.mark.parametrize("head_dim", (128, 256, 512))
def test_rotate_matches_oracle_wht_fp32(head_dim):
    # ref.wht_head is pinned bit-exact against the BeeLlama fixtures, so
    # matching it here chains kvarn_rotate to the normative composite WHT.
    x = (_rng(head_dim).standard_normal((512, head_dim)) * 3).astype(np.float32)
    want = ref.wht_head(x)
    got = np.array(kq.kvarn_rotate(mx.array(x)))
    if head_dim == 128:
        assert np.array_equal(got, want)
    else:
        # mlx's butterfly stage order differs in last fp32 ulps at 256/512.
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=5e-6)


@pytest.mark.parametrize("head_dim", (128, 256, 512))
def test_rotate_fp16_staging_path(head_dim):
    # The fp16 stage must hold the fp32 rotation rounded once, not an fp16
    # butterfly. At 128 this is bit-exact vs the oracle.
    x = (_rng(head_dim + 50).standard_normal((2048, head_dim)) * 3).astype(np.float16)
    want = ref.wht_head(x.astype(np.float32)).astype(np.float16)
    got = np.array(kq.kvarn_rotate(mx.array(x)))
    assert got.dtype == np.float16
    if head_dim == 128:
        assert np.array_equal(got, want)
    else:
        u = _ulp16(got, want)
        absd = np.abs(got.astype(np.float32) - want.astype(np.float32))
        assert np.all((u <= 1) | (absd <= 1e-6))
        assert float((u > 0).mean()) <= 0.002


@pytest.mark.parametrize("head_dim", (128, 256, 512))
def test_rotate_self_inverse(head_dim):
    x = _rng(head_dim + 70).standard_normal((256, head_dim)).astype(np.float16)
    y = np.array(kq.kvarn_rotate(kq.kvarn_rotate(mx.array(x))))
    np.testing.assert_allclose(
        y.astype(np.float32), x.astype(np.float32), rtol=2e-3, atol=2e-3
    )


def test_rotate_dtype_override():
    x = _rng(8).standard_normal((64, 128)).astype(np.float16)
    out = kq.kvarn_rotate(mx.array(x), dtype=mx.float32)
    assert out.dtype == mx.float32
    assert np.array_equal(np.array(out), ref.wht_head(x.astype(np.float32)))


def test_rotate_rejects_bad_head_dim():
    with pytest.raises(ValueError):
        kq.kvarn_rotate(mx.zeros((4, 64), dtype=mx.float16))


@pytest.mark.parametrize("kind", SIDES)
@pytest.mark.parametrize("bits", (2, 6))
def test_full_pipeline_reconstruction(kind, bits):
    # rotate -> quantize -> dequant -> unrotate must reconstruct as well as
    # the oracle chain on the same data.
    x = _rng(bits + 400).standard_normal((1, 2, 256, 128)).astype(np.float16)
    stage = kq.kvarn_rotate(mx.array(x))
    codes, axes = kq.kvarn_quantize(stage, bits, kind)
    recon = kq.kvarn_rotate(kq.kvarn_dequant(codes, axes, bits, kind))
    mx.eval(recon)
    got = np.array(recon).astype(np.float32)

    stage_np = np.array(stage)
    oracle = np.stack(
        [ref.dequant_tile(rec, kind) for _, rec in _oracle_tiles(stage_np, bits, kind)]
    ).reshape(x.shape)
    oracle = ref.wht_head(oracle.reshape(-1, 128)).reshape(x.shape)

    xf = x.astype(np.float32)
    rmse_kernel = float(np.sqrt(np.mean((got - xf) ** 2)))
    rmse_oracle = float(np.sqrt(np.mean((oracle - xf) ** 2)))
    assert rmse_kernel <= rmse_oracle * 1.05
    if bits == 6:
        assert rmse_kernel < 0.05 * float(xf.std())


def test_dequant_rejects_malformed():
    x = mx.zeros((1, 1, 256, 128), dtype=mx.float16)
    codes, axes = kq.kvarn_quantize(x, 6, "k")
    with pytest.raises(ValueError):
        kq.kvarn_dequant(codes, axes, 8, "k")  # width mismatch
    with pytest.raises(ValueError):
        kq.kvarn_dequant(codes.astype(mx.int32), axes, 6, "k")
    with pytest.raises(ValueError):
        kq.kvarn_dequant(codes, axes[..., :64], 6, "k")
    with pytest.raises(ValueError):
        kq.kvarn_dequant(codes, axes[:, :, :1], 6, "k")  # leading shape mismatch
    with pytest.raises(ValueError):
        kq.kvarn_dequant(codes, axes, 6, "k", dtype=mx.float32)
