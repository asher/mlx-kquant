"""Fused low-rank hyper-connection ops vs the eager op-chain reference.

The reference reproduces the gmlx qwen4exp HyperConnection eager path on
dequantized weights, including its rounding points (the down/up qmv
outputs round to the half dtype before their activations). The kernels
accumulate in f32 with different reduction orders, so comparisons carry a
half-grid tolerance rather than bit-exactness.

Metal-only kernels (eval_cpu throws): skipped under KQUANT_FORCE_CPU.

Usage: test_hc_lowrank.py
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType, quants
from kqref import is_synth, synth_wire

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

HC = 4
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="hc lowrank ops are Metal-only kernels; no CPU path.",
)


def _wire(w):
    """f32 [N, K] -> (q8_0 wire uint8 [N, K // 32 * 34], dequantized f32)."""
    n, k = w.shape
    raw = quants.quantize(w.astype(np.float32), GGMLQuantizationType.Q8_0)
    packed = np.ascontiguousarray(raw, dtype=np.uint8).reshape(n, k // 32 * 34)
    deq = quants.dequantize(raw, GGMLQuantizationType.Q8_0).reshape(n, k)
    return mx.array(packed), mx.array(deq.astype(np.float32))


def _mk(seed, R, D, LR, h_dtype, half_dtype):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    h = (mx.random.normal((R, HC, D)) * 1.5).astype(h_dtype)
    if h_dtype != mx.float32:
        h = h.astype(h_dtype)
    gamma = (mx.random.normal((HC * D,)) * 0.3 + 1.0).astype(half_dtype)
    wd, wd_deq = _wire(rng.standard_normal((LR, HC * D)) * 0.05)
    wu, wu_deq = _wire(rng.standard_normal((HC * D, LR)) * 0.2)
    wi = mx.array(rng.standard_normal((HC, HC * D)).astype(np.float32) * 0.02)
    return h, gamma, wd, wd_deq, wu, wu_deq, wi


def _ref(h, gamma, wd_deq, wu_deq, wi, half_dtype):
    """Eager chain on dequantized weights, mirroring gmlx rounding points."""
    R, _, D = h.shape
    xn = mx.fast.rms_norm(h, None, EPS) * gamma.reshape(HC, D)
    xf = xn.reshape(R, HC * D)
    down = (xf.astype(mx.float32) @ wd_deq.T).astype(half_dtype)
    t = down * 0.25
    lo = t * mx.sigmoid(t)
    up = (lo.astype(mx.float32) @ wu_deq.T).astype(half_dtype)
    gate = mx.sigmoid(up).reshape(R, HC, D)
    mixed = (gate.astype(h.dtype) * xn).mean(axis=1)
    inj = 2.0 * mx.sigmoid((xf.astype(mx.float32) @ wi.T) * 0.25)
    return lo, mixed, inj, xn


CASES = [
    (1, 256, 64, mx.float32, mx.bfloat16),
    (4, 256, 64, mx.float32, mx.bfloat16),
    (1, 256, 64, mx.bfloat16, mx.bfloat16),
    (1, 256, 64, mx.float32, mx.float16),
    (1, 256, 64, mx.float16, mx.float16),
    (2, 2560, 320, mx.float32, mx.bfloat16),  # production geometry
    (8, 2560, 320, mx.float32, mx.bfloat16),  # max verify width
    (9, 256, 512, mx.float32, mx.bfloat16),  # max LR
]


# Epilogue forms by lever: the simdgroup kernel at 4, 8 and 16 lanes and 1
# or 8 rows a threadgroup, the one-lane kernel, and the default pick.
EPI_ARMS = {
    "default": {},
    "lpr4": {"KQ_HC_EPI_LPR": "4"},
    "lpr8_rb1": {"KQ_HC_EPI_LPR": "8", "KQ_HC_EPI_RB": "1"},
    "lpr8_rb8": {"KQ_HC_EPI_LPR": "8", "KQ_HC_EPI_RB": "8"},
    "lpr16": {"KQ_HC_EPI_LPR": "16"},
    "serial": {"KQ_HC_EPI_LPR": "0"},
}


@pytest.mark.parametrize("R,D,LR,h_dtype,half_dtype", CASES)
def test_hc_lowrank_front(R, D, LR, h_dtype, half_dtype):
    h, gamma, wd, wd_deq, wu, wu_deq, wi = _mk(3, R, D, LR, h_dtype, half_dtype)
    ref_lo, _, ref_inj, ref_xn = _ref(h, gamma, wd_deq, wu_deq, wi, half_dtype)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    assert xn.dtype == h.dtype and xn.shape == (R, HC, D)
    lo, inj = kq.hc_lowrank_front(xn, wd, wi, half_dtype)
    assert lo.dtype == half_dtype and lo.shape == (R, LR)
    assert inj.dtype == mx.float32 and inj.shape == (R, HC)
    np.testing.assert_allclose(
        np.array(inj),
        np.array(ref_inj),
        rtol=5e-4,
        atol=5e-5,
    )
    # lo rounds to the half grid in both paths; different f32 reduction
    # orders can move a value one ulp.
    np.testing.assert_allclose(
        np.array(lo.astype(mx.float32)),
        np.array(ref_lo.astype(mx.float32)),
        rtol=2e-2 if half_dtype == mx.bfloat16 else 3e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("arm", list(EPI_ARMS))
@pytest.mark.parametrize("R,D,LR,h_dtype,half_dtype", CASES)
def test_hc_lowrank_epilogue(R, D, LR, h_dtype, half_dtype, arm, monkeypatch):
    for k, v in EPI_ARMS[arm].items():
        monkeypatch.setenv(k, v)
    h, gamma, wd, wd_deq, wu, wu_deq, wi = _mk(7, R, D, LR, h_dtype, half_dtype)
    ref_lo, ref_mixed, _, ref_xn = _ref(h, gamma, wd_deq, wu_deq, wi, half_dtype)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    lo, inj = kq.hc_lowrank_front(xn, wd, wi, half_dtype)
    mixed = kq.hc_lowrank_epilogue(lo, wu, xn)
    assert mixed.dtype == h.dtype and mixed.shape == (R, D)
    # Both paths round the up dot to the half grid before the sigmoid
    # gate; different f32 reduction orders (and the reference's TF32 GEMM)
    # flip near-boundary roundings, each worth up to one gate ulp times
    # xn / 4. Bound the tail and the bulk separately.
    d = np.abs(
        np.array(mixed.astype(mx.float32)) - np.array(ref_mixed.astype(mx.float32))
    )
    scale = float(np.abs(np.array(ref_mixed.astype(mx.float32))).mean())
    assert float(d.max()) < 1e-2 * max(scale, 1.0)
    assert float(np.quantile(d, 0.99)) < 5e-3 * max(scale, 1.0)
    assert float(d.mean()) < 1.5e-3 * max(scale, 1.0)


@pytest.mark.parametrize("arm", list(EPI_ARMS))
@pytest.mark.parametrize("R", [2, 4, 5, 8, 16])
def test_hc_lowrank_epilogue_batch_bit_exact(R, arm, monkeypatch):
    """A batched epilogue call matches per-row calls exactly: a row's sums
    take the same order at every row count, in each epilogue form."""
    for k, v in EPI_ARMS[arm].items():
        monkeypatch.setenv(k, v)
    h, gamma, wd, _, wu, _, wi = _mk(17, R, 2560, 320, mx.float32, mx.bfloat16)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    lo, _ = kq.hc_lowrank_front(xn, wd, wi, mx.bfloat16)
    mixed = kq.hc_lowrank_epilogue(lo, wu, xn)
    per_row = mx.concatenate(
        [kq.hc_lowrank_epilogue(lo[r : r + 1], wu, xn[r : r + 1]) for r in range(R)]
    )
    assert mx.array_equal(mixed, per_row)


def test_hc_lowrank_leading_dims():
    """[B, T, 4, D] leading shape round-trips like [R, 4, D]."""
    h, gamma, wd, wd_deq, wu, wu_deq, wi = _mk(11, 4, 256, 64, mx.float32, mx.bfloat16)
    h4 = h.reshape(2, 2, HC, 256)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    xn4 = kq.hc_lowrank_norm(h4, gamma, EPS)
    assert xn4.shape == (2, 2, HC, 256)
    assert mx.array_equal(xn4.reshape(4, HC, 256), xn)
    lo, inj = kq.hc_lowrank_front(xn, wd, wi, mx.bfloat16)
    lo4, inj4 = kq.hc_lowrank_front(xn4, wd, wi, mx.bfloat16)
    assert lo4.shape == (2, 2, 64) and inj4.shape == (2, 2, HC)
    assert mx.array_equal(lo4.reshape(4, 64), lo)
    mixed = kq.hc_lowrank_epilogue(lo, wu, xn)
    mixed4 = kq.hc_lowrank_epilogue(lo4, wu, xn4)
    assert mixed4.shape == (2, 2, 256)
    assert mx.array_equal(mixed4.reshape(4, 256), mixed)


def test_hc_lowrank_bad_shapes():
    h, gamma, wd, _, wu, _, wi = _mk(13, 1, 256, 64, mx.float32, mx.bfloat16)
    with pytest.raises(ValueError):
        kq.hc_lowrank_norm(h[:, :3], gamma, EPS)  # 3 streams
    with pytest.raises(ValueError):
        kq.hc_lowrank_norm(h, gamma[:-1], EPS)  # short gamma
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    with pytest.raises(ValueError):
        kq.hc_lowrank_front(xn, wd[:, :-1], wi, mx.bfloat16)  # bad wire
    with pytest.raises(ValueError):
        kq.hc_lowrank_front(xn, wd, wi[:3], mx.bfloat16)  # 3 inject rows
    with pytest.raises(ValueError):
        kq.hc_lowrank_front(xn, wd, wi, mx.float32)  # non-half lo dtype
    lo, _ = kq.hc_lowrank_front(xn, wd, wi, mx.bfloat16)
    with pytest.raises(ValueError):
        kq.hc_lowrank_epilogue(lo[:, :-2], wu, xn)  # LR mismatch


def _chain(h, gamma, wd, wi, wu):
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    lo, inj = kq.hc_lowrank_front(xn, wd, wi, gamma.dtype)
    return kq.hc_lowrank_epilogue(lo, wu, xn), inj


MIX_CASES = CASES + [
    (3, 2560, 320, mx.float32, mx.bfloat16),
    (5, 2560, 320, mx.float32, mx.float16),
    (16, 2560, 320, mx.float32, mx.bfloat16),
    (13, 2560, 320, mx.bfloat16, mx.bfloat16),
    (2, 192, 32, mx.float32, mx.bfloat16),  # D not a multiple of 256
    (7, 320, 96, mx.float16, mx.float16),
]


@pytest.mark.parametrize("arm", list(EPI_ARMS))
@pytest.mark.parametrize("R,D,LR,h_dtype,half_dtype", MIX_CASES)
def test_hc_lowrank_mix_bits(R, D, LR, h_dtype, half_dtype, arm, monkeypatch):
    """The one-op mixer without the folded norm gives the bits of the three
    ops at each row count, under each epilogue form."""
    monkeypatch.setenv("KQ_HC_MIX_FN", "0")
    for k, v in EPI_ARMS[arm].items():
        monkeypatch.setenv(k, v)
    h, gamma, wd, _, wu, _, wi = _mk(23, R, D, LR, h_dtype, half_dtype)
    want_mixed, want_inj = _chain(h, gamma, wd, wi, wu)
    mx.eval(want_mixed, want_inj)
    mixed, inj = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    assert mixed.dtype == h_dtype and mixed.shape == (R, D)
    assert inj.dtype == mx.float32 and inj.shape == (R, HC)
    assert mx.array_equal(mixed, want_mixed)
    assert mx.array_equal(inj, want_inj)


MMA_ARMS = {
    "lanes": {"KQ_HC_FRONT_MMA": "0", "KQ_HC_EPI_MMA": "0"},
    "front": {"KQ_HC_FRONT_MMA": "1", "KQ_HC_EPI_MMA": "0"},
    "epilogue": {"KQ_HC_FRONT_MMA": "0", "KQ_HC_EPI_MMA": "1"},
    "both": {"KQ_HC_FRONT_MMA": "1", "KQ_HC_EPI_MMA": "1"},
}


@pytest.mark.parametrize("sps", ["1", "2", "4"])
@pytest.mark.parametrize("mma", list(MMA_ARMS))
@pytest.mark.parametrize("R", [1, 2, 5, 8, 9, 16])
def test_hc_lowrank_mix_folded_rows_keep_bits(R, mma, sps, monkeypatch):
    """A row of the folded-norm mixer keeps its bits at every row count up
    to 16, with the MMA front and epilogue and with the lane kernels, at
    each count of lane front simdgroups a stream."""
    for k, v in MMA_ARMS[mma].items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("KQ_HC_FRONT_FN_SPS", sps)
    h, gamma, wd, _, wu, _, wi = _mk(53, 16, 2560, 320, mx.float32, mx.bfloat16)
    want = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    mx.eval(want)
    for r0 in (0, 16 - R):
        m, i = kq.hc_lowrank_mix(h[r0 : r0 + R], gamma, wd, wi, wu, EPS)
        assert mx.array_equal(m, want[0][r0 : r0 + R])
        assert mx.array_equal(i, want[1][r0 : r0 + R])


@pytest.mark.parametrize("mma", list(MMA_ARMS))
@pytest.mark.parametrize("arm", list(EPI_ARMS))
@pytest.mark.parametrize("R,D,LR,h_dtype,half_dtype", MIX_CASES)
def test_hc_lowrank_mix_folded_norm(
    R, D, LR, h_dtype, half_dtype, arm, mma, monkeypatch
):
    """The default mixer (the norm folded into the front for float32
    streams) against the three kernels, which the tests above hold to the
    eager reference, under each epilogue form. The folded sums differ in
    the last float32 bit, which can flip a gate's half rounding."""
    for k, v in EPI_ARMS[arm].items():
        monkeypatch.setenv(k, v)
    for k, v in MMA_ARMS[mma].items():
        monkeypatch.setenv(k, v)
    h, gamma, wd, _, wu, _, wi = _mk(7, R, D, LR, h_dtype, half_dtype)
    mixed, inj = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    mx.eval(mixed, inj)
    assert mixed.dtype == h_dtype and mixed.shape == (R, D)
    monkeypatch.setenv("KQ_HC_MIX_FN", "0")
    want, want_inj = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    mx.eval(want, want_inj)
    np.testing.assert_allclose(np.array(inj), np.array(want_inj), rtol=1e-5, atol=1e-6)
    d = np.abs(np.array(mixed.astype(mx.float32)) - np.array(want.astype(mx.float32)))
    scale = float(np.abs(np.array(want.astype(mx.float32))).mean())
    assert float(d.max()) < 1e-2 * max(scale, 1.0)
    assert float(d.mean()) < 1e-4 * max(scale, 1.0)


@pytest.mark.parametrize("sps", ["2", "4"])
@pytest.mark.parametrize(
    "R,D,LR",
    [
        (1, 2560, 320),
        (3, 2560, 320),
        (8, 2560, 320),
        (16, 2560, 320),
        (2, 192, 32),
        (5, 320, 96),
    ],
)
def test_hc_lowrank_mix_front_split(R, D, LR, sps, monkeypatch):
    """The lane front with two or four simdgroups a stream against the
    three kernels, as test_hc_lowrank_mix_folded_norm holds the default."""
    monkeypatch.setenv("KQ_HC_FRONT_MMA", "0")
    monkeypatch.setenv("KQ_HC_FRONT_FN_SPS", sps)
    h, gamma, wd, _, wu, _, wi = _mk(19, R, D, LR, mx.float32, mx.bfloat16)
    mixed, inj = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    mx.eval(mixed, inj)
    monkeypatch.setenv("KQ_HC_MIX_FN", "0")
    want, want_inj = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    mx.eval(want, want_inj)
    np.testing.assert_allclose(np.array(inj), np.array(want_inj), rtol=1e-5, atol=1e-6)
    d = np.abs(np.array(mixed) - np.array(want))
    scale = float(np.abs(np.array(want)).mean())
    assert float(d.max()) < 1e-2 * max(scale, 1.0)
    assert float(d.mean()) < 1e-4 * max(scale, 1.0)


@pytest.mark.parametrize("no", ["4", "8"])
@pytest.mark.parametrize("rb", ["1", "2", "4", "8"])
@pytest.mark.parametrize("R", [1, 3, 5, 8, 16])
def test_hc_lowrank_mix_folded_rows_a_group_keep_bits(R, rb, no, monkeypatch):
    """The rows and outputs a folded-norm front threadgroup takes leave a
    row's bits as they are, and a row alone gets them too."""
    monkeypatch.setenv("KQ_HC_EPI_MMA", "0")
    monkeypatch.setenv("KQ_HC_FRONT_MMA", "0")
    monkeypatch.setenv("KQ_HC_FRONT_FN_SPS", "1")
    h, gamma, wd, _, wu, _, wi = _mk(43, R, 2560, 320, mx.float32, mx.bfloat16)
    want = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    mx.eval(want)
    monkeypatch.setenv("KQ_HC_FRONT_FN_RB", rb)
    monkeypatch.setenv("KQ_HC_FRONT_FN_NO", no)
    got = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    assert mx.array_equal(got[0], want[0]) and mx.array_equal(got[1], want[1])
    m1, i1 = kq.hc_lowrank_mix(h[R - 1 :], gamma, wd, wi, wu, EPS)
    assert mx.array_equal(m1[0], want[0][R - 1])
    assert mx.array_equal(i1[0], want[1][R - 1])


@pytest.mark.parametrize("sps", ["1", "2"])
@pytest.mark.parametrize("epi_mma", ["0", "1"])
@pytest.mark.parametrize("R", [1, 2, 3, 5, 8, 16])
def test_hc_lowrank_mix_rows_keep_bits(R, epi_mma, sps, monkeypatch):
    """A row's bits do not depend on the rows of the call, with one
    epilogue form and one front split at every row count (the default
    takes the MMA epilogue from 2 rows and two front simdgroups a stream
    at 1 row, whose bits differ from the other counts')."""
    monkeypatch.setenv("KQ_HC_EPI_MMA", epi_mma)
    monkeypatch.setenv("KQ_HC_FRONT_FN_SPS", sps)
    h, gamma, wd, _, wu, _, wi = _mk(29, R, 2560, 320, mx.float32, mx.bfloat16)
    mixed, inj = kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS)
    mx.eval(mixed, inj)
    for r in range(R):
        m1, i1 = kq.hc_lowrank_mix(h[r : r + 1], gamma, wd, wi, wu, EPS)
        assert mx.array_equal(m1[0], mixed[r]), r
        assert mx.array_equal(i1[0], inj[r]), r


@pytest.mark.parametrize("fn", ["1", "0"])
def test_hc_lowrank_mix_operand_views(fn, monkeypatch):
    """Streams that are a broadcast of one stream, a slice of a wider
    batch and a [B, T, 4, D] shape give the bits of their dense copies."""
    monkeypatch.setenv("KQ_HC_MIX_FN", fn)
    h, gamma, wd, _, wu, _, wi = _mk(31, 6, 256, 64, mx.float32, mx.bfloat16)

    def _chain(h, gamma, wd, wi, wu):
        return kq.hc_lowrank_mix(mx.contiguous(h), gamma, wd, wi, wu, EPS)

    want, want_inj = _chain(h, gamma, wd, wi, wu)
    mixed, inj = kq.hc_lowrank_mix(h.reshape(2, 3, HC, 256), gamma, wd, wi, wu, EPS)
    assert mixed.shape == (2, 3, 256) and inj.shape == (2, 3, HC)
    assert mx.array_equal(mixed.reshape(6, 256), want)
    assert mx.array_equal(inj.reshape(6, HC), want_inj)
    mixed, inj = kq.hc_lowrank_mix(h[1:4], gamma, wd, wi, wu, EPS)
    assert mx.array_equal(mixed, want[1:4]) and mx.array_equal(inj, want_inj[1:4])
    one = h[:, :1]
    wide = mx.broadcast_to(one, (6, HC, 256))
    want, want_inj = _chain(mx.contiguous(wide), gamma, wd, wi, wu)
    mixed, inj = kq.hc_lowrank_mix(wide, gamma, wd, wi, wu, EPS)
    assert mx.array_equal(mixed, want) and mx.array_equal(inj, want_inj)


def test_hc_lowrank_mix_bad_shapes():
    h, gamma, wd, _, wu, _, wi = _mk(13, 1, 256, 64, mx.float32, mx.bfloat16)
    with pytest.raises(ValueError):
        kq.hc_lowrank_mix(h[:, :3], gamma, wd, wi, wu, EPS)  # 3 streams
    with pytest.raises(ValueError):
        kq.hc_lowrank_mix(h, gamma[:-1], wd, wi, wu, EPS)  # short gamma
    with pytest.raises(ValueError):
        kq.hc_lowrank_mix(h, gamma, wd[:, :-1], wi, wu, EPS)  # bad down wire
    with pytest.raises(ValueError):
        kq.hc_lowrank_mix(h, gamma, wd, wi[:3], wu, EPS)  # 3 inject rows
    with pytest.raises(ValueError):
        kq.hc_lowrank_mix(h, gamma, wd, wi, wu[:-1], EPS)  # short up wire


@pytest.mark.parametrize("L", ["4", "8", "16"])
@pytest.mark.parametrize("R", [1, 3, 4, 7, 16])
def test_hc_lowrank_epilogue_rows_a_group_keep_bits(R, L, monkeypatch):
    """The rows a threadgroup takes, which a call picks by its rows, leave a
    row's bits as they are."""
    h, gamma, wd, _, wu, _, wi = _mk(37, R, 2560, 320, mx.float32, mx.bfloat16)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    lo, _ = kq.hc_lowrank_front(xn, wd, wi, mx.bfloat16)
    monkeypatch.setenv("KQ_HC_EPI_LPR", L)
    outs = []
    for rb in ("1", "2", "4", "8"):
        monkeypatch.setenv("KQ_HC_EPI_RB", rb)
        outs.append(kq.hc_lowrank_epilogue(lo, wu, xn))
    mx.eval(outs)
    for o in outs[1:]:
        assert mx.array_equal(outs[0], o)


@pytest.mark.parametrize("R", [1, 3, 5, 9, 16])
def test_hc_lowrank_front_rows_a_group_keep_bits(R, monkeypatch):
    """The rows of xn a front threadgroup takes, which a call picks by its
    rows, leave a row's bits as they are, and a row alone gets them too."""
    h, gamma, wd, _, _, _, wi = _mk(47, R, 2560, 320, mx.float32, mx.bfloat16)
    xn = kq.hc_lowrank_norm(h, gamma, EPS)
    outs = []
    for rb in ("1", "2", "4"):
        monkeypatch.setenv("KQ_HC_FRONT_RB", rb)
        outs.append(kq.hc_lowrank_front(xn, wd, wi, mx.bfloat16))
    mx.eval(outs)
    for lo, inj in outs[1:]:
        assert mx.array_equal(outs[0][0], lo)
        assert mx.array_equal(outs[0][1], inj)
    monkeypatch.setenv("KQ_HC_FRONT_RB", "4")
    lo1, inj1 = kq.hc_lowrank_front(xn[R - 1 : R], wd, wi, mx.bfloat16)
    assert mx.array_equal(lo1[0], outs[0][0][R - 1])
    assert mx.array_equal(inj1[0], outs[0][1][R - 1])


# Mixer weights of every codec hc_lowrank_mix reads. A test puts one weight
# in the codec and keeps q8_0 down and up and a float32 inject for the
# other two, the shipped mix.
FLOATS = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}
MIX_CODECS = kq.hc_lowrank_mix_codecs()


def _codec_weight(seed, codec, n, k, rms):
    """[n, k] weights of a codec with about the given rms, and their
    dequantized float32 values."""
    rng = np.random.default_rng(seed)
    if codec in FLOATS:
        w = mx.array((rng.standard_normal((n, k)) * rms).astype(np.float32))
        w = w.astype(FLOATS[codec])
        return w, w.astype(mx.float32)
    bpb, wpb = CODEC_GEOMETRY[codec][2:]

    def deq(w):
        return kq.dequantize(w, mx.zeros((1,), mx.uint8), codec, dtype=mx.float32)

    if not (is_synth(codec) or codec == "mxfp4"):
        w = mx.array((rng.standard_normal((n, k)) * rms).astype(np.float32))
        wq, _ = kq.quantize(w, codec)
        return wq, deq(wq).reshape(n, k)
    nb = n * (k // wpb)

    def synth(f):
        raw = synth_wire(
            np.random.default_rng(seed), codec, bpb, nb, d_range=(0.004 * f, 0.01 * f)
        )
        return raw

    raw = synth(1.0)
    got = float(np.std(np.array(deq(mx.array(raw)))))
    if codec == "mxfp4":
        # e8m0 block scale: one step halves the values.
        raw[:, 0] -= np.uint8(max(0, round(np.log2(got / rms))))
    else:
        raw = synth(rms / got)
    wq = mx.array(raw.reshape(n, k // wpb * bpb))
    return wq, deq(wq).reshape(n, k)


def _mk_codecs(seed, R, D, LR, h_dtype, half_dtype, down, inject, up):
    mx.random.seed(seed)
    h = (mx.random.normal((R, HC, D)) * 1.5).astype(h_dtype)
    gamma = (mx.random.normal((HC * D,)) * 0.3 + 1.0).astype(half_dtype)
    wd, wd_deq = _codec_weight(seed + 1, down, LR, HC * D, 0.05)
    wi, wi_deq = _codec_weight(seed + 2, inject, HC, HC * D, 0.02)
    wu, wu_deq = _codec_weight(seed + 3, up, HC * D, LR, 0.2)
    return h, gamma, (wd, wi, wu), (wd_deq, wi_deq, wu_deq)


def _types(down, inject, up):
    return {
        "down_type": "" if down in FLOATS else down,
        "inject_type": "" if inject in FLOATS else inject,
        "up_type": "" if up in FLOATS else up,
    }


def _ref_codecs(h, gamma, deqs, half_dtype, inj_dtype):
    """_ref with any inject: the eager matmul rounds to its result dtype
    and the gate runs in it. The inject dot runs in float64, clear of the
    GEMM's reduced-precision forms."""
    wd_deq, wi_deq, wu_deq = deqs
    R, _, D = h.shape
    _, mixed, _, xn = _ref(h, gamma, wd_deq, wu_deq, wi_deq, half_dtype)
    xf = np.array(xn.reshape(R, HC * D).astype(mx.float32), dtype=np.float64)
    dot = xf @ np.array(wi_deq, dtype=np.float64).T
    inj = mx.array(dot.astype(np.float32)).astype(inj_dtype)
    return mixed, 2.0 * mx.sigmoid(inj * 0.25)


def _check_mixed(mixed, want):
    """test_hc_lowrank_epilogue's bounds, with the tail at 2.5e-2: over
    thousands of rows a flipped rounding of the up dot meets a large xn."""
    d = np.abs(np.array(mixed.astype(mx.float32)) - np.array(want.astype(mx.float32)))
    scale = max(float(np.abs(np.array(want.astype(mx.float32))).mean()), 1.0)
    assert float(d.max()) < 2.5e-2 * scale
    assert float(np.quantile(d, 0.99)) < 5e-3 * scale


INJ_TOL = {mx.float32: 5e-4, mx.float16: 2e-3, mx.bfloat16: 2.0**-7}


def _mix_codec_case(R, D, LR, h_dtype, half_dtype, down, inject, up):
    h, gamma, ws, deqs = _mk_codecs(41, R, D, LR, h_dtype, half_dtype, down, inject, up)
    kw = _types(down, inject, up)
    mixed, inj = kq.hc_lowrank_mix(h, gamma, *ws, EPS, **kw)
    inj_dtype = h_dtype
    if inject in FLOATS:
        inj_dtype = (mx.zeros((1,), h_dtype) + mx.zeros((1,), FLOATS[inject])).dtype
    assert mixed.dtype == h_dtype and mixed.shape == (R, D)
    assert inj.dtype == inj_dtype and inj.shape == (R, HC)
    want_mixed, want_inj = _ref_codecs(h, gamma, deqs, half_dtype, inj_dtype)
    _check_mixed(mixed, want_mixed)
    np.testing.assert_allclose(
        np.array(inj.astype(mx.float32)),
        np.array(want_inj.astype(mx.float32)),
        rtol=INJ_TOL[inj_dtype],
        atol=1e-5,
    )
    # A row's bits do not depend on the rows of the call (from 2 rows: the
    # folded q8_0 form takes another epilogue for one row).
    n = min(R, 2)
    for r in (0, R - n):
        m1, i1 = kq.hc_lowrank_mix(h[r : r + n], gamma, *ws, EPS, **kw)
        assert mx.array_equal(m1, mixed[r : r + n]), r
        assert mx.array_equal(i1, inj[r : r + n]), r


@pytest.mark.parametrize("role", ["down", "inject", "up"])
@pytest.mark.parametrize("codec", MIX_CODECS)
def test_hc_lowrank_mix_codecs(codec, role):
    """Each codec in each weight, at 6 rows, against the eager chain on the
    dequantized weights. LR 256 takes the up rows of a 256-value block."""
    w = {"down": "q8_0", "inject": "f32", "up": "q8_0", role: codec}
    _mix_codec_case(
        6, 256, 256, mx.float32, mx.bfloat16, w["down"], w["inject"], w["up"]
    )


# The published quants of the model: down and inject of a k-quant, up of a
# 32-value block (its 320 columns), at the production geometry.
PUBLISHED = [
    ("q4_k", "q4_k", "q5_0"),
    ("q6_k", "q4_k", "q5_0"),
    ("q8_0", "f32", "q8_0"),
    ("iq4_xs", "iq4_xs", "iq4_nl"),
    ("q3_k", "q3_k", "q4_0"),
    ("q5_k", "q5_k", "q5_1"),
    ("bf16", "bf16", "bf16"),
    ("f16", "f32", "f16"),
]


@pytest.mark.parametrize("R", [1, 8, 16])
@pytest.mark.parametrize("down,inject,up", PUBLISHED)
def test_hc_lowrank_mix_published(down, inject, up, R):
    _mix_codec_case(R, 2560, 320, mx.float32, mx.bfloat16, down, inject, up)


@pytest.mark.parametrize(
    "h_dtype,half_dtype", [(mx.bfloat16, mx.bfloat16), (mx.float16, mx.float16)]
)
@pytest.mark.parametrize("down,inject,up", PUBLISHED[:2] + PUBLISHED[-2:])
def test_hc_lowrank_mix_codecs_half_streams(down, inject, up, h_dtype, half_dtype):
    _mix_codec_case(5, 2560, 320, h_dtype, half_dtype, down, inject, up)


def test_hc_lowrank_mix_codec_refusals():
    h, gamma, ws, _ = _mk_codecs(
        43, 1, 256, 64, mx.float32, mx.bfloat16, "q8_0", "f32", "q8_0"
    )
    wd, wi, wu = ws
    with pytest.raises(ValueError):
        kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS, down_type="nvfp4")
    with pytest.raises(ValueError):
        kq.hc_lowrank_mix(h, gamma, wd, wi, wu, EPS, up_type="q4_k")  # 256 > LR
    wu_k, _ = _codec_weight(5, "q4_k", HC * 256, 64 * 4, 0.2)
    with pytest.raises(ValueError):
        kq.hc_lowrank_mix(h, gamma, wd, wi, wu_k, EPS, up_type="q4_k")  # 1024 != LR
