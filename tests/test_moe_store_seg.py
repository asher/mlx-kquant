"""Wide-row expert store op (kq_moe_store_seg.h).

moe_store_seg is gather_qmm_seg with the weights read from store planes.
Per wide codec (the four with field planes and the generic-format codecs
with a wide loader), bfloat16 and float16, 64 and 128 row tiles, sorted
rows and a row map: the output equals gather_qmm_seg on the experts' GGUF
blocks bit for bit. Experts with no rows, one row, a partial tile, a whole
tile and more than one tile are all in the routing, so 64-row tiles of up
to 16 rows take the direct-register path (KQ_SEG_DR) and larger ones the
slab. A store codec with no wide loader reports unsupported and raises.

moe_store_seg_pair runs the gate and up products of a paired codec in one
kernel: its two outputs equal two moe_store_seg calls, and its fused form
equals MLX's compiled silu(gate) * up on them, bit for bit.
"""

import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kqref import is_synth, synth_wire  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="expert store ops are Metal-only kernels; no CPU path.",
)

SPLIT = ("q5_0", "iq4_nl", "iq3_xxs", "iq4_xs")
BLOCK = (
    "iq2_s",
    "q2_0",
    "iq3_s",
    "iq2_xxs",
    "iq2_xs",
    "iq1_s",
    "iq1_m",
    "stq1_0",
    "pq2_0",
    "ptq1_0",
)
E, N, K = 6, 128, 512
COUNTS = (0, 1, 17, 64, 70, 130)


def _wire(rng, codec):
    bpb, wpb = CODEC_GEOMETRY[codec][2:]
    if is_synth(codec):
        w = synth_wire(
            rng,
            codec,
            bpb,
            E * N * (K // wpb),
            d_range=(0.004, 0.01),
            nvfp4_scales=(0x08, 0x19),
        )
        return mx.array(w.reshape(E, N, (K // wpb) * bpb))
    w = mx.array((rng.standard_normal((E, N, K)) * 0.1).astype(np.float32))
    wq, _ = kq.quantize(w, codec)
    mx.eval(wq)
    return wq


def _bits(a):
    mx.eval(a)
    return np.array(a.astype(mx.float32)).tobytes()


@pytest.mark.parametrize("codec", SPLIT + BLOCK)
def test_store_seg_equals_gather_qmm_seg(codec):
    if not kq.moe_store_seg_supported(codec, K):
        pytest.skip("no tensor-op units on this GPU")
    rng = np.random.default_rng(7)
    w = _wire(rng, codec)
    planes = kq.moe_store_pack(w, codec, K, mx.arange(E).astype(mx.uint32))
    scales = mx.zeros((1,), dtype=mx.uint8)
    idx = mx.array(np.repeat(np.arange(E), COUNTS).astype(np.uint32))
    rows = int(idx.size)
    xs = 0.05 if is_synth(codec) else 0.5
    for dt in (mx.bfloat16, mx.float16):
        x = mx.array((rng.standard_normal((rows, K)) * xs).astype(np.float32)).astype(
            dt
        )
        # A row map over fewer source rows, as a prefill's order // k.
        src = mx.array((rng.standard_normal((40, K)) * xs).astype(np.float32)).astype(
            dt
        )
        xr = mx.array(rng.integers(0, 40, size=rows).astype(np.uint32))
        for tr in (64, 128):
            maps = kq.expert_tile_map(idx, E, tile_rows=tr)
            want = kq.gather_qmm_seg(x, w, scales, codec, *maps, tile_rows=tr)
            got = kq.moe_store_seg(x, planes, codec, *maps, n_out=N, tile_rows=tr)
            assert got.shape == want.shape and got.dtype == want.dtype
            assert _bits(got) == _bits(want), (codec, str(dt), tr)
            want = kq.gather_qmm_seg(
                src, w, scales, codec, *maps, tile_rows=tr, x_rows=xr
            )
            got = kq.moe_store_seg(
                src, planes, codec, *maps, n_out=N, tile_rows=tr, x_rows=xr
            )
            assert _bits(got) == _bits(want), (codec, str(dt), tr, "x_rows")


def test_store_seg_refuses_a_codec_with_no_wide_loader():
    if not kq.moe_store_supported("q8_0", ""):
        pytest.skip("no tensor-op units on this GPU")
    assert not kq.moe_store_seg_supported("q8_0", K)
    assert not kq.moe_store_seg_supported("q5_0", 96)
    rng = np.random.default_rng(3)
    w = _wire(rng, "q8_0")
    planes = kq.moe_store_pack(w, "q8_0", K, mx.arange(E).astype(mx.uint32))
    idx = mx.array(np.repeat(np.arange(E), 4).astype(np.uint32))
    maps = kq.expert_tile_map(idx, E, tile_rows=64)
    x = mx.zeros((int(idx.size), K), dtype=mx.bfloat16)
    with pytest.raises(ValueError):
        kq.moe_store_seg(x, planes, "q8_0", *maps, n_out=N)


PAIRED = ("iq3_xxs", "iq4_xs", "iq2_s")


@mx.compile
def _swiglu(gate, up):
    return nn.silu(gate) * up


@pytest.mark.parametrize("codec", PAIRED)
def test_store_seg_pair_equals_two_calls_and_swiglu(codec):
    if not kq.moe_store_seg_supported(codec, K):
        pytest.skip("no tensor-op units on this GPU")
    rng = np.random.default_rng(11)
    slots = mx.arange(E).astype(mx.uint32)
    gp = kq.moe_store_pack(_wire(rng, codec), codec, K, slots)
    up = kq.moe_store_pack(_wire(rng, codec), codec, K, slots)
    idx = mx.array(np.repeat(np.arange(E), COUNTS).astype(np.uint32))
    rows = int(idx.size)
    for dt in (mx.bfloat16, mx.float16):
        x = mx.array((rng.standard_normal((rows, K)) * 0.5).astype(np.float32)).astype(
            dt
        )
        src = mx.array((rng.standard_normal((40, K)) * 0.5).astype(np.float32)).astype(
            dt
        )
        xr = mx.array(rng.integers(0, 40, size=rows).astype(np.uint32))
        for tr in (64, 128):
            maps = kq.expert_tile_map(idx, E, tile_rows=tr)
            for xin, kw in ((x, {}), (src, {"x_rows": xr})):
                tag = (codec, str(dt), tr, bool(kw))
                g = kq.moe_store_seg(xin, gp, codec, *maps, n_out=N, tile_rows=tr, **kw)
                u = kq.moe_store_seg(xin, up, codec, *maps, n_out=N, tile_rows=tr, **kw)
                g2, u2 = kq.moe_store_seg_pair(
                    xin, gp, up, codec, *maps, n_out=N, tile_rows=tr, **kw
                )
                assert _bits(g2) == _bits(g) and _bits(u2) == _bits(u), tag
                (h,) = kq.moe_store_seg_pair(
                    xin, gp, up, codec, *maps, n_out=N, tile_rows=tr, fused=True, **kw
                )
                assert h.shape == g.shape and h.dtype == dt, tag
                assert _bits(h) == _bits(_swiglu(g, u)), tag


def test_store_seg_pair_refuses_an_unpaired_codec():
    if not kq.moe_store_seg_supported("q5_0", K):
        pytest.skip("no tensor-op units on this GPU")
    rng = np.random.default_rng(5)
    planes = kq.moe_store_pack(
        _wire(rng, "q5_0"), "q5_0", K, mx.arange(E).astype(mx.uint32)
    )
    idx = mx.array(np.repeat(np.arange(E), 4).astype(np.uint32))
    maps = kq.expert_tile_map(idx, E, tile_rows=64)
    x = mx.zeros((int(idx.size), K), dtype=mx.bfloat16)
    with pytest.raises(ValueError):
        kq.moe_store_seg_pair(x, planes, planes, "q5_0", *maps, n_out=N)
