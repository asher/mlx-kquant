"""Packed-weight start alignment on the GPU (kq_weight_base_align).

The GPU kernels need a packed weight to start on a 2-, 4-, 8- or 16-byte
boundary, by codec, and the ops raise ValueError on a start off it instead
of returning wrong values. mxfp4 and nvfp4 run from any start, as do the
CPU kernels of every codec. A weight is moved off its boundary by copying
its bytes into a larger buffer at an offset and passing a view of them.

The GPU tests check the C++ table against BASE_ALIGN. A start at the
table value runs, and a start at half of it raises with the value in the
message.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY, DECODE_ONLY_CODECS

BASE_ALIGN = {
    "q4_k": 16,
    "q5_k": 16,
    "iq4_xs": 8,
    "iq1_m": 8,
    "q2_k": 4,
    "q4_1": 4,
    "q5_1": 4,
    "ptq1_0": 4,
    "mxfp4": 1,
    "nvfp4": 1,
}
# verify_nax's own start check, stricter than BASE_ALIGN for these codecs.
VNAX_BASE = {"pq2_0": 4, "q4_0": 4, "q8_0": 4}
N, K = 256, 512
E, R = 4, 2

GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or mx.default_device() != mx.gpu,
    reason="the alignment check guards the Metal kernels",
)


def _align(codec):
    return BASE_ALIGN.get(codec, 2)


MOE_CODECS = [
    c for c in sorted(CODEC_GEOMETRY) if _align(c) > 1 and kq.codec_has_moe_glu(c)
]


def _wire(codec, shape, seed=3):
    """Packed weight of logical shape (..., K) for codec."""
    rng = np.random.default_rng(seed)
    if codec in DECODE_ONLY_CODECS:
        _, _, bpb, wpb = CODEC_GEOMETRY[codec]
        nb = shape[-1] // wpb
        wire = rng.integers(0, 256, size=(*shape[:-1], nb, bpb), dtype=np.uint8)
        if codec == "mxfp4":
            wire[..., 0] = rng.integers(120, 128, size=wire.shape[:-1])
        w = mx.array(wire.reshape(*shape[:-1], nb * bpb))
        s = mx.zeros((1,), dtype=mx.float32)
    else:
        wf = mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32))
        im = mx.ones((shape[-1],), dtype=mx.float32) if codec.startswith("iq") else None
        w, s = kq.quantize(wf, codec, im)
    mx.eval(w, s)
    return w, s


def _shift(w, offset, evaluate=True):
    buf = mx.concatenate([mx.zeros((offset,), dtype=mx.uint8), w.reshape(-1)])
    mx.eval(buf)
    v = buf[offset:].reshape(w.shape)
    if evaluate:
        mx.eval(v)
    return v


def _equal(a, b):
    a = np.array(a.astype(mx.float32))
    b = np.array(b.astype(mx.float32))
    return np.array_equal(a, b, equal_nan=True)


def _close(a, b):
    # A start that verify_nax declines runs on another route, which rounds
    # differently in bf16 but stays finite and close.
    a = np.array(a.astype(mx.float32))
    b = np.array(b.astype(mx.float32))
    return np.isfinite(a).all() and np.abs(a - b).max() <= 2e-2 * np.abs(b).max()


def _msg(op, codec, off):
    unit = "byte" if off == 1 else "bytes"
    return (
        f"{op}: the {codec} weight starts {off} {unit} past a "
        f"{_align(codec)}-byte boundary"
    )


def _check(call, good, off, msg):
    """call(o) with o None runs on the aligned weight. Offsets in good match
    it, and offset off raises msg from the op call."""
    ref = call(None)
    mx.eval(ref)
    for o in good:
        assert _equal(call(o), ref), o
    with pytest.raises(ValueError, match=msg):
        call(off)


@pytest.mark.parametrize("codec", sorted(CODEC_GEOMETRY))
def test_align_divides_block(codec):
    # A weight that starts on a block boundary of an aligned buffer passes.
    assert codec in kq.codecs()
    assert CODEC_GEOMETRY[codec][2] % _align(codec) == 0


@GPU
@pytest.mark.parametrize("codec", sorted(CODEC_GEOMETRY))
def test_gpu_weight_base(codec):
    w, s = _wire(codec, (N, K))
    a = _align(codec)
    good = sorted({a, 2 * a, 16, 32} if a > 1 else {1, 3, 16})
    for m in (1, 4, 8, 64):
        x = (mx.random.normal((m, K), key=mx.random.key(m)) * 0.5).astype(mx.bfloat16)
        ref = kq.quantized_matmul(x, w, s, codec, transpose=True)
        mx.eval(ref)
        for off in good:
            y = kq.quantized_matmul(x, _shift(w, off), s, codec, transpose=True)
            # verify_nax runs from M 3 through M 8.
            routed = 3 <= m <= 8 and off % VNAX_BASE.get(codec, 1)
            same = _close if routed else _equal
            assert same(y, ref), (m, off)
    ref_d = kq.dequantize(w, s, codec)
    mx.eval(ref_d)
    for off in good:
        assert _equal(kq.dequantize(_shift(w, off), s, codec), ref_d), off
    if a == 1:
        return
    x = (mx.random.normal((8, K), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
    for off in sorted({1, a // 2}):
        wv = _shift(w, off)
        # An evaluated weight raises from the op call, before any graph work.
        with pytest.raises(ValueError, match=_msg("quantized_matmul", codec, off)):
            kq.quantized_matmul(x, wv, s, codec, transpose=True)
        with pytest.raises(ValueError, match=_msg("dequantize", codec, off)):
            kq.dequantize(wv, s, codec)
    # A weight that is not yet evaluated raises from the eval instead.
    y = kq.quantized_matmul(
        x, _shift(w, a // 2, evaluate=False), s, codec, transpose=True
    )
    with pytest.raises(ValueError, match=_msg("quantized_matmul", codec, a // 2)):
        mx.eval(y)
    # The stream keeps working, and a copy in a new buffer runs.
    wc = mx.array(np.array(_shift(w, a // 2)))
    y = kq.quantized_matmul(x, wc, s, codec, transpose=True)
    assert _equal(y, kq.quantized_matmul(x, w, s, codec, transpose=True))


@GPU
def test_gpu_weight_base_qmv_bias():
    codec = "q8_0"
    w, s = _wire(codec, (N, K))
    x = (mx.random.normal((1, K), key=mx.random.key(4)) * 0.5).astype(mx.bfloat16)
    b = (mx.random.normal((N,), key=mx.random.key(5)) * 0.1).astype(mx.bfloat16)

    def call(o):
        wv = w if o is None else _shift(w, o)
        return kq.quantized_matmul_qmv_bias(x, wv, s, b, codec)

    _check(call, (2, 4, 16), 1, _msg("quantized_matmul_qmv_bias", codec, 1))


@GPU
@pytest.mark.parametrize("codec", MOE_CODECS)
def test_gpu_weight_base_gather(codec):
    a = _align(codec)
    ew, es = _wire(codec, (E, N, K))
    uw, _ = _wire(codec, (E, N, K), seed=4)
    dw, _ = _wire(codec, (E, K, N), seed=5)
    sg, _ = _wire(codec, (N, K), seed=6)
    su, _ = _wire(codec, (N, K), seed=7)
    sd, _ = _wire(codec, (K, N), seed=8)
    ids = np.array([[3, 1], [0, 1], [2, 3]], dtype=np.uint32)
    inds = mx.array(ids)
    x = (mx.random.normal((3, K), key=mx.random.key(2)) * 0.05).astype(mx.float16)
    hx = (mx.random.normal((3, R, N), key=mx.random.key(3)) * 0.05).astype(mx.float16)
    hs = (mx.random.normal((3, R + 1, N), key=mx.random.key(6)) * 0.05).astype(
        mx.float16
    )
    sc = mx.array(np.full((3, R), 0.5, dtype=np.float32))
    scs = mx.array(np.full((3, R + 1), 0.5, dtype=np.float32))
    xg = (mx.random.normal((6, 1, K), key=mx.random.key(7)) * 0.05).astype(mx.float16)
    rhs = mx.array(ids.reshape(-1))
    order = np.argsort(ids.reshape(-1), kind="stable")
    xs = (mx.random.normal((6, K), key=mx.random.key(8)) * 0.05).astype(mx.float16)
    tmap = kq.expert_tile_map(mx.array(ids.reshape(-1)[order]), E)
    mx.eval(tmap)

    def at(w, o, j, i):
        # Weight i of the call, moved to offset o when it is the one under test.
        return _shift(w, o) if o is not None and i == j else w

    cases = {
        "moe_glu_gather_kq": (
            2,
            lambda o, j: kq.moe_glu_gather_kq(
                x, at(ew, o, j, 0), at(uw, o, j, 1), codec, inds, act="silu"
            ),
        ),
        "moe_glu_gather_shexp_kq": (
            4,
            lambda o, j: kq.moe_glu_gather_shexp_kq(
                x,
                at(ew, o, j, 0),
                at(uw, o, j, 1),
                at(sg, o, j, 2),
                at(su, o, j, 3),
                codec,
                inds,
                act="silu",
                shexp_kquant_type=codec,
            ),
        ),
        "gather_qmv_kq": (
            1,
            lambda o, j: kq.gather_qmv_kq(hx, at(dw, o, j, 0), codec, inds),
        ),
        "gather_qmv_mix_ns_kq": (
            1,
            lambda o, j: kq.gather_qmv_mix_ns_kq(hx, at(dw, o, j, 0), codec, inds, sc),
        ),
        "gather_qmv_mix_kq": (
            2,
            lambda o, j: kq.gather_qmv_mix_kq(
                hs,
                at(dw, o, j, 0),
                at(sd, o, j, 1),
                codec,
                inds,
                scs,
                shexp_kquant_type=codec,
            ),
        ),
        "gather_qmm": (
            1,
            lambda o, j: kq.gather_qmm(xg, at(ew, o, j, 0), es, codec, rhs_indices=rhs),
        ),
        "gather_qmm_seg": (
            1,
            lambda o, j: kq.gather_qmm_seg(xs, at(ew, o, j, 0), es, codec, *tmap),
        ),
    }
    for op, (n_weights, fn) in cases.items():
        for j in range(n_weights):
            _check(
                lambda o, fn=fn, j=j: fn(o, j),
                sorted({a, 16}),
                a // 2,
                _msg(op, codec, a // 2),
            )


@GPU
def test_gguf_small_alignment_loads(tmp_path):
    # A GGUF with general.alignment 8 puts one of two q4_k tensors 8 bytes
    # off a 16-byte boundary. The loader copies that tensor, so both run.
    gguf = pytest.importorskip("gguf")
    gt = gguf.GGMLQuantizationType
    wa, _ = _wire("q4_k", (N, K), seed=9)
    wb, _ = _wire("q4_k", (N, K), seed=10)
    pad = np.zeros((1, 8), dtype=np.int8)
    path = tmp_path / "align8.gguf"
    wr = gguf.GGUFWriter(str(path), "smoke")
    wr.add_custom_alignment(8)
    wr.add_tensor("pad0", pad, raw_dtype=gt.I8)
    wr.add_tensor("a.weight", np.array(wa), raw_dtype=gt.Q4_K)
    wr.add_tensor("pad1", pad, raw_dtype=gt.I8)
    wr.add_tensor("b.weight", np.array(wb), raw_dtype=gt.Q4_K)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    arrays, codecs, _, _ = kq.load_gguf(str(path))
    x = (mx.random.normal((8, K), key=mx.random.key(9)) * 0.5).astype(mx.bfloat16)
    for name, w in (("a", wa), ("b", wb)):
        assert codecs[f"{name}.weight"] == "q4_k"
        wl = arrays[f"{name}.weight"]
        s = arrays[f"{name}.scales"]
        assert np.array_equal(np.array(wl), np.array(w))
        y = kq.quantized_matmul(x, wl, s, "q4_k", transpose=True)
        assert _equal(y, kq.quantized_matmul(x, w, s, "q4_k", transpose=True))


@pytest.mark.parametrize("codec", ["q2_k", "q4_k", "q8_0", "iq4_xs", "iq1_m", "q6_k"])
def test_cpu_any_weight_base(codec):
    w, s = _wire(codec, (N, K))
    x = (mx.random.normal((1, K), key=mx.random.key(1)) * 0.5).astype(mx.float32)
    ref_q = kq.quantized_matmul(x, w, s, codec, transpose=True, stream=mx.cpu)
    ref_d = kq.dequantize(w, s, codec, stream=mx.cpu)
    for off in (1, 2, 3, 8):
        wv = _shift(w, off)
        y = kq.quantized_matmul(x, wv, s, codec, transpose=True, stream=mx.cpu)
        assert _equal(y, ref_q)
        assert _equal(kq.dequantize(wv, s, codec, stream=mx.cpu), ref_d)
