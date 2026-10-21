"""gguf.quants-compatible oracle shim for the codecs gguf-py lacks.

gguf-py has no STQ1_0, PQ2_0, PTQ1_0, Q2_0 or MXFP8 codec. Each gets a sentinel
type (the single Python-side type-id constant), registered into
``GGML_QUANT_SIZES`` so GGUFWriter can synthesize files and into the enum's
value map so GGUFReader opens real files. STQ1_0 and MXFP8 share type id 43;
the value map keeps STQ1_0, so GGUFReader reads a type-43 tensor as STQ1_0.
``GT`` proxies GGMLQuantizationType with the sentinels added; ``quants``
proxies gguf.quants with quantize/dequantize routed to the NumPy reference
codecs below for the sentinels and delegated otherwise.

STQ1_0: any random qs/sign wire is valid, and quantize(dequantize(wire)) ==
wire whenever d > 0. PQ2_0: any wire is valid; codes 0..3 decode to -1..+2
and the encoder emits only 0..2. Q2_0 is the PQ2_0 layout on a 64-weight
block. PTQ1_0: every byte decodes (the trit
extraction is total over 0..255), but only encoder-produced bytes round-trip.
"""

from __future__ import annotations

import numpy as np
from gguf import GGMLQuantizationType
from gguf import quants as _gguf_quants
from gguf.constants import GGML_QUANT_SIZES

STQ1_0_TYPE_ID = 43  # llama.cpp PR #22836 (unmerged -- may shift)
PQ2_0_TYPE_ID = 142  # PrismML/llama.cpp 8bbb28b76 (Prism-private)
PTQ1_0_TYPE_ID = 143  # PrismML/llama.cpp e19819227 (Prism-private)
Q2_0_TYPE_ID = 42  # llama.cpp GGML_TYPE_Q2_0
MXFP8_TYPE_ID = 43  # llama.cpp PR #30817 (unmerged), the STQ1_0 id too


class _SentinelType:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return self.name


STQ1_0 = _SentinelType("STQ1_0", STQ1_0_TYPE_ID)
PQ2_0 = _SentinelType("PQ2_0", PQ2_0_TYPE_ID)
PTQ1_0 = _SentinelType("PTQ1_0", PTQ1_0_TYPE_ID)
Q2_0 = _SentinelType("Q2_0", Q2_0_TYPE_ID)
MXFP8 = _SentinelType("MXFP8", MXFP8_TYPE_ID)


def _f16_bytes(d: np.ndarray) -> np.ndarray:
    return d.astype(np.float16).view(np.uint16).astype("<u2").view(np.uint8)


def _round_half_away(v: np.ndarray) -> np.ndarray:
    """C ``roundf``/``lroundf``: halves go away from zero, unlike np.round."""
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


# ---------------------------------------------------------------------------
# STQ1_0
# ---------------------------------------------------------------------------

_CODEBOOK = np.array(
    # sign = 0
    [
        0xA9,
        0x89,
        0x29,
        0x09,
        0xA6,
        0x86,
        0x26,
        0x06,
        0x9A,
        0x92,
        0x1A,
        0x12,
        0x6A,
        0x62,
        0x4A,
        0x42,
        # sign = 1
        0x01,
        0x21,
        0x81,
        0xA1,
        0x04,
        0x24,
        0x84,
        0xA4,
        0x10,
        0x18,
        0x90,
        0x98,
        0x40,
        0x48,
        0x60,
        0x68,
    ],
    dtype=np.uint8,
)
_QPACK_TO_SLOT = np.full(256, 0xFF, dtype=np.uint8)
_QPACK_TO_SIGN = np.zeros(256, dtype=np.uint8)
_QPACK_TO_SLOT[_CODEBOOK] = np.arange(32) & 0xF
_QPACK_TO_SIGN[_CODEBOOK] = np.arange(32) >> 4


def _dequantize_stq1_0(data: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(data, dtype=np.uint8)
    shape = data.shape
    blocks = data.reshape(-1, 42)
    nb = blocks.shape[0]
    qs = blocks[:, :32]
    sg = blocks[:, 32:40]
    d = blocks[:, 40:42].copy().view(np.float16).astype(np.float32)
    g = np.arange(64)
    slots = (qs[:, g >> 1] >> (4 * (g & 1))) & 0xF
    signs = (sg[:, g >> 3] >> (g & 7)) & 1
    cb = _CODEBOOK[(signs.astype(np.int32) << 4) | slots]
    p = np.arange(4)
    lanes = ((cb[:, :, None] >> (2 * p)) & 3).astype(np.float32) - 1.0
    # group axis (chunk, gloc) with lane p -> weight chunk*64 + p*16 + gloc
    w = lanes.reshape(nb, 4, 16, 4).transpose(0, 1, 3, 2).reshape(nb, 256)
    return (w * d).reshape(shape[:-1] + (shape[-1] // 42 * 256,))


def _quantize_stq1_0(data: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(data, dtype=np.float32)
    shape = x.shape
    x = x.reshape(-1, 256)
    nb = x.shape[0]
    d = np.abs(x).max(axis=1).astype(np.float16)
    # weight (chunk, p, gloc) -> group lanes v[nb, 64, 4]
    v = x.reshape(nb, 4, 4, 16).transpose(0, 1, 3, 2).reshape(nb, 64, 4)
    zero_pos = np.argmin(np.abs(v), axis=2)  # first min == strict-< scan
    lane = np.where(v < 0.0, 0, 2).astype(np.uint8)
    np.put_along_axis(lane, zero_pos[..., None], np.uint8(1), axis=2)
    qpack = (
        lane[..., 0] | (lane[..., 1] << 2) | (lane[..., 2] << 4) | (lane[..., 3] << 6)
    )
    slot = _QPACK_TO_SLOT[qpack].reshape(nb, 32, 2)
    sign = _QPACK_TO_SIGN[qpack].reshape(nb, 8, 8)
    out = np.zeros((nb, 42), dtype=np.uint8)
    out[:, :32] = slot[:, :, 0] | (slot[:, :, 1] << 4)
    out[:, 32:40] = (sign << np.arange(8)).sum(axis=2).astype(np.uint8)
    out[:, 40:42] = _f16_bytes(d).reshape(nb, 2)
    return out.reshape(shape[:-1] + (shape[-1] // 256 * 42,))


# ---------------------------------------------------------------------------
# PQ2_0: [fp16 d][32 x u8 qs], element j at byte j/4 bits (j%4)*2, code - 1.
# Q2_0: the same layout with 16 qs bytes (wpb 64).
# ---------------------------------------------------------------------------


def _dequantize_pq2_0(data: np.ndarray, wpb: int = 128) -> np.ndarray:
    data = np.ascontiguousarray(data, dtype=np.uint8)
    shape = data.shape
    bpb = 2 + wpb // 4
    blocks = data.reshape(-1, bpb)
    nb = blocks.shape[0]
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    qs = blocks[:, 2:bpb]
    codes = (qs[:, :, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    w = codes.reshape(nb, wpb).astype(np.float32) - 1.0
    return (w * d).reshape(shape[:-1] + (shape[-1] // bpb * wpb,))


def _quantize_pq2_0(data: np.ndarray, wpb: int = 128) -> np.ndarray:
    x = np.ascontiguousarray(data, dtype=np.float32)
    shape = x.shape
    bpb = 2 + wpb // 4
    x = x.reshape(-1, wpb)
    nb = x.shape[0]
    amax = np.abs(x).max(axis=1)
    d = amax.astype(np.float16)
    inv = np.zeros_like(amax)
    np.divide(1.0, amax, out=inv, where=amax > 0.0)
    inv = inv.astype(np.float32)
    q = _round_half_away(x * inv[:, None]).astype(np.int32) + 1
    q = np.clip(q, 0, 3).astype(np.uint8).reshape(nb, wpb // 4, 4)
    out = np.zeros((nb, bpb), dtype=np.uint8)
    out[:, 0:2] = _f16_bytes(d).reshape(nb, 2)
    out[:, 2:bpb] = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    return out.reshape(shape[:-1] + (shape[-1] // wpb * bpb,))


def _dequantize_q2_0(data: np.ndarray) -> np.ndarray:
    return _dequantize_pq2_0(data, 64)


def _quantize_q2_0(data: np.ndarray) -> np.ndarray:
    return _quantize_pq2_0(data, 64)


# ---------------------------------------------------------------------------
# PTQ1_0: [24 x u8 qs][2 x u8 qh][fp16 d]. qs bytes hold five base-3 trits,
# qh bytes four; the element order follows the reference's stage walk
# {32, 16, 8} over qs, then qh. Trit n of byte b is ((b * 3^n) mod 256) * 3
# >> 8; the value is trit - 1.
# ---------------------------------------------------------------------------

_PTQ1_0_STAGES = (32, 16, 8)
_PTQ1_0_QS = 24
_PTQ1_0_QH = 2
_POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint16)


def _ptq1_0_order() -> np.ndarray:
    """(payload byte, trit) for each of the 128 elements, derived from the
    stage walk so it holds at any qs length the stages can tile."""
    order = []
    j = 0
    for c in _PTQ1_0_STAGES:
        while j + c <= _PTQ1_0_QS:
            for n in range(5):
                for m in range(c):
                    order.append((j + m, n))
            j += c
    for n in range(4):
        for h in range(_PTQ1_0_QH):
            order.append((_PTQ1_0_QS + h, n))
    out = np.array(order, dtype=np.int64)
    assert out.shape == (128, 2)
    return out


PTQ1_0_ORDER = _ptq1_0_order()


def _dequantize_ptq1_0(data: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(data, dtype=np.uint8)
    shape = data.shape
    blocks = data.reshape(-1, 28)
    payload = blocks[:, : _PTQ1_0_QS + _PTQ1_0_QH].astype(np.uint16)
    d = blocks[:, 26:28].copy().view(np.float16).astype(np.float32)
    byte_idx, trit = PTQ1_0_ORDER[:, 0], PTQ1_0_ORDER[:, 1]
    q = (payload[:, byte_idx] * _POW3[trit]) & 0xFF
    xi = ((q * 3) >> 8).astype(np.float32) - 1.0
    return (xi * d).reshape(shape[:-1] + (shape[-1] // 28 * 128,))


def _quantize_ptq1_0(data: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(data, dtype=np.float32)
    shape = x.shape
    x = x.reshape(-1, 128)
    nb = x.shape[0]
    amax = np.abs(x).max(axis=1)
    d = amax.astype(np.float16)
    inv = np.zeros_like(amax)
    np.divide(1.0, amax, out=inv, where=amax > 0.0)
    inv = inv.astype(np.float32)
    trits = (_round_half_away(x * inv[:, None]).astype(np.int64) + 1).astype(np.uint16)
    # Trit n weighs 3^(4-n) in its byte; the qh bytes carry four trits and the
    # reference shifts them up one place, which the same weight table does.
    byte_idx, trit = PTQ1_0_ORDER[:, 0], PTQ1_0_ORDER[:, 1]
    weight = np.zeros((128, _PTQ1_0_QS + _PTQ1_0_QH), dtype=np.uint16)
    weight[np.arange(128), byte_idx] = _POW3[4 - trit]
    q5 = trits @ weight  # [nb, 26], each < 243
    packed = ((q5.astype(np.uint32) * 256 + 242) // 243).astype(np.uint8)
    out = np.zeros((nb, 28), dtype=np.uint8)
    out[:, :26] = packed
    out[:, 26:28] = _f16_bytes(d).reshape(nb, 2)
    return out.reshape(shape[:-1] + (shape[-1] // 128 * 28,))


# ---------------------------------------------------------------------------
# MXFP8 (OCP MX): [E8M0 e][32 x E4M3 qs], value = e4m3(q) * 2^(e - 127).
# E4M3 "fn" has no infinity: 0x7F and 0xFF are NaN, and so is e == 0xFF.
# A transcription of the reference's gguf-py MXFP8 class.
# ---------------------------------------------------------------------------


def mxfp8_e4m3_table() -> np.ndarray:
    """float32 value of each of the 256 E4M3 codes."""
    x = np.arange(256, dtype=np.uint32)
    exp = ((x >> 3) & 0xF).astype(np.int32)
    man = (x & 0x7).astype(np.float64)
    v = np.where(exp == 0, man * 2.0**-9, (8.0 + man) * np.exp2(exp - 10.0))
    v = np.where((x & 0x7F) == 0x7F, np.nan, v)
    return np.where(x & 0x80, -v, v).astype(np.float32)


def mxfp8_e8m0_to_fp32(x: np.ndarray) -> np.ndarray:
    """2^(e - 127) as float32; e == 0 is the subnormal 2^-127, 0xFF NaN."""
    x = np.asarray(x).astype(np.uint32)
    bits = np.where(x == 0, np.uint32(0x00400000), x << np.uint32(23))
    return np.where(x == 0xFF, np.float32(np.nan), bits.view(np.float32))


def _fp32_to_e4m3(x: np.ndarray) -> np.ndarray:
    # Round to nearest even, saturating to +-448; NaN -> 0x7F.
    sign = np.where(np.signbit(x), np.uint8(0x80), np.uint8(0))
    a = np.abs(x).astype(np.float32)
    sub = np.rint(a * np.float32(512)).astype(np.int32)
    f, e = np.frexp(a)
    m = np.rint((f * 2 - 1) * 8).astype(np.int32)
    E = e - 1 + 7
    E = np.where(m == 8, E + 1, E)
    m = np.where(m == 8, 0, m)
    norm = (E << 3) | m
    norm = np.where((E > 15) | ((E == 15) & (m == 7)), 0x7E, norm)
    code = np.where(a < 2.0**-6, sub, norm)
    code = np.where(a >= 448.0, 0x7E, code)
    code = np.where(np.isnan(x), 0x7F, code | sign)
    return code.astype(np.uint8)


def _dequantize_mxfp8(data: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(data, dtype=np.uint8)
    shape = data.shape
    blocks = data.reshape(-1, 33)
    d = mxfp8_e8m0_to_fp32(blocks[:, :1])
    with np.errstate(over="ignore", invalid="ignore"):
        w = d * mxfp8_e4m3_table()[blocks[:, 1:]]
    return w.reshape(shape[:-1] + (shape[-1] // 33 * 32,))


def _quantize_mxfp8(data: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(data, dtype=np.float32)
    shape = x.shape
    blocks = x.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.ceil(np.log2(amax / np.float32(448))) + 127, 0)
    e = np.clip(e, 0, 254).astype(np.uint8)
    qs = _fp32_to_e4m3(blocks * (np.float32(1) / mxfp8_e8m0_to_fp32(e)))
    out = np.concatenate([e, qs], axis=-1)
    return out.reshape(shape[:-1] + (shape[-1] // 32 * 33,))


# ---------------------------------------------------------------------------
# Registry and the gguf-py proxies
# ---------------------------------------------------------------------------

# sentinel -> (weights_per_block, bytes_per_block, dequantize, quantize).
# STQ1_0 comes before MXFP8, so the enum value map keeps STQ1_0 for id 43.
SENTINELS = {
    STQ1_0: (256, 42, _dequantize_stq1_0, _quantize_stq1_0),
    PQ2_0: (128, 34, _dequantize_pq2_0, _quantize_pq2_0),
    PTQ1_0: (128, 28, _dequantize_ptq1_0, _quantize_ptq1_0),
    Q2_0: (64, 18, _dequantize_q2_0, _quantize_q2_0),
    MXFP8: (32, 33, _dequantize_mxfp8, _quantize_mxfp8),
}
_BY_NAME = {s.name: s for s in SENTINELS}

for _s, (_wpb, _bpb, _dq, _q) in SENTINELS.items():
    # Lets GGUFWriter synthesize files of the sentinel type (a non-enum key is
    # fine), and GGMLQuantizationType(id) resolve to it so GGUFReader opens
    # real files.
    GGML_QUANT_SIZES[_s] = (_wpb, _bpb)
    GGMLQuantizationType._value2member_map_.setdefault(_s.value, _s)


class _GTProxy:
    STQ1_0 = STQ1_0
    PQ2_0 = PQ2_0
    PTQ1_0 = PTQ1_0
    Q2_0 = Q2_0
    MXFP8 = MXFP8

    def __getattr__(self, name: str):
        return getattr(GGMLQuantizationType, name)


GT = _GTProxy()


def _sentinel(qtype):
    if isinstance(qtype, str):
        return _BY_NAME.get(qtype)
    if qtype in SENTINELS:
        return qtype
    return _BY_NAME.get(getattr(qtype, "name", None))


def dequantize(data: np.ndarray, qtype) -> np.ndarray:
    s = _sentinel(qtype)
    if s is not None:
        return SENTINELS[s][2](data)
    return _gguf_quants.dequantize(data, qtype)


def quantize(data: np.ndarray, qtype) -> np.ndarray:
    s = _sentinel(qtype)
    if s is not None:
        return SENTINELS[s][3](data)
    return _gguf_quants.quantize(data, qtype)


class _QuantsProxy:
    dequantize = staticmethod(dequantize)
    quantize = staticmethod(quantize)

    def __getattr__(self, name: str):
        return getattr(_gguf_quants, name)


quants = _QuantsProxy()


# ---------------------------------------------------------------------------
# Synthesized test wire for the codecs the tests cannot encode through gguf-py
# ---------------------------------------------------------------------------

# fp16 d sits at byte 0 unless listed here; IQ1_M and nvfp4 carry no fp16 d.
_D_OFFSET = {"stq1_0": 40, "ptq1_0": 26}


def is_synth(codec: str) -> bool:
    """Whether the tests feed this codec random structurally-valid wire
    instead of quantizing real values through gguf-py."""
    return codec.startswith("iq") or codec in (
        "nvfp4",
        "stq1_0",
        "pq2_0",
        "ptq1_0",
        "q2_0",
        "mxfp8",
    )


def synth_wire(
    rng: np.random.Generator,
    codec: str,
    bpb: int,
    n_blocks: int,
    *,
    d_range: tuple[float, float] = (0.02, 0.08),
    nvfp4_scales: tuple[int, int] = (0x30, 0x41),
):
    """Random wire with a sane scale so dequant cannot hit Inf/NaN: fp16 d in
    ``d_range`` at the codec's d offset; nvfp4's four ue4m3 group scales at
    bytes 0..3 drawn from ``nvfp4_scales``; mxfp4's and mxfp8's e8m0 scale
    near 1, with mxfp8's two NaN codes redrawn; IQ1_M's fp16 scale in the top
    nibble of each of the four uint16 scale words at offset 48."""
    wire = rng.integers(0, 256, size=(n_blocks, bpb), dtype=np.uint8)
    if codec == "nvfp4":
        lo, hi = nvfp4_scales
        wire[:, 0:4] = rng.integers(lo, hi, (n_blocks, 4), dtype=np.uint8)
        return wire
    if codec == "mxfp4":
        wire[:, 0] = rng.integers(121, 132, n_blocks, dtype=np.uint8)
        return wire
    if codec == "mxfp8":
        wire[:, 0] = rng.integers(121, 132, n_blocks, dtype=np.uint8)
        qs = wire[:, 1:]
        qs[(qs & 0x7F) == 0x7F] &= 0xFE  # 0x7F -> 0x7E, 0xFF -> 0xFE
        return wire
    d = rng.uniform(*d_range, n_blocks).astype(np.float16)
    if codec == "iq1_m":
        dbits = d.view(np.uint16)
        for k, byteidx in enumerate((49, 51, 53, 55)):  # high byte of each word
            nib = ((dbits >> (4 * k)) & 0xF).astype(np.uint8)
            wire[:, byteidx] = (wire[:, byteidx] & 0x0F) | (nib << 4)
        return wire
    off = _D_OFFSET.get(codec, 0)
    wire[:, off : off + 2] = d.view(np.uint8).reshape(n_blocks, 2)
    return wire
