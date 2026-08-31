"""gguf.quants-compatible oracle shim adding STQ1_0, which gguf-py lacks.

Exports the STQ1_0 sentinel (the single Python-side type-id constant), a GT
proxy over GGMLQuantizationType, and a quants proxy whose quantize/dequantize
handle the sentinel with the NumPy reference codec below and delegate real enum
members to gguf.quants. Any random qs/sign wire is valid, and
quantize(dequantize(wire)) == wire whenever d > 0.
"""

from __future__ import annotations

import numpy as np
from gguf import GGMLQuantizationType
from gguf import quants as _gguf_quants
from gguf.constants import GGML_QUANT_SIZES

STQ1_0_TYPE_ID = 43  # llama.cpp PR #22836 (unmerged -- may shift)

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


class _Stq1_0Type:
    name = "STQ1_0"
    value = STQ1_0_TYPE_ID

    def __int__(self) -> int:
        return STQ1_0_TYPE_ID

    def __index__(self) -> int:
        return STQ1_0_TYPE_ID

    def __repr__(self) -> str:
        return "STQ1_0"


STQ1_0 = _Stq1_0Type()
# Lets GGUFWriter synthesize STQ1_0 files (a non-enum key is fine).
GGML_QUANT_SIZES[STQ1_0] = (256, 42)
# GGMLQuantizationType(43) resolves to the sentinel instead of raising, so
# GGUFReader opens real STQ1_0 files.
GGMLQuantizationType._value2member_map_.setdefault(STQ1_0_TYPE_ID, STQ1_0)


class _GTProxy:
    STQ1_0 = STQ1_0

    def __getattr__(self, name: str):
        return getattr(GGMLQuantizationType, name)


GT = _GTProxy()


def _is_stq1_0(qtype) -> bool:
    return qtype is STQ1_0 or getattr(qtype, "name", None) == "STQ1_0"


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
    out[:, 40:42] = d.view(np.uint16).astype("<u2").view(np.uint8).reshape(nb, 2)
    return out.reshape(shape[:-1] + (shape[-1] // 256 * 42,))


def dequantize(data: np.ndarray, qtype) -> np.ndarray:
    if _is_stq1_0(qtype):
        return _dequantize_stq1_0(data)
    return _gguf_quants.dequantize(data, qtype)


def quantize(data: np.ndarray, qtype) -> np.ndarray:
    if _is_stq1_0(qtype):
        return _quantize_stq1_0(data)
    return _gguf_quants.quantize(data, qtype)


class _QuantsProxy:
    dequantize = staticmethod(dequantize)
    quantize = staticmethod(quantize)

    def __getattr__(self, name: str):
        return getattr(_gguf_quants, name)


quants = _QuantsProxy()
