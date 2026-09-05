"""Numpy reference for the KVarN KV-cache quantizer (BeeLlama variant).

Pinned to the fp32 GPU path in beellama.cpp (ggml/src/ggml-cuda/kvarn.cu),
which produced the published KLD results: fp32 accumulators with sequential
accumulation order in the std reductions, round-half-away-from-zero for RTN
codes, and round-to-nearest-even fp16 conversion for stage rows and axis
vectors. The BeeLlama CPU backend accumulates in double and is therefore only
tolerance-comparable; the committed fixtures under tests/fixtures/kvarn/ come
from that CPU code and are checked with tolerances, while the Metal kernels
are held to near-bit-exactness against this module.

Geometry: one record covers a 128-token group of one kv-head slice (128 dims).
K tiles are [row=dim][col=token] (per-channel RTN), V tiles are
[row=token][col=dim] (per-token RTN). A record is the packed payload plus
three fp16 axis vectors: scale_axis[row], zp_axis[row], other_axis[col].
"""

from __future__ import annotations

import numpy as np

GROUP = 128
BITS_OK = (2, 3, 4, 5, 6, 8)
SINKHORN_ITERS = 16

_INV_SQRT_128 = np.float32(0.08838834764831845)
_CROSS_SCALE = {
    1: np.float32(1.0),
    2: np.float32(0.7071067811865475),
    4: np.float32(0.5),
}

_STD_CLIP_LO = np.float32(1e-3)
_STD_CLIP_HI = np.float32(1e3)
_LOG_S_LO = np.float32(-0.3)
_LOG_S_HI = np.float32(10.0)
_IMB_EPS = np.float32(1e-8)
_SCALE_EPS = np.float32(1e-10)


def row_bytes(bits: int) -> int:
    return GROUP * bits // 8


def payload_bytes(bits: int) -> int:
    return GROUP * GROUP * bits // 8


def record_nbytes(bits: int) -> int:
    return payload_bytes(bits) + 3 * GROUP * 2


def _butterfly(x: np.ndarray) -> np.ndarray:
    """In-place-order WHT butterfly over the last axis (a power of two)."""
    y = np.ascontiguousarray(x, dtype=np.float32)
    n = y.shape[-1]
    stride = 1
    while stride < n:
        v = y.reshape(*y.shape[:-1], n // (2 * stride), 2, stride)
        a = v[..., 0, :]
        b = v[..., 1, :]
        y = np.stack((a + b, a - b), axis=-2).reshape(*y.shape[:-1], n)
        stride *= 2
    return y


def wht128(x: np.ndarray) -> np.ndarray:
    """Normalized WHT-128 over the last axis. Self-inverse."""
    if x.shape[-1] != GROUP:
        raise ValueError(f"last axis must be {GROUP}, got {x.shape[-1]}")
    return _butterfly(x) * _INV_SQRT_128


def wht_head(x: np.ndarray) -> np.ndarray:
    """Composite rotation H_slices x H_128 over a head of 128/256/512 dims.

    Matches BeeLlama kvarn_cpu_hadamard_head: per-slice WHT-128, then a
    cross-slice butterfly scaled 1/sqrt(slices). Equals
    mx.hadamard_transform(x, scale=D**-0.5) up to fp accumulation order.
    """
    d = x.shape[-1]
    slices = d // GROUP
    if d not in (128, 256, 512):
        raise ValueError(f"head_dim must be 128/256/512, got {d}")
    y = wht128(
        np.ascontiguousarray(x, dtype=np.float32).reshape(*x.shape[:-1], slices, GROUP)
    )
    if slices > 1:
        y = _butterfly(np.swapaxes(y, -1, -2)) * _CROSS_SCALE[slices]
        y = np.swapaxes(y, -1, -2)
    return y.reshape(*x.shape[:-1], d)


def _seq_stds(cur: np.ndarray, axis: int) -> np.ndarray:
    """Unbiased sample std with sequential fp32 accumulation (CUDA order)."""
    s = np.cumsum(cur, axis=axis, dtype=np.float32)
    s2 = np.cumsum(cur * cur, axis=axis, dtype=np.float32)
    s = np.take(s, -1, axis=axis)
    s2 = np.take(s2, -1, axis=axis)
    mean = s / np.float32(GROUP)
    var = (s2 - np.float32(GROUP) * mean * mean) / np.float32(GROUP - 1)
    return np.sqrt(np.maximum(var, np.float32(0.0)))


def _imbalance(col_std: np.ndarray, row_std: np.ndarray) -> np.float32:
    col = col_std.max() / max(col_std.min(), _IMB_EPS)
    row = row_std.max() / max(row_std.min(), _IMB_EPS)
    return np.float32(col + row)


def variance_normalize(tile: np.ndarray, iterations: int = SINKHORN_ITERS):
    """Log-domain Sinkhorn variance normalization with best-imbalance keep.

    tile is [row, col] fp32. Returns (balanced, s_col, s_row) where
    balanced = tile / (s_col[col] * s_row[row]) for the best-imbalance scales.
    """
    tile = np.ascontiguousarray(tile, dtype=np.float32)
    log_s_col = np.zeros(GROUP, np.float32)
    log_s_row = np.zeros(GROUP, np.float32)
    s_col_best = np.ones(GROUP, np.float32)
    s_row_best = np.ones(GROUP, np.float32)

    cur = tile.copy()
    imb_best = _imbalance(_seq_stds(cur, 0), _seq_stds(cur, 1))

    for _ in range(iterations):
        std_c = np.clip(_seq_stds(cur, 0), _STD_CLIP_LO, _STD_CLIP_HI)
        log_s_col = np.clip(log_s_col + np.log(std_c), _LOG_S_LO, _LOG_S_HI)
        s_col = np.exp(log_s_col)
        s_row = np.exp(log_s_row)
        cur = tile / (s_col[None, :] * s_row[:, None])

        std_r = np.clip(_seq_stds(cur, 1), _STD_CLIP_LO, _STD_CLIP_HI)
        log_s_row = np.clip(log_s_row + np.log(std_r), _LOG_S_LO, _LOG_S_HI)
        s_row = np.exp(log_s_row)
        cur = tile / (s_col[None, :] * s_row[:, None])

        imb = _imbalance(_seq_stds(cur, 0), _seq_stds(cur, 1))
        if imb <= imb_best:
            imb_best = imb
            s_col_best = s_col.copy()
            s_row_best = s_row.copy()

    balanced = tile / (s_col_best[None, :] * s_row_best[:, None])
    return balanced, s_col_best, s_row_best


def _round_half_away(v: np.ndarray) -> np.ndarray:
    """Exact round-half-away for non-negative fp32 (matches roundf/metal round)."""
    f = np.floor(v)
    return f + ((v - f) >= np.float32(0.5)).astype(np.float32)


def rtn_rows(balanced: np.ndarray, s_col: np.ndarray, s_row: np.ndarray, bits: int):
    """Per-row asymmetric RTN plus scale absorption into the axis vectors.

    Returns (codes [row, col] uint8, scale_axis, zp_axis, other_axis) with the
    axis vectors already converted to fp16.
    """
    qmax = (1 << bits) - 1
    lo = balanced.min(axis=1)
    hi = balanced.max(axis=1)
    scale = np.maximum((hi - lo) / np.float32(qmax), _SCALE_EPS)
    q = _round_half_away((balanced - lo[:, None]) / scale[:, None])
    codes = np.clip(q, np.float32(0.0), np.float32(qmax)).astype(np.uint8)
    scale_axis = (s_row * scale).astype(np.float16)
    zp_axis = (s_row * lo).astype(np.float16)
    other_axis = s_col.astype(np.float16)
    return codes, scale_axis, zp_axis, other_axis


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack [row, col] codes into the LSB-first record bitstream (uint8)."""
    flat = codes.reshape(-1, 1).astype(np.uint8)
    stream = np.unpackbits(flat, axis=1, count=bits, bitorder="little").reshape(-1)
    return np.packbits(stream, bitorder="little")


def unpack_codes(payload: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of pack_codes: payload bytes back to [row, col] uint8 codes."""
    stream = np.unpackbits(np.asarray(payload, np.uint8), bitorder="little")
    stream = stream[: GROUP * GROUP * bits].reshape(-1, bits)
    return np.packbits(stream, axis=1, bitorder="little").reshape(GROUP, GROUP)


def quantize_tile(
    x_rot: np.ndarray, bits: int, kind: str, iterations: int = SINKHORN_ITERS
):
    """Quantize one rotated fp16 group [token, dim] into a KVarN record.

    kind selects tile orientation: "k" puts dims on rows (per-channel RTN),
    "v" puts tokens on rows (per-token RTN). Returns a dict with codes,
    payload, the three fp16 axis vectors, and the Sinkhorn intermediates.
    """
    if bits not in BITS_OK:
        raise ValueError(f"bits must be one of {BITS_OK}, got {bits}")
    if kind not in ("k", "v"):
        raise ValueError(f"kind must be 'k' or 'v', got {kind!r}")
    if x_rot.shape != (GROUP, GROUP):
        raise ValueError(f"expected [{GROUP}, {GROUP}] group, got {x_rot.shape}")
    x = np.asarray(x_rot, np.float16).astype(np.float32)
    tile = x if kind == "v" else x.T.copy()
    balanced, s_col, s_row = variance_normalize(tile, iterations)
    codes, scale_axis, zp_axis, other_axis = rtn_rows(balanced, s_col, s_row, bits)
    return {
        "codes": codes,
        "payload": pack_codes(codes, bits),
        "scale_axis": scale_axis,
        "zp_axis": zp_axis,
        "other_axis": other_axis,
        "balanced": balanced,
        "s_col": s_col,
        "s_row": s_row,
    }


def record_bytes(rec: dict) -> np.ndarray:
    """Assemble the BeeLlama interleaved record: payload then the three axes."""
    return np.concatenate(
        [
            rec["payload"],
            rec["scale_axis"].view(np.uint8),
            rec["zp_axis"].view(np.uint8),
            rec["other_axis"].view(np.uint8),
        ]
    )


def parse_record(record: np.ndarray, bits: int) -> dict:
    record = np.asarray(record, np.uint8)
    pb = payload_bytes(bits)
    if record.size != record_nbytes(bits):
        raise ValueError(
            f"record size {record.size} != {record_nbytes(bits)} for bits={bits}"
        )
    return {
        "codes": unpack_codes(record[:pb], bits),
        "payload": record[:pb].copy(),
        "scale_axis": record[pb : pb + 256].view(np.float16).copy(),
        "zp_axis": record[pb + 256 : pb + 512].view(np.float16).copy(),
        "other_axis": record[pb + 512 : pb + 768].view(np.float16).copy(),
    }


def dequant_tile(rec: dict, kind: str) -> np.ndarray:
    """Reconstruct the rotated group [token, dim] fp32 from a (parsed) record."""
    q = rec["codes"].astype(np.float32)
    scale = rec["scale_axis"].astype(np.float32)
    zp = rec["zp_axis"].astype(np.float32)
    other = rec["other_axis"].astype(np.float32)
    tile = (q * scale[:, None] + zp[:, None]) * other[None, :]
    return tile if kind == "v" else tile.T.copy()


def codes_words(payload: np.ndarray) -> np.ndarray:
    """View a record payload as the little-endian uint32 wire words."""
    return np.ascontiguousarray(payload, np.uint8).view(np.uint32)
