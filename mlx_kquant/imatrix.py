"""Importance-matrix (imatrix) loading and HF-path mapping for kquant encoding.

An imatrix is a per-tensor importance vector (one weight per input channel) that
steers the K-quant encoder toward preserving the channels that matter most for a
calibration corpus. :func:`mlx_kquant.quantize.quantize` passes these vectors to
``kq.quantize`` (a no-op on the non-K-quant codecs, which have no
importance-weighted rounding path).

This module is pure numpy (+ optional ``gguf`` for GGUF-format imatrix files) —
no ops, no GPU. It reads both the legacy llama.cpp ``.dat`` format and the newer
GGUF imatrix format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def load_imatrix(path: str) -> dict[str, np.ndarray]:
    """Load an imatrix file (legacy ``.dat`` or GGUF) into ``{name: ndarray}``.

    Args:
        path: Path to the imatrix file.

    Returns:
        dict: Maps tensor name to a float32 importance vector.
    """
    p = Path(path)
    if ".gguf" in p.name:
        return _load_imatrix_gguf(p)
    return _load_imatrix_dat(p)


def _load_imatrix_dat(path: Path) -> dict[str, np.ndarray]:
    import numpy as np

    out: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        data = f.read()
    pos = 0
    (n_entries,) = np.frombuffer(data, dtype=np.int32, count=1, offset=pos)
    pos += 4
    for _ in range(int(n_entries)):
        (name_len,) = np.frombuffer(data, dtype=np.int32, count=1, offset=pos)
        pos += 4
        name = data[pos : pos + int(name_len)].decode("utf-8")
        pos += int(name_len)
        _ncall, nval = np.frombuffer(data, dtype=np.int32, count=2, offset=pos)
        pos += 8
        nval = int(nval)
        vals = np.frombuffer(data, dtype=np.float32, count=nval, offset=pos).copy()
        pos += nval * 4
        out[name] = vals
    return out


def _load_imatrix_gguf(path: Path) -> dict[str, np.ndarray]:
    import numpy as np

    try:
        import gguf  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "GGUF imatrix support requires the 'gguf' package: pip install gguf"
        ) from None
    reader = gguf.GGUFReader(path, "r")
    raw: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        raw[tensor.name] = np.asarray(tensor.data, dtype=np.float32).copy()

    # GGUF imatrix stores .counts (scalar) and .in_sum2 (importance vector)
    # pairs per tensor. Normalize: importance = in_sum2 / counts.
    sum2_keys = [k for k in raw if k.endswith(".in_sum2")]
    if sum2_keys:
        out: dict[str, np.ndarray] = {}
        for s2_key in sum2_keys:
            base = s2_key.removesuffix(".in_sum2")
            counts_key = base + ".counts"
            counts = float(raw[counts_key].item()) if counts_key in raw else 1.0
            if counts > 0:
                out[base] = raw[s2_key] / counts
        return out

    return raw


def map_imatrix_to_hf(
    imatrix: dict[str, np.ndarray],
    hf_paths: list[str],
    arch_string: str | None = None,
) -> dict[str, np.ndarray]:
    """Resolve imatrix tensor names against HF module paths.

    Tries an identity match first (imatrix keyed by HF-format names with a
    ``.weight`` suffix — what this package's ``calibrate-imatrix`` produces).
    Falls back to a GGUF->HF name remap (for llama.cpp imatrix files keyed by
    GGUF names) *if* the ``gguf-mlx`` remap tables are importable — an optional
    integration, not a hard dependency.

    Args:
        imatrix: Raw imatrix data from :func:`load_imatrix`.
        hf_paths: Module paths that will be quantized (bare, no ``.weight``).
        arch_string: Model architecture string, for the GGUF name remap.

    Returns:
        dict: Maps HF module path to a float32 importance vector. Empty if
        nothing could be resolved.
    """
    hf_set = set(hf_paths)
    identity = {
        k.removesuffix(".weight"): v
        for k, v in imatrix.items()
        if k.removesuffix(".weight") in hf_set
    }
    if identity:
        return identity
    if arch_string is None:
        return {}
    try:
        from gguf_mlx.remap import (  # type: ignore[import-not-found]
            RemapDecision,
            parse_gguf_name,
        )
    except ImportError:
        return {}
    out: dict[str, np.ndarray] = {}
    for gguf_name, vec in imatrix.items():
        decision = parse_gguf_name(arch_string, gguf_name)
        if decision.kind != RemapDecision.KIND_MAP or decision.hf_name is None:
            continue
        hf_path = decision.hf_name.removesuffix(".weight")
        if hf_path in hf_set:
            out[hf_path] = vec
    return out
