"""mlx-kquant: standalone GGUF K-quant ops for MLX.

Custom Metal kernels packaged as an MLX C++ extension — runs against a stock,
unmodified ``mlx`` wheel (no MLX-core fork required).

Current API:
  * ``codecs`` / ``metallib_dir`` / ``metallib_loads`` — toolchain self-checks.
  * ``dequantize`` — GGUF K-quant wire bytes -> float array.
  * ``quantized_matmul`` — x @ dequant(w) for K-quant weights.
  * ``gather_qmm`` — mixture-of-experts gathered quantized matmul.
  * ``quantize`` — encode a float tensor into K-quant wire bytes (GPU-only).
  * ``load_gguf`` — load a GGUF file's tensors + metadata (C++ mmap memcpy).
"""

# Import mlx.core first: it registers the nanobind type caster for
# ``mlx.core.array``, which every op here accepts and returns. Without it, the
# first array crossing the C++/Python boundary raises an opaque ``std::bad_cast``
# instead of working — so make ``import mlx_kquant`` sufficient on its own.
import mlx.core as _mx  # noqa: F401

from ._ext import (  # noqa: F401
    codecs,
    dequantize,
    gather_qmm,
    load_gguf,
    metallib_dir,
    metallib_loads,
    quantize,
    quantized_matmul,
)

__all__ = [
    "codecs",
    "dequantize",
    "gather_qmm",
    "load_gguf",
    "metallib_dir",
    "metallib_loads",
    "quantize",
    "quantized_matmul",
]
