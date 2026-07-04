"""mlx-kquant: GGUF K-quant ops for MLX.

Custom Metal kernels packaged as an MLX C++ extension, installed on top of a
stock ``mlx`` wheel.

Current API:
  * ``codecs`` / ``metallib_dir`` / ``metallib_loads`` /
    ``cpu_neon_available`` - toolchain self-checks.
  * ``dequantize`` - GGUF K-quant wire bytes -> float array.
  * ``quantized_matmul`` - x @ dequant(w) for K-quant weights.
  * ``quantized_matmul_qmv_bias`` - decode-only (M=1) bias-fused variant of
    ``quantized_matmul``, q8_0 only.
  * ``gather_qmm`` - mixture-of-experts gathered quantized matmul.
  * ``quantize`` - encode a float tensor into K-quant wire bytes (CPU or Metal).
  * ``load_gguf`` - load a GGUF file's tensors + metadata (C++ mmap memcpy).
"""

# Import mlx.core first: it registers the nanobind type caster for
# ``mlx.core.array``, which every op here accepts and returns. Without it, the
# first array crossing the C++/Python boundary raises an opaque ``std::bad_cast``
# instead of working - so make ``import mlx_kquant`` sufficient on its own.
import mlx.core as _mx  # noqa: F401

from ._ext import (  # noqa: F401
    add_rmsnorm,
    codecs,
    cpu_neon_available,
    dequantize,
    gather_qmm,
    gather_qmv_bias,
    gather_qmv_kq,
    gather_qmv_mix_kq,
    gather_qmv_mix_ns_kq,
    load_gguf,
    metallib_dir,
    metallib_loads,
    moe_glu_gather,
    moe_glu_gather_kq,
    moe_glu_gather_shexp_kq,
    moe_router_topk,
    quantize,
    quantized_matmul,
    quantized_matmul_qmv_bias,
    rmsnorm2_add,
    rmsnorm_multi3,
    sdpa_decode_gqa,
    sdpa_fa_verify,
    sdpa_vector,
)
from ._version import __version__

__all__ = [
    "__version__",
    "add_rmsnorm",
    "codecs",
    "cpu_neon_available",
    "dequantize",
    "gather_qmm",
    "gather_qmv_bias",
    "gather_qmv_kq",
    "gather_qmv_mix_kq",
    "gather_qmv_mix_ns_kq",
    "load_gguf",
    "metallib_dir",
    "metallib_loads",
    "moe_glu_gather",
    "moe_glu_gather_kq",
    "moe_glu_gather_shexp_kq",
    "moe_router_topk",
    "quantize",
    "quantized_matmul",
    "quantized_matmul_qmv_bias",
    "rmsnorm2_add",
    "rmsnorm_multi3",
    "sdpa_decode_gqa",
    "sdpa_fa_verify",
    "sdpa_vector",
]
