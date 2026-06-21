// Internal helpers shared across the kq primitive eval paths. Not exported via
// the Python bindings.
#pragma once

#include <cstdint>
#include <string>

#include "mlx/array.h"

namespace mx = mlx::core;

namespace mlx_kquant {

// Map a dtype to the C++ type token the kq Metal kernels are instantiated with:
//   float32 -> "float", float16 -> "float16_t", bfloat16 -> "bfloat16_t".
// Throws on other dtypes.
std::string kq_type_string(mx::Dtype d);

// Kernel-name prefix for a codec: "kquant_<codec>_".
inline std::string kq_kname_prefix(const std::string& kquant_type) {
  return "kquant_" + kquant_type + "_";
}

// 64-bit variant of mx::elem_to_loc (mlx/backend/common/utils.h), whose linear
// index parameter is a 32-bit int. That truncates offsets into tensors whose
// flattened element count exceeds 2^31 - e.g. a stacked MoE expert weight whose
// per-expert byte span is itself > 2 GB. This mirrors the stock per-dimension
// loop with a 64-bit index so the decomposition stays exact for large tensors.
inline int64_t elem_to_loc64(
    int64_t elem,
    const mx::Shape& shape,
    const mx::Strides& strides) {
  int64_t loc = 0;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    const int64_t dim = static_cast<int64_t>(shape[i]);
    const int64_t q = elem / dim;
    loc += (elem - q * dim) * strides[i];
    elem = q;
  }
  return loc;
}

} // namespace mlx_kquant
