// Internal helpers shared across the kq primitive eval paths. Not exported via
// the Python bindings.
#pragma once

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

} // namespace mlx_kquant
