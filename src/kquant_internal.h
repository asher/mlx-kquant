// Internal helpers shared across the kq primitive eval paths. Not exported via
// the Python bindings.
#pragma once

#include <string>

#include "mlx/array.h"

namespace mx = mlx::core;

namespace mlx_kquant {

// Replica of mlx::core::get_type_string (mlx/backend/common/compiled.cpp): the
// C++ type token the kq Metal kernels were instantiated with —
//   float32 -> "float", float16 -> "float16_t", bfloat16 -> "bfloat16_t".
// NOTE: the exported type_to_name() yields "float32"/"float16"/"bfloat16" and
// would build a kernel name that does not exist. Throws on other dtypes.
std::string kq_type_string(mx::Dtype d);

// Kernel-name prefix for a codec: "kquant_<codec>_" (mirrors the fork's
// quantized_kname_prefix for mode == "kquant").
inline std::string kq_kname_prefix(const std::string& kquant_type) {
  return "kquant_" + kquant_type + "_";
}

} // namespace mlx_kquant
