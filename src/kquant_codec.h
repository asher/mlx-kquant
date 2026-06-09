// K-quant codec registry: the source of truth for per-codec block geometry
// (weights/bytes per block, bit width, kernel availability).
#pragma once

#include <string>
#include <vector>

namespace mlx_kquant {

struct KQuantCodec {
  std::string name; // "q4_k", "q8_0", ...
  int weights_per_block; // 32 or 256
  int bytes_per_block; // packed wire bytes per block
  int bits; // nominal bit width
  bool has_matmul_kernel; // fused qmm/gather kernels exist
  bool has_encode; // encode (quantize) kernel exists
};

// Returns nullptr if `name` is not a known codec.
const KQuantCodec* codec_by_name(const std::string& name);

// Canonical ordered list of supported codec names.
std::vector<std::string> codec_names();

} // namespace mlx_kquant
