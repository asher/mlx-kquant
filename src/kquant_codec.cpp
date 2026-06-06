#include "kquant_codec.h"

namespace mlx_kquant {

// Geometry copied verbatim from mlx/primitives.cpp:3458-3473 (kquant fork):
//   {name, weights_per_block, bytes_per_block, bits, has_matmul, has_encode}
static const std::vector<KQuantCodec>& registry() {
  static const std::vector<KQuantCodec> codecs = {
      {"q2_k", 256, 84, 2, true, true},
      {"q3_k", 256, 110, 3, true, true},
      {"q4_k", 256, 144, 4, true, true},
      {"q5_k", 256, 176, 5, true, true},
      {"q6_k", 256, 210, 6, true, true},
      {"q4_0", 32, 18, 4, true, true},
      {"q4_1", 32, 20, 4, true, true},
      {"q5_0", 32, 22, 5, true, true},
      {"q5_1", 32, 24, 5, true, true},
      {"q8_0", 32, 34, 8, true, true},
  };
  return codecs;
}

const KQuantCodec* codec_by_name(const std::string& name) {
  for (const auto& c : registry()) {
    if (c.name == name) {
      return &c;
    }
  }
  return nullptr;
}

std::vector<std::string> codec_names() {
  std::vector<std::string> out;
  out.reserve(registry().size());
  for (const auto& c : registry()) {
    out.push_back(c.name);
  }
  return out;
}

} // namespace mlx_kquant
