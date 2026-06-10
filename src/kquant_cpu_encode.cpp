// Scalar CPU encoders for the flat GGUF codecs (q8_0 + the four legacy block
// codecs). Each function is a direct port of the matching Metal kernel impl in
// kq_quantized_encode.h: same amax/min-max scale derivation, same
// round-and-pack (Metal `round` is round-half-away-from-zero -> std::round on
// float), and the scale `d` (and `m`) are written through fp16 exactly as
// `half(...)` does, so a CPU-encoded block is byte-identical to a GPU-encoded
// one. The K-quant codecs are GPU-only for now; the dispatch throws for them on
// CPU.
#include "kquant_cpu_encode.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kquant_codec.h"

#include "mlx/types/half_types.h"

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// Write a float as a 2-byte fp16 (round-to-nearest-even), matching the kernel's
// `half(v)` store; the K-quant re-quantize phase reads this back, so the round
// trip must be identical between CPU and GPU.
inline void write_f16(uint8_t* ptr, float v) {
  _Float16 h = static_cast<_Float16>(v);
  std::memcpy(ptr, &h, sizeof(_Float16));
}

template <typename T>
void quantize_q8_0(const T* w, uint8_t* out, std::size_t num_weights) {
  constexpr int group = 32;
  constexpr int block_bytes = 34; // 2 (d) + 32 (qs)
  std::size_t num_blocks = num_weights / group;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x = w + b * group;
    uint8_t* block = out + b * block_bytes;

    float amax = 0.0f;
    for (int j = 0; j < group; j++) {
      amax = std::max(amax, std::fabs(static_cast<float>(x[j])));
    }
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    write_f16(block, d);
    int8_t* qs = reinterpret_cast<int8_t*>(block + 2);
    for (int j = 0; j < group; j++) {
      float v = static_cast<float>(x[j]) * id;
      qs[j] = static_cast<int8_t>(std::clamp(std::round(v), -127.0f, 127.0f));
    }
  }
}

template <typename T>
void quantize_q4_0(const T* w, uint8_t* out, std::size_t num_weights) {
  constexpr int group = 32;
  constexpr int block_bytes = 18; // 2 (d) + 16 (qs)
  std::size_t num_blocks = num_weights / group;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x = w + b * group;
    uint8_t* block = out + b * block_bytes;

    float amax = 0.0f;
    for (int j = 0; j < group; j++) {
      amax = std::max(amax, std::fabs(static_cast<float>(x[j])));
    }
    const float d = amax / 7.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    write_f16(block, d);
    uint8_t* qs = block + 2;
    for (int j = 0; j < 16; j++) {
      float x0 = static_cast<float>(x[j]) * id;
      float x1 = static_cast<float>(x[j + 16]) * id;
      uint8_t q0 =
          static_cast<uint8_t>(std::clamp(std::round(x0) + 8.0f, 0.0f, 15.0f));
      uint8_t q1 =
          static_cast<uint8_t>(std::clamp(std::round(x1) + 8.0f, 0.0f, 15.0f));
      qs[j] = q0 | (q1 << 4);
    }
  }
}

template <typename T>
void quantize_q4_1(const T* w, uint8_t* out, std::size_t num_weights) {
  constexpr int group = 32;
  constexpr int block_bytes = 20; // 2 (d) + 2 (m) + 16 (qs)
  std::size_t num_blocks = num_weights / group;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x = w + b * group;
    uint8_t* block = out + b * block_bytes;

    float vmin = static_cast<float>(x[0]);
    float vmax = static_cast<float>(x[0]);
    for (int j = 1; j < group; j++) {
      float v = static_cast<float>(x[j]);
      vmin = std::min(vmin, v);
      vmax = std::max(vmax, v);
    }
    const float d = (vmax - vmin) / 15.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    write_f16(block, d);
    write_f16(block + 2, vmin);
    uint8_t* qs = block + 4;
    for (int j = 0; j < 16; j++) {
      float x0 = (static_cast<float>(x[j]) - vmin) * id;
      float x1 = (static_cast<float>(x[j + 16]) - vmin) * id;
      uint8_t q0 =
          static_cast<uint8_t>(std::clamp(std::round(x0), 0.0f, 15.0f));
      uint8_t q1 =
          static_cast<uint8_t>(std::clamp(std::round(x1), 0.0f, 15.0f));
      qs[j] = q0 | (q1 << 4);
    }
  }
}

// Pack the 5th bit of each of 32 quants into a little-endian uint32 high-bit
// field, matching the kernel's byte-wise qh store (endian-safe).
inline void write_qh32(uint8_t* qh_p, uint32_t qh) {
  qh_p[0] = static_cast<uint8_t>(qh & 0xFF);
  qh_p[1] = static_cast<uint8_t>((qh >> 8) & 0xFF);
  qh_p[2] = static_cast<uint8_t>((qh >> 16) & 0xFF);
  qh_p[3] = static_cast<uint8_t>((qh >> 24) & 0xFF);
}

template <typename T>
void quantize_q5_0(const T* w, uint8_t* out, std::size_t num_weights) {
  constexpr int group = 32;
  constexpr int block_bytes = 22; // 2 (d) + 4 (qh) + 16 (qs)
  std::size_t num_blocks = num_weights / group;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x = w + b * group;
    uint8_t* block = out + b * block_bytes;

    float amax = 0.0f;
    for (int j = 0; j < group; j++) {
      amax = std::max(amax, std::fabs(static_cast<float>(x[j])));
    }
    const float d = amax / 15.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    write_f16(block, d);
    uint8_t* qh_p = block + 2;
    uint8_t* qs = block + 6;
    uint32_t qh = 0;
    for (int j = 0; j < 16; j++) {
      float v0 = static_cast<float>(x[j]) * id;
      float v1 = static_cast<float>(x[j + 16]) * id;
      uint8_t q0 =
          static_cast<uint8_t>(std::clamp(std::round(v0) + 16.0f, 0.0f, 31.0f));
      uint8_t q1 =
          static_cast<uint8_t>(std::clamp(std::round(v1) + 16.0f, 0.0f, 31.0f));
      qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
      qh |= (static_cast<uint32_t>(q0 >> 4) << j);
      qh |= (static_cast<uint32_t>(q1 >> 4) << (j + 16));
    }
    write_qh32(qh_p, qh);
  }
}

template <typename T>
void quantize_q5_1(const T* w, uint8_t* out, std::size_t num_weights) {
  constexpr int group = 32;
  constexpr int block_bytes = 24; // 2 (d) + 2 (m) + 4 (qh) + 16 (qs)
  std::size_t num_blocks = num_weights / group;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x = w + b * group;
    uint8_t* block = out + b * block_bytes;

    float vmin = static_cast<float>(x[0]);
    float vmax = static_cast<float>(x[0]);
    for (int j = 1; j < group; j++) {
      float v = static_cast<float>(x[j]);
      vmin = std::min(vmin, v);
      vmax = std::max(vmax, v);
    }
    const float d = (vmax - vmin) / 31.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    write_f16(block, d);
    write_f16(block + 2, vmin);
    uint8_t* qh_p = block + 4;
    uint8_t* qs = block + 8;
    uint32_t qh = 0;
    for (int j = 0; j < 16; j++) {
      float v0 = (static_cast<float>(x[j]) - vmin) * id;
      float v1 = (static_cast<float>(x[j + 16]) - vmin) * id;
      uint8_t q0 =
          static_cast<uint8_t>(std::clamp(std::round(v0), 0.0f, 31.0f));
      uint8_t q1 =
          static_cast<uint8_t>(std::clamp(std::round(v1), 0.0f, 31.0f));
      qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
      qh |= (static_cast<uint32_t>(q0 >> 4) << j);
      qh |= (static_cast<uint32_t>(q1 >> 4) << (j + 16));
    }
    write_qh32(qh_p, qh);
  }
}

} // namespace

template <typename T>
void kquant_quantize_dispatch(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const std::string& kquant_type,
    const float* imatrix,
    std::size_t K) {
  // The flat codecs have no importance-weighted rounding path.
  (void)imatrix;
  (void)K;
  if (kquant_type == "q8_0") {
    quantize_q8_0(w, out, num_weights);
  } else if (kquant_type == "q4_0") {
    quantize_q4_0(w, out, num_weights);
  } else if (kquant_type == "q4_1") {
    quantize_q4_1(w, out, num_weights);
  } else if (kquant_type == "q5_0") {
    quantize_q5_0(w, out, num_weights);
  } else if (kquant_type == "q5_1") {
    quantize_q5_1(w, out, num_weights);
  } else if (codec_by_name(kquant_type) != nullptr) {
    // A known K-quant superblock codec: GPU-only until the CPU K-quant
    // encoders.
    throw std::runtime_error(
        "[mlx_kquant] quantize: CPU encode is not yet implemented for codec '" +
        kquant_type + "'; run on the GPU stream (the default device).");
  } else {
    throw std::runtime_error(
        "[mlx_kquant] quantize: unsupported codec: " + kquant_type);
  }
}

// Explicit instantiations for the float types the eval path dispatches over.
template void kquant_quantize_dispatch<float>(
    const float*,
    uint8_t*,
    std::size_t,
    const std::string&,
    const float*,
    std::size_t);
template void kquant_quantize_dispatch<mx::float16_t>(
    const mx::float16_t*,
    uint8_t*,
    std::size_t,
    const std::string&,
    const float*,
    std::size_t);
template void kquant_quantize_dispatch<mx::bfloat16_t>(
    const mx::bfloat16_t*,
    uint8_t*,
    std::size_t,
    const std::string&,
    const float*,
    std::size_t);

} // namespace mlx_kquant
