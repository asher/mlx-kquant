// Scalar CPU decoders for the 10 GGUF codecs + dequant-then-matmul. Bit-exact
// per-codec against the gguf-py reference; no Metal deps.
#include "kquant_cpu_decode.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "kquant_codec.h"

#include "mlx/types/half_types.h" // float16_t, bfloat16_t

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

inline float read_f16(const uint8_t* ptr) {
  _Float16 tmp;
  std::memcpy(&tmp, ptr, sizeof(_Float16));
  return static_cast<float>(tmp);
}

template <typename T>
void dequantize_q8_0(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 34;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
    T* dst = out + b * block_weights;
    for (int i = 0; i < block_weights; i++) {
      dst[i] = static_cast<T>(d * static_cast<float>(qs[i]));
    }
  }
}

template <typename T>
void dequantize_q4_0(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 18;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs = block + 2;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int x0 = (qs[j] & 0x0F) - 8;
      int x1 = (qs[j] >> 4) - 8;
      dst[j] = static_cast<T>(d * static_cast<float>(x0));
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1));
    }
  }
}

template <typename T>
void dequantize_q4_1(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 20;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float m = read_f16(block + 2);
    const uint8_t* qs = block + 4;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int x0 = qs[j] & 0x0F;
      int x1 = qs[j] >> 4;
      dst[j] = static_cast<T>(d * static_cast<float>(x0) + m);
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1) + m);
    }
  }
}

template <typename T>
void dequantize_q5_0(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 22;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qh_bytes = block + 2;
    uint32_t qh = static_cast<uint32_t>(qh_bytes[0]) |
        (static_cast<uint32_t>(qh_bytes[1]) << 8) |
        (static_cast<uint32_t>(qh_bytes[2]) << 16) |
        (static_cast<uint32_t>(qh_bytes[3]) << 24);
    const uint8_t* qs = block + 6;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int xh_0 = ((qh >> j) << 4) & 0x10;
      int xh_1 = (qh >> (j + 12)) & 0x10;
      int x0 = (qs[j] & 0x0F) | xh_0;
      int x1 = (qs[j] >> 4) | xh_1;
      dst[j] = static_cast<T>(d * static_cast<float>(x0 - 16));
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1 - 16));
    }
  }
}

template <typename T>
void dequantize_q5_1(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 24;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float m = read_f16(block + 2);
    const uint8_t* qh_bytes = block + 4;
    const uint8_t* qs = block + 8;
    uint32_t qh;
    std::memcpy(&qh, qh_bytes, 4);
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
      uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
      uint8_t x0 = (qs[j] & 0x0F) | xh_0;
      uint8_t x1 = (qs[j] >> 4) | xh_1;
      dst[j] = static_cast<T>(d * static_cast<float>(x0) + m);
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1) + m);
    }
  }
}

inline void unpack_q4k_scales(
    const uint8_t* scales_packed,
    float* sc,
    float* mn,
    float d,
    float dmin) {
  for (int i = 0; i < 8; i++) {
    uint8_t raw_sc, raw_m;
    if (i < 4) {
      raw_sc = scales_packed[i] & 0x3F;
      raw_m = scales_packed[i + 4] & 0x3F;
    } else {
      raw_sc =
          (scales_packed[i + 4] & 0x0F) | ((scales_packed[i - 4] >> 6) << 4);
      raw_m = (scales_packed[i + 4] >> 4) | ((scales_packed[i] >> 6) << 4);
    }
    sc[i] = d * static_cast<float>(raw_sc);
    mn[i] = dmin * static_cast<float>(raw_m);
  }
}

template <typename T>
void dequantize_q4_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 144;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float dmin = read_f16(block + 2);
    const uint8_t* scales_packed = block + 4;
    const uint8_t* qs = block + 16;

    float sc[8], mn[8];
    unpack_q4k_scales(scales_packed, sc, mn, d, dmin);

    T* dst = out + b * block_weights;
    for (int g = 0; g < 4; g++) {
      for (int i = 0; i < 32; i++) {
        dst[(2 * g) * 32 + i] = static_cast<T>(
            sc[2 * g] * static_cast<float>(qs[g * 32 + i] & 0x0F) - mn[2 * g]);
        dst[(2 * g + 1) * 32 + i] = static_cast<T>(
            sc[2 * g + 1] * static_cast<float>(qs[g * 32 + i] >> 4) -
            mn[2 * g + 1]);
      }
    }
  }
}

template <typename T>
void dequantize_q5_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 176;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float dmin = read_f16(block + 2);
    const uint8_t* scales_packed = block + 4;
    const uint8_t* qh = block + 16;
    const uint8_t* qs = block + 48;

    float sc[8], mn[8];
    unpack_q4k_scales(scales_packed, sc, mn, d, dmin);

    T* dst = out + b * block_weights;
    for (int g = 0; g < 4; g++) {
      for (int i = 0; i < 32; i++) {
        uint8_t lo0 = qs[g * 32 + i] & 0x0F;
        uint8_t lo1 = qs[g * 32 + i] >> 4;
        uint8_t hi0 = (qh[i] >> (2 * g)) & 1;
        uint8_t hi1 = (qh[i] >> (2 * g + 1)) & 1;
        dst[(2 * g) * 32 + i] = static_cast<T>(
            sc[2 * g] * static_cast<float>(lo0 | (hi0 << 4)) - mn[2 * g]);
        dst[(2 * g + 1) * 32 + i] = static_cast<T>(
            sc[2 * g + 1] * static_cast<float>(lo1 | (hi1 << 4)) -
            mn[2 * g + 1]);
      }
    }
  }
}

template <typename T>
void dequantize_q6_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 210;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* ql_base = block;
    const uint8_t* qh_base = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    float d = read_f16(block + 208);

    T* dst = out + b * block_weights;
    for (int half = 0; half < 2; half++) {
      const uint8_t* ql = ql_base + half * 64;
      const uint8_t* qh = qh_base + half * 32;
      const int8_t* sc = scales + half * 8;
      T* out_half = dst + half * 128;

      for (int l = 0; l < 32; l++) {
        int is0 = l / 16;
        int8_t q1 =
            static_cast<int8_t>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) -
            32;
        int8_t q2 = static_cast<int8_t>(
                        (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) -
            32;
        int8_t q3 =
            static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int8_t q4 =
            static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) -
            32;
        out_half[l] = static_cast<T>(
            d * static_cast<float>(sc[is0]) * static_cast<float>(q1));
        out_half[l + 32] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 2]) * static_cast<float>(q2));
        out_half[l + 64] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 4]) * static_cast<float>(q3));
        out_half[l + 96] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 6]) * static_cast<float>(q4));
      }
    }
  }
}

inline void unpack_q3k_scales(const uint8_t* s, int32_t* sc) {
  for (int k = 0; k < 4; k++) {
    sc[k] = static_cast<int32_t>(s[k] & 0x0F) |
        (static_cast<int32_t>((s[8 + k]) & 0x03) << 4);
    sc[k + 4] = static_cast<int32_t>(s[k + 4] & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 2) & 0x03) << 4);
    sc[k + 8] = static_cast<int32_t>((s[k] >> 4) & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 4) & 0x03) << 4);
    sc[k + 12] = static_cast<int32_t>((s[k + 4] >> 4) & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 6) & 0x03) << 4);
  }
  for (int i = 0; i < 16; i++) {
    sc[i] -= 32;
  }
}

template <typename T>
void dequantize_q3_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 110;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* hmask = block;
    const uint8_t* qs_full = block + 32;
    const uint8_t* scales_packed = block + 96;
    float d = read_f16(block + 108);

    int32_t sc[16];
    unpack_q3k_scales(scales_packed, sc);

    T* dst = out + b * block_weights;
    int out_idx = 0;
    for (int outer_half = 0; outer_half < 2; outer_half++) {
      const uint8_t* qs_chunk = qs_full + outer_half * 32;
      for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
        int shift = shift_idx * 2;
        uint8_t m = 1 << (outer_half * 4 + shift_idx);
        int is_left = outer_half * 8 + shift_idx * 2;
        float dl_left = d * static_cast<float>(sc[is_left]);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l] >> shift) & 3;
          int h = (hmask[l] & m) ? 0 : 4;
          dst[out_idx++] = static_cast<T>(dl_left * static_cast<float>(q2 - h));
        }
        float dl_right = d * static_cast<float>(sc[is_left + 1]);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l + 16] >> shift) & 3;
          int h = (hmask[l + 16] & m) ? 0 : 4;
          dst[out_idx++] =
              static_cast<T>(dl_right * static_cast<float>(q2 - h));
        }
      }
    }
  }
}

template <typename T>
void dequantize_q2_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 84;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* scales_raw = block;
    const uint8_t* qs_full = block + 16;
    float d = read_f16(block + 80);
    float dmin = read_f16(block + 82);

    T* dst = out + b * block_weights;
    int out_idx = 0;
    int is_idx = 0;
    for (int outer_half = 0; outer_half < 2; outer_half++) {
      const uint8_t* qs_chunk = qs_full + outer_half * 32;
      for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
        int shift = shift_idx * 2;
        uint8_t sc_byte_left = scales_raw[is_idx++];
        float dl_left = d * static_cast<float>(sc_byte_left & 0x0F);
        float ml_left = dmin * static_cast<float>(sc_byte_left >> 4);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l] >> shift) & 3;
          dst[out_idx++] =
              static_cast<T>(dl_left * static_cast<float>(q2) - ml_left);
        }
        uint8_t sc_byte_right = scales_raw[is_idx++];
        float dl_right = d * static_cast<float>(sc_byte_right & 0x0F);
        float ml_right = dmin * static_cast<float>(sc_byte_right >> 4);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l + 16] >> shift) & 3;
          dst[out_idx++] =
              static_cast<T>(dl_right * static_cast<float>(q2) - ml_right);
        }
      }
    }
  }
}

} // namespace

template <typename T>
void kquant_dequantize_dispatch(
    const uint8_t* w,
    T* out,
    std::size_t num_weights,
    const std::string& kquant_type) {
  if (kquant_type == "q8_0") {
    dequantize_q8_0(w, out, num_weights);
  } else if (kquant_type == "q4_0") {
    dequantize_q4_0(w, out, num_weights);
  } else if (kquant_type == "q4_1") {
    dequantize_q4_1(w, out, num_weights);
  } else if (kquant_type == "q5_0") {
    dequantize_q5_0(w, out, num_weights);
  } else if (kquant_type == "q5_1") {
    dequantize_q5_1(w, out, num_weights);
  } else if (kquant_type == "q4_k") {
    dequantize_q4_k(w, out, num_weights);
  } else if (kquant_type == "q5_k") {
    dequantize_q5_k(w, out, num_weights);
  } else if (kquant_type == "q6_k") {
    dequantize_q6_k(w, out, num_weights);
  } else if (kquant_type == "q3_k") {
    dequantize_q3_k(w, out, num_weights);
  } else if (kquant_type == "q2_k") {
    dequantize_q2_k(w, out, num_weights);
  } else {
    throw std::runtime_error(
        "[mlx_kquant] dequantize: unsupported codec: " + kquant_type);
  }
}

template <typename T>
void kquant_qmm_cpu(
    T* result,
    const T* x,
    const uint8_t* w,
    int M,
    int N,
    int K,
    bool transpose_w,
    const std::string& kquant_type) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::runtime_error(
        "[mlx_kquant] quantized_matmul: unsupported codec: " + kquant_type);
  }
  int w_rows = transpose_w ? N : K;
  int w_cols = transpose_w ? K : N;
  std::size_t weights_per_row = static_cast<std::size_t>(w_cols);
  std::size_t row_bytes =
      (weights_per_row / codec->weights_per_block) * codec->bytes_per_block;

  std::vector<float> w_dec(static_cast<std::size_t>(w_rows) * w_cols);
  for (int r = 0; r < w_rows; r++) {
    kquant_dequantize_dispatch(
        w + r * row_bytes, w_dec.data() + r * w_cols, w_cols, kquant_type);
  }

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = 0.0f;
      if (transpose_w) {
        for (int k = 0; k < K; k++) {
          acc += static_cast<float>(x[m * K + k]) * w_dec[n * K + k];
        }
      } else {
        for (int k = 0; k < K; k++) {
          acc += static_cast<float>(x[m * K + k]) * w_dec[k * N + n];
        }
      }
      result[m * N + n] = static_cast<T>(acc);
    }
  }
}

// Explicit instantiations for the float types the eval paths dispatch over.
template void kquant_dequantize_dispatch<float>(
    const uint8_t*,
    float*,
    std::size_t,
    const std::string&);
template void kquant_dequantize_dispatch<mx::float16_t>(
    const uint8_t*,
    mx::float16_t*,
    std::size_t,
    const std::string&);
template void kquant_dequantize_dispatch<mx::bfloat16_t>(
    const uint8_t*,
    mx::bfloat16_t*,
    std::size_t,
    const std::string&);

template void kquant_qmm_cpu<float>(
    float*,
    const float*,
    const uint8_t*,
    int,
    int,
    int,
    bool,
    const std::string&);
template void kquant_qmm_cpu<mx::float16_t>(
    mx::float16_t*,
    const mx::float16_t*,
    const uint8_t*,
    int,
    int,
    int,
    bool,
    const std::string&);
template void kquant_qmm_cpu<mx::bfloat16_t>(
    mx::bfloat16_t*,
    const mx::bfloat16_t*,
    const uint8_t*,
    int,
    int,
    int,
    bool,
    const std::string&);

} // namespace mlx_kquant
