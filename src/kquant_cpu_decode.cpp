// CPU decoders for the 10 GGUF codecs + dequant-then-matmul. The per-block
// decode is scalar and bit-exact per-codec against the gguf-py reference; no
// Metal deps, so this builds and runs on any platform with the stock mlx
// wheel. The matmul wrapper is performance-tuned but portable: a shared
// worker pool parallelizes over output rows / blocks, small-M matmuls run a
// fused decode-one-block-then-dot loop (no full-matrix scratch), and large-M
// matmuls dequantize once then run a GEMM - through Accelerate where
// available (KQ_USE_ACCELERATE), else a threaded scalar loop. The block
// decode math derives from ggml (llama.cpp, MIT) - see
// mlx_kquant/licenses/llama.cpp-LICENSE.
#include "kquant_cpu_decode.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "kquant_codec.h"
#include "kquant_cpu_neon.h"
#include "kquant_iq_tables.h"

#include "mlx/types/half_types.h" // float16_t, bfloat16_t

#ifdef KQ_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

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

// IQ codecs (load-only): grid/LUT decode, signs read from a packed table.
// Float-op order mirrors ggml dequantize_row_iq* so f32 output is bit-exact.

template <typename T>
void dequantize_iq4_nl(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 18;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs = block + 2;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      dst[j] = static_cast<T>(d * kvalues_iq4nl[qs[j] & 0xf]);
      dst[j + 16] = static_cast<T>(d * kvalues_iq4nl[qs[j] >> 4]);
    }
  }
}

template <typename T>
void dequantize_iq4_xs(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 136;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    uint16_t scales_h;
    std::memcpy(&scales_h, block + 2, sizeof(uint16_t));
    const uint8_t* scales_l = block + 4;
    const uint8_t* qs = block + 8;
    T* dst = out + b * block_weights;
    for (int ib = 0; ib < 8; ib++) {
      int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
          (((scales_h >> (2 * ib)) & 3) << 4);
      float dl = d * (ls - 32);
      const uint8_t* q = qs + ib * 16;
      T* o = dst + ib * 32;
      for (int j = 0; j < 16; j++) {
        o[j] = static_cast<T>(dl * kvalues_iq4nl[q[j] & 0xf]);
        o[j + 16] = static_cast<T>(dl * kvalues_iq4nl[q[j] >> 4]);
      }
    }
  }
}

template <typename T>
void dequantize_iq3_xxs(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 98;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs = block + 2;
    const uint8_t* gas = block + 2 + 64; // scales+signs, interleaved per ib32
    T* dst = out + b * block_weights;
    for (int ib32 = 0; ib32 < 8; ib32++) {
      uint32_t aux32;
      std::memcpy(&aux32, gas + 4 * ib32, sizeof(uint32_t));
      float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
      const uint8_t* q = qs + ib32 * 8;
      T* o = dst + ib32 * 32;
      for (int l = 0; l < 4; l++) {
        uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const uint8_t* g1 =
            reinterpret_cast<const uint8_t*>(iq3xxs_grid + q[2 * l + 0]);
        const uint8_t* g2 =
            reinterpret_cast<const uint8_t*>(iq3xxs_grid + q[2 * l + 1]);
        for (int j = 0; j < 4; j++) {
          o[j + 0] = static_cast<T>(
              db * g1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
          o[j + 4] = static_cast<T>(
              db * g2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
        o += 8;
      }
    }
  }
}

template <typename T>
void dequantize_iq3_s(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 110;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs = block + 2;
    const uint8_t* qh = block + 66;
    const uint8_t* signs = block + 74;
    const uint8_t* scales = block + 106;
    T* y = out + b * block_weights;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
      float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
      float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
      for (int l = 0; l < 4; l++) {
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(
            iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
        const uint8_t* g2 = reinterpret_cast<const uint8_t*>(
            iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
        for (int j = 0; j < 4; j++) {
          y[j + 0] = static_cast<T>(
              db1 * g1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f));
          y[j + 4] = static_cast<T>(
              db1 * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
        y += 8;
      }
      qs += 8;
      signs += 4;
      for (int l = 0; l < 4; l++) {
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(
            iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
        const uint8_t* g2 = reinterpret_cast<const uint8_t*>(
            iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
        for (int j = 0; j < 4; j++) {
          y[j + 0] = static_cast<T>(
              db2 * g1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f));
          y[j + 4] = static_cast<T>(
              db2 * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
        y += 8;
      }
      qh += 2;
      qs += 8;
      signs += 4;
    }
  }
}

// IQ2 grids are uint64 (one entry = 8 weight bytes), so one lookup per group of
// 8 (vs IQ3's two uint32 lookups). Scale is d*(0.5+s)*0.25; signs come from the
// ksigns table (XXS/XS) or the block's own signs[] bytes (S).
template <typename T>
void dequantize_iq2_xxs(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 66;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs =
        block + 2; // uint16_t[32]: per ib32, 2 u32 (idx + scale/signs)
    T* y = out + b * block_weights;
    for (int ib32 = 0; ib32 < 8; ib32++) {
      uint32_t aux32[2];
      std::memcpy(aux32, qs + 8 * ib32, 2 * sizeof(uint32_t));
      const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
      float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
      for (int l = 0; l < 4; l++) {
        const uint8_t* g =
            reinterpret_cast<const uint8_t*>(iq2xxs_grid + aux8[l]);
        uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
        for (int j = 0; j < 8; j++) {
          y[j] =
              static_cast<T>(db * g[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
        }
        y += 8;
      }
    }
  }
}

template <typename T>
void dequantize_iq2_xs(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 74;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs =
        block + 2; // uint16_t[32]: 9-bit grid idx | 7-bit sign idx
    const uint8_t* scales = block + 66; // uint8_t[8]
    T* y = out + b * block_weights;
    for (int ib32 = 0; ib32 < 8; ib32++) {
      float db[2];
      db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
      db[1] = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
      for (int l = 0; l < 4; l++) {
        uint16_t q;
        std::memcpy(&q, qs + 2 * (4 * ib32 + l), sizeof(uint16_t));
        const uint8_t* g =
            reinterpret_cast<const uint8_t*>(iq2xs_grid + (q & 511));
        uint8_t signs = ksigns_iq2xs[q >> 9];
        float dl = db[l / 2];
        for (int j = 0; j < 8; j++) {
          y[j] =
              static_cast<T>(dl * g[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
        }
        y += 8;
      }
    }
  }
}

template <typename T>
void dequantize_iq2_s(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 82;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs = block + 2; // grid-low bytes [0..31]
    const uint8_t* signs = block + 2 + 32; // sign bytes [32..63] (= qs + 32)
    const uint8_t* qh = block + 66; // uint8_t[8]
    const uint8_t* scales = block + 74; // uint8_t[8]
    T* y = out + b * block_weights;
    for (int ib32 = 0; ib32 < 8; ib32++) {
      float db[2];
      db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
      db[1] = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
      for (int l = 0; l < 4; l++) {
        float dl = db[l / 2];
        const uint8_t* g = reinterpret_cast<const uint8_t*>(
            iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
        for (int j = 0; j < 8; j++) {
          y[j] = static_cast<T>(
              dl * g[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f));
        }
        y += 8;
      }
      qs += 4;
      signs += 4;
    }
  }
}

// --------------------------------------------------------------------------
// Shared CPU worker pool
// --------------------------------------------------------------------------
//
// MLX's CPU command encoder runs each primitive's lambda on a single
// per-stream scheduler thread, so without this pool every kq.* CPU op is
// single-threaded. The pool is lazily created on first use and lives for the
// process; the dispatching thread participates in every job, so
// KQ_CPU_THREADS=1 (or a single-core box) degrades to plain inline execution
// with no thread traffic. Not reentrant by design - callers below only ever
// invoke it from the scheduler thread, never from inside a worker.

class KQThreadPool {
 public:
  static KQThreadPool& get() {
    static KQThreadPool pool;
    return pool;
  }

  int n_threads() const {
    return static_cast<int>(workers_.size()) + 1; // + calling thread
  }

  // Run fn(part) for part in [0, n_parts), distributing parts across the
  // workers and the calling thread. Blocks until every part has finished.
  void parallel(int n_parts, const std::function<void(int)>& fn) {
    if (n_parts <= 0) {
      return;
    }
    if (n_parts == 1 || workers_.empty()) {
      for (int p = 0; p < n_parts; p++) {
        fn(p);
      }
      return;
    }
    // Each job carries its own shared state, so the caller only has to wait
    // for its parts to finish - never for slow-to-wake workers to pass
    // through (a worker that wakes late just finds the part counter
    // exhausted and goes back to sleep; the shared_ptr keeps the state it
    // touches alive). Waiting on woken-worker exit instead was costing
    // ~50-100us of wake-straggler latency per job on asymmetric cores.
    auto job = std::make_shared<Job>();
    job->fn = fn; // copy: stragglers may outlive the caller's frame
    job->parts = n_parts;
    {
      std::lock_guard<std::mutex> lk(m_);
      job_ = job;
      generation_++;
      cv_.notify_all();
    }
    run_parts(*job); // calling thread participates
    if (job->done.load(std::memory_order_acquire) < job->parts) {
      std::unique_lock<std::mutex> lk(m_);
      done_cv_.wait(lk, [&] {
        return job->done.load(std::memory_order_acquire) >= job->parts;
      });
    }
  }

 private:
  KQThreadPool() {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    if (const char* env = std::getenv("KQ_CPU_THREADS")) {
      int v = std::atoi(env);
      if (v > 0) {
        n = v;
      }
    }
    if (n < 1) {
      n = 1;
    }
    if (n > 64) {
      n = 64;
    }
    for (int i = 0; i < n - 1; i++) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ~KQThreadPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
      cv_.notify_all();
    }
    for (auto& t : workers_) {
      t.join();
    }
  }

  struct Job {
    std::function<void(int)> fn;
    int parts{0};
    std::atomic<int> next{0};
    std::atomic<int> done{0};
  };

  void run_parts(Job& job) {
    int p;
    while ((p = job.next.fetch_add(1)) < job.parts) {
      job.fn(p);
      if (job.done.fetch_add(1) + 1 == job.parts) {
        // Last part: wake the caller. Notify under the mutex so the caller's
        // predicate check can't race past the increment and then sleep.
        std::lock_guard<std::mutex> lk(m_);
        done_cv_.notify_all();
      }
    }
  }

  void worker_loop() {
    // Optional spin-before-park (KQ_CPU_SPIN_US, default 0 = park on the
    // condition variable immediately). Spinning for the next job avoids the
    // ~50-100us cv wake latency and buys ~15% on back-to-back same-op GEMV
    // microbenches - but in real model graphs the spinning workers steal
    // cores from everything that runs BETWEEN kq jobs (the single-threaded
    // MLX CPU ops, and on hybrid CPU/GPU runs the GPU encode threads), which
    // measured net-negative end-to-end. Off by default; the knob stays for
    // dedicated-CPU experiments.
    static const auto kSpinWindow = [] {
      long us = 0;
      if (const char* env = std::getenv("KQ_CPU_SPIN_US")) {
        us = std::atol(env);
        if (us < 0) {
          us = 0;
        }
      }
      return std::chrono::microseconds(us);
    }();
    std::uint64_t seen = 0;
    auto spin_deadline = std::chrono::steady_clock::now() + kSpinWindow;
    while (true) {
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
      if (generation_.load(std::memory_order_acquire) == seen) {
        if (std::chrono::steady_clock::now() < spin_deadline) {
#if defined(__aarch64__)
          asm volatile("yield");
#endif
          continue;
        }
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] {
          return stop_.load(std::memory_order_relaxed) ||
              generation_.load(std::memory_order_relaxed) != seen;
        });
        if (stop_.load(std::memory_order_relaxed)) {
          return;
        }
      }
      std::shared_ptr<Job> job;
      {
        std::lock_guard<std::mutex> lk(m_);
        seen = generation_.load(std::memory_order_relaxed);
        job = job_;
      }
      if (job) {
        run_parts(*job);
      }
      spin_deadline = std::chrono::steady_clock::now() + kSpinWindow;
    }
  }

  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::shared_ptr<Job> job_; // current job (guarded by m_)
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<bool> stop_{false};
};

} // namespace

int kq_cpu_threads() {
  return KQThreadPool::get().n_threads();
}

void kq_parallel_for(
    std::size_t n_items,
    const std::function<void(std::size_t, std::size_t)>& fn) {
  kq_parallel_for(n_items, 1, fn);
}

void kq_parallel_for(
    std::size_t n_items,
    std::size_t grain,
    const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n_items == 0) {
    return;
  }
  auto& pool = KQThreadPool::get();
  // Oversubscribe parts (the workers pull them off an atomic counter) so the
  // schedule load-balances across asymmetric P/E cores and memory-stall
  // jitter; one static part per thread leaves fast cores idle waiting on the
  // slowest core's tail.
  std::size_t parts = static_cast<std::size_t>(pool.n_threads()) * 8;
  if (grain > 1 && parts > n_items / grain) {
    parts = n_items / grain;
  }
  if (parts > n_items) {
    parts = n_items;
  }
  if (parts <= 1 || pool.n_threads() <= 1) {
    fn(0, n_items);
    return;
  }
  pool.parallel(static_cast<int>(parts), [&](int p) {
    std::size_t begin = n_items * p / parts;
    std::size_t end = n_items * (p + 1) / parts;
    if (begin < end) {
      fn(begin, end);
    }
  });
}

namespace {

// Function-pointer resolver for the float32 block decoders, so the hot matmul
// loops pay the codec-name string compare once per call instead of per block.
using DequantFnF32 = void (*)(const uint8_t*, float*, std::size_t);

DequantFnF32 dequant_fn_f32(const std::string& t) {
  if (t == "q8_0") {
    return &dequantize_q8_0<float>;
  } else if (t == "q4_0") {
    return &dequantize_q4_0<float>;
  } else if (t == "q4_1") {
    return &dequantize_q4_1<float>;
  } else if (t == "q5_0") {
    return &dequantize_q5_0<float>;
  } else if (t == "q5_1") {
    return &dequantize_q5_1<float>;
  } else if (t == "q4_k") {
    return &dequantize_q4_k<float>;
  } else if (t == "q5_k") {
    return &dequantize_q5_k<float>;
  } else if (t == "q6_k") {
    return &dequantize_q6_k<float>;
  } else if (t == "q3_k") {
    return &dequantize_q3_k<float>;
  } else if (t == "q2_k") {
    return &dequantize_q2_k<float>;
  } else if (t == "iq4_nl") {
    return &dequantize_iq4_nl<float>;
  } else if (t == "iq4_xs") {
    return &dequantize_iq4_xs<float>;
  } else if (t == "iq3_s") {
    return &dequantize_iq3_s<float>;
  } else if (t == "iq3_xxs") {
    return &dequantize_iq3_xxs<float>;
  } else if (t == "iq2_xxs") {
    return &dequantize_iq2_xxs<float>;
  } else if (t == "iq2_xs") {
    return &dequantize_iq2_xs<float>;
  } else if (t == "iq2_s") {
    return &dequantize_iq2_s<float>;
  }
  return nullptr;
}

// Multi-accumulator dot so the compiler can vectorize without -ffast-math
// (a single float accumulator forbids reassociation). n is a whole number of
// codec blocks, always a multiple of 8.
inline float dot_block(const float* a, const float* b, int n) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  float s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
  for (int i = 0; i < n; i += 8) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4];
    s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6];
    s7 += a[i + 7] * b[i + 7];
  }
  return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

// Fused decode-then-dot over output rows [n0, n1) for transpose_w=true
// (w decodes to [N, K] row-major, the weight convention of every model
// matmul). One 256-weight block is decoded into a stack buffer and
// immediately dotted against all M activation rows - wire bytes are read
// once and no [N, K] scratch is ever materialized, which is what makes
// memory-bound decode honest.
void qmv_fused_rows(
    float* outf, // [M, N]
    const float* xf, // [M, K]
    const uint8_t* w,
    int M,
    int N,
    int K,
    std::size_t row_bytes,
    int block_weights,
    std::size_t bytes_per_block,
    DequantFnF32 fn,
    std::size_t n0,
    std::size_t n1) {
  float buf[256]; // max weights_per_block across codecs
  float acc[kQmvFusedMaxM];
  const int nblocks = K / block_weights;
  for (std::size_t n = n0; n < n1; n++) {
    const uint8_t* wr = w + n * row_bytes;
    for (int m = 0; m < M; m++) {
      acc[m] = 0.0f;
    }
    for (int b = 0; b < nblocks; b++) {
      fn(wr + b * bytes_per_block, buf, block_weights);
      const float* xb = xf + static_cast<std::size_t>(b) * block_weights;
      for (int m = 0; m < M; m++) {
        acc[m] +=
            dot_block(xb + static_cast<std::size_t>(m) * K, buf, block_weights);
      }
    }
    for (int m = 0; m < M; m++) {
      outf[static_cast<std::size_t>(m) * N + n] = acc[m];
    }
  }
}

// Quantize M f32 activation rows into the NEON kernel's q8 layout (done once
// per matmul call, before the parallel row loop). Returns the row stride in
// bytes; rows stay 4-byte aligned (act_block_bytes is a multiple of 4).
std::size_t quantize_act_rows(
    const KQNeonKernel* nk,
    const float* xf,
    int M,
    int K,
    int block_weights,
    std::unique_ptr<uint8_t[]>& act) {
  const std::size_t row_stride =
      (static_cast<std::size_t>(K) / block_weights) * nk->act_block_bytes;
  act.reset(new uint8_t[static_cast<std::size_t>(M) * row_stride]);
  for (int m = 0; m < M; m++) {
    nk->quantize_act_row(
        xf + static_cast<std::size_t>(m) * K, act.get() + m * row_stride, K);
  }
  return row_stride;
}

// NEON int8 fused GEMV over output rows [n0, n1): one whole-row vec_dot of
// wire bytes against the pre-quantized q8 activations per (row, m). Wire
// bytes are read once per row (M passes hit cache).
void qmv_neon_rows(
    float* outf, // [M, N]
    const uint8_t* act, // M q8 activation rows
    std::size_t act_row_stride,
    const uint8_t* w,
    int M,
    int N,
    std::size_t row_bytes,
    int nblocks,
    const KQNeonKernel* nk,
    std::size_t n0,
    std::size_t n1) {
  for (std::size_t n = n0; n < n1; n++) {
    const uint8_t* wr = w + n * row_bytes;
    for (int m = 0; m < M; m++) {
      outf[static_cast<std::size_t>(m) * N + n] = nk->vec_dot(
          wr, act + static_cast<std::size_t>(m) * act_row_stride, nblocks);
    }
  }
}

// Per-element casts are ~1 ns of work: keep parts coarse and run decode-shape
// conversions (M <= 16 rows) inline rather than waking the pool.
constexpr std::size_t kConvertGrain = std::size_t(1) << 16;

template <typename T>
void convert_to_f32(const T* src, float* dst, std::size_t n) {
  kq_parallel_for(n, kConvertGrain, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; i++) {
      dst[i] = static_cast<float>(src[i]);
    }
  });
}

template <typename T>
void convert_from_f32(const float* src, T* dst, std::size_t n) {
  kq_parallel_for(n, kConvertGrain, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; i++) {
      dst[i] = static_cast<T>(src[i]);
    }
  });
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
  } else if (kquant_type == "iq4_nl") {
    dequantize_iq4_nl(w, out, num_weights);
  } else if (kquant_type == "iq4_xs") {
    dequantize_iq4_xs(w, out, num_weights);
  } else if (kquant_type == "iq3_s") {
    dequantize_iq3_s(w, out, num_weights);
  } else if (kquant_type == "iq3_xxs") {
    dequantize_iq3_xxs(w, out, num_weights);
  } else if (kquant_type == "iq2_xxs") {
    dequantize_iq2_xxs(w, out, num_weights);
  } else if (kquant_type == "iq2_xs") {
    dequantize_iq2_xs(w, out, num_weights);
  } else if (kquant_type == "iq2_s") {
    dequantize_iq2_s(w, out, num_weights);
  } else {
    throw std::runtime_error(
        "[mlx_kquant] dequantize: unsupported codec: " + kquant_type);
  }
}

template <typename T>
void kquant_dequantize_parallel(
    const uint8_t* w,
    T* out,
    std::size_t num_weights,
    const std::string& kquant_type) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::runtime_error(
        "[mlx_kquant] dequantize: unsupported codec: " + kquant_type);
  }
  const std::size_t bw = codec->weights_per_block;
  const std::size_t bb = codec->bytes_per_block;
  const std::size_t n_blocks = num_weights / bw;
  kq_parallel_for(n_blocks, [&](std::size_t b0, std::size_t b1) {
    kquant_dequantize_dispatch(
        w + b0 * bb, out + b0 * bw, (b1 - b0) * bw, kquant_type);
  });
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
  const int w_rows = transpose_w ? N : K;
  const int w_cols = transpose_w ? K : N;
  const std::size_t row_bytes =
      (static_cast<std::size_t>(w_cols) / codec->weights_per_block) *
      codec->bytes_per_block;
  DequantFnF32 fn = dequant_fn_f32(kquant_type);

  // Stage activations (and, for half outputs, the result) in float32. The
  // accumulation dtype is float either way; this just hoists the per-element
  // casts out of the inner loops.
  const std::size_t x_els = static_cast<std::size_t>(M) * K;
  const std::size_t out_els = static_cast<std::size_t>(M) * N;
  std::unique_ptr<float[]> xf_store;
  const float* xf;
  if constexpr (std::is_same_v<T, float>) {
    xf = x;
  } else {
    xf_store.reset(new float[x_els]);
    convert_to_f32(x, xf_store.get(), x_els);
    xf = xf_store.get();
  }
  std::unique_ptr<float[]> of_store;
  float* of;
  if constexpr (std::is_same_v<T, float>) {
    of = result;
  } else {
    of_store.reset(new float[out_els]);
    of = of_store.get();
  }

  if (transpose_w && M <= kQmvFusedMaxM && fn != nullptr) {
    // Decode/GEMV shape: fused decode-then-dot, parallel over output rows.
    // No scratch matrix; wire bytes are read exactly once. When an int8 NEON
    // kernel exists for the codec, quantize the activations once to q8 and
    // dot wire nibbles directly (vdotq_s32); else scalar f32 decode-then-dot.
    const KQNeonKernel* nk = kq_neon_kernel(kquant_type);
    if (nk != nullptr) {
      std::unique_ptr<uint8_t[]> act;
      const std::size_t act_stride =
          quantize_act_rows(nk, xf, M, K, codec->weights_per_block, act);
      const int nblocks = K / codec->weights_per_block;
      kq_parallel_for(
          static_cast<std::size_t>(N), [&](std::size_t n0, std::size_t n1) {
            qmv_neon_rows(
                of,
                act.get(),
                act_stride,
                w,
                M,
                N,
                row_bytes,
                nblocks,
                nk,
                n0,
                n1);
          });
    } else {
      kq_parallel_for(
          static_cast<std::size_t>(N), [&](std::size_t n0, std::size_t n1) {
            qmv_fused_rows(
                of,
                xf,
                w,
                M,
                N,
                K,
                row_bytes,
                codec->weights_per_block,
                codec->bytes_per_block,
                fn,
                n0,
                n1);
          });
    }
  } else {
    // Prefill/GEMM shape: dequantize the weight matrix once (parallel over
    // rows, uninitialized scratch - every element is overwritten), then GEMM.
    std::unique_ptr<float[]> w_dec(
        new float[static_cast<std::size_t>(w_rows) * w_cols]);
    kq_parallel_for(
        static_cast<std::size_t>(w_rows), [&](std::size_t r0, std::size_t r1) {
          for (std::size_t r = r0; r < r1; r++) {
            kquant_dequantize_dispatch(
                w + r * row_bytes,
                w_dec.get() + r * w_cols,
                w_cols,
                kquant_type);
          }
        });

#ifdef KQ_USE_ACCELERATE
    // Accelerate's sgemm engages the AMX/SME matrix units - far past what
    // scalar (or even NEON) per-core loops can reach for M x N x K work.
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        transpose_w ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        1.0f,
        xf,
        K,
        w_dec.get(),
        w_cols,
        0.0f,
        of,
        N);
#else
    kq_parallel_for(
        static_cast<std::size_t>(N), [&](std::size_t n0, std::size_t n1) {
          for (int m = 0; m < M; m++) {
            const float* xm = xf + static_cast<std::size_t>(m) * K;
            for (std::size_t n = n0; n < n1; n++) {
              float acc = 0.0f;
              if (transpose_w) {
                const float* wn = w_dec.get() + n * K;
                for (int k = 0; k < K; k++) {
                  acc += xm[k] * wn[k];
                }
              } else {
                for (int k = 0; k < K; k++) {
                  acc += xm[k] * w_dec[static_cast<std::size_t>(k) * N + n];
                }
              }
              of[static_cast<std::size_t>(m) * N + n] = acc;
            }
          }
        });
#endif
  }

  if constexpr (!std::is_same_v<T, float>) {
    convert_from_f32(of, result, out_els);
  }
}

template <typename T>
void kquant_qmm_cpu_batch(
    const KQmvTask<T>* tasks,
    int n_tasks,
    int N,
    int K,
    const std::string& kquant_type) {
  if (n_tasks <= 0) {
    return;
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  DequantFnF32 fn = codec ? dequant_fn_f32(kquant_type) : nullptr;
  if (fn == nullptr) {
    throw std::runtime_error(
        "[mlx_kquant] quantized_matmul: unsupported codec: " + kquant_type);
  }
  const int block_weights = codec->weights_per_block;
  const int nblocks = K / block_weights;
  const std::size_t row_bytes =
      static_cast<std::size_t>(nblocks) * codec->bytes_per_block;
  const KQNeonKernel* nk = kq_neon_kernel(kquant_type);

  // Deduplicate activation blocks shared across tasks (MoE decode: top-k
  // experts all read the same token row), so each x row is staged/quantized
  // once per call. Task counts are tiny at decode; linear scan is fine.
  std::vector<int> ux_of(n_tasks);
  std::vector<int> ux_first;
  for (int t = 0; t < n_tasks; t++) {
    int u = -1;
    for (int s = 0; s < static_cast<int>(ux_first.size()); s++) {
      if (tasks[ux_first[s]].x == tasks[t].x &&
          tasks[ux_first[s]].m == tasks[t].m) {
        u = s;
        break;
      }
    }
    if (u < 0) {
      u = static_cast<int>(ux_first.size());
      ux_first.push_back(t);
    }
    ux_of[t] = u;
  }
  const int n_ux = static_cast<int>(ux_first.size());

  // Stage the unique activation blocks in f32. All loops here are serial:
  // batch shapes are decode-sized (m <= kQmvFusedMaxM, a handful of tasks),
  // so staging is negligible next to the N x K row work below.
  std::vector<const float*> uxf(n_ux);
  std::unique_ptr<float[]> xf_store;
  if constexpr (std::is_same_v<T, float>) {
    for (int u = 0; u < n_ux; u++) {
      uxf[u] = tasks[ux_first[u]].x;
    }
  } else {
    std::size_t total = 0;
    for (int u = 0; u < n_ux; u++) {
      total += static_cast<std::size_t>(tasks[ux_first[u]].m) * K;
    }
    xf_store.reset(new float[total]);
    std::size_t off = 0;
    for (int u = 0; u < n_ux; u++) {
      const T* src = tasks[ux_first[u]].x;
      const std::size_t els =
          static_cast<std::size_t>(tasks[ux_first[u]].m) * K;
      float* dst = xf_store.get() + off;
      for (std::size_t i = 0; i < els; i++) {
        dst[i] = static_cast<float>(src[i]);
      }
      uxf[u] = dst;
      off += els;
    }
  }

  // Quantize the unique activation blocks to q8 once (NEON path only).
  std::unique_ptr<uint8_t[]> act;
  std::vector<const uint8_t*> uact(n_ux, nullptr);
  std::size_t act_stride = 0;
  if (nk != nullptr) {
    act_stride = static_cast<std::size_t>(nblocks) * nk->act_block_bytes;
    std::size_t total_rows = 0;
    for (int u = 0; u < n_ux; u++) {
      total_rows += tasks[ux_first[u]].m;
    }
    act.reset(new uint8_t[total_rows * act_stride]);
    std::size_t row = 0;
    for (int u = 0; u < n_ux; u++) {
      uact[u] = act.get() + row * act_stride;
      for (int m = 0; m < tasks[ux_first[u]].m; m++, row++) {
        nk->quantize_act_row(
            uxf[u] + static_cast<std::size_t>(m) * K,
            act.get() + row * act_stride,
            K);
      }
    }
  }

  // f32 output staging for half-precision tasks.
  std::vector<float*> ofs(n_tasks);
  std::unique_ptr<float[]> of_store;
  if constexpr (std::is_same_v<T, float>) {
    for (int t = 0; t < n_tasks; t++) {
      ofs[t] = tasks[t].out;
    }
  } else {
    std::size_t total = 0;
    for (int t = 0; t < n_tasks; t++) {
      total += static_cast<std::size_t>(tasks[t].m) * N;
    }
    of_store.reset(new float[total]);
    std::size_t off = 0;
    for (int t = 0; t < n_tasks; t++) {
      ofs[t] = of_store.get() + off;
      off += static_cast<std::size_t>(tasks[t].m) * N;
    }
  }

  // ONE parallel job over all (task, output-row) work items. Every row costs
  // the same (same N x K wire geometry), so flat-index partitioning balances.
  kq_parallel_for(
      static_cast<std::size_t>(n_tasks) * N,
      [&](std::size_t r0, std::size_t r1) {
        std::size_t r = r0;
        while (r < r1) {
          const int t = static_cast<int>(r / N);
          const std::size_t n0 = r % N;
          const std::size_t n1 = std::min<std::size_t>(N, n0 + (r1 - r));
          if (nk != nullptr) {
            qmv_neon_rows(
                ofs[t],
                uact[ux_of[t]],
                act_stride,
                tasks[t].w,
                tasks[t].m,
                N,
                row_bytes,
                nblocks,
                nk,
                n0,
                n1);
          } else {
            qmv_fused_rows(
                ofs[t],
                uxf[ux_of[t]],
                tasks[t].w,
                tasks[t].m,
                N,
                K,
                row_bytes,
                block_weights,
                codec->bytes_per_block,
                fn,
                n0,
                n1);
          }
          r += n1 - n0;
        }
      });

  if constexpr (!std::is_same_v<T, float>) {
    for (int t = 0; t < n_tasks; t++) {
      const std::size_t els = static_cast<std::size_t>(tasks[t].m) * N;
      for (std::size_t i = 0; i < els; i++) {
        tasks[t].out[i] = static_cast<T>(ofs[t][i]);
      }
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

template void kquant_dequantize_parallel<float>(
    const uint8_t*,
    float*,
    std::size_t,
    const std::string&);
template void kquant_dequantize_parallel<mx::float16_t>(
    const uint8_t*,
    mx::float16_t*,
    std::size_t,
    const std::string&);
template void kquant_dequantize_parallel<mx::bfloat16_t>(
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

template void kquant_qmm_cpu_batch<float>(
    const KQmvTask<float>*,
    int,
    int,
    int,
    const std::string&);
template void kquant_qmm_cpu_batch<mx::float16_t>(
    const KQmvTask<mx::float16_t>*,
    int,
    int,
    int,
    const std::string&);
template void kquant_qmm_cpu_batch<mx::bfloat16_t>(
    const KQmvTask<mx::bfloat16_t>*,
    int,
    int,
    int,
    const std::string&);

} // namespace mlx_kquant
