// Native-fp (mxfp4 / nvfp4) decode table and scale helpers, shared by the
// scalar and NEON CPU kernels. ggml's doubled-kvalues convention: the int8
// LUT stores 2x the OCP E2M1 values and the scale decoders fold in the
// compensating 0.5, so integer dot products against q8 activations stay
// exact. Decode math mirrors gguf-py MXFP4/NVFP4 (the parity oracle). The
// mxfp8 helpers at the end decode plain values, with no doubling.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace mlx_kquant {

// 2x OCP E2M1 values, ggml kvalues_mxfp4 order (low nibble codes 0-7
// positive, 8-15 negative). Shared by mxfp4 and nvfp4.
inline constexpr int8_t kvalues_mxfp4[16] =
    {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// E8M0 scale byte -> 2^(e-127)/2, bit-exact with ggml_e8m0_to_fp32_half:
// e < 2 lands in the f32 subnormal range, so build the bits directly.
inline float kq_e8m0_half(uint8_t e) {
  uint32_t bits =
      e < 2 ? (0x00200000u << e) : (static_cast<uint32_t>(e - 1) << 23);
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Unsigned E4M3 (bias 7) scale byte -> value/2, matching gguf-py
// NVFP4.ue4m3_to_fp32: 0x00 and 0x7F (the NaN encoding) decode to 0. All
// steps are exact powers-of-two scalings of a 4-bit mantissa.
inline float kq_ue4m3_half(uint8_t x) {
  if (x == 0 || x == 0x7F) {
    return 0.0f;
  }
  int exp = (x >> 3) & 0xF;
  float man = static_cast<float>(x & 0x7);
  float raw =
      exp == 0 ? man * 0x1p-9f : std::ldexp(1.0f + man * 0.125f, exp - 7);
  return raw * 0.5f;
}

// MXFP8 (OCP MX): one E8M0 scale byte per 32 E4M3 codes, value =
// e4m3(q) * 2^(e - 127). Decode mirrors ggml dequantize_row_mxfp8, so f32
// output is bit-exact: each code is exact in f32 and the product with a
// power of two is exact unless it overflows to inf.

// E8M0 scale byte -> 2^(e-127); 0xFF is NaN. e == 0 is the f32 subnormal
// 2^-127, so build the bits directly.
inline float kq_e8m0_mxfp8(uint8_t e) {
  if (e == 0xFF) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  uint32_t bits = e == 0 ? 0x00400000u : static_cast<uint32_t>(e) << 23;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// E4M3 (fn) code -> value, all 256 codes: sign, 4 exponent bits (bias 7),
// 3 mantissa bits, no infinity; 0x7F and 0xFF are NaN.
inline const std::array<float, 256>& kq_e4m3fn_lut() {
  static const std::array<float, 256> lut = [] {
    std::array<float, 256> t{};
    for (int x = 0; x < 256; ++x) {
      const int exp = (x >> 3) & 0xF;
      const int man = x & 0x7;
      float v;
      if (exp == 0xF && man == 0x7) {
        v = std::numeric_limits<float>::quiet_NaN();
      } else if (exp == 0) {
        v = std::ldexp(static_cast<float>(man), -9);
      } else {
        v = std::ldexp(static_cast<float>(8 + man), exp - 10);
      }
      t[x] = (x & 0x80) ? -v : v;
    }
    return t;
  }();
  return lut;
}

} // namespace mlx_kquant
