// Native-fp (mxfp4 / nvfp4) decode table and scale helpers, shared by the
// scalar and NEON CPU kernels. ggml's doubled-kvalues convention: the int8
// LUT stores 2x the OCP E2M1 values and the scale decoders fold in the
// compensating 0.5, so integer dot products against q8 activations stay
// exact. Decode math mirrors gguf-py MXFP4/NVFP4 (the parity oracle).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

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

} // namespace mlx_kquant
