// One kernel body for float16 and bfloat16 activations. A kernel whose math
// runs in float32 reads and writes its activations as kq_x16, and function
// constant 390 (kq_x_bf16) picks the 16-bit format when the pipeline is
// built. The conversions are the ones float16_t and bfloat16_t do, so a
// pipeline computes what the typed kernel computed. The host names such a
// kernel "x16" in place of the type (kq_get_kernel).
#pragma once

#include <metal_stdlib>

using namespace metal;

constant bool kq_x_bf16 [[function_constant(390)]];

inline float kq_x16_float(ushort b) {
  return kq_x_bf16 ? float(as_type<bfloat>(b)) : float(as_type<half>(b));
}

inline ushort kq_x16_bits(float f) {
  return kq_x_bf16 ? as_type<ushort>(static_cast<bfloat>(f))
                   : as_type<ushort>(static_cast<half>(f));
}

struct kq_x16 {
  ushort bits;

  kq_x16() thread = default;
  kq_x16() threadgroup = default;
  kq_x16() device = default;
  kq_x16(float f) thread : bits(kq_x16_bits(f)) {}
  kq_x16(float f) threadgroup : bits(kq_x16_bits(f)) {}
  kq_x16(float f) device : bits(kq_x16_bits(f)) {}

  operator float() const thread {
    return kq_x16_float(bits);
  }
  operator float() const threadgroup {
    return kq_x16_float(bits);
  }
  operator float() const device {
    return kq_x16_float(bits);
  }
  operator float() const constant {
    return kq_x16_float(bits);
  }
};

// Four activations from p as float4, one 8-byte load.
template <typename T>
inline float4 kq_ld4(const device T* p) {
  return float4(*(const device vec<T, 4>*)p);
}

inline float4 kq_ld4(const device kq_x16* p) {
  const ushort4 b = *(const device ushort4*)p;
  return kq_x_bf16 ? float4(as_type<bfloat4>(b)) : float4(as_type<half4>(b));
}

template <typename T>
inline float4 kq_ld4(const threadgroup T* p) {
  return float4(*(const threadgroup vec<T, 4>*)p);
}

inline float4 kq_ld4(const threadgroup kq_x16* p) {
  const ushort4 b = *(const threadgroup ushort4*)p;
  return kq_x_bf16 ? float4(as_type<bfloat4>(b)) : float4(as_type<half4>(b));
}

// Eight activations from p as floats, one 16-byte load.
template <typename T>
inline void kq_ld8(const device T* p, thread float* o) {
  const vec<T, 8> v = *(const device vec<T, 8>*)p;
#pragma unroll
  for (short j = 0; j < 8; ++j) {
    o[j] = float(v[j]);
  }
}

inline void kq_ld8(const device kq_x16* p, thread float* o) {
  const vec<ushort, 8> b = *(const device vec<ushort, 8>*)p;
#pragma unroll
  for (short j = 0; j < 8; ++j) {
    o[j] = kq_x16_float(b[j]);
  }
}
