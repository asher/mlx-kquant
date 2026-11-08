// Row quantization to int8 for int8_matmul, on any Metal GPU.
//
// One simdgroup quantizes one group of one row: gs columns from column
// g gs, the last group of a row K - g gs. m = max |float(x)| over the
// group; s = max(m / 127, FLT_MIN), or 1 / 127 when m is 0; codes =
// clamp(rint(float(x) / s), -127, 127), ties to even. The FLT_MIN floor
// keeps s normal: the GPU flushes a subnormal m / 127 to 0. x [rows, K]
// and q [rows, K] row-major, s [rows, G] row-major. Unit u = (row, g) =
// (u / G, u % G); a threadgroup of 8 simdgroups takes 8 consecutive units,
// and the grid covers all of them. VEC reads and writes four columns at a
// time, for K and gs multiples of 4 and x on a boundary of four elements.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T, bool VEC>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_int8_quantize_rows(
    const device T* x [[buffer(0)]],
    device int8_t* q [[buffer(1)]],
    device float* s [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& gs [[buffer(4)]],
    const constant int& G [[buffer(5)]],
    const constant int64_t& units [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const size_t u = size_t(tg) * 8 + sg;
  if (u >= size_t(units)) {
    return;
  }
  const size_t row = u / size_t(G);
  const int g = int(u - row * size_t(G));
  const int c0 = g * gs;
  const int len = metal::min(gs, K - c0);
  const size_t base = row * size_t(K) + size_t(c0);
  const device T* xr = x + base;
  device int8_t* qr = q + base;

  float m = 0.f;
  if (VEC) {
    for (int i = int(lane) * 4; i < len; i += 128) {
      const float4 f = float4(*(const device vec<T, 4>*)(xr + i));
      m = metal::max(
          m,
          metal::max(
              metal::max(metal::abs(f.x), metal::abs(f.y)),
              metal::max(metal::abs(f.z), metal::abs(f.w))));
    }
  } else {
    for (int i = int(lane); i < len; i += 32) {
      m = metal::max(m, metal::abs(float(xr[i])));
    }
  }
  m = simd_max(m);
  const float sc = m > 0.f ? metal::max(m / 127.f, FLT_MIN) : 1.f / 127.f;

  if (VEC) {
    for (int i = int(lane) * 4; i < len; i += 128) {
      const float4 f = float4(*(const device vec<T, 4>*)(xr + i));
      const float4 r = metal::clamp(metal::rint(f / sc), -127.f, 127.f);
      *(device char4*)(qr + i) = char4(r);
    }
  } else {
    for (int i = int(lane); i < len; i += 32) {
      const float r =
          metal::clamp(metal::rint(float(xr[i]) / sc), -127.f, 127.f);
      qr[i] = int8_t(r);
    }
  }
  if (lane == 0) {
    s[u] = sc;
  }
}

// clang-format off
#define instantiate_int8_quant(tname, type)                                  \
  instantiate_kernel("kq_int8_quantize_rows_" #tname, kq_int8_quantize_rows, \
                     type, false)                                            \
  instantiate_kernel("kq_int8_quantize_rows_vec_" #tname,                    \
                     kq_int8_quantize_rows, type, true)

instantiate_int8_quant(bfloat16_t, bfloat16_t)
instantiate_int8_quant(float16_t, float16_t)
instantiate_int8_quant(float, float)
    // clang-format on
