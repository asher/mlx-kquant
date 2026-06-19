// Shared scalar-encode helpers, used by both the K-quant CPU encoders
// (kquant_cpu_encode.cpp) and the IQ CPU encoders (kquant_iq_encode.cpp). The
// IQ translation unit is built -ffp-contract=off (for byte-exactness against
// llama-quantize); the K-quant one is not. To keep each unit's floating-point
// flags isolated, these live in an anonymous namespace -- every including TU
// gets its OWN copy compiled with its OWN flags, so the IQ unit's no-contract
// kq_make_qp_quants can never be linked into the K-quant path (which would risk
// its byte-exact CPU/GPU A/B). The fp16 round trips derive from ggml
// (llama.cpp, MIT); see mlx_kquant/licenses/llama.cpp-LICENSE.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "mlx/types/half_types.h"

namespace mlx_kquant {
namespace {

// Write a float as a 2-byte fp16 (round-to-nearest-even), matching the kernel's
// `half(v)` store; the K-quant re-quantize phase reads this back, so the round
// trip must be identical between CPU and GPU.
inline void write_f16(uint8_t* ptr, float v) {
  _Float16 h = static_cast<_Float16>(v);
  std::memcpy(ptr, &h, sizeof(_Float16));
}

// Read a 2-byte fp16 back to float, matching the kernel's `float(*(half*)...)`.
// The K-quant encoders write the super-scale as fp16 then re-read it before
// re-quantizing each weight, so this round trip must match the GPU exactly.
inline float read_f16(const uint8_t* ptr) {
  _Float16 h;
  std::memcpy(&h, ptr, sizeof(_Float16));
  return static_cast<float>(h);
}

// int(rint(v)): round-half-to-even, matching Metal `int(rint(v))` (the K-quant
// rounding rule; the flat codecs use round-half-away-from-zero instead).
inline int kq_nearest_int(float v) {
  return static_cast<int>(std::rint(v));
}

// Fit a single positive-only scale to N values via the ggml make_qp_quants
// search (the Q4_K/Q5_K/Q2_K super-scale and super-min; also reused by the IQ
// encoders). Writes the quantized levels into L_out and returns the scale.
// group_max_eps is the below-which-treat-as-zero floor: 1e-30f for the K-quants
// (preserves their shipped byte-exactness), per-codec values for the IQ family.
template <int N>
float kq_make_qp_quants(
    const float* x,
    const float* qw,
    int nmax,
    uint8_t* L_out,
    float group_max_eps = 1e-30f) {
  float max_v = 0.0f;
  for (int i = 0; i < N; i++) {
    if (x[i] > max_v) {
      max_v = x[i];
    }
  }
  if (max_v < group_max_eps) {
    for (int i = 0; i < N; i++) {
      L_out[i] = 0;
    }
    return 0.0f;
  }
  float iscale = float(nmax) / max_v;
  uint8_t L[N];
  for (int i = 0; i < N; i++) {
    int l = kq_nearest_int(iscale * x[i]);
    L[i] = uint8_t(std::max(0, std::min(nmax, l)));
  }
  float scale = 1.0f / iscale;
  float best_mse = 0.0f;
  for (int i = 0; i < N; i++) {
    float diff = x[i] - scale * float(L[i]);
    best_mse += qw[i] * diff * diff;
  }
  for (int is = -4; is <= 4; is++) {
    if (is == 0) {
      continue;
    }
    float iscale_is = (0.1f * float(is) + float(nmax)) / max_v;
    float scale_is = 1.0f / iscale_is;
    float mse = 0.0f;
    for (int i = 0; i < N; i++) {
      int l = kq_nearest_int(iscale_is * x[i]);
      l = std::min(nmax, l);
      float diff = x[i] - scale_is * float(l);
      mse += qw[i] * diff * diff;
    }
    if (mse < best_mse) {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0.0f, suml2 = 0.0f;
  for (int i = 0; i < N; i++) {
    int l = kq_nearest_int(iscale * x[i]);
    l = std::min(nmax, l);
    L[i] = uint8_t(l);
    sumlx += qw[i] * x[i] * float(l);
    suml2 += qw[i] * float(l) * float(l);
  }
  for (int itry = 0; itry < 5; itry++) {
    int n_changed = 0;
    for (int i = 0; i < N; i++) {
      float w = qw[i];
      float slx = sumlx - w * x[i] * float(L[i]);
      float sl2 = suml2 - w * float(L[i]) * float(L[i]);
      if (slx > 0.0f && sl2 > 0.0f) {
        int new_l = kq_nearest_int(x[i] * sl2 / slx);
        new_l = std::min(nmax, new_l);
        if (new_l != int(L[i])) {
          slx += w * x[i] * float(new_l);
          sl2 += w * float(new_l) * float(new_l);
          if (slx * slx * suml2 > sumlx * sumlx * sl2) {
            L[i] = uint8_t(new_l);
            sumlx = slx;
            suml2 = sl2;
            n_changed++;
          }
        }
      }
    }
    if (n_changed == 0) {
      break;
    }
  }
  for (int i = 0; i < N; i++) {
    L_out[i] = L[i];
  }
  return (suml2 > 0.0f) ? (sumlx / suml2) : 0.0f;
}

} // namespace
} // namespace mlx_kquant
