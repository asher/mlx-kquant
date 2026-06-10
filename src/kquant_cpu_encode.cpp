// Scalar CPU encoders for all ten GGUF codecs: the flat ones (q8_0 + the four
// legacy block codecs) here, the five K-quant superblocks in the section below.
// Each function is a direct port of the matching Metal kernel impl in
// kq_quantized_encode.h: same scale derivation, same round-and-pack (Metal
// `round` is round-half-away-from-zero -> std::round; the K-quants use
// round-half-to-even via std::rint), and the scales are written through fp16
// exactly as `half(...)` does. The flat codecs and q6_k are byte-identical to a
// GPU-encoded block; the four codecs that reduce sigma2 (q2_k/q4_k/q5_k, and
// q3_k under an imatrix) can differ by an ULP-tied level but are numerically
// equivalent.
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

// Read a 2-byte fp16 back to float, matching the kernel's `float(*(half*)...)`.
// The K-quant encoders write the super-scale as fp16 then re-read it before
// re-quantizing each weight, so this round trip must match the GPU exactly.
inline float read_f16(const uint8_t* ptr) {
  _Float16 h;
  std::memcpy(&h, ptr, sizeof(_Float16));
  return static_cast<float>(h);
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

// ----------------------------- K-quant codecs -----------------------------
// The five super-block codecs (256 weights/block). Each is a serial port of the
// matching Metal kernel impl in kq_quantized_encode.h: the GPU runs one
// 256-thread threadgroup per super-block with simd_sum reductions and barriers;
// here that collapses to plain loops over the 256 weights, computing the same
// intermediate fit. K-quant rounding uses round-half-to-even (Metal `rint`),
// the super-scale is written as fp16 and re-read before re-quantizing, and
// (where a codec has one) the imatrix steers the per-weight importance like the
// kernel.

// Block geometry + field offsets, mirroring kq_quantized.h (the byte layout the
// decode path and the gguf-py reference both agree on). Kept here as plain
// constants so this file has no Metal-header dependency.
constexpr int KSB = 256; // super-block weights, all five codecs

constexpr int Q2K_BB = 84, Q2K_SC = 0, Q2K_QS = 16, Q2K_D = 80, Q2K_DMIN = 82;
constexpr int Q3K_BB = 110, Q3K_HMASK = 0, Q3K_QS = 32, Q3K_SC = 96,
              Q3K_D = 108;
constexpr int Q4K_BB = 144, Q4K_D = 0, Q4K_DMIN = 2, Q4K_SC = 4, Q4K_QS = 16;
constexpr int Q5K_BB = 176, Q5K_D = 0, Q5K_DMIN = 2, Q5K_SC = 4, Q5K_QH = 16,
              Q5K_QS = 48;
constexpr int Q6K_BB = 210, Q6K_QL = 0, Q6K_QH = 128, Q6K_SC = 192, Q6K_D = 208;

// int(rint(v)): round-half-to-even, matching Metal `int(rint(v))` (the K-quant
// rounding rule; the flat codecs above use round-half-away-from-zero instead).
inline int kq_nearest_int(float v) {
  return static_cast<int>(std::rint(v));
}

// sum(x^2) over the 256-weight super-block -> sigma2 = factor*sum/256, and
// av_x = sqrt(sigma2). The GPU computes the sum via a simd_sum tree reduction;
// the serial sum here can differ by an ULP, which is why the K-quant CPU/GPU
// byte-exactness oracle is limited to the codecs that do not consume av_x.
inline void
compute_sigma2_av_x(const float* Xs, float& sigma2, float& av_x, float factor) {
  float total = 0.0f;
  for (int i = 0; i < KSB; i++) {
    total += Xs[i] * Xs[i];
  }
  sigma2 = factor * total / 256.0f;
  av_x = std::sqrt(sigma2);
}

// Fit a single positive-only scale to N values via the ggml make_qp_quants
// search (used for the Q4_K/Q5_K/Q2_K super-scale and super-min). Writes the
// quantized levels into L_out and returns the scale.
template <int N>
float kq_make_qp_quants(
    const float* x,
    const float* qw,
    int nmax,
    uint8_t* L_out) {
  float max_v = 0.0f;
  for (int i = 0; i < N; i++) {
    if (x[i] > max_v) {
      max_v = x[i];
    }
  }
  if (max_v < 1e-30f) { // GROUP_MAX_EPS
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

// Fit a scale + min to N values via the ggml make_qkx3_quants search (the
// per-sub-block fit for Q4_K/Q5_K/Q2_K). Returns the scale; writes -best_min
// into the_min.
template <int N>
float kq_make_qkx3_quants(
    const float* x,
    const float* qw,
    int nmax,
    float& the_min) {
  const float rmin = -0.9f;
  const float rdelta = 0.05f;
  const int nstep = 36;

  float min_v = x[0];
  float max_v = x[0];
  float sum_w = qw[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < N; i++) {
    if (x[i] < min_v) {
      min_v = x[i];
    }
    if (x[i] > max_v) {
      max_v = x[i];
    }
    float w = qw[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min_v > 0.0f) {
    min_v = 0.0f;
  }
  if (max_v <= min_v) {
    the_min = -min_v;
    return 0.0f;
  }

  float iscale = float(nmax) / (max_v - min_v);
  float scale = 1.0f / iscale;
  uint8_t L[N];
  float best_mad = 0.0f;
  for (int i = 0; i < N; i++) {
    int l = kq_nearest_int(iscale * (x[i] - min_v));
    L[i] = uint8_t(std::max(0, std::min(nmax, l)));
    float diff = scale * float(L[i]) + min_v - x[i];
    best_mad += qw[i] * diff * diff;
  }
  float best_min = min_v;

  uint8_t Laux[N];
  for (int is = 0; is <= nstep; is++) {
    float iscale_is =
        (rmin + rdelta * float(is) + float(nmax)) / (max_v - min_v);
    float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
    for (int i = 0; i < N; i++) {
      int l = kq_nearest_int(iscale_is * (x[i] - min_v));
      l = std::max(0, std::min(nmax, l));
      Laux[i] = uint8_t(l);
      float w = qw[i];
      sum_l += w * float(l);
      sum_l2 += w * float(l) * float(l);
      sum_xl += w * float(l) * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0.0f) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0.0f) {
        this_min = 0.0f;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0.0f;
      for (int i = 0; i < N; i++) {
        float diff = this_scale * float(Laux[i]) + this_min - x[i];
        mad += qw[i] * diff * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < N; i++) {
          L[i] = Laux[i];
        }
        best_mad = mad;
        scale = this_scale;
        best_min = this_min;
      }
    }
  }
  the_min = -best_min;
  return scale;
}

// Fit a signed scale to 16 values via the ggml make_qx_quants search (the
// per-sub-block fit for Q6_K/Q3_K). qw may be null -> weight by x^2.
inline float kq_make_qx_quants_16(
    const float* x,
    const float* qw,
    int nmax,
    uint8_t* L_out) {
  const int n = 16;
  float max_v = 0.0f;
  float amax = 0.0f;
  for (int i = 0; i < n; i++) {
    float ax = std::fabs(x[i]);
    if (ax > amax) {
      amax = ax;
      max_v = x[i];
    }
  }
  if (amax < 1e-30f) {
    for (int i = 0; i < n; i++) {
      L_out[i] = uint8_t(nmax);
    }
    return 0.0f;
  }
  float iscale = -float(nmax) / max_v;
  float sumlx = 0.0f, suml2 = 0.0f;
  for (int i = 0; i < n; i++) {
    int l = kq_nearest_int(iscale * x[i]);
    l = std::max(-nmax, std::min(nmax - 1, l));
    L_out[i] = uint8_t(l + nmax);
    float w = (qw != nullptr) ? qw[i] : x[i] * x[i];
    sumlx += w * x[i] * float(l);
    suml2 += w * float(l) * float(l);
  }
  float scale = (suml2 > 0.0f) ? (sumlx / suml2) : 0.0f;
  float best = scale * sumlx;
  for (int is = -9; is <= 9; is++) {
    if (is == 0) {
      continue;
    }
    float iscale_is = -(float(nmax) + 0.1f * float(is)) / max_v;
    float slx = 0.0f, sl2 = 0.0f;
    for (int i = 0; i < n; i++) {
      int l = kq_nearest_int(iscale_is * x[i]);
      l = std::max(-nmax, std::min(nmax - 1, l));
      float w = (qw != nullptr) ? qw[i] : x[i] * x[i];
      slx += w * x[i] * float(l);
      sl2 += w * float(l) * float(l);
    }
    if (sl2 > 0.0f && slx * slx > best * sl2) {
      for (int i = 0; i < n; i++) {
        int l = kq_nearest_int(iscale_is * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L_out[i] = uint8_t(l + nmax);
      }
      scale = slx / sl2;
      best = scale * slx;
    }
  }
  return scale;
}

// Pack the eight 6-bit scale/min pairs into the 12-byte Q4_K/Q5_K scales field
// (ggml get_scale_min_k4 inverse), matching the kernel's kq_pack_scales12.
inline void
kq_pack_scales12(uint8_t* scales12, const uint8_t* Ls, const uint8_t* Lm) {
  for (int i = 0; i < 12; i++) {
    scales12[i] = 0;
  }
  for (int j = 0; j < 8; j++) {
    uint8_t ls = Ls[j];
    uint8_t lm = Lm[j];
    if (j < 4) {
      scales12[j] = ls;
      scales12[j + 4] = lm;
    } else {
      scales12[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
      scales12[j - 4] |= ((ls >> 4) << 6);
      scales12[j] |= ((lm >> 4) << 6);
    }
  }
}

// Unpack one 6-bit scale + min from the Q4_K/Q5_K scales field (re-quant
// phase).
inline void kq_unpack_scale_min_k4(
    int j,
    const uint8_t* scales12,
    uint8_t& sc_out,
    uint8_t& mn_out) {
  if (j < 4) {
    sc_out = scales12[j] & 0x3F;
    mn_out = scales12[j + 4] & 0x3F;
  } else {
    sc_out = (scales12[j + 4] & 0x0F) | ((scales12[j - 4] >> 6) << 4);
    mn_out = (scales12[j + 4] >> 4) | ((scales12[j] >> 6) << 4);
  }
}

// Unpack one 6-bit (biased) scale from the Q3_K 12-byte scales field.
inline uint8_t kq_q3_k_unpack_scale(int j, const uint8_t* q12) {
  const int quad = j / 4;
  const int byte = j & 3;
  const uint8_t low4 = (q12[(quad & 1) * 4 + byte] >> ((quad >> 1) * 4)) & 0x0F;
  const uint8_t high2 = (q12[8 + byte] >> (quad * 2)) & 0x03;
  return uint8_t(low4 | (high2 << 4));
}

// Q4_K (bits=4) and Q5_K (bits=5) share an encoder; Q5_K additionally packs the
// 5th bit of each level into the 32-byte qh field.
template <typename T, int bits>
void quantize_q45_k(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const float* imatrix,
    std::size_t K) {
  constexpr int nmax = (bits == 4) ? 15 : 31;
  constexpr int BB = (bits == 4) ? Q4K_BB : Q5K_BB;
  constexpr int D = (bits == 4) ? Q4K_D : Q5K_D;
  constexpr int DMIN = (bits == 4) ? Q4K_DMIN : Q5K_DMIN;
  constexpr int SC = (bits == 4) ? Q4K_SC : Q5K_SC;
  constexpr int QS = (bits == 4) ? Q4K_QS : Q5K_QS;
  const bool has_imatrix = imatrix != nullptr;
  std::size_t num_blocks = num_weights / KSB;

  float Xs[KSB], QWs[KSB];
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x_global = w + b * KSB;
    uint8_t* block = out + b * BB;
    for (int i = 0; i < KSB; i++) {
      Xs[i] = float(x_global[i]);
    }
    float sigma2, av_x;
    compute_sigma2_av_x(Xs, sigma2, av_x, 2.0f);
    if (has_imatrix) {
      std::size_t k_off = (b * KSB) % K;
      for (int i = 0; i < KSB; i++) {
        QWs[i] = imatrix[k_off + i] * std::sqrt(sigma2 + Xs[i] * Xs[i]);
      }
    } else {
      for (int i = 0; i < KSB; i++) {
        QWs[i] = av_x + std::fabs(Xs[i]);
      }
    }

    float scales_sb[8], mins_sb[8], sw_sb[8];
    for (int sb = 0; sb < 8; sb++) {
      float sumw = 0.0f;
      for (int l = 0; l < 32; l++) {
        sumw += QWs[sb * 32 + l];
      }
      sw_sb[sb] = sumw;
      float the_min;
      scales_sb[sb] =
          kq_make_qkx3_quants<32>(&Xs[sb * 32], &QWs[sb * 32], nmax, the_min);
      mins_sb[sb] = the_min;
    }

    uint8_t Ls[8], Lm[8];
    float d_block = kq_make_qp_quants<8>(scales_sb, sw_sb, 63, Ls);
    float m_block = kq_make_qp_quants<8>(mins_sb, sw_sb, 63, Lm);
    write_f16(block + D, d_block);
    write_f16(block + DMIN, m_block);
    kq_pack_scales12(block + SC, Ls, Lm);

    float d_wire = read_f16(block + D);
    float dmin_wire = read_f16(block + DMIN);
    uint8_t L[KSB];
    for (int lid = 0; lid < KSB; lid++) {
      int my_sb = lid / 32;
      uint8_t sc, mn;
      kq_unpack_scale_min_k4(my_sb, block + SC, sc, mn);
      float d_final = d_wire * float(sc);
      float dm_final = dmin_wire * float(mn);
      if (d_final == 0.0f) {
        L[lid] = 0;
      } else {
        int l = kq_nearest_int((Xs[lid] + dm_final) / d_final);
        l = std::max(0, std::min(nmax, l));
        L[lid] = uint8_t(l);
      }
    }

    uint8_t* qs = block + QS;
    for (int lid = 0; lid < 128; lid++) {
      int stride = lid / 32;
      int l = lid % 32;
      uint8_t lo = L[64 * stride + l] & 0x0F;
      uint8_t hi = L[64 * stride + l + 32] & 0x0F;
      qs[32 * stride + l] = lo | (hi << 4);
    }
    if (bits == 5) {
      uint8_t* qh = block + Q5K_QH;
      for (int j = 0; j < 32; j++) {
        uint8_t bbyte = 0;
        for (int block_idx = 0; block_idx < 8; block_idx++) {
          if (L[block_idx * 32 + j] > 15) {
            bbyte |= uint8_t(1 << block_idx);
          }
        }
        qh[j] = bbyte;
      }
    }
  }
}

template <typename T>
void quantize_q6_k(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const float* imatrix,
    std::size_t K) {
  const bool has_imatrix = imatrix != nullptr;
  std::size_t num_blocks = num_weights / KSB;

  float Xs[KSB], QWs[KSB];
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x_global = w + b * KSB;
    uint8_t* block = out + b * Q6K_BB;
    for (int i = 0; i < KSB; i++) {
      Xs[i] = float(x_global[i]);
    }
    if (has_imatrix) {
      std::size_t k_off = (b * KSB) % K;
      for (int i = 0; i < KSB; i++) {
        QWs[i] = imatrix[k_off + i];
      }
    }

    float scales_sb[16];
    for (int sb = 0; sb < 16; sb++) {
      uint8_t L_local[16];
      const float* qw = has_imatrix ? &QWs[sb * 16] : nullptr;
      scales_sb[sb] = kq_make_qx_quants_16(&Xs[sb * 16], qw, 32, L_local);
    }

    float max_scale = 0.0f, max_abs_scale = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
      float s = scales_sb[sb];
      float a = std::fabs(s);
      if (a > max_abs_scale) {
        max_abs_scale = a;
        max_scale = s;
      }
    }
    if (max_abs_scale < 1e-30f) {
      for (int i = 0; i < Q6K_BB; i++) {
        block[i] = 0;
      }
      continue;
    }

    float iscale = -128.0f / max_scale;
    write_f16(block + Q6K_D, 1.0f / iscale);
    int8_t* scales_out = reinterpret_cast<int8_t*>(block + Q6K_SC);
    for (int sb = 0; sb < 16; sb++) {
      int s = kq_nearest_int(iscale * scales_sb[sb]);
      s = std::min(127, s);
      scales_out[sb] = int8_t(s);
    }
    float d_wire = read_f16(block + Q6K_D);

    uint8_t L[KSB];
    for (int lid = 0; lid < KSB; lid++) {
      int my_sb = lid / 16;
      float d_eff = d_wire * float(scales_out[my_sb]);
      if (d_eff == 0.0f) {
        L[lid] = 32;
      } else {
        int l = kq_nearest_int(Xs[lid] / d_eff);
        l = std::max(-32, std::min(31, l));
        L[lid] = uint8_t(l + 32);
      }
    }

    for (int lid = 0; lid < 64; lid++) {
      int stride = lid / 32;
      int l = lid % 32;
      int base = stride * 128;
      uint8_t La = L[base + l];
      uint8_t Lb = L[base + l + 32];
      uint8_t Lc = L[base + l + 64];
      uint8_t Ld = L[base + l + 96];
      uint8_t* ql_out = block + Q6K_QL + stride * 64;
      uint8_t* qh_out = block + Q6K_QH + stride * 32;
      ql_out[l] = (La & 0x0F) | ((Lc & 0x0F) << 4);
      ql_out[l + 32] = (Lb & 0x0F) | ((Ld & 0x0F) << 4);
      qh_out[l] =
          (La >> 4) | ((Lb >> 4) << 2) | ((Lc >> 4) << 4) | ((Ld >> 4) << 6);
    }
  }
}

template <typename T>
void quantize_q3_k(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const float* imatrix,
    std::size_t K) {
  const bool has_imatrix = imatrix != nullptr;
  std::size_t num_blocks = num_weights / KSB;

  float Xs[KSB], QWs[KSB];
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x_global = w + b * KSB;
    uint8_t* block = out + b * Q3K_BB;
    for (int i = 0; i < KSB; i++) {
      Xs[i] = float(x_global[i]);
    }
    if (has_imatrix) {
      float sigma2, av_x;
      compute_sigma2_av_x(Xs, sigma2, av_x, 2.0f);
      std::size_t k_off = (b * KSB) % K;
      for (int i = 0; i < KSB; i++) {
        QWs[i] = imatrix[k_off + i] * std::sqrt(sigma2 + Xs[i] * Xs[i]);
      }
    }

    float scales_sb[16];
    for (int sb = 0; sb < 16; sb++) {
      uint8_t L_local[16];
      const float* qw = has_imatrix ? &QWs[sb * 16] : nullptr;
      scales_sb[sb] = kq_make_qx_quants_16(&Xs[sb * 16], qw, 4, L_local);
    }

    float amax_sc = 0.0f, max_sc = 0.0f;
    for (int sb = 0; sb < 16; sb++) {
      float v = scales_sb[sb];
      float av = std::fabs(v);
      if (av > amax_sc) {
        amax_sc = av;
        max_sc = v;
      }
    }
    uint8_t* scales12 = block + Q3K_SC;
    for (int i = 0; i < 12; i++) {
      scales12[i] = 0;
    }
    float d_block = 0.0f;
    if (max_sc != 0.0f) {
      float iscale = -32.0f / max_sc;
      for (int sb = 0; sb < 16; sb++) {
        int l = kq_nearest_int(iscale * scales_sb[sb]);
        l = std::max(-32, std::min(31, l)) + 32; // biased [0, 63]
        if (sb < 8) {
          scales12[sb] = uint8_t(l & 0x0F);
        } else {
          scales12[sb - 8] |= uint8_t((l & 0x0F) << 4);
        }
        uint8_t lh = uint8_t((l >> 4) & 0x03);
        scales12[8 + (sb % 4)] |= uint8_t(lh << (2 * (sb / 4)));
      }
      d_block = 1.0f / iscale;
    }
    write_f16(block + Q3K_D, d_block);

    float d_fp16 = read_f16(block + Q3K_D);
    uint8_t L[KSB];
    for (int lid = 0; lid < KSB; lid++) {
      int my_sb = lid / 16;
      uint8_t sc_unsigned = kq_q3_k_unpack_scale(my_sb, scales12);
      int sc_signed = int(sc_unsigned) - 32;
      float d_eff = d_fp16 * float(sc_signed);
      if (d_eff == 0.0f) {
        L[lid] = 4;
      } else {
        int l = kq_nearest_int(Xs[lid] / d_eff);
        l = std::max(-4, std::min(3, l));
        L[lid] = uint8_t(l + 4);
      }
    }

    uint8_t* qs = block + Q3K_QS;
    for (int lid = 0; lid < 64; lid++) {
      int outer_half = lid / 32;
      int within_shift = lid % 32;
      uint8_t byte = 0;
      for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
        int w_idx = outer_half * 128 + shift_idx * 32 + within_shift;
        byte |= uint8_t((L[w_idx] & 0x03) << (shift_idx * 2));
      }
      qs[lid] = byte;
    }
    uint8_t* hmask = block + Q3K_HMASK;
    for (int within_shift = 0; within_shift < 32; within_shift++) {
      uint8_t byte = 0;
      for (int bbit = 0; bbit < 8; bbit++) {
        int outer_half = bbit / 4;
        int shift_idx = bbit % 4;
        int w_idx = outer_half * 128 + shift_idx * 32 + within_shift;
        uint8_t hbit = (L[w_idx] >> 2) & 0x01;
        byte |= uint8_t(hbit << bbit);
      }
      hmask[within_shift] = byte;
    }
  }
}

template <typename T>
void quantize_q2_k(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const float* imatrix,
    std::size_t K) {
  const bool has_imatrix = imatrix != nullptr;
  std::size_t num_blocks = num_weights / KSB;

  float Xs[KSB], QWs[KSB];
  for (std::size_t b = 0; b < num_blocks; b++) {
    const T* x_global = w + b * KSB;
    uint8_t* block = out + b * Q2K_BB;
    for (int i = 0; i < KSB; i++) {
      Xs[i] = float(x_global[i]);
    }
    float sigma2, av_x;
    compute_sigma2_av_x(Xs, sigma2, av_x, 1.0f);
    if (has_imatrix) {
      std::size_t k_off = (b * KSB) % K;
      for (int i = 0; i < KSB; i++) {
        QWs[i] = imatrix[k_off + i] * std::sqrt(sigma2 + Xs[i] * Xs[i]);
      }
    } else {
      for (int i = 0; i < KSB; i++) {
        QWs[i] = av_x + std::fabs(Xs[i]);
      }
    }

    float scales_sb[16], mins_sb[16], sw_sb[16];
    for (int sb = 0; sb < 16; sb++) {
      float sumw = 0.0f;
      for (int l = 0; l < 16; l++) {
        sumw += QWs[sb * 16 + l];
      }
      sw_sb[sb] = sumw;
      float the_min;
      scales_sb[sb] =
          kq_make_qkx3_quants<16>(&Xs[sb * 16], &QWs[sb * 16], 3, the_min);
      mins_sb[sb] = the_min;
    }

    uint8_t Ls[16], Lm[16];
    float d_block = kq_make_qp_quants<16>(scales_sb, sw_sb, 15, Ls);
    float m_block = kq_make_qp_quants<16>(mins_sb, sw_sb, 15, Lm);
    write_f16(block + Q2K_D, d_block);
    write_f16(block + Q2K_DMIN, m_block);
    uint8_t* scales16 = block + Q2K_SC;
    for (int j = 0; j < 16; j++) {
      scales16[j] = uint8_t((Ls[j] & 0x0F) | ((Lm[j] & 0x0F) << 4));
    }

    float d_wire = read_f16(block + Q2K_D);
    float dmin_wire = read_f16(block + Q2K_DMIN);
    uint8_t L[KSB];
    for (int lid = 0; lid < KSB; lid++) {
      int my_sb = lid / 16;
      uint8_t sc_byte = scales16[my_sb];
      uint8_t sc = sc_byte & 0x0F;
      uint8_t mn = sc_byte >> 4;
      float d_eff = d_wire * float(sc);
      float m_eff = dmin_wire * float(mn);
      if (d_eff == 0.0f) {
        L[lid] = 0;
      } else {
        int l = kq_nearest_int((Xs[lid] + m_eff) / d_eff);
        l = std::max(0, std::min(3, l));
        L[lid] = uint8_t(l);
      }
    }

    uint8_t* qs = block + Q2K_QS;
    for (int lid = 0; lid < 64; lid++) {
      int outer_half = lid / 32;
      int within_shift = lid % 32;
      uint8_t byte = 0;
      for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
        int w_idx = outer_half * 128 + shift_idx * 32 + within_shift;
        byte |= uint8_t((L[w_idx] & 0x03) << (shift_idx * 2));
      }
      qs[lid] = byte;
    }
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
  // Flat codecs (q8_0 + legacy) have no importance-weighted rounding path, so
  // they ignore imatrix/K; the five K-quants consume both.
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
  } else if (kquant_type == "q2_k") {
    quantize_q2_k(w, out, num_weights, imatrix, K);
  } else if (kquant_type == "q3_k") {
    quantize_q3_k(w, out, num_weights, imatrix, K);
  } else if (kquant_type == "q4_k") {
    quantize_q45_k<T, 4>(w, out, num_weights, imatrix, K);
  } else if (kquant_type == "q5_k") {
    quantize_q45_k<T, 5>(w, out, num_weights, imatrix, K);
  } else if (kquant_type == "q6_k") {
    quantize_q6_k(w, out, num_weights, imatrix, K);
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
