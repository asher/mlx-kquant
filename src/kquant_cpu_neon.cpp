// NEON-dotprod int8 vec-dot kernels for the fused CPU GEMV (see the header
// for the dispatch contract). Activation rows are quantized once per matmul
// call to int8 with a per-block float scale (k-quants additionally keep
// per-16-element sums so the codec mins can be applied in one f32 fixup),
// then each weight row is dotted against the wire nibbles with vdotq_s32.
// The per-codec bit-twiddling mirrors ggml's arm64 vec_dot kernels
// (llama.cpp, MIT) - see mlx_kquant/licenses/llama.cpp-LICENSE.
#include "kquant_cpu_neon.h"

#ifdef KQ_CPU_NEON_TU

#include <cstdlib>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#define KQ_NEON_DOTPROD 1
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif
#endif

namespace mlx_kquant {

#ifdef KQ_NEON_DOTPROD

namespace {

// ---------------------------------------------------------------------------
// Activation q8 layouts
// ---------------------------------------------------------------------------

// Per 256-weight superblock (k-quant codecs). bsums (sums of 16-element
// groups) let the q4_k/q5_k mins and the q6_k -32 offset be applied as one
// f32 fixup instead of per-weight work.
struct ActQ8K {
  float d; // dequant scale: x ~= d * qs
  int8_t qs[256];
  int16_t bsums[16];
};
static_assert(sizeof(ActQ8K) == 292, "ActQ8K layout");

// Per 32-weight block (q8_0).
struct ActQ80 {
  float d;
  int8_t qs[32];
};
static_assert(sizeof(ActQ80) == 36, "ActQ80 layout");

inline float read_f16(const uint8_t* ptr) {
  _Float16 tmp;
  std::memcpy(&tmp, ptr, sizeof(_Float16));
  return static_cast<float>(tmp);
}

// Round-to-nearest-even via the float mantissa trick (|fval| must be < 2^22;
// quantized activations are bounded by 127).
inline int nearest_int(float fval) {
  float val = fval + 12582912.f;
  int i;
  std::memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

// ---------------------------------------------------------------------------
// Activation row quantizers (scalar: M <= 16 rows per call, negligible next
// to the N x K dot work)
// ---------------------------------------------------------------------------

void quantize_act_row_q8k(const float* x, void* vy, int k) {
  ActQ8K* y = static_cast<ActQ8K*>(vy);
  const int nb = k / 256;
  for (int i = 0; i < nb; i++) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int j = 0; j < 256; ++j) {
      float ax = x[j] < 0 ? -x[j] : x[j];
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (amax == 0.0f) {
      y[i].d = 0.0f;
      std::memset(y[i].qs, 0, sizeof(y[i].qs));
      std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
      x += 256;
      continue;
    }
    const float iscale = -127.f / max;
    for (int j = 0; j < 256; ++j) {
      int v = nearest_int(iscale * x[j]);
      y[i].qs[j] = static_cast<int8_t>(v > 127 ? 127 : v);
    }
    for (int j = 0; j < 16; ++j) {
      int sum = 0;
      for (int l = 0; l < 16; ++l) {
        sum += y[i].qs[j * 16 + l];
      }
      y[i].bsums[j] = static_cast<int16_t>(sum);
    }
    y[i].d = 1.0f / iscale;
    x += 256;
  }
}

void quantize_act_row_q80(const float* x, void* vy, int k) {
  ActQ80* y = static_cast<ActQ80*>(vy);
  const int nb = k / 32;
  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
      float ax = x[j] < 0 ? -x[j] : x[j];
      if (ax > amax) {
        amax = ax;
      }
    }
    const float d = amax / 127.f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    for (int j = 0; j < 32; ++j) {
      y[i].qs[j] = static_cast<int8_t>(nearest_int(x[j] * id));
    }
    y[i].d = d;
    x += 32;
  }
}

// ---------------------------------------------------------------------------
// Per-codec vec-dot kernels
// ---------------------------------------------------------------------------

constexpr uint32_t kmask1 = 0x3f3f3f3f;
constexpr uint32_t kmask2 = 0x0f0f0f0f;
constexpr uint32_t kmask3 = 0x03030303;

// q4_k block: d f16 @0, dmin f16 @2, packed scales/mins[12] @4, qs[128] @16.
float vec_dot_q4_k(const uint8_t* w, const void* act, int nb) {
  const ActQ8K* y = static_cast<const ActQ8K*>(act);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  uint32_t utmp[4];
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 144;
    const float d = y[i].d * read_f16(xb);
    const float dmin = y[i].d * read_f16(xb + 2);

    const int16x8_t q8sums =
        vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

    std::memcpy(utmp, xb + 4, 12);
    uint32x2_t mins8 = vdup_n_u32(0);
    mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
    mins8 = vset_lane_u32(
        ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[0] &= kmask1;

    const int16x8_t mins =
        vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
    const int32x4_t prod = vaddq_s32(
        vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
        vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
    sumf -= dmin * static_cast<float>(vaddvq_s32(prod));

    const uint8_t* scales = reinterpret_cast<const uint8_t*>(utmp);
    const uint8_t* q4 = xb + 16;
    const int8_t* q8 = y[i].qs;

    int32_t sumi1 = 0;
    int32_t sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
      const uint8x16_t q4bits0 = vld1q_u8(q4);
      const uint8x16_t q4bits1 = vld1q_u8(q4 + 16);
      q4 += 32;

      int8x16_t q8b0 = vld1q_s8(q8);
      int8x16_t q8b1 = vld1q_s8(q8 + 16);
      q8 += 32;
      const int8x16_t q4lo0 = vreinterpretq_s8_u8(vandq_u8(q4bits0, m4b));
      const int8x16_t q4lo1 = vreinterpretq_s8_u8(vandq_u8(q4bits1, m4b));
      const int32x4_t p1 =
          vdotq_s32(vdotq_s32(mzero, q4lo0, q8b0), q4lo1, q8b1);
      sumi1 += vaddvq_s32(p1) * scales[2 * j + 0];

      q8b0 = vld1q_s8(q8);
      q8b1 = vld1q_s8(q8 + 16);
      q8 += 32;
      const int8x16_t q4hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits0, 4));
      const int8x16_t q4hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits1, 4));
      const int32x4_t p2 =
          vdotq_s32(vdotq_s32(mzero, q4hi0, q8b0), q4hi1, q8b1);
      sumi2 += vaddvq_s32(p2) * scales[2 * j + 1];
    }
    sumf += d * static_cast<float>(sumi1 + sumi2);
  }
  return sumf;
}

// q5_k block: d f16 @0, dmin f16 @2, packed scales/mins[12] @4, qh[32] @16,
// qs[128] @48.
float vec_dot_q5_k(const uint8_t* w, const void* act, int nb) {
  const ActQ8K* y = static_cast<const ActQ8K*>(act);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const uint8x16_t mone = vdupq_n_u8(1);
  const uint8x16_t mtwo = vdupq_n_u8(2);
  const int32x4_t mzero = vdupq_n_s32(0);
  uint32_t utmp[4];
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 176;
    const float d = y[i].d * read_f16(xb);
    const float dmin = y[i].d * read_f16(xb + 2);

    const int16x8_t q8sums =
        vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

    std::memcpy(utmp, xb + 4, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const uint8x8_t mins8 = vld1_u8(reinterpret_cast<const uint8_t*>(utmp) + 8);
    const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
    const int32x4_t prod = vaddq_s32(
        vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
        vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
    const int32_t sumi_mins = vaddvq_s32(prod);

    const uint8_t* scales = reinterpret_cast<const uint8_t*>(utmp);
    const uint8_t* q5 = xb + 48;
    const int8_t* q8 = y[i].qs;

    uint8x16_t qhbits0 = vld1q_u8(xb + 16);
    uint8x16_t qhbits1 = vld1q_u8(xb + 32);

    int32_t sumi = 0;
    for (int j = 0; j < 4; ++j) {
      const uint8x16_t q5bits0 = vld1q_u8(q5);
      const uint8x16_t q5bits1 = vld1q_u8(q5 + 16);
      q5 += 32;
      const int8x16_t q8b0 = vld1q_s8(q8);
      const int8x16_t q8b1 = vld1q_s8(q8 + 16);
      const int8x16_t q8b2 = vld1q_s8(q8 + 32);
      const int8x16_t q8b3 = vld1q_s8(q8 + 48);
      q8 += 64;

      const uint8x16_t q5h0 = vshlq_n_u8(vandq_u8(mone, qhbits0), 4);
      const uint8x16_t q5h1 = vshlq_n_u8(vandq_u8(mone, qhbits1), 4);
      const uint8x16_t q5h2 = vshlq_n_u8(vandq_u8(mtwo, qhbits0), 3);
      const uint8x16_t q5h3 = vshlq_n_u8(vandq_u8(mtwo, qhbits1), 3);
      qhbits0 = vshrq_n_u8(qhbits0, 2);
      qhbits1 = vshrq_n_u8(qhbits1, 2);

      const int8x16_t q5b0 =
          vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits0, m4b), q5h0));
      const int8x16_t q5b1 =
          vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits1, m4b), q5h1));
      const int8x16_t q5b2 =
          vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits0, 4), q5h2));
      const int8x16_t q5b3 =
          vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits1, 4), q5h3));

      sumi += vaddvq_s32(vdotq_s32(vdotq_s32(mzero, q5b0, q8b0), q5b1, q8b1)) *
          *scales++;
      sumi += vaddvq_s32(vdotq_s32(vdotq_s32(mzero, q5b2, q8b2), q5b3, q8b3)) *
          *scales++;
    }
    sumf += d * static_cast<float>(sumi) - dmin * static_cast<float>(sumi_mins);
  }
  return sumf;
}

// q6_k block: ql[128] @0, qh[64] @128, scales int8[16] @192, d f16 @208.
float vec_dot_q6_k(const uint8_t* w, const void* act, int nb) {
  const ActQ8K* y = static_cast<const ActQ8K*>(act);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const uint8x16_t mone = vdupq_n_u8(3);
  const int32x4_t mzero = vdupq_n_s32(0);
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 210;
    const float d_all = read_f16(xb + 208);

    const uint8_t* q6 = xb;
    const uint8_t* qh = xb + 128;
    const int8_t* scale = reinterpret_cast<const int8_t*>(xb + 192);
    const int8_t* q8 = y[i].qs;

    // -32 offset fixup: sum over the 16 sub-blocks of scale * bsum.
    const int16x8_t q8sums0 = vld1q_s16(y[i].bsums);
    const int16x8_t q8sums1 = vld1q_s16(y[i].bsums + 8);
    const int8x16_t scales_v = vld1q_s8(scale);
    const int16x8_t q6scales0 = vmovl_s8(vget_low_s8(scales_v));
    const int16x8_t q6scales1 = vmovl_s8(vget_high_s8(scales_v));
    const int32x4_t prod = vaddq_s32(
        vaddq_s32(
            vmull_s16(vget_low_s16(q8sums0), vget_low_s16(q6scales0)),
            vmull_s16(vget_high_s16(q8sums0), vget_high_s16(q6scales0))),
        vaddq_s32(
            vmull_s16(vget_low_s16(q8sums1), vget_low_s16(q6scales1)),
            vmull_s16(vget_high_s16(q8sums1), vget_high_s16(q6scales1))));
    const int32_t isum_mins = vaddvq_s32(prod);

    int32_t isum = 0;
    for (int j = 0; j < 2; ++j) {
      const uint8x16_t qhbits0 = vld1q_u8(qh);
      const uint8x16_t qhbits1 = vld1q_u8(qh + 16);
      qh += 32;
      const uint8x16_t q6bits0 = vld1q_u8(q6);
      const uint8x16_t q6bits1 = vld1q_u8(q6 + 16);
      const uint8x16_t q6bits2 = vld1q_u8(q6 + 32);
      const uint8x16_t q6bits3 = vld1q_u8(q6 + 48);
      q6 += 64;
      int8x16_t q8b0 = vld1q_s8(q8);
      int8x16_t q8b1 = vld1q_s8(q8 + 16);
      int8x16_t q8b2 = vld1q_s8(q8 + 32);
      int8x16_t q8b3 = vld1q_s8(q8 + 48);
      q8 += 64;

      uint8x16_t q6h0 = vshlq_n_u8(vandq_u8(mone, qhbits0), 4);
      uint8x16_t q6h1 = vshlq_n_u8(vandq_u8(mone, qhbits1), 4);
      uint8x16_t shifted = vshrq_n_u8(qhbits0, 2);
      uint8x16_t q6h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits1, 2);
      uint8x16_t q6h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      int8x16_t q6b0 =
          vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits0, m4b), q6h0));
      int8x16_t q6b1 =
          vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits1, m4b), q6h1));
      int8x16_t q6b2 =
          vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits2, m4b), q6h2));
      int8x16_t q6b3 =
          vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits3, m4b), q6h3));

      isum += vaddvq_s32(vdotq_s32(mzero, q6b0, q8b0)) * scale[0] +
          vaddvq_s32(vdotq_s32(mzero, q6b1, q8b1)) * scale[1] +
          vaddvq_s32(vdotq_s32(mzero, q6b2, q8b2)) * scale[2] +
          vaddvq_s32(vdotq_s32(mzero, q6b3, q8b3)) * scale[3];
      scale += 4;

      q8b0 = vld1q_s8(q8);
      q8b1 = vld1q_s8(q8 + 16);
      q8b2 = vld1q_s8(q8 + 32);
      q8b3 = vld1q_s8(q8 + 48);
      q8 += 64;

      shifted = vshrq_n_u8(qhbits0, 4);
      q6h0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits1, 4);
      q6h1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits0, 6);
      q6h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits1, 6);
      q6h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      q6b0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits0, 4), q6h0));
      q6b1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits1, 4), q6h1));
      q6b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits2, 4), q6h2));
      q6b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits3, 4), q6h3));

      isum += vaddvq_s32(vdotq_s32(mzero, q6b0, q8b0)) * scale[0] +
          vaddvq_s32(vdotq_s32(mzero, q6b1, q8b1)) * scale[1] +
          vaddvq_s32(vdotq_s32(mzero, q6b2, q8b2)) * scale[2] +
          vaddvq_s32(vdotq_s32(mzero, q6b3, q8b3)) * scale[3];
      scale += 4;
    }
    sumf += d_all * y[i].d * static_cast<float>(isum - 32 * isum_mins);
  }
  return sumf;
}

// q8_0 block: d f16 @0, qs int8[32] @2.
float vec_dot_q8_0(const uint8_t* w, const void* act, int nb) {
  const ActQ80* y = static_cast<const ActQ80*>(act);
  const int32x4_t mzero = vdupq_n_s32(0);
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 1 < nb; i += 2) {
    const uint8_t* xb0 = w + static_cast<std::size_t>(i) * 34;
    const uint8_t* xb1 = xb0 + 34;
    const int8_t* q0 = reinterpret_cast<const int8_t*>(xb0 + 2);
    const int8_t* q1 = reinterpret_cast<const int8_t*>(xb1 + 2);

    const int32x4_t p0 = vdotq_s32(
        vdotq_s32(mzero, vld1q_s8(q0), vld1q_s8(y[i].qs)),
        vld1q_s8(q0 + 16),
        vld1q_s8(y[i].qs + 16));
    const int32x4_t p1 = vdotq_s32(
        vdotq_s32(mzero, vld1q_s8(q1), vld1q_s8(y[i + 1].qs)),
        vld1q_s8(q1 + 16),
        vld1q_s8(y[i + 1].qs + 16));
    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), read_f16(xb0) * y[i].d);
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), read_f16(xb1) * y[i + 1].d);
  }
  for (; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 34;
    const int8_t* q = reinterpret_cast<const int8_t*>(xb + 2);
    const int32x4_t p = vdotq_s32(
        vdotq_s32(mzero, vld1q_s8(q), vld1q_s8(y[i].qs)),
        vld1q_s8(q + 16),
        vld1q_s8(y[i].qs + 16));
    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), read_f16(xb) * y[i].d);
  }
  return vaddvq_f32(vaddq_f32(sumv0, sumv1));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

constexpr KQNeonKernel kKernelQ4K = {
    &vec_dot_q4_k,
    sizeof(ActQ8K),
    &quantize_act_row_q8k};
constexpr KQNeonKernel kKernelQ5K = {
    &vec_dot_q5_k,
    sizeof(ActQ8K),
    &quantize_act_row_q8k};
constexpr KQNeonKernel kKernelQ6K = {
    &vec_dot_q6_k,
    sizeof(ActQ8K),
    &quantize_act_row_q8k};
constexpr KQNeonKernel kKernelQ80 = {
    &vec_dot_q8_0,
    sizeof(ActQ80),
    &quantize_act_row_q80};

bool cpu_has_dotprod() {
#if defined(__APPLE__)
  return true; // FEAT_DotProd is in the Apple Silicon baseline (M1+)
#elif defined(__linux__)
  static const bool has = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
  return has;
#else
  return false;
#endif
}

} // namespace

const KQNeonKernel* kq_neon_kernel(const std::string& codec) {
  if (!cpu_has_dotprod()) {
    return nullptr;
  }
  // Read per call (not cached) so the kill switch can be flipped at runtime,
  // e.g. by tests A/B-ing the int8 and scalar paths in one process.
  const char* env = std::getenv("KQ_CPU_NEON");
  if (env != nullptr && env[0] == '0' && env[1] == '\0') {
    return nullptr;
  }
  if (codec == "q4_k") {
    return &kKernelQ4K;
  } else if (codec == "q5_k") {
    return &kKernelQ5K;
  } else if (codec == "q6_k") {
    return &kKernelQ6K;
  } else if (codec == "q8_0") {
    return &kKernelQ80;
  }
  // q2_k / q3_k / q4_0 / q4_1 / q5_0 / q5_1: scalar f32 fallback.
  return nullptr;
}

bool kq_cpu_neon_available() {
  return kq_neon_kernel("q4_k") != nullptr;
}

#else // !KQ_NEON_DOTPROD (arm64 TU built without dotprod support)

const KQNeonKernel* kq_neon_kernel(const std::string&) {
  return nullptr;
}

bool kq_cpu_neon_available() {
  return false;
}

#endif // KQ_NEON_DOTPROD

} // namespace mlx_kquant

#endif // KQ_CPU_NEON_TU
