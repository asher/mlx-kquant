// NEON-dotprod int8 vec-dot kernels for the fused CPU GEMV (see the header
// for the dispatch contract). Activation rows are quantized once per matmul
// call to int8 with a per-block float scale (k-quants additionally keep
// per-16-element sums, and q4_1/q5_1 a per-block sum, so the codec
// mins/offsets can be applied in one f32 fixup), then each weight row is
// dotted against the wire nibbles with vdotq_s32.
// The per-codec bit-twiddling mirrors ggml's arm64 vec_dot kernels
// (llama.cpp, MIT) - see mlx_kquant/licenses/llama.cpp-LICENSE.
#include "kquant_cpu_neon.h"

#ifdef KQ_CPU_NEON_TU

#include <cstdlib>
#include <cstring>

#include "kquant_iq_tables.h"

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

// Per 32-weight block (q8_0, and the q4_0/q5_0 weight codecs, whose offset
// is folded into the signed int8 weight values).
struct ActQ80 {
  float d;
  int8_t qs[32];
};
static_assert(sizeof(ActQ80) == 36, "ActQ80 layout");

// Per 32-weight block with the block sum pre-multiplied by the scale
// (ggml's q8_1): lets the q4_1/q5_1 per-block min `m` fold into one
// `m * s` f32 fixup instead of per-weight work.
struct ActQ81 {
  float d;
  float s; // d * sum(qs)
  int8_t qs[32];
};
static_assert(sizeof(ActQ81) == 40, "ActQ81 layout");

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

void quantize_act_row_q81(const float* x, void* vy, int k) {
  ActQ81* y = static_cast<ActQ81*>(vy);
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
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
      const int v = nearest_int(x[j] * id);
      y[i].qs[j] = static_cast<int8_t>(v);
      sum += v;
    }
    y[i].d = d;
    y[i].s = d * static_cast<float>(sum);
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

// q2_k block: scales/mins u4x2[16] @0, qs[64] @16, d f16 @80, dmin f16 @82.
float vec_dot_q2_k(const uint8_t* w, const void* act, int nb) {
  const ActQ8K* y = static_cast<const ActQ8K*>(act);
  const uint8x16_t m3 = vdupq_n_u8(0x3);
  const uint8x16_t m4 = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 84;
    const float d = y[i].d * read_f16(xb + 80);
    const float dmin = -y[i].d * read_f16(xb + 82);

    // Per-16 mins fixup straight off the activation bsums (the 16 q2_k
    // sub-block mins line up 1:1 with the 16 bsums).
    const uint8x16_t mins_and_scales = vld1q_u8(xb);
    uint8_t scales[16];
    vst1q_u8(scales, vandq_u8(mins_and_scales, m4));
    const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
    const int16x8_t q8sums0 = vld1q_s16(y[i].bsums);
    const int16x8_t q8sums1 = vld1q_s16(y[i].bsums + 8);
    const int16x8_t mins0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
    const int16x8_t mins1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
    const int32x4_t s0 = vaddq_s32(
        vmull_s16(vget_low_s16(mins0), vget_low_s16(q8sums0)),
        vmull_s16(vget_high_s16(mins0), vget_high_s16(q8sums0)));
    const int32x4_t s1 = vaddq_s32(
        vmull_s16(vget_low_s16(mins1), vget_low_s16(q8sums1)),
        vmull_s16(vget_high_s16(mins1), vget_high_s16(q8sums1)));
    sumf += dmin * static_cast<float>(vaddvq_s32(vaddq_s32(s0, s1)));

    const uint8_t* q2 = xb + 16;
    const int8_t* q8 = y[i].qs;
    int32_t isum = 0;
    int is = 0;
    for (int j = 0; j < 2; ++j) {
      const uint8x16_t q2bits0 = vld1q_u8(q2);
      const uint8x16_t q2bits1 = vld1q_u8(q2 + 16);
      q2 += 32;
      const auto accum = [&](uint8x16_t bits0, uint8x16_t bits1) {
        const int8x16_t q8b0 = vld1q_s8(q8);
        const int8x16_t q8b1 = vld1q_s8(q8 + 16);
        q8 += 32;
        const int8x16_t q2b0 = vreinterpretq_s8_u8(vandq_u8(bits0, m3));
        const int8x16_t q2b1 = vreinterpretq_s8_u8(vandq_u8(bits1, m3));
        isum += vaddvq_s32(vdotq_s32(mzero, q2b0, q8b0)) * scales[is++];
        isum += vaddvq_s32(vdotq_s32(mzero, q2b1, q8b1)) * scales[is++];
      };
      accum(q2bits0, q2bits1);
      accum(vshrq_n_u8(q2bits0, 2), vshrq_n_u8(q2bits1, 2));
      accum(vshrq_n_u8(q2bits0, 4), vshrq_n_u8(q2bits1, 4));
      accum(vshrq_n_u8(q2bits0, 6), vshrq_n_u8(q2bits1, 6));
    }
    sumf += d * static_cast<float>(isum);
  }
  return sumf;
}

// q3_k block: hmask[32] @0, qs[64] @32, packed scales[12] @96, d f16 @108.
// The stored high bit means "do NOT subtract 4": vbicq selects the cleared
// bits so q = (low 2 bits) - 4*(hbit==0), all in int8.
float vec_dot_q3_k(const uint8_t* w, const void* act, int nb) {
  const ActQ8K* y = static_cast<const ActQ8K*>(act);
  const uint8x16_t m3b = vdupq_n_u8(0x3);
  const int32x4_t mzero = vdupq_n_s32(0);
  const uint8x16_t mh0 = vdupq_n_u8(1);
  const uint8x16_t mh1 = vshlq_n_u8(mh0, 1);
  const uint8x16_t mh2 = vshlq_n_u8(mh0, 2);
  const uint8x16_t mh3 = vshlq_n_u8(mh0, 3);
  uint32_t aux[3];
  uint32_t utmp[4];
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 110;
    const float d = y[i].d * read_f16(xb + 108);

    uint8x16_t qhbits0 = vld1q_u8(xb);
    uint8x16_t qhbits1 = vld1q_u8(xb + 16);
    const uint8_t* q3 = xb + 32;
    const int8_t* q8 = y[i].qs;

    // Unpack the 16 6-bit scales (centered at 32).
    std::memcpy(aux, xb + 96, 12);
    utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask3) << 4);
    utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask3) << 4);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask3) << 4);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask3) << 4);
    int8_t* scale = reinterpret_cast<int8_t*>(utmp);
    for (int j = 0; j < 16; ++j) {
      scale[j] -= 32;
    }

    int32_t isum = 0;
    for (int j = 0; j < 2; ++j) {
      const uint8x16_t q3bits0 = vld1q_u8(q3);
      const uint8x16_t q3bits1 = vld1q_u8(q3 + 16);
      q3 += 32;

      uint8x16_t q3h0 = vshlq_n_u8(vbicq_u8(mh0, qhbits0), 2);
      uint8x16_t q3h1 = vshlq_n_u8(vbicq_u8(mh0, qhbits1), 2);
      uint8x16_t q3h2 = vshlq_n_u8(vbicq_u8(mh1, qhbits0), 1);
      uint8x16_t q3h3 = vshlq_n_u8(vbicq_u8(mh1, qhbits1), 1);

      int8x16_t q3b0 = vsubq_s8(
          vreinterpretq_s8_u8(vandq_u8(q3bits0, m3b)),
          vreinterpretq_s8_u8(q3h0));
      int8x16_t q3b1 = vsubq_s8(
          vreinterpretq_s8_u8(vandq_u8(q3bits1, m3b)),
          vreinterpretq_s8_u8(q3h1));
      int8x16_t q3b2 = vsubq_s8(
          vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits0, 2), m3b)),
          vreinterpretq_s8_u8(q3h2));
      int8x16_t q3b3 = vsubq_s8(
          vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits1, 2), m3b)),
          vreinterpretq_s8_u8(q3h3));

      isum += vaddvq_s32(vdotq_s32(mzero, q3b0, vld1q_s8(q8))) * scale[0] +
          vaddvq_s32(vdotq_s32(mzero, q3b1, vld1q_s8(q8 + 16))) * scale[1] +
          vaddvq_s32(vdotq_s32(mzero, q3b2, vld1q_s8(q8 + 32))) * scale[2] +
          vaddvq_s32(vdotq_s32(mzero, q3b3, vld1q_s8(q8 + 48))) * scale[3];
      scale += 4;
      q8 += 64;

      q3h0 = vbicq_u8(mh2, qhbits0);
      q3h1 = vbicq_u8(mh2, qhbits1);
      q3h2 = vshrq_n_u8(vbicq_u8(mh3, qhbits0), 1);
      q3h3 = vshrq_n_u8(vbicq_u8(mh3, qhbits1), 1);

      q3b0 = vsubq_s8(
          vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits0, 4), m3b)),
          vreinterpretq_s8_u8(q3h0));
      q3b1 = vsubq_s8(
          vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits1, 4), m3b)),
          vreinterpretq_s8_u8(q3h1));
      q3b2 = vsubq_s8(
          vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits0, 6), m3b)),
          vreinterpretq_s8_u8(q3h2));
      q3b3 = vsubq_s8(
          vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits1, 6), m3b)),
          vreinterpretq_s8_u8(q3h3));

      isum += vaddvq_s32(vdotq_s32(mzero, q3b0, vld1q_s8(q8))) * scale[0] +
          vaddvq_s32(vdotq_s32(mzero, q3b1, vld1q_s8(q8 + 16))) * scale[1] +
          vaddvq_s32(vdotq_s32(mzero, q3b2, vld1q_s8(q8 + 32))) * scale[2] +
          vaddvq_s32(vdotq_s32(mzero, q3b3, vld1q_s8(q8 + 48))) * scale[3];
      scale += 4;
      q8 += 64;

      if (j == 0) {
        qhbits0 = vshrq_n_u8(qhbits0, 4);
        qhbits1 = vshrq_n_u8(qhbits1, 4);
      }
    }
    sumf += d * static_cast<float>(isum);
  }
  return sumf;
}

// q4_0 block: d f16 @0, qs u4x2[16] @2 (values centered at 8).
float vec_dot_q4_0(const uint8_t* w, const void* act, int nb) {
  const ActQ80* y = static_cast<const ActQ80*>(act);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int8x16_t s8b = vdupq_n_s8(0x8);
  const int32x4_t mzero = vdupq_n_s32(0);
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 1 < nb; i += 2) {
    const uint8_t* xb0 = w + static_cast<std::size_t>(i) * 18;
    const uint8_t* xb1 = xb0 + 18;
    const uint8x16_t v0 = vld1q_u8(xb0 + 2);
    const uint8x16_t v1 = vld1q_u8(xb1 + 2);
    const int8x16_t v0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), s8b);
    const int8x16_t v0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8b);
    const int8x16_t v1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4b)), s8b);
    const int8x16_t v1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), s8b);
    const int32x4_t p0 = vdotq_s32(
        vdotq_s32(mzero, v0l, vld1q_s8(y[i].qs)), v0h, vld1q_s8(y[i].qs + 16));
    const int32x4_t p1 = vdotq_s32(
        vdotq_s32(mzero, v1l, vld1q_s8(y[i + 1].qs)),
        v1h,
        vld1q_s8(y[i + 1].qs + 16));
    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), read_f16(xb0) * y[i].d);
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), read_f16(xb1) * y[i + 1].d);
  }
  for (; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 18;
    const uint8x16_t v = vld1q_u8(xb + 2);
    const int8x16_t vl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4b)), s8b);
    const int8x16_t vh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), s8b);
    const int32x4_t p = vdotq_s32(
        vdotq_s32(mzero, vl, vld1q_s8(y[i].qs)), vh, vld1q_s8(y[i].qs + 16));
    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), read_f16(xb) * y[i].d);
  }
  return vaddvq_f32(vaddq_f32(sumv0, sumv1));
}

// q4_1 block: d f16 @0, m f16 @2, qs u4x2[16] @4 (unsigned values; the min
// term folds via the activation block sums).
float vec_dot_q4_1(const uint8_t* w, const void* act, int nb) {
  const ActQ81* y = static_cast<const ActQ81*>(act);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);
  float summs = 0.0f;
  int i = 0;
  for (; i + 1 < nb; i += 2) {
    const uint8_t* xb0 = w + static_cast<std::size_t>(i) * 20;
    const uint8_t* xb1 = xb0 + 20;
    summs += read_f16(xb0 + 2) * y[i].s + read_f16(xb1 + 2) * y[i + 1].s;
    const uint8x16_t v0 = vld1q_u8(xb0 + 4);
    const uint8x16_t v1 = vld1q_u8(xb1 + 4);
    const int8x16_t v0l = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
    const int8x16_t v0h = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
    const int8x16_t v1l = vreinterpretq_s8_u8(vandq_u8(v1, m4b));
    const int8x16_t v1h = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
    const int32x4_t p0 = vdotq_s32(
        vdotq_s32(mzero, v0l, vld1q_s8(y[i].qs)), v0h, vld1q_s8(y[i].qs + 16));
    const int32x4_t p1 = vdotq_s32(
        vdotq_s32(mzero, v1l, vld1q_s8(y[i + 1].qs)),
        v1h,
        vld1q_s8(y[i + 1].qs + 16));
    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), read_f16(xb0) * y[i].d);
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), read_f16(xb1) * y[i + 1].d);
  }
  for (; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 20;
    summs += read_f16(xb + 2) * y[i].s;
    const uint8x16_t v = vld1q_u8(xb + 4);
    const int8x16_t vl = vreinterpretq_s8_u8(vandq_u8(v, m4b));
    const int8x16_t vh = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
    const int32x4_t p = vdotq_s32(
        vdotq_s32(mzero, vl, vld1q_s8(y[i].qs)), vh, vld1q_s8(y[i].qs + 16));
    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), read_f16(xb) * y[i].d);
  }
  return vaddvq_f32(vaddq_f32(sumv0, sumv1)) + summs;
}

// Expand the 8 bits of a byte to 8 bytes: b0 maps bit -> (b << 4) (q5_1's
// high-bit OR), b1 maps bit -> ((!b) << 4) (q5_0: subtracting it applies the
// high bit AND the -16 offset in one vsubq_s8).
struct B2BTables {
  uint64_t b0[256];
  uint64_t b1[256];
};

constexpr B2BTables make_b2b_tables() {
  B2BTables t{};
  for (int v = 0; v < 256; ++v) {
    uint64_t e0 = 0;
    uint64_t e1 = 0;
    for (int j = 0; j < 8; ++j) {
      const uint64_t bit = (static_cast<uint64_t>(v) >> j) & 1;
      e0 |= (bit << 4) << (8 * j);
      e1 |= ((1 - bit) << 4) << (8 * j);
    }
    t.b0[v] = e0;
    t.b1[v] = e1;
  }
  return t;
}

constexpr B2BTables kB2B = make_b2b_tables();

// q5_0 block: d f16 @0, qh u1[32] @2, qs u4x2[16] @6 (values centered at 16).
float vec_dot_q5_0(const uint8_t* w, const void* act, int nb) {
  const ActQ80* y = static_cast<const ActQ80*>(act);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  float32x4_t sumv = vdupq_n_f32(0.0f);
  uint64_t tmp[4];
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 22;
    uint32_t qh;
    std::memcpy(&qh, xb + 2, 4);
    tmp[0] = kB2B.b1[(qh >> 0) & 0xff];
    tmp[1] = kB2B.b1[(qh >> 8) & 0xff];
    tmp[2] = kB2B.b1[(qh >> 16) & 0xff];
    tmp[3] = kB2B.b1[(qh >> 24)];
    const int8x16_t qhl = vld1q_s8(reinterpret_cast<const int8_t*>(tmp + 0));
    const int8x16_t qhh = vld1q_s8(reinterpret_cast<const int8_t*>(tmp + 2));
    const uint8x16_t v = vld1q_u8(xb + 6);
    const int8x16_t vl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4b)), qhl);
    const int8x16_t vh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), qhh);
    const int32x4_t p = vdotq_s32(
        vdotq_s32(mzero, vl, vld1q_s8(y[i].qs)), vh, vld1q_s8(y[i].qs + 16));
    sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), read_f16(xb) * y[i].d);
  }
  return vaddvq_f32(sumv);
}

// q5_1 block: d f16 @0, m f16 @2, qh u1[32] @4, qs u4x2[16] @8 (unsigned
// values; the min term folds via the activation block sums).
float vec_dot_q5_1(const uint8_t* w, const void* act, int nb) {
  const ActQ81* y = static_cast<const ActQ81*>(act);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  float32x4_t sumv = vdupq_n_f32(0.0f);
  float summs = 0.0f;
  uint64_t tmp[4];
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 24;
    summs += read_f16(xb + 2) * y[i].s;
    uint32_t qh;
    std::memcpy(&qh, xb + 4, 4);
    tmp[0] = kB2B.b0[(qh >> 0) & 0xff];
    tmp[1] = kB2B.b0[(qh >> 8) & 0xff];
    tmp[2] = kB2B.b0[(qh >> 16) & 0xff];
    tmp[3] = kB2B.b0[(qh >> 24)];
    const uint8x16_t qhl = vld1q_u8(reinterpret_cast<const uint8_t*>(tmp + 0));
    const uint8x16_t qhh = vld1q_u8(reinterpret_cast<const uint8_t*>(tmp + 2));
    const uint8x16_t v = vld1q_u8(xb + 8);
    const int8x16_t vl = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v, m4b), qhl));
    const int8x16_t vh = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(v, 4), qhh));
    const int32x4_t p = vdotq_s32(
        vdotq_s32(mzero, vl, vld1q_s8(y[i].qs)), vh, vld1q_s8(y[i].qs + 16));
    sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), read_f16(xb) * y[i].d);
  }
  return vaddvq_f32(sumv) + summs;
}

// IQ codecs. Decode-only: grid/LUT weights, signs folded into the int8 weight
// (vmulq_s8) so the inner loop stays a vdotq_s32. The 256-wide IQ codecs reuse
// the q8_K activation (no min term, so bsums are ignored). Mirrors ggml's arm64
// vec_dot_iq* kernels; all grid bytes are < 128, so the int8 reinterpret of the
// uint32 grid agrees with the scalar uint8 decode.

inline uint32x4_t grid4(const uint32_t* g, int a, int b, int c, int d) {
  const uint32_t v[4] = {g[a], g[b], g[c], g[d]};
  return vld1q_u32(v);
}

// iq4_nl block: d f16 @0, qs u4x2[16] @2 -> kvalues_iq4nl LUT. q8_0 activation.
float vec_dot_iq4_nl(const uint8_t* w, const void* act, int nb) {
  const ActQ80* y = static_cast<const ActQ80*>(act);
  const int8x16_t values = vld1q_s8(kvalues_iq4nl);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 1 < nb; i += 2) {
    const uint8_t* xb0 = w + static_cast<std::size_t>(i) * 18;
    const uint8_t* xb1 = xb0 + 18;
    const uint8x16_t q0 = vld1q_u8(xb0 + 2);
    const uint8x16_t q1 = vld1q_u8(xb1 + 2);
    const int8x16_t v0l = vqtbl1q_s8(values, vandq_u8(q0, m4b));
    const int8x16_t v0h = vqtbl1q_s8(values, vshrq_n_u8(q0, 4));
    const int8x16_t v1l = vqtbl1q_s8(values, vandq_u8(q1, m4b));
    const int8x16_t v1h = vqtbl1q_s8(values, vshrq_n_u8(q1, 4));
    const int32x4_t p0 = vdotq_s32(
        vdotq_s32(mzero, v0l, vld1q_s8(y[i].qs)), v0h, vld1q_s8(y[i].qs + 16));
    const int32x4_t p1 = vdotq_s32(
        vdotq_s32(mzero, v1l, vld1q_s8(y[i + 1].qs)),
        v1h,
        vld1q_s8(y[i + 1].qs + 16));
    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), read_f16(xb0) * y[i].d);
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), read_f16(xb1) * y[i + 1].d);
  }
  for (; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 18;
    const uint8x16_t q = vld1q_u8(xb + 2);
    const int8x16_t vl = vqtbl1q_s8(values, vandq_u8(q, m4b));
    const int8x16_t vh = vqtbl1q_s8(values, vshrq_n_u8(q, 4));
    const int32x4_t p = vdotq_s32(
        vdotq_s32(mzero, vl, vld1q_s8(y[i].qs)), vh, vld1q_s8(y[i].qs + 16));
    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), read_f16(xb) * y[i].d);
  }
  return vaddvq_f32(vaddq_f32(sumv0, sumv1));
}

// iq4_xs block: d f16 @0, scales_h u16 @2, scales_l[4] @4, qs[128] @8. Signed
// 6-bit per-32 scale (ls - 32), kvalues LUT, no min.
float vec_dot_iq4_xs(const uint8_t* w, const void* act, int nb) {
  const ActQ8K* y = static_cast<const ActQ8K*>(act);
  const int8x16_t values = vld1q_s8(kvalues_iq4nl);
  const uint8x16_t m4b = vdupq_n_u8(0xf);
  const int32x4_t mzero = vdupq_n_s32(0);
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 136;
    const float d = read_f16(xb) * y[i].d;
    uint16_t h;
    std::memcpy(&h, xb + 2, sizeof(uint16_t));
    const uint8_t* scales_l = xb + 4;
    const uint8_t* q4 = xb + 8;
    const int8_t* q8 = y[i].qs;
    int32_t sumi1 = 0, sumi2 = 0;
    for (int ib = 0; ib < 4; ++ib) {
      const uint8x16_t q4b0 = vld1q_u8(q4);
      const uint8x16_t q4b1 = vld1q_u8(q4 + 16);
      q4 += 32;
      const int8x16_t v0 = vqtbl1q_s8(values, vandq_u8(q4b0, m4b));
      const int8x16_t v1 = vqtbl1q_s8(values, vshrq_n_u8(q4b0, 4));
      const int8x16_t v2 = vqtbl1q_s8(values, vandq_u8(q4b1, m4b));
      const int8x16_t v3 = vqtbl1q_s8(values, vshrq_n_u8(q4b1, 4));
      const int32x4_t p1 =
          vdotq_s32(vdotq_s32(mzero, v0, vld1q_s8(q8)), v1, vld1q_s8(q8 + 16));
      const int32x4_t p2 = vdotq_s32(
          vdotq_s32(mzero, v2, vld1q_s8(q8 + 32)), v3, vld1q_s8(q8 + 48));
      q8 += 64;
      const int ls1 = ((scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
      const int ls2 = ((scales_l[ib] >> 4) | ((h << 2) & 0x30)) - 32;
      h >>= 4;
      sumi1 += vaddvq_s32(p1) * ls1;
      sumi2 += vaddvq_s32(p2) * ls2;
    }
    sumf += d * static_cast<float>(sumi1 + sumi2);
  }
  return sumf;
}

// iq3_xxs block: d f16 @0, qs[64] @2 (grid idx), gas[32] @66 (per-ib32 u32:
// signs in low 28 bits, scale in top 4). Global *0.5 applied post-sum.
float vec_dot_iq3_xxs(const uint8_t* w, const void* act, int nb) {
  const ActQ8K* y = static_cast<const ActQ8K*>(act);
  const uint64_t* signs64 = reinterpret_cast<const uint64_t*>(keven_signs_q2xs);
  const int32x4_t mzero = vdupq_n_s32(0);
  uint32_t aux32[2];
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 98;
    const float d = read_f16(xb) * y[i].d;
    const uint8_t* q3 = xb + 2;
    const uint8_t* gas = xb + 2 + 64;
    const int8_t* q8 = y[i].qs;
    float sumf1 = 0.0f, sumf2 = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
      const int8x16_t q8b0 = vld1q_s8(q8);
      const int8x16_t q8b1 = vld1q_s8(q8 + 16);
      const int8x16_t q8b2 = vld1q_s8(q8 + 32);
      const int8x16_t q8b3 = vld1q_s8(q8 + 48);
      q8 += 64;
      std::memcpy(aux32, gas, 8);
      gas += 8;
      const uint32x4_t g0 = grid4(iq3xxs_grid, q3[0], q3[1], q3[2], q3[3]);
      const uint32x4_t g1 = grid4(iq3xxs_grid, q3[4], q3[5], q3[6], q3[7]);
      const uint32x4_t g2 = grid4(iq3xxs_grid, q3[8], q3[9], q3[10], q3[11]);
      const uint32x4_t g3 = grid4(iq3xxs_grid, q3[12], q3[13], q3[14], q3[15]);
      q3 += 16;
      int8x16_t s0 = vcombine_s8(
          vld1_s8(reinterpret_cast<const int8_t*>(signs64 + (aux32[0] & 127))),
          vld1_s8(reinterpret_cast<const int8_t*>(
              signs64 + ((aux32[0] >> 7) & 127))));
      int8x16_t s1 = vcombine_s8(
          vld1_s8(reinterpret_cast<const int8_t*>(
              signs64 + ((aux32[0] >> 14) & 127))),
          vld1_s8(reinterpret_cast<const int8_t*>(
              signs64 + ((aux32[0] >> 21) & 127))));
      int8x16_t s2 = vcombine_s8(
          vld1_s8(reinterpret_cast<const int8_t*>(signs64 + (aux32[1] & 127))),
          vld1_s8(reinterpret_cast<const int8_t*>(
              signs64 + ((aux32[1] >> 7) & 127))));
      int8x16_t s3 = vcombine_s8(
          vld1_s8(reinterpret_cast<const int8_t*>(
              signs64 + ((aux32[1] >> 14) & 127))),
          vld1_s8(reinterpret_cast<const int8_t*>(
              signs64 + ((aux32[1] >> 21) & 127))));
      s0 = vmulq_s8(s0, vreinterpretq_s8_u32(g0));
      s1 = vmulq_s8(s1, vreinterpretq_s8_u32(g1));
      s2 = vmulq_s8(s2, vreinterpretq_s8_u32(g2));
      s3 = vmulq_s8(s3, vreinterpretq_s8_u32(g3));
      const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, s0, q8b0), s1, q8b1);
      const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, s2, q8b2), s3, q8b3);
      sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[0] >> 28));
      sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[1] >> 28));
    }
    sumf += d * (sumf1 + sumf2);
  }
  return 0.5f * sumf;
}

// iq3_s block: d f16 @0, qs[64] @2 (low 8 grid bits), qh[8] @66 (9th bit),
// signs[32] @74, scales[4] @106 (4-bit -> 2*s+1). 9-bit grid, no min.
float vec_dot_iq3_s(const uint8_t* w, const void* act, int nb) {
  const ActQ8K* y = static_cast<const ActQ8K*>(act);
  static const uint8_t k_mask1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                      1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                      2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
  static const uint8_t k_mask2[16] = {
      1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  static const int16_t k_shift[8] = {8, 7, 6, 5, 4, 3, 2, 1};
  const uint8x16_t mask1_0 = vld1q_u8(k_mask1);
  const uint8x16_t mask1_1 = vld1q_u8(k_mask1 + 16);
  const uint8x16_t mask2 = vld1q_u8(k_mask2);
  const int16x8_t hshift = vld1q_s16(k_shift);
  const uint16x8_t m256 = vdupq_n_u16(256);
  const uint8x16_t m1 = vdupq_n_u8(1);
  const int32x4_t mzero = vdupq_n_s32(0);
  uint32_t scales32[2];
  const uint8_t* scales8 = reinterpret_cast<const uint8_t*>(scales32);
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* xb = w + static_cast<std::size_t>(i) * 110;
    const float d = read_f16(xb) * y[i].d;
    const uint8_t* qs = xb + 2;
    const uint8_t* qh = xb + 66;
    const uint16_t* signs = reinterpret_cast<const uint16_t*>(xb + 74);
    const int8_t* q8 = y[i].qs;
    std::memcpy(scales32, xb + 106, 4);
    scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
    scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;
    int32_t sumi1 = 0, sumi2 = 0;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
      const int8x16_t q8b0 = vld1q_s8(q8);
      const int8x16_t q8b1 = vld1q_s8(q8 + 16);
      const int8x16_t q8b2 = vld1q_s8(q8 + 32);
      const int8x16_t q8b3 = vld1q_s8(q8 + 48);
      q8 += 64;
      const uint8x16_t idx_l = vld1q_u8(qs);
      qs += 16;
      uint16_t idx[8];
      vst1q_u16(
          idx,
          vorrq_u16(
              vmovl_u8(vget_low_u8(idx_l)),
              vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32]), hshift), m256)));
      const uint32x4_t g0 = grid4(iq3s_grid, idx[0], idx[1], idx[2], idx[3]);
      const uint32x4_t g1 = grid4(iq3s_grid, idx[4], idx[5], idx[6], idx[7]);
      vst1q_u16(
          idx,
          vorrq_u16(
              vmovl_u8(vget_high_u8(idx_l)),
              vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32 + 1]), hshift), m256)));
      const uint32x4_t g2 = grid4(iq3s_grid, idx[0], idx[1], idx[2], idx[3]);
      const uint32x4_t g3 = grid4(iq3s_grid, idx[4], idx[5], idx[6], idx[7]);

      uint8x16_t vs = vreinterpretq_u8_u32(
          vdupq_n_u32(signs[0] | (static_cast<uint32_t>(signs[1]) << 16)));
      uint8x16_t vsb = vandq_u8(vqtbl1q_u8(vs, mask1_1), mask2);
      uint8x16_t vsa = vandq_u8(vqtbl1q_u8(vs, mask1_0), mask2);
      vsa = vorrq_u8(vceqq_u8(vsa, mask2), m1);
      vsb = vorrq_u8(vceqq_u8(vsb, mask2), m1);
      const int8x16_t q3s0 =
          vmulq_s8(vreinterpretq_s8_u8(vsa), vreinterpretq_s8_u32(g0));
      const int8x16_t q3s1 =
          vmulq_s8(vreinterpretq_s8_u8(vsb), vreinterpretq_s8_u32(g1));

      vs = vreinterpretq_u8_u32(
          vdupq_n_u32(signs[2] | (static_cast<uint32_t>(signs[3]) << 16)));
      vsb = vandq_u8(vqtbl1q_u8(vs, mask1_1), mask2);
      vsa = vandq_u8(vqtbl1q_u8(vs, mask1_0), mask2);
      vsa = vorrq_u8(vceqq_u8(vsa, mask2), m1);
      vsb = vorrq_u8(vceqq_u8(vsb, mask2), m1);
      signs += 4;
      const int8x16_t q3s2 =
          vmulq_s8(vreinterpretq_s8_u8(vsa), vreinterpretq_s8_u32(g2));
      const int8x16_t q3s3 =
          vmulq_s8(vreinterpretq_s8_u8(vsb), vreinterpretq_s8_u32(g3));

      const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q3s0, q8b0), q3s1, q8b1);
      const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q3s2, q8b2), q3s3, q8b3);
      sumi1 += vaddvq_s32(p1) * scales8[ib32 / 2 + 0];
      sumi2 += vaddvq_s32(p2) * scales8[ib32 / 2 + 4];
    }
    sumf += d * static_cast<float>(sumi1 + sumi2);
  }
  return sumf;
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
constexpr KQNeonKernel kKernelQ2K = {
    &vec_dot_q2_k,
    sizeof(ActQ8K),
    &quantize_act_row_q8k};
constexpr KQNeonKernel kKernelQ3K = {
    &vec_dot_q3_k,
    sizeof(ActQ8K),
    &quantize_act_row_q8k};
constexpr KQNeonKernel kKernelQ40 = {
    &vec_dot_q4_0,
    sizeof(ActQ80),
    &quantize_act_row_q80};
constexpr KQNeonKernel kKernelQ41 = {
    &vec_dot_q4_1,
    sizeof(ActQ81),
    &quantize_act_row_q81};
constexpr KQNeonKernel kKernelQ50 = {
    &vec_dot_q5_0,
    sizeof(ActQ80),
    &quantize_act_row_q80};
constexpr KQNeonKernel kKernelQ51 = {
    &vec_dot_q5_1,
    sizeof(ActQ81),
    &quantize_act_row_q81};
constexpr KQNeonKernel kKernelIQ4NL = {
    &vec_dot_iq4_nl,
    sizeof(ActQ80),
    &quantize_act_row_q80};
constexpr KQNeonKernel kKernelIQ4XS = {
    &vec_dot_iq4_xs,
    sizeof(ActQ8K),
    &quantize_act_row_q8k};
constexpr KQNeonKernel kKernelIQ3S = {
    &vec_dot_iq3_s,
    sizeof(ActQ8K),
    &quantize_act_row_q8k};
constexpr KQNeonKernel kKernelIQ3XXS = {
    &vec_dot_iq3_xxs,
    sizeof(ActQ8K),
    &quantize_act_row_q8k};

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
  } else if (codec == "q2_k") {
    return &kKernelQ2K;
  } else if (codec == "q3_k") {
    return &kKernelQ3K;
  } else if (codec == "q4_0") {
    return &kKernelQ40;
  } else if (codec == "q4_1") {
    return &kKernelQ41;
  } else if (codec == "q5_0") {
    return &kKernelQ50;
  } else if (codec == "q5_1") {
    return &kKernelQ51;
  } else if (codec == "iq4_nl") {
    return &kKernelIQ4NL;
  } else if (codec == "iq4_xs") {
    return &kKernelIQ4XS;
  } else if (codec == "iq3_s") {
    return &kKernelIQ3S;
  } else if (codec == "iq3_xxs") {
    return &kKernelIQ3XXS;
  }
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
