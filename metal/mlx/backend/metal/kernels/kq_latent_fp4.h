// FP4 latent rows at rest: groups of 16 values, each an E2M1 code (one
// nibble, low nibble first) times one E4M3 scale byte per group. Rows on
// the grid (ds4's latent QAT: scale = e4m3(amax / 6), codes = e2m1(v /
// scale)) pack exactly; rows off it project the same way the QAT does.
// A row of D values packs to D / 2 code bytes and D / 16 scale bytes.
#pragma once

#include <metal_stdlib>

constant constexpr float kq_fp4_e2m1_val[16] = {
    0.0f,
    0.5f,
    1.0f,
    1.5f,
    2.0f,
    3.0f,
    4.0f,
    6.0f,
    -0.0f,
    -0.5f,
    -1.0f,
    -1.5f,
    -2.0f,
    -3.0f,
    -4.0f,
    -6.0f};

METAL_FUNC float kq_fp4_e4m3_decode(uint b) {
  const uint e = (b >> 3) & 15u;
  const uint m = b & 7u;
  return e == 0 ? float(m) * 0.001953125f
                : as_type<float>(((e + 120u) << 23) | (m << 20));
}

// Code of s when s is an E4M3 value in [2^-9, 448]; 256 otherwise.
METAL_FUNC uint kq_fp4_e4m3_encode(float s) {
  if (!(s >= 0.001953125f && s <= 448.0f)) {
    return 256u;
  }
  int e;
  const float f = metal::frexp(s, e); // s = f * 2^e, f in [0.5, 1)
  uint code;
  if (e + 6 >= 1) {
    code = (uint(e + 6) << 3) | uint((f * 2.0f - 1.0f) * 8.0f);
  } else {
    code = uint(s * 512.0f);
  }
  return kq_fp4_e4m3_decode(code) == s ? code : 256u;
}

// Nearest E4M3 value of s > 0 (ties to even), clamped to 448.
METAL_FUNC float kq_fp4_e4m3_round(float s) {
  int e;
  metal::frexp(metal::max(s, 0.001953125f), e);
  const int E = metal::clamp(e - 1, -6, 8);
  const float q = metal::ldexp(1.0f, E - 3);
  return metal::min(metal::rint(s / q) * q, 448.0f);
}

// Nearest E2M1 magnitude (ties to the even value index).
METAL_FUNC float kq_fp4_e2m1_snap(float a) {
  return a <= 0.25f ? 0.0f
      : a < 0.75f   ? 0.5f
      : a <= 1.25f  ? 1.0f
      : a < 1.75f   ? 1.5f
      : a <= 2.5f   ? 2.0f
      : a < 3.5f    ? 3.0f
      : a <= 5.0f   ? 4.0f
                    : 6.0f;
}

// Code m with e2m1[m] * s == a exactly; 16 when there is none.
METAL_FUNC uint kq_fp4_e2m1_code_exact(float a, float s) {
  for (uint m = 0; m < 8; ++m) {
    if (kq_fp4_e2m1_val[m] * s == a) {
      return m;
    }
  }
  return 16u;
}

// 8 values of one code word (nibbles, low first) at one scale.
template <typename T>
METAL_FUNC metal::vec<T, 8> kq_fp4_chunk8(uint codes, float scale) {
  metal::vec<T, 8> v;
  for (int k = 0; k < 8; ++k) {
    v[k] = T(kq_fp4_e2m1_val[(codes >> (4 * k)) & 15u] * scale);
  }
  return v;
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_latent_fp4_pack(
    const device T* X [[buffer(0)]],
    device uint* CODES [[buffer(1)]], // 2 words per group
    device uchar* SCALES [[buffer(2)]],
    const constant int& n_groups [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n_groups) {
    return;
  }
  const device metal::vec<T, 8>* xr =
      reinterpret_cast<const device metal::vec<T, 8>*>(X + size_t(gid) * 16);
  const metal::vec<T, 8> x0 = xr[0];
  const metal::vec<T, 8> x1 = xr[1];
  float v[16];
  float amax = 0.0f;
  for (int k = 0; k < 8; ++k) {
    v[k] = float(x0[k]);
    v[8 + k] = float(x1[k]);
  }
  for (int k = 0; k < 16; ++k) {
    amax = metal::max(amax, metal::fabs(v[k]));
  }

  uint scode = 256u;
  uint c[16];
  if (amax == 0.0f) {
    scode = 1u; // the QAT's floor scale, 2^-9
    for (int k = 0; k < 16; ++k) {
      c[k] = metal::signbit(v[k]) ? 8u : 0u;
    }
  } else {
    // The largest magnitude is top * s for one of the E2M1 magnitudes
    // top; try each, the E4M3 codes around amax / top, and keep the first
    // scale under which every value sits on the grid.
    for (int t = 7; t >= 1 && scode == 256u; --t) {
      const float top = kq_fp4_e2m1_val[t];
      const uint base = kq_fp4_e4m3_encode(kq_fp4_e4m3_round(amax / top));
      for (int dc = -1; dc <= 1 && scode == 256u; ++dc) {
        const int code = int(base) + dc;
        if (base == 256u || code < 1 || code > 126) {
          continue;
        }
        const float s = kq_fp4_e4m3_decode(uint(code));
        if (top * s != amax) {
          continue;
        }
        bool ok = true;
        for (int k = 0; k < 16; ++k) {
          const uint m = kq_fp4_e2m1_code_exact(metal::fabs(v[k]), s);
          ok = ok && m < 16u;
          c[k] = (m & 7u) | (metal::signbit(v[k]) ? 8u : 0u);
        }
        if (ok) {
          scode = uint(code);
        }
      }
    }
  }
  if (scode == 256u) {
    // Off the grid: the QAT projection.
    const float s = kq_fp4_e4m3_round(metal::max(amax, 0.01171875f) / 6.0f);
    scode = kq_fp4_e4m3_encode(s);
    for (int k = 0; k < 16; ++k) {
      const float q = metal::clamp(v[k] / s, -6.0f, 6.0f);
      const uint m =
          kq_fp4_e2m1_code_exact(kq_fp4_e2m1_snap(metal::fabs(q)), 1.0f);
      c[k] = m | (q < 0.0f ? 8u : 0u);
    }
  }
  uint w0 = 0u;
  uint w1 = 0u;
  for (int k = 0; k < 8; ++k) {
    w0 |= c[k] << (4 * k);
    w1 |= c[8 + k] << (4 * k);
  }
  CODES[size_t(gid) * 2] = w0;
  CODES[size_t(gid) * 2 + 1] = w1;
  SCALES[gid] = uchar(scode);
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_latent_fp4_unpack(
    const device uint* CODES [[buffer(0)]],
    const device uchar* SCALES [[buffer(1)]],
    device T* OUT [[buffer(2)]],
    const constant int& n_groups [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n_groups) {
    return;
  }
  const float s = kq_fp4_e4m3_decode(SCALES[gid]);
  device metal::vec<T, 8>* o =
      reinterpret_cast<device metal::vec<T, 8>*>(OUT + size_t(gid) * 16);
  o[0] = kq_fp4_chunk8<T>(CODES[size_t(gid) * 2], s);
  o[1] = kq_fp4_chunk8<T>(CODES[size_t(gid) * 2 + 1], s);
}
