// DeepSeek-V4-Flash indexer activation QAT round-trip, fused into one
// kernel: the 128-wide Hadamard transform followed by the per-32-block
// FP4-E2M1 round-trip (scale = 2^ceil(log2(amax/6)) with an FLT_MIN*6 amax
// floor, clamp to +-6, threshold-ladder nearest-even rounding, rescale).
// Replaces the ~5-pass mx.hadamard_transform + compiled-core chain on the
// indexer query path (ds4.c dsv4_indexer_qat_row).
//
// Bit-compatibility with the mx graph it replaces is deliberate: the
// butterfly structure and write-time scale replicate mlx's hadamard_n for
// n=128 exactly (radix-16 stage then final radix-8, fp32 threadgroup
// buffer), log2 uses metal::precise::log2 (mlx's Log2 op), and
// ldexp(1, e) equals the exact power-of-two table for every reachable e
// (amax floor pins e >= -126; fp32 range pins e <= 127).

#pragma once

#include "mlx/backend/metal/kernels/steel/defines.h"

// mlx hadamard.h radix_func, verbatim structure (thread-local 2^R FWHT).
template <short R>
METAL_FUNC void kq_dsa_qat_radix(thread float* x) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < logR; s++) {
    STEEL_PRAGMA_UNROLL
    for (short b = 0; b < R / 2; b++) {
      short k = b & (h - 1);
      short j = ((b - k) << 1) + k;
      float a = x[j];
      float c = x[j + h];
      x[j] = a + c;
      x[j + h] = a - c;
    }
    h <<= 1;
  }
}

// E2M1 nearest snap (threshold ladder, ties to the even value index).
METAL_FUNC float kq_dsa_e2m1_snap(float a) {
  return a <= 0.25f ? 0.0f
      : a < 0.75f   ? 0.5f
      : a <= 1.25f  ? 1.0f
      : a < 1.75f   ? 1.5f
      : a <= 2.5f   ? 2.0f
      : a < 3.5f    ? 3.0f
      : a <= 5.0f   ? 4.0f
                    : 6.0f;
}

// Shared indexer-QAT core: fp32 Hadamard (mlx hadamard_n structure for
// n=128) + per-32-block FP4 power-of-two scale. Leaves the scaled
// transform in buf[lr] and block scales in qs[lr]; the two kernels below
// differ only in what they store.
template <typename T>
METAL_FUNC void kq_dsa_qat_row_core(
    const device T* x_row,
    bool active,
    short lr,
    short i,
    float scale,
    threadgroup float (&buf)[32][128],
    threadgroup float (&part)[32][8],
    threadgroup float (&qs)[32][4]) {
  constexpr short N = 128;
  constexpr short RAD = 16; // max radix (mlx hadamard_n, n=128)
  constexpr short FINAL = 8; // final radix (logN=7, logR=4 -> 2^3)

  if (active) {
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < RAD; r++) {
      buf[lr][i * RAD + r] = float(x_row[i * RAD + r]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float x[RAD];

  // Radix-16 stage (h = 1): worker i owns the contiguous [16i, 16i+16).
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < RAD; r++) {
    x[r] = buf[lr][i * RAD + r];
  }
  kq_dsa_qat_radix<RAD>(x);
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < RAD; r++) {
    buf[lr][i * RAD + r] = x[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final radix-8 stage (h = 16): butterfly idx covers {idx + 16r}; the
  // 16 butterflies partition the row, so the two per worker need no
  // barrier between them (same invariant as mlx's final-radix loop).
  STEEL_PRAGMA_UNROLL
  for (short t = 0; t < 2; t++) {
    const short idx = i + t * (N / RAD);
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < FINAL; r++) {
      x[r] = buf[lr][idx + RAD * r];
    }
    kq_dsa_qat_radix<FINAL>(x);
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < FINAL; r++) {
      buf[lr][idx + RAD * r] = x[r];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Hadamard scale (mlx applies it at write time) + per-worker |max| part.
  float local_max = 0.0f;
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < RAD; r++) {
    const float v = buf[lr][i * RAD + r] * scale;
    buf[lr][i * RAD + r] = v;
    local_max = metal::max(local_max, metal::fabs(v));
  }
  part[lr][i] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Per-32-block FP4 scale: block b spans workers 2b and 2b+1.
  if (i < N / 32) {
    float amax = metal::max(part[lr][2 * i], part[lr][2 * i + 1]);
    amax = metal::max(amax, 7.052966104933725e-38f); // FLT_MIN * 6
    const float e = metal::ceil(metal::precise::log2(amax / 6.0f));
    qs[lr][i] = metal::ldexp(1.0f, int(e));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_dsa_indexer_qat(
    const device T* X [[buffer(0)]],
    device T* O [[buffer(1)]],
    const constant int& n_rows [[buffer(2)]],
    const constant float& scale [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  constexpr short N = 128;
  constexpr short ROWS = 32; // rows per threadgroup
  constexpr short RAD = 16;

  threadgroup float buf[ROWS][N];
  threadgroup float part[ROWS][N / RAD];
  threadgroup float qs[ROWS][N / 32];

  const short lr = short(lane >> 3); // local row
  const short i = short(lane & 7); // worker within the row
  const int row = int(tgid.x) * ROWS + lr;
  const bool active = row < n_rows;

  kq_dsa_qat_row_core(X + size_t(row) * N, active, lr, i, scale, buf, part, qs);

  if (active) {
    device T* o_row = O + size_t(row) * N;
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < RAD; r++) {
      const short e = i * RAD + r;
      const float q = qs[lr][e >> 5];
      const float c = metal::clamp(buf[lr][e] / q, -6.0f, 6.0f);
      const float sgn = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
      // _e2m1_round's threshold ladder: nearest E2M1 value, ties to the
      // even value index (thresholds bake the tie direction in).
      const float qv = kq_dsa_e2m1_snap(metal::fabs(c));
      o_row[e] = T(sgn * qv * q);
    }
  }
}

// Emit variant: same core, but instead of the dequantized round-trip it
// stores the quantized wire form the i8mx score kernel consumes: int8
// codes = E2M1 values doubled (in [-12, 12]) and per-32-block fp32 scales
// pre-folded as scale * 0.5, so code * scale_half reproduces the
// kq_dsa_indexer_qat output bit-exactly except that negatives snapped to
// zero re-dequantize as +0.0 where the round-trip stores -0.0 (int8 has
// no signed zero; value-equal, and score kernels are unaffected).
template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]] void
kq_dsa_indexer_qat_quant(
    const device T* X [[buffer(0)]],
    device int8_t* CODES [[buffer(1)]],
    device float* SCALES [[buffer(2)]],
    const constant int& n_rows [[buffer(3)]],
    const constant float& scale [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  constexpr short N = 128;
  constexpr short ROWS = 32;
  constexpr short RAD = 16;

  threadgroup float buf[ROWS][N];
  threadgroup float part[ROWS][N / RAD];
  threadgroup float qs[ROWS][N / 32];

  const short lr = short(lane >> 3);
  const short i = short(lane & 7);
  const int row = int(tgid.x) * ROWS + lr;
  const bool active = row < n_rows;

  kq_dsa_qat_row_core(X + size_t(row) * N, active, lr, i, scale, buf, part, qs);

  if (active) {
    device int8_t* c_row = CODES + size_t(row) * N;
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < RAD; r++) {
      const short e = i * RAD + r;
      const float q = qs[lr][e >> 5];
      const float c = metal::clamp(buf[lr][e] / q, -6.0f, 6.0f);
      const float sgn = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
      const float qv = kq_dsa_e2m1_snap(metal::fabs(c));
      c_row[e] = int8_t(sgn * qv * 2.0f);
    }
    if (i < N / 32) {
      SCALES[size_t(row) * (N / 32) + i] = qs[lr][i] * 0.5f;
    }
  }
}

// Main-attention KV row QAT round-trip fused into one kernel (ds4.c
// dsv4_fp8_kv_quantize_row + f16_round): per-64-block FP8-E4M3FN
// round-trip on the leading D - NROT dims (scale = 2^ceil(log2(amax/448))
// with a 1e-4 amax floor, clamp to +-448, nearest-even mantissa rounding
// with the 2^-9 subnormal step floor, rescale), the trailing NROT RoPE
// dims untouched by fp8, then every output re-rounded through fp16 (the
// f16 KV-cache step). Bit-compatible with the split + _fp8_block_core +
// concat + astype graph it replaces: the fp8 result rounds through T
// before the fp16 round (the graph's per-slice astype), log2 is
// metal::precise::log2 (mlx's Log2 op), and ldexp(1, e) equals the exact
// power-of-two table for every reachable e. One threadgroup of 256
// threads per row: one simdgroup per 64-block (lane owns elements
// 64b + lane and 64b + 32 + lane), lanes re-derive the block scale from
// the simd_max broadcast.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_dsa_kv_qat(
    const device T* X [[buffer(0)]],
    device T* O [[buffer(1)]],
    const constant int& D [[buffer(2)]],
    const constant int& NROT [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int NSG = 8; // 256 / 32
  const device T* x_row = X + size_t(tgid) * D;
  device T* o_row = O + size_t(tgid) * D;

  const int nblk = (D - NROT) / 64;
  for (int b = int(simd_gid); b < nblk; b += NSG) {
    float v[2];
    float local = 0.0f;
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 2; j++) {
      v[j] = float(x_row[64 * b + 32 * j + int(simd_lid)]);
      local = metal::max(local, metal::fabs(v[j]));
    }
    const float amax = metal::max(simd_max(local), 1e-4f);
    const float scale = metal::ldexp(
        1.0f, int(metal::ceil(metal::precise::log2(amax / 448.0f))));
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 2; j++) {
      const float c = metal::clamp(v[j] / scale, -448.0f, 448.0f);
      const float a = metal::fabs(c);
      const float sgn = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
      // _e4m3_round: e = clip(floor(log2(max(a, 2^-9))), -6, 8), step
      // q = 2^(e-3), rint ties to even mantissa.
      float e = metal::floor(metal::precise::log2(metal::max(a, 0.001953125f)));
      e = metal::clamp(e, -6.0f, 8.0f);
      const float q = metal::ldexp(1.0f, int(e) - 3);
      const float r = sgn * metal::rint(a / q) * q * scale;
      o_row[64 * b + 32 * j + int(simd_lid)] = T(half(float(T(r))));
    }
  }
  for (int i = int(lid); i < NROT; i += 256) {
    const int e = (D - NROT) + i;
    o_row[e] = T(half(float(x_row[e])));
  }
}
