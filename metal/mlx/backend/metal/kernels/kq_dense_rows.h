#pragma once

#include "mlx/backend/metal/kernels/kq_verify_nax.h"

// Dense row-class kernels: quantized_matmul at 1 to 16 rows of x on NAX
// GPUs, reading the GGUF wire blocks in place.
//
// A threadgroup owns 32 weight rows. Each of its nsg simdgroups decodes
// every weight of those rows in its share of K once, straight into the
// right operand of a matmul2d 16x32x16, and one NAX op applies the decoded
// weights to all the rows of x: lane (fm, q) loads activation row fm into
// left elements 0..3 and row fm + 8 into 4..7, and holds destination rows
// fm and fm + 8. Simdgroup g takes units g, g + nsg, ... of K, and the
// threadgroup sums the simdgroups' tiles in simdgroup order through
// threadgroup memory, so a call is one dispatch with no partials in device
// memory.
//
// Rows past M are zero in the left operand, and the NAX op sums each row
// of the destination alone, so a row's bits do not depend on M. The row
// class (function constant 41, rows 9 to 16) only adds the loads and
// stores of rows 8..15, and nsg (function constant 43) depends on N, K and
// the codec only.
//
// Two decode forms: the codecs of kq_verify_nax.h decode a lane's words of
// a unit straight into half pairs (KqXxxNax), and every other codec
// dequantizes 16 weights at a time with its mat-vec chunk decode
// (KqXxxExt::deq_chunk16, natural order) into half registers.

constant bool kq_rows16_fc [[function_constant(41)]];
constant bool kq_rows16 =
    is_function_constant_defined(kq_rows16_fc) && kq_rows16_fc;
constant int kq_rows_nsg_fc [[function_constant(43)]];
constant int kq_rows_nsg =
    is_function_constant_defined(kq_rows_nsg_fc) ? kq_rows_nsg_fc : 4;
// The weights of a dispatch (function constant 44): 2 to 4 run the multi
// form, where threadgroup t serves the weight whose tg_end is the first
// past t. Function constant 43 is then the nsg all the weights share, or 0
// where each weight takes its own from nsgs: a threadgroup launches the
// largest, and the simdgroups past a weight's own nsg hold no units.
constant int kq_rows_multi_fc [[function_constant(44)]];
constant bool kq_rows_multi =
    is_function_constant_defined(kq_rows_multi_fc) && kq_rows_multi_fc > 1;

// Left elements 0..3 of step s for a codec of kq_verify_nax.h, from the
// unit at kb of the lane's row (xstep's order). Under kq_vnax_x16 the even
// step loads steps s and s + 1 in one 16-byte load into xu.
template <typename T, typename Codec>
METAL_FUNC vec<T, 4> kq_rows_xstep(
    const device T* xr,
    bool live,
    int kb,
    short q,
    short s,
    thread uint4& xu) {
  if (Codec::x_ahead || !kq_vnax_x16) {
    return live ? Codec::template xstep<T>(xr + kb, q, s) : vec<T, 4>(0);
  }
  if ((s & 1) == 0) {
    xu = live ? *(const device uint4*)(xr + kb + Codec::xoff(q, s)) : uint4(0);
  }
  const vec<T, 2> a = as_type<vec<T, 2>>((s & 1) ? xu.z : xu.x);
  const vec<T, 2> c = as_type<vec<T, 2>>((s & 1) ? xu.w : xu.y);
  return vec<T, 4>(a[0], c[0], a[1], c[1]);
}

// Left elements 0..3 of step s in natural order: activations off .. off + 3
// of the row. Under kq_vnax_x16 the even step loads steps s and s + 1 (off
// a multiple of 8) in one 16-byte load into xu.
template <typename T>
METAL_FUNC vec<T, 4> kq_rows_xnat(
    const device T* xr,
    bool live,
    int off,
    short s,
    thread uint4& xu) {
  if (!kq_vnax_x16) {
    return live ? *(const device vec<T, 4>*)(xr + off) : vec<T, 4>(0);
  }
  if ((s & 1) == 0) {
    xu = live ? *(const device uint4*)(xr + off) : uint4(0);
  }
  return (s & 1) ? as_type<vec<T, 4>>(xu.zw) : as_type<vec<T, 4>>(xu.xy);
}

// Sums the nsg simdgroups' tiles in simdgroup order and stores rows below
// M. A simdgroup at or past nsg (the multi form) holds a zero tile, meets
// the barriers and stores nothing. Lane (fm, fn) holds rows fm + 8h, columns
// 16f + fn + j, at element 8f + 4h + j. red holds nsg tiles of 8 rows;
// rows 8..15 go through it after rows 0..7, in the same order.
template <typename U, bool kOutScale, typename CC>
METAL_FUNC void kq_rows_store(
    thread CC& cc,
    threadgroup float* red,
    device U* y,
    const int N,
    const int M,
    const int n0,
    short fm,
    short fn,
    uint simd_gid,
    uint simd_lid,
    const int nsg,
    float out_mul,
    const device float* out_scale) {
  const short rh = kq_rows16 ? 2 : 1;
  const bool live = int(simd_gid) < nsg;
  for (short h = 0; h < rh; ++h) {
    if (nsg == 1) {
      const int m = fm + 8 * h;
      if (m >= M || !live) {
        continue;
      }
      for (short f = 0; f < 2; ++f) {
        for (short j = 0; j < 4; ++j) {
          const int n = n0 + 16 * f + fn + j;
          if (n < N) {
            float v = cc[f * 8 + 4 * h + j] * out_mul;
            if constexpr (kOutScale) {
              if (kq_has_out_scale) {
                v *= kq_out_scale_col(out_scale, n);
              }
            }
            y[static_cast<int64_t>(m) * N + n] = static_cast<U>(v);
          }
        }
      }
      continue;
    }
    if (h > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (short f = 0; f < 2; ++f) {
      for (short j = 0; j < 4; ++j) {
        red[simd_gid * 256 + fm * 32 + 16 * f + fn + j] = cc[f * 8 + 4 * h + j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = live ? simd_gid * 32 + simd_lid : 256; e < 256;
         e += 32 * nsg) {
      const int m = 8 * h + e / 32;
      const int n = n0 + (e % 32);
      if (m >= M || n >= N) {
        continue;
      }
      float v = 0.0f;
      for (short g = 0; g < nsg; ++g) {
        v += red[g * 256 + e];
      }
      v *= out_mul;
      if constexpr (kOutScale) {
        if (kq_has_out_scale) {
          v *= kq_out_scale_col(out_scale, n);
        }
      }
      y[static_cast<int64_t>(m) * N + n] = static_cast<U>(v);
    }
  }
}

// The lane geometry, the NAX operands and the lane's rows of x and of the
// weight, shared by both decode forms.
#define KQ_ROWS_NAX_SETUP                                                  \
  namespace to = mpp::tensor_ops;                                          \
  const int n0 = tid.x * 32;                                               \
  const short qid = simd_lid >> 2;                                         \
  const short fm = (qid & 4) | ((simd_lid >> 1) & 3);                      \
  const short fn = ((qid & 2) | (simd_lid & 1)) * 4;                       \
  const short q = fn >> 2;                                                 \
  constexpr auto desc = to::matmul2d_descriptor(                           \
      16,                                                                  \
      32,                                                                  \
      16,                                                                  \
      false,                                                               \
      true,                                                                \
      true,                                                                \
      to::matmul2d_descriptor::mode::multiply_accumulate);                 \
  to::matmul2d<desc, execution_simdgroup> op;                              \
  auto ca = op.template get_left_input_cooperative_tensor<T, W, float>();  \
  auto cb = op.template get_right_input_cooperative_tensor<T, W, float>(); \
  auto cc = op.template get_destination_cooperative_tensor<                \
      decltype(ca),                                                        \
      decltype(cb),                                                        \
      float>();                                                            \
  for (short i = 0; i < 16; ++i) {                                         \
    cc[i] = 0.0f;                                                          \
  }                                                                        \
  for (short i = 0; i < 8; ++i) {                                          \
    ca[i] = T(0);                                                          \
  }                                                                        \
  const bool xlo = fm < M;                                                 \
  const bool xhi = kq_rows16 && fm + 8 < M;                                \
  const device T* xr0 = x + static_cast<int64_t>(xlo ? fm : 0) * K;        \
  const device T* xr1 = x + static_cast<int64_t>(xhi ? fm + 8 : 0) * K;    \
  /* Row r of the lane is n0 + fm + 8r, clamped into the last row tile. */ \
  const int nl = min(n0 + fm, N - 1);                                      \
  const device uint8_t* wr = w + static_cast<int64_t>(nl) * row_bytes;     \
  auto row = [&](short r) {                                                \
    return wr + static_cast<int64_t>(min(8 * r, N - 1 - nl)) * row_bytes;  \
  };

// Codecs of kq_verify_nax.h: a unit's words per row, decoded per step.
template <typename T, typename Codec, bool kOutScale = false>
METAL_FUNC void kq_rows_nax_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const int K,
    const int N,
    const int M,
    const int nsg,
    threadgroup float* red,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    const device float* out_scale = nullptr) {
  typedef half W;
  constexpr int BK = Codec::block_k;
  constexpr int UB = Codec::ub;
  constexpr int SPB = BK / 16;
  constexpr int KS = UB * BK;
  static_assert(
      Codec::x_ahead || (SPB % 2 == 0 && kq_vnax_x16_pairs<Codec, SPB>()),
      "x16 loads step pairs");
  const int row_bytes = (K / BK) * Codec::block_bytes;
  KQ_ROWS_NAX_SETUP

  const int units = K / KS;
  // A simdgroup at or past nsg (the multi form) holds no units.
  for (int ui = int(simd_gid) < nsg ? int(simd_gid) : units; ui < units;
       ui += nsg) {
    const int kb = ui * KS;
    typename Codec::Words wv[UB][4];
#pragma unroll
    for (short u = 0; u < UB; ++u) {
#pragma unroll
      for (short r = 0; r < 4; ++r) {
        wv[u][r] = Codec::load(row(r), ui * UB + u, q);
      }
    }
    uint4 xu0 = uint4(0);
    uint4 xu1 = uint4(0);
#pragma unroll
    for (short u = 0; u < UB; ++u) {
#pragma unroll
      for (short s = 0; s < SPB; ++s) {
        const vec<T, 4> a0 =
            kq_rows_xstep<T, Codec>(xr0, xlo, kb + u * BK, q, s, xu0);
        ca[0] = a0[0];
        ca[1] = a0[1];
        ca[2] = a0[2];
        ca[3] = a0[3];
        if (kq_rows16) {
          const vec<T, 4> a1 =
              kq_rows_xstep<T, Codec>(xr1, xhi, kb + u * BK, q, s, xu1);
          ca[4] = a1[0];
          ca[5] = a1[1];
          ca[6] = a1[2];
          ca[7] = a1[3];
        }
#pragma unroll
        for (short r = 0; r < 4; ++r) {
#pragma unroll
          for (short jp = 0; jp < 2; ++jp) {
            const half2 v = Codec::pair(wv[u][r], 2 * s + jp);
            cb[4 * r + 2 * jp] = v.x;
            cb[4 * r + 2 * jp + 1] = v.y;
          }
        }
        op.run(ca, cb, cc);
      }
    }
  }
  kq_rows_store<T, kOutScale>(
      cc,
      red,
      y,
      N,
      M,
      n0,
      fm,
      fn,
      simd_gid,
      simd_lid,
      nsg,
      KqVnaxOutMul<Codec>::value,
      out_scale);
}

// Every other codec: a unit is UB blocks, UB * BK weights (UB 0 takes
// the fewest blocks that reach 128 weights). Lane quad q owns weights
// [q * LW, (q + 1) * LW) of the unit, LW = UB * BK / 4, and step s their
// 4s..4s + 3. Each 16-weight chunk (half a chunk where LW is 8) is
// dequantized once per row into half registers when the lane reaches it.
template <typename Ext, int UB>
constexpr int kq_rows_gen_ub() {
  return UB > 0 ? UB : (Ext::superblock >= 128 ? 1 : 128 / Ext::superblock);
}

template <bool B, typename X, typename Y>
struct kq_rows_pick {
  typedef X type;
};
template <typename X, typename Y>
struct kq_rows_pick<false, X, Y> {
  typedef Y type;
};

// kWide holds the weights in x's type instead of half: a codec whose block
// scale is a power of two of any exponent (mxfp4) keeps every weight exact
// in bfloat16, where half would overflow or flush it.
template <
    typename T,
    typename Ext,
    bool kOutScale = false,
    int UB_ = 0,
    bool kWide = false>
METAL_FUNC void kq_rows_gen_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const int K,
    const int N,
    const int M,
    const int nsg,
    threadgroup float* red,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    const device float* out_scale = nullptr) {
  typedef typename kq_rows_pick<kWide, T, half>::type W;
  constexpr int BK = Ext::superblock;
  constexpr int UB = kq_rows_gen_ub<Ext, UB_>();
  constexpr int UK = UB * BK;
  constexpr int SPB = UK / 16;
  constexpr int LW = UK / 4;
  static_assert(LW % 8 == 0, "a lane's weights are whole 16-byte x loads");
  static_assert(LW < 16 || BK % 16 == 0, "chunks stay inside one block");
  const int row_bytes = (K / BK) * Ext::block_bytes;
  KQ_ROWS_NAX_SETUP

  const int units = K / UK;
  for (int ui = int(simd_gid) < nsg ? int(simd_gid) : units; ui < units;
       ui += nsg) {
    const int kb = ui * UK;
    vec<W, 4> wh[4][4];
    uint4 xu0 = uint4(0);
    uint4 xu1 = uint4(0);
#pragma unroll
    for (short s = 0; s < SPB; ++s) {
      if (LW >= 16 ? (s % 4 == 0) : (s == 0)) {
        const short o = q * LW + 4 * s;
        const short blk = LW >= 16 ? o / BK : 0;
        const short chunk = LW >= 16 ? (o % BK) / 16 : q >> 1;
        const int64_t boff =
            static_cast<int64_t>(ui * UB + blk) * Ext::block_bytes;
#pragma unroll
        for (short r = 0; r < 4; ++r) {
          float4x4 reg;
          Ext::deq_chunk16(row(r) + boff, chunk, reg);
#pragma unroll
          for (short i = 0; i < 4; ++i) {
            wh[r][i] = vec<W, 4>(reg[i]);
          }
        }
      }
      const int off = kb + q * LW + 4 * s;
      const vec<T, 4> a0 = kq_rows_xnat<T>(xr0, xlo, off, s, xu0);
      ca[0] = a0[0];
      ca[1] = a0[1];
      ca[2] = a0[2];
      ca[3] = a0[3];
      if (kq_rows16) {
        const vec<T, 4> a1 = kq_rows_xnat<T>(xr1, xhi, off, s, xu1);
        ca[4] = a1[0];
        ca[5] = a1[1];
        ca[6] = a1[2];
        ca[7] = a1[3];
      }
#pragma unroll
      for (short r = 0; r < 4; ++r) {
        const vec<W, 4> v =
            LW >= 16 ? wh[r][s % 4] : ((q & 1) ? wh[r][2 + s] : wh[r][s]);
        cb[4 * r] = v[0];
        cb[4 * r + 1] = v[1];
        cb[4 * r + 2] = v[2];
        cb[4 * r + 3] = v[3];
      }
      op.run(ca, cb, cc);
    }
  }
  kq_rows_store<T, kOutScale>(
      cc, red, y, N, M, n0, fm, fn, simd_gid, simd_lid, nsg, 1.0f, out_scale);
}

// One block per unit, for a K that is whole blocks but not whole units of
// the form above (codecs under 128 weights a block).
template <typename T, typename Ext, bool kOutScale = false, bool kWide = false>
METAL_FUNC void kq_rows_gen1_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const int K,
    const int N,
    const int M,
    const int nsg,
    threadgroup float* red,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    const device float* out_scale = nullptr) {
  kq_rows_gen_impl<T, Ext, kOutScale, 1, kWide>(
      w, x, y, K, N, M, nsg, red, tid, simd_gid, simd_lid, out_scale);
}

// Entry points: grid (ceil(N / 32), 1, 1), 32 * nsg threads, nsg * 1 KiB
// of threadgroup memory. OS 1 gives the kernel the output scale. The
// multi form (kq_rows_multi) binds up to four weights and outputs that
// share x: weight i takes ceil(N_i / 32) threadgroups after those of the
// weights before it, with its own N and nsg, so each output holds the
// bits of its single call.
#define KQ_ROWS_MULTI_BUF(TYPE, NAME, SLOT) \
  TYPE NAME [[buffer(SLOT), function_constant(kq_rows_multi)]]

#define KQ_DEFINE_ROWS_KERNEL_ARGS(NAME, IMPL, OS, ...)                        \
  template <typename T>                                                        \
  [[kernel]] void kq_##NAME##_rows(                                            \
      const device uint8_t* w0,                                                \
      const device uint8_t* /* scales */,                                      \
      const device T* x,                                                       \
      device T* y0,                                                            \
      const constant int& K,                                                   \
      const constant int& N0,                                                  \
      const constant int& M,                                                   \
      KQ_ROWS_MULTI_BUF(const device uint8_t*, w1, 7),                         \
      KQ_ROWS_MULTI_BUF(const device uint8_t*, w2, 8),                         \
      KQ_ROWS_MULTI_BUF(const device uint8_t*, w3, 9),                         \
      KQ_ROWS_MULTI_BUF(device T*, y1, 10),                                    \
      KQ_ROWS_MULTI_BUF(device T*, y2, 11),                                    \
      KQ_ROWS_MULTI_BUF(device T*, y3, 12),                                    \
      KQ_ROWS_MULTI_BUF(const constant int4&, n_rows, 13),                     \
      KQ_ROWS_MULTI_BUF(const constant int4&, tg_end, 14),                     \
      KQ_ROWS_MULTI_BUF(const constant int4&, nsgs, 15),                       \
      threadgroup float* red [[threadgroup(0)]],                               \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]] KQ_OS_ARG_##OS) {            \
    if (!kq_rows_multi) {                                                      \
      IMPL<T, __VA_ARGS__>(                                                    \
          w0,                                                                  \
          x,                                                                   \
          y0,                                                                  \
          K,                                                                   \
          N0,                                                                  \
          M,                                                                   \
          kq_rows_nsg,                                                         \
          red,                                                                 \
          tid,                                                                 \
          simd_gid,                                                            \
          simd_lid KQ_OS_PTR_##OS);                                            \
      return;                                                                  \
    }                                                                          \
    const int t = int(tid.x);                                                  \
    /* tg_end is cumulative, so the weight is the ends passed. */              \
    const short i =                                                            \
        short(t >= tg_end[0]) + short(t >= tg_end[1]) + short(t >= tg_end[2]); \
    const int t0 = i == 0 ? 0 : tg_end[i - 1];                                 \
    IMPL<T, __VA_ARGS__>(                                                      \
        i == 0       ? w0                                                      \
            : i == 1 ? w1                                                      \
            : i == 2 ? w2                                                      \
                     : w3,                                                     \
        x,                                                                     \
        i == 0       ? y0                                                      \
            : i == 1 ? y1                                                      \
            : i == 2 ? y2                                                      \
                     : y3,                                                     \
        K,                                                                     \
        n_rows[i],                                                             \
        M,                                                                     \
        kq_rows_nsg > 0 ? kq_rows_nsg : nsgs[i],                               \
        red,                                                                   \
        uint3(uint(t - t0), tid.y, tid.z),                                     \
        simd_gid,                                                              \
        simd_lid KQ_OS_PTR_##OS);                                              \
  }

#define KQ_DEFINE_ROWS_KERNEL(NAME, IMPL, TRAITS, OS) \
  KQ_DEFINE_ROWS_KERNEL_ARGS(NAME, IMPL, OS, TRAITS, OS)

// kq_verify_nax.h decode forms.
KQ_DEFINE_ROWS_KERNEL(pq2_0, kq_rows_nax_impl, KqPq2_0Nax, 0)
KQ_DEFINE_ROWS_KERNEL(q2_0, kq_rows_nax_impl, KqQ2_0Nax, 0)
KQ_DEFINE_ROWS_KERNEL(nvfp4, kq_rows_nax_impl, KqNvfp4Nax, 1)
KQ_DEFINE_ROWS_KERNEL(nvfp4_sb, kq_rows_nax_impl, KqNvfp4SbNax, 1)
KQ_DEFINE_ROWS_KERNEL(q4_0, kq_rows_nax_impl, KqQ4_0Nax, 0)
KQ_DEFINE_ROWS_KERNEL(q4_0_sb, kq_rows_nax_impl, KqQ4_0SbNax, 0)
KQ_DEFINE_ROWS_KERNEL(q4_1, kq_rows_nax_impl, KqQ4_1Nax, 0)
KQ_DEFINE_ROWS_KERNEL(q4_1_sb, kq_rows_nax_impl, KqQ4_1SbNax, 0)
KQ_DEFINE_ROWS_KERNEL(q8_0, kq_rows_nax_impl, KqQ8_0Nax, 0)
KQ_DEFINE_ROWS_KERNEL(q4_k, kq_rows_nax_impl, KqQ4_KNax, 0)
KQ_DEFINE_ROWS_KERNEL(q5_k, kq_rows_nax_impl, KqQ5_KNax, 0)
KQ_DEFINE_ROWS_KERNEL(q6_k, kq_rows_nax_impl, KqQ6_KNax, 0)
KQ_DEFINE_ROWS_KERNEL(q3_k, kq_rows_nax_impl, KqQ3_KNax, 0)
KQ_DEFINE_ROWS_KERNEL(q2_k, kq_rows_nax_impl, KqQ2_KNax, 0)
KQ_DEFINE_ROWS_KERNEL(iq4_xs, kq_rows_nax_impl, KqIq4XsNax, 0)
KQ_DEFINE_ROWS_KERNEL(iq4_nl, kq_rows_nax_impl, KqIq4NlNax, 0)

// Chunk decode forms (_gen): every codec with a mat-vec chunk decode.
KQ_DEFINE_ROWS_KERNEL(q2_k_gen, kq_rows_gen_impl, KqQ2_KExt, 0)
KQ_DEFINE_ROWS_KERNEL(q3_k_gen, kq_rows_gen_impl, KqQ3_KExt, 0)
KQ_DEFINE_ROWS_KERNEL(q4_k_gen, kq_rows_gen_impl, KqQ4_KExt, 0)
KQ_DEFINE_ROWS_KERNEL(q5_k_gen, kq_rows_gen_impl, KqQ5_KExt, 0)
KQ_DEFINE_ROWS_KERNEL(q6_k_gen, kq_rows_gen_impl, KqQ6_KExt, 0)
KQ_DEFINE_ROWS_KERNEL(q4_0_gen, kq_rows_gen_impl, KqQ4_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q4_1_gen, kq_rows_gen_impl, KqQ4_1Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q5_0_gen, kq_rows_gen_impl, KqQ5_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q5_1_gen, kq_rows_gen_impl, KqQ5_1Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q8_0_gen, kq_rows_gen_impl, KqQ8_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(iq4_nl_gen, kq_rows_gen_impl, KqIq4_nlExt, 0)
KQ_DEFINE_ROWS_KERNEL(iq4_xs_gen, kq_rows_gen_impl, KqIq4_xsExt, 0)
KQ_DEFINE_ROWS_KERNEL(iq3_s_gen, kq_rows_gen_impl, KqIq3_sExt, 0)
KQ_DEFINE_ROWS_KERNEL(iq3_xxs_gen, kq_rows_gen_impl, KqIq3_xxsExt, 0)
KQ_DEFINE_ROWS_KERNEL(iq2_xxs_gen, kq_rows_gen_impl, KqIq2_xxsExt, 0)
KQ_DEFINE_ROWS_KERNEL(iq2_xs_gen, kq_rows_gen_impl, KqIq2_xsExt, 0)
KQ_DEFINE_ROWS_KERNEL(iq2_s_gen, kq_rows_gen_impl, KqIq2_sExt, 0)
KQ_DEFINE_ROWS_KERNEL(iq1_s_gen, kq_rows_gen_impl, KqIq1_sExt, 0)
KQ_DEFINE_ROWS_KERNEL(iq1_m_gen, kq_rows_gen_impl, KqIq1_mExt, 0)
KQ_DEFINE_ROWS_KERNEL(stq1_0_gen, kq_rows_gen_impl, KqStq1_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(pq2_0_gen, kq_rows_gen_impl, KqPq2_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q2_0_gen, kq_rows_gen_impl, KqQ2_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(ptq1_0_gen, kq_rows_gen_impl, KqPtq1_0Ext, 0)
// mxfp4 with the weights in x's type (kWide).
KQ_DEFINE_ROWS_KERNEL_ARGS(
    mxfp4_gen,
    kq_rows_gen_impl,
    0,
    KqMxfp4Ext,
    false,
    0,
    true)
KQ_DEFINE_ROWS_KERNEL(nvfp4_gen, kq_rows_gen_impl, KqNvfp4Ext, 1)

// One block per unit, codecs under 128 weights a block.
KQ_DEFINE_ROWS_KERNEL(q4_0_gen1, kq_rows_gen1_impl, KqQ4_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q4_1_gen1, kq_rows_gen1_impl, KqQ4_1Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q5_0_gen1, kq_rows_gen1_impl, KqQ5_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q5_1_gen1, kq_rows_gen1_impl, KqQ5_1Ext, 0)
KQ_DEFINE_ROWS_KERNEL(q8_0_gen1, kq_rows_gen1_impl, KqQ8_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL(iq4_nl_gen1, kq_rows_gen1_impl, KqIq4_nlExt, 0)
KQ_DEFINE_ROWS_KERNEL(q2_0_gen1, kq_rows_gen1_impl, KqQ2_0Ext, 0)
KQ_DEFINE_ROWS_KERNEL_ARGS(
    mxfp4_gen1,
    kq_rows_gen1_impl,
    0,
    KqMxfp4Ext,
    false,
    true)
KQ_DEFINE_ROWS_KERNEL(nvfp4_gen1, kq_rows_gen1_impl, KqNvfp4Ext, 1)
