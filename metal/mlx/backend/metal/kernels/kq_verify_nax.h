#pragma once

#include <metal_stdlib>
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Register-fed NAX verify kernels for the M <= 8 band (NAX GPUs).
//
// Each of the 4 simdgroups owns 32 weight rows. Lane (fm, q) of the
// matmul2d 16x32x16 fragment holds rows fm + 8r (r < 4) of the right
// operand and decodes lane-quad q's bytes of each block of those rows
// straight into it. The left operand is activation rows 0..7, with rows
// 8..15 as padding, loaded from device memory per block in the codec's k
// order. Nothing is staged in threadgroup memory and there are no
// barriers. Each simdgroup keeps one f32 accumulator. Split-K over grid z
// writes T partials that kquant_qmm_splitk_accum folds.
//
// A unit is one or more wire blocks. For pq2_0 and q4_0, step s of a unit
// consumes the codec's verify_mma fragments 2s and 2s + 1
// (kq_verify_mma.h), and left element 2 jp + e is the activation at that
// codec's perm(2s + jp, 2q + e). The q8_0 unit is four wire blocks with
// its own order (KqQ8_0Nax). The block scale is folded into the half
// weights. Device loads and NAX ops of one simdgroup do not overlap, so
// occupancy hides the loads and the host sizes the split count for it
// (kq_verify_nax_splits). pq2_0 and q4_0 load a unit's activation steps
// ahead of its NAX ops, and q8_0, whose Words are larger, loads one step
// at a time; each order measured faster for its codecs on M5 Max. The
// right operand is written in element order, which ran 1.6x faster than
// a strided order.
//
// Codec contract (KqXxxNax):
//   block_k, block_bytes: the unit's weights and bytes
//   ub: units per loop iteration
//   x_ahead: load the iteration's activation steps before its NAX ops
//   Words: a lane's decode state for one unit of one row
//   load(bp, q): Words of unit bp for lane-quad q
//   pair(words, f): fragment f's element pair, block scale applied
//   xstep<T>(xb, q, s): left elements 0..3 of step s, xb the unit's
//     first activation of the lane's row

MLX_MTL_CONST int KQ_VNAX_NSG = 4;

// PQ2_0: lane-quad q owns the two 16-code words at qs bytes 8q..8q+7.
// Pair f is code f % 8 of word f / 8 in each half (KqPq2_0Mma).
struct KqPq2_0Nax {
  static constant constexpr int block_k = KQ_PQ2_0_SUPERBLOCK;
  static constant constexpr int block_bytes = KQ_PQ2_0_BLOCK_BYTES;
  static constant constexpr int ub = 1;
  static constant constexpr bool x_ahead = true;
  struct Words {
    uint w[2];
    half d;
  };
  static METAL_FUNC Words load(const device uint8_t* bp, short q) {
    Words o;
    o.d = *(const device half*)bp;
    const packed_ushort4 v =
        *(const device packed_ushort4*)(bp + KQ_PQ2_0_QS_OFFSET + 8 * q);
    o.w[0] = uint(v.x) | (uint(v.y) << 16);
    o.w[1] = uint(v.z) | (uint(v.w) << 16);
    return o;
  }
  static METAL_FUNC half2 pair(thread const Words& o, short f) {
    const short h = f >> 3;
    const short j = f & 7;
    const short jj = j < 5 ? j : j - 5;
    const uint src = j < 5 ? o.w[h] : (o.w[h] >> 10);
    const uint mask = 0x00030003u << (2 * jj);
    const half scale = half(1.0f / float(1 << (2 * jj)));
    const half off = -half(1024.0f / float(1 << (2 * jj))) - 1.0h;
    return fma(as_type<half2>((src & mask) | 0x64006400u),
               half2(scale),
               half2(off)) *
        o.d;
  }
  template <typename T>
  static METAL_FUNC vec<T, 4> xstep(const device T* xb, short q, short s) {
    const int o = 32 * q + 16 * (s >> 2) + 2 * (s & 3);
    const vec<T, 2> p0 = *(const device vec<T, 2>*)(xb + o);
    const vec<T, 2> p1 = *(const device vec<T, 2>*)(xb + o + 8);
    return vec<T, 4>(p0.x, p1.x, p0.y, p1.y);
  }
};

// Q4_0: lane-quad q owns qs bytes 4q..4q+3. Pairs (n0, n2), (n1, n3),
// (h0, h2), (h1, h3) of its low and high nibbles (KqQ4_0Mma), as
// 1024 + code in the half mantissa, minus 1032, times d. Two 32-weight
// blocks per iteration.
struct KqQ4_0Nax {
  static constant constexpr int block_k = KQ_Q4_0_GROUP;
  static constant constexpr int block_bytes = KQ_Q4_0_BLOCK_BYTES;
  static constant constexpr int ub = 2;
  static constant constexpr bool x_ahead = true;
  struct Words {
    uint lo;
    uint hi;
    half d;
  };
  static METAL_FUNC Words load(const device uint8_t* bp, short q) {
    Words o;
    o.d = *(const device half*)(bp + KQ_Q4_0_D_OFFSET);
    const packed_ushort2 v =
        *(const device packed_ushort2*)(bp + KQ_Q4_0_QS_OFFSET + 4 * q);
    const uint wv = uint(v.x) | (uint(v.y) << 16);
    o.lo = wv & 0x0F0F0F0Fu;
    o.hi = (wv >> 4) & 0x0F0F0F0Fu;
    return o;
  }
  static METAL_FUNC half2 pair(thread const Words& o, short f) {
    const uint src = (f < 2 ? o.lo : o.hi) >> (8 * (f & 1));
    return (as_type<half2>((src & 0x000F000Fu) | 0x64006400u) -
            half2(1032.0h)) *
        o.d;
  }
  template <typename T>
  static METAL_FUNC vec<T, 4> xstep(const device T* xb, short q, short s) {
    const vec<T, 4> p = *(const device vec<T, 4>*)(xb + 16 * s + 4 * q);
    return vec<T, 4>(p[0], p[2], p[1], p[3]);
  }
};

// Q8_0: the unit is four wire blocks (128 weights, 136 bytes), and
// lane-quad q owns block q of it, so a lane's loads run along one row.
// The qs bytes are read as aligned words, with a 2-byte funnel shift for
// the blocks whose qs start 2 mod 4, so a row's bytes must start 4-byte
// aligned (whole rows of a K that is a multiple of 128). Step s takes
// word s: pairs (b0, b2), (b1, b3), sign bits flipped, as 1152 + code in
// the half mantissa, minus 1152, times d.
struct KqQ8_0Nax {
  static constant constexpr int block_k = 4 * KQ_Q8_0_GROUP;
  static constant constexpr int block_bytes = 4 * KQ_Q8_0_BLOCK_BYTES;
  static constant constexpr int ub = 1;
  static constant constexpr bool x_ahead = false;
  struct Words {
    uint w[8];
    half d;
  };
  static METAL_FUNC Words load(const device uint8_t* bp, short q) {
    Words o;
    const int b = KQ_Q8_0_BLOCK_BYTES * q;
    o.d = *(const device half*)(bp + b + KQ_Q8_0_D_OFFSET);
    const int a = b + KQ_Q8_0_Q_OFFSET;
    const bool sh = (a & 2) != 0;
    const device uint* wp = (const device uint*)(bp + (a & ~3));
    uint t[9];
    for (short i = 0; i < 8; ++i) {
      t[i] = wp[i];
    }
    t[8] = sh ? wp[8] : 0u;
    for (short i = 0; i < 8; ++i) {
      o.w[i] = sh ? ((t[i] >> 16) | (t[i + 1] << 16)) : t[i];
    }
    return o;
  }
  static METAL_FUNC half2 pair(thread const Words& o, short f) {
    const uint src = (o.w[f >> 1] >> (8 * (f & 1))) & 0x00FF00FFu;
    return (as_type<half2>((src ^ 0x00800080u) | 0x64006400u) -
            half2(1152.0h)) *
        o.d;
  }
  template <typename T>
  static METAL_FUNC vec<T, 4> xstep(const device T* xb, short q, short s) {
    const vec<T, 4> p =
        *(const device vec<T, 4>*)(xb + KQ_Q8_0_GROUP * q + 4 * s);
    return vec<T, 4>(p[0], p[2], p[1], p[3]);
  }
};

template <typename T, typename Codec>
METAL_FUNC void kq_verify_nax_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  namespace to = mpp::tensor_ops;
  constexpr int BK = Codec::block_k;
  constexpr int UB = Codec::ub;
  constexpr int SPB = BK / 16;
  const int n0 = (tid.x * KQ_VNAX_NSG + simd_gid) * 32;
  if (n0 >= N) {
    return;
  }
  const int row_bytes = (K / BK) * Codec::block_bytes;
  const short qid = simd_lid >> 2;
  const short fm = (qid & 4) | ((simd_lid >> 1) & 3);
  const short fn = ((qid & 2) | (simd_lid & 1)) * 4;
  const short q = fn >> 2;
  const int kbeg = tid.z * k_partition_size;
  const int kend = kbeg + k_partition_size;

  constexpr auto desc = to::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      true,
      true,
      to::matmul2d_descriptor::mode::multiply_accumulate);
  to::matmul2d<desc, execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<T, half, float>();
  auto cb = op.template get_right_input_cooperative_tensor<T, half, float>();
  auto cc = op.template get_destination_cooperative_tensor<
      decltype(ca),
      decltype(cb),
      float>();
  for (short i = 0; i < 16; ++i) {
    cc[i] = 0.0f;
  }
  for (short i = 0; i < 8; ++i) {
    ca[i] = T(0);
  }

  const bool xrow = fm < M;
  const device T* xr = x + static_cast<int64_t>(xrow ? fm : 0) * K;
  // Row r of the lane is n0 + fm + 8r, clamped into the last row tile.
  const int nl = min(n0 + fm, N - 1);
  const device uint8_t* wr = w + static_cast<int64_t>(nl) * row_bytes;
  auto row = [&](short r) {
    return wr + static_cast<int64_t>(min(8 * r, N - 1 - nl)) * row_bytes;
  };

  for (int kb = kbeg; kb < kend; kb += UB * BK) {
    typename Codec::Words wv[UB][4];
#pragma unroll
    for (short u = 0; u < UB; ++u) {
#pragma unroll
      for (short r = 0; r < 4; ++r) {
        wv[u][r] =
            Codec::load(row(r) + ((kb + u * BK) / BK) * Codec::block_bytes, q);
      }
    }
    vec<T, 4> xa[UB][SPB];
    if (Codec::x_ahead) {
#pragma unroll
      for (short u = 0; u < UB; ++u) {
#pragma unroll
        for (short s = 0; s < SPB; ++s) {
          xa[u][s] = xrow ? Codec::template xstep<T>(xr + kb + u * BK, q, s)
                          : vec<T, 4>(0);
        }
      }
    }
#pragma unroll
    for (short u = 0; u < UB; ++u) {
#pragma unroll
      for (short s = 0; s < SPB; ++s) {
        const vec<T, 4> xv = Codec::x_ahead ? xa[u][s]
            : xrow ? Codec::template xstep<T>(xr + kb + u * BK, q, s)
                   : vec<T, 4>(0);
        ca[0] = xv[0];
        ca[1] = xv[1];
        ca[2] = xv[2];
        ca[3] = xv[3];
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

  // Destination: lane (fm, fn) holds row fm, columns n0 + 16f + fn + j.
  if (!xrow) {
    return;
  }
  for (short f = 0; f < 2; ++f) {
    for (short j = 0; j < 4; ++j) {
      const int n = n0 + 16 * f + fn + j;
      if (n < N) {
        y[static_cast<int64_t>(fm) * N + n] = static_cast<T>(cc[f * 8 + j]);
      }
    }
  }
}

// Entry point: qmm_t_splitk's buffer layout, grid (ceil(N / 128), 1,
// splits), 128 threads.
#define KQ_DEFINE_VERIFY_NAX_KERNEL(CODEC, TRAITS)                    \
  template <typename T, int group_size, int bits>                     \
  [[kernel]] void kq_##CODEC##_verify_nax(                            \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      device T* y,                                                    \
      const constant int& K,                                          \
      const constant int& N,                                          \
      const constant int& M,                                          \
      const constant int& k_partition_size,                           \
      const constant int& split_k_partition_stride,                   \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    static_assert(TRAITS::block_k % group_size == 0, #CODEC " unit"); \
    kq_verify_nax_impl<T, TRAITS>(                                    \
        w,                                                            \
        x,                                                            \
        y + tid.z * static_cast<int64_t>(split_k_partition_stride),   \
        K,                                                            \
        N,                                                            \
        M,                                                            \
        k_partition_size,                                             \
        tid,                                                          \
        simd_gid,                                                     \
        simd_lid);                                                    \
  }

KQ_DEFINE_VERIFY_NAX_KERNEL(pq2_0, KqPq2_0Nax)
KQ_DEFINE_VERIFY_NAX_KERNEL(q4_0, KqQ4_0Nax)
KQ_DEFINE_VERIFY_NAX_KERNEL(q8_0, KqQ8_0Nax)
