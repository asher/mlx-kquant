// Tiled score-mixed down projection of the fused MoE ops, at 2 to 16 rows
// on the tensor-op (NAX) units. The expert weights are read as GGUF blocks
// in place.
//
// The design is Splash's expert down pass (Apache-2.0; its
// runtime/metal/kernels/common/gguf_column_tile.h, gguf_staged_tile.h and
// shared/moe_gguf.metal): the (row, slot) pairs of a call are sorted by
// expert into tiles of up to KQ_TL_ROWS pairs, a tile's activation rows
// are made contiguous, and each tile reads its expert once.
//   kq_moe_tile_plan: sorts the pairs. It writes the tile table, each
//     pair's place in tile order, and the activation rows in that order.
//   gather_qmv_mix_tl: grid (N / 64, pairs). A threadgroup is one tile's
//     64 output columns, a lane a column. A tile of 1 or 2 rows keeps its
//     sums in registers (column tile). A larger tile stages 32 weights a
//     lane as half into threadgroup memory and runs an 8-row matmul2d on
//     them, one simdgroup each (staged tile). It writes each pair's row
//     sums [pairs, N].
//   kq_moe_tile_combine: adds a row's slots under their scores.
// The same tiles on the GLU lost to the grouped kernels at every row count
// and are not built.
// A weight is rounded to half before its products, so a weight past the
// half range (65504) turns to inf. A row's bits depend on the tile it is
// in, so on the call's width.

#pragma once

#include <metal_stdlib>
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define KQ_TL_ROWS 8
#define KQ_TL_COL_ROWS 2
// Bit 31 of a tile's expert marks the shared expert; the low bits are then
// the shared slot.
#define KQ_TL_SHARED 0x80000000u

// A lane's decode of one 32-weight step of a weight row: load reads what
// the step needs, dequant turns it into 8 half4 (weights 4i to 4i + 3 in
// v[i]). The generic form decodes through the codec's chunk decode and
// reads the block in dequant.
template <typename Codec>
struct KqTile {
  struct Payload {
    const device uint8_t* row;
    uint g;
  };
  static METAL_FUNC Payload load(const device uint8_t* row, uint g) {
    Payload p;
    p.row = row;
    p.g = g;
    return p;
  }
  static METAL_FUNC void
  dequant(Payload p, thread half4* v, const threadgroup uint8_t* luts) {
    constexpr int chpb = Codec::superblock / 16;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int ich = 2 * int(p.g) + h;
      float4x4 lw;
      float sc;
      KqTgLuts<Codec>::deq_chunk16s(
          p.row + (int64_t)(ich / chpb) * Codec::block_bytes,
          short(ich % chpb),
          lw,
          luts,
          sc);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[4 * h + i] = half4(lw[i] * sc);
      }
    }
  }
};

// A GGUF block's fields are 2-byte aligned, so a step's payload is read as
// halfwords.
METAL_FUNC uint kq_tl_u32(const device ushort* p) {
  return uint(p[0]) | uint(p[1]) << 16;
}

// q5_0: the block's bytes are loaded a step ahead of their decode.
template <>
struct KqTile<KqQ5_0Ext> {
  struct Payload {
    uint4 qs;
    uint qh;
    half d;
  };
  static METAL_FUNC Payload load(const device uint8_t* row, uint g) {
    const device ushort* q =
        (const device ushort*)(row + (int64_t)g * KQ_Q5_0_BLOCK_BYTES);
    Payload p;
    p.d = as_type<half>(q[0]);
    p.qh = kq_tl_u32(q + 1);
    p.qs = uint4(
        kq_tl_u32(q + 3), kq_tl_u32(q + 5), kq_tl_u32(q + 7), kq_tl_u32(q + 9));
    return p;
  }
  static METAL_FUNC void
  dequant(Payload p, thread half4* v, const threadgroup uint8_t*) {
    const float d = float(p.d);
#pragma unroll
    for (ushort c = 0; c < 4; ++c) {
      const uint hl = (p.qh >> (4 * c)) & 0xF;
      const uint hh = (p.qh >> (16 + 4 * c)) & 0xF;
      const uint lo =
          (p.qs[c] & 0x0F0F0F0Fu) | (((hl * 0x00204081u) & 0x01010101u) << 4);
      const uint hi = ((p.qs[c] >> 4) & 0x0F0F0F0Fu) |
          (((hh * 0x00204081u) & 0x01010101u) << 4);
      v[c] = half4((float4(as_type<uchar4>(lo)) - 16.0f) * d);
      v[4 + c] = half4((float4(as_type<uchar4>(hi)) - 16.0f) * d);
    }
  }
};

// iq4_nl: the same, 16 bytes of nibbles through the value table.
template <>
struct KqTile<KqIq4_nlExt> {
  struct Payload {
    uint4 qs;
    half d;
  };
  static METAL_FUNC Payload load(const device uint8_t* row, uint g) {
    const device ushort* q =
        (const device ushort*)(row + (int64_t)g * KQ_IQ4_NL_BLOCK_BYTES);
    Payload p;
    p.d = as_type<half>(q[0]);
    p.qs = uint4(
        kq_tl_u32(q + 1), kq_tl_u32(q + 3), kq_tl_u32(q + 5), kq_tl_u32(q + 7));
    return p;
  }
  static METAL_FUNC void
  dequant(Payload p, thread half4* v, const threadgroup uint8_t*) {
    const float d = float(p.d);
#pragma unroll
    for (ushort c = 0; c < 4; ++c) {
      const uint b = p.qs[c];
      v[c] = half4(
          float4(
              kvalues_iq4nl[b & 0xF],
              kvalues_iq4nl[(b >> 8) & 0xF],
              kvalues_iq4nl[(b >> 16) & 0xF],
              kvalues_iq4nl[(b >> 24) & 0xF]) *
          d);
      v[4 + c] = half4(
          float4(
              kvalues_iq4nl[(b >> 4) & 0xF],
              kvalues_iq4nl[(b >> 12) & 0xF],
              kvalues_iq4nl[(b >> 20) & 0xF],
              kvalues_iq4nl[b >> 28]) *
          d);
    }
  }
};

// Column tile: the lane's weight row over all of K against `rows` (1 or 2)
// activation rows, sums in registers, the next step loaded while this
// one's products run.
template <typename T, typename Codec>
METAL_FUNC float2 kq_tl_column(
    const device uint8_t* w_row,
    const device T* x,
    uint rows,
    int K,
    const threadgroup uint8_t* luts) {
  const uint steps = uint(K) / 32;
  float2 sums = float2(0.0f);
  typename KqTile<Codec>::Payload w = KqTile<Codec>::load(w_row, 0);
  for (uint g = 0; g < steps; ++g) {
    half4 v[8];
    KqTile<Codec>::dequant(w, v, luts);
    if (g + 1 < steps) {
      w = KqTile<Codec>::load(w_row, g + 1);
    }
#pragma unroll
    for (ushort r = 0; r < KQ_TL_COL_ROWS; ++r) {
      if (r >= rows) {
        break;
      }
      const device T* xr = x + (int64_t)r * K + g * 32;
      float sum = 0.0f;
#pragma unroll
      for (ushort i = 0; i < 8; ++i) {
        const float4 xv = float4(*(const device vec<T, 4>*)(xr + 4 * i));
        const float4 wv = float4(v[i]);
        sum = sum + wv.x * xv.x;
        sum = sum + wv.y * xv.y;
        sum = sum + wv.z * xv.z;
        sum = sum + wv.w * xv.w;
      }
      sums[r] += sum;
    }
  }
  return sums;
}

// Staged tile of one simdgroup: lane l stages its weight row, 32 weights a
// step, into the stage (two buffers of 32 x 32 half, alternating), then the
// 8-row matmul2d of the step's activations runs on it. The sums are parked
// at park[row * 32 + lane], which may alias the stage.
template <typename T, typename Codec>
METAL_FUNC void kq_tl_staged(
    const device uint8_t* w_row,
    const device T* x,
    int K,
    uint lane,
    threadgroup half* stage,
    threadgroup float* park,
    const threadgroup uint8_t* luts) {
  using namespace mpp::tensor_ops;
  auto ta = tensor(
      (device T*)x, dextents<int, 2>{K, KQ_TL_ROWS}, array<int, 2>{1, K});
  constexpr auto desc = matmul2d_descriptor(
      KQ_TL_ROWS,
      32,
      32,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
      stage, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
      stage + 1024, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
  auto b0 = bt0.slice<32, 32>(0, 0);
  auto b1 = bt1.slice<32, 32>(0, 0);
  auto a0 = ta.template slice<32, KQ_TL_ROWS>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(a0),
      decltype(b0),
      float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  const uint steps = uint(K) / 32;
  typename KqTile<Codec>::Payload w = KqTile<Codec>::load(w_row, 0);
  for (uint s = 0; s < steps; ++s) {
    threadgroup half* buf = stage + (s & 1) * 1024 + lane * 32;
    half4 v[8];
    KqTile<Codec>::dequant(w, v, luts);
#pragma unroll
    for (ushort i = 0; i < 8; ++i) {
      *((threadgroup half4*)(buf + 4 * i)) = v[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (s + 1 < steps) {
      w = KqTile<Codec>::load(w_row, s + 1);
    }
    auto as = ta.template slice<32, KQ_TL_ROWS>(s * 32, 0);
    if (s & 1) {
      op.run(as, b1, acc);
    } else {
      op.run(as, b0, acc);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) {
      continue;
    }
    const auto index = acc.get_multidimensional_index(i);
    park[uint(index[1]) * 32 + uint(index[0])] = float(acc[i]);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

// One tile's 32 columns of a simdgroup: weight row w_row against the tile's
// rows at x, written to y (the tile's first pair, this lane's column).
template <typename T, typename Codec>
METAL_FUNC void kq_tl_tile(
    const device uint8_t* w_row,
    const device T* x,
    device T* y,
    uint rows,
    uint col_rows,
    int K,
    int N,
    uint lane,
    bool live,
    threadgroup half* stage,
    const threadgroup uint8_t* luts) {
  if (rows <= col_rows) {
    const float2 sums = kq_tl_column<T, Codec>(w_row, x, rows, K, luts);
    if (live) {
      for (uint r = 0; r < rows; ++r) {
        y[(int64_t)r * N] = static_cast<T>(sums[r]);
      }
    }
    return;
  }
  threadgroup float* park = (threadgroup float*)stage;
  kq_tl_staged<T, Codec>(w_row, x, K, lane, stage, park, luts);
  if (live) {
    for (uint r = 0; r < rows; ++r) {
      y[(int64_t)r * N] = static_cast<T>(park[r * 32 + lane]);
    }
  }
}

// hg is [pairs + KQ_TL_ROWS - 1, K] in tile order, tiles [pairs, 4] as
// kq_moe_tile_plan writes it, y [pairs, N]. A tile of up to COLR (at most
// KQ_TL_COL_ROWS) rows is a column tile. sw is the shared expert, [N * NS, K]
// with a column's NS rows adjacent, as the mix kernels read it. Grid (ceil(N /
// 64), pairs), threads (32, 2).
template <typename T, typename Codec, typename SCodec>
[[kernel]] void kq_tl_gather_qmv_mix(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* sw [[buffer(1)]],
    const device T* hg [[buffer(2)]],
    const device uint32_t* tiles [[buffer(3)]],
    device T* y [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& NS [[buffer(9)]],
    const constant int& COLR [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint4 td = *(const device uint4*)(tiles + 4 * tid.y);
  const uint rows = td.z & 0xFFu;
  if (rows == 0) {
    return;
  }
  const uint col_rows = min(uint(COLR), uint(KQ_TL_COL_ROWS));
  const bool shared = (td.x & KQ_TL_SHARED) != 0;
  const uint expert = td.x & ~KQ_TL_SHARED;
  const uint col = tid.x * 64 + simd_gid * 32 + simd_lid;
  const bool live = col < uint(N);
  const uint c = live ? col : uint(N) - 1;
  // Two 32 x 32 half buffers a simdgroup, as float for the park's alignment.
  threadgroup float stage_f[2048];
  threadgroup half* stage = (threadgroup half*)stage_f + simd_gid * 2048;
  const device T* x = hg + (int64_t)td.y * K;
  device T* yo = y + (int64_t)td.y * N + c;
  KQ_EXT_DECL_LUTS2(Codec, SCodec, kq_luts)
  if (shared) {
    KQ_EXT_STAGE_INTO(SCodec, kq_luts)
    const int64_t rb = (int64_t)(K / SCodec::superblock) * SCodec::block_bytes;
    kq_tl_tile<T, SCodec>(
        sw + ((int64_t)c * NS + expert) * rb,
        x,
        yo,
        rows,
        col_rows,
        K,
        N,
        simd_lid,
        live,
        stage,
        kq_luts);
  } else {
    KQ_EXT_STAGE_INTO(Codec, kq_luts)
    const int64_t rb = (int64_t)(K / Codec::superblock) * Codec::block_bytes;
    kq_tl_tile<T, Codec>(
        w + ((int64_t)expert * N + c) * rb,
        x,
        yo,
        rows,
        col_rows,
        K,
        N,
        simd_lid,
        live,
        stage,
        kq_luts);
  }
}

// Sorts the pairs of a call by expert, a threadgroup a pair. Routed pair
// p = t * (S - NS) + s of indices [T, S - NS] takes rank = pairs on a lower
// expert + earlier pairs on its own; a shared pair (slot j, row t) follows
// the routed ones at T * (S - NS) + j * T + t. The pair writes its rank to
// place[p] (routed pairs only) and tiles[rank] = {expert, rank, rows |
// total << 8, t * S + slot}: total is the pairs on its expert, and rows is
// the size of the tile it starts (every KQ_TL_ROWS-th pair of an expert
// starts one of the pairs up to the next start) or 0. It copies its
// activation row h[t, slot] of h [T, S, K] to hg[rank]. Grid (T * S),
// threads 32. K is a multiple of 8.
template <typename T>
[[kernel]] void kq_moe_tile_plan(
    const device uint32_t* indices [[buffer(0)]],
    const device T* h [[buffer(1)]],
    device T* hg [[buffer(2)]],
    device uint32_t* tiles [[buffer(3)]],
    device uint32_t* place [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& S [[buffer(6)]],
    const constant int& NS [[buffer(7)]],
    const constant int& T_ [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n_route = uint(S - NS);
  const uint pr = uint(T_) * n_route;
  const uint p = tid.x;
  uint rank;
  uint rows;
  uint total;
  uint e;
  uint t;
  uint slot;
  if (p < pr) {
    e = indices[p];
    uint lt = 0;
    uint eqb = 0;
    total = 0;
    for (uint q0 = 0; q0 < pr; q0 += 32) {
      const uint q = q0 + lane;
      const bool in = q < pr;
      const uint eq = in ? indices[q] : 0u;
      lt += popcount(static_cast<ulong>(simd_ballot(in && eq < e)));
      eqb += popcount(static_cast<ulong>(simd_ballot(in && eq == e && q < p)));
      total += popcount(static_cast<ulong>(simd_ballot(in && eq == e)));
    }
    rank = lt + eqb;
    rows = eqb % KQ_TL_ROWS == 0 ? min(uint(KQ_TL_ROWS), total - eqb) : 0u;
    t = p / n_route;
    slot = p - t * n_route;
    if (lane == 0) {
      place[p] = rank;
    }
  } else {
    const uint q = p - pr;
    const uint j = q / uint(T_);
    t = q - j * uint(T_);
    rank = p;
    total = uint(T_);
    rows = t % KQ_TL_ROWS == 0 ? min(uint(KQ_TL_ROWS), total - t) : 0u;
    e = KQ_TL_SHARED | j;
    slot = n_route + j;
  }
  const uint pair = t * uint(S) + slot;
  if (lane == 0) {
    *(device uint4*)(tiles + 4 * rank) =
        uint4(e, rank, rows | (total << 8), pair);
  }
  const device T* from = h + (int64_t)pair * K;
  device T* to = hg + (int64_t)rank * K;
  for (int i = int(lane) * 8; i < K; i += 256) {
    *(device vec<T, 8>*)(to + i) = *(const device vec<T, 8>*)(from + i);
  }
}

template <typename T>
[[kernel]] void kq_moe_tile_combine(
    const device T* y [[buffer(0)]],
    const device float* scores [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant int& N [[buffer(3)]],
    const constant int& S [[buffer(4)]],
    const constant int& SC [[buffer(5)]],
    const constant int& NS [[buffer(6)]],
    const device uint32_t* place [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int n = int(gid.x) * 4;
  const int t = int(gid.z);
  const int T_ = int(grid.z);
  const int n_route = S - NS;
  float4 acc = float4(0.0f);
  for (int s = 0; s < n_route; ++s) {
    const int64_t pl = place[t * n_route + s];
    acc +=
        scores[t * SC + s] * float4(*(const device vec<T, 4>*)(y + pl * N + n));
  }
  for (int j = 0; j < NS; ++j) {
    const int64_t pl = (int64_t)T_ * n_route + j * T_ + t;
    acc += (SC == S ? scores[t * SC + n_route + j] : 1.0f) *
        float4(*(const device vec<T, 4>*)(y + pl * N + n));
  }
  *(device vec<T, 4>*)(out + (int64_t)t * N + n) = vec<T, 4>(acc);
}

// clang-format off
#define instantiate_kq_tl_pair(stem, traits, straits)                         \
  instantiate_kernel(                                                         \
      "kq_" stem "_gather_qmv_mix_tl_bfloat16_t",                             \
      kq_tl_gather_qmv_mix, bfloat16_t, traits, straits)                      \
  instantiate_kernel(                                                         \
      "kq_" stem "_gather_qmv_mix_tl_float16_t",                              \
      kq_tl_gather_qmv_mix, float16_t, traits, straits)

// The pairs of instantiate_kq_ext_all (kq_moe_glu_kq.h), under the grouped
// kernels' stems.
#define instantiate_kq_tl_all(codec, traits)                                  \
  instantiate_kq_tl_pair(#codec "_sx_" #codec, traits, traits)                \
  instantiate_kq_tl_pair(#codec "_sx_q5_k", traits, KqQ5_KExt)                \
  instantiate_kq_tl_pair(#codec "_sx_q6_k", traits, KqQ6_KExt)                \
  instantiate_kq_tl_pair(#codec "_sx_q8_0", traits, KqQ8_0Ext)
// clang-format on
