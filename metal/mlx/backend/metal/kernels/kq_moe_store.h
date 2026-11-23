// Expert store kernels of the fused MoE ops (moe_store_pack, moe_store_glu,
// moe_store_down): the tile design of kq_moe_tile.h reading routed experts
// from a store, a copy of their GGUF blocks in a layout the tiles read
// faster. The GGUF blocks stay the fallback: an expert with no slot is read
// in place by the same kernel, and the shared expert is always read in
// place.
//
// Store layout of one projection [E, N, row bytes] of a codec: each field of
// the codec's block is a plane of its own, and inside a plane the 64 output
// columns of a tile sit side by side at each step along K:
//   plane[slot][N / 64][unit][64][R]
// unit is a 32-weight group, or a block for a field a block holds once; R is
// the field's bytes. A plane holds the bytes of the GGUF field unchanged, so
// a slot is as large as the expert's GGUF blocks. slots[expert] is the
// expert's slot or KQ_ST_ABSENT.
//   kq_moe_store_pack: copies one plane of the listed experts.
//   kq_moe_store_plan: kq_moe_tile_plan with the rows of x [T, K] as the
//     activation rows (the GLU's input).
//   kq_st_glu: grid (N / 32, pairs), threads (32, 2). Simdgroup 0 runs a
//     tile's gate, simdgroup 1 its up, on the same 32 columns.
//   kq_st_down: the grid and outputs of kq_tl_gather_qmv_mix.
// The down's arithmetic is that of kq_tl_gather_qmv_mix, so its bits equal
// the in-place tile kernel's. The GLU rounds a weight to half before its
// products like the tile down; the grouped GLU kernels do not, so the GLU's
// bits differ from theirs.

#pragma once

#include <metal_stdlib>
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define KQ_ST_ABSENT 0x80000000u
#define KQ_ST_TILE 64u

// A codec's planes: R bytes a record, UG groups a record. load reads a
// step's records at q0, q1, q2; dequant gives the 32 weights as in KqTile.
//
// The generic format (every codec without a specialization below) has one
// plane whose record is the codec's whole block: the blocks of a tile's 64
// columns sit side by side at each block step along K, and a step decodes
// through the codec's chunk decode as the generic KqTile does, so the bits
// are the in-place tile kernel's. LUTS: the format reads the codec's
// threadgroup tables, which the kernel then stages.
template <typename Codec>
struct KqStoreFmt {
  struct Payload {
    const device uint8_t* block;
  };
  MLX_MTL_CONST bool LUTS = true;
  MLX_MTL_CONST uint R0 = Codec::block_bytes;
  MLX_MTL_CONST uint UG0 = Codec::superblock / 32;
  MLX_MTL_CONST uint R1 = Codec::block_bytes;
  MLX_MTL_CONST uint UG1 = Codec::superblock / 32;
  MLX_MTL_CONST uint R2 = Codec::block_bytes;
  MLX_MTL_CONST uint UG2 = Codec::superblock / 32;
  static METAL_FUNC Payload
  load(const device uint8_t* q0, const device uint8_t*, const device uint8_t*) {
    Payload p;
    p.block = q0;
    return p;
  }
  static METAL_FUNC void
  dequant(Payload p, uint g, thread half4* v, const threadgroup uint8_t* luts) {
    const int ch0 = 2 * int(g % UG0);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float4x4 lw;
      float sc;
      KqTgLuts<Codec>::deq_chunk16s(p.block, short(ch0 + h), lw, luts, sc);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[4 * h + i] = half4(lw[i] * sc);
      }
    }
  }
};

template <>
struct KqStoreFmt<KqQ5_0Ext> {
  typedef KqTile<KqQ5_0Ext>::Payload Payload;
  MLX_MTL_CONST bool LUTS = false;
  MLX_MTL_CONST uint R0 = 16;
  MLX_MTL_CONST uint UG0 = 1;
  MLX_MTL_CONST uint R1 = 4;
  MLX_MTL_CONST uint UG1 = 1;
  MLX_MTL_CONST uint R2 = 2;
  MLX_MTL_CONST uint UG2 = 1;
  static METAL_FUNC Payload load(
      const device uint8_t* q0,
      const device uint8_t* q1,
      const device uint8_t* q2) {
    Payload p;
    p.qs = *(const device uint4*)q0;
    p.qh = *(const device uint*)q1;
    p.d = *(const device half*)q2;
    return p;
  }
  static METAL_FUNC void
  dequant(Payload p, uint, thread half4* v, const threadgroup uint8_t* luts) {
    KqTile<KqQ5_0Ext>::dequant(p, v, luts);
  }
};

template <>
struct KqStoreFmt<KqIq4_nlExt> {
  typedef KqTile<KqIq4_nlExt>::Payload Payload;
  MLX_MTL_CONST bool LUTS = false;
  MLX_MTL_CONST uint R0 = 16;
  MLX_MTL_CONST uint UG0 = 1;
  MLX_MTL_CONST uint R1 = 2;
  MLX_MTL_CONST uint UG1 = 1;
  MLX_MTL_CONST uint R2 = 2;
  MLX_MTL_CONST uint UG2 = 1;
  static METAL_FUNC Payload load(
      const device uint8_t* q0,
      const device uint8_t* q1,
      const device uint8_t*) {
    Payload p;
    p.qs = *(const device uint4*)q0;
    p.d = *(const device half*)q1;
    return p;
  }
  static METAL_FUNC void
  dequant(Payload p, uint, thread half4* v, const threadgroup uint8_t* luts) {
    KqTile<KqIq4_nlExt>::dequant(p, v, luts);
  }
};

// The 16 nibbles of a group through the iq4 value table, scaled.
METAL_FUNC void kq_st_nl_deq(uint4 qs, float d, thread half4* v) {
#pragma unroll
  for (ushort c = 0; c < 4; ++c) {
    const uint b = qs[c];
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

// iq4_xs: plane 0 the 16 nibble bytes of a group, plane 1 the block's first
// 8 bytes (d, scales_h, scales_l).
template <>
struct KqStoreFmt<KqIq4_xsExt> {
  struct Payload {
    uint4 qs;
    uint2 meta;
  };
  MLX_MTL_CONST bool LUTS = false;
  MLX_MTL_CONST uint R0 = 16;
  MLX_MTL_CONST uint UG0 = 1;
  MLX_MTL_CONST uint R1 = 8;
  MLX_MTL_CONST uint UG1 = 8;
  MLX_MTL_CONST uint R2 = 8;
  MLX_MTL_CONST uint UG2 = 8;
  static METAL_FUNC Payload load(
      const device uint8_t* q0,
      const device uint8_t* q1,
      const device uint8_t*) {
    Payload p;
    p.qs = *(const device uint4*)q0;
    p.meta = *(const device uint2*)q1;
    return p;
  }
  static METAL_FUNC void
  dequant(Payload p, uint g, thread half4* v, const threadgroup uint8_t*) {
    const uint ib = g % 8u;
    const int ls =
        int(((p.meta.y >> (4 * ib)) & 0xF) |
            (((p.meta.x >> (16 + 2 * ib)) & 3) << 4));
    const float d = float(as_type<half>(ushort(p.meta.x & 0xFFFFu)));
    kq_st_nl_deq(p.qs, d * float(ls - 32), v);
  }
};

// iq3_xxs: plane 0 the 8 grid bytes of a group, plane 1 its sign and scale
// word, plane 2 the block's d.
template <>
struct KqStoreFmt<KqIq3_xxsExt> {
  struct Payload {
    uint2 qs;
    uint gas;
    half d;
  };
  MLX_MTL_CONST bool LUTS = false;
  MLX_MTL_CONST uint R0 = 8;
  MLX_MTL_CONST uint UG0 = 1;
  MLX_MTL_CONST uint R1 = 4;
  MLX_MTL_CONST uint UG1 = 1;
  MLX_MTL_CONST uint R2 = 2;
  MLX_MTL_CONST uint UG2 = 8;
  static METAL_FUNC Payload load(
      const device uint8_t* q0,
      const device uint8_t* q1,
      const device uint8_t* q2) {
    Payload p;
    p.qs = *(const device uint2*)q0;
    p.gas = *(const device uint*)q1;
    p.d = *(const device half*)q2;
    return p;
  }
  static METAL_FUNC void
  dequant(Payload p, uint, thread half4* v, const threadgroup uint8_t*) {
    const float s = float(p.d) * (0.5f + float(p.gas >> 28)) * 0.5f;
#pragma unroll
    for (ushort l = 0; l < 4; ++l) {
      const uint i7 = (p.gas >> (7 * l)) & 127;
      // ksigns_iq2xs[i]: i with its parity in bit 7.
      const uint sg = i7 | ((popcount(i7) & 1) << 7);
      const uint w = p.qs[l / 2] >> (16 * (l % 2));
      const float4 v0 = float4(as_type<uchar4>(iq3xxs_grid[w & 0xFF]));
      const float4 v1 = float4(as_type<uchar4>(iq3xxs_grid[(w >> 8) & 0xFF]));
      v[2 * l] =
          half4(select(v0, -v0, bool4(sg & 1, sg & 2, sg & 4, sg & 8)) * s);
      v[2 * l + 1] = half4(
          select(v1, -v1, bool4(sg & 16, sg & 32, sg & 64, sg & 128)) * s);
    }
  }
};

// A lane's weight row in the store: output column col of slot.
template <typename Codec>
struct KqStoreSrc {
  typedef KqStoreFmt<Codec> F;
  typedef typename F::Payload Payload;
  const device uint8_t* q0;
  const device uint8_t* q1;
  const device uint8_t* q2;
  void begin(
      const device uint8_t* b0,
      const device uint8_t* b1,
      const device uint8_t* b2,
      uint slot,
      uint col,
      int N,
      int K) thread {
    const ulong tile = ulong(slot) * (uint(N) / KQ_ST_TILE) + col / KQ_ST_TILE;
    const ulong r = col % KQ_ST_TILE;
    const ulong G = uint(K) / 32;
    q0 = b0 + (tile * (G / F::UG0) * KQ_ST_TILE + r) * F::R0;
    q1 = b1 + (tile * (G / F::UG1) * KQ_ST_TILE + r) * F::R1;
    q2 = b2 + (tile * (G / F::UG2) * KQ_ST_TILE + r) * F::R2;
  }
  Payload load(uint g) thread {
    return F::load(
        q0 + ulong(g / F::UG0) * (KQ_ST_TILE * F::R0),
        q1 + ulong(g / F::UG1) * (KQ_ST_TILE * F::R1),
        q2 + ulong(g / F::UG2) * (KQ_ST_TILE * F::R2));
  }
  static METAL_FUNC void
  dequant(Payload p, uint g, thread half4* v, const threadgroup uint8_t* luts) {
    F::dequant(p, g, v, luts);
  }
};

// A lane's weight row as GGUF blocks in place.
template <typename Codec>
struct KqBlockSrc {
  typedef typename KqTile<Codec>::Payload Payload;
  const device uint8_t* row;
  Payload load(uint g) thread {
    return KqTile<Codec>::load(row, g);
  }
  static METAL_FUNC void
  dequant(Payload p, uint, thread half4* v, const threadgroup uint8_t* luts) {
    KqTile<Codec>::dequant(p, v, luts);
  }
};

// kq_tl_column on a source.
template <typename T, typename Src>
METAL_FUNC float2 kq_st_column(
    thread Src& src,
    const device T* x,
    uint rows,
    int K,
    const threadgroup uint8_t* luts) {
  const uint steps = uint(K) / 32;
  float2 sums = float2(0.0f);
  typename Src::Payload w = src.load(0);
  for (uint g = 0; g < steps; ++g) {
    half4 v[8];
    Src::dequant(w, g, v, luts);
    if (g + 1 < steps) {
      w = src.load(g + 1);
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

// kq_tl_staged on a source.
template <typename T, typename Src>
METAL_FUNC void kq_st_staged(
    thread Src& src,
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
  typename Src::Payload w = src.load(0);
  for (uint s = 0; s < steps; ++s) {
    threadgroup half* buf = stage + (s & 1) * 1024 + lane * 32;
    half4 v[8];
    Src::dequant(w, s, v, luts);
#pragma unroll
    for (ushort i = 0; i < 8; ++i) {
      *((threadgroup half4*)(buf + 4 * i)) = v[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (s + 1 < steps) {
      w = src.load(s + 1);
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

// The lane's column sums of a tile's rows, into s[0 .. rows).
template <typename T, typename Src>
METAL_FUNC void kq_st_sums(
    thread Src& src,
    const device T* x,
    uint rows,
    uint col_rows,
    int K,
    uint lane,
    threadgroup half* stage,
    const threadgroup uint8_t* luts,
    thread float* s) {
  if (rows <= col_rows) {
    const float2 v = kq_st_column<T, Src>(src, x, rows, K, luts);
    s[0] = v.x;
    s[1] = v.y;
    return;
  }
  threadgroup float* park = (threadgroup float*)stage;
  kq_st_staged<T, Src>(src, x, K, lane, stage, park, luts);
  for (uint r = 0; r < KQ_TL_ROWS; ++r) {
    s[r] = park[r * 32 + lane];
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

// The tiled down on a store: kq_tl_gather_qmv_mix's buffers, grid and
// outputs. A tile of an expert that has a slot reads the planes p0 to p2;
// the shared tiles and the experts with no slot read their GGUF blocks in
// place (w, sw). slots is [E]. One dispatch serves the three: a second one
// for the in-place tiles costs a lane's whole walk of K again (67 us beside
// 163 on the GLU at 8 rows on M5 Max, against 20 in the same dispatch).
template <typename T, typename Codec, typename SCodec>
[[kernel]] void kq_st_down(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* sw [[buffer(1)]],
    const device T* hg [[buffer(2)]],
    const device uint32_t* tiles [[buffer(3)]],
    device T* y [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& NS [[buffer(9)]],
    const constant int& COLR [[buffer(10)]],
    const device uint32_t* slots [[buffer(11)]],
    const device uint8_t* p0 [[buffer(12)]],
    const device uint8_t* p1 [[buffer(13)]],
    const device uint8_t* p2 [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint4 td = *(const device uint4*)(tiles + 4 * tid.y);
  const uint rows = td.z & 0xFFu;
  if (rows == 0) {
    return;
  }
  const bool shared = (td.x & KQ_TL_SHARED) != 0;
  const uint expert = td.x & ~KQ_TL_SHARED;
  const uint slot = shared ? KQ_ST_ABSENT : slots[expert];
  const uint col_rows = min(uint(COLR), uint(KQ_TL_COL_ROWS));
  const uint col = tid.x * 64 + simd_gid * 32 + simd_lid;
  const bool live = col < uint(N);
  const uint c = live ? col : uint(N) - 1;
  threadgroup float stage_f[2048];
  threadgroup half* stage = (threadgroup half*)stage_f + simd_gid * 2048;
  const device T* x = hg + (int64_t)td.y * K;
  device T* yo = y + (int64_t)td.y * N + c;
  KQ_EXT_DECL_LUTS2(Codec, SCodec, kq_luts)
  float s[KQ_TL_ROWS];
  if (shared) {
    KQ_EXT_STAGE_INTO(SCodec, kq_luts)
    const int64_t rb = (int64_t)(K / SCodec::superblock) * SCodec::block_bytes;
    KqBlockSrc<SCodec> src;
    src.row = sw + ((int64_t)c * NS + expert) * rb;
    kq_st_sums<T, KqBlockSrc<SCodec>>(
        src, x, rows, col_rows, K, simd_lid, stage, kq_luts, s);
  } else if (slot != KQ_ST_ABSENT) {
    if (KqStoreFmt<Codec>::LUTS) {
      KQ_EXT_STAGE_INTO(Codec, kq_luts)
    }
    KqStoreSrc<Codec> src;
    src.begin(p0, p1, p2, slot, c, N, K);
    kq_st_sums<T, KqStoreSrc<Codec>>(
        src, x, rows, col_rows, K, simd_lid, stage, kq_luts, s);
  } else {
    KQ_EXT_STAGE_INTO(Codec, kq_luts)
    const int64_t rb = (int64_t)(K / Codec::superblock) * Codec::block_bytes;
    KqBlockSrc<Codec> src;
    src.row = w + ((int64_t)expert * N + c) * rb;
    kq_st_sums<T, KqBlockSrc<Codec>>(
        src, x, rows, col_rows, K, simd_lid, stage, kq_luts, s);
  }
  if (live) {
    for (uint r = 0; r < rows; ++r) {
      yo[(int64_t)r * N] = static_cast<T>(s[r]);
    }
  }
}

// The tiled GLU on a store: out[pair] = silu(gate) * up of the pair's row,
// out [T * S, N] in pair order (t * S + slot). xg and tiles are
// kq_moe_store_plan's. gw, uw are the routed stacks [E, N, row bytes], sgw
// and suw the shared ones [N, row bytes] (one slot), g0 to g2 and u0 to u2
// the planes. Grid (ceil(N / 32), tiles), threads (32, 2): simdgroup 0 runs
// a tile's gate, simdgroup 1 its up, on the same 32 columns.
template <typename T, typename Codec, typename SCodec>
[[kernel]] void kq_st_glu(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device uint8_t* sgw [[buffer(2)]],
    const device uint8_t* suw [[buffer(3)]],
    const device T* xg [[buffer(4)]],
    const device uint32_t* tiles [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& COLR [[buffer(10)]],
    const device uint32_t* slots [[buffer(11)]],
    const device uint8_t* g0 [[buffer(12)]],
    const device uint8_t* g1 [[buffer(13)]],
    const device uint8_t* g2 [[buffer(14)]],
    const device uint8_t* u0 [[buffer(15)]],
    const device uint8_t* u1 [[buffer(16)]],
    const device uint8_t* u2 [[buffer(17)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint ti = tid.y;
  const uint4 td = *(const device uint4*)(tiles + 4 * ti);
  const uint rows = td.z & 0xFFu;
  if (rows == 0) {
    return;
  }
  const bool shared = (td.x & KQ_TL_SHARED) != 0;
  const uint expert = td.x & ~KQ_TL_SHARED;
  const uint slot = shared ? KQ_ST_ABSENT : slots[expert];
  const uint col_rows = min(uint(COLR), uint(KQ_TL_COL_ROWS));
  const bool up = simd_gid != 0;
  const uint col = tid.x * 32 + simd_lid;
  const bool live = col < uint(N);
  const uint c = live ? col : uint(N) - 1;
  threadgroup float stage_f[2048 + 32 * KQ_TL_ROWS];
  threadgroup half* stage = (threadgroup half*)stage_f + simd_gid * 2048;
  threadgroup float* gate = stage_f + 2048;
  const device T* x = xg + (int64_t)td.y * K;
  KQ_EXT_DECL_LUTS2(Codec, SCodec, kq_luts)
  float s[KQ_TL_ROWS];
  if (shared) {
    KQ_EXT_STAGE_INTO(SCodec, kq_luts)
    const int64_t rb = (int64_t)(K / SCodec::superblock) * SCodec::block_bytes;
    KqBlockSrc<SCodec> src;
    src.row = (up ? suw : sgw) + (int64_t)c * rb;
    kq_st_sums<T, KqBlockSrc<SCodec>>(
        src, x, rows, col_rows, K, simd_lid, stage, kq_luts, s);
  } else if (slot != KQ_ST_ABSENT) {
    if (KqStoreFmt<Codec>::LUTS) {
      KQ_EXT_STAGE_INTO(Codec, kq_luts)
    }
    KqStoreSrc<Codec> src;
    if (up) {
      src.begin(u0, u1, u2, slot, c, N, K);
    } else {
      src.begin(g0, g1, g2, slot, c, N, K);
    }
    kq_st_sums<T, KqStoreSrc<Codec>>(
        src, x, rows, col_rows, K, simd_lid, stage, kq_luts, s);
  } else {
    KQ_EXT_STAGE_INTO(Codec, kq_luts)
    const int64_t rb = (int64_t)(K / Codec::superblock) * Codec::block_bytes;
    KqBlockSrc<Codec> src;
    src.row = (up ? uw : gw) + ((int64_t)expert * N + c) * rb;
    kq_st_sums<T, KqBlockSrc<Codec>>(
        src, x, rows, col_rows, K, simd_lid, stage, kq_luts, s);
  }
  if (!up) {
    for (uint r = 0; r < rows; ++r) {
      gate[r * 32 + simd_lid] = s[r];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (up && live) {
    for (uint r = 0; r < rows; ++r) {
      const float g = gate[r * 32 + simd_lid];
      const float sig = 1.0f / (1.0f + metal::exp(-g));
      const uint pair = tiles[4 * (ti + r) + 3];
      out[(int64_t)pair * N + c] = static_cast<T>((g * sig) * s[r]);
    }
  }
}

// kq_moe_tile_plan with XR: the activation row of a pair is x[t] of x
// [T, K] (XR != 0) or h[t, slot] of h [T, S, K].
template <typename T>
[[kernel]] void kq_moe_store_plan(
    const device uint32_t* indices [[buffer(0)]],
    const device T* h [[buffer(1)]],
    device T* hg [[buffer(2)]],
    device uint32_t* tiles [[buffer(3)]],
    device uint32_t* place [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& S [[buffer(6)]],
    const constant int& NS [[buffer(7)]],
    const constant int& T_ [[buffer(8)]],
    const constant int& XR [[buffer(9)]],
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
  const device T* from = h + (int64_t)(XR != 0 ? t : pair) * K;
  device T* to = hg + (int64_t)rank * K;
  for (int i = int(lane) * 8; i < K; i += 256) {
    *(device vec<T, 8>*)(to + i) = *(const device vec<T, 8>*)(from + i);
  }
}

// One plane of the listed experts: record (slot, n, u) of R bytes from GGUF
// row bytes w [E, N, ROWB]. A row's block of BB bytes holds GPB records of
// the plane from byte OFF. experts[i] goes to slot S0 + i. Grid (N, U,
// experts). Not a template, so only one metal file may hold it: a second
// file that includes this header defines KQ_MOE_STORE_NO_PACK.
#ifndef KQ_MOE_STORE_NO_PACK
[[kernel]] void kq_moe_store_pack(
    const device uint8_t* w [[buffer(0)]],
    device uint8_t* dst [[buffer(1)]],
    const device uint32_t* experts [[buffer(2)]],
    const constant int& N [[buffer(3)]],
    const constant int& U [[buffer(4)]],
    const constant int& R [[buffer(5)]],
    const constant int& BB [[buffer(6)]],
    const constant int& OFF [[buffer(7)]],
    const constant int& GPB [[buffer(8)]],
    const constant int& ROWB [[buffer(9)]],
    const constant int& S0 [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint u = gid.y;
  const ulong e = experts[gid.z];
  const ulong slot = ulong(S0) + gid.z;
  const device uint8_t* from = w + (e * uint(N) + n) * uint(ROWB) +
      ulong(u / uint(GPB)) * uint(BB) + uint(OFF) + (u % uint(GPB)) * uint(R);
  device uint8_t* to = dst +
      ((slot * (uint(N) / KQ_ST_TILE) + n / KQ_ST_TILE) * uint(U) * KQ_ST_TILE +
       ulong(u) * KQ_ST_TILE + n % KQ_ST_TILE) *
          uint(R);
  for (int i = 0; i < R; ++i) {
    to[i] = from[i];
  }
}
#endif

// clang-format off
#define instantiate_kq_st_pair(op, name, traits, straits)                     \
  instantiate_kernel(                                                         \
      "kq_store_" #op "_" name "_bfloat16_t", kq_st_##op, bfloat16_t,         \
      traits, straits)                                                        \
  instantiate_kernel(                                                         \
      "kq_store_" #op "_" name "_float16_t", kq_st_##op, float16_t,           \
      traits, straits)

// A store codec with the shared codecs of the tile kernels: the routed
// codec, q5_k, q6_k, q8_0.
#define instantiate_kq_st_all(codec, traits)                                  \
  instantiate_kq_st_pair(down, #codec "_sx_" #codec, traits, traits)          \
  instantiate_kq_st_pair(down, #codec "_sx_q5_k", traits, KqQ5_KExt)          \
  instantiate_kq_st_pair(down, #codec "_sx_q6_k", traits, KqQ6_KExt)          \
  instantiate_kq_st_pair(down, #codec "_sx_q8_0", traits, KqQ8_0Ext)          \
  instantiate_kq_st_pair(glu, #codec "_sx_" #codec, traits, traits)           \
  instantiate_kq_st_pair(glu, #codec "_sx_q5_k", traits, KqQ5_KExt)           \
  instantiate_kq_st_pair(glu, #codec "_sx_q6_k", traits, KqQ6_KExt)           \
  instantiate_kq_st_pair(glu, #codec "_sx_q8_0", traits, KqQ8_0Ext)
// The three codecs that are themselves the shared codecs.
#define instantiate_kq_st_wide(codec, traits)                                 \
  instantiate_kq_st_pair(down, #codec "_sx_q5_k", traits, KqQ5_KExt)          \
  instantiate_kq_st_pair(down, #codec "_sx_q6_k", traits, KqQ6_KExt)          \
  instantiate_kq_st_pair(down, #codec "_sx_q8_0", traits, KqQ8_0Ext)          \
  instantiate_kq_st_pair(glu, #codec "_sx_q5_k", traits, KqQ5_KExt)           \
  instantiate_kq_st_pair(glu, #codec "_sx_q6_k", traits, KqQ6_KExt)           \
  instantiate_kq_st_pair(glu, #codec "_sx_q8_0", traits, KqQ8_0Ext)
// clang-format on

// ---------------------------------------------------------------------------
// Pair-grid forms (kq_st_glu_pg, kq_st_down_pg): the tiles of the kernels
// above with no plan dispatch ahead and a weight row split along K over the
// simdgroups of a threadgroup.
//   Grid (T * S, N-tiles): a threadgroup is a (row, slot) pair's share of
//   the output columns, the pair along x. The GPU takes threadgroups in grid
//   order, and with the column tiles along x (runs of N-tiles threadgroups
//   that all return, for a pair that leads no tile) the same kernel ran 8
//   to 11% slower at 2 to 8 rows (M5 Max, iq3_xxs GLU).
//   It finds the tile its pair leads from the indices
//   (kq_grp_run: the pairs on one expert in row-major order, KQ_TL_ROWS a
//   tile, the same tiles as kq_moe_tile_plan) and returns when the pair
//   leads none. A shared slot runs as tiles of consecutive rows.
//   The activation rows of a tile are read where they are: a column tile
//   (COLR rows or fewer) through its rows' addresses, a staged tile by
//   staging each step's 8 x 32 activations beside the weights.
//   A lane walks the groups [j G / q, (j + 1) G / q) of its weight row, j
//   the simdgroup's place among the q that share its columns: q = QR for a
//   routed expert (stored or read in place, so a row's bits do not depend
//   on which experts the store holds) and QH for the shared expert. The
//   parts are added in order of j. At q = 1 the sums are those of the tile
//   kernels above, bit for bit.
//   Outputs are in pair order (t * S + slot).
// Why K is split: a lane's walk of all of K is the latency floor of a tile
// kernel, and the shared expert's tile in a slow codec (q6_k, K 2560) sets
// the time of the whole dispatch at 1 to 4 rows.
// Template int SG, the most simdgroups a threadgroup may have (2, 4, 8),
// sizes the threadgroup memory (2.5 KB a simdgroup), which costs time by
// its size alone (M5 Max, 8 rows of distinct experts: the same sums ran
// 13% slower with 20 KB a threadgroup than with 5 KB).
// A column tile may have up to KQ_ST_COL_ROWS rows (COLR; 2 is the tile
// kernels' and the measured best, staged tiles win from 3 rows).
// ---------------------------------------------------------------------------
#define KQ_ST_COL_ROWS 4

// kq_grp_run with the tile size cap: the tile pair p leads, its size or 0,
// its pairs in m (unused entries p).
METAL_FUNC int kq_st_run(
    const device uint32_t* indices,
    int P,
    int p,
    uint lane,
    int cap,
    thread int* m) {
  const uint e = indices[p];
  int rank = 0;
  for (int q0 = 0; q0 < p; q0 += 32) {
    const int q = q0 + int(lane);
    rank += int(
        popcount(static_cast<ulong>(simd_ballot(q < p && indices[q] == e))));
  }
  if (rank % cap != 0) {
    return 0;
  }
  for (int i = 0; i < KQ_TL_ROWS; ++i) {
    m[i] = p;
  }
  int n = 1;
  for (int q0 = p + 1; q0 < P && n < cap; q0 += 32) {
    const int q = q0 + int(lane);
    ulong b = static_cast<ulong>(simd_ballot(q < P && indices[q] == e));
    for (; b != 0 && n < cap; ++n) {
      m[n] = q0 + int(ctz(b));
      b &= b - 1;
    }
  }
  return n;
}

// kq_st_column over the groups [g0, g1) with the rows' addresses, up to
// KQ_ST_COL_ROWS rows.
template <typename T, typename Src>
METAL_FUNC float4 kq_st_column_r(
    thread Src& src,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    const device T* x3,
    uint rows,
    uint g0,
    uint g1,
    const threadgroup uint8_t* luts) {
  float4 sums = float4(0.0f);
  typename Src::Payload w = src.load(g0);
  for (uint g = g0; g < g1; ++g) {
    half4 v[8];
    Src::dequant(w, g, v, luts);
    if (g + 1 < g1) {
      w = src.load(g + 1);
    }
#pragma unroll
    for (ushort r = 0; r < KQ_ST_COL_ROWS; ++r) {
      if (r >= rows) {
        break;
      }
      const device T* xr =
          (r == 0 ? x0 : (r == 1 ? x1 : (r == 2 ? x2 : x3))) + g * 32;
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

// kq_st_staged over the groups [g0, g1) with the activations staged: lane l
// copies 8 values of activation row l / 4 a step (xl is that row from its
// value (l % 4) * 8) into xs [8][32], and the 8-row matmul2d reads xs.
template <typename T, typename Src>
METAL_FUNC void kq_st_staged_r(
    thread Src& src,
    const device T* xl,
    uint g0,
    uint g1,
    uint lane,
    threadgroup half* stage,
    threadgroup T* xs,
    threadgroup float* park,
    const threadgroup uint8_t* luts) {
  using namespace mpp::tensor_ops;
  constexpr auto desc = matmul2d_descriptor(
      KQ_TL_ROWS,
      32,
      32,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt(
      stage, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
  tensor<threadgroup T, dextents<int, 2>, tensor_inline> at(
      xs, dextents<int, 2>{32, KQ_TL_ROWS}, array<int, 2>{1, 32});
  auto b0 = bt.slice<32, 32>(0, 0);
  auto a0 = at.template slice<32, KQ_TL_ROWS>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(a0),
      decltype(b0),
      float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  threadgroup half* buf = stage + lane * 32;
  threadgroup T* xb = xs + (lane / 4) * 32 + (lane % 4) * 8;
  typename Src::Payload w = src.load(g0);
  for (uint s = g0; s < g1; ++s) {
    half4 v[8];
    Src::dequant(w, s, v, luts);
#pragma unroll
    for (ushort i = 0; i < 8; ++i) {
      *((threadgroup half4*)(buf + 4 * i)) = v[i];
    }
    *((threadgroup vec<T, 8>*)xb) = *(const device vec<T, 8>*)(xl + s * 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (s + 1 < g1) {
      w = src.load(s + 1);
    }
    op.run(a0, b0, acc);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
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

// The lane's part of a tile's column sums over [g0, g1), parked at
// park[r * 32 + lane]. xr holds the tile's KQ_TL_ROWS activation row
// addresses (rows past the tile's repeat one of them).
#define KQ_ST_PG_PART(SrcT, src)                                \
  if (uint(cnt) <= uint(col_rows)) {                            \
    const float4 v4 = kq_st_column_r<T, SrcT>(                  \
        src, xr[0], xr[1], xr[2], xr[3], cnt, g0, g1, kq_luts); \
    for (int r = 0; r < cnt; ++r) {                             \
      park[r * 32 + simd_lid] = v4[r];                          \
    }                                                           \
  } else {                                                      \
    kq_st_staged_r<T, SrcT>(                                    \
        src,                                                    \
        xr[simd_lid / 4] + (simd_lid % 4) * 8,                  \
        g0,                                                     \
        g1,                                                     \
        simd_lid,                                               \
        stage,                                                  \
        xs,                                                     \
        park,                                                   \
        kq_luts);                                               \
  }

// Threadgroup memory of a pair-grid kernel: PS halfs a simdgroup (the
// staged weights, then the parked sums as PS / 2 floats) and its staged
// activations.
#define KQ_ST_PG_MEM                                       \
  constexpr int PS = 1024;                                 \
  constexpr int PF = PS / 2;                               \
  constexpr int cap = KQ_TL_ROWS;                          \
  threadgroup half stage_h[SG * PS];                       \
  threadgroup T xs_t[SG * 32 * KQ_TL_ROWS];                \
  threadgroup half* stage = stage_h + simd_gid * PS;       \
  threadgroup T* xs = xs_t + simd_gid * (32 * KQ_TL_ROWS); \
  threadgroup float* park = (threadgroup float*)stage;

// The tile the pair (row pt, slot ps) of the grid leads: cnt its
// pairs or 0, member i on activation row mr[i] and output pair mo[i] of the
// [T, S] grid (entries past cnt repeat the first).
#define KQ_ST_PG_TILE                                                  \
  const int pt = int(tid.x) / SLOTS;                                   \
  const int ps = int(tid.x) - pt * SLOTS;                              \
  const uint ct = tid.y;                                               \
  const int S_ = SLOTS;                                                \
  const int n_route = S_ - NS;                                         \
  const int T_ = (int(tpg.x) / SLOTS);                                 \
  const bool shared = ps >= n_route;                                   \
  int cnt;                                                             \
  int mr[KQ_TL_ROWS];                                                  \
  int mo[KQ_TL_ROWS];                                                  \
  if (shared) {                                                        \
    cnt = pt % cap == 0 ? min(cap, T_ - pt) : 0;                       \
    for (int i = 0; i < KQ_TL_ROWS; ++i) {                             \
      mr[i] = min(pt + i, T_ - 1);                                     \
      mo[i] = mr[i] * S_ + ps;                                         \
    }                                                                  \
  } else {                                                             \
    int mem[KQ_TL_ROWS];                                               \
    cnt = kq_st_run(                                                   \
        indices, T_ * n_route, pt * n_route + ps, simd_lid, cap, mem); \
    for (int i = 0; i < KQ_TL_ROWS; ++i) {                             \
      const int m = cnt == 0 ? 0 : mem[i];                             \
      mr[i] = m / n_route;                                             \
      mo[i] = mr[i] * S_ + (m - mr[i] * n_route);                      \
    }                                                                  \
  }                                                                    \
  if (cnt == 0) {                                                      \
    return;                                                            \
  }

// The GLU: out[t, slot] = silu(gate) * up, out [T, S, N]. x is [T, K],
// indices [T, S - 1]; the weights, slots and planes are kq_st_glu's. Grid
// (T * S, ceil(N / 32)), threads (32, 2 q): simdgroups 2 j and 2 j + 1 run
// the gate and the up of part j of the threadgroup's 32 columns.
template <typename T, typename Codec, typename SCodec, typename SQ, int SG>
[[kernel]] void kq_st_glu_pg(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device uint8_t* sgw [[buffer(2)]],
    const device uint8_t* suw [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uint32_t* indices [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& QR [[buffer(9)]],
    const constant int& COLR [[buffer(10)]],
    const device uint32_t* slots [[buffer(11)]],
    const device uint8_t* g0p [[buffer(12)]],
    const device uint8_t* g1p [[buffer(13)]],
    const device uint8_t* g2p [[buffer(14)]],
    const device uint8_t* u0p [[buffer(15)]],
    const device uint8_t* u1p [[buffer(16)]],
    const device uint8_t* u2p [[buffer(17)]],
    const constant int& QH [[buffer(18)]],
    const constant int& SLOTS [[buffer(19)]],
    const constant int& SHQF [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int NS = 1;
  KQ_ST_PG_MEM
  KQ_ST_PG_TILE
  if (shared && SHQF != 0) {
    // The shared expert on its codec's qf row loops: 32 lanes share a
    // row block's walk of K, one activation row a pass, a simdgroup takes
    // every tptg.y-th block of the threadgroup's 32 columns.
    constexpr int RPS = SQ::rps;
    const uint nb = 32 / RPS;
    for (uint b = simd_gid; b < nb; b += uint(tptg.y)) {
      const uint blk = ct * nb + b;
      if (int(blk) * RPS >= N) {
        break;
      }
      for (int r = 0; r < cnt; ++r) {
        const device T* xp = x + (int64_t)mr[r] * K;
        float g[RPS] = {0};
        float u[RPS] = {0};
        SQ::template rows<T, RPS>(sgw, xp, K, N, blk / 2, blk % 2, simd_lid, g);
        SQ::template rows<T, RPS>(suw, xp, K, N, blk / 2, blk % 2, simd_lid, u);
        if (simd_lid == 0) {
          device T* o = out + (int64_t)mo[r] * N + blk * RPS;
          for (int i = 0; i < RPS; ++i) {
            const float sig = 1.0f / (1.0f + metal::exp(-g[i]));
            o[i] = static_cast<T>((g[i] * sig) * u[i]);
          }
        }
      }
    }
    return;
  }
  const uint expert = shared ? 0u : indices[pt * n_route + ps];
  const uint slot = shared ? KQ_ST_ABSENT : slots[expert];
  const uint col_rows = min(uint(COLR), uint(KQ_ST_COL_ROWS));
  const bool up = (simd_gid & 1u) != 0;
  const uint j = simd_gid / 2;
  const uint G = uint(K) / 32;
  const uint q = min(min(uint(shared ? QH : QR), uint(tptg.y) / 2), G);
  const uint g0 = j * G / q;
  const uint g1 = (j + 1) * G / q;
  const uint col = ct * 32 + simd_lid;
  const bool live = col < uint(N);
  const uint c = live ? col : uint(N) - 1;
  const device T* xr[KQ_TL_ROWS];
  for (int i = 0; i < KQ_TL_ROWS; ++i) {
    xr[i] = x + (int64_t)mr[i] * K;
  }
  KQ_EXT_DECL_LUTS2(Codec, SCodec, kq_luts)
  if (shared) {
    KQ_EXT_STAGE_INTO(SCodec, kq_luts)
    if (j < q) {
      const int64_t rb =
          (int64_t)(K / SCodec::superblock) * SCodec::block_bytes;
      KqBlockSrc<SCodec> src;
      src.row = (up ? suw : sgw) + (int64_t)c * rb;
      KQ_ST_PG_PART(KqBlockSrc<SCodec>, src)
    }
  } else if (slot != KQ_ST_ABSENT) {
    if (j < q) {
      KqStoreSrc<Codec> src;
      if (up) {
        src.begin(u0p, u1p, u2p, slot, c, N, K);
      } else {
        src.begin(g0p, g1p, g2p, slot, c, N, K);
      }
      KQ_ST_PG_PART(KqStoreSrc<Codec>, src)
    }
  } else {
    KQ_EXT_STAGE_INTO(Codec, kq_luts)
    if (j < q) {
      const int64_t rb = (int64_t)(K / Codec::superblock) * Codec::block_bytes;
      KqBlockSrc<Codec> src;
      src.row = (up ? uw : gw) + ((int64_t)expert * N + c) * rb;
      KQ_ST_PG_PART(KqBlockSrc<Codec>, src)
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0 && live) {
    const threadgroup float* parts = (const threadgroup float*)stage_h;
    for (int r = 0; r < cnt; ++r) {
      float g = parts[r * 32 + simd_lid];
      float u = parts[PF + r * 32 + simd_lid];
      for (uint p = 1; p < q; ++p) {
        g += parts[(2 * p) * PF + r * 32 + simd_lid];
        u += parts[(2 * p + 1) * PF + r * 32 + simd_lid];
      }
      const float sig = 1.0f / (1.0f + metal::exp(-g));
      out[(int64_t)mo[r] * N + c] = static_cast<T>((g * sig) * u);
    }
  }
}

// The down's row sums of every pair, y [T * S, N] in pair order. h is
// [T, S, K], indices [T, S - NS]; the weights, slots and planes are
// kq_st_down's. Grid (T * S, ceil(N / (32 CW))), threads (32, CW q): a
// threadgroup takes 32 CW columns, simdgroup cw * q + j part j of its
// columns [32 cw, 32 cw + 32).
template <typename T, typename Codec, typename SCodec, typename SQ, int SG>
[[kernel]] void kq_st_down_pg(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* sw [[buffer(1)]],
    const device T* h [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    const constant int& SLOTS [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& QR [[buffer(8)]],
    const constant int& NS [[buffer(9)]],
    const constant int& COLR [[buffer(10)]],
    const device uint32_t* slots [[buffer(11)]],
    const device uint8_t* p0 [[buffer(12)]],
    const device uint8_t* p1 [[buffer(13)]],
    const device uint8_t* p2 [[buffer(14)]],
    const constant int& QH [[buffer(18)]],
    const constant int& CW [[buffer(19)]],
    const constant int& SHQF [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  KQ_ST_PG_MEM
  KQ_ST_PG_TILE
  if (shared && SHQF != 0) {
    // The shared expert on its codec's qf row loops (kq_st_glu_pg); the
    // host keeps NS == 1 here.
    constexpr int RPS = SQ::rps;
    const uint nb = uint(32 * CW) / RPS;
    for (uint b = simd_gid; b < nb; b += uint(tptg.y)) {
      const uint blk = ct * nb + b;
      if (int(blk) * RPS >= N) {
        break;
      }
      for (int r = 0; r < cnt; ++r) {
        const device T* xp = h + (int64_t)mo[r] * K;
        float sums[RPS] = {0};
        SQ::template rows<T, RPS>(
            sw, xp, K, N, blk / 2, blk % 2, simd_lid, sums);
        if (simd_lid == 0) {
          device T* o = y + (int64_t)mo[r] * N + blk * RPS;
          for (int i = 0; i < RPS; ++i) {
            o[i] = static_cast<T>(sums[i]);
          }
        }
      }
    }
    return;
  }
  const uint expert = shared ? uint(ps - n_route) : indices[pt * n_route + ps];
  const uint slot = shared ? KQ_ST_ABSENT : slots[expert];
  const uint col_rows = min(uint(COLR), uint(KQ_ST_COL_ROWS));
  const uint G = uint(K) / 32;
  const uint qw = uint(tptg.y) / uint(CW);
  const uint cw = simd_gid / qw;
  const uint j = simd_gid - cw * qw;
  const uint q = min(min(uint(shared ? QH : QR), qw), G);
  const uint g0 = j * G / q;
  const uint g1 = (j + 1) * G / q;
  const uint col = (ct * uint(CW) + cw) * 32 + simd_lid;
  const bool live = col < uint(N);
  const uint c = live ? col : uint(N) - 1;
  const device T* xr[KQ_TL_ROWS];
  for (int i = 0; i < KQ_TL_ROWS; ++i) {
    xr[i] = h + (int64_t)mo[i] * K;
  }
  KQ_EXT_DECL_LUTS2(Codec, SCodec, kq_luts)
  if (shared) {
    KQ_EXT_STAGE_INTO(SCodec, kq_luts)
    if (j < q) {
      const int64_t rb =
          (int64_t)(K / SCodec::superblock) * SCodec::block_bytes;
      KqBlockSrc<SCodec> src;
      src.row = sw + ((int64_t)c * NS + expert) * rb;
      KQ_ST_PG_PART(KqBlockSrc<SCodec>, src)
    }
  } else if (slot != KQ_ST_ABSENT) {
    if (j < q) {
      KqStoreSrc<Codec> src;
      src.begin(p0, p1, p2, slot, c, N, K);
      KQ_ST_PG_PART(KqStoreSrc<Codec>, src)
    }
  } else {
    KQ_EXT_STAGE_INTO(Codec, kq_luts)
    if (j < q) {
      const int64_t rb = (int64_t)(K / Codec::superblock) * Codec::block_bytes;
      KqBlockSrc<Codec> src;
      src.row = w + ((int64_t)expert * N + c) * rb;
      KQ_ST_PG_PART(KqBlockSrc<Codec>, src)
    }
  }
  if (q > 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (j == 0 && live) {
    const threadgroup float* parts =
        (const threadgroup float*)(stage_h + (cw * qw) * PS);
    for (int r = 0; r < cnt; ++r) {
      float s = parts[r * 32 + simd_lid];
      for (uint p = 1; p < q; ++p) {
        s += parts[p * PF + r * 32 + simd_lid];
      }
      y[(int64_t)mo[r] * N + c] = static_cast<T>(s);
    }
  }
}

// out[t] = the sum over slots of score * y[t * S + slot], y the pair-order
// row sums of kq_st_down_pg; kq_moe_tile_combine's sums in its order.
template <typename T>
[[kernel]] void kq_moe_store_combine(
    const device T* y [[buffer(0)]],
    const device float* scores [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant int& N [[buffer(3)]],
    const constant int& S [[buffer(4)]],
    const constant int& SC [[buffer(5)]],
    const constant int& NS [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int n = int(gid.x) * 4;
  const int t = int(gid.z);
  const int n_route = S - NS;
  float4 acc = float4(0.0f);
  for (int s = 0; s < n_route; ++s) {
    acc += scores[t * SC + s] *
        float4(*(const device vec<T, 4>*)(y + ((int64_t)t * S + s) * N + n));
  }
  for (int s = n_route; s < S; ++s) {
    acc += (SC == S ? scores[t * SC + s] : 1.0f) *
        float4(*(const device vec<T, 4>*)(y + ((int64_t)t * S + s) * N + n));
  }
  *(device vec<T, 4>*)(out + (int64_t)t * N + n) = vec<T, 4>(acc);
}

// ---------------------------------------------------------------------------
// The in-place pair form on the planes (iq3_xxs): the sums of the in-place
// Ext kernels in their order, so a row's bits are the in-place op's.
// A simdgroup is 32 / NX output columns by NX lanes along K; lane tx takes
// the 16-weight chunks tx, tx + NX, ... of its column, the lanes reduce
// with kq_ext_reduce. A threadgroup is one (row, slot) pair's share of the
// columns; the pair that owns its expert (kq_dd_partner) dots the next
// pair on that expert against the same weights. C32 (one row): 32-weight
// groups, the sums of the in-place "_c32" kernel. The shared slot is the
// in-place kernel's own code on the GGUF bytes, and so is an expert the
// store does not hold. No plan, no threadgroup
// memory but the tables.
// Dispatch: grid (N / (sg * 32 / NX), S, T), threads (32, sg).
// ---------------------------------------------------------------------------
typedef KqStoreSrc<KqIq3_xxsExt> KqStIq3Src;

// p.qs[0]: the 4 index bytes of the group's half hi.
METAL_FUNC void kq_stp_iq3_chunk16s(
    KqStIq3Src::Payload p,
    short hi,
    thread float4x4& reg,
    const threadgroup uint8_t* luts,
    thread float& scale) {
  const threadgroup uint32_t* grid =
      reinterpret_cast<const threadgroup uint32_t*>(luts);
  const threadgroup uint8_t* ksigns = luts + 1024;
  const uint aux32 = p.gas;
  scale = float(p.d) * (0.5f + float(aux32 >> 28)) * 0.5f;
  const uint q = p.qs[0];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int l = 2 * hi + t;
    const uint signs = ksigns[(aux32 >> (7 * l)) & 127];
    const uint w = q >> (16 * t);
    reg[2 * t] = kq_grid_sign4(kq_grid_u8x4(grid[w & 0xFF]), signs);
    reg[2 * t + 1] =
        kq_grid_sign4(kq_grid_u8x4(grid[(w >> 8) & 0xFF]), signs >> 4);
  }
}

METAL_FUNC float kq_stp_iq3_group32s(
    KqStIq3Src::Payload p,
    thread float4x4& r0,
    thread float4x4& r1,
    const threadgroup uint8_t* luts) {
  const threadgroup uint32_t* grid =
      reinterpret_cast<const threadgroup uint32_t*>(luts);
  const threadgroup uint8_t* ksigns = luts + 1024;
  const uint aux = p.gas;
  kq_c32_u32_pair(grid, p.qs[0] & 0xFFFF, ksigns[aux & 127], r0[0], r0[1]);
  kq_c32_u32_pair(grid, p.qs[0] >> 16, ksigns[(aux >> 7) & 127], r0[2], r0[3]);
  kq_c32_u32_pair(
      grid, p.qs[1] & 0xFFFF, ksigns[(aux >> 14) & 127], r1[0], r1[1]);
  kq_c32_u32_pair(grid, p.qs[1] >> 16, ksigns[(aux >> 21) & 127], r1[2], r1[3]);
  return float(p.d) * (0.5f + float(aux >> 28)) * 0.5f;
}

// (gate, up) of row x0 in .xy and, with two, of row x1 in .zw: a lane's
// partial sums (kq_ext_glu_row_partial, kq_ext_glu_row_partial_pair).
template <typename T, int NX, bool C32>
METAL_FUNC float4 kq_stp_iq3_glu_rows(
    KqStIq3Src gs,
    KqStIq3Src us,
    const device T* x0,
    const device T* x1,
    bool two,
    int K,
    short tx,
    const threadgroup uint8_t* luts) {
  float4 acc = float4(0.0f);
  if (C32) {
    for (int ig = tx; 32 * ig < K; ig += NX) {
      const device T* xp = x0 + ig * 32;
      float4 a[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        a[i] = float4(*(const device vec<T, 4>*)(xp + 4 * i));
      }
      float4x4 l0, l1;
      float sc = kq_stp_iq3_group32s(gs.load(uint(ig)), l0, l1, luts);
      acc.x += sc *
              (dot(l0[0], a[0]) + dot(l0[1], a[1]) + dot(l0[2], a[2]) +
               dot(l0[3], a[3])) +
          sc *
              (dot(l1[0], a[4]) + dot(l1[1], a[5]) + dot(l1[2], a[6]) +
               dot(l1[3], a[7]));
      sc = kq_stp_iq3_group32s(us.load(uint(ig)), l0, l1, luts);
      acc.y += sc *
              (dot(l0[0], a[0]) + dot(l0[1], a[1]) + dot(l0[2], a[2]) +
               dot(l0[3], a[3])) +
          sc *
              (dot(l1[0], a[4]) + dot(l1[1], a[5]) + dot(l1[2], a[6]) +
               dot(l1[3], a[7]));
    }
    return acc;
  }
  // Lane tx walks its half (tx & 1) of group tx / 2 + 8 * it at NX 16, and
  // the fields of consecutive steps sit a fixed stride apart.
  const short hi = short(tx & 1);
  const uint gstep = uint(NX) / 2;
  const ulong s0 = ulong(gstep) * (KQ_ST_TILE * 8);
  const ulong s1 = ulong(gstep) * (KQ_ST_TILE * 4);
  const ulong s2 = ulong(gstep / 8) * (KQ_ST_TILE * 2);
  const ulong o0 = ulong(tx / 2) * (KQ_ST_TILE * 8) + ulong(hi) * 4;
  const ulong o1 = ulong(tx / 2) * (KQ_ST_TILE * 4);
  const device uint8_t* gq = gs.q0 + o0;
  const device uint8_t* gg = gs.q1 + o1;
  const device uint8_t* gd = gs.q2;
  const device uint8_t* uq = us.q0 + o0;
  const device uint8_t* ug = us.q1 + o1;
  const device uint8_t* ud = us.q2;
  for (int ich = tx; 16 * ich < K; ich += NX) {
    const device T* xp = x0 + ich * 16;
    const float4 a0 = float4(*(const device vec<T, 4>*)(xp + 0));
    const float4 a1 = float4(*(const device vec<T, 4>*)(xp + 4));
    const float4 a2 = float4(*(const device vec<T, 4>*)(xp + 8));
    const float4 a3 = float4(*(const device vec<T, 4>*)(xp + 12));
    const device T* yp = x1 + ich * 16;
    float4x4 lw;
    float sc;
    KqStIq3Src::Payload pl;
    pl.qs = uint2(*(const device uint*)gq);
    pl.gas = *(const device uint*)gg;
    pl.d = *(const device half*)gd;
    kq_stp_iq3_chunk16s(pl, hi, lw, luts, sc);
    acc.x += sc *
        (dot(lw[0], a0) + dot(lw[1], a1) + dot(lw[2], a2) + dot(lw[3], a3));
    if (two) {
      acc.z += sc *
          (dot(lw[0], float4(*(const device vec<T, 4>*)(yp + 0))) +
           dot(lw[1], float4(*(const device vec<T, 4>*)(yp + 4))) +
           dot(lw[2], float4(*(const device vec<T, 4>*)(yp + 8))) +
           dot(lw[3], float4(*(const device vec<T, 4>*)(yp + 12))));
    }
    pl.qs = uint2(*(const device uint*)uq);
    pl.gas = *(const device uint*)ug;
    pl.d = *(const device half*)ud;
    kq_stp_iq3_chunk16s(pl, hi, lw, luts, sc);
    acc.y += sc *
        (dot(lw[0], a0) + dot(lw[1], a1) + dot(lw[2], a2) + dot(lw[3], a3));
    if (two) {
      acc.w += sc *
          (dot(lw[0], float4(*(const device vec<T, 4>*)(yp + 0))) +
           dot(lw[1], float4(*(const device vec<T, 4>*)(yp + 4))) +
           dot(lw[2], float4(*(const device vec<T, 4>*)(yp + 8))) +
           dot(lw[3], float4(*(const device vec<T, 4>*)(yp + 12))));
    }
    gq += s0;
    gg += s1;
    gd += s2;
    uq += s0;
    ug += s1;
    ud += s2;
  }
  return acc;
}

template <typename T, typename SCodec, int NX, bool C32>
[[kernel]] void kq_st_glu_dd_iq3(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device uint8_t* sgw [[buffer(2)]],
    const device uint8_t* suw [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uint32_t* indices [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const device uint32_t* slots [[buffer(11)]],
    const device uint8_t* g0p [[buffer(12)]],
    const device uint8_t* g1p [[buffer(13)]],
    const device uint8_t* g2p [[buffer(14)]],
    const device uint8_t* u0p [[buffer(15)]],
    const device uint8_t* u1p [[buffer(16)]],
    const device uint8_t* u2p [[buffer(17)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int n_route = int(tpg.y) - 1;
  const int T_ = int(tpg.z);
  const bool shared_slot = int(tid.y) >= n_route;
  const int p = int(tid.z) * int(tpg.y) + int(tid.y);
  int q = -1;
  int q_row = 0;
  uint32_t expert = 0;
  if (shared_slot) {
    if (int(tid.z) & 1) {
      return;
    }
    if (int(tid.z) + 1 < T_) {
      q_row = int(tid.z) + 1;
      q = q_row * int(tpg.y) + int(tid.y);
    }
  } else {
    const int pr = int(tid.z) * n_route + int(tid.y);
    expert = indices[pr];
    const int qr = kq_dd_partner(indices, pr, n_route * T_, expert);
    if (qr == -2) {
      return;
    }
    if (qr >= 0) {
      q_row = qr / n_route;
      q = q_row * int(tpg.y) + (qr % n_route);
    }
  }
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;
  x += (int64_t)tid.z * K;
  const device T* x1 = q < 0 ? x : x + (int64_t)(q_row - int(tid.z)) * K;
  float4 gu = float4(0.0f);
  KQ_EXT_DECL_LUTS2(KqIq3_xxsExt, SCodec, kq_luts)
  if (shared_slot) {
    KQ_EXT_STAGE_INTO(SCodec, kq_luts)
    if (q < 0) {
      gu.xy = kq_ext_glu_row_partial<T, SCodec, NX, C32>(
          sgw, suw, x, (int64_t)out_row, K, tx, kq_luts);
    } else {
      gu = kq_ext_glu_row_partial_pair<T, SCodec, NX>(
          sgw, suw, x, x1, (int64_t)out_row, K, tx, kq_luts);
    }
  } else {
    KQ_EXT_STAGE_INTO(KqIq3_xxsExt, kq_luts)
    const uint slot = slots[expert];
    if (slot & 0x80000000u) {
      // An expert the store does not hold: its GGUF bytes in place.
      const int64_t wrow = (int64_t)expert * N + out_row;
      if (q < 0) {
        gu.xy = kq_ext_glu_row_partial<T, KqIq3_xxsExt, NX, C32>(
            gw, uw, x, wrow, K, tx, kq_luts);
      } else {
        gu = kq_ext_glu_row_partial_pair<T, KqIq3_xxsExt, NX>(
            gw, uw, x, x1, wrow, K, tx, kq_luts);
      }
    } else {
      KqStIq3Src gs, us;
      gs.begin(g0p, g1p, g2p, slot, uint(out_row), N, K);
      us.begin(u0p, u1p, u2p, slot, uint(out_row), N, K);
      gu = kq_stp_iq3_glu_rows<T, NX, C32>(
          gs, us, x, x1, q >= 0, K, tx, kq_luts);
    }
  }
  const float g0 = kq_ext_reduce<NX>(gu.x);
  const float u0 = kq_ext_reduce<NX>(gu.y);
  const float g1 = kq_ext_reduce<NX>(gu.z);
  const float u1 = kq_ext_reduce<NX>(gu.w);
  if (tx == 0) {
    const float s0 = 1.0f / (1.0f + metal::exp(-g0));
    out[(int64_t)p * N + out_row] = static_cast<T>((g0 * s0) * u0);
    if (q >= 0) {
      const float s1 = 1.0f / (1.0f + metal::exp(-g1));
      out[(int64_t)q * N + out_row] = static_cast<T>((g1 * s1) * u1);
    }
  }
}

// clang-format off
// The pair-grid kernels of a store codec, by simdgroups a threadgroup (2, 4,
// 8: the threadgroup memory they hold), with the shared codecs of the tile
// kernels. sq: the shared codec's qf row core (kq_moe_qf.h) for the SHQF
// route; a shared expert in the routed codec has no such route, and its
// kernels carry a core they never call.
#define instantiate_kq_st_pg(op, name, traits, straits, sq, sg)               \
  instantiate_kernel(                                                         \
      "kq_store_" #op "_" name "_g" #sg "_bfloat16_t", kq_st_##op,            \
      bfloat16_t, traits, straits, sq, sg)                                    \
  instantiate_kernel(                                                         \
      "kq_store_" #op "_" name "_g" #sg "_float16_t", kq_st_##op,             \
      float16_t, traits, straits, sq, sg)
#define instantiate_kq_st_pgs(op, name, traits, straits, sq)                  \
  instantiate_kq_st_pg(op, name, traits, straits, sq, 2)                      \
  instantiate_kq_st_pg(op, name, traits, straits, sq, 4)                      \
  instantiate_kq_st_pg(op, name, traits, straits, sq, 8)
#define instantiate_kq_st_pg_sx(codec, sx, traits, straits, sq)               \
  instantiate_kq_st_pgs(down_pg, #codec "_sx_" sx, traits, straits, sq)       \
  instantiate_kq_st_pgs(glu_pg, #codec "_sx_" sx, traits, straits, sq)
#define instantiate_kq_st_pg_all(codec, traits)                               \
  instantiate_kq_st_pg_sx(codec, #codec, traits, traits, KqQfQ8_0)            \
  instantiate_kq_st_pg_sx(codec, "q5_k", traits, KqQ5_KExt, KqQfQ5_K)         \
  instantiate_kq_st_pg_sx(codec, "q6_k", traits, KqQ6_KExt, KqQfQ6_K)         \
  instantiate_kq_st_pg_sx(codec, "q8_0", traits, KqQ8_0Ext, KqQfQ8_0)

#define instantiate_kq_st_dd_t(sx, straits, type)                             \
  instantiate_kernel(                                                         \
      "kq_store_glu_dd_iq3_xxs_sx_" sx "_" #type, kq_st_glu_dd_iq3, type,     \
      straits, 16, false)                                                     \
  instantiate_kernel(                                                         \
      "kq_store_glu_dd_iq3_xxs_sx_" sx "_c32_" #type, kq_st_glu_dd_iq3, type, \
      straits, 16, true)
#define instantiate_kq_st_dd(sx, straits)                                     \
  instantiate_kq_st_dd_t(sx, straits, bfloat16_t)                             \
  instantiate_kq_st_dd_t(sx, straits, float16_t)                              \
  instantiate_kq_st_dd_t(sx, straits, float)
#define instantiate_kq_st_dd_all()                                            \
  instantiate_kq_st_dd("iq3_xxs", KqIq3_xxsExt)                               \
  instantiate_kq_st_dd("q5_k", KqQ5_KExt)                                     \
  instantiate_kq_st_dd("q6_k", KqQ6_KExt)                                     \
  instantiate_kq_st_dd("q8_0", KqQ8_0Ext)
// clang-format on
