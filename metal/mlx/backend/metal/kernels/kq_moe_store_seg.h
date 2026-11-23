// Wide-row expert store kernel (moe_store_seg): the expert-major segment
// GEMM of gather_qmm_seg (kq_quantized_nax.h) with its weight slab loaded
// from store planes. A wide call (prefill) routes many rows to every expert;
// each tile holds up to BM rows of one expert, the 64 x 64 weight slab of a
// tile step is decoded once into threadgroup memory and every row band
// multiplies against it.
//
// The slab holds the values the in-place loader of the codec writes (the
// same float product, rounded to T once), and the accumulation is
// kq_gather_qmm_seg_nax_tgp_impl's, so the output bits equal
// gather_qmm_seg's on the same rows.
//
// Loaders:
//   KqStSegSplitLoader: the field planes of q5_0, iq4_nl, iq3_xxs, iq4_xs.
//   KqStSegBlockLoader: the generic format (a plane of whole blocks) of a
//     codec whose NAX block loader walks a block by 32-weight tiles
//     (kt_base); the codec's own loader decodes, this one points it at the
//     plane's blocks.
// A codec with neither keeps the store's tile kernels at every width.

#pragma once

#include "mlx/backend/metal/kernels/kq_moe_store.h"

// The slab values of one 32-weight group, as the codec's NAX block loader
// writes them.
template <typename Codec>
struct KqStSegDeq;

template <>
struct KqStSegDeq<KqQ5_0Ext> {
  template <typename T>
  static METAL_FUNC void
  run(typename KqStoreFmt<KqQ5_0Ext>::Payload p, uint, threadgroup T* dst) {
    const float d = float(p.d);
    const uint qh = p.qh;
#pragma unroll
    for (short s = 0; s < 8; s++) {
      const uint v = (p.qs[s / 2] >> (16 * (s & 1))) & 0xFFFFu;
      const uint x0 = (v & 0x000Fu) | (((qh >> (2 * s)) << 4) & 0x10u);
      const uint x1 =
          ((v >> 8) & 0x000Fu) | (((qh >> (2 * s + 1)) << 4) & 0x10u);
      const uint x16 =
          ((v >> 4) & 0x000Fu) | (((qh >> (2 * s + 16)) << 4) & 0x10u);
      const uint x17 = (v >> 12) | (((qh >> (2 * s + 17)) << 4) & 0x10u);
      dst[2 * s + 0] = T(d * (float(x0) - 16.0f));
      dst[2 * s + 1] = T(d * (float(x1) - 16.0f));
      dst[2 * s + 16] = T(d * (float(x16) - 16.0f));
      dst[2 * s + 17] = T(d * (float(x17) - 16.0f));
    }
  }
  // The slab values at group positions pos to pos + 3 (pos a multiple of 4).
  template <typename T>
  static METAL_FUNC vec<T, 4>
  dr4(typename KqStoreFmt<KqQ5_0Ext>::Payload p, uint, short pos) {
    const float d = float(p.d);
    const uint word = p.qs[(pos & 15) >> 2];
    const uint sh = pos >= 16 ? 4u : 0u;
    vec<T, 4> o;
#pragma unroll
    for (short j = 0; j < 4; j++) {
      const uint xq = ((word >> (8 * j + sh)) & 0xFu) |
          (((p.qh >> uint(pos + j)) << 4) & 0x10u);
      o[j] = T(d * (float(xq) - 16.0f));
    }
    return o;
  }
};

template <>
struct KqStSegDeq<KqIq4_nlExt> {
  template <typename T>
  static METAL_FUNC void
  run(typename KqStoreFmt<KqIq4_nlExt>::Payload p, uint, threadgroup T* dst) {
    const float d = float(p.d);
#pragma unroll
    for (short s = 0; s < 8; s++) {
      const uint v = (p.qs[s / 2] >> (16 * (s & 1))) & 0xFFFFu;
      dst[2 * s + 0] = T(d * float(kvalues_iq4nl[v & 0x000Fu]));
      dst[2 * s + 1] = T(d * float(kvalues_iq4nl[(v >> 8) & 0x000Fu]));
      dst[2 * s + 16] = T(d * float(kvalues_iq4nl[(v >> 4) & 0x000Fu]));
      dst[2 * s + 17] = T(d * float(kvalues_iq4nl[v >> 12]));
    }
  }
  template <typename T>
  static METAL_FUNC vec<T, 4>
  dr4(typename KqStoreFmt<KqIq4_nlExt>::Payload p, uint, short pos) {
    const float d = float(p.d);
    const uint word = p.qs[(pos & 15) >> 2];
    const uint sh = pos >= 16 ? 4u : 0u;
    vec<T, 4> o;
#pragma unroll
    for (short j = 0; j < 4; j++) {
      o[j] = T(d * float(kvalues_iq4nl[(word >> (8 * j + sh)) & 0xFu]));
    }
    return o;
  }
};

template <>
struct KqStSegDeq<KqIq4_xsExt> {
  template <typename T>
  static METAL_FUNC void
  run(typename KqStoreFmt<KqIq4_xsExt>::Payload p, uint g, threadgroup T* dst) {
    const uint sb = g % 8u;
    const float d = float(as_type<half>(ushort(p.meta.x & 0xFFFFu)));
    const uint scales_h = p.meta.x >> 16;
    const int ls = int(
        ((p.meta.y >> (4 * sb)) & 0xf) | (((scales_h >> (2 * sb)) & 3) << 4));
    const float dl = d * float(ls - 32);
#pragma unroll
    for (short v = 0; v < 4; v++) {
      const uint q = p.qs[v];
#pragma unroll
      for (short k = 0; k < 4; k++) {
        const short i = 4 * v + k;
        const uint b = (q >> (8 * k)) & 0xFFu;
        dst[i] = T(dl * float(kvalues_iq4nl[b & 0x0Fu]));
        dst[16 + i] = T(dl * float(kvalues_iq4nl[b >> 4]));
      }
    }
  }
  template <typename T>
  static METAL_FUNC vec<T, 4>
  dr4(typename KqStoreFmt<KqIq4_xsExt>::Payload p, uint g, short pos) {
    const uint sb = g % 8u;
    const float d = float(as_type<half>(ushort(p.meta.x & 0xFFFFu)));
    const uint scales_h = p.meta.x >> 16;
    const int ls = int(
        ((p.meta.y >> (4 * sb)) & 0xf) | (((scales_h >> (2 * sb)) & 3) << 4));
    const float dl = d * float(ls - 32);
    const uint word = p.qs[(pos & 15) >> 2];
    const uint sh = pos >= 16 ? 4u : 0u;
    vec<T, 4> o;
#pragma unroll
    for (short j = 0; j < 4; j++) {
      o[j] = T(dl * float(kvalues_iq4nl[(word >> (8 * j + sh)) & 0xFu]));
    }
    return o;
  }
};

// Byte j is 0xFF where bit j of b (4 bits) is set, else 0.
inline uint kq_sign4_mask(uint b) {
  return ((b * 0x00204081u) & 0x01010101u) * 0xFFu;
}
template <>
struct KqStSegDeq<KqIq3_xxsExt> {
  template <typename T>
  static METAL_FUNC void
  run(typename KqStoreFmt<KqIq3_xxsExt>::Payload p, uint, threadgroup T* dst) {
    const float d = float(p.d);
    const uint aux32 = p.gas;
    const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
#pragma unroll
    for (short l = 0; l < 4; l++) {
      const uint w = p.qs[l / 2] >> (16 * (l % 2));
      const uint g0 = iq3xxs_grid[w & 0xFFu];
      const uint g1 = iq3xxs_grid[(w >> 8) & 0xFFu];
      const uint64_t g = uint64_t(g0) | (uint64_t(g1) << 32);
      const uint64_t m = kq_iq2xs_signbytes[(aux32 >> (7 * l)) & 127];
      const uint64_t gs = (g ^ m) + (m & 0x0101010101010101ull);
      const float4 f0 = float4(as_type<char4>(uint(gs))) * db;
      const float4 f1 = float4(as_type<char4>(uint(gs >> 32))) * db;
      *(threadgroup vec<T, 4>*)(dst + l * 8) = vec<T, 4>(f0);
      *(threadgroup vec<T, 4>*)(dst + l * 8 + 4) = vec<T, 4>(f1);
    }
  }
  // Grid bytes are never 0, so the slab's 64-bit sign add carries no byte
  // into the next and each 32-bit half can take it alone; the mask of a
  // half is the sign table's (7 bits and their parity), built in registers.
  template <typename T>
  static METAL_FUNC vec<T, 4>
  dr4(typename KqStoreFmt<KqIq3_xxsExt>::Payload p, uint, short pos) {
    const float d = float(p.d);
    const uint aux32 = p.gas;
    const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
    const short l = pos >> 3;
    const short sub = (pos >> 2) & 1;
    const uint w = p.qs[l / 2] >> (16 * (l % 2));
    const uint gr = iq3xxs_grid[(w >> (8 * sub)) & 0xFFu];
    const uint i7 = (aux32 >> (7 * l)) & 127u;
    const uint sg = i7 | ((popcount(i7) & 1u) << 7);
    const uint m = kq_sign4_mask(sub ? (sg >> 4) : (sg & 0xFu));
    const uint s = (gr ^ m) + (m & 0x01010101u);
    const float4 f = float4(as_type<char4>(s)) * db;
    return vec<T, 4>(f);
  }
};

// Slab loader on field planes: 128 threads, each one 32-weight group of one
// of the tile's 64 columns (two groups a 64-wide K step), as the block
// loaders divide the slab.
template <typename T, typename Codec, short dst_ld>
struct KqStSegSplitLoader {
  typedef KqStoreSrc<Codec> Src;
  typedef Codec CodecT;
  MLX_MTL_CONST bool kColumn = !KqStoreFmt<Codec>::LUTS;
  KqStoreSrc<Codec> src;
  threadgroup T* dst;
  uint g;

  KqStSegSplitLoader(
      const device uint8_t* p0,
      const device uint8_t* p1,
      const device uint8_t* p2,
      uint slot,
      uint col0,
      int N,
      int K,
      threadgroup T* dst_,
      uint simd_group_id,
      uint simd_lane_id) thread {
    const uint thread_idx = simd_group_id * 32 + simd_lane_id;
    const uint bi = (thread_idx / 2) % KQ_ST_TILE;
    g = thread_idx % 2;
    dst = dst_ + bi * dst_ld + g * 32;
    src.begin(p0, p1, p2, slot, col0 + bi, N, K);
  }

  void load_unsafe() thread {
    KqStSegDeq<Codec>::run(src.load(g), g, dst);
  }

  void next() thread {
    g += 2;
  }
};

// Slab loader on the generic format: the codec's NAX block loader pointed at
// the plane. Block u of column r of a tile is record (u, r) of the tile:
// the blocks of a column are 64 records apart.
template <typename T, typename Inner>
struct KqStSegBlockLoader {
  typedef void Src;
  MLX_MTL_CONST bool kColumn = false;
  Inner in;

  KqStSegBlockLoader(
      const device uint8_t* p0,
      const device uint8_t*,
      const device uint8_t*,
      uint slot,
      uint col0,
      int N,
      int K,
      threadgroup T* dst_,
      uint simd_group_id,
      uint simd_lane_id) thread
      : in(p0, K, dst_, ushort(simd_group_id), ushort(simd_lane_id)) {
    const ulong tile = ulong(slot) * (uint(N) / KQ_ST_TILE) + col0 / KQ_ST_TILE;
    const ulong U = uint(K) / uint(Inner::weights_per_block);
    const ulong r = uint(in.bi) % KQ_ST_TILE;
    in.src = p0 + (tile * U * KQ_ST_TILE + r) * uint(Inner::bytes_per_block);
  }

  void load_unsafe() thread {
    in.load_unsafe();
  }

  void next() thread {
    in.kt_base += 2;
    if (in.kt_base == Inner::k_tiles_per_block) {
      in.kt_base = 0;
      in.src += KQ_ST_TILE * Inner::bytes_per_block;
    }
  }
};

// kq_st_column of a source on up to 4 rows at their own pointers (rows past
// `rows` unread), over K groups [g_lo, g_hi) of 32, for a format without
// threadgroup tables.
template <typename T, typename Src>
METAL_FUNC float4 kq_st_seg_column(
    thread Src& src,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    const device T* x3,
    int rows,
    uint g_lo,
    uint g_hi) {
  float4 sums = float4(0.0f);
  typename Src::Payload w = src.load(g_lo);
  for (uint g = g_lo; g < g_hi; ++g) {
    half4 v[8];
    Src::dequant(w, g, v, nullptr);
    if (g + 1 < g_hi) {
      w = src.load(g + 1);
    }
#pragma unroll
    for (short r = 0; r < 4; ++r) {
      if (r >= rows) {
        break;
      }
      const device T* xr = (r == 0       ? x0
                                : r == 1 ? x1
                                : r == 2 ? x2
                                         : x3) +
          g * 32;
      float sum = 0.0f;
#pragma unroll
      for (short i = 0; i < 8; ++i) {
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

// The column path of kq_st_seg (kq_seg_colr): thread t walks column t % 64
// of the tile over part t / 64 of the K groups, parts summed in order
// through the threadgroup. y points at the tile's first output.
template <typename T, typename Src, int NT>
METAL_FUNC void kq_st_seg_one_column(
    const device T* x,
    const device uint32_t* x_rows,
    size_t row0,
    int num_rows,
    int N,
    int K,
    const device uint8_t* p0,
    const device uint8_t* p1,
    const device uint8_t* p2,
    uint slot,
    uint y_col,
    device T* y,
    threadgroup float* part,
    uint t) {
  constexpr uint NP = NT / 64;
  const uint c = t % 64;
  const uint p = t / 64;
  const uint steps = uint(K) / 32;
  Src src;
  src.begin(p0, p1, p2, slot, y_col + c, N, K);
  const device T* xs[4];
#pragma unroll
  for (short r = 0; r < 4; ++r) {
    const int rr = r < num_rows ? r : 0;
    xs[r] = kq_seg_xmap ? x + size_t(x_rows[row0 + rr]) * size_t(K)
                        : x + size_t(rr) * size_t(K);
  }
  float4 s = kq_st_seg_column<T, Src>(
      src,
      xs[0],
      xs[1],
      xs[2],
      xs[3],
      num_rows,
      p * steps / NP,
      (p + 1) * steps / NP);
  if (NP > 1) {
    if (p > 0) {
      for (int r = 0; r < num_rows; ++r) {
        part[((p - 1) * 4 + r) * 64 + c] = s[r];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p > 0) {
      return;
    }
    for (uint q = 1; q < NP; ++q) {
      for (int r = 0; r < num_rows; ++r) {
        s[r] += part[((q - 1) * 4 + r) * 64 + c];
      }
    }
  }
  for (int r = 0; r < num_rows; ++r) {
    y[size_t(r) * N + c] = static_cast<T>(s[r]);
  }
}

// kq_gather_qmm_seg_nax_tgp_impl with the slab from a store loader: N a
// multiple of 64 and K a multiple of 64 (the store's shapes), no output
// scale. seg_map rows are (slot, row_start, num_rows <= BM).
// The matmul path of kq_st_seg on a simdgroup layout of WMx x WNx (SM rows
// and 64 / WNx columns a simdgroup).
template <typename T, typename LoaderW, short SM, short WMx, short WNx>
METAL_FUNC void kq_st_seg_mma(
    const device T* x,
    const device uint32_t* x_rows,
    size_t row0,
    int num_rows,
    int N,
    int K,
    const device uint8_t* p0,
    const device uint8_t* p1,
    const device uint8_t* p2,
    uint32_t slot,
    int y_col,
    threadgroup T* Ws,
    device T* y,
    uint simd_group_id,
    uint simd_lane_id) {
  constexpr int BN = 64;
  constexpr int BK = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  const int K_it = K / BK;
  constexpr short SN = BN / WNx;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WNx);
  const short tn = SN * (simd_group_id % WNx);

  const short sgp_sm = short(min(int(SM), max(0, num_rows - int(tm))));
  const bool sg_active = sgp_sm > 0;

  constexpr short BR = TN;
  constexpr short BC = TK;

  NAXTile<float, 1, TN> Dsub[TM];
#pragma unroll
  for (short mi = 0; mi < TM; mi++) {
    Dsub[mi].clear();
  }

  const device T* xn = x + tm * K;
  const device T* xr[TM][2];
  int xk = 0;
  if (kq_seg_xmap) {
    const short fm = BaseNAXFrag::get_coord().y;
#pragma unroll
    for (short mi = 0; mi < TM; mi++) {
#pragma unroll
      for (short i = 0; i < 2; i++) {
        const int r = int(tm) + mi * 16 + fm + i * 8;
        const int rr = r < num_rows ? r : 0;
        xr[mi][i] = x + size_t(x_rows[row0 + rr]) * size_t(K);
      }
    }
  }

  thread LoaderW loader_w(
      p0, p1, p2, slot, uint(y_col), N, K, Ws, simd_group_id, simd_lane_id);
  const bool sg_loads = int(simd_group_id) < kSegLoadSimdgroups;

  for (int k = 0; k < K_it; k++) {
    const threadgroup T* Wk = Ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_loads) {
      loader_w.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < BK; kk1 += SK) {
        NAXTile<T, BR, BC> Btile;
        volatile int compiler_barrier;
        Btile.template load<T, BK_padded, 1>(Wk + tn * BK_padded + kk1);
#pragma unroll
        for (short mi = 0; mi < TM; mi++) {
          if (mi * 16 < sgp_sm) {
            NAXTile<T, 1, TK> At;
            const short rows = min(short(16), short(sgp_sm - mi * 16));
            if (kq_seg_xmap) {
              kq_seg_load_rows(At, xr[mi][0], xr[mi][1], xk + kk1, SK);
            } else if (rows == 16) {
              At.load(xn + mi * 16 * K + kk1, K);
            } else {
              At.load_safe(xn + mi * 16 * K + kk1, K, short2(SK, rows));
            }
            tile_matmad_nax(
                Dsub[mi],
                At,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<true>{});
          }
        }
        (void)compiler_barrier;
      }
    }

    xn += BK;
    xk += BK;
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg_active) {
#pragma unroll
    for (short mi = 0; mi < TM; mi++) {
      if (mi * 16 < sgp_sm) {
        const short rows = min(short(16), short(sgp_sm - mi * 16));
        device T* yp = y + (tm + mi * 16) * N + tn;
        if (rows == 16) {
          Dsub[mi].store(yp, N);
        } else {
          Dsub[mi].store_slice(yp, N, short2(0, 0), short2(SN, rows));
        }
      }
    }
  }
}

// Tiles of at most 32 rows (kq_seg_dr): no slab and no barrier in the K
// walk. One simdgroup's 32 columns from col0 on TMr 16-row bands: each lane
// decodes the weights of its own B fragments from the planes (the slab
// loader's values) and multiplies in the matmul path's K order (32-wide
// steps, one tile_matmad_nax a band), so the same bits.
template <typename T, typename Codec, short TMr>
METAL_FUNC void kq_st_dr_walk(
    thread NAXTile<float, 1, 2> (&D)[TMr],
    const device T* x,
    const device uint32_t* x_rows,
    size_t row0,
    int num_rows,
    int N,
    int K,
    const device uint8_t* b0,
    const device uint8_t* b1,
    const device uint8_t* b2,
    uint32_t slot,
    int col0) {
  constexpr short TN = 2;
  constexpr short TK = 2;
  constexpr short SK = 32;
  typedef KqStoreSrc<Codec> Src;
  const short2 sc = BaseNAXFrag::get_coord();
  const short fn = sc.x;
  const short fm = sc.y;

  Src src[TN][2];
#pragma unroll
  for (short r = 0; r < TN; r++) {
#pragma unroll
    for (short i = 0; i < 2; i++) {
      src[r][i].begin(b0, b1, b2, slot, uint(col0 + 16 * r + fm + 8 * i), N, K);
    }
  }
  const device T* xr[TMr][2];
#pragma unroll
  for (short mi = 0; mi < TMr; mi++) {
#pragma unroll
    for (short i = 0; i < 2; i++) {
      const int r = 16 * mi + fm + 8 * i;
      const int rr = r < num_rows ? r : 0;
      xr[mi][i] = kq_seg_xmap ? x + size_t(x_rows[row0 + rr]) * size_t(K)
                              : x + size_t(rr) * size_t(K);
    }
    D[mi].clear();
  }

  const int G = K / SK;
  for (int g = 0; g < G; g++) {
    NAXTile<T, TN, TK> B;
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < TN; r++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        const typename Src::Payload p = src[r][i].load(uint(g));
        const vec<T, 4> lo = KqStSegDeq<Codec>::template dr4<T>(p, uint(g), fn);
        const vec<T, 4> hi =
            KqStSegDeq<Codec>::template dr4<T>(p, uint(g), short(16 + fn));
        thread auto& f0 = B.frag_at(r, 0);
        thread auto& f1 = B.frag_at(r, 1);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          f0[i * 4 + j] = lo[j];
          f1[i * 4 + j] = hi[j];
        }
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short mi = 0; mi < TMr; mi++) {
      NAXTile<T, 1, TK> At;
      kq_seg_load_rows(At, xr[mi][0], xr[mi][1], g * SK, SK);
      tile_matmad_nax(
          D[mi],
          At,
          metal::bool_constant<false>{},
          B,
          metal::bool_constant<true>{});
    }
  }
}

template <typename T, short TMr>
METAL_FUNC void kq_st_dr_store(
    thread NAXTile<float, 1, 2> (&D)[TMr],
    int num_rows,
    int N,
    device T* o) {
  STEEL_PRAGMA_UNROLL
  for (short mi = 0; mi < TMr; mi++) {
    const short rows = short(min(16, num_rows - 16 * mi));
    device T* om = o + 16 * mi * N;
    if (rows == 16) {
      D[mi].store(om, N);
    } else if (rows > 0) {
      D[mi].store_slice(om, N, short2(0, 0), short2(32, rows));
    }
  }
}

// The single kernel's DR: simdgroups 0-1 on 32 columns each, the matmul
// path's own layout at these rows (2-3 idle there too).
template <typename T, typename Codec, short TMr>
METAL_FUNC void kq_st_seg_dr(
    const device T* x,
    const device uint32_t* x_rows,
    size_t row0,
    int num_rows,
    int N,
    int K,
    const device uint8_t* p0,
    const device uint8_t* p1,
    const device uint8_t* p2,
    uint32_t slot,
    int y_col,
    device T* y,
    uint simd_group_id) {
  if (simd_group_id >= 2) {
    return;
  }
  const short tn = 32 * short(simd_group_id);
  NAXTile<float, 1, 2> D[TMr];
  kq_st_dr_walk<T, Codec, TMr>(
      D, x, x_rows, row0, num_rows, N, K, p0, p1, p2, slot, y_col + tn);
  kq_st_dr_store<T, TMr>(D, num_rows, N, y + tn);
}

template <typename T, typename LoaderW, int BM, int WM>
[[kernel]] void kq_st_seg(
    const device T* x [[buffer(0)]],
    const device uint8_t* p0 [[buffer(1)]],
    const device uint8_t* p1 [[buffer(2)]],
    const device uint32_t* seg_map [[buffer(3)]],
    const device uint32_t* tile_count [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const device uint32_t* x_rows_
    [[buffer(8), function_constant(kq_seg_xmap)]],
    const device uint8_t* p2 [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int BN = 64;
  constexpr int BK = 64;
  if (kq_seg_gx > 0) {
    const uint n_cols = uint(N / BN);
    const uint t = (tid.y / n_cols) * uint(kq_seg_gx) + tid.x;
    tid.x = tid.y % n_cols;
    tid.y = t;
  }
  if (tid.y >= tile_count[0]) {
    return;
  }
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];
  const device uint32_t* x_rows = kq_seg_xmap ? x_rows_ : nullptr;

  const device uint32_t* seg = seg_map + 3 * tid.y;
  const uint32_t slot = seg[0];
  const size_t row0 = size_t(seg[1]);
  const int num_rows = int(seg[2]);

  const int y_col = tid.x * BN;

  if (!kq_seg_xmap) {
    x += row0 * static_cast<size_t>(K);
  }
  y += row0 * static_cast<size_t>(N) + size_t(y_col);

  if constexpr (LoaderW::kColumn) {
    if (kq_seg_colr > 0 && num_rows <= kq_seg_colr) {
      kq_st_seg_one_column<T, typename LoaderW::Src, WM * 2 * 32>(
          x,
          x_rows,
          row0,
          num_rows,
          N,
          K,
          p0,
          p1,
          p2,
          slot,
          uint(y_col),
          y,
          (threadgroup float*)Ws,
          simd_group_id * 32 + simd_lane_id);
      return;
    }
    if constexpr (BM == 64 && WM == 2) {
      if (kq_seg_dr > 0 && num_rows <= kq_seg_dr) {
        if (num_rows <= 16) {
          kq_st_seg_dr<T, typename LoaderW::CodecT, 1>(
              x,
              x_rows,
              row0,
              num_rows,
              N,
              K,
              p0,
              p1,
              p2,
              slot,
              y_col,
              y,
              simd_group_id);
        } else {
          kq_st_seg_dr<T, typename LoaderW::CodecT, 2>(
              x,
              x_rows,
              row0,
              num_rows,
              N,
              K,
              p0,
              p1,
              p2,
              slot,
              y_col,
              y,
              simd_group_id);
        }
        return;
      }
    }
  }

  kq_st_seg_mma<T, LoaderW, BM / WM, WM, 2>(
      x,
      x_rows,
      row0,
      num_rows,
      N,
      K,
      p0,
      p1,
      p2,
      slot,
      y_col,
      Ws,
      y,
      simd_group_id,
      simd_lane_id);
}

// The gate and up products of one wide expert pass in one kernel: kq_st_seg's
// walk with the slabs of both weights decoded at each step and each A tile
// multiplied against both, each product summed in kq_st_seg's order. OUT 0
// writes the gate's product to y and the up's to y2, each with the bits of
// its own kq_st_seg call. OUT 1 writes silu(gate) * up to y, each product
// rounded to T first and the activation in T as MLX's ops take it.
// The column path of kq_st_seg_pair (kq_seg_colr): threads 0-63 walk the
// tile's 64 gate columns, 64-127 its up columns, each over the tile's rows
// (at most 4) in registers. y and y2 point at the tile's first output.
template <typename T, typename Src, int OUT>
METAL_FUNC void kq_st_seg_pair_column(
    const device T* x,
    const device uint32_t* x_rows,
    size_t row0,
    int num_rows,
    int N,
    int K,
    const device uint8_t* g0,
    const device uint8_t* g1,
    const device uint8_t* g2,
    const device uint8_t* u0,
    const device uint8_t* u1,
    const device uint8_t* u2,
    uint slot,
    uint y_col,
    device T* y,
    device T* y2,
    threadgroup T* xch,
    uint t) {
  const bool active = t < 128;
  const bool up = t >= 64;
  const uint c = t % 64;
  float4 s = float4(0.0f);
  if (active) {
    Src src;
    if (up) {
      src.begin(u0, u1, u2, slot, y_col + c, N, K);
    } else {
      src.begin(g0, g1, g2, slot, y_col + c, N, K);
    }
    const device T* xs[4];
#pragma unroll
    for (short r = 0; r < 4; ++r) {
      const int rr = r < num_rows ? r : 0;
      xs[r] = kq_seg_xmap ? x + size_t(x_rows[row0 + rr]) * size_t(K)
                          : x + size_t(rr) * size_t(K);
    }
    s = kq_st_seg_column<T, Src>(
        src, xs[0], xs[1], xs[2], xs[3], num_rows, 0, uint(K) / 32);
  }
  if constexpr (OUT == 1) {
    if (active && !up) {
      for (int r = 0; r < num_rows; ++r) {
        xch[r * 64 + c] = static_cast<T>(s[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && up) {
      for (int r = 0; r < num_rows; ++r) {
        const T gv = xch[r * 64 + c];
        const T uv = static_cast<T>(s[r]);
        const T sv = Sigmoid()(gv);
        y[size_t(r) * N + c] = Multiply()(Multiply()(gv, sv), uv);
      }
    }
  } else {
    if (active) {
      device T* o = up ? y2 : y;
      for (int r = 0; r < num_rows; ++r) {
        o[size_t(r) * N + c] = static_cast<T>(s[r]);
      }
    }
  }
}

// The matmul path of kq_st_seg_pair on a simdgroup layout of WMx x WNx
// (SM rows and 64 / WNx columns a simdgroup).
template <typename T, typename LoaderW, short SM, short WMx, short WNx, int OUT>
METAL_FUNC void kq_st_seg_pair_mma(
    const device T* x,
    const device uint32_t* x_rows,
    size_t row0,
    int num_rows,
    int N,
    int K,
    const device uint8_t* g0,
    const device uint8_t* g1,
    const device uint8_t* g2,
    const device uint8_t* u0,
    const device uint8_t* u1,
    const device uint8_t* u2,
    uint32_t slot,
    int y_col,
    threadgroup T* Wg,
    threadgroup T* Wu,
    device T* y,
    device T* y2,
    uint simd_group_id,
    uint simd_lane_id) {
  constexpr int BN = 64;
  constexpr int BK = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  const int K_it = K / BK;
  constexpr short SN = BN / WNx;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WNx);
  const short tn = SN * (simd_group_id % WNx);

  const short sgp_sm = short(min(int(SM), max(0, num_rows - int(tm))));
  const bool sg_active = sgp_sm > 0;

  constexpr short BR = TN;
  constexpr short BC = TK;

  NAXTile<float, 1, TN> Dg[TM];
  NAXTile<float, 1, TN> Du[TM];
#pragma unroll
  for (short mi = 0; mi < TM; mi++) {
    Dg[mi].clear();
    Du[mi].clear();
  }

  const device T* xn = x + tm * K;
  const device T* xr[TM][2];
  int xk = 0;
  if (kq_seg_xmap) {
    const short fm = BaseNAXFrag::get_coord().y;
#pragma unroll
    for (short mi = 0; mi < TM; mi++) {
#pragma unroll
      for (short i = 0; i < 2; i++) {
        const int r = int(tm) + mi * 16 + fm + i * 8;
        const int rr = r < num_rows ? r : 0;
        xr[mi][i] = x + size_t(x_rows[row0 + rr]) * size_t(K);
      }
    }
  }

  thread LoaderW lg(
      g0, g1, g2, slot, uint(y_col), N, K, Wg, simd_group_id, simd_lane_id);
  thread LoaderW lu(
      u0, u1, u2, slot, uint(y_col), N, K, Wu, simd_group_id, simd_lane_id);
  const bool sg_loads = int(simd_group_id) < kSegLoadSimdgroups;

  for (int k = 0; k < K_it; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_loads) {
      lg.load_unsafe();
      lu.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < BK; kk1 += SK) {
        NAXTile<T, BR, BC> Bg;
        NAXTile<T, BR, BC> Bu;
        volatile int compiler_barrier;
        Bg.template load<T, BK_padded, 1>(Wg + tn * BK_padded + kk1);
        Bu.template load<T, BK_padded, 1>(Wu + tn * BK_padded + kk1);
#pragma unroll
        for (short mi = 0; mi < TM; mi++) {
          if (mi * 16 < sgp_sm) {
            NAXTile<T, 1, TK> At;
            const short rows = min(short(16), short(sgp_sm - mi * 16));
            if (kq_seg_xmap) {
              kq_seg_load_rows(At, xr[mi][0], xr[mi][1], xk + kk1, SK);
            } else if (rows == 16) {
              At.load(xn + mi * 16 * K + kk1, K);
            } else {
              At.load_safe(xn + mi * 16 * K + kk1, K, short2(SK, rows));
            }
            tile_matmad_nax(
                Dg[mi],
                At,
                metal::bool_constant<false>{},
                Bg,
                metal::bool_constant<true>{});
            tile_matmad_nax(
                Du[mi],
                At,
                metal::bool_constant<false>{},
                Bu,
                metal::bool_constant<true>{});
          }
        }
        (void)compiler_barrier;
      }
    }

    xn += BK;
    xk += BK;
    lg.next();
    lu.next();
  }

  if (!sg_active) {
    return;
  }
  if constexpr (OUT == 1) {
    // Every tile under a full unroll and no row guard: a tile index that is
    // not a constant here puts Dg and Du in stack memory for the whole walk.
    using Tile = NAXTile<float, 1, TN>;
    STEEL_PRAGMA_UNROLL
    for (short mi = 0; mi < TM; mi++) {
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < Tile::kNumFrags; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < Tile::kElemsPerFrag; ++e) {
          const T gv = static_cast<T>(Dg[mi].val_frags[f][e]);
          const T uv = static_cast<T>(Du[mi].val_frags[f][e]);
          const T sv = Sigmoid()(gv);
          Dg[mi].val_frags[f][e] =
              static_cast<float>(Multiply()(Multiply()(gv, sv), uv));
        }
      }
    }
  }
#pragma unroll
  for (short mi = 0; mi < TM; mi++) {
    if (mi * 16 < sgp_sm) {
      const short rows = min(short(16), short(sgp_sm - mi * 16));
      const size_t off = size_t(tm + mi * 16) * N + tn;
      if (rows == 16) {
        Dg[mi].store(y + off, N);
      } else {
        Dg[mi].store_slice(y + off, N, short2(0, 0), short2(SN, rows));
      }
      if constexpr (OUT == 0) {
        if (rows == 16) {
          Du[mi].store(y2 + off, N);
        } else {
          Du[mi].store_slice(y2 + off, N, short2(0, 0), short2(SN, rows));
        }
      }
    }
  }
}

// The pair kernel's DR: simdgroups 0-1 on the tile's gate columns, 2-3 on
// its up columns, 32 each. The fused activation takes the up values
// through xch.
template <typename T, typename Codec, int OUT, short TMr>
METAL_FUNC void kq_st_seg_pair_dr(
    const device T* x,
    const device uint32_t* x_rows,
    size_t row0,
    int num_rows,
    int N,
    int K,
    const device uint8_t* g0,
    const device uint8_t* g1,
    const device uint8_t* g2,
    const device uint8_t* u0,
    const device uint8_t* u1,
    const device uint8_t* u2,
    uint32_t slot,
    int y_col,
    threadgroup T* xch,
    device T* y,
    device T* y2,
    uint simd_group_id) {
  const bool up = simd_group_id >= 2;
  const short tn = 32 * short(simd_group_id % 2);
  NAXTile<float, 1, 2> D[TMr];
  kq_st_dr_walk<T, Codec, TMr>(
      D,
      x,
      x_rows,
      row0,
      num_rows,
      N,
      K,
      up ? u0 : g0,
      up ? u1 : g1,
      up ? u2 : g2,
      slot,
      y_col + tn);

  using Tile = NAXTile<float, 1, 2>;
  if constexpr (OUT == 1) {
    const short2 sc = BaseNAXFrag::get_coord();
    const short fn = sc.x;
    const short fm = sc.y;
    if (up) {
      STEEL_PRAGMA_UNROLL
      for (short mi = 0; mi < TMr; mi++) {
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < Tile::kNumFrags; ++f) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < Tile::kElemsPerFrag; ++e) {
            const short row = 16 * mi + fm + 8 * (e / 4);
            const short col = tn + 16 * f + fn + (e % 4);
            xch[row * 64 + col] = static_cast<T>(D[mi].val_frags[f][e]);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up) {
      return;
    }
    STEEL_PRAGMA_UNROLL
    for (short mi = 0; mi < TMr; mi++) {
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < Tile::kNumFrags; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < Tile::kElemsPerFrag; ++e) {
          const short row = 16 * mi + fm + 8 * (e / 4);
          const short col = tn + 16 * f + fn + (e % 4);
          const T gv = static_cast<T>(D[mi].val_frags[f][e]);
          const T uv = xch[row * 64 + col];
          const T sv = Sigmoid()(gv);
          D[mi].val_frags[f][e] =
              static_cast<float>(Multiply()(Multiply()(gv, sv), uv));
        }
      }
    }
  }
  kq_st_dr_store<T, TMr>(D, num_rows, N, (up ? y2 : y) + tn);
}

template <typename T, typename LoaderW, int BM, int WM, int OUT>
[[kernel]] void kq_st_seg_pair(
    const device T* x [[buffer(0)]],
    const device uint8_t* g0 [[buffer(1)]],
    const device uint8_t* g1 [[buffer(2)]],
    const device uint32_t* seg_map [[buffer(3)]],
    const device uint32_t* tile_count [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const device uint32_t* x_rows_
    [[buffer(8), function_constant(kq_seg_xmap)]],
    const device uint8_t* g2 [[buffer(9)]],
    const device uint8_t* u0 [[buffer(10)]],
    const device uint8_t* u1 [[buffer(11)]],
    const device uint8_t* u2 [[buffer(12)]],
    device T* y2 [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int BN = 64;
  constexpr int BK = 64;
  if (kq_seg_gx > 0) {
    const uint n_cols = uint(N / BN);
    const uint t = (tid.y / n_cols) * uint(kq_seg_gx) + tid.x;
    tid.x = tid.y % n_cols;
    tid.y = t;
  }
  if (tid.y >= tile_count[0]) {
    return;
  }
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Wg[BN * BK_padded];
  threadgroup T Wu[BN * BK_padded];
  const device uint32_t* x_rows = kq_seg_xmap ? x_rows_ : nullptr;

  const device uint32_t* seg = seg_map + 3 * tid.y;
  const uint32_t slot = seg[0];
  const size_t row0 = size_t(seg[1]);
  const int num_rows = int(seg[2]);

  const int y_col = tid.x * BN;

  if (!kq_seg_xmap) {
    x += row0 * static_cast<size_t>(K);
  }
  y += row0 * static_cast<size_t>(N) + size_t(y_col);
  y2 += row0 * static_cast<size_t>(N) + size_t(y_col);

  if constexpr (LoaderW::kColumn) {
    if (kq_seg_colr > 0 && num_rows <= kq_seg_colr) {
      kq_st_seg_pair_column<T, typename LoaderW::Src, OUT>(
          x,
          x_rows,
          row0,
          num_rows,
          N,
          K,
          g0,
          g1,
          g2,
          u0,
          u1,
          u2,
          slot,
          uint(y_col),
          y,
          y2,
          Wg,
          simd_group_id * 32 + simd_lane_id);
      return;
    }
    if constexpr (BM == 64 && WM == 2) {
      if (kq_seg_dr > 0 && num_rows <= kq_seg_dr) {
        if (num_rows <= 16) {
          kq_st_seg_pair_dr<T, typename LoaderW::CodecT, OUT, 1>(
              x,
              x_rows,
              row0,
              num_rows,
              N,
              K,
              g0,
              g1,
              g2,
              u0,
              u1,
              u2,
              slot,
              y_col,
              Wg,
              y,
              y2,
              simd_group_id);
        } else {
          kq_st_seg_pair_dr<T, typename LoaderW::CodecT, OUT, 2>(
              x,
              x_rows,
              row0,
              num_rows,
              N,
              K,
              g0,
              g1,
              g2,
              u0,
              u1,
              u2,
              slot,
              y_col,
              Wg,
              y,
              y2,
              simd_group_id);
        }
        return;
      }
    }
  }

  kq_st_seg_pair_mma<T, LoaderW, BM / WM, WM, 2, OUT>(
      x,
      x_rows,
      row0,
      num_rows,
      N,
      K,
      g0,
      g1,
      g2,
      u0,
      u1,
      u2,
      slot,
      y_col,
      Wg,
      Wu,
      y,
      y2,
      simd_group_id,
      simd_lane_id);
}
