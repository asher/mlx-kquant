// Tensor-op (NAX) index-gathered attention over a shared K/V latent, built
// on MLX's steel NAX attention tiles (MIT; see mlx_kquant/licenses/).
// The simdgroup-matrix form lives in kq_sdpa.h (kq_sdpa_fa_indexed_2pass_1)
// and serves GPUs without tensor-op hardware; both write the same per-split
// partials, so the shared kq_sdpa_gqa_2pass_2 merge finishes either.
//
// One threadgroup per (32-head strip, query, key split): eight simdgroups
// each own a 64-column eighth of the head dim, so Q (resident in
// registers), O and P @ V stay at 64 floats per thread at head_dim 512.
// Each simdgroup loads its 64 columns of the 16 gathered key rows of a
// tile straight into fragment registers through the query's index list,
// one tile ahead of the matmuls so the gathered-row latency overlaps the
// compute; the fragments serve both K and V, and every row's bytes are
// read once per threadgroup. A simdgroup computes S^T = K @ Q^T over its
// columns (the fragment form MLX's attention uses, keys as rows) and parks
// the partial in threadgroup scratch; the simdgroups sum the eight
// partials by sub-block and read the total back transposed (heads as
// rows) for the row softmax. A negative or out-of-range index entry
// zero-fills and masks.

#pragma once

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;
using namespace mlx::steel;

constant int gqa_splits [[function_constant(2)]];

namespace kq_sdpa_nax {

struct MaxOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return metal::max(x, y);
  }
};
struct SumOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x + y;
  }
};
struct MulOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x * y;
  }
};
struct ExpSubOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return fast::exp2(x - y);
  }
};

} // namespace kq_sdpa_nax

template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]] void
kq_sdpa_fa_indexed_nax_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* kv [[buffer(1)]],
    const device int32_t* idx [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& kv_len [[buffer(7)]],
    const constant size_t& kv_seq_stride [[buffer(8)]],
    const constant float& scale [[buffer(9)]],
    const constant int& n_heads [[buffer(10)]],
    const constant int& n_queries [[buffer(11)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using namespace kq_sdpa_nax;
  constexpr int D = 512;
  constexpr int NSG = 8; // simdgroups, one per 64-column eighth
  constexpr int DE = D / NSG; // columns per simdgroup
  constexpr short kU = 16;
  constexpr int BK = 16; // keys per tile
  constexpr short TDE = DE / kU; // 4 fragments of columns per simdgroup
  constexpr int BQ = 32; // heads per strip (two row fragments)
  constexpr int SLD = 34; // exchange slot row stride (floats)
  constexpr int SLOT = BK * SLD;
  using T4 = metal::vec<T, 4>;

  using otile_t = NAXTile<float, 2, TDE>;
  using stile_t = NAXTile<float, 2, 1>; // heads x keys
  using sttile_t = NAXTile<float, 1, 2>; // keys x heads
  using kfrag_t = typename BaseNAXFrag::dtype_frag_t<T>;

  // Eight partial slots plus two result buffers (tile parity), each
  // [16 keys][34] floats.
  threadgroup float slots[(NSG + 2) * SLOT];
  threadgroup float* results = slots + NSG * SLOT;

  const short eighth = simd_group_id;
  const int h0 = int(tid.x) * BQ;
  const int query_idx = tid.y;
  const int split_idx = tid.z;
  const int rows_valid = metal::clamp(n_heads - h0, 0, BQ);

  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);
  const device int32_t* idx_row = idx + (size_t)query_idx * N;

  const short2 sc = BaseNAXFrag::get_coord();
  const short sm = sc.y;
  const short sn = sc.x;
  const device T* kvcol = kv + eighth * DE + sn;

  // Q fragments of this strip over this simdgroup's columns, resident for
  // the whole walk (natural [1, Hq, Q, D] layout; heads past n_heads zero).
  NAXTile<T, 2, TDE> Qtile;
  {
    const int q_ld = n_queries * D;
    const device T* qbase =
        queries + ((size_t)h0 * n_queries + query_idx) * D + eighth * DE;
    if (rows_valid == BQ) {
      Qtile.load(qbase, q_ld);
    } else {
      Qtile.load_rows(qbase, q_ld, short(rows_valid));
    }
  }

  otile_t Otile;
  Otile.clear();
  metal::vec<float, 4> max_score;
  metal::vec<float, 4> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 4; i++) {
    max_score[i] = Limits<float>::finite_min;
  }
  const float scale2 = scale * M_LOG2E_F;

  // A tile in registers: this lane's two fragment rows (keys sm and
  // sm + 8 of the tile) over the four column fragments, plus the validity
  // of its four S columns (keys sn + jj). A padded, out-of-range or
  // past-the-split slot reads row 0 and selects zero, so every lane issues
  // the same loads and the compiler can batch them.
  struct Tile {
    kfrag_t kf[TDE];
    bool cv[4];
  };
  auto load_tile = [&](int kt, thread Tile& t) {
    int g[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      const int slot = kt + sm + i * 8;
      const int gi = slot < k1 ? int(idx_row[slot]) : -1;
      g[i] = (gi >= 0 && gi < kv_len) ? gi : -1;
    }
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < 4; jj++) {
      const int slot = kt + sn + jj;
      const int gi = slot < k1 ? int(idx_row[slot]) : -1;
      t.cv[jj] = gi >= 0 && gi < kv_len;
    }
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDE; dd++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        const T4 v =
            *((const device T4*)(kvcol +
                                 (size_t)metal::max(g[i], 0) * kv_seq_stride +
                                 dd * kU));
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          t.kf[dd][i * 4 + j] = g[i] >= 0 ? v[j] : T(0);
        }
      }
    }
  };

  int parity = 0;
  auto step = [&](int kt, thread Tile& cur) {
    // Partial S^T = K @ Q^T over this simdgroup's columns: 16 keys x 32
    // heads.
    sttile_t STt;
    STt.clear();
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDE; dd++) {
      sttile_t::NAXFrag_t::mma(
          STt.frag_at(0, 0),
          STt.frag_at(0, 1),
          cur.kf[dd],
          metal::false_type{},
          Qtile.frag_at(0, dd),
          Qtile.frag_at(1, dd),
          metal::true_type{});
    }

    // Exchange: park the partial in slot `eighth` as [key][head], sum the
    // eight slots by sub-block into this tile's result buffer, then read
    // the total back transposed (heads as rows), scaled and masked.
    stile_t Stile;
    threadgroup float* res = results + parity * SLOT;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STt.template store<float, SLD, 1>(slots + eighth * SLOT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      // Sub-block: keys 2 * eighth + (lane >> 4), heads
      // 2 * (lane & 15) + {0, 1}.
      const int key = 2 * int(eighth) + int(simd_lane_id >> 4);
      const int head = 2 * int(simd_lane_id & 15);
      float a0 = 0, a1 = 0;
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < NSG; q++) {
        a0 += slots[q * SLOT + key * SLD + head];
        a1 += slots[q * SLOT + key * SLD + head + 1];
      }
      res[key * SLD + head] = a0;
      res[key * SLD + head + 1] = a1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Stile.template load<float, 1, SLD>(res);
    parity ^= 1;
    STEEL_PRAGMA_UNROLL
    for (short ir = 0; ir < 2; ir++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; jj++) {
          const short e = i * 4 + jj;
          Stile.frag_at(ir, 0)[e] = cur.cv[jj]
              ? Stile.frag_at(ir, 0)[e] * scale2
              : Limits<float>::finite_min;
        }
      }
    }

    // Online softmax (exp2 space). A row with no valid key yet keeps its
    // max at finite_min and contributes a zero P row.
    metal::vec<float, 4> new_max = max_score;
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    metal::vec<float, 4> factor;
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 4; r++) {
      if (new_max[r] > Limits<float>::finite_min) {
        factor[r] = fast::exp2(max_score[r] - new_max[r]);
        max_score[r] = new_max[r];
      } else {
        factor[r] = 1.0f;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; jj++) {
          Stile.frag_at(r >> 1, 0)[(r & 1) * 4 + jj] = 0.0f;
        }
      }
    }
    sum_score = sum_score * factor;
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);

    // O_eighth += P @ V[:, eighth] from the resident fragments (K == V).
    STEEL_PRAGMA_UNROLL
    for (short ir = 0; ir < 2; ir++) {
      STEEL_PRAGMA_UNROLL
      for (short pp = 0; pp < TDE; pp += 2) {
        otile_t::NAXFrag_t::mma(
            Otile.frag_at(ir, pp),
            Otile.frag_at(ir, pp + 1),
            Stile.frag_at(ir, 0),
            metal::false_type{},
            cur.kf[pp],
            cur.kf[pp + 1],
            metal::false_type{});
      }
    }
  };

  // Two tiles ping-pong through registers: the next tile's loads issue
  // before the current tile's step and land while it computes.
  Tile tA, tB;
  load_tile(k0, tA);
  for (int kt = k0; kt < k1; kt += 2 * BK) {
    if (kt + BK < k1) {
      load_tile(kt + BK, tB);
    }
    step(kt, tA);
    if (kt + BK >= k1) {
      break;
    }
    if (kt + 2 * BK < k1) {
      load_tile(kt + 2 * BK, tA);
    }
    step(kt + BK, tB);
  }

  // Unnormalized partials at row h * Q + j (head stride Q * splits * D);
  // the sn == 0 lanes of simdgroup 0 write the row stats with the max in
  // natural log for the shared merge.
  if (rows_valid > 0) {
    const int ld = n_queries * gqa_splits * D;
    device float* dst = out +
        (((size_t)h0 * n_queries + query_idx) * gqa_splits + split_idx) * D +
        eighth * DE;
    Otile.store_rows(dst, ld, short(rows_valid));
    if (eighth == 0 && sn == 0) {
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < 4; r++) {
        const int row = (r >> 1) * kU + (r & 1) * 8 + sm;
        if (row < rows_valid) {
          const size_t po =
              ((size_t)(h0 + row) * n_queries + query_idx) * gqa_splits +
              split_idx;
          sums[po] = sum_score[r];
          maxs[po] = max_score[r] == Limits<float>::finite_min
              ? Limits<float>::finite_min
              : max_score[r] * M_LN2_F;
        }
      }
    }
  }
}
