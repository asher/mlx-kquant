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

// Decode-width (one query) GQA attention at head_dim 512 on the tensor-op
// units, pass 1 of the sdpa_decode_gqa split-K form. It writes the same
// per-split partials as kq_sdpa_gqa_2pass_1, so kq_sdpa_gqa_2pass_2 merges
// either. The scalar kernel does four multiply-adds per K/V byte on the
// shader ALUs and falls short of bandwidth when the GPU clock drops; here
// both matmuls run as 16x32x16 tensor ops with the group's q heads as the
// fragment rows (gqa <= 8, rows past it zero).
//
// One threadgroup per (kv head, batch row, key split), eight simdgroups, a
// tile of 256 keys. For S = Q K^T each simdgroup owns 32 keys over the full
// head dim, with Q staged once in threadgroup memory. The simdgroups
// exchange their tile maxima, park P (heads x 256 keys, float) in
// threadgroup memory, and for O += P V each owns a 64-column slice of the
// head dim over all 256 keys. Every K and V row is read once per
// threadgroup. Inside a fragment the 16-wide contraction index is a
// permutation of the head dim, and Q uses the same one: lane quad g reads
// dims 16g..16g+15 of each 64-dim block, 32 contiguous bytes per row.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_sdpa_gqa_nax_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& gqa [[buffer(12)]],
    const constant size_t& k_batch_stride [[buffer(13)]],
    const constant size_t& v_batch_stride [[buffer(14)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  using namespace kq_sdpa_nax;
  constexpr int D = 512;
  constexpr int NSG = 8;
  constexpr int KS = 32; // keys per simdgroup in S
  constexpr int BK = NSG * KS; // keys per tile
  constexpr int DC = D / NSG; // output columns per simdgroup in P V
  constexpr int GM = 8; // q heads per group (fragment rows used)
  constexpr int PLD = BK + 4; // P row stride (floats)
  using T4 = metal::vec<T, 4>;
  using frag_t = typename BaseNAXFrag::dtype_frag_t<T>;
  using ffrag_t = typename BaseNAXFrag::dtype_frag_t<float>;

  threadgroup T sQ[GM * D];
  threadgroup float sP[GM * PLD];
  threadgroup float sM[NSG * GM]; // tile maxima, then row sums

  const int n_kv_heads = tpg.x;
  const size_t hb = (size_t)tid.y * n_kv_heads + tid.x;
  const size_t k_bh = tid.y * k_batch_stride + tid.x * k_head_stride;
  const size_t v_bh = tid.y * v_batch_stride + tid.x * v_head_stride;
  const int split_idx = tid.z;
  const short sg = simd_group_id;
  const short2 sc = BaseNAXFrag::get_coord();
  const short fm = sc.y; // head row of this lane (and fm + 8, unused)
  const short fn = sc.x;
  const short goff = (fn >> 2) * 16; // lane quad's 16-dim slice
  const bool row_lead = (simd_lane_id & 9) == 0; // fn == 0

  // Whole tiles per split: a partial tile costs as long as a full one.
  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  // Partials row for head fm of this group ([B, Hq, 1, splits, D]).
  const size_t po = (hb * gqa + fm) * gqa_splits + split_idx;
  device float* dst = out + po * D + sg * DC + goff;

  if (k0 >= k1) {
    // Empty split: pass 2 folds it at zero weight.
    if (fm < gqa) {
      STEEL_PRAGMA_UNROLL
      for (short cc = 0; cc < 4; cc++) {
        *(device float4*)(dst + cc * 4) = float4(0);
      }
      if (sg == 0 && row_lead) {
        sums[po] = 0;
        maxs[po] = Limits<float>::finite_min;
      }
    }
    return;
  }

  {
    const device T4* q4 = (const device T4*)(queries + hb * gqa * D);
    threadgroup T4* sQ4 = (threadgroup T4*)sQ;
    for (int i = simd_lane_id + 32 * sg; i < GM * D / 4; i += 32 * NSG) {
      sQ4[i] = i < gqa * (D / 4) ? q4[i] : T4(T(0));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* kbase = keys + k_bh + goff;
  const device T* vbase = values + v_bh + sg * DC + goff;
  const threadgroup T* qrow = sQ + fm * D + goff;
  const float scale2 = scale * M_LOG2E_F;

  ffrag_t O[4];
  STEEL_PRAGMA_UNROLL
  for (short cc = 0; cc < 4; cc++) {
    O[cc] = ffrag_t(0);
  }
  float m_run = Limits<float>::finite_min; // row fm, log2 domain
  float l_run = 0; // this lane's share of row fm's sum

  for (int kt = k0; kt < k1; kt += BK) {
    // S (heads x 32 keys) for this simdgroup's keys. Lane rows are keys
    // kb + fm, + 8, + 16, + 24; rows past N read row N - 1 and mask.
    const int kb = kt + sg * KS;
    ffrag_t S0 = ffrag_t(0);
    ffrag_t S1 = ffrag_t(0);
    if (kb < k1) {
      const device T* kr[4];
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < 4; r++) {
        kr[r] = kbase + (size_t)min(kb + fm + 8 * r, N - 1) * k_seq_stride;
      }
      STEEL_PRAGMA_UNROLL
      for (short blk = 0; blk < D / 64; blk++) {
        T4 kv[4][4];
        T4 qv[4];
        STEEL_PRAGMA_UNROLL
        for (short s = 0; s < 4; s++) {
          STEEL_PRAGMA_UNROLL
          for (short r = 0; r < 4; r++) {
            kv[r][s] = *(const device T4*)(kr[r] + blk * 64 + s * 4);
          }
          qv[s] = *(const threadgroup T4*)(qrow + blk * 64 + s * 4);
        }
        STEEL_PRAGMA_UNROLL
        for (short s = 0; s < 4; s++) {
          frag_t a, b0, b1;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; j++) {
            a[j] = qv[s][j];
            a[4 + j] = T(0);
            b0[j] = kv[0][s][j];
            b0[4 + j] = kv[1][s][j];
            b1[j] = kv[2][s][j];
            b1[4 + j] = kv[3][s][j];
          }
          BaseNAXFrag::mma(
              S0, S1, a, metal::false_type{}, b0, b1, metal::true_type{});
        }
      }
    }

    // Scale and mask (keys kb + fn + j in S0, kb + 16 + fn + j in S1), then
    // the tile max of row fm across the simdgroups.
    float sv[8];
    float tmax = Limits<float>::finite_min;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 4; j++) {
      sv[j] = kb + fn + j < k1 ? S0[j] * scale2 : Limits<float>::finite_min;
      sv[4 + j] =
          kb + 16 + fn + j < k1 ? S1[j] * scale2 : Limits<float>::finite_min;
      tmax = max(tmax, max(sv[j], sv[4 + j]));
    }
    tmax = max(tmax, simd_shuffle_xor(tmax, ushort(1)));
    tmax = max(tmax, simd_shuffle_xor(tmax, ushort(8)));
    if (row_lead) {
      sM[sg * GM + fm] = tmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float new_m = m_run;
    STEEL_PRAGMA_UNROLL
    for (short q = 0; q < NSG; q++) {
      new_m = max(new_m, sM[q * GM + fm]);
    }
    const float factor = fast::exp2(m_run - new_m);
    m_run = new_m;
    float4 p0, p1;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 4; j++) {
      p0[j] = fast::exp2(sv[j] - new_m);
      p1[j] = fast::exp2(sv[4 + j] - new_m);
    }
    l_run = l_run * factor + (p0[0] + p0[1] + p0[2] + p0[3]) +
        (p1[0] + p1[1] + p1[2] + p1[3]);
    *(threadgroup float4*)(sP + fm * PLD + sg * KS + fn) = p0;
    *(threadgroup float4*)(sP + fm * PLD + sg * KS + 16 + fn) = p1;
    STEEL_PRAGMA_UNROLL
    for (short cc = 0; cc < 4; cc++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; j++) {
        O[cc][j] *= factor;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P V over the tile's keys for this simdgroup's 64 columns. Lane
    // rows are keys kt + 16 kc + fm and + 8; masked keys carry P = 0.
    STEEL_PRAGMA_UNROLL
    for (short kc = 0; kc < BK / 16; kc++) {
      if (kt + kc * 16 < k1) {
        const int r0 = kt + kc * 16 + fm;
        const device T* vr0 = vbase + (size_t)min(r0, N - 1) * v_seq_stride;
        const device T* vr1 = vbase + (size_t)min(r0 + 8, N - 1) * v_seq_stride;
        T4 v0[4], v1[4];
        STEEL_PRAGMA_UNROLL
        for (short s = 0; s < 4; s++) {
          v0[s] = *(const device T4*)(vr0 + s * 4);
          v1[s] = *(const device T4*)(vr1 + s * 4);
        }
        const float4 pa =
            *(const threadgroup float4*)(sP + fm * PLD + kc * 16 + fn);
        ffrag_t a;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          a[j] = pa[j];
          a[4 + j] = 0.0f;
        }
        STEEL_PRAGMA_UNROLL
        for (short pp = 0; pp < 2; pp++) {
          frag_t b0, b1;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; j++) {
            b0[j] = v0[2 * pp][j];
            b0[4 + j] = v1[2 * pp][j];
            b1[j] = v0[2 * pp + 1][j];
            b1[4 + j] = v1[2 * pp + 1][j];
          }
          BaseNAXFrag::mma(
              O[2 * pp],
              O[2 * pp + 1],
              a,
              metal::false_type{},
              b0,
              b1,
              metal::false_type{});
        }
      }
    }
  }

  // Row sums across the lane quad pair and the simdgroups; the max is the
  // same in every simdgroup.
  l_run += simd_shuffle_xor(l_run, ushort(1));
  l_run += simd_shuffle_xor(l_run, ushort(8));
  if (row_lead) {
    sM[sg * GM + fm] = l_run;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (fm < gqa) {
    STEEL_PRAGMA_UNROLL
    for (short cc = 0; cc < 4; cc++) {
      *(device float4*)(dst + cc * 4) =
          float4(O[cc][0], O[cc][1], O[cc][2], O[cc][3]);
    }
    if (sg == 0 && row_lead) {
      float l = 0;
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < NSG; q++) {
        l += sM[q * GM + fm];
      }
      sums[po] = l;
      maxs[po] = m_run * M_LN2_F;
    }
  }
}
