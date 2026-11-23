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

#include "mlx/backend/metal/kernels/kq_sdpa_rows_meta.h"
#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;
using namespace mlx::steel;

constant int gqa_splits [[function_constant(2)]];
constant bool gqa_has_starts [[function_constant(4)]];

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
//
// With starts (a left-padded batched KV cache), batch row b attends keys
// [starts[b], N). Whole tiles below the start are skipped, keys below it
// in the first tile are masked, and K and V rows are read from the start
// up, so the pad rows are never read.
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
    const device int* starts [[buffer(15)]],
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
  // First key of this batch row, and the first tile that holds one.
  const int row_start = gqa_has_starts ? max(0, starts[tid.y]) : 0;
  const int kt0 = max(k0, (row_start / BK) * BK);

  // Partials row for head fm of this group ([B, Hq, 1, splits, D]).
  const size_t po = (hb * gqa + fm) * gqa_splits + split_idx;
  device float* dst = out + po * D + sg * DC + goff;

  if (max(k0, row_start) >= k1) {
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

  // The first tile holds a key at or above row_start, so the running max is
  // finite from the first tile on.
  for (int kt = kt0; kt < k1; kt += BK) {
    // S (heads x 32 keys) for this simdgroup's keys. Lane rows are keys
    // kb + fm, + 8, + 16, + 24; rows past N read row N - 1, rows below
    // row_start read row row_start, and both mask.
    const int kb = kt + sg * KS;
    ffrag_t S0 = ffrag_t(0);
    ffrag_t S1 = ffrag_t(0);
    if (kb < k1 && kb + KS > row_start) {
      const device T* kr[4];
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < 4; r++) {
        kr[r] = kbase +
            (size_t)min(max(kb + fm + 8 * r, row_start), N - 1) * k_seq_stride;
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
      const int ka = kb + fn + j;
      const int kc = ka + 16;
      sv[j] = ka < k1 && ka >= row_start ? S0[j] * scale2
                                         : Limits<float>::finite_min;
      sv[4 + j] = kc < k1 && kc >= row_start ? S1[j] * scale2
                                             : Limits<float>::finite_min;
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
      if (kt + kc * 16 < k1 && kt + kc * 16 + 16 > row_start) {
        const int r0 = kt + kc * 16 + fm;
        const device T* vr0 =
            vbase + (size_t)min(max(r0, row_start), N - 1) * v_seq_stride;
        const device T* vr1 =
            vbase + (size_t)min(max(r0 + 8, row_start), N - 1) * v_seq_stride;
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

// Speculative-verify attention on the tensor-op units, pass 1 of the
// sdpa_fa_verify split-K form, after MLX's head-dim split NAX attention
// (attention_nax_dsplit). It writes the partials of
// kq_sdpa_fa_verify_2pass_1 (kq_sdpa.h), so kq_sdpa_gqa_2pass_2 merges
// either. The caller folds the GQA group into the query rows: q is
// [B, Hkv, n_rows, D], and folded row r attends keys up to
// N - q_len + (r % q_len). sdpa_decode_gqa runs it at head_dim 128 with
// q_len 1 and the group as the rows. With starts, batch row b
// attends keys [starts[b], N): its tiles begin at the start, so pad rows
// are never read.
//
// One threadgroup per (kv head, batch row, key split), WM x WN simdgroups.
// Simdgroup (row group, d group) owns query rows 16 * row group onward and
// the d group's D / WN columns: its Q slice stays in registers, it reads
// its columns of each 32-key K and V tile from device memory into
// fragments, and keeps O for its columns only. The WN partial scores of a
// row group are summed in threadgroup memory in a fixed order, and every
// d group runs the same row softmax on the sum. A tile past a row's limit
// leaves the row's running max at finite_min and its P row at zero; an
// empty split writes O = 0, sum = 0 and max = finite_min.
template <typename T, int D, int WM, int WN, bool PV_SPLIT = false>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
kq_sdpa_fa_verify_2pass_1_nax(
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
    const constant int& q_len [[buffer(12)]],
    const constant int& n_rows [[buffer(13)]],
    const constant size_t& k_batch_stride [[buffer(14)]],
    const constant size_t& v_batch_stride [[buffer(15)]],
    const device int* starts [[buffer(16)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  using namespace kq_sdpa_nax;
  constexpr short kU = 16;
  constexpr int BK = 32; // keys per tile
  constexpr short TK = BK / kU;
  constexpr short TDh = D / kU / WN; // column fragments per simdgroup
  constexpr int BDh = D / WN;
  static_assert(TDh % 2 == 0, "P @ V accumulates column fragments in pairs");

  using otile_t = NAXTile<float, 1, TDh>;
  using stile_t = NAXTile<float, 1, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  threadgroup float s_xchg[WM][WN][TK * kEPF * 32];

  const short row_group = simd_group_id / WN;
  const short d_group = simd_group_id % WN;
  const int n_kv_heads = tpg.x;
  const int split_idx = tid.z;
  const size_t qbh = (size_t)tid.y * n_kv_heads + tid.x;
  const int tm = kU * row_group;

  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k1 = min(split_idx * chunk + chunk, N);
  const int k0 =
      max(split_idx * chunk, gqa_has_starts ? max(0, starts[tid.y]) : 0);

  const int k_ld = int(k_seq_stride);
  const int v_ld = int(v_seq_stride);
  const device T* K = keys + tid.y * k_batch_stride + tid.x * k_head_stride +
      (size_t)k0 * k_seq_stride + d_group * BDh;
  const device T* V = values + tid.y * v_batch_stride + tid.x * v_head_stride +
      (size_t)k0 * v_seq_stride + d_group * BDh;

  const short2 sc = otile_t::NAXFrag_t::get_coord();
  const short sm = sc.y;
  const short sn = sc.x;

  // Q slice resident in registers; rows past n_rows load as zero.
  const short q_rows = short(metal::clamp(n_rows - tm, 0, int(kU)));
  NAXTile<T, 1, 1> Qtiles[TDh];
  {
    const device T* qs = queries + (qbh * n_rows + tm) * D + d_group * BDh;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id++) {
      Qtiles[id].load_rows(qs + id * kU, D, q_rows);
    }
  }

  otile_t Otile;
  Otile.clear();
  metal::vec<float, kRowsPT> max_score;
  metal::vec<float, kRowsPT> sum_score{0};
  // Highest key of this lane's rows (tm + sm and tm + sm + 8).
  int lim[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<float>::finite_min;
    lim[i] = N - q_len + ((tm + sm + i * 8) % q_len);
  }
  const int lim_min = N - q_len; // every row reaches it
  const float scale2 = scale * M_LOG2E_F;

  for (int kt = k0; kt < k1; kt += BK) {
    const int krem = min(k1 - kt, BK);
    const bool full = krem == BK;

    stile_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const device T* kp = K + ik * kU * k_ld + id * kU;
        if (full) {
          Ktile.load(kp, k_ld);
        } else {
          Ktile.load_rows(kp, k_ld, short(krem - ik * kU));
        }
        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }
    }

    if (WN > 1) {
      // Sum the WN partial scores of the row group in a fixed order.
      threadgroup float* slot = s_xchg[row_group][d_group];
      const short base = short(simd_lane_id) * (TK * kEPF);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) {
          slot[base + ik * kEPF + i] = Stile.frag_at(0, ik)[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& s = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) {
          s[i] = s_xchg[row_group][0][base + ik * kEPF + i];
        }
        STEEL_PRAGMA_UNROLL
        for (short p = 1; p < WN; p++) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kEPF; i++) {
            s[i] += s_xchg[row_group][p][base + ik * kEPF + i];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }
    // Only the split tail or a tile reaching past N - q_len masks.
    if (!full || kt + BK - 1 > lim_min) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const int c = kt + ik * kU + sn + jj;
            const short loc = ii * stile_t::kFragThrCols + jj;
            if (c >= k1 || c > lim[ii]) {
              fg[loc] = Limits<float>::finite_min;
            }
          }
        }
      }
    }

    // Online softmax (exp2 space). A row with no valid key yet keeps its
    // max at finite_min and contributes a zero P row.
    metal::vec<float, kRowsPT> new_max = max_score;
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    metal::vec<float, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      if (new_max[i] > Limits<float>::finite_min) {
        factor[i] = fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
      } else {
        factor[i] = 1.0f;
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            Stile.frag_at(0, ik)[i * stile_t::kFragThrCols + jj] = 0.0f;
          }
        }
      }
    }
    sum_score = sum_score * factor;
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // O += P @ V over this simdgroup's columns. The tensor op takes a float
    // P at reduced precision; with PV_SPLIT, P runs as two T parts, hi and
    // lo, which keeps the simdgroup kernel's rounding at a second matmul.
    using pfrag_t = typename BaseNAXFrag::dtype_frag_t<T>;
    pfrag_t p_hi[TK], p_lo[TK];
    if (PV_SPLIT) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) {
          const float x = Stile.frag_at(0, ik)[i];
          p_hi[ik][i] = T(x);
          p_lo[ik][i] = T(x - float(p_hi[ik][i]));
        }
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        const device T* vp = V + ik * kU * v_ld + id * kU;
        if (full) {
          Vtile.load(vp, v_ld);
        } else {
          Vtile.load_rows(vp, v_ld, short(krem - ik * kU));
        }
        if (PV_SPLIT) {
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            otile_t::NAXFrag_t::mma(
                Otile.frag_at(0, id),
                Otile.frag_at(0, id + 1),
                h == 0 ? p_hi[ik] : p_lo[ik],
                metal::false_type{},
                Vtile.frag_at(0, 0),
                Vtile.frag_at(0, 1),
                metal::false_type{});
          }
        } else {
          otile_t::NAXFrag_t::mma(
              Otile.frag_at(0, id),
              Otile.frag_at(0, id + 1),
              Stile.frag_at(0, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    K += BK * k_ld;
    V += BK * v_ld;
  }

  // Unnormalized partials [B, Hkv, n_rows, splits, D]; d group 0 writes the
  // row stats, with the max in natural log for the shared merge.
  if (q_rows > 0) {
    device float* dst = out +
        ((qbh * n_rows + tm) * gqa_splits + split_idx) * D + d_group * BDh;
    Otile.store_rows(dst, gqa_splits * D, q_rows);
    if (d_group == 0 && sn == 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        const int row = tm + sm + i * 8;
        if (row < n_rows) {
          const size_t po = (qbh * n_rows + row) * gqa_splits + split_idx;
          sums[po] = sum_score[i];
          maxs[po] = max_score[i] == Limits<float>::finite_min
              ? Limits<float>::finite_min
              : max_score[i] * M_LN2_F;
        }
      }
    }
  }
}

// Shared-tile pass 1 of sdpa_paged_rows on the tensor-op units. It walks
// what kq_sdpa_paged_rows_u (kq_sdpa.h) walks: the same grid, the same
// split of the key axis, the union of a window's page lists marked and
// compacted the same way, and the same partials, so kq_sdpa_gqa_2pass_2
// merges either. The window's gqa * win rows are folded head-major, window
// query fastest. Simdgroup (row group, d group) owns 16 of them and D / WN
// columns: its Q slice stays in registers, and it reads its columns of a
// tile's 32 key rows from device memory straight into fragments through the
// compacted page list, so K and V take no staging pass. With WN 2, d group
// 1 parks its partial scores in threadgroup memory and d group 0 adds them
// to its own and parks the sum for both. A key counts for a row when the
// row's query listed its page and it lies below the query's end; rows at or
// past the window's last end are never read.
template <typename T, int D, int PR, int WN>
[[kernel, max_total_threads_per_threadgroup(512)]] void
kq_sdpa_paged_rows_u_nax(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const device int* pages [[buffer(6)]],
    const device int* ends [[buffer(7)]],
    const device T* keys_tail [[buffer(8)]],
    const device T* values_tail [[buffer(9)]],
    const constant KQPagedRowsMeta& pm [[buffer(10)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  using namespace kq_sdpa_nax;
  constexpr short kU = 16;
  constexpr int BK = 32; // keys per tile
  constexpr int PPT = BK / PR; // pages per tile
  constexpr short TK = BK / kU;
  constexpr short TDh = D / kU / WN; // column fragments per simdgroup
  constexpr int BDh = D / WN;
  constexpr int WM_MAX = 8; // row groups, 128 rows
  static_assert(WN == 1 || WN == 2, "one or two d groups");
  static_assert(TK == 2, "a tile is one 16x32 score fragment pair");
  static_assert(TDh % 2 == 0, "P @ V accumulates column fragments in pairs");
  static_assert(
      PR % 4 == 0 && BK % PR == 0, "a lane's 4 score columns share a page");
  using T4 = metal::vec<T, 4>;
  using otile_t = NAXTile<float, 1, TDh>;
  using stile_t = NAXTile<float, 1, TK>;
  using kfrag_t = typename BaseNAXFrag::dtype_frag_t<T>;
  constexpr short kEPF = stile_t::kElemsPerFrag;
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  threadgroup float s_xchg[WN > 1 ? WM_MAX : 1][TK * kEPF * 32];
  threadgroup atomic_uint marks[KQ_PR_UNION_MAX];
  threadgroup int ulist[KQ_PR_UNION_MAX];
  threadgroup uint umask[KQ_PR_UNION_MAX];
  threadgroup int ucount[1];

  const int nq = pm.q_len;
  const int W = pm.win;
  const int G = pm.gqa;
  const int n_win = (nq + W - 1) / W;
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y / n_win;
  const int l0 = (tid.y % n_win) * W;
  const int wl = min(W, nq - l0); // queries in this window
  const int split_idx = tid.z;
  const int num_q_heads = int(tpg.x) * G;
  const int N = pm.N;
  const int n_pages = pm.n_pages;
  const int n_all = n_pages + pm.tail_page;
  const int base_len = pm.base_len;
  const int flat = int(simd_group_id) * 32 + int(simd_lane_id);
  const int NT = int(tptg.x * tptg.y);
  const short row_group = short(simd_group_id) / WN;
  const short d_group = short(simd_group_id) % WN;
  const int tm = kU * row_group;
  const int n_rows = G * wl;

  // This split's pages, as kq_sdpa_paged_rows_u takes them.
  const int NP = (N + PR - 1) / PR;
  const int n_gran = (NP + PPT - 1) / PPT;
  const int nr = ((n_gran + gqa_splits - 1) / gqa_splits) * PPT;
  const int stride = PPT * gqa_splits;

  const device int* plist = pages +
      ((size_t)(batch_idx * pm.page_heads +
                (pm.page_heads > 1 ? kv_head_idx : 0)) *
       nq) *
          n_pages;
  const device int* wends = ends + batch_idx * nq;
  int wend = 0;
  for (int j = 0; j < wl; j++) {
    wend = max(wend, wends[l0 + j]);
  }
  const int row_end = min(wend, N);

  const short2 sc = BaseNAXFrag::get_coord();
  const short sm = sc.y;
  const short sn = sc.x;

  // This lane's rows, tm + sm and tm + sm + 8: window query lq of head g.
  bool live[kRowsPT];
  uint qbit[kRowsPT];
  int lim[kRowsPT];
  size_t qrow[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; i++) {
    const int r = tm + sm + i * 8;
    live[i] = r < n_rows;
    const int rr = live[i] ? r : 0;
    const int g = rr / wl;
    const int lq = rr - g * wl;
    qbit[i] = live[i] ? 1u << lq : 0u;
    lim[i] = min(wends[l0 + lq], N) - 1;
    qrow[i] =
        (size_t)(batch_idx * num_q_heads + kv_head_idx * G + g) * nq + l0 + lq;
  }

  // Q slice in registers; rows past the window's zero.
  kfrag_t Qf[TDh];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; i++) {
    const device T* qp = queries + qrow[i] * D + d_group * BDh + sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id++) {
      const T4 v = *((const device T4*)(qp + id * kU));
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; j++) {
        Qf[id][i * 4 + j] = live[i] ? v[j] : T(0);
      }
    }
  }

  otile_t Otile;
  Otile.clear();
  metal::vec<float, kRowsPT> max_score;
  metal::vec<float, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<float>::finite_min;
  }
  const float scale2 = pm.scale * M_LOG2E_F;

  const size_t col0 = d_group * BDh + sn;
  const device T* kbase =
      keys + batch_idx * pm.k_batch + (size_t)kv_head_idx * pm.k_head + col0;
  const device T* vbase =
      values + batch_idx * pm.v_batch + (size_t)kv_head_idx * pm.v_head + col0;
  const device T* ktail = keys_tail + batch_idx * pm.kt_batch +
      (size_t)kv_head_idx * pm.kt_head + col0;
  const device T* vtail = values_tail + batch_idx * pm.vt_batch +
      (size_t)kv_head_idx * pm.vt_head + col0;

  for (int r0 = 0; r0 < nr; r0 += KQ_PR_UNION_MAX) {
    const int rn = min(nr - r0, KQ_PR_UNION_MAX);
    // Mark, per page of the round, which of the window's queries listed it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = flat; i < rn; i += NT) {
      atomic_store_explicit(&marks[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = flat; i < wl * n_all; i += NT) {
      const int mq = i / n_all;
      const int pi = i - mq * n_all;
      const int ml = l0 + mq;
      const int pg =
          pi < n_pages ? plist[(size_t)ml * n_pages + pi] : wends[ml] / PR;
      if (pg >= 0 && pg < NP && (pg / PPT) % gqa_splits == split_idx) {
        const int slot = (pg / stride) * PPT + pg % PPT - r0;
        if (slot >= 0 && slot < rn) {
          atomic_fetch_or_explicit(
              &marks[slot], 1u << mq, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Compact the marked pages, in key order.
    if (simd_group_id == 0) {
      int base = 0;
      for (int j0 = 0; j0 < rn; j0 += 32) {
        const int i = j0 + int(simd_lane_id);
        const uint m =
            i < rn ? atomic_load_explicit(&marks[i], memory_order_relaxed) : 0u;
        const int hit = m != 0u ? 1 : 0;
        const int at = base + simd_prefix_exclusive_sum(hit);
        if (hit) {
          const int si = r0 + i;
          ulist[at] = (si / PPT) * stride + split_idx * PPT + si % PPT;
          umask[at] = m;
        }
        base += simd_sum(hit);
      }
      if (simd_lane_id == 0) {
        ucount[0] = base;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int cnt = ucount[0];

    for (int t0 = 0; t0 < cnt; t0 += PPT) {
      // This lane's K / V rows of the tile, keys kf * 16 + sm + 8 i. A row
      // past the compacted list or the window's last end reads row 0 and
      // selects zero, so every lane issues the same loads.
      const device T* kp[TK][2];
      const device T* vp[TK][2];
      bool kok[TK][2];
      STEEL_PRAGMA_UNROLL
      for (short kf = 0; kf < TK; kf++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
          const int j = kf * kU + sm + i * 8;
          const int pi = t0 + j / PR;
          const int pg = pi < cnt ? ulist[pi] : -1;
          const int kg = pg * PR + j % PR;
          kok[kf][i] = pg >= 0 && kg < row_end;
          const int kc = kok[kf][i] ? kg : 0;
          const bool in_base = kc < base_len;
          kp[kf][i] = in_base ? kbase + (size_t)kc * pm.k_seq
                              : ktail + (size_t)(kc - base_len) * pm.kt_seq;
          vp[kf][i] = in_base ? vbase + (size_t)kc * pm.v_seq
                              : vtail + (size_t)(kc - base_len) * pm.vt_seq;
        }
      }
      // This lane's score columns kf * 16 + sn + jj, all on one page: the
      // page's first such key (-1 past the list) and its query marks.
      int ckg[TK];
      uint cmask[TK];
      STEEL_PRAGMA_UNROLL
      for (short kf = 0; kf < TK; kf++) {
        const int j = kf * kU + sn;
        const int pi = t0 + j / PR;
        const int pg = pi < cnt ? ulist[pi] : -1;
        ckg[kf] = pg >= 0 ? pg * PR + j % PR : -1;
        cmask[kf] = pi < cnt ? umask[pi] : 0u;
      }

      // Partial S = Q K^T over this simdgroup's columns: 16 rows x 32 keys.
      stile_t Stile;
      Stile.clear();
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        kfrag_t kfr[TK];
        STEEL_PRAGMA_UNROLL
        for (short kf = 0; kf < TK; kf++) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 2; i++) {
            const T4 v = *((const device T4*)(kp[kf][i] + id * kU));
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
              kfr[kf][i * 4 + j] = kok[kf][i] ? v[j] : T(0);
            }
          }
        }
        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, 0),
            Stile.frag_at(0, 1),
            Qf[id],
            metal::false_type{},
            kfr[0],
            kfr[1],
            metal::true_type{});
      }

      if (WN > 1) {
        // d group 0's partial plus d group 1's, in that order, for both.
        threadgroup float* slot = s_xchg[row_group];
        const short base = short(simd_lane_id) * (TK * kEPF);
        if (d_group == 1) {
          STEEL_PRAGMA_UNROLL
          for (short kf = 0; kf < TK; kf++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kEPF; i++) {
              slot[base + kf * kEPF + i] = Stile.frag_at(0, kf)[i];
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d_group == 0) {
          STEEL_PRAGMA_UNROLL
          for (short kf = 0; kf < TK; kf++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kEPF; i++) {
              const float s =
                  Stile.frag_at(0, kf)[i] + slot[base + kf * kEPF + i];
              Stile.frag_at(0, kf)[i] = s;
              slot[base + kf * kEPF + i] = s;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d_group == 1) {
          STEEL_PRAGMA_UNROLL
          for (short kf = 0; kf < TK; kf++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kEPF; i++) {
              Stile.frag_at(0, kf)[i] = slot[base + kf * kEPF + i];
            }
          }
        }
      }

      // Scale; a key counts for a row when the row's query listed its page
      // and it lies at or below the query's last key.
      STEEL_PRAGMA_UNROLL
      for (short kf = 0; kf < TK; kf++) {
        thread auto& fg = Stile.frag_at(0, kf);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < kRowsPT; ii++) {
          const bool member = ckg[kf] >= 0 && (cmask[kf] & qbit[ii]) != 0u;
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 4; jj++) {
            const short loc = ii * 4 + jj;
            fg[loc] = member && ckg[kf] + jj <= lim[ii]
                ? fg[loc] * scale2
                : Limits<float>::finite_min;
          }
        }
      }

      // Online softmax (exp2 space). A row with no key yet keeps its max at
      // finite_min and contributes a zero P row.
      metal::vec<float, kRowsPT> new_max = max_score;
      Stile.template row_reduce<MaxOp>(new_max);
      Stile.template row_bin_op<ExpSubOp>(new_max);
      metal::vec<float, kRowsPT> factor;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        if (new_max[i] > Limits<float>::finite_min) {
          factor[i] = fast::exp2(max_score[i] - new_max[i]);
          max_score[i] = new_max[i];
        } else {
          factor[i] = 1.0f;
          STEEL_PRAGMA_UNROLL
          for (short kf = 0; kf < TK; kf++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < 4; jj++) {
              Stile.frag_at(0, kf)[i * 4 + jj] = 0.0f;
            }
          }
        }
      }
      sum_score = sum_score * factor;
      Stile.template row_reduce<SumOp>(sum_score);
      Otile.template row_bin_op<MulOp>(factor);

      // O += P @ V over this simdgroup's columns.
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kf = 0; kf < TK; kf++) {
          kfrag_t v0, v1;
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 2; i++) {
            const T4 a = *((const device T4*)(vp[kf][i] + id * kU));
            const T4 b = *((const device T4*)(vp[kf][i] + (id + 1) * kU));
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
              v0[i * 4 + j] = kok[kf][i] ? a[j] : T(0);
              v1[i * 4 + j] = kok[kf][i] ? b[j] : T(0);
            }
          }
          otile_t::NAXFrag_t::mma(
              Otile.frag_at(0, id),
              Otile.frag_at(0, id + 1),
              Stile.frag_at(0, kf),
              metal::false_type{},
              v0,
              v1,
              metal::false_type{});
        }
      }
    }
  }

  // Unnormalized partials [B, Hq, q_len, gqa_splits, D]; d group 0 writes
  // the row stats, with the max in natural log for the shared merge.
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; i++) {
    if (!live[i]) {
      continue;
    }
    const size_t po = qrow[i] * gqa_splits + split_idx;
    device float* orow = out + po * D + col0;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id++) {
      const thread auto& fg = Otile.frag_at(0, id);
      *((device float4*)(orow + id * kU)) =
          float4(fg[i * 4], fg[i * 4 + 1], fg[i * 4 + 2], fg[i * 4 + 3]);
    }
    if (d_group == 0 && sn == 0) {
      sums[po] = sum_score[i];
      maxs[po] = max_score[i] == Limits<float>::finite_min
          ? Limits<float>::finite_min
          : max_score[i] * M_LN2_F;
    }
  }
}
