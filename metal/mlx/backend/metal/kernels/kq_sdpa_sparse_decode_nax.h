// The sparse decode split kernel (kq_sdpa_sparse_decode.h) on the tensor-op
// (NAX) units, at a 512-wide latent with or without a score-only tail of 64.
// It writes the per-split partials that kq_sdpa_sparse_decode_merge reads,
// so the merge finishes either kernel.
//
// One threadgroup per (32-head strip, query, key split). Eight simdgroups
// each own a 64-column eighth of the latent: the strip's queries over those
// columns stay in registers, and the 16 key rows of a tile load straight
// into fragment registers from where they lie, window rows by position and
// pool rows through the query's index list, one tile ahead of the matmuls.
// The fragments serve both K and V. A simdgroup computes the partial
// S^T = K @ Q^T over its columns and parks it in threadgroup scratch; the
// partials are summed by sub-block and read back transposed (heads as
// rows) for the row softmax, then P @ V runs from the same fragments. With
// a tail a ninth simdgroup holds the tail columns of the queries and the
// rows and adds its partial to the sum; it has no output columns.
//
// A masked, padded or out-of-range slot loads a row that exists and scores
// finite_min, so its weight is zero and its values never count.
#pragma once

#include "mlx/backend/metal/kernels/kq_sdpa_nax.h"
#include "mlx/backend/metal/kernels/kq_sdpa_sparse_decode_params.h"

template <typename T, typename IdxT, int TAIL>
[[ kernel, max_total_threads_per_threadgroup((TAIL > 0 ? 9 : 8) * 32) ]] void
kq_sdpa_sparse_decode_nax(
    const device T* Q [[buffer(0)]],
    const device T* Win [[buffer(1)]],
    const device T* Pool [[buffer(2)]],
    const device IdxT* Idx [[buffer(3)]],
    const device bool* WinMask [[buffer(4)]],
    const device bool* SelMask [[buffer(5)]],
    device float* Oacc [[buffer(6)]],
    device float* Ms [[buffer(7)]],
    device float* Ls [[buffer(8)]],
    const constant KQSdpaSparseDecodeParams* p [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  using namespace kq_sdpa_nax;
  constexpr int DL = 512; // latent columns: scored and summed
  constexpr int D = DL + TAIL; // row width
  constexpr int NSG = 8; // simdgroups with output columns
  constexpr int NSS = TAIL > 0 ? NSG + 1 : NSG; // simdgroups that score
  constexpr int DE = DL / NSG; // columns per simdgroup
  constexpr short kU = 16;
  constexpr int BK = 16; // keys per tile
  constexpr short TDE = DE / kU; // column fragments per simdgroup
  constexpr int BQ = 32; // heads per strip (two row fragments)
  constexpr int SLD = 34; // exchange slot row stride (floats)
  constexpr int SLOT = BK * SLD;
  constexpr float kEmpty = -1e30f; // kSdpaSparseNegBig of the merge
  static_assert(TAIL == 0 || TAIL == DE, "the tail is one simdgroup wide");
  using T4 = metal::vec<T, 4>;

  using otile_t = NAXTile<float, 2, TDE>;
  using stile_t = NAXTile<float, 2, 1>; // heads x keys
  using sttile_t = NAXTile<float, 1, 2>; // keys x heads
  using kfrag_t = typename BaseNAXFrag::dtype_frag_t<T>;

  // A partial slot per scoring simdgroup plus two result buffers (tile
  // parity), each [16 keys][34] floats.
  threadgroup float slots[(NSS + 2) * SLOT];
  threadgroup float* results = slots + NSS * SLOT;

  const int part = int(simd_group_id); // column block; NSG is the tail
  const bool is_tail = part >= NSG;
  const int s = int(tid.x);
  const int h0 = int(tid.y) * BQ;
  const int b = int(tid.z) / p->L;
  const int l = int(tid.z) % p->L;
  const int rows_valid = metal::clamp(p->H - h0, 0, BQ);

  const int W = p->W;
  const int total = W + p->N;
  const int k0 = s * p->keys_per_split;
  const int k1 = metal::min(total, k0 + p->keys_per_split);
  const device T* win_b = Win + size_t(b) * p->win_strides[0];
  const device T* pool_b = Pool + size_t(b) * p->pool_strides[0];
  const device IdxT* idx_bl =
      Idx + size_t(b) * p->idx_strides[0] + size_t(l) * p->idx_strides[1];
  const device bool* wm = p->has_win_mask
      ? WinMask + size_t(b) * p->win_mask_strides[0] +
          size_t(l) * p->win_mask_strides[1]
      : nullptr;
  const device bool* sm_ = p->has_sel_mask
      ? SelMask + size_t(b) * p->sel_mask_strides[0] +
          size_t(l) * p->sel_mask_strides[1]
      : nullptr;
  const device T* safe_row = W > 0 ? win_b : pool_b;

  const short2 sc = BaseNAXFrag::get_coord();
  const short sm = sc.y;
  const short sn = sc.x;
  const int col0 = part * DE + sn;

  // Q fragments of this strip over this simdgroup's columns, resident for
  // the whole walk (heads past H zero).
  NAXTile<T, 2, TDE> Qtile;
  {
    const device T* qbase = Q + size_t(b) * p->q_strides[0] +
        size_t(h0) * p->q_strides[1] + size_t(l) * p->q_strides[2] + part * DE;
    const int q_ld = int(p->q_strides[1]);
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
  const float scale2 = p->scale_log2;

  // The row of key slot j, and whether it counts.
  auto slot_row = [&](int j, thread bool& ok) -> const device T* {
    ok = j < k1;
    const device T* row = safe_row;
    if (ok) {
      if (j < W) {
        ok = (wm == nullptr) || wm[j];
        row = win_b + size_t(j) * p->win_strides[1];
      } else {
        const int n = j - W;
        const int64_t r = int64_t(idx_bl[n]);
        ok = ((sm_ == nullptr) || sm_[n]) && r >= 0 && r < int64_t(p->P);
        row = ok ? pool_b + size_t(r) * p->pool_strides[1] : safe_row;
      }
    }
    return row;
  };

  // A tile in registers: this lane's two fragment rows (keys sm and sm + 8
  // of the tile) over the column fragments, plus the validity of its four
  // S columns (keys sn + jj). Every lane issues the same loads.
  struct Tile {
    kfrag_t kf[TDE];
    bool cv[4];
  };
  auto load_tile = [&](int kt, thread Tile& t) {
    const device T* rows[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
      bool ok;
      rows[i] = slot_row(kt + sm + i * 8, ok) + col0;
    }
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < 4; jj++) {
      slot_row(kt + sn + jj, t.cv[jj]);
    }
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDE; dd++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        const T4 v = *((const device T4*)(rows[i] + dd * kU));
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          t.kf[dd][i * 4 + j] = v[j];
        }
      }
    }
  };

  int parity = 0;
  auto step = [&](thread Tile& cur) {
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

    // Exchange: park the partial in this simdgroup's slot as [key][head],
    // sum the slots by sub-block into this tile's result buffer, then read
    // the total back transposed (heads as rows), scaled and masked.
    threadgroup float* res = results + parity * SLOT;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STt.template store<float, SLD, 1>(slots + part * SLOT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!is_tail) {
      // Sub-block: keys 2 * part + (lane >> 4), heads
      // 2 * (lane & 15) + {0, 1}.
      const int key = 2 * part + int(simd_lane_id >> 4);
      const int head = 2 * int(simd_lane_id & 15);
      float a0 = 0, a1 = 0;
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < NSS; q++) {
        a0 += slots[q * SLOT + key * SLD + head];
        a1 += slots[q * SLOT + key * SLD + head + 1];
      }
      res[key * SLD + head] = a0;
      res[key * SLD + head + 1] = a1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    parity ^= 1;
    if (is_tail) {
      return;
    }
    stile_t Stile;
    Stile.template load<float, 1, SLD>(res);
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

    // O += P @ V over this simdgroup's columns from the resident
    // fragments (K == V).
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
  if (k0 < k1) {
    load_tile(k0, tA);
  }
  for (int kt = k0; kt < k1; kt += 2 * BK) {
    if (kt + BK < k1) {
      load_tile(kt + BK, tB);
    }
    step(tA);
    if (kt + BK >= k1) {
      break;
    }
    if (kt + 2 * BK < k1) {
      load_tile(kt + 2 * BK, tA);
    }
    step(tB);
  }

  // Unnormalized partials [B, L, splits, Hp, D] and the row stats
  // [B, L, splits, Hp], the max in log2 units as the merge reads it.
  if (is_tail || rows_valid <= 0) {
    return;
  }
  const size_t bls = (size_t(b) * p->L + l) * p->n_splits + s;
  device float* dst = Oacc + (bls * p->Hp + h0) * D + part * DE;
  Otile.store_rows(dst, D, short(rows_valid));
  if (part == 0 && sn == 0) {
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 4; r++) {
      const int row = (r >> 1) * kU + (r & 1) * 8 + sm;
      if (row < rows_valid) {
        const size_t po = bls * p->Hp + h0 + row;
        Ls[po] = sum_score[r];
        Ms[po] =
            max_score[r] == Limits<float>::finite_min ? kEmpty : max_score[r];
      }
    }
  }
}
