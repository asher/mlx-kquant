// Sliding-window causal prefill attention on the tensor-op (NAX) units, a
// port of MLX's attention_nax (MIT, steel_attention_nax.h; see
// mlx_kquant/licenses/) that visits only the key blocks inside each query
// block's window. Stock attention takes the window as a dense mask over every
// key block up to the diagonal, so a 2048-row chunk over a 513-key window
// runs about five times the useful work.
//
// The queries are the last qL of the kL key positions. Query row r, in key
// positions r = row + kL - qL, sees keys c with r - window < c <= r, plus an
// optional per-head sink logit in the softmax denominator. With starts, batch
// row b also skips the keys below starts[b] (a left-padded batch), and a query
// row below its start writes zeros. One threadgroup of
// four simdgroups per 64 query rows, query head and batch row; each simdgroup
// owns 16 rows. Q, K, V and O are read and written through their batch, head
// and row strides with a unit head_dim stride, as stock reads them.
#pragma once

#include "mlx/backend/metal/kernels/kq_sdpa_nax.h"

template <typename T, int D>
[[kernel, max_total_threads_per_threadgroup(128)]] void kq_sdpa_window_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const device float* sinks [[buffer(4)]],
    // Q, K, V, O: batch, head, row
    const constant int64_t* strides [[buffer(5)]],
    const constant int& qL [[buffer(6)]],
    const constant int& kL [[buffer(7)]],
    const constant int& gqa_factor [[buffer(8)]],
    const constant int& window [[buffer(9)]],
    const constant int& has_sinks [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const device int* starts [[buffer(12)]],
    const constant int& has_starts [[buffer(13)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using namespace kq_sdpa_nax;
  constexpr short kU = 16;
  constexpr int BQ = 64;
  constexpr int BK = 32;
  constexpr int TD = D / kU;
  constexpr short TK = BK / kU;
  static_assert(TD % 2 == 0, "P @ V accumulates column fragments in pairs");

  const short tm = kU * short(simd_group_id);
  const int row0 = int(tid.x) * BQ + tm;
  const int64_t kvh = int64_t(tid.y / uint(gqa_factor));
  Q += tid.z * strides[0] + tid.y * strides[1] + row0 * strides[2];
  K += tid.z * strides[3] + kvh * strides[4];
  V += tid.z * strides[6] + kvh * strides[7];
  O += tid.z * strides[9] + tid.y * strides[10] + row0 * strides[11];
  const int ldq = int(strides[2]);
  const int ldk = int(strides[5]);
  const int ldv = int(strides[8]);
  const int ldo = int(strides[11]);
  const float scale2 = scale * 1.44269504089f;

  using otile_t = NAXTile<float, 1, TD>;
  otile_t Otile;
  Otile.clear();
  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<float, kRowsPT> max_score;
  metal::vec<float, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<float>::finite_min;
  }
  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * sinks[tid.y];
      sum_score[i] = 1;
    }
  }

  // The threadgroup's rows in key positions are [r_lo, r_hi]. Key blocks run
  // from the first one the top row sees to the diagonal of the bottom row.
  // Blocks from kb_causal on hold keys past some row's diagonal; blocks up to
  // kb_window hold keys below some row's window, and up to kb_start keys
  // below the row's start.
  const int qL_off = kL - qL;
  const int s0 = has_starts ? clamp(starts[tid.z], 0, kL) : 0;
  const int NK_aligned = kL / BK;
  const int kL_rem = kL - NK_aligned * BK;
  const int r_lo = int(tid.x) * BQ + qL_off;
  const int r_hi = min(int(tid.x + 1) * BQ, qL) - 1 + qL_off;
  const int kb_lo = max(s0, r_lo - window + 1) / BK;
  const int kb_lim = r_hi / BK + 1;
  const int kb_causal = r_lo / BK;
  const int kb_window = r_hi - window + 1 > 0 ? (r_hi - window + 1) / BK : -1;
  const int kb_start = s0 > 0 ? s0 / BK : -1;
  const short lim_rows_q = short(min(qL - row0, int(kU)));
  const bool short_q = lim_rows_q < kU;
  constexpr auto neg_inf = Limits<float>::finite_min;

  K += int64_t(kb_lo) * BK * ldk;
  V += int64_t(kb_lo) * BK * ldv;
  for (int kb = kb_lo; kb < kb_lim; kb++) {
    const bool is_last_k = kb == NK_aligned;

    // S = Q @ K^T
    using stile_t = NAXTile<float, 1, TK>;
    stile_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
      for (short id = 0; id < TD; id++) {
        NAXTile<T, 1, 1> Qtile;
        NAXTile<T, 2, 1> Ktile;
        if (short_q) {
          Qtile.load_rows(Q + id * kU, ldq, max(lim_rows_q, short(0)));
        } else {
          Qtile.load(Q + id * kU, ldq);
        }
        const int k_off = ik * kU * ldk + id * kU;
        if (is_last_k) {
          Ktile.load_rows(K + k_off, ldk, kL_rem - ik * kU);
        } else {
          Ktile.load(K + k_off, ldk);
        }
        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtile.frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    if (is_last_k) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (col_pos + jj) < kL_rem ? fg[loc] : neg_inf;
          }
        }
      }
    }

    if (kb >= kb_causal || kb <= kb_window || kb <= kb_start) {
      const int base_row = row0 + qL_off;
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const int r = base_row + ii * stile_t::kFragRowsJump + sm;
            const int c = base_col + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (r < c || c <= r - window || c < s0) ? neg_inf : fg[loc];
          }
        }
      }
    }

    // Online softmax
    metal::vec<float, kRowsPT> new_max;
    metal::vec<float, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);
    simdgroup_barrier(mem_flags::mem_none);

    // O += P @ V
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id += 2) {
      if constexpr (D == 128) {
        if (id == 4) {
          threadgroup_barrier(mem_flags::mem_none);
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        const int v_off = ik * kU * ldv + id * kU;
        if (is_last_k) {
          Vtile.load_rows(V + v_off, ldv, kL_rem - ik * kU);
        } else {
          Vtile.load(V + v_off, ldv);
        }
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
    K += BK * ldk;
    V += BK * ldv;
  }

  threadgroup_barrier(mem_flags::mem_none);
  if (lim_rows_q <= 0) {
    return;
  }
  // A row below its start saw no key: zero, not the masked average.
  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    const int r = row0 + qL_off + i * otile_t::kFragRowsJump + sm;
    rcp[i] = r >= s0 ? 1.f / sum_score[i] : 0.f;
  }
  Otile.template row_bin_op<MulOp>(rcp);
  if (short_q) {
    Otile.store_rows(O, ldo, lim_rows_q);
  } else {
    Otile.store(O, ldo);
  }
}

// The same attention at head dim 256 with one key head a query head and no
// starts, in the column-block form of kq_sdpa_sparse_decode_nax.h. One
// threadgroup per 32 query rows, query head and batch row. Four simdgroups
// each own a 64-column quarter of the head dim: the 32 query rows over
// those columns stay in registers and 16 key rows load into fragment
// registers, then 16 value rows into the same ones. A simdgroup computes
// the partial S^T = K @ Q^T over its columns and parks it in threadgroup
// scratch; the partials are summed and read back transposed (queries as
// rows) for the mask and the row softmax, then P @ V runs over its columns.
// A thread holds 64 floats of output and no second key tile, where the
// kernel above holds 128 and reloads Q for every key block.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void kq_sdpa_window_cb_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const device float* sinks [[buffer(4)]],
    // Q, K, V, O: batch, head, row
    const constant int64_t* strides [[buffer(5)]],
    const constant int& qL [[buffer(6)]],
    const constant int& kL [[buffer(7)]],
    const constant int& window [[buffer(9)]],
    const constant int& has_sinks [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using namespace kq_sdpa_nax;
  constexpr int NSG = 4; // simdgroups
  constexpr int DE = 64; // columns per simdgroup
  constexpr short kU = 16;
  constexpr int BK = 16; // keys per tile
  constexpr short TDE = DE / kU;
  constexpr int BQ = 32; // query rows per threadgroup
  constexpr int SLD = 34; // exchange slot row stride (floats)
  constexpr int SLOT = BK * SLD;

  using otile_t = NAXTile<float, 2, TDE>;
  using stile_t = NAXTile<float, 2, 1>; // queries x keys
  using sttile_t = NAXTile<float, 1, 2>; // keys x queries

  // A partial slot per simdgroup plus the result buffer, each [16 keys][34]
  // floats.
  threadgroup float slots[(NSG + 1) * SLOT];
  threadgroup float* res = slots + NSG * SLOT;

  const int part = int(simd_group_id);
  const int row0 = int(tid.x) * BQ;
  const int rows_valid = metal::min(qL - row0, BQ);
  // The threadgroup's rows in key positions are [r_lo, r_hi]; its keys run
  // from the first one the top row sees to the diagonal of the bottom row.
  const int r_lo = row0 + kL - qL;
  const int r_hi = r_lo + rows_valid - 1;
  const int k_lo = metal::max(0, r_lo - window + 1);
  const int k_end = r_hi + 1;

  Q += tid.z * strides[0] + tid.y * strides[1] + row0 * strides[2] + part * DE;
  K += tid.z * strides[3] + tid.y * strides[4] + part * DE;
  V += tid.z * strides[6] + tid.y * strides[7] + part * DE;
  O +=
      tid.z * strides[9] + tid.y * strides[10] + row0 * strides[11] + part * DE;
  const int ldq = int(strides[2]);
  const int ldk = int(strides[5]);
  const int ldv = int(strides[8]);
  const int ldo = int(strides[11]);
  const float scale2 = scale * 1.44269504089f;

  const short2 sc = BaseNAXFrag::get_coord();
  const short sm = sc.y;
  const short sn = sc.x;

  // The query rows over this simdgroup's columns, resident for the walk
  // (rows past qL zero).
  NAXTile<T, 2, TDE> Qtile;
  if (rows_valid == BQ) {
    Qtile.load(Q, ldq);
  } else {
    Qtile.load_rows(Q, ldq, short(rows_valid));
  }

  otile_t Otile;
  Otile.clear();
  metal::vec<float, 4> max_score;
  metal::vec<float, 4> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 4; i++) {
    max_score[i] = Limits<float>::finite_min;
  }

  for (int kt = k_lo; kt < k_end; kt += BK) {
    const short kn = short(metal::min(k_end - kt, BK));
    // One tile of registers serves the keys, then the values.
    NAXTile<T, 1, TDE> KVtile;
    if (kn == BK) {
      KVtile.load(K + int64_t(kt) * ldk, ldk);
    } else {
      KVtile.load_rows(K + int64_t(kt) * ldk, ldk, kn);
    }

    // Partial S^T = K @ Q^T over this simdgroup's columns: 16 keys x 32
    // queries.
    sttile_t STt;
    STt.clear();
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDE; dd++) {
      sttile_t::NAXFrag_t::mma(
          STt.frag_at(0, 0),
          STt.frag_at(0, 1),
          KVtile.frag_at(0, dd),
          metal::false_type{},
          Qtile.frag_at(0, dd),
          Qtile.frag_at(1, dd),
          metal::true_type{});
    }

    // Exchange: park the partial as [key][query], sum the slots by
    // sub-block into the result buffer, then read the total back
    // transposed, scaled and masked.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STt.template store<float, SLD, 1>(slots + part * SLOT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      // Sub-block: key 4 * part + (lane >> 3), queries 4 * (lane & 7) on.
      const int key = 4 * part + int(simd_lane_id >> 3);
      const int qr = 4 * int(simd_lane_id & 7);
      metal::vec<float, 4> a{0};
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < NSG; q++) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 4; e++) {
          a[e] += slots[q * SLOT + key * SLD + qr + e];
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 4; e++) {
        res[key * SLD + qr + e] = a[e];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stile_t Stile;
    Stile.template load<float, 1, SLD>(res);
    STEEL_PRAGMA_UNROLL
    for (short ir = 0; ir < 2; ir++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; jj++) {
          const int r = r_lo + ir * kU + i * 8 + sm;
          const int c = kt + sn + jj;
          const short e = i * 4 + jj;
          Stile.frag_at(ir, 0)[e] = (c <= r && c > r - window)
              ? Stile.frag_at(ir, 0)[e] * scale2
              : Limits<float>::finite_min;
        }
      }
    }

    // Online softmax (exp2 space). A row with no visible key yet keeps its
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

    // O += P @ V over this simdgroup's columns.
    if (kn == BK) {
      KVtile.load(V + int64_t(kt) * ldv, ldv);
    } else {
      KVtile.load_rows(V + int64_t(kt) * ldv, ldv, kn);
    }
    STEEL_PRAGMA_UNROLL
    for (short ir = 0; ir < 2; ir++) {
      STEEL_PRAGMA_UNROLL
      for (short pp = 0; pp < TDE; pp += 2) {
        otile_t::NAXFrag_t::mma(
            Otile.frag_at(ir, pp),
            Otile.frag_at(ir, pp + 1),
            Stile.frag_at(ir, 0),
            metal::false_type{},
            KVtile.frag_at(0, pp),
            KVtile.frag_at(0, pp + 1),
            metal::false_type{});
      }
    }
  }

  // Normalize, the sink as one more logit, and write the rows.
  metal::vec<float, 4> wgt;
  const float sink_m = has_sinks ? M_LOG2E_F * sinks[tid.y] : 0.0f;
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < 4; r++) {
    float m_all = max_score[r];
    float l_all = sum_score[r];
    if (has_sinks) {
      m_all = metal::max(max_score[r], sink_m);
      l_all = sum_score[r] * fast::exp2(max_score[r] - m_all) +
          fast::exp2(sink_m - m_all);
    }
    wgt[r] = fast::exp2(max_score[r] - m_all) / l_all;
  }
  Otile.template row_bin_op<MulOp>(wgt);
  if (rows_valid == BQ) {
    Otile.store(O, ldo);
  } else {
    Otile.store_rows(O, ldo, short(rows_valid));
  }
}
