// Sliding-window causal prefill attention on the tensor-op (NAX) units, a
// port of MLX's attention_nax (MIT, steel_attention_nax.h; see
// mlx_kquant/licenses/) that visits only the key blocks inside each query
// block's window. Stock attention takes the window as a dense mask over every
// key block up to the diagonal, so a 2048-row chunk over a 513-key window
// runs about five times the useful work.
//
// The queries are the last qL of the kL key positions. Query row r, in key
// positions r = row + kL - qL, sees keys c with r - window < c <= r, plus an
// optional per-head sink logit in the softmax denominator. One threadgroup of
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
  // kb_window hold keys below some row's window.
  const int qL_off = kL - qL;
  const int NK_aligned = kL / BK;
  const int kL_rem = kL - NK_aligned * BK;
  const int r_lo = int(tid.x) * BQ + qL_off;
  const int r_hi = min(int(tid.x + 1) * BQ, qL) - 1 + qL_off;
  const int kb_lo = max(0, r_lo - window + 1) / BK;
  const int kb_lim = r_hi / BK + 1;
  const int kb_causal = r_lo / BK;
  const int kb_window = r_hi - window + 1 > 0 ? (r_hi - window + 1) / BK : -1;
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

    if (kb >= kb_causal || kb <= kb_window) {
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
            fg[loc] = (r < c || c <= r - window) ? neg_inf : fg[loc];
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
  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }
  Otile.template row_bin_op<MulOp>(rcp);
  if (short_q) {
    Otile.store_rows(O, ldo, lim_rows_q);
  } else {
    Otile.store(O, ldo);
  }
}
