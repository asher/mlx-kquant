// DeepSeek-V4-Flash lightning-indexer kernels, ported from omlx glm_moe_dsa
// steel_dsa_indexer_score.h (also (c) 2026 Apple Inc.) with the kernels
// renamed into the kq_ namespace and the top-k params struct self-hosted
// (kq_dsa_params.h). omlx is Apache-2.0: see mlx_kquant/licenses/
// omlx-LICENSE. Two kernels:
//
//  * kq_dsa_indexer_score -- steel GEMM computing, per query row m and key
//    column n, sum_h relu(q[h,m] . k[n]) * w[h,m]: the indexer relevance
//    score whose per-row top-k picks the pooled tokens the sparse-attention
//    kernel gathers. Function constants select causal masking (300) and the
//    [L, H] weights layout (301).
//
//  * kq_dsa_topk_indices_16bit -- one-threadgroup-per-row 2-pass radix
//    select over the 16-bit orderable transform of fp16/bf16 scores,
//    emitting TOPK uint32 indices per row. Ties at the threshold key are
//    admitted in scan order, so the selected set matches a full sort while
//    the order inside the row does not. Function constant 302 switches to a
//    bucketed (deterministic-order) emission.
//
// Bodies are byte-identical to omlx apart from the renames.

#pragma once

#include "mlx/backend/metal/kernels/kq_dsa_params.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace mlx::steel;

constant bool kq_dsa_do_causal [[function_constant(300)]];
constant bool kq_dsa_weights_lh [[function_constant(301)]];
constant bool kq_dsa_bucketed_topk [[function_constant(302)]];

template <typename T>
METAL_FUNC uint kq_dsa_ordered_key_16(T x) {
  const ushort bits = as_type<ushort>(x);
  return (bits & 0x8000) ? uint((~bits) & 0xffff) : uint(bits | 0x8000);
}

template <typename T, typename O, int TOPK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
kq_dsa_topk_indices_16bit(
    const device T* scores [[buffer(0)]],
    device O* out [[buffer(1)]],
    const constant KQDsaTopKParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  threadgroup atomic_uint hist[256];
  threadgroup atomic_uint counters[2];
  threadgroup uint state[4];

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  if (tid < 2) {
    atomic_store_explicit(&counters[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* row_scores = scores + size_t(row) * params->K;
  device O* row_out = out + size_t(row) * TOPK;

  int scan_limit = params->K;
  if (params->causal_valid_prefix) {
    const int q = int(row % uint(params->L));
    const int valid_length =
        metal::min(params->K, metal::max(0, params->K - params->L + q + 1));
    if (valid_length <= TOPK) {
      for (int i = int(tid); i < TOPK; i += THREADS) {
        row_out[i] = O(i < valid_length ? i : 0);
      }
      return;
    }
    scan_limit = valid_length;
  }

  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = kq_dsa_ordered_key_16(row_scores[i]);
    atomic_fetch_add_explicit(&hist[key >> 8], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = 0;
    uint threshold_hi = 0;
    for (int h = 255; h >= 0; --h) {
      const uint count = atomic_load_explicit(&hist[h], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_hi = uint(h);
        break;
      }
      greater += count;
    }
    state[0] = threshold_hi;
    state[1] = greater;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_hi = state[0];
  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = kq_dsa_ordered_key_16(row_scores[i]);
    if ((key >> 8) == threshold_hi) {
      atomic_fetch_add_explicit(&hist[key & 0xff], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = state[1];
    uint threshold_lo = 0;
    for (int l = 255; l >= 0; --l) {
      const uint count = atomic_load_explicit(&hist[l], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_lo = uint(l);
        break;
      }
      greater += count;
    }
    const uint threshold_key = (threshold_hi << 8) | threshold_lo;
    state[2] = threshold_key;
    state[3] = greater;
    atomic_store_explicit(&counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[1], greater, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_key = state[2];
  if (kq_dsa_bucketed_topk) {
    for (int base = 0; base < scan_limit; base += THREADS) {
      const int i = base + int(tid);
      if (i < scan_limit) {
        const uint key = kq_dsa_ordered_key_16(row_scores[i]);
        if (key > threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        } else if (key == threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (int i = int(tid); i < scan_limit; i += THREADS) {
      const uint key = kq_dsa_ordered_key_16(row_scores[i]);
      if (key > threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = O(i);
        }
      } else if (key == threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = O(i);
        }
      }
    }
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
kq_dsa_indexer_score(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* W [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int& H [[buffer(5)]],
    const constant int& unused_causal_prefix_topk [[buffer(6)]],
    const constant bool& skip_causal_future_store [[buffer(7)]],
    const constant int& causal_q_offset [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  using gemm_kernel =
      GEMMKernel<T, T, BM, BN, BK, WM, WN, false, true, true, true, float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;

  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;
  constexpr int THREADS = WM * WN * 32;
  const int thread_idx = int(simd_group_id) * 32 + int(simd_lane_id);

  if (kq_dsa_do_causal) {
    const int row_limit = metal::min(c_row + BM, M);
    if (c_col > q_offset + row_limit - 1) {
      if (skip_causal_future_store) {
        return;
      }
      device T* Dst =
          O + size_t(tid.z) * M * N + size_t(c_row) * params->ldd + c_col;
      for (int e = thread_idx; e < BM * BN; e += THREADS) {
        const int row = e / BN;
        const int col = e - row * BN;
        if (c_row + row < M && c_col + col < N) {
          Dst[size_t(row) * params->ldd + col] = static_cast<T>(-INFINITY);
        }
      }
      return;
    }
  }

  if (kq_dsa_do_causal && unused_causal_prefix_topk > 0) {
    const int row_limit = metal::min(c_row + BM, M);
    if (q_offset + row_limit <= unused_causal_prefix_topk) {
      return;
    }
  }

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  W += size_t(tid.z) * H * M;
  O += size_t(tid.z) * M * N + size_t(c_row) * params->ldd + c_col;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  for (int h = 0; h < H; ++h) {
    mma_op.Ctile.clear();

    const device T* A = Q + size_t(h) * M * D + size_t(c_row) * D;
    const device T* B = K + size_t(c_col) * D;

    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
      const float weight = kq_dsa_weights_lh
          ? static_cast<float>(W[size_t(row) * H + h])
          : static_cast<float>(W[size_t(h) * M + row]);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          accum[ai++] += max(frag[e], 0.0f) * weight;
        }
      }
    }
  }

  device T* Dst = O + size_t(mma_op.sm) * params->ldd + mma_op.sn;
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      const int out_base =
          (i * decltype(mma_op.Ctile)::kFragRows) * WM * params->ldd +
          (j * decltype(mma_op.Ctile)::kFragCols) * WN;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        const bool future = kq_dsa_do_causal && col > q_offset + row;
        const T value =
            future ? static_cast<T>(-INFINITY) : static_cast<T>(accum[ai]);
        Dst[out_base + e] = value;
        ai++;
      }
    }
  }
}

// Decode-width direct score: sum_h relu(q[h, j] . k[r]) * w[j, h] for
// QL <= 4 query rows against every pooled row, one simdgroup per R-row
// group, never materializing the [H, P] per-head scores. Each lane holds
// one half4 of each key row; q rows stream from device (L2-resident) and
// are shared across the group's R rows. Pooled visibility follows
// PoolingCache.make_mask: row r is visible to query j iff
// r < (q_offset + j + 1) / ratio, and every row is visible when QL == 1;
// invisible rows score the dtype's finite min so the radix top-k orders
// them last (matching the inline path's masked argpartition).
template <typename T, int QL, int H = 64, int D = 128, int SGS = 4, int R = 8>
[[kernel, max_total_threads_per_threadgroup(SGS * 32)]] void
kq_dsa_indexer_score_decode(
    const device T* Q [[buffer(0)]], // [B, H, QL, D]
    const device T* K [[buffer(1)]], // [B, P, D]
    const device T* W [[buffer(2)]], // [B, QL, H]
    device T* out [[buffer(3)]], // [B, 1, QL, P]
    const constant int& P [[buffer(4)]],
    const constant int& q_offset [[buffer(5)]],
    const constant int& ratio [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(D == 4 * 32, "one half4 of the key row per lane");

  const int b = int(tid.z);
  const int row0 = (int(tid.x) * SGS + int(simd_gid)) * R;
  if (row0 >= P) {
    return;
  }
  const int nrows = metal::min(R, P - row0);

  const device T* kb = K + (size_t(b) * P + size_t(row0)) * D + simd_lid * 4;
  vec<T, 4> kf[R];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < R; ++i) {
    kf[i] = i < nrows
        ? *reinterpret_cast<const device vec<T, 4>*>(kb + size_t(i) * D)
        : vec<T, 4>(0);
  }

  float acc[R][QL];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < R; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < QL; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  const device T* qb = Q + size_t(b) * H * QL * D + simd_lid * 4;
  const device T* wb = W + size_t(b) * QL * H;
  for (int h = 0; h < H; ++h) {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < QL; ++j) {
      const float4 qf = float4(*reinterpret_cast<const device vec<T, 4>*>(
          qb + (size_t(h) * QL + j) * D));
      const float w = float(wb[size_t(j) * H + h]);
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < R; ++i) {
        const float dotv = simd_sum(metal::dot(qf, float4(kf[i])));
        acc[i][j] += metal::max(dotv, 0.0f) * w;
      }
    }
  }

  if (simd_lid != 0) {
    return;
  }
  device T* ob = out + size_t(b) * QL * size_t(P) + row0;
  STEEL_PRAGMA_UNROLL
  for (int j = 0; j < QL; ++j) {
    const int vlim = QL == 1 ? P : metal::min(P, (q_offset + j + 1) / ratio);
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < R; ++i) {
      if (i < nrows) {
        ob[size_t(j) * P + i] =
            row0 + i < vlim ? static_cast<T>(acc[i][j]) : Limits<T>::finite_min;
      }
    }
  }
}
