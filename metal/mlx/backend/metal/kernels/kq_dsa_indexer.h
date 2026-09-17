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
//    admitted lowest-index-first, matching argpartition's index-order
//    tie-break; the order inside the row is not sorted. Function constant
//    302 is accepted for ABI parity and has no effect.
//
// Score kernel body byte-identical to omlx apart from the renames. The
// top-k emission is intentionally not: omlx compacts through atomic
// counters, which is nondeterministic under 16-bit tie pileup; do not
// re-take it on an omlx sync.

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
    uint row [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
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

  // Stable block-ordered compaction; every step is index-ordered so the
  // emitted array is bitwise reproducible. Greater keys fill [0, greater),
  // threshold ties fill [greater, TOPK) lowest-index-first.
  const uint threshold_key = state[2];
  threadgroup uint sg_gt[THREADS / 32];
  threadgroup uint sg_tie[THREADS / 32];
  threadgroup uint bases[2];
  if (tid == 0) {
    bases[0] = 0; // next slot for strictly-greater keys
    bases[1] = state[3]; // count of strictly-greater keys
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int base = 0; base < scan_limit; base += THREADS) {
    const int i = base + int(tid);
    bool is_gt = false;
    bool is_tie = false;
    if (i < scan_limit) {
      const uint key = kq_dsa_ordered_key_16(row_scores[i]);
      is_gt = (key > threshold_key);
      is_tie = (key == threshold_key);
    }
    const uint gt_lane = simd_prefix_exclusive_sum(uint(is_gt));
    const uint tie_lane = simd_prefix_exclusive_sum(uint(is_tie));
    if (simd_lane == 31) {
      sg_gt[simd_group] = gt_lane + uint(is_gt);
      sg_tie[simd_group] = tie_lane + uint(is_tie);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (is_gt || is_tie) {
      uint pos = is_gt ? (bases[0] + gt_lane) : (bases[1] + tie_lane);
      for (uint g = 0; g < simd_group; ++g) {
        pos += is_gt ? sg_gt[g] : sg_tie[g];
      }
      if (pos < uint(TOPK)) {
        row_out[pos] = O(i);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint gt_tot = 0;
      uint tie_tot = 0;
      for (uint g = 0; g < THREADS / 32; ++g) {
        gt_tot += sg_gt[g];
        tie_tot += sg_tie[g];
      }
      bases[0] += gt_tot;
      bases[1] += tie_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  (void)kq_dsa_bucketed_topk; // unused: emission is identical in both modes
  (void)counters;
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

// Decode-width score: sum_h relu(q[h, j] . k[r]) * w[j, h] for QL <= 4
// query rows against every pooled row, never materializing the [H, P]
// per-head scores. Each threadgroup stages the query block for one row j
// in threadgroup memory and streams its key range in NB x 8 row blocks:
// S[heads, keys] = Q K^T over 8x8 tiles, relu, then the weighted head sum
// is one more tile product with the head weights in row 0 of an otherwise
// zero tile, so row 0 of the accumulator holds the scores. The keys are
// the only DRAM traffic; H below 8 is padded with zero rows. Pooled
// visibility follows PoolingCache.make_mask: row r is visible to query j
// iff r < (q_offset + j + 1) / ratio, and every row is visible when
// QL == 1; invisible rows score the dtype's finite min so the radix top-k
// orders them last. WT is the head-weight storage type: fp32 keeps
// sign-free head gates exact for the runtimes that pin them (glm5).
template <
    typename T,
    int QL,
    int H = 64,
    int D = 128,
    int SGS = 8,
    int NB = 1,
    typename WT = T>
[[kernel, max_total_threads_per_threadgroup(SGS * 32)]] void
kq_dsa_indexer_score_decode(
    const device T* Q [[buffer(0)]], // [B, H, QL, D]
    const device T* K [[buffer(1)]], // [B, P, D]
    const device WT* W [[buffer(2)]], // [B, QL, H]
    device T* out [[buffer(3)]], // [B, 1, QL, P]
    const constant int& P [[buffer(4)]],
    const constant int& q_offset [[buffer(5)]],
    const constant int& ratio [[buffer(6)]],
    const constant int& keys_per_tg [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort simd_lid [[thread_index_in_simdgroup]]) {
  using MatT = metal::simdgroup_matrix<T, 8, 8>;
  using MatF = metal::simdgroup_matrix<float, 8, 8>;
  constexpr int NHT = (H + 7) / 8;
  constexpr int HP = NHT * 8;
  constexpr int NDT = D / 8;
  constexpr int CH = D / 8; // 8-wide chunks per row
  constexpr int LD = D + 8; // padded row (bank spread, 16-byte aligned)
  constexpr int KB = NB * 8;
  constexpr int NTH = SGS * 32;

  threadgroup T Qs[HP * LD];
  threadgroup float Wsm[NHT * 64];
  threadgroup T Ktail[KB * LD];
  threadgroup float Os[SGS * 64];

  const int b = int(tid.z);
  const int tg0 = int(tid.x) * keys_per_tg;
  if (tg0 >= P) {
    return;
  }
  const int tg1 = metal::min(P, tg0 + keys_per_tg);
  const int nblk = (tg1 - tg0 + KB - 1) / KB;
  const int tidx = int(simd_gid) * 32 + int(simd_lid);

  if (HP > H) {
    for (int i = tidx; i < (HP - H) * LD; i += NTH) {
      Qs[H * LD + i] = T(0);
    }
  }

  const device T* qb = Q + size_t(b) * H * QL * D;
  const device WT* wb = W + size_t(b) * QL * H;
  const device T* kb = K + size_t(b) * size_t(P) * D;
  device T* ob = out + size_t(b) * QL * size_t(P);

  for (int j = 0; j < QL; ++j) {
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (int c = tidx; c < H * CH; c += NTH) {
      const int h = c / CH;
      const int d = (c % CH) * 8;
      *reinterpret_cast<threadgroup vec<T, 8>*>(Qs + h * LD + d) =
          *reinterpret_cast<const device vec<T, 8>*>(
              qb + (size_t(h) * QL + j) * D + d);
    }
    for (int i = tidx; i < NHT * 64; i += NTH) {
      const int r = (i % 64) / 8;
      const int h = (i / 64) * 8 + (i % 8);
      Wsm[i] = (r == 0 && h < H) ? float(wb[size_t(j) * H + h]) : 0.0f;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    MatF Wf[NHT];
    STEEL_PRAGMA_UNROLL
    for (int ht = 0; ht < NHT; ++ht) {
      simdgroup_load(Wf[ht], Wsm + ht * 64, 8);
    }
    const int vlim = QL == 1 ? P : metal::min(P, (q_offset + j + 1) / ratio);

    for (int blk = int(simd_gid); blk < nblk; blk += SGS) {
      const int key0 = tg0 + blk * KB;
      MatT Kt[NB][NDT];
      if (key0 + KB <= P) {
        STEEL_PRAGMA_UNROLL
        for (int nb = 0; nb < NB; ++nb) {
          const device T* kr = kb + size_t(key0 + nb * 8) * D;
          STEEL_PRAGMA_UNROLL
          for (int dt = 0; dt < NDT; ++dt) {
            simdgroup_load(Kt[nb][dt], kr + dt * 8, D, ulong2(0, 0), true);
          }
        }
      } else {
        // Only the last block of the pool lands here: stage it with zero
        // rows past P so the tile loads stay in bounds.
        for (int c = int(simd_lid); c < KB * CH; c += 32) {
          const int r = c / CH;
          const int d = (c % CH) * 8;
          *reinterpret_cast<threadgroup vec<T, 8>*>(Ktail + r * LD + d) =
              key0 + r < P ? *reinterpret_cast<const device vec<T, 8>*>(
                                 kb + size_t(key0 + r) * D + d)
                           : vec<T, 8>(0);
        }
        simdgroup_barrier(metal::mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (int nb = 0; nb < NB; ++nb) {
          STEEL_PRAGMA_UNROLL
          for (int dt = 0; dt < NDT; ++dt) {
            simdgroup_load(
                Kt[nb][dt],
                Ktail + nb * 8 * LD + dt * 8,
                LD,
                ulong2(0, 0),
                true);
          }
        }
      }

      MatF acc[NB];
      STEEL_PRAGMA_UNROLL
      for (int nb = 0; nb < NB; ++nb) {
        acc[nb] = MatF(0.0f);
      }
      STEEL_PRAGMA_UNROLL
      for (int ht = 0; ht < NHT; ++ht) {
        MatF St[NB];
        STEEL_PRAGMA_UNROLL
        for (int nb = 0; nb < NB; ++nb) {
          St[nb] = MatF(0.0f);
        }
        STEEL_PRAGMA_UNROLL
        for (int dt = 0; dt < NDT; ++dt) {
          MatT Qt;
          simdgroup_load(Qt, Qs + ht * 8 * LD + dt * 8, LD);
          STEEL_PRAGMA_UNROLL
          for (int nb = 0; nb < NB; ++nb) {
            simdgroup_multiply_accumulate(St[nb], Qt, Kt[nb][dt], St[nb]);
          }
        }
        STEEL_PRAGMA_UNROLL
        for (int nb = 0; nb < NB; ++nb) {
          St[nb].thread_elements()[0] =
              metal::max(St[nb].thread_elements()[0], 0.0f);
          St[nb].thread_elements()[1] =
              metal::max(St[nb].thread_elements()[1], 0.0f);
          simdgroup_multiply_accumulate(acc[nb], Wf[ht], St[nb], acc[nb]);
        }
      }

      threadgroup float* os = Os + int(simd_gid) * 64;
      STEEL_PRAGMA_UNROLL
      for (int nb = 0; nb < NB; ++nb) {
        simdgroup_store(acc[nb], os, 8);
        simdgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (simd_lid < 8) {
          const int r = key0 + nb * 8 + int(simd_lid);
          if (r < P) {
            ob[size_t(j) * P + r] =
                r < vlim ? static_cast<T>(os[simd_lid]) : Limits<T>::finite_min;
          }
        }
        simdgroup_barrier(metal::mem_flags::mem_threadgroup);
      }
    }
  }
}
