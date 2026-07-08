// DeepSeek-V4-Flash sparse attention: one dispatch covering the sliding
// local window plus the indexer-selected pooled rows, with per-head
// attention sinks seeding the flash online softmax (f32 accumulation).
//
// Ported from omlx glm_moe_dsa steel_deepseek_v4_sparse_attention.h (also
// (c) 2026 OpenAI) with the kernel renamed into the kq_ namespace and the
// params struct self-hosted (kq_dsa_params.h). The kernel is parametric in
// qL: one threadgroup per (query position, batch), WM simdgroups handling
// all H heads via 8x8 MMA fragments, D chunked into DC-wide slabs. K == V
// (the V4 shared latent), so a single KV buffer serves both passes.
//
// omlx is Apache-2.0: see mlx_kquant/licenses/omlx-LICENSE.
//
// Unlike the omlx host wrapper, the kq dispatch accepts qL >= 1: decode
// (qL = 1) and MTP verify (qL = 2) run this same kernel -- omlx gated them
// out only because it lacked a decode-time indexer, not for kernel reasons.
//
// Beyond the port, key tiles track their last live slot: all-dead tiles
// (visibility-clamped pooled tiles, the dummy pooled tile of window-only
// layers) skip both MMA phases entirely, and partially-live tiles bound the
// gathers and fragment sweeps to a bucketed live prefix (arms of TK/4
// fragments). Window and compressed layers pass identity top-k, so their
// live slots always form a prefix; scattered real top-k simply lands in the
// widest arm.

#pragma once

#include "mlx/backend/metal/kernels/kq_dsa_params.h"
#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace mlx::steel;

struct KQDsaMaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct KQDsaSumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct KQDsaMulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct KQDsaExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct KQDsaDivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// Live-bounded QK / PV fragment sweeps. Identity top-k (window and
// compressed layers) and the visibility clamp leave whole key-tile suffixes
// dead; the arms bound the fragment loops to the bucketed live prefix. KF
// must be a compile-time constant: a runtime bound would index the MMA
// fragment registers dynamically and spill them to thread memory.
// Dead columns stay exact: Stile is cleared, the selected[]<0 mask fill
// drives them to -inf before the row reduce, and the PV arm only skips
// fragments whose softmax weights are exactly zero.
#define KQ_DSA_QK_SWEEP(KF)                                                \
  STEEL_PRAGMA_UNROLL                                                      \
  for (short dd = 0; dd < TDC; ++dd) {                                     \
    simdgroup_barrier(mem_flags::mem_none);                                \
    Qtile.template load<T, 1, 1, LDQ, 1>(&Qs[Qs_offset + dd * kFragSize]); \
    STEEL_PRAGMA_UNROLL                                                    \
    for (short ik = 0; ik < (KF); ++ik) {                                  \
      MMAFragAcc::load(                                                    \
          Ktile.frag_at(0, ik),                                            \
          &KVs[Ks_offset + dd * kFragSize * LDK + ik * kFragSize],         \
          Int<LDK>{},                                                      \
          Int<1>{});                                                       \
    }                                                                      \
    simdgroup_barrier(mem_flags::mem_none);                                \
    STEEL_PRAGMA_UNROLL                                                    \
    for (short iq = 0; iq < TQ; ++iq) {                                    \
      STEEL_PRAGMA_UNROLL                                                  \
      for (short ik = 0; ik < (KF); ++ik) {                                \
        MMAFragAcc::mma(                                                   \
            Stile.frag_at(iq, ik),                                         \
            Qtile.frag_at(iq, 0),                                          \
            Ktile.frag_at(0, ik),                                          \
            Stile.frag_at(iq, ik));                                        \
      }                                                                    \
    }                                                                      \
  }

#define KQ_DSA_PV_SWEEP(KF)                                                    \
  STEEL_PRAGMA_UNROLL                                                          \
  for (short iq = 0; iq < TQ; ++iq) {                                          \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short id = 0; id < TDC; ++id) {                                       \
      STEEL_PRAGMA_UNROLL                                                      \
      for (short ik = 0; ik < (KF); ++ik) {                                    \
        const short kk = ik * kFragSize;                                       \
        const short dd = id * kFragSize;                                       \
        Vtile.template load<T, 1, 1, LDV, 1>(&KVs[Vs_offset + kk * LDV + dd]); \
        MMAFragAcc::mma(                                                       \
            Otile.frag_at(iq, vchunk* TDC + id),                               \
            Stile.frag_at(iq, ik),                                             \
            Vtile.frag_at(0, 0),                                               \
            Otile.frag_at(iq, vchunk* TDC + id));                              \
      }                                                                        \
    }                                                                          \
  }

// clang-format off
template <
    typename T,
    int BK,
    int DC,
    int H,
    int D,
    int WM,
    typename IndexT,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void kq_dsa_sparse_attention(
    const device T* Q [[buffer(0)]],
    const device T* LocalKV [[buffer(1)]],
    const device T* PooledKV [[buffer(2)]],
    const device IndexT* Topk [[buffer(3)]],
    const device T* Sinks [[buffer(4)]],
    device T* O [[buffer(5)]],
    const constant KQDsaSparseAttentionParams* params [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on

  constexpr short kFragSize = 8;
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ = DC + padQ;
  constexpr short LDK = BK + padK;
  constexpr short LDV = DC + padV;

  constexpr int kNWarps = WM;
  constexpr int TQ = H / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TDC = DC / kFragSize;
  constexpr int D_CHUNKS = D / DC;

  static_assert(TQ >= 1, "DeepSeek sparse attention needs a head tile.");
  static_assert(
      H % (kNWarps * kFragSize) == 0,
      "Head count must divide evenly across simdgroups.");
  static_assert(BK % kFragSize == 0, "BK must be a multiple of 8.");
  static_assert(DC % kFragSize == 0, "DC must be a multiple of 8.");
  static_assert(D % DC == 0, "Head dimension must divide DC.");
  static_assert(TK % 4 == 0, "Live-fragment arms slice the key tile in 4.");
  static_assert(
      sizeof(T) == 2,
      "Staging moves rows as uint4 (8 half/bfloat elements per load).");

  // uint4 staging geometry: 8 elements per 16-byte vector, DC/8 vectors per
  // row slice. Row strides (LDQ/LDV elements = 80 bytes) and dbase (64-byte
  // multiples) keep every vector access 16-byte aligned; the host contract
  // (mx::contiguous, D = 512) keeps the device side aligned too.
  constexpr short kElemsPerVec = 16 / short(sizeof(T));
  constexpr short kVecsPerSlice = DC / kElemsPerVec;

  constexpr int tgp_size = WM * 32;
  const int lane = int(simd_group_id * 32 + simd_lane_id);

  const int q_pos = int(tid.x);
  const int b = int(tid.y);

  threadgroup uint4 Qs_v[(H * LDQ * short(sizeof(T)) + 15) / 16];
  threadgroup uint4 KVs_v
      [(((BK * LDV > DC * LDK) ? BK * LDV : DC * LDK) * short(sizeof(T)) + 15) /
       16];
  threadgroup T* Qs = reinterpret_cast<threadgroup T*>(Qs_v);
  threadgroup T* KVs = reinterpret_cast<threadgroup T*>(KVs_v);
  threadgroup int selected[BK];
  // Double-buffered per-simdgroup last-live slot: tile t's readers and tile
  // t+2's writers of the same buffer are separated by tile t+1's post-build
  // barrier (executed on the skip path too), so no extra barrier is needed.
  threadgroup int sg_last_live[2][WM];

  using MMAFragAcc = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  MMATile<AccumType, TQ, 1, MMAFragAcc> Qtile;
  MMATile<AccumType, 1, TK, MMAFragAcc> Ktile;
  MMATile<AccumType, TQ, TK, MMAFragAcc> Stile;
  MMATile<AccumType, 1, 1, MMAFragAcc> Vtile;
  MMATile<AccumType, TQ, D_CHUNKS * TDC, MMAFragAcc> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFragAcc::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ + sn;
  const short Ks_offset = sm * LDK + sn;
  const short Vs_offset = sm * LDV + sn;

  const AccumType scale = AccumType(params->scale * M_LOG2E_F);

  constexpr short rows_per_thread = decltype(Stile)::kRowsPerThread;
  AccumType max_score[rows_per_thread];
  AccumType sum_score[rows_per_thread] = {0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < rows_per_thread; ++i) {
    const int head = int(tm + sm + i * kFragSize);
    if (head < params->H) {
      max_score[i] = AccumType(M_LOG2E_F) * AccumType(Sinks[head]);
      sum_score[i] = AccumType(1);
    } else {
      max_score[i] = Limits<AccumType>::finite_min;
    }
  }

  const device T* q_base = Q + size_t(b) * params->Q_strides[0] +
      size_t(q_pos) * params->Q_strides[2];
  const device T* local_base = LocalKV + size_t(b) * params->Local_strides[0];
  const device T* pooled_base =
      PooledKV + size_t(b) * params->Pooled_strides[0];
  const device IndexT* topk_base = Topk + size_t(b) * params->Topk_strides[0] +
      size_t(q_pos) * params->Topk_strides[2];

  const int local_offset = params->localL - params->qL;
  const int local_end = metal::min(params->localL, local_offset + q_pos + 1);
  const int local_start = metal::max(0, local_end - params->local_window);
  const int local_count = metal::max(0, local_end - local_start);
  const int local_tiles = (local_count + BK - 1) / BK;
  const int pooled_tiles = (params->topk + BK - 1) / BK;
  const int pooled_valid = metal::min(
      params->pooledL, (params->q_offset + q_pos + 1) / params->compress_ratio);
  const int total_tiles = local_tiles + pooled_tiles;

  for (int ktile = 0; ktile < total_tiles; ++ktile) {
    const bool is_pooled_tile = ktile >= local_tiles;
    const int tile_slot = is_pooled_tile ? (ktile - local_tiles) : ktile;
    const int slot_base = tile_slot * BK;

    int my_last_live = -1;
    for (int k = lane; k < BK; k += tgp_size) {
      const int slot = slot_base + k;
      int k_pos = -1;
      if (is_pooled_tile) {
        if (slot < params->topk) {
          const int pooled_pos = int(topk_base[slot]);
          if (pooled_pos >= 0 && pooled_pos < pooled_valid) {
            k_pos = pooled_pos;
          }
        }
      } else {
        if (slot < local_count) {
          k_pos = local_start + slot;
        }
      }
      selected[k] = k_pos;
      if (k_pos >= 0) {
        my_last_live = k;
      }
    }
    {
      const int sg_max = simd_max(my_last_live);
      if (simd_lane_id == 0) {
        sg_last_live[ktile & 1][simd_group_id] = sg_max;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int tile_last_live = sg_last_live[ktile & 1][0];
    STEEL_PRAGMA_UNROLL
    for (short w = 1; w < WM; ++w) {
      tile_last_live = metal::max(tile_last_live, sg_last_live[ktile & 1][w]);
    }
    if (tile_last_live < 0) {
      continue;
    }
    // 1..4, each arm covering another TK/4 fragments of live prefix.
    const short live_arm = short((tile_last_live / kFragSize) / (TK / 4)) + 1;
    const int live_slots = int(live_arm) * (BK / 4);

    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short dchunk = 0; dchunk < D_CHUNKS; ++dchunk) {
      const int dbase = int(dchunk) * DC;

      for (int u = lane; u < H * kVecsPerSlice; u += tgp_size) {
        const int h = u / kVecsPerSlice;
        const int v = u - h * kVecsPerSlice;
        reinterpret_cast<threadgroup uint4*>(&Qs[h * LDQ])[v] =
            reinterpret_cast<const device uint4*>(
                q_base + size_t(h) * params->Q_strides[1] + dbase)[v];
      }

      // One gathered row slice per thread: one address computation, DC/8
      // vector loads, transposed scalar stores (consecutive k per d stays
      // conflict-free).
      for (int k = lane; k < live_slots; k += tgp_size) {
        const int k_pos = selected[k];
        if (k_pos >= 0) {
          const device uint4* srcv = reinterpret_cast<const device uint4*>(
              (is_pooled_tile
                   ? pooled_base + size_t(k_pos) * params->Pooled_strides[1]
                   : local_base + size_t(k_pos) * params->Local_strides[2]) +
              dbase);
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < kVecsPerSlice; ++v) {
            const uint4 packed = srcv[v];
            const thread T* pe = reinterpret_cast<const thread T*>(&packed);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kElemsPerVec; ++j) {
              KVs[k + (v * kElemsPerVec + j) * LDK] = pe[j];
            }
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short d = 0; d < DC; ++d) {
            KVs[k + d * LDK] = T(0);
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      switch (live_arm) {
        case 1:
          KQ_DSA_QK_SWEEP(TK / 4);
          break;
        case 2:
          KQ_DSA_QK_SWEEP(TK / 2);
          break;
        case 3:
          KQ_DSA_QK_SWEEP(3 * TK / 4);
          break;
        default:
          KQ_DSA_QK_SWEEP(TK);
          break;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ++ii) {
      Stile.elems()[ii] *= scale;
    }

    {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; ++j) {
          const short col_pos = sn + j * stile_t::kFragCols;
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; ++jj) {
            if (selected[col_pos + jj] < 0) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    AccumType new_max[rows_per_thread];
    AccumType factor[rows_per_thread];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<KQDsaMaxOp>(new_max);
    Stile.template row_bin_op<KQDsaExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[rows_per_thread] = {0};
    Stile.template row_reduce<KQDsaSumOp>(sum_score_tmp);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<KQDsaMulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short vchunk = 0; vchunk < D_CHUNKS; ++vchunk) {
      const int dbase = int(vchunk) * DC;

      for (int k = lane; k < live_slots; k += tgp_size) {
        const int k_pos = selected[k];
        threadgroup uint4* dstv =
            reinterpret_cast<threadgroup uint4*>(&KVs[k * LDV]);
        if (k_pos >= 0) {
          const device uint4* srcv = reinterpret_cast<const device uint4*>(
              (is_pooled_tile
                   ? pooled_base + size_t(k_pos) * params->Pooled_strides[1]
                   : local_base + size_t(k_pos) * params->Local_strides[2]) +
              dbase);
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < kVecsPerSlice; ++v) {
            dstv[v] = srcv[v];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < kVecsPerSlice; ++v) {
            dstv[v] = uint4(0);
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      switch (live_arm) {
        case 1:
          KQ_DSA_PV_SWEEP(TK / 4);
          break;
        case 2:
          KQ_DSA_PV_SWEEP(TK / 2);
          break;
        case 3:
          KQ_DSA_PV_SWEEP(3 * TK / 4);
          break;
        default:
          KQ_DSA_PV_SWEEP(TK);
          break;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  Otile.template row_bin_op<KQDsaDivOp>(sum_score);

  device T* out = O + size_t(b) * params->O_strides[0] +
      size_t(q_pos) * params->O_strides[2] +
      size_t(tm + sm) * params->O_strides[1] + sn;
  Otile.template store<T, 1, 1>(out, params->O_strides[1]);
}

// Split-KV variant for small grids (decode / MTP verify): grid
// (n_splits, qL, B), each threadgroup covering key tiles
// ktile = split, split + n_splits, ... and writing unnormalized fp32
// partials (O_acc, m, l) instead of the final output. Sinks move to the
// merge kernel so they are counted once; a split with no live tiles writes
// m = finite_min, l = 0 and merges at zero weight. The live-slot double
// buffer is indexed by iteration parity, not ktile parity: with an even
// tile stride, ktile parity would alias every iteration onto one buffer
// and reintroduce the skip-path race the pairing exists to prevent.
// clang-format off
template <
    typename T,
    int BK,
    int DC,
    int H,
    int D,
    int WM,
    typename IndexT,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void kq_dsa_sparse_attention_split(
    const device T* Q [[buffer(0)]],
    const device T* LocalKV [[buffer(1)]],
    const device T* PooledKV [[buffer(2)]],
    const device IndexT* Topk [[buffer(3)]],
    device float* Oacc [[buffer(4)]],
    device float* Ms [[buffer(5)]],
    device float* Ls [[buffer(6)]],
    const constant KQDsaSparseAttentionParams* params [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 gpg [[threadgroups_per_grid]]) { // clang-format on

  constexpr short kFragSize = 8;
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ = DC + padQ;
  constexpr short LDK = BK + padK;
  constexpr short LDV = DC + padV;

  constexpr int kNWarps = WM;
  constexpr int TQ = H / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TDC = DC / kFragSize;
  constexpr int D_CHUNKS = D / DC;

  static_assert(TQ >= 1, "DeepSeek sparse attention needs a head tile.");
  static_assert(
      H % (kNWarps * kFragSize) == 0,
      "Head count must divide evenly across simdgroups.");
  static_assert(BK % kFragSize == 0, "BK must be a multiple of 8.");
  static_assert(DC % kFragSize == 0, "DC must be a multiple of 8.");
  static_assert(D % DC == 0, "Head dimension must divide DC.");
  static_assert(TK % 4 == 0, "Live-fragment arms slice the key tile in 4.");
  static_assert(
      sizeof(T) == 2,
      "Staging moves rows as uint4 (8 half/bfloat elements per load).");

  constexpr short kElemsPerVec = 16 / short(sizeof(T));
  constexpr short kVecsPerSlice = DC / kElemsPerVec;

  constexpr int tgp_size = WM * 32;
  const int lane = int(simd_group_id * 32 + simd_lane_id);

  const int split = int(tid.x);
  const int n_splits = int(gpg.x);
  const int q_pos = int(tid.y);
  const int b = int(tid.z);

  threadgroup uint4 Qs_v[(H * LDQ * short(sizeof(T)) + 15) / 16];
  threadgroup uint4 KVs_v
      [(((BK * LDV > DC * LDK) ? BK * LDV : DC * LDK) * short(sizeof(T)) + 15) /
       16];
  threadgroup T* Qs = reinterpret_cast<threadgroup T*>(Qs_v);
  threadgroup T* KVs = reinterpret_cast<threadgroup T*>(KVs_v);
  threadgroup int selected[BK];
  threadgroup int sg_last_live[2][WM];

  using MMAFragAcc = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  MMATile<AccumType, TQ, 1, MMAFragAcc> Qtile;
  MMATile<AccumType, 1, TK, MMAFragAcc> Ktile;
  MMATile<AccumType, TQ, TK, MMAFragAcc> Stile;
  MMATile<AccumType, 1, 1, MMAFragAcc> Vtile;
  MMATile<AccumType, TQ, D_CHUNKS * TDC, MMAFragAcc> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFragAcc::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ + sn;
  const short Ks_offset = sm * LDK + sn;
  const short Vs_offset = sm * LDV + sn;

  const AccumType scale = AccumType(params->scale * M_LOG2E_F);

  constexpr short rows_per_thread = decltype(Stile)::kRowsPerThread;
  AccumType max_score[rows_per_thread];
  AccumType sum_score[rows_per_thread] = {0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < rows_per_thread; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  const device T* q_base = Q + size_t(b) * params->Q_strides[0] +
      size_t(q_pos) * params->Q_strides[2];
  const device T* local_base = LocalKV + size_t(b) * params->Local_strides[0];
  const device T* pooled_base =
      PooledKV + size_t(b) * params->Pooled_strides[0];
  const device IndexT* topk_base = Topk + size_t(b) * params->Topk_strides[0] +
      size_t(q_pos) * params->Topk_strides[2];

  const int local_offset = params->localL - params->qL;
  const int local_end = metal::min(params->localL, local_offset + q_pos + 1);
  const int local_start = metal::max(0, local_end - params->local_window);
  const int local_count = metal::max(0, local_end - local_start);
  const int local_tiles = (local_count + BK - 1) / BK;
  const int pooled_tiles = (params->topk + BK - 1) / BK;
  const int pooled_valid = metal::min(
      params->pooledL, (params->q_offset + q_pos + 1) / params->compress_ratio);
  const int total_tiles = local_tiles + pooled_tiles;

  for (int ktile = split, it = 0; ktile < total_tiles;
       ktile += n_splits, ++it) {
    const bool is_pooled_tile = ktile >= local_tiles;
    const int tile_slot = is_pooled_tile ? (ktile - local_tiles) : ktile;
    const int slot_base = tile_slot * BK;

    int my_last_live = -1;
    for (int k = lane; k < BK; k += tgp_size) {
      const int slot = slot_base + k;
      int k_pos = -1;
      if (is_pooled_tile) {
        if (slot < params->topk) {
          const int pooled_pos = int(topk_base[slot]);
          if (pooled_pos >= 0 && pooled_pos < pooled_valid) {
            k_pos = pooled_pos;
          }
        }
      } else {
        if (slot < local_count) {
          k_pos = local_start + slot;
        }
      }
      selected[k] = k_pos;
      if (k_pos >= 0) {
        my_last_live = k;
      }
    }
    {
      const int sg_max = simd_max(my_last_live);
      if (simd_lane_id == 0) {
        sg_last_live[it & 1][simd_group_id] = sg_max;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int tile_last_live = sg_last_live[it & 1][0];
    STEEL_PRAGMA_UNROLL
    for (short w = 1; w < WM; ++w) {
      tile_last_live = metal::max(tile_last_live, sg_last_live[it & 1][w]);
    }
    if (tile_last_live < 0) {
      continue;
    }
    // 1..4, each arm covering another TK/4 fragments of live prefix.
    const short live_arm = short((tile_last_live / kFragSize) / (TK / 4)) + 1;
    const int live_slots = int(live_arm) * (BK / 4);

    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short dchunk = 0; dchunk < D_CHUNKS; ++dchunk) {
      const int dbase = int(dchunk) * DC;

      for (int u = lane; u < H * kVecsPerSlice; u += tgp_size) {
        const int h = u / kVecsPerSlice;
        const int v = u - h * kVecsPerSlice;
        reinterpret_cast<threadgroup uint4*>(&Qs[h * LDQ])[v] =
            reinterpret_cast<const device uint4*>(
                q_base + size_t(h) * params->Q_strides[1] + dbase)[v];
      }

      for (int k = lane; k < live_slots; k += tgp_size) {
        const int k_pos = selected[k];
        if (k_pos >= 0) {
          const device uint4* srcv = reinterpret_cast<const device uint4*>(
              (is_pooled_tile
                   ? pooled_base + size_t(k_pos) * params->Pooled_strides[1]
                   : local_base + size_t(k_pos) * params->Local_strides[2]) +
              dbase);
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < kVecsPerSlice; ++v) {
            const uint4 packed = srcv[v];
            const thread T* pe = reinterpret_cast<const thread T*>(&packed);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kElemsPerVec; ++j) {
              KVs[k + (v * kElemsPerVec + j) * LDK] = pe[j];
            }
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short d = 0; d < DC; ++d) {
            KVs[k + d * LDK] = T(0);
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      switch (live_arm) {
        case 1:
          KQ_DSA_QK_SWEEP(TK / 4);
          break;
        case 2:
          KQ_DSA_QK_SWEEP(TK / 2);
          break;
        case 3:
          KQ_DSA_QK_SWEEP(3 * TK / 4);
          break;
        default:
          KQ_DSA_QK_SWEEP(TK);
          break;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ++ii) {
      Stile.elems()[ii] *= scale;
    }

    {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; ++j) {
          const short col_pos = sn + j * stile_t::kFragCols;
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; ++jj) {
            if (selected[col_pos + jj] < 0) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    AccumType new_max[rows_per_thread];
    AccumType factor[rows_per_thread];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<KQDsaMaxOp>(new_max);
    Stile.template row_bin_op<KQDsaExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[rows_per_thread] = {0};
    Stile.template row_reduce<KQDsaSumOp>(sum_score_tmp);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<KQDsaMulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short vchunk = 0; vchunk < D_CHUNKS; ++vchunk) {
      const int dbase = int(vchunk) * DC;

      for (int k = lane; k < live_slots; k += tgp_size) {
        const int k_pos = selected[k];
        threadgroup uint4* dstv =
            reinterpret_cast<threadgroup uint4*>(&KVs[k * LDV]);
        if (k_pos >= 0) {
          const device uint4* srcv = reinterpret_cast<const device uint4*>(
              (is_pooled_tile
                   ? pooled_base + size_t(k_pos) * params->Pooled_strides[1]
                   : local_base + size_t(k_pos) * params->Local_strides[2]) +
              dbase);
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < kVecsPerSlice; ++v) {
            dstv[v] = srcv[v];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < kVecsPerSlice; ++v) {
            dstv[v] = uint4(0);
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      switch (live_arm) {
        case 1:
          KQ_DSA_PV_SWEEP(TK / 4);
          break;
        case 2:
          KQ_DSA_PV_SWEEP(TK / 2);
          break;
        case 3:
          KQ_DSA_PV_SWEEP(3 * TK / 4);
          break;
        default:
          KQ_DSA_PV_SWEEP(TK);
          break;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  device float* oacc_out = Oacc +
      ((size_t(b) * params->qL + q_pos) * n_splits + split) * size_t(H) * D +
      size_t(tm + sm) * D + sn;
  Otile.template store<float, 1, 1>(oacc_out, D);

  if (sn == 0) {
    const size_t ml_base =
        ((size_t(b) * params->qL + q_pos) * n_splits + split) * H;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      const int head = int(tm + sm) + i * kFragSize;
      if (head < params->H) {
        Ms[ml_base + head] = max_score[i];
        Ls[ml_base + head] = sum_score[i];
      }
    }
  }
}

// Renormalizing merge for the split variant: one simdgroup per
// (head, query position, batch) combines the n_splits unnormalized partials
// via the running-max identity, with the head's sink seeding the denominator
// exactly as in the single-dispatch kernel's softmax.
// clang-format off
template <typename T, int D>
[[kernel]] void kq_dsa_sparse_attention_merge(
    const device float* Oacc [[buffer(0)]],
    const device float* Ms [[buffer(1)]],
    const device float* Ls [[buffer(2)]],
    const device T* Sinks [[buffer(3)]],
    device T* O [[buffer(4)]],
    const constant KQDsaSparseAttentionParams* params [[buffer(5)]],
    const constant int& n_splits [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) { // clang-format on
  const int h = int(tid.x);
  const int q_pos = int(tid.y);
  const int b = int(tid.z);
  const int S = n_splits;
  const int H = params->H;

  const size_t ml_base = (size_t(b) * params->qL + q_pos) * size_t(S) * H + h;
  const float sink_m = M_LOG2E_F * float(Sinks[h]);

  float m_all = sink_m;
  for (int s = 0; s < S; ++s) {
    m_all = metal::max(m_all, Ms[ml_base + size_t(s) * H]);
  }
  float l_all = fast::exp2(sink_m - m_all);
  for (int s = 0; s < S; ++s) {
    l_all += Ls[ml_base + size_t(s) * H] *
        fast::exp2(Ms[ml_base + size_t(s) * H] - m_all);
  }

  const size_t acc_base =
      (size_t(b) * params->qL + q_pos) * size_t(S) * H * D + size_t(h) * D;
  device T* out = O + size_t(b) * params->O_strides[0] +
      size_t(h) * params->O_strides[1] + size_t(q_pos) * params->O_strides[2];
  for (int d = int(lane); d < D; d += 32) {
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
      acc += Oacc[acc_base + size_t(s) * H * D + d] *
          fast::exp2(Ms[ml_base + size_t(s) * H] - m_all);
    }
    out[d] = static_cast<T>(acc / l_all);
  }
}

#undef KQ_DSA_QK_SWEEP
#undef KQ_DSA_PV_SWEEP
