// Tensor-op (MPP matmul2d) ports of the DeepSeek-V4-Flash lightning-indexer
// score GEMM. Two kernels:
//
//  * kq_dsa_indexer_score_mma -- drop-in replacement for
//    kq_dsa_indexer_score (kq_dsa_indexer.h): per query row m and key
//    column n, sum_h relu(q[h,m] . k[n]) * w[h,m], fp32 accumulation,
//    T output, identical causal semantics (future tiles store -inf or
//    skip, unused-causal-prefix tiles skip, per-element future -inf).
//
//  * kq_dsa_indexer_score_i8mx -- same score math on pre-quantized
//    operands: int8 codes (E2M1 values doubled, in [-12, 12]) with
//    per-32-block power-of-two scales (stored pre-folded as scale * 0.5,
//    so value = code * scale_half exactly). Each 128-dim dot runs as four
//    K=32 int8 MMA segments accumulated in int32 (exact), rescaled by
//    sq * sk per segment. Products are dyadic rationals: the result is
//    bit-identical to running the fp16 kernel on the dequantized codes.
//
// Both use one 64x32 output tile per threadgroup, 4 simdgroups. M and N
// keep the legacy multiples-of-64 contract. Requires tensor-op hardware
// (kq_is_nax_available); host falls back to the simdgroup kernel.

#pragma once

#include <metal_stdlib>
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "mlx/backend/metal/kernels/steel/defines.h"

using namespace metal;

constant bool kq_dsa_mma_do_causal [[function_constant(310)]];
constant bool kq_dsa_mma_weights_lh [[function_constant(311)]];

namespace kq_dsa_mma {

constant constexpr int BM = 64;
constant constexpr int BN = 32;
constant constexpr int HD = 128;
constant constexpr int NSEG = 4; // 128 / 32 scale blocks
constant constexpr int THREADS = 128; // 4 simdgroups
// Destination cooperative tensor slots per thread for a 64x32 fp32/int32
// tile over 128 threads (2048 / 128). The layout is implementation
// defined; runtime capacity above this aborts the store (never observed).
constant constexpr int CAP_MAX = 16;

// Legacy-kernel causal tile handling, shared by both kernels. Returns
// true when the tile is fully handled (skip or -inf store) and the
// kernel should exit. O points at the batch base.
template <typename T>
METAL_FUNC bool causal_tile_prologue(
    device T* O,
    int c_row,
    int c_col,
    int M,
    int N,
    int q_offset,
    int unused_causal_prefix_topk,
    bool skip_causal_future_store,
    int thread_idx) {
  if (!kq_dsa_mma_do_causal) {
    return false;
  }
  const int row_limit = metal::min(c_row + BM, M);
  if (c_col > q_offset + row_limit - 1) {
    if (!skip_causal_future_store) {
      device T* Dst = O + size_t(c_row) * N + c_col;
      for (int e = thread_idx; e < BM * BN; e += THREADS) {
        const int row = e / BN;
        const int col = e - row * BN;
        if (c_row + row < M && c_col + col < N) {
          Dst[size_t(row) * N + col] = static_cast<T>(-INFINITY);
        }
      }
    }
    return true;
  }
  if (unused_causal_prefix_topk > 0 &&
      q_offset + row_limit <= unused_causal_prefix_topk) {
    return true;
  }
  return false;
}

} // namespace kq_dsa_mma

template <typename T>
[[kernel, max_total_threads_per_threadgroup(kq_dsa_mma::THREADS)]] void
kq_dsa_indexer_score_mma(
    device T* Q [[buffer(0)]],
    device T* K [[buffer(1)]],
    device T* W [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant int& M [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& H [[buffer(6)]],
    const constant int& unused_causal_prefix_topk [[buffer(7)]],
    const constant bool& skip_causal_future_store [[buffer(8)]],
    const constant int& causal_q_offset [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using namespace kq_dsa_mma;
  namespace to = mpp::tensor_ops;

  const int c_row = int(tid.y) * BM;
  const int c_col = int(tid.x) * BN;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;

  Q += size_t(tid.z) * H * M * HD;
  K += size_t(tid.z) * N * HD;
  W += size_t(tid.z) * H * M;
  O += size_t(tid.z) * M * N;

  if (causal_tile_prologue(
          O,
          c_row,
          c_col,
          M,
          N,
          q_offset,
          unused_causal_prefix_topk,
          skip_causal_future_store,
          int(lid.x))) {
    return;
  }

  constexpr auto desc = to::matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      true,
      true,
      to::matmul2d_descriptor::mode::multiply_accumulate);
  to::matmul2d<desc, execution_simdgroups<4>> op;

  auto tB_full = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
      K, dextents<int32_t, 2>(HD, N));
  auto tB = tB_full.template slice<dynamic_extent, BN>(0, c_col);
  auto tA0_full = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
      Q, dextents<int32_t, 2>(HD, M));
  auto tA0 = tA0_full.template slice<dynamic_extent, BM>(0, c_row);

  auto sT = op.template get_destination_cooperative_tensor<
      decltype(tA0),
      decltype(tB),
      float>();
  const uint16_t cap = sT.get_capacity();
  if (cap > CAP_MAX) {
    return;
  }

  // Per-slot output coords, fixed across heads: idx[0] = n, idx[1] = m.
  int rowg[CAP_MAX];
  int colg[CAP_MAX];
  bool live[CAP_MAX];
  float acc[CAP_MAX];
#pragma unroll
  for (uint16_t i = 0; i < cap; ++i) {
    live[i] = sT.is_valid_element(i);
    if (live[i]) {
      auto idx = sT.get_multidimensional_index(i);
      colg[i] = c_col + int(idx[0]);
      rowg[i] = c_row + int(idx[1]);
    } else {
      colg[i] = 0;
      rowg[i] = 0;
    }
    acc[i] = 0.0f;
  }

  for (int h = 0; h < H; ++h) {
    auto tA_full = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
        Q + size_t(h) * M * HD, dextents<int32_t, 2>(HD, M));
    auto tA = tA_full.template slice<dynamic_extent, BM>(0, c_row);

#pragma unroll
    for (uint16_t i = 0; i < cap; ++i) {
      sT[i] = 0.0f;
    }
    op.run(tA, tB, sT);

#pragma unroll
    for (uint16_t i = 0; i < cap; ++i) {
      if (live[i]) {
        const float w = kq_dsa_mma_weights_lh
            ? static_cast<float>(W[size_t(rowg[i]) * H + h])
            : static_cast<float>(W[size_t(h) * M + rowg[i]]);
        acc[i] += metal::max(sT[i], 0.0f) * w;
      }
    }
  }

#pragma unroll
  for (uint16_t i = 0; i < cap; ++i) {
    if (live[i]) {
      const bool future = kq_dsa_mma_do_causal && colg[i] > q_offset + rowg[i];
      O[size_t(rowg[i]) * N + colg[i]] =
          future ? static_cast<T>(-INFINITY) : static_cast<T>(acc[i]);
    }
  }
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(kq_dsa_mma::THREADS)]] void
kq_dsa_indexer_score_i8mx(
    device int8_t* Qc [[buffer(0)]],
    device float* SQ [[buffer(1)]],
    device int8_t* Kc [[buffer(2)]],
    device float* SK [[buffer(3)]],
    device float* W [[buffer(4)]],
    device T* O [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& H [[buffer(8)]],
    const constant int& unused_causal_prefix_topk [[buffer(9)]],
    const constant bool& skip_causal_future_store [[buffer(10)]],
    const constant int& causal_q_offset [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using namespace kq_dsa_mma;
  namespace to = mpp::tensor_ops;

  const int c_row = int(tid.y) * BM;
  const int c_col = int(tid.x) * BN;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;
  const int thread_idx = int(lid.x);

  Qc += size_t(tid.z) * H * M * HD;
  SQ += size_t(tid.z) * H * M * NSEG;
  Kc += size_t(tid.z) * N * HD;
  SK += size_t(tid.z) * N * NSEG;
  W += size_t(tid.z) * H * M;
  O += size_t(tid.z) * M * N;

  if (causal_tile_prologue(
          O,
          c_row,
          c_col,
          M,
          N,
          q_offset,
          unused_causal_prefix_topk,
          skip_causal_future_store,
          thread_idx)) {
    return;
  }

  constexpr auto desc = to::matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      true,
      false,
      to::matmul2d_descriptor::mode::multiply_accumulate);
  to::matmul2d<desc, execution_simdgroups<4>> op;

  auto tB_full = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
      Kc, dextents<int32_t, 2>(HD, N));
  auto tA0_full = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
      Qc, dextents<int32_t, 2>(HD, M));
  auto tA0 = tA0_full.template slice<32, BM>(0, c_row);
  auto tB0 = tB_full.template slice<32, BN>(0, c_col);

  auto segT = op.template get_destination_cooperative_tensor<
      decltype(tA0),
      decltype(tB0),
      int>();
  const uint16_t cap = segT.get_capacity();
  if (cap > CAP_MAX) {
    return;
  }

  int rowl[CAP_MAX]; // tile-local row
  int coll[CAP_MAX]; // tile-local col
  bool live[CAP_MAX];
  float acc[CAP_MAX];
#pragma unroll
  for (uint16_t i = 0; i < cap; ++i) {
    live[i] = segT.is_valid_element(i);
    if (live[i]) {
      auto idx = segT.get_multidimensional_index(i);
      coll[i] = int(idx[0]);
      rowl[i] = int(idx[1]);
    } else {
      coll[i] = 0;
      rowl[i] = 0;
    }
    acc[i] = 0.0f;
  }

  // Scale/weight staging: SK once (col-invariant across heads), SQ and W
  // per head. Converts per-slot scattered device loads into threadgroup
  // reads (the epilogue is the i8 arm's overhead term).
  threadgroup float sk_tile[BN * NSEG];
  threadgroup float sq_tile[BM * NSEG];
  threadgroup float w_tile[BM];

  for (int e = thread_idx; e < BN * NSEG; e += THREADS) {
    sk_tile[e] = SK[size_t(c_col) * NSEG + e];
  }

  for (int h = 0; h < H; ++h) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* SQh = SQ + size_t(h) * M * NSEG + size_t(c_row) * NSEG;
    for (int e = thread_idx; e < BM * NSEG; e += THREADS) {
      sq_tile[e] = SQh[e];
    }
    for (int e = thread_idx; e < BM; e += THREADS) {
      w_tile[e] = kq_dsa_mma_weights_lh ? W[size_t(c_row + e) * H + h]
                                        : W[size_t(h) * M + c_row + e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float score[CAP_MAX];
#pragma unroll
    for (uint16_t i = 0; i < cap; ++i) {
      score[i] = 0.0f;
    }

    device int8_t* Qh = Qc + size_t(h) * M * HD;
#pragma unroll
    for (int s = 0; s < NSEG; ++s) {
      auto tA_full = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
          Qh, dextents<int32_t, 2>(HD, M));
      auto tA = tA_full.template slice<32, BM>(s * 32, c_row);
      auto tB = tB_full.template slice<32, BN>(s * 32, c_col);

#pragma unroll
      for (uint16_t i = 0; i < cap; ++i) {
        segT[i] = 0;
      }
      op.run(tA, tB, segT);

#pragma unroll
      for (uint16_t i = 0; i < cap; ++i) {
        if (live[i]) {
          score[i] += float(segT[i]) * sq_tile[rowl[i] * NSEG + s] *
              sk_tile[coll[i] * NSEG + s];
        }
      }
    }

#pragma unroll
    for (uint16_t i = 0; i < cap; ++i) {
      if (live[i]) {
        acc[i] += metal::max(score[i], 0.0f) * w_tile[rowl[i]];
      }
    }
  }

#pragma unroll
  for (uint16_t i = 0; i < cap; ++i) {
    if (live[i]) {
      const int row = c_row + rowl[i];
      const int col = c_col + coll[i];
      const bool future = kq_dsa_mma_do_causal && col > q_offset + row;
      O[size_t(row) * N + col] =
          future ? static_cast<T>(-INFINITY) : static_cast<T>(acc[i]);
    }
  }
}
