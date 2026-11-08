// int8 x int8 matmul on the tensor-op (NAX) units: y = x8 w8^T with
// per-row activation scales, one a row or one a row per K group, and
// per-channel weight scales.
//
// x8 [M, K] and w8 [N, K] are int8 and row-major, K a multiple of 32. One
// threadgroup computes a 128 x 128 block of the output with a single
// matmul2d over the block on 8 simdgroups, K in steps of 32, into an int32
// cooperative tensor. Each operand tensor starts at its block, A at
// x8 + row0 K and B at w8 + col0 K, with extents (K, 128), so no 32-bit
// offset inside the tensor ops passes 2 GiB. The output is written by the
// epilogue through 64-bit indices, never through a tensor.
//
// kq_int8_mm_nax (one scale a row): y[r, c] = OutT(float(acc) * sa[r] *
// sb[c]). kq_int8_mm_grouped_nax (G groups of gs columns, the last one
// short): per group the int32 tensor is cleared, the group's K runs into
// it, and acc += float(int32) * sa[r, g] in float32, in group order; then
// y[r, c] = OutT(acc * sb[c]). The block's row scales are staged in
// threadgroup memory when they fit (function constant 313).
//
// Tails: a block whose rows pass M reads its A rows from At, a copy of the
// last row block zero padded to 128 rows (kq_int8_pad_rows); a block whose
// columns pass N reads its B rows from Bt likewise. Stores skip r >= M,
// and c >= N unless N is a multiple of 128 (function constant 312); row
// scales past M are not read.

#pragma once

#include <metal_stdlib>
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "mlx/backend/metal/kernels/steel/defines.h"

// K of the call, fixed when the pipeline is built.
constant int kq_i8_k [[function_constant(310)]];
// Columns a group (grouped kernel), likewise.
constant int kq_i8_gs [[function_constant(311)]];
// N a multiple of 128: the stores skip their column check.
constant bool kq_i8_align_n [[function_constant(312)]];
// The grouped kernel's row scales, 128 x G floats, staged in threadgroup
// memory; device reads otherwise.
constant bool kq_i8_tg_scales [[function_constant(313)]];

struct KqInt8MMParams {
  int M;
  int N;
  int G; // activation scales a row
  int tiles_m;
};

// The live rows of one tile of a row-major [rows, K] int8 operand into a
// [tile_rows, K] buffer, zero past them. src starts at the tile's first
// row; the grid is (K / 16, tile_rows) threads.
[[kernel]] void kq_int8_pad_rows(
    const device int8_t* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    const constant int& K [[buffer(2)]],
    const constant int& live [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const size_t off = size_t(gid.y) * size_t(K) + size_t(gid.x) * 16;
  uint4 v = uint4(0);
  if (int(gid.y) < live) {
    v = *(const device uint4*)(src + off);
  }
  *(device uint4*)(dst + off) = v;
}

namespace kq_int8_mm {

constexpr constant int BM = 128;
constexpr constant int BN = 128;
constexpr constant int NSG = 8;
constexpr constant int SWZ = 2;
constexpr constant int SK = 32;

namespace to = mpp::tensor_ops;

constexpr constant auto kDesc = to::matmul2d_descriptor(
    BM,
    BN,
    SK,
    false,
    true,
    false,
    to::matmul2d_descriptor::mode::multiply_accumulate);

using Op = to::matmul2d<kDesc, metal::execution_simdgroups<NSG>>;
using Tensor = metal::
    tensor<device int8_t, metal::dextents<int32_t, 2>, metal::tensor_inline>;

// The block of threadgroup tid, (column, row): consecutive groups of 2^SWZ
// threadgroups walk down the rows of one column of blocks.
METAL_FUNC int2 block_of(uint3 tid) {
  return int2(
      int(tid.x) >> SWZ, (int(tid.y) << SWZ) + (int(tid.x) & ((1 << SWZ) - 1)));
}

// The block's operand: base + first * K, or the padded tile when the block
// passes `rows`.
METAL_FUNC Tensor operand(
    const device int8_t* base,
    const device int8_t* tile,
    int first,
    int rows,
    int K,
    int extent) {
  const device int8_t* p =
      first + extent > rows ? tile : base + size_t(first) * size_t(K);
  return Tensor((device int8_t*)p, metal::dextents<int32_t, 2>(K, extent));
}

} // namespace kq_int8_mm

template <typename OutT>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_int8_mm_nax(
    const device int8_t* A [[buffer(0)]],
    const device int8_t* B [[buffer(1)]],
    const device float* sa [[buffer(2)]],
    const device float* sb [[buffer(3)]],
    device OutT* D [[buffer(4)]],
    const device int8_t* At [[buffer(5)]],
    const device int8_t* Bt [[buffer(6)]],
    const constant KqInt8MMParams& p [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using namespace kq_int8_mm;
  const int2 blk = block_of(tid);
  if (blk.y >= p.tiles_m) {
    return;
  }
  const int row0 = blk.y * BM;
  const int col0 = blk.x * BN;
  const int K = kq_i8_k;
  const int M = p.M;
  const int N = p.N;
  Op op;
  auto tA = operand(A, At, row0, M, K, BM);
  auto tB = operand(B, Bt, col0, N, K, BN);
  auto a0 = tA.template slice<SK, BM>(0, 0);
  auto b0 = tB.template slice<SK, BN>(0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(a0),
      decltype(b0),
      int>();
  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    cT[i] = 0;
  }
  for (int k = 0; k < K; k += SK) {
    auto a = tA.template slice<SK, BM>(k, 0);
    auto b = tB.template slice<SK, BN>(k, 0);
    op.run(a, b, cT);
  }

  const int G = p.G;
  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto idx = cT.get_multidimensional_index(i);
      const int c = col0 + int(idx[0]);
      const int r = row0 + int(idx[1]);
      if (r < M && (kq_i8_align_n || c < N)) {
        D[size_t(r) * size_t(N) + size_t(c)] =
            OutT(float(cT[i]) * sa[size_t(r) * size_t(G)] * sb[c]);
      }
    }
  }
}

template <typename OutT>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_int8_mm_grouped_nax(
    const device int8_t* A [[buffer(0)]],
    const device int8_t* B [[buffer(1)]],
    const device float* sa [[buffer(2)]],
    const device float* sb [[buffer(3)]],
    device OutT* D [[buffer(4)]],
    const device int8_t* At [[buffer(5)]],
    const device int8_t* Bt [[buffer(6)]],
    const constant KqInt8MMParams& p [[buffer(7)]],
    threadgroup float* tsa [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]) {
  using namespace kq_int8_mm;
  constexpr int CAP = BM * BN / (NSG * 32);
  const int2 blk = block_of(tid);
  if (blk.y >= p.tiles_m) {
    return;
  }
  const int row0 = blk.y * BM;
  const int col0 = blk.x * BN;
  const int K = kq_i8_k;
  const int M = p.M;
  const int N = p.N;
  const int G = p.G;
  const int gs = kq_i8_gs;
  Op op;
  auto tA = operand(A, At, row0, M, K, BM);
  auto tB = operand(B, Bt, col0, N, K, BN);
  auto a0 = tA.template slice<SK, BM>(0, 0);
  auto b0 = tB.template slice<SK, BN>(0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(a0),
      decltype(b0),
      int>();
  // acc holds a thread's elements of the block, 128 x 128 over 256 threads.
  // The loops below stop at the smaller of CAP and the tensor's capacity
  // (a runtime value) and skip invalid elements; a capacity above CAP
  // would leave elements unstored, never overrun acc.
  const uint16_t cap = metal::min(uint16_t(CAP), cT.get_capacity());

  // Row scales of the block, [BM, G]; rows past M read as 0 and are never
  // stored.
  const size_t live_scales = size_t(metal::min(BM, M - row0)) * size_t(G);
  const device float* sar = sa + size_t(row0) * size_t(G);
  if (kq_i8_tg_scales) {
    for (int e = int(lid); e < BM * G; e += NSG * 32) {
      tsa[e] = size_t(e) < live_scales ? sar[e] : 0.f;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }

  float acc[CAP];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < CAP; ++i) {
    acc[i] = 0.f;
  }
  // Whole groups run a fixed number of steps; a short last group (K not a
  // multiple of gs) runs its own.
  const int whole = K / gs;
  for (int g = 0; g < G; g++) {
    const int k0 = g * gs;
    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      cT[i] = 0;
    }
    if (g < whole) {
      for (int k = 0; k < gs; k += SK) {
        auto a = tA.template slice<SK, BM>(k0 + k, 0);
        auto b = tB.template slice<SK, BN>(k0 + k, 0);
        op.run(a, b, cT);
      }
    } else {
      for (int k = k0; k < K; k += SK) {
        auto a = tA.template slice<SK, BM>(k, 0);
        auto b = tB.template slice<SK, BN>(k, 0);
        op.run(a, b, cT);
      }
    }
    STEEL_PRAGMA_UNROLL
    for (uint16_t i = 0; i < cap; ++i) {
      if (!cT.is_valid_element(i)) {
        continue;
      }
      const int r = int(cT.get_multidimensional_index(i)[1]);
      float s;
      if (kq_i8_tg_scales) {
        s = tsa[r * G + g];
      } else {
        s = row0 + r < M ? sar[size_t(r) * size_t(G) + size_t(g)] : 0.f;
      }
      acc[i] += float(cT[i]) * s;
    }
  }

  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < cap; ++i) {
    if (!cT.is_valid_element(i)) {
      continue;
    }
    auto idx = cT.get_multidimensional_index(i);
    const int c = col0 + int(idx[0]);
    const int r = row0 + int(idx[1]);
    if (r < M && (kq_i8_align_n || c < N)) {
      D[size_t(r) * size_t(N) + size_t(c)] = OutT(acc[i] * sb[c]);
    }
  }
}
