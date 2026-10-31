// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// MiniMax-H3 video sparse attention (VSA), the block-sparse part, on the
// tensor-op (NAX) units. q, k, v are [n_tiles * TILE, heads, 128], and tile
// t holds vbs[t] live rows, then pad slots. Tiles [0, n_prefix) are the dense
// prefix. Video query tile j of head h reads the first block_num[h, j] key
// tiles of block_idx[h, j], and of each only its live rows. Only the live
// rows of the video query tiles are written. With GATED, each written row
// also adds the pooled-compression term bf16(coarse[h, tile] * gate[row]) to
// its bfloat16 output, rounded as MLX's unfused ops round it.
//
// One threadgroup per BQ query rows, video tile and head, 16 query rows per
// simdgroup and keys in blocks of 32. The fragment math, the online softmax
// and the block order are those of MLX's attention_nax, so the output is
// MLX's fused attention over the gathered key tiles with the dead rows
// masked, bit for bit.
//
// Ported from FastVideo (https://github.com/hao-ai-lab/FastVideo), the
// h3_vsa_sparse_nax kernel of fastvideo/mlx_runtime/minimax_h3_vsa_nax.py,
// licensed under the Apache License, Version 2.0 (see
// mlx_kquant/licenses/FastVideo-LICENSE). Changes for mlx-kquant: a
// compiled kernel with buffer arguments in place of an mx.fast.metal_kernel
// source, block_num clamped to [0, k_max], a key tile index outside
// [0, n_tiles) skipped, vbs clamped to [0, TILE], and the gate term of
// FastVideo's h3_vsa_attention added in the epilogue (GATED). The attention
// math is unchanged.
//
// FastVideo's kernel is adapted from MLX (steel_attention_nax.h, Copyright
// 2024-26 Apple Inc.). Its MIT notice follows.
//
// MIT License
//
// Copyright © 2023 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/attn/nax.h"

using namespace metal;
using namespace mlx::steel;

namespace kq_vsa {

struct MaxOp {
  template <typename V>
  METAL_FUNC static constexpr V apply(V x, V y) {
    return metal::max(x, y);
  }
};
struct SumOp {
  template <typename V>
  METAL_FUNC static constexpr V apply(V x, V y) {
    return x + y;
  }
};
struct MulOp {
  template <typename V>
  METAL_FUNC static constexpr V apply(V x, V y) {
    return x * y;
  }
};
struct ExpSubOp {
  template <typename V>
  METAL_FUNC static constexpr V apply(V x, V y) {
    return metal::fast::exp2(x - y);
  }
};

} // namespace kq_vsa

template <typename T, int TILE, int BQ, bool GATED>
[[kernel, max_total_threads_per_threadgroup(BQ / 16 * 32)]] void kq_vsa_nax(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device int* block_idx [[buffer(3)]],
    const device int* block_num [[buffer(4)]],
    const device int* vbs [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& k_max [[buffer(7)]],
    const constant int& n_prefix [[buffer(8)]],
    const constant int& n_video [[buffer(9)]],
    const constant int& heads [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const device float* coarse [[buffer(12)]],
    const device T* gate [[buffer(13)]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  using namespace kq_vsa;
  constexpr int D = 128;
  constexpr int BK = 32;
  constexpr int TK = BK / 16;
  constexpr int TD = D / 16;
  constexpr float NEG = -3.402823466e38f;
  static_assert(TILE % BQ == 0 && BQ % 16 == 0, "BQ splits the tile");

  const int qsub = int(threadgroup_position_in_grid.x);
  const int query_tile = int(threadgroup_position_in_grid.y);
  const int head = int(threadgroup_position_in_grid.z);
  const int sid = int(simdgroup_index_in_threadgroup);
  const int n_tiles = n_prefix + n_video;
  const int qt = n_prefix + query_tile;
  const int q_valid = clamp(vbs[qt], 0, TILE);
  const int row_base = qsub * BQ;
  const int nsel = clamp(block_num[head * n_video + query_tile], 0, k_max);

  // Uniform across the threadgroup: later barriers cannot be skipped by only
  // some simdgroups. Inactive simdgroups use guarded loads and stores. A tile
  // with no key tile writes zero, or the gate term alone.
  if (row_base >= q_valid) {
    return;
  }

  const int row_stride = heads * D;
  const int qrow = row_base + sid * 16;
  const short q_remaining = short(q_valid - qrow);
  const device T* Q = q + (long(qt * TILE + qrow) * heads + head) * D;
  device T* O = out + (long(qt * TILE + qrow) * heads + head) * D;
  const float scale2 = scale * 1.44269504088896341f;

  using otile_t = NAXTile<float, 1, TD>;
  using stile_t = NAXTile<float, 1, TK>;
  otile_t Otile;
  Otile.clear();
  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<float, kRowsPT> max_score;
  metal::vec<float, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = NEG;
  }
  const short sn = otile_t::NAXFrag_t::get_coord().x;

  for (int selected = 0; selected < nsel; ++selected) {
    const int kt = block_idx[(head * n_video + query_tile) * k_max + selected];
    if (kt < 0 || kt >= n_tiles) {
      continue;
    }
    const int k_valid = clamp(vbs[kt], 0, TILE);
    const long kv_offset = (long(kt) * TILE * heads + head) * D;
    for (int krow = 0; krow < k_valid; krow += BK) {
      const short k_remaining = short(metal::min(BK, k_valid - krow));
      const device T* K = k + kv_offset + long(krow) * row_stride;
      const device T* V = v + kv_offset + long(krow) * row_stride;
      stile_t Stile;
      Stile.clear();

      // S = Q K^T, with MLX's four-way partial unroll of the head dim.
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; ++id) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;
          if (q_remaining < 16) {
            Qtile.load_rows(Q + id * 16, row_stride, q_remaining);
          } else {
            Qtile.load(Q + id * 16, row_stride);
          }
          if (k_remaining < BK) {
            Ktile.load_rows(
                K + ik * 16 * row_stride + id * 16,
                row_stride,
                k_remaining - ik * 16);
          } else {
            Ktile.load(K + ik * 16 * row_stride + id * 16, row_stride);
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
      for (short i = 0; i < stile_t::kElemsPerTile; ++i) {
        Stile.elems()[i] *= scale2;
      }
      if (k_remaining < BK) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          thread auto& frag = Stile.frag_at(0, ik);
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < stile_t::kFragThrRows; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < stile_t::kFragThrCols; ++j) {
              if (ik * 16 + sn + j >= k_remaining) {
                frag[i * stile_t::kFragThrCols + j] = NEG;
              }
            }
          }
        }
      }

      // Online softmax.
      metal::vec<float, kRowsPT> new_max = max_score;
      metal::vec<float, kRowsPT> factor;
      Stile.template row_reduce<MaxOp>(new_max);
      Stile.template row_bin_op<ExpSubOp>(new_max);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        factor[i] = metal::fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
        sum_score[i] *= factor[i];
      }
      Stile.template row_reduce<SumOp>(sum_score);
      Otile.template row_bin_op<MulOp>(factor);
      simdgroup_barrier(mem_flags::mem_none);

      // O += P V, P kept in float32 as MLX's NAX attention keeps it.
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if (id == 4) {
          threadgroup_barrier(mem_flags::mem_none);
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          NAXTile<T, 1, 2> Vtile;
          if (k_remaining < BK) {
            Vtile.load_rows(
                V + ik * 16 * row_stride + id * 16,
                row_stride,
                k_remaining - ik * 16);
          } else {
            Vtile.load(V + ik * 16 * row_stride + id * 16, row_stride);
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
    }
  }
  threadgroup_barrier(mem_flags::mem_none);
  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = sum_score[i] > 0.f ? 1.f / sum_score[i] : 0.f;
  }
  Otile.template row_bin_op<MulOp>(rcp);
  if (q_remaining <= 0) {
    return;
  }
  if constexpr (GATED) {
    // out = bf16(o) + bf16(coarse * float(gate)): the attention output and
    // the product each rounded to T, then their sum, as MLX's Multiply,
    // AsType and Add kernels compute it.
    const short2 sc = otile_t::NAXFrag_t::get_coord();
    const device float* C = coarse + (long(head) * n_tiles + qt) * D;
    const device T* G = gate + (long(qt * TILE + qrow) * heads + head) * D;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; ++id) {
      const thread auto& frag = Otile.frag_at(0, id);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < otile_t::kFragThrRows; ++i) {
        const short r = sc.y + i * otile_t::kFragRowsJump;
        if (r < q_remaining) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < otile_t::kFragThrCols; ++j) {
            const int c = id * 16 + sc.x + j;
            const T o = static_cast<T>(frag[i * otile_t::kFragThrCols + j]);
            const T g = static_cast<T>(
                C[c] * static_cast<float>(G[r * row_stride + c]));
            O[r * row_stride + c] = static_cast<T>(o + g);
          }
        }
      }
    }
  } else if (q_remaining < 16) {
    Otile.store_rows(O, row_stride, q_remaining);
  } else {
    Otile.store(O, row_stride);
  }
}

// Zero-fills the output before kq_vsa_nax writes its live rows, 16 bytes a
// thread.
[[kernel]] void kq_vsa_zero(
    device uint4* out [[buffer(0)]],
    const constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (i < n) {
    out[i] = uint4(0);
  }
}
