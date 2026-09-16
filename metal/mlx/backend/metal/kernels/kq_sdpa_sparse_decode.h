// Sparse decode attention over two key sources: a window [W, D] read whole
// and a pool [P, D] read through a per-query index list [N]. K == V (the
// absorbed-MLA latent).
//
// Split kernel: one threadgroup of two simdgroups per (8-head group, query,
// key split). Blocks of 8 key rows are staged through one threadgroup
// buffer (each simdgroup stages and owns one half of D), scored
// against the 8 queries with simdgroup matrix ops (partial scores summed
// across the two halves), run through an exp2-space online softmax held per
// row in the fragment layout, and accumulated into fp32 output tiles.
// Masked or out-of-range rows score kNegBig and weigh zero, which matches
// an additive -inf mask without inf arithmetic. The footprint stays small
// (one 8-row buffer, the queries pass through it once) so several
// threadgroups share a core. The merge kernel renormalizes the split
// partials with the head's sink counted once.
#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#include "mlx/backend/metal/kernels/kq_sdpa_sparse_decode_params.h"

constant constexpr float kSdpaSparseNegBig = -1e30f;

#define KQ_UNROLL _Pragma("clang loop unroll(full)")

template <typename T, typename IdxT, int D>
[[kernel]] void kq_sdpa_sparse_decode_split(
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
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int HG = 8; // heads per threadgroup
  constexpr int KB = 8; // keys per block
  constexpr int LD = D + 8; // padded row (bank spread, 16-byte aligned)
  constexpr int DH = D / 2; // dims per simdgroup
  constexpr int NT = DH / 8; // 8x8 tiles per simdgroup along D
  constexpr int CH = DH / 8; // 16-byte chunks per half row
  using MatT = metal::simdgroup_matrix<T, 8, 8>;
  using MatF = metal::simdgroup_matrix<float, 8, 8>;

  static_assert(HG == KB, "the query rows pass through the key buffer");
  threadgroup T Ks[KB * LD];
  threadgroup float Sx[2][32][2];

  const int s = int(tid.x);
  const int h0 = int(tid.y) * HG;
  const int b = int(tid.z) / p->L;
  const int l = int(tid.z) % p->L;
  const int g = int(simd_gid);
  const int H = p->H;
  const int tix = g * 32 + int(lane);

  // Queries of the 8 heads (rows past H repeat the last head; their
  // partials are never read).
  {
    const device T* qb =
        Q + size_t(b) * p->q_strides[0] + size_t(l) * p->q_strides[2];
    constexpr int QCH = D / 8;
    for (int v = tix; v < HG * QCH; v += 64) {
      const int r = v / QCH;
      const int c = v % QCH;
      const int h = metal::min(h0 + r, H - 1);
      const uint4 x = *reinterpret_cast<const device uint4*>(
          qb + size_t(h) * p->q_strides[1] + c * 8);
      *reinterpret_cast<threadgroup uint4*>(Ks + r * LD + c * 8) = x;
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  MatT Qt[NT];
  KQ_UNROLL
  for (int t = 0; t < NT; ++t) {
    simdgroup_load(Qt[t], Ks + g * DH + t * 8, LD);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  MatF Ot[NT];
  KQ_UNROLL
  for (int t = 0; t < NT; ++t) {
    Ot[t] = MatF(0.0f);
  }

  // Fragment layout of an 8x8 simdgroup matrix: this lane holds row fm,
  // columns fn and fn + 1; lanes lane^1 and lane^8 share the row.
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  float m_row = kSdpaSparseNegBig;
  float l_row = 0.0f;

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
  const device bool* sm = p->has_sel_mask
      ? SelMask + size_t(b) * p->sel_mask_strides[0] +
          size_t(l) * p->sel_mask_strides[1]
      : nullptr;
  // Dead slots stage a row that exists so every load stays unconditional.
  const device T* safe_row = W > 0 ? win_b : pool_b;
  const float sl2 = p->scale_log2;

  for (int j0 = k0; j0 < k1; j0 += KB) {
    // Rows of this block (uniform across the threadgroup).
    const device T* rows[KB];
    int okbits = 0;
    KQ_UNROLL
    for (int k = 0; k < KB; ++k) {
      const int j = j0 + k;
      bool v = j < k1;
      const device T* row = safe_row;
      if (v) {
        if (j < W) {
          v = (wm == nullptr) || wm[j];
          row = win_b + size_t(j) * p->win_strides[1];
        } else {
          const int n = j - W;
          v = (sm == nullptr) || sm[n];
          if (v) {
            const int64_t r = int64_t(idx_bl[n]);
            v = r >= 0 && r < int64_t(p->P);
            if (v) {
              row = pool_b + size_t(r) * p->pool_strides[1];
            }
          }
        }
      }
      okbits |= int(v) << k;
      rows[k] = row + g * DH;
    }
    // Stage this simdgroup's half of the 8 rows.
    if (int(lane) < CH) {
      KQ_UNROLL
      for (int k = 0; k < KB; ++k) {
        const uint4 x = reinterpret_cast<const device uint4*>(rows[k])[lane];
        *reinterpret_cast<threadgroup uint4*>(Ks + k * LD + g * DH + lane * 8) =
            x;
      }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Partial scores over this half of D, then the other half's.
    MatF St = MatF(0.0f);
    KQ_UNROLL
    for (int t = 0; t < NT; ++t) {
      MatT Kt;
      simdgroup_load(Kt, Ks + g * DH + t * 8, LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(St, Qt[t], Kt, St);
    }
    Sx[g][lane][0] = St.thread_elements()[0];
    Sx[g][lane][1] = St.thread_elements()[1];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const bool ok0 = (okbits >> fn) & 1;
    const bool ok1 = (okbits >> (fn + 1)) & 1;
    const float s0 = ok0 ? (St.thread_elements()[0] + Sx[1 - g][lane][0]) * sl2
                         : kSdpaSparseNegBig;
    const float s1 = ok1 ? (St.thread_elements()[1] + Sx[1 - g][lane][1]) * sl2
                         : kSdpaSparseNegBig;

    // Online softmax per row (the 4 lanes of a row hold the same state).
    float mb = metal::max(s0, s1);
    mb = metal::max(mb, simd_shuffle_xor(mb, 1));
    mb = metal::max(mb, simd_shuffle_xor(mb, 8));
    const float m_new = metal::max(m_row, mb);
    const float alpha = fast::exp2(m_row - m_new);
    const float p0 = ok0 ? fast::exp2(s0 - m_new) : 0.0f;
    const float p1 = ok1 ? fast::exp2(s1 - m_new) : 0.0f;
    float ps = p0 + p1;
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * alpha + ps;
    m_row = m_new;
    KQ_UNROLL
    for (int t = 0; t < NT; ++t) {
      Ot[t].thread_elements()[0] *= alpha;
      Ot[t].thread_elements()[1] *= alpha;
    }
    MatT Pt;
    Pt.thread_elements()[0] = T(p0);
    Pt.thread_elements()[1] = T(p1);
    KQ_UNROLL
    for (int t = 0; t < NT; ++t) {
      MatT Kt;
      simdgroup_load(Kt, Ks + g * DH + t * 8, LD);
      simdgroup_multiply_accumulate(Ot[t], Pt, Kt, Ot[t]);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }

  const size_t bls = (size_t(b) * p->L + l) * p->n_splits + s;
  device float* ob = Oacc + (bls * p->Hp + h0) * D + g * DH;
  KQ_UNROLL
  for (int t = 0; t < NT; ++t) {
    simdgroup_store(Ot[t], ob + t * 8, D);
  }
  if (g == 0 && fn == 0) {
    Ms[bls * p->Hp + h0 + fm] = m_row;
    Ls[bls * p->Hp + h0 + fm] = l_row;
  }
}

// One threadgroup of D / 4 threads per (head, query, batch): every thread
// owns one float4 of the row and renormalizes the n_splits partials with the
// running-max identity; the sink (if any) is one more logit.
template <typename T, int D>
[[kernel]] void kq_sdpa_sparse_decode_merge(
    const device float* Oacc [[buffer(0)]],
    const device float* Ms [[buffer(1)]],
    const device float* Ls [[buffer(2)]],
    const device T* Sinks [[buffer(3)]],
    device T* O [[buffer(4)]],
    const constant KQSdpaSparseDecodeParams* p [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
  using V4 = metal::vec<T, 4>;
  const int h = int(tid.x);
  const int l = int(tid.y);
  const int b = int(tid.z);
  const int S = p->n_splits;
  const int Hp = p->Hp;
  const size_t ml_base = (size_t(b) * p->L + l) * size_t(S) * Hp + h;

  threadgroup float w_sh[32];
  if (t < 32) {
    // Lane t owns split t; the reductions ride simd shuffles.
    const bool live = int(t) < S;
    const float ms = live ? Ms[ml_base + size_t(t) * Hp] : kSdpaSparseNegBig;
    float m_all = simd_max(ms);
    float l_all = 0.0f;
    if (p->has_sinks) {
      const float sink_m = M_LOG2E_F * float(Sinks[h]);
      m_all = metal::max(m_all, sink_m);
      l_all = fast::exp2(sink_m - m_all);
    }
    const float w = live ? fast::exp2(ms - m_all) : 0.0f;
    const float lw = live ? Ls[ml_base + size_t(t) * Hp] * w : 0.0f;
    l_all += simd_sum(lw);
    w_sh[t] = w / l_all;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  const device float4* acc4 =
      reinterpret_cast<const device float4*>(Oacc + ml_base * D) + t;
  const size_t split_step = size_t(Hp) * D / 4;
  float4 acc = 0.0f;
  for (int s = 0; s < S; ++s) {
    acc += acc4[size_t(s) * split_step] * w_sh[s];
  }
  device T* out = O + size_t(b) * p->o_strides[0] +
      size_t(h) * p->o_strides[1] + size_t(l) * p->o_strides[2];
  reinterpret_cast<device V4*>(out)[t] = V4(acc);
}
