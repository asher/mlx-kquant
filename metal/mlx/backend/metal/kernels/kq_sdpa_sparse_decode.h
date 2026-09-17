// Sparse decode attention over two key sources: a window [W, D] read whole
// and a pool [P, D] read through a per-query index list [N]. K == V (the
// absorbed-MLA latent).
//
// Split kernel: one threadgroup per (HG-head group, query, key split), two
// simdgroups per 8 heads (each owns one half of D). Blocks of 8 key rows are
// staged through one threadgroup buffer shared by every simdgroup, scored
// against the 8 queries of each simdgroup with simdgroup matrix ops (partial
// scores summed across the two halves), run through an exp2-space online
// softmax held per row in the fragment layout, and accumulated into fp32
// output tiles. The next block's rows are fetched into registers while the
// current block computes. Masked or out-of-range rows score kNegBig and
// weigh zero, which matches an additive -inf mask without inf arithmetic.
// A lone split (p->direct) normalizes in place, sinks included, and writes
// the output rows itself; otherwise the merge kernel renormalizes the split
// partials with the head's sink counted once. PK reads the pool in the
// latent_fp4_pack form (Pool holds the code bytes, PoolScales the scale
// bytes) and dequantizes each row chunk as it stages.
#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#include "mlx/backend/metal/kernels/kq_latent_fp4.h"
#include "mlx/backend/metal/kernels/kq_sdpa_sparse_decode_params.h"

constant constexpr float kSdpaSparseNegBig = -1e30f;

#define KQ_UNROLL _Pragma("clang loop unroll(full)")

template <typename T, typename IdxT, int D, int HG, bool PK>
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
    const device T* Sinks [[buffer(10)]],
    device T* O [[buffer(11)]],
    const device uchar* PoolScales [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int KB = 8; // keys per block
  constexpr int KT = KB / 8; // 8-key tiles per block
  constexpr int NSG = HG / 4; // simdgroups: two per 8-head subgroup
  constexpr int NTH = NSG * 32;
  constexpr int LD = D + 8; // padded row (bank spread, 16-byte aligned)
  constexpr int DH = D / 2; // dims per simdgroup
  constexpr int NT = DH / 8; // 8x8 tiles per simdgroup along D
  constexpr int CHR = D / 8; // 16-byte chunks per row
  constexpr int SV = (KB * CHR + NTH - 1) / NTH; // staged chunks per thread
  static_assert(HG % 8 == 0, "head groups are 8-head tiles");
  using MatT = metal::simdgroup_matrix<T, 8, 8>;
  using MatF = metal::simdgroup_matrix<float, 8, 8>;

  threadgroup T Ks[KB * LD];
  threadgroup float Sx[NSG][32][2 * KT];

  const int s = int(tid.x);
  const int h0 = int(tid.y) * HG;
  const int b = int(tid.z) / p->L;
  const int l = int(tid.z) % p->L;
  const int g = int(simd_gid);
  const int hs = g >> 1; // 8-head subgroup of this simdgroup
  const int dh = g & 1; // D half of this simdgroup
  const int H = p->H;
  const int tix = g * 32 + int(lane);

  // Queries, 8 heads per round through the key buffer (rows past H repeat
  // the last head; their partials are never read).
  MatT Qt[NT];
  {
    const device T* qb =
        Q + size_t(b) * p->q_strides[0] + size_t(l) * p->q_strides[2];
    for (int sg = 0; sg < HG / 8; ++sg) {
      for (int v = tix; v < 8 * CHR; v += NTH) {
        const int r = v / CHR;
        const int c = v % CHR;
        const int h = metal::min(h0 + sg * 8 + r, H - 1);
        const uint4 x = *reinterpret_cast<const device uint4*>(
            qb + size_t(h) * p->q_strides[1] + c * 8);
        *reinterpret_cast<threadgroup uint4*>(Ks + r * LD + c * 8) = x;
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      if (hs == sg) {
        KQ_UNROLL
        for (int t = 0; t < NT; ++t) {
          simdgroup_load(Qt[t], Ks + dh * DH + t * 8, LD);
        }
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
  }
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
  const device uchar* pool_c = reinterpret_cast<const device uchar*>(Pool) +
      size_t(b) * (p->pool_strides[0] / 2);
  const device uchar* pool_s =
      PoolScales + size_t(b) * (p->pool_strides[0] / 16);
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
  const bool safe_pk = PK && W == 0;
  const device T* safe_row = W > 0 ? win_b
      : PK                         ? reinterpret_cast<const device T*>(pool_c)
                                   : pool_b;
  const float sl2 = p->scale_log2;

  // Rows of block j0 (uniform across the threadgroup) into registers: a
  // 16-byte chunk of T, or (PK) its code word in .x and scale byte in .y.
  uint4 pre[SV];
  bool pkf[SV];
  int okbits = 0;
#define KQ_SPARSE_FETCH(j0)                                            \
  {                                                                    \
    const device T* rows[KB];                                          \
    const device uchar* srows[KB];                                     \
    int pkbits = 0;                                                    \
    okbits = 0;                                                        \
    KQ_UNROLL                                                          \
    for (int k = 0; k < KB; ++k) {                                     \
      const int j = (j0) + k;                                          \
      bool v = j < k1;                                                 \
      const device T* row = safe_row;                                  \
      const device uchar* srow = pool_s;                               \
      bool pk = safe_pk;                                               \
      if (v) {                                                         \
        if (j < W) {                                                   \
          v = (wm == nullptr) || wm[j];                                \
          row = win_b + size_t(j) * p->win_strides[1];                 \
          pk = false;                                                  \
        } else {                                                       \
          const int n = j - W;                                         \
          v = (sm == nullptr) || sm[n];                                \
          if (v) {                                                     \
            const int64_t r = int64_t(idx_bl[n]);                      \
            v = r >= 0 && r < int64_t(p->P);                           \
            if (v) {                                                   \
              if (PK) {                                                \
                row = reinterpret_cast<const device T*>(               \
                    pool_c + size_t(r) * (D / 2));                     \
                srow = pool_s + size_t(r) * (D / 16);                  \
                pk = true;                                             \
              } else {                                                 \
                row = pool_b + size_t(r) * p->pool_strides[1];         \
              }                                                        \
            }                                                          \
          }                                                            \
        }                                                              \
      }                                                                \
      okbits |= int(v) << k;                                           \
      pkbits |= int(pk) << k;                                          \
      rows[k] = row;                                                   \
      srows[k] = srow;                                                 \
    }                                                                  \
    KQ_UNROLL                                                          \
    for (int i = 0; i < SV; ++i) {                                     \
      const int v = tix + i * NTH;                                     \
      if (v < KB * CHR) {                                              \
        const int k = v / CHR;                                         \
        const int c = v % CHR;                                         \
        pkf[i] = PK && ((pkbits >> k) & 1);                            \
        if (pkf[i]) {                                                  \
          pre[i].x = reinterpret_cast<const device uint*>(rows[k])[c]; \
          pre[i].y = srows[k][c / 2];                                  \
        } else {                                                       \
          pre[i] = reinterpret_cast<const device uint4*>(rows[k])[c];  \
        }                                                              \
      }                                                                \
    }                                                                  \
  }

  if (k0 < k1) {
    KQ_SPARSE_FETCH(k0);
  }
  for (int j0 = k0; j0 < k1; j0 += KB) {
    KQ_UNROLL
    for (int i = 0; i < SV; ++i) {
      const int v = tix + i * NTH;
      if (v < KB * CHR) {
        const uint4 x = pkf[i] ? as_type<uint4>(kq_fp4_chunk8<T>(
                                     pre[i].x, kq_fp4_e4m3_decode(pre[i].y)))
                               : pre[i];
        *reinterpret_cast<threadgroup uint4*>(
            Ks + (v / CHR) * LD + (v % CHR) * 8) = x;
      }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const int cur_ok = okbits;
    if (j0 + KB < k1) {
      KQ_SPARSE_FETCH(j0 + KB);
    }

    // Partial scores over this half of D, then the other half's.
    MatF St[KT];
    KQ_UNROLL
    for (int kt = 0; kt < KT; ++kt) {
      St[kt] = MatF(0.0f);
      KQ_UNROLL
      for (int t = 0; t < NT; ++t) {
        MatT Kt;
        simdgroup_load(
            Kt, Ks + kt * 8 * LD + dh * DH + t * 8, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(St[kt], Qt[t], Kt, St[kt]);
      }
      Sx[g][lane][2 * kt] = St[kt].thread_elements()[0];
      Sx[g][lane][2 * kt + 1] = St[kt].thread_elements()[1];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float sc[2 * KT];
    bool ok[2 * KT];
    float mb = kSdpaSparseNegBig;
    KQ_UNROLL
    for (int kt = 0; kt < KT; ++kt) {
      ok[2 * kt] = (cur_ok >> (kt * 8 + fn)) & 1;
      ok[2 * kt + 1] = (cur_ok >> (kt * 8 + fn + 1)) & 1;
      sc[2 * kt] = ok[2 * kt]
          ? (St[kt].thread_elements()[0] + Sx[g ^ 1][lane][2 * kt]) * sl2
          : kSdpaSparseNegBig;
      sc[2 * kt + 1] = ok[2 * kt + 1]
          ? (St[kt].thread_elements()[1] + Sx[g ^ 1][lane][2 * kt + 1]) * sl2
          : kSdpaSparseNegBig;
      mb = metal::max(mb, metal::max(sc[2 * kt], sc[2 * kt + 1]));
    }

    // Online softmax per row (the 4 lanes of a row hold the same state).
    mb = metal::max(mb, simd_shuffle_xor(mb, 1));
    mb = metal::max(mb, simd_shuffle_xor(mb, 8));
    const float m_new = metal::max(m_row, mb);
    const float alpha = fast::exp2(m_row - m_new);
    float ps = 0.0f;
    MatT Pt[KT];
    KQ_UNROLL
    for (int kt = 0; kt < KT; ++kt) {
      const float p0 = ok[2 * kt] ? fast::exp2(sc[2 * kt] - m_new) : 0.0f;
      const float p1 =
          ok[2 * kt + 1] ? fast::exp2(sc[2 * kt + 1] - m_new) : 0.0f;
      ps += p0 + p1;
      Pt[kt].thread_elements()[0] = T(p0);
      Pt[kt].thread_elements()[1] = T(p1);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * alpha + ps;
    m_row = m_new;
    KQ_UNROLL
    for (int t = 0; t < NT; ++t) {
      Ot[t].thread_elements()[0] *= alpha;
      Ot[t].thread_elements()[1] *= alpha;
    }
    KQ_UNROLL
    for (int kt = 0; kt < KT; ++kt) {
      KQ_UNROLL
      for (int t = 0; t < NT; ++t) {
        MatT Kt;
        simdgroup_load(Kt, Ks + kt * 8 * LD + dh * DH + t * 8, LD);
        simdgroup_multiply_accumulate(Ot[t], Pt[kt], Kt, Ot[t]);
      }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
#undef KQ_SPARSE_FETCH

  const int hq = h0 + hs * 8; // first head of this simdgroup
  if (p->direct) {
    // The only split: normalize here, the sink as one more logit.
    const int h = hq + fm;
    float m_all = m_row;
    float l_all = l_row;
    if (p->has_sinks) {
      const float sink_m = M_LOG2E_F * float(Sinks[metal::min(h, H - 1)]);
      m_all = metal::max(m_row, sink_m);
      l_all = l_row * fast::exp2(m_row - m_all) + fast::exp2(sink_m - m_all);
    }
    const float wgt = fast::exp2(m_row - m_all) / l_all;
    if (h < H) {
      device T* out = O + size_t(b) * p->o_strides[0] +
          size_t(h) * p->o_strides[1] + size_t(l) * p->o_strides[2] + dh * DH;
      KQ_UNROLL
      for (int t = 0; t < NT; ++t) {
        metal::vec<T, 2> o2(
            T(Ot[t].thread_elements()[0] * wgt),
            T(Ot[t].thread_elements()[1] * wgt));
        *reinterpret_cast<device metal::vec<T, 2>*>(out + t * 8 + fn) = o2;
      }
    }
    return;
  }

  const size_t bls = (size_t(b) * p->L + l) * p->n_splits + s;
  device float* ob = Oacc + (bls * p->Hp + hq) * D + dh * DH;
  KQ_UNROLL
  for (int t = 0; t < NT; ++t) {
    simdgroup_store(Ot[t], ob + t * 8, D);
  }
  if (dh == 0 && fn == 0) {
    Ms[bls * p->Hp + hq + fm] = m_row;
    Ls[bls * p->Hp + hq + fm] = l_row;
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

constant constexpr int kSdpaSparsePrefillPass = 1024; // row-table entries
constant constexpr int kSdpaSparseDead = -2147483647 - 1;

// Prefill form: one threadgroup per (HG-head group, query), no key split.
// The keys are the query's own band of the window array, [pos - band + 1,
// pos] with pos = koff + l, then its listed pool rows. Each 8-head subgroup
// spreads over DS simdgroups (one D slice each, partial scores summed
// through Sx) and KB rows go through a staged buffer per block. Row
// addresses resolve once per pass into a threadgroup table (window rows as
// ~row, pool rows as row, dead slots as kSdpaSparseDead), so a block fetch
// reads the table instead of chasing the index and mask loads.
template <typename T, typename IdxT, int D, int HG, int DS, int KB, bool PK>
[[kernel]] void kq_sdpa_sparse_prefill(
    const device T* Q [[buffer(0)]],
    const device T* Win [[buffer(1)]],
    const device T* Pool [[buffer(2)]],
    const device IdxT* Idx [[buffer(3)]],
    const device bool* SelMask [[buffer(4)]],
    const device T* Sinks [[buffer(5)]],
    device T* O [[buffer(6)]],
    const constant KQSdpaSparsePrefillParams* p [[buffer(7)]],
    const device uchar* PoolScales [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int KT = KB / 8; // 8-key tiles per block
  constexpr int NSG = (HG / 8) * DS; // simdgroups
  constexpr int NTH = NSG * 32;
  constexpr int LD = D + 8; // padded row (bank spread, 16-byte aligned)
  constexpr int DH = D / DS; // dims per simdgroup
  constexpr int NT = DH / 8; // 8x8 tiles per simdgroup along D
  constexpr int CHR = D / 8; // 16-byte chunks per row
  constexpr int SV = (KB * CHR + NTH - 1) / NTH; // staged chunks per thread
  constexpr int MAXK = kSdpaSparsePrefillPass;
  constexpr int DEAD = kSdpaSparseDead;
  static_assert(HG % 8 == 0, "head groups are 8-head tiles");
  static_assert(DH % 8 == 0, "D slices are 8-wide tiles");
  static_assert(KB % 8 == 0, "key blocks are 8-key tiles");
  using MatT = metal::simdgroup_matrix<T, 8, 8>;
  using MatF = metal::simdgroup_matrix<float, 8, 8>;

  threadgroup T Ks[KB * LD];
  threadgroup float Sx[NSG][32][2 * KT];
  threadgroup int tab[MAXK];

  const int h0 = int(tid.y) * HG;
  const int b = int(tid.z) / p->L;
  const int l = int(tid.z) % p->L;
  const int g = int(simd_gid);
  const int hs = g / DS; // 8-head subgroup of this simdgroup
  const int dh = g % DS; // D slice of this simdgroup
  const int H = p->H;
  const int tix = g * 32 + int(lane);

  // Queries, 8 heads per round through the key buffer (rows past H repeat
  // the last head; their output is never written).
  MatT Qt[NT];
  {
    const device T* qb =
        Q + size_t(b) * p->q_strides[0] + size_t(l) * p->q_strides[2];
    for (int sg = 0; sg < HG / 8; ++sg) {
      for (int v = tix; v < 8 * CHR; v += NTH) {
        const int r = v / CHR;
        const int c = v % CHR;
        const int h = metal::min(h0 + sg * 8 + r, H - 1);
        const uint4 x = *reinterpret_cast<const device uint4*>(
            qb + size_t(h) * p->q_strides[1] + c * 8);
        *reinterpret_cast<threadgroup uint4*>(Ks + r * LD + c * 8) = x;
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      if (hs == sg) {
        KQ_UNROLL
        for (int t = 0; t < NT; ++t) {
          simdgroup_load(Qt[t], Ks + dh * DH + t * 8, LD);
        }
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
  }
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

  const int pos = p->koff + l;
  const int w0 = metal::max(0, pos - p->band + 1);
  const int Wl = pos + 1 - w0;
  const int total = Wl + p->N;
  const device T* win_b = Win + size_t(b) * p->win_strides[0];
  const device T* pool_b = Pool + size_t(b) * p->pool_strides[0];
  const device uchar* pool_c = reinterpret_cast<const device uchar*>(Pool) +
      size_t(b) * (p->pool_strides[0] / 2);
  const device uchar* pool_s =
      PoolScales + size_t(b) * (p->pool_strides[0] / 16);
  const device IdxT* idx_bl =
      Idx + size_t(b) * p->idx_strides[0] + size_t(l) * p->idx_strides[1];
  const device bool* sm = p->has_sel_mask
      ? SelMask + size_t(b) * p->sel_mask_strides[0] +
          size_t(l) * p->sel_mask_strides[1]
      : nullptr;
  const float sl2 = p->scale_log2;

  // Rows of block j0 into registers through the table; dead slots stage
  // window row 0 so every load stays unconditional. A packed pool row
  // lands as its code word in .x and scale byte in .y.
  uint4 pre[SV];
  bool pkf[SV];
#define KQ_SPARSE_PF_FETCH(j0)                                    \
  {                                                               \
    KQ_UNROLL                                                     \
    for (int i = 0; i < SV; ++i) {                                \
      const int v = tix + i * NTH;                                \
      if (v < KB * CHR) {                                         \
        const int j = (j0) + v / CHR;                             \
        const int c = v % CHR;                                    \
        const int e = j < pe ? tab[j - pb] : DEAD;                \
        pkf[i] = PK && e >= 0;                                    \
        if (pkf[i]) {                                             \
          pre[i].x = reinterpret_cast<const device uint*>(        \
              pool_c + size_t(e) * (D / 2))[c];                   \
          pre[i].y = pool_s[size_t(e) * (D / 16) + c / 2];        \
        } else {                                                  \
          const device T* row = win_b;                            \
          if (e >= 0) {                                           \
            row = pool_b + size_t(e) * p->pool_strides[1];        \
          } else if (e != DEAD) {                                 \
            row = win_b + size_t(~e) * p->win_strides[1];         \
          }                                                       \
          pre[i] = reinterpret_cast<const device uint4*>(row)[c]; \
        }                                                         \
      }                                                           \
    }                                                             \
  }

  for (int pb = 0; pb < total; pb += MAXK) {
    const int pe = metal::min(total, pb + MAXK);
    for (int k = tix; k < pe - pb; k += NTH) {
      const int j = pb + k;
      int e = DEAD;
      if (j < Wl) {
        e = ~(w0 + j);
      } else {
        const int n = j - Wl;
        if (sm == nullptr || sm[n]) {
          const int64_t r = int64_t(idx_bl[n]);
          if (r >= 0 && r < int64_t(p->P)) {
            e = int(r);
          }
        }
      }
      tab[k] = e;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    KQ_SPARSE_PF_FETCH(pb);
    for (int j0 = pb; j0 < pe; j0 += KB) {
      KQ_UNROLL
      for (int i = 0; i < SV; ++i) {
        const int v = tix + i * NTH;
        if (v < KB * CHR) {
          const uint4 x = pkf[i] ? as_type<uint4>(kq_fp4_chunk8<T>(
                                       pre[i].x, kq_fp4_e4m3_decode(pre[i].y)))
                                 : pre[i];
          *reinterpret_cast<threadgroup uint4*>(
              Ks + (v / CHR) * LD + (v % CHR) * 8) = x;
        }
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      if (j0 + KB < pe) {
        KQ_SPARSE_PF_FETCH(j0 + KB);
      }

      // Partial scores over this slice of D, then the other slices'.
      MatF St[KT];
      KQ_UNROLL
      for (int kt = 0; kt < KT; ++kt) {
        St[kt] = MatF(0.0f);
        KQ_UNROLL
        for (int t = 0; t < NT; ++t) {
          MatT Kt;
          simdgroup_load(
              Kt, Ks + kt * 8 * LD + dh * DH + t * 8, LD, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(St[kt], Qt[t], Kt, St[kt]);
        }
        Sx[g][lane][2 * kt] = St[kt].thread_elements()[0];
        Sx[g][lane][2 * kt + 1] = St[kt].thread_elements()[1];
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      float sc[2 * KT];
      bool ok[2 * KT];
      float mb = kSdpaSparseNegBig;
      KQ_UNROLL
      for (int kt = 0; kt < KT; ++kt) {
        KQ_UNROLL
        for (int c = 0; c < 2; ++c) {
          const int j = j0 + kt * 8 + fn + c;
          const bool v = j < pe && tab[j - pb] != DEAD;
          float s = St[kt].thread_elements()[c];
          KQ_UNROLL
          for (int ds = 0; ds < DS; ++ds) {
            if (ds != dh) {
              s += Sx[hs * DS + ds][lane][2 * kt + c];
            }
          }
          ok[2 * kt + c] = v;
          sc[2 * kt + c] = v ? s * sl2 : kSdpaSparseNegBig;
          mb = metal::max(mb, sc[2 * kt + c]);
        }
      }

      // Online softmax per row (the 4 lanes of a row hold the same state).
      mb = metal::max(mb, simd_shuffle_xor(mb, 1));
      mb = metal::max(mb, simd_shuffle_xor(mb, 8));
      const float m_new = metal::max(m_row, mb);
      const float alpha = fast::exp2(m_row - m_new);
      float ps = 0.0f;
      MatT Pt[KT];
      KQ_UNROLL
      for (int kt = 0; kt < KT; ++kt) {
        const float p0 = ok[2 * kt] ? fast::exp2(sc[2 * kt] - m_new) : 0.0f;
        const float p1 =
            ok[2 * kt + 1] ? fast::exp2(sc[2 * kt + 1] - m_new) : 0.0f;
        ps += p0 + p1;
        Pt[kt].thread_elements()[0] = T(p0);
        Pt[kt].thread_elements()[1] = T(p1);
      }
      ps += simd_shuffle_xor(ps, 1);
      ps += simd_shuffle_xor(ps, 8);
      l_row = l_row * alpha + ps;
      m_row = m_new;
      KQ_UNROLL
      for (int t = 0; t < NT; ++t) {
        Ot[t].thread_elements()[0] *= alpha;
        Ot[t].thread_elements()[1] *= alpha;
      }
      KQ_UNROLL
      for (int kt = 0; kt < KT; ++kt) {
        KQ_UNROLL
        for (int t = 0; t < NT; ++t) {
          MatT Kt;
          simdgroup_load(Kt, Ks + kt * 8 * LD + dh * DH + t * 8, LD);
          simdgroup_multiply_accumulate(Ot[t], Pt[kt], Kt, Ot[t]);
        }
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
  }
#undef KQ_SPARSE_PF_FETCH

  // Normalize here, the sink as one more logit.
  const int h = h0 + hs * 8 + fm;
  float m_all = m_row;
  float l_all = l_row;
  if (p->has_sinks) {
    const float sink_m = M_LOG2E_F * float(Sinks[metal::min(h, H - 1)]);
    m_all = metal::max(m_row, sink_m);
    l_all = l_row * fast::exp2(m_row - m_all) + fast::exp2(sink_m - m_all);
  }
  const float wgt = fast::exp2(m_row - m_all) / l_all;
  if (h < H) {
    device T* out = O + size_t(b) * p->o_strides[0] +
        size_t(h) * p->o_strides[1] + size_t(l) * p->o_strides[2] + dh * DH;
    KQ_UNROLL
    for (int t = 0; t < NT; ++t) {
      metal::vec<T, 2> o2(
          T(Ot[t].thread_elements()[0] * wgt),
          T(Ot[t].thread_elements()[1] * wgt));
      *reinterpret_cast<device metal::vec<T, 2>*>(out + t * 8 + fn) = o2;
    }
  }
}
