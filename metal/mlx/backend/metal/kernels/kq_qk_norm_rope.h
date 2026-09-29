// Fused per-head RMSNorm and RoPE of the DiT attention q and k, both in one
// dispatch:
//
//   y = x * rsqrt(mean(x^2) + eps) * w
//   out[2i]     = y[2i] * cos[l, i] - y[2i + 1] * sin[l, i]
//   out[2i + 1] = y[2i] * sin[l, i] + y[2i + 1] * cos[l, i]
//
// q and k are [B, L, H, D], read in place through their b, l, h and d
// strides (a slice of a fused qkv output needs no copy). q_out and k_out
// are [B, H, L, D] row-contiguous, the layout attention reads. cos and sin
// are float [L, D / 2] tables with one angle per row and pair, so any
// position scheme works (multi-axis image positions included); the norm
// weights are float [D]. The batch shares the tables.
//
// One threadgroup per (b, l) row. A group of KQ_QKNR_LPR lanes covers one
// head row of D, EPL = D / KQ_QKNR_LPR elements per lane, so a simdgroup
// covers 32 / KQ_QKNR_LPR head rows per pass and the simdgroups walk the
// 2 * H head rows of q, then k. A lane loads its cos, sin and weight values
// once per row and reuses them across heads. The mean square is a
// shuffle-xor sum inside the lane group. All math in f32, one round at the
// write.
//
// kq_qknr_vec (function constant 350) selects vector loads, which the host
// sets when the d stride is 1 and every row start is aligned to the vector
// width; otherwise the kernel reads element by element through the strides.

#pragma once

#define KQ_QKNR_LPR 16
#define KQ_QKNR_UNROLL 2

constant bool kq_qknr_vec [[function_constant(350)]];

// Mirror of the host-side KQQkNormRopeParams (kquant_qk_norm_rope.cpp); the
// layouts must match field for field. Strides are in elements.
struct KQQkNormRopeParams {
  int64_t q_b;
  int64_t q_l;
  int64_t q_h;
  int64_t q_d;
  int64_t k_b;
  int64_t k_l;
  int64_t k_h;
  int64_t k_d;
  int L;
  int H;
  float eps;
};

template <typename T, int EPL, int VW>
METAL_FUNC void kq_qknr_load(const device T* src, int64_t sd, thread float* x) {
  if (kq_qknr_vec) {
    using V = metal::vec<T, VW>;
    for (int i = 0; i < EPL; i += VW) {
      const V v = *reinterpret_cast<const device V*>(src + i);
      for (int t = 0; t < VW; ++t) {
        x[i + t] = float(v[t]);
      }
    }
  } else {
    for (int i = 0; i < EPL; ++i) {
      x[i] = float(src[i * sd]);
    }
  }
}

template <typename T, int D>
[[kernel]] void kq_qk_norm_rope(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device float* qw [[buffer(2)]],
    const device float* kw [[buffer(3)]],
    const device float* cosv [[buffer(4)]],
    const device float* sinv [[buffer(5)]],
    device T* q_out [[buffer(6)]],
    device T* k_out [[buffer(7)]],
    const constant KQQkNormRopeParams& p [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int LPR = KQ_QKNR_LPR;
  constexpr int EPL = D / LPR;
  constexpr int VW = (EPL % 4 == 0) ? 4 : 2;
  constexpr int RPS = 32 / LPR;
  constexpr int U = KQ_QKNR_UNROLL;
  static_assert(D % (2 * LPR) == 0 && EPL % VW == 0, "unsupported head dim");

  const int L = p.L;
  const int H = p.H;
  const int b = int(row) / L;
  const int l = int(row) - b * L;
  const int c0 = (int(lane) % LPR) * EPL;

  float cs[EPL / 2], sn[EPL / 2], wq[EPL], wk[EPL];
  const device float* cr = cosv + int64_t(l) * (D / 2) + c0 / 2;
  const device float* sr = sinv + int64_t(l) * (D / 2) + c0 / 2;
  for (int i = 0; i < EPL / 2; ++i) {
    cs[i] = cr[i];
    sn[i] = sr[i];
  }
  for (int i = 0; i < EPL; ++i) {
    wq[i] = qw[c0 + i];
    wk[i] = kw[c0 + i];
  }

  const device T* qr = q + b * p.q_b + l * p.q_l + c0 * p.q_d;
  const device T* kr = k + b * p.k_b + l * p.k_l + c0 * p.k_d;
  // Output row (b, 0, l) of [B, H, L, D]; head h adds h * L * D.
  const int64_t obase = (int64_t(b) * H * L + l) * D + c0;
  const int64_t ohead = int64_t(L) * D;
  const int step = int(nsg) * RPS;

  for (int j0 = int(sg) * RPS + int(lane) / LPR; j0 < 2 * H; j0 += step * U) {
    float x[U][EPL];
    bool act[U], isk[U];
    int hh[U];
    for (int u = 0; u < U; ++u) {
      const int j = j0 + u * step;
      act[u] = j < 2 * H;
      isk[u] = j >= H;
      hh[u] = isk[u] ? j - H : j;
      if (act[u]) {
        const device T* src = isk[u] ? kr + hh[u] * p.k_h : qr + hh[u] * p.q_h;
        kq_qknr_load<T, EPL, VW>(src, isk[u] ? p.k_d : p.q_d, x[u]);
      } else {
        for (int i = 0; i < EPL; ++i) {
          x[u][i] = 0.0f;
        }
      }
    }
    for (int u = 0; u < U; ++u) {
      float ss = 0.0f;
      for (int i = 0; i < EPL; ++i) {
        ss = metal::fma(x[u][i], x[u][i], ss);
      }
      for (int o = LPR / 2; o >= 1; o >>= 1) {
        ss += simd_shuffle_xor(ss, ushort(o));
      }
      if (!act[u]) {
        continue;
      }
      const float inv = metal::precise::rsqrt(ss / float(D) + p.eps);
      float y[EPL];
      for (int i = 0; i < EPL; ++i) {
        y[i] = x[u][i] * inv * (isk[u] ? wk[i] : wq[i]);
      }
      device T* dst = (isk[u] ? k_out : q_out) + obase + hh[u] * ohead;
      using V = metal::vec<T, VW>;
      for (int i = 0; i < EPL; i += VW) {
        V v;
        for (int t = 0; t < VW; t += 2) {
          const int e = i + t;
          const float a = y[e];
          const float c = y[e + 1];
          v[t] = T(a * cs[e / 2] - c * sn[e / 2]);
          v[t + 1] = T(a * sn[e / 2] + c * cs[e / 2]);
        }
        *reinterpret_cast<device V*>(dst + i) = v;
      }
    }
  }
}
