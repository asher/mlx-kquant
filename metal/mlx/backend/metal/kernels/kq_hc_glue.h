// Fused hyper-connection glue kernels for the deepseek4 M=1 decode route.
// Four streams (hc_mult 4) baked in; D is the per-stream hidden size.
// Numerics mirror the certified gmlx JIT kernels: f32 accumulate,
// fast::exp sinkhorn, round-before-use in the fused expand, single
// rounding at each T-dtype write. The f32 reductions run in the order
// written here (one thread per float4 column across the four streams,
// then simd_sum, then the simdgroup partials in order), and the split and
// fused routes share that order by construction.
//
//   kq_hc_front_reduce:       mixes_raw[m] = dot(x, fn[m]), plus the row
//                             sum of squares (deferred rms factor). One
//                             threadgroup per (row, m) with m == MIX for
//                             the sumsq lane.
//   kq_hc_front_expand_reduce: the previous cycle's expand recomputed
//                             ahead of the same reduction; the sumsq
//                             threadgroup also writes the expanded h.
//   kq_hc_sinkhorn_collapse:  sinkhorn mix normalization plus collapse to
//                             one stream with the sublayer RMSNorm folded
//                             into the output. One threadgroup per row;
//                             the last simdgroup runs the sinkhorn while
//                             the others collapse.
//   kq_hc_expand:             pre/comb expand of the sublayer output back
//                             to four streams. Two threadgroups per row.
//   kq_hc_front_expand_collapse: the front and the collapse as one
//                             dispatch: every threadgroup publishes its
//                             dot and arrives at a per-row device counter,
//                             and the last one to arrive runs the
//                             collapse. No threadgroup waits.
//
// Every kernel runs KQ_HC_NT threads per threadgroup (the dispatch side
// uses the same count): at hidden sizes up to 4096 each thread owns one
// float4 column of every stream, so a threadgroup issues all of its loads
// at once instead of walking the row in strides, which is what the
// dependent decode chain pays for. The bodies live in shared inline
// functions so the split and fused routes cannot drift apart.

#define KQ_HC 4
#define KQ_HC_MIX ((2 + KQ_HC) * KQ_HC)
#define KQ_HC_NT 1024
#define KQ_HC_NSG (KQ_HC_NT / 32)
// The collapse: KQ_HC_NT - 32 worker threads own the float4 columns and
// the last simdgroup runs the sinkhorn.
#define KQ_HC_NC (KQ_HC_NT - 32)
#define KQ_HC_MAX_CHUNKS ((8192 / 4 + KQ_HC_NC - 1) / KQ_HC_NC)

// One f32 per thread to one value, valid on lane 0 of simdgroup 0 after
// the call. partial is threadgroup scratch [NSG]; simdgroups at or past
// NSG contribute nothing. Every thread of the threadgroup must call this
// (it barriers).
template <int NSG>
inline float
kq_hc_tg_sum(float acc, uint lane, uint sg, threadgroup float* partial) {
  acc = simd_sum(acc);
  if (lane == 0 && sg < (uint)NSG) {
    partial[sg] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float v = 0.0f;
  if (sg == 0) {
    v = (lane < (uint)NSG) ? partial[lane] : 0.0f;
    v = simd_sum(v);
  }
  return v;
}

#define KQ_HC_DOT16(acc, a0, a1, a2, a3, b0, b1, b2, b3) \
  acc = fma(a0.x, b0.x, acc);                            \
  acc = fma(a0.y, b0.y, acc);                            \
  acc = fma(a0.z, b0.z, acc);                            \
  acc = fma(a0.w, b0.w, acc);                            \
  acc = fma(a1.x, b1.x, acc);                            \
  acc = fma(a1.y, b1.y, acc);                            \
  acc = fma(a1.z, b1.z, acc);                            \
  acc = fma(a1.w, b1.w, acc);                            \
  acc = fma(a2.x, b2.x, acc);                            \
  acc = fma(a2.y, b2.y, acc);                            \
  acc = fma(a2.z, b2.z, acc);                            \
  acc = fma(a2.w, b2.w, acc);                            \
  acc = fma(a3.x, b3.x, acc);                            \
  acc = fma(a3.y, b3.y, acc);                            \
  acc = fma(a3.z, b3.z, acc);                            \
  acc = fma(a3.w, b3.w, acc);

template <typename T>
[[kernel]] void kq_hc_front_reduce(
    const device T* x [[buffer(0)]],
    const device float* fn [[buffer(1)]],
    device float* mixes_raw [[buffer(2)]],
    device float* sumsq [[buffer(3)]],
    const constant int& D [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  const uint lane = tid % 32;
  const uint sg = tid / 32;
  const int KTOT = KQ_HC * D;
  const uint D4q = (uint)D / 4;

  const uint row = tg / (KQ_HC_MIX + 1);
  const uint m = tg % (KQ_HC_MIX + 1);

  using T4 = vec<T, 4>;
  const device T4* x4 = (const device T4*)(x + (int64_t)row * KTOT);

  float acc = 0.0f;
  if (m < (uint)KQ_HC_MIX) {
    const device float4* f4 = (const device float4*)(fn + (int64_t)m * KTOT);
    for (uint d4 = tid; d4 < D4q; d4 += KQ_HC_NT) {
      float4 x0 = float4(x4[0 * D4q + d4]);
      float4 x1 = float4(x4[1 * D4q + d4]);
      float4 x2 = float4(x4[2 * D4q + d4]);
      float4 x3 = float4(x4[3 * D4q + d4]);
      float4 f0 = f4[0 * D4q + d4];
      float4 f1 = f4[1 * D4q + d4];
      float4 f2 = f4[2 * D4q + d4];
      float4 f3 = f4[3 * D4q + d4];
      KQ_HC_DOT16(acc, x0, x1, x2, x3, f0, f1, f2, f3)
    }
  } else {
    for (uint d4 = tid; d4 < D4q; d4 += KQ_HC_NT) {
      float4 x0 = float4(x4[0 * D4q + d4]);
      float4 x1 = float4(x4[1 * D4q + d4]);
      float4 x2 = float4(x4[2 * D4q + d4]);
      float4 x3 = float4(x4[3 * D4q + d4]);
      KQ_HC_DOT16(acc, x0, x1, x2, x3, x0, x1, x2, x3)
    }
  }

  threadgroup float partial[KQ_HC_NSG];
  float v = kq_hc_tg_sum<KQ_HC_NSG>(acc, lane, sg, partial);
  if (sg == 0 && lane == 0) {
    if (m < (uint)KQ_HC_MIX) {
      mixes_raw[row * KQ_HC_MIX + m] = v;
    } else {
      sumsq[row] = v;
    }
  }
}

// Shared body of kq_hc_front_expand_reduce, also run by every threadgroup
// of the fused kernel. partial is threadgroup scratch [KQ_HC_NSG].
template <typename T>
inline void kq_hc_front_expand_reduce_body(
    const device T* x_sub,
    const device T* resid,
    const device float* post,
    const device float* comb,
    const device float* fn,
    device T* h_out,
    device float* mixes_raw,
    device float* sumsq,
    const int D,
    uint row,
    uint m,
    uint tid,
    threadgroup float* partial) {
  const uint lane = tid % 32;
  const uint sg = tid / 32;
  const int KTOT = KQ_HC * D;
  const uint D4q = (uint)D / 4;

  const device T* xs = x_sub + (int64_t)row * D;
  const device T* rr = resid + (int64_t)row * KTOT;
  device T* hout = h_out + (int64_t)row * KTOT;

  const uint pb = row * 4, cb = row * 16;
  const float p0 = post[pb + 0], p1 = post[pb + 1];
  const float p2 = post[pb + 2], p3 = post[pb + 3];
  // comb is [j][i]; the expand applies comb^T: h_i = sum_j comb[j][i] r_j
  float c[16];
  for (int j = 0; j < 16; ++j) {
    c[j] = comb[cb + j];
  }

  using T4 = vec<T, 4>;
  const device T4* xs4 = (const device T4*)xs;
  const device T4* rr4 = (const device T4*)rr;
  device T4* h4 = (device T4*)hout;
  const device float4* f4 = (const device float4*)(fn + (int64_t)m * KTOT);

  float acc = 0.0f;
  for (uint d4 = tid; d4 < D4q; d4 += KQ_HC_NT) {
    float4 xv = float4(xs4[d4]);
    float4 r0 = float4(rr4[0 * D4q + d4]);
    float4 r1 = float4(rr4[1 * D4q + d4]);
    float4 r2 = float4(rr4[2 * D4q + d4]);
    float4 r3 = float4(rr4[3 * D4q + d4]);
    // the same fma chain per stream as kq_hc_expand, rounded to T once
    T4 hv0 = T4(fma(
        float4(p0),
        xv,
        fma(float4(c[0]),
            r0,
            fma(float4(c[4]), r1, fma(float4(c[8]), r2, float4(c[12]) * r3)))));
    T4 hv1 = T4(fma(
        float4(p1),
        xv,
        fma(float4(c[1]),
            r0,
            fma(float4(c[5]), r1, fma(float4(c[9]), r2, float4(c[13]) * r3)))));
    T4 hv2 = T4(
        fma(float4(p2),
            xv,
            fma(float4(c[2]),
                r0,
                fma(float4(c[6]),
                    r1,
                    fma(float4(c[10]), r2, float4(c[14]) * r3)))));
    T4 hv3 = T4(
        fma(float4(p3),
            xv,
            fma(float4(c[3]),
                r0,
                fma(float4(c[7]),
                    r1,
                    fma(float4(c[11]), r2, float4(c[15]) * r3)))));
    float4 h0 = float4(hv0), h1 = float4(hv1);
    float4 h2 = float4(hv2), h3 = float4(hv3);
    if (m < (uint)KQ_HC_MIX) {
      float4 f0 = f4[0 * D4q + d4];
      float4 f1 = f4[1 * D4q + d4];
      float4 f2 = f4[2 * D4q + d4];
      float4 f3 = f4[3 * D4q + d4];
      KQ_HC_DOT16(acc, h0, h1, h2, h3, f0, f1, f2, f3)
    } else {
      h4[0 * D4q + d4] = hv0;
      h4[1 * D4q + d4] = hv1;
      h4[2 * D4q + d4] = hv2;
      h4[3 * D4q + d4] = hv3;
      KQ_HC_DOT16(acc, h0, h1, h2, h3, h0, h1, h2, h3)
    }
  }

  float v = kq_hc_tg_sum<KQ_HC_NSG>(acc, lane, sg, partial);
  if (sg == 0 && lane == 0) {
    if (m < (uint)KQ_HC_MIX) {
      mixes_raw[row * KQ_HC_MIX + m] = v;
    } else {
      sumsq[row] = v;
    }
  }
}

template <typename T>
[[kernel]] void kq_hc_front_expand_reduce(
    const device T* x_sub [[buffer(0)]],
    const device T* resid [[buffer(1)]],
    const device float* post [[buffer(2)]],
    const device float* comb [[buffer(3)]],
    const device float* fn [[buffer(4)]],
    device T* h_out [[buffer(5)]],
    device float* mixes_raw [[buffer(6)]],
    device float* sumsq [[buffer(7)]],
    const constant int& D [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float partial[KQ_HC_NSG];
  kq_hc_front_expand_reduce_body<T>(
      x_sub,
      resid,
      post,
      comb,
      fn,
      h_out,
      mixes_raw,
      sumsq,
      D,
      tg / (KQ_HC_MIX + 1),
      tg % (KQ_HC_MIX + 1),
      tid,
      partial);
}

// Shared body of kq_hc_sinkhorn_collapse, also run by the collapsing
// threadgroup of the fused kernel. The first KQ_HC_NC threads collapse
// the streams with the pre gates, which every thread derives on its own
// from the mixes, while the last simdgroup runs the sinkhorn (post and
// comb). ssq_shared is [KQ_HC_NSG] and inv_shared [1] of threadgroup
// scratch.
template <typename T>
inline void kq_hc_sinkhorn_collapse_body(
    const device T* x,
    const device float* mixes_raw,
    const device float* sumsq,
    const device float* scale,
    const device float* base,
    const device T* w,
    device T* collapsed,
    device float* post,
    device float* comb,
    const int D,
    const int iters,
    const float hc_eps,
    const float norm_eps,
    uint row,
    uint tid,
    threadgroup float* ssq_shared,
    threadgroup float* inv_shared) {
  constexpr int NSG = KQ_HC_NSG;
  constexpr int NC = KQ_HC_NC;
  constexpr int MAXC = KQ_HC_MAX_CHUNKS;
  const uint lane = tid % 32;
  const uint sg = tid / 32;
  const int BASE_OFF = 2 * KQ_HC;
  const float EPS = hc_eps;
  const float NEPS = norm_eps;
  const bool worker = tid < (uint)NC;
  const bool spare = (sg == (uint)(NSG - 1));

  const device float* mix = mixes_raw + row * KQ_HC_MIX;
  device float* post_out = post + row * KQ_HC;
  device float* comb_out = comb + row * KQ_HC * KQ_HC;

  const float factor = metal::rsqrt(sumsq[row] / (float)(KQ_HC * D) + NEPS);

  const device T* x_row = x + (int64_t)row * (KQ_HC * D);
  device T* out_row = collapsed + (int64_t)row * D;

  using T4 = vec<T, 4>;
  const device T4* x_row0 = (const device T4*)(x_row + 0 * D);
  const device T4* x_row1 = (const device T4*)(x_row + 1 * D);
  const device T4* x_row2 = (const device T4*)(x_row + 2 * D);
  const device T4* x_row3 = (const device T4*)(x_row + 3 * D);
  device T4* out4 = (device T4*)out_row;

  const uint D4 = (uint)D / 4;
  const uint chunks = (D4 + NC - 1) / NC;

  const float pre_scale = scale[0] * factor;
  const float p0 =
      1.0f / (1.0f + metal::fast::exp(-(mix[0] * pre_scale + base[0]))) + EPS;
  const float p1 =
      1.0f / (1.0f + metal::fast::exp(-(mix[1] * pre_scale + base[1]))) + EPS;
  const float p2 =
      1.0f / (1.0f + metal::fast::exp(-(mix[2] * pre_scale + base[2]))) + EPS;
  const float p3 =
      1.0f / (1.0f + metal::fast::exp(-(mix[3] * pre_scale + base[3]))) + EPS;

  if (spare) {
    const float post_scale = scale[1] * factor;
    const float comb_scale = scale[2] * factor;

    const float active = (lane < (uint)KQ_HC) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, (uint)(KQ_HC - 1));

    float post_z = mix[KQ_HC + llane] * post_scale + base[KQ_HC + llane];
    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)KQ_HC) {
      post_out[lane] = post_v;
    }

    float4 v =
        (*(const device float4*)(mix + BASE_OFF + llane * KQ_HC) * comb_scale +
         *(const device float4*)(base + BASE_OFF + llane * KQ_HC)) *
        active;

    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    float4 e = metal::fast::exp(v - row_max) * active;
    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;

    float4 col_inv = 1.0f /
        (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) +
         EPS);
    r *= col_inv;

    for (int iter = 1; iter < iters; ++iter) {
      r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
      col_inv = 1.0f /
          (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) +
           EPS);
      r *= col_inv;
    }

    if (lane < (uint)KQ_HC) {
      *(device float4*)(comb_out + lane * KQ_HC) = r;
    }
  }

  float4 vals[MAXC];
  float ssq = 0.0f;
  for (uint c = 0; c < chunks; ++c) {
    uint d4 = c * NC + tid;
    float4 result = float4(0.0f);
    if (worker && d4 < D4) {
      float4 x0 = float4(x_row0[d4]);
      float4 x1 = float4(x_row1[d4]);
      float4 x2 = float4(x_row2[d4]);
      float4 x3 = float4(x_row3[d4]);
      result =
          fma(float4(p0),
              x0,
              fma(float4(p1), x1, fma(float4(p2), x2, float4(p3) * x3)));
      ssq += result.x * result.x + result.y * result.y + result.z * result.z +
          result.w * result.w;
    }
    vals[c] = result;
  }

  float tot = kq_hc_tg_sum<NSG>(ssq, lane, sg, ssq_shared);
  if (sg == 0 && lane == 0) {
    inv_shared[0] = metal::rsqrt(tot / (float)D + NEPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = inv_shared[0];

  for (uint c = 0; c < chunks; ++c) {
    uint d4 = c * NC + tid;
    if (worker && d4 < D4) {
      uint d = d4 * 4;
      float4 wv = float4(
          (float)w[d], (float)w[d + 1], (float)w[d + 2], (float)w[d + 3]);
      out4[d4] = T4(vals[c] * inv * wv);
    }
  }
}

template <typename T>
[[kernel]] void kq_hc_sinkhorn_collapse(
    const device T* x [[buffer(0)]],
    const device float* mixes_raw [[buffer(1)]],
    const device float* sumsq [[buffer(2)]],
    const device float* scale [[buffer(3)]],
    const device float* base [[buffer(4)]],
    const device T* w [[buffer(5)]],
    device T* collapsed [[buffer(6)]],
    device float* post [[buffer(7)]],
    device float* comb [[buffer(8)]],
    const constant int& D [[buffer(9)]],
    const constant int& iters [[buffer(10)]],
    const constant float& hc_eps [[buffer(11)]],
    const constant float& norm_eps [[buffer(12)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float ssq_shared[KQ_HC_NSG];
  threadgroup float inv_shared[1];
  kq_hc_sinkhorn_collapse_body<T>(
      x,
      mixes_raw,
      sumsq,
      scale,
      base,
      w,
      collapsed,
      post,
      comb,
      D,
      iters,
      hc_eps,
      norm_eps,
      row,
      tid,
      ssq_shared,
      inv_shared);
}

// The two-dispatch front cycle as one dispatch. Every threadgroup of a
// row runs the expand_reduce body, publishes its device writes and
// increments the row's arrival counter; the threadgroup whose increment
// completes the count of KQ_HC_MIX + 1 continues into the collapse body
// over the mixes and h the others published. Nothing waits, so the grid
// needs no co-residency and any row count works; the counter is zero on
// entry (the host clears a fresh one per dispatch) and is left at
// KQ_HC_MIX + 1. The collapsing threadgroup reads h and the mixes after
// a device-scope fence on the acquiring side that matches the release
// fence each publisher issues before its increment.
template <typename T>
[[kernel]] void kq_hc_front_expand_collapse(
    const device T* x_sub [[buffer(0)]],
    const device T* resid [[buffer(1)]],
    const device float* post_in [[buffer(2)]],
    const device float* comb_in [[buffer(3)]],
    const device float* fn [[buffer(4)]],
    const device float* scale [[buffer(5)]],
    const device float* base [[buffer(6)]],
    const device T* w [[buffer(7)]],
    device T* h_out [[buffer(8)]],
    device float* mixes_raw [[buffer(9)]],
    device float* sumsq [[buffer(10)]],
    device atomic_uint* arrive [[buffer(11)]],
    device T* collapsed [[buffer(12)]],
    device float* post_out [[buffer(13)]],
    device float* comb_out [[buffer(14)]],
    const constant int& D [[buffer(15)]],
    const constant int& iters [[buffer(16)]],
    const constant float& hc_eps [[buffer(17)]],
    const constant float& norm_eps [[buffer(18)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float partial[KQ_HC_NSG];
  threadgroup float ssq_shared[KQ_HC_NSG];
  threadgroup float inv_shared[1];
  threadgroup uint last_flag;

  const uint row = tg / (KQ_HC_MIX + 1);
  const uint m = tg % (KQ_HC_MIX + 1);

  kq_hc_front_expand_reduce_body<T>(
      x_sub,
      resid,
      post_in,
      comb_in,
      fn,
      h_out,
      mixes_raw,
      sumsq,
      D,
      row,
      m,
      tid,
      partial);

  // publish this threadgroup's mix dot (or h and sumsq), then arrive
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
    uint prev =
        atomic_fetch_add_explicit(arrive + row, 1u, memory_order_relaxed);
    last_flag = (prev == (uint)KQ_HC_MIX) ? 1u : 0u;
    if (last_flag != 0u) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last_flag == 0u) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_device);

  kq_hc_sinkhorn_collapse_body<T>(
      h_out,
      mixes_raw,
      sumsq,
      scale,
      base,
      w,
      collapsed,
      post_out,
      comb_out,
      D,
      iters,
      hc_eps,
      norm_eps,
      row,
      tid,
      ssq_shared,
      inv_shared);
}

template <typename T>
[[kernel]] void kq_hc_expand(
    const device T* x [[buffer(0)]],
    const device T* resid [[buffer(1)]],
    const device float* post [[buffer(2)]],
    const device float* comb [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& D [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  const uint NTG = 2;
  const uint row = tg / NTG;
  const uint sub = tg % NTG;

  const device T* xr = x + (int64_t)row * D;
  const device T* rr = resid + (int64_t)row * 4 * D;
  device T* orow = out + (int64_t)row * 4 * D;

  const uint pb = row * 4, cb = row * 16;
  float p0 = post[pb + 0], p1 = post[pb + 1];
  float p2 = post[pb + 2], p3 = post[pb + 3];
  // comb is [j][i]; expand applies comb^T: sum_j comb[j][i] * res[j]
  float c00 = comb[cb + 0], c01 = comb[cb + 1];
  float c02 = comb[cb + 2], c03 = comb[cb + 3];
  float c10 = comb[cb + 4], c11 = comb[cb + 5];
  float c12 = comb[cb + 6], c13 = comb[cb + 7];
  float c20 = comb[cb + 8], c21 = comb[cb + 9];
  float c22 = comb[cb + 10], c23 = comb[cb + 11];
  float c30 = comb[cb + 12], c31 = comb[cb + 13];
  float c32 = comb[cb + 14], c33 = comb[cb + 15];

  const uint SPAN = (uint)D / NTG;
  const uint d0 = sub * SPAN;
  using T4 = vec<T, 4>;
  const device T4* x4 = (const device T4*)(xr + d0);
  const device T4* r04 = (const device T4*)(rr + 0 * D + d0);
  const device T4* r14 = (const device T4*)(rr + 1 * D + d0);
  const device T4* r24 = (const device T4*)(rr + 2 * D + d0);
  const device T4* r34 = (const device T4*)(rr + 3 * D + d0);
  device T4* o04 = (device T4*)(orow + 0 * D + d0);
  device T4* o14 = (device T4*)(orow + 1 * D + d0);
  device T4* o24 = (device T4*)(orow + 2 * D + d0);
  device T4* o34 = (device T4*)(orow + 3 * D + d0);
  for (uint k = tid; k < SPAN / 4; k += 256) {
    float4 xv = float4(x4[k]);
    float4 r0 = float4(r04[k]), r1 = float4(r14[k]);
    float4 r2 = float4(r24[k]), r3 = float4(r34[k]);
    o04[k] = T4(
        fma(float4(p0),
            xv,
            fma(float4(c00),
                r0,
                fma(float4(c10), r1, fma(float4(c20), r2, float4(c30) * r3)))));
    o14[k] = T4(
        fma(float4(p1),
            xv,
            fma(float4(c01),
                r0,
                fma(float4(c11), r1, fma(float4(c21), r2, float4(c31) * r3)))));
    o24[k] = T4(
        fma(float4(p2),
            xv,
            fma(float4(c02),
                r0,
                fma(float4(c12), r1, fma(float4(c22), r2, float4(c32) * r3)))));
    o34[k] = T4(
        fma(float4(p3),
            xv,
            fma(float4(c03),
                r0,
                fma(float4(c13), r1, fma(float4(c23), r2, float4(c33) * r3)))));
  }
}
