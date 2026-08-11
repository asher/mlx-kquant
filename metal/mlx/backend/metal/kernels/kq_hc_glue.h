// Fused hyper-connection glue kernels for the deepseek4 M=1 decode route.
// Four streams (hc_mult 4) baked in; D is the per-stream hidden size.
// Numerics mirror the certified gmlx JIT kernels exactly: f32 accumulate,
// fast::exp sinkhorn, round-before-use in the fused expand, single
// rounding at each T-dtype write.
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
//                             into the output. One threadgroup per row.
//   kq_hc_expand:             pre/comb expand of the sublayer output back
//                             to four streams. Two threadgroups per row.
//   kq_hc_front_expand_collapse: the two above as one dispatch, the
//                             sumsq threadgroup continuing into the
//                             collapse once an arrival counter says every
//                             mix threadgroup has published its dot.
//                             Single row, fixed 25-threadgroup grid.
//
// The expand_reduce and sinkhorn_collapse bodies live in shared inline
// functions so the split and continuation routes run the same arithmetic
// in the same order by construction and cannot drift apart.

#define KQ_HC 4
#define KQ_HC_MIX ((2 + KQ_HC) * KQ_HC)
#define KQ_HC_MAX_CHUNKS 8

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

  const uint row = tg / (KQ_HC_MIX + 1);
  const uint m = tg % (KQ_HC_MIX + 1);

  const device T* xr = x + (int64_t)row * KTOT;
  using T4 = vec<T, 4>;
  const device T4* x4 = (const device T4*)xr;

  float acc = 0.0f;
  if (m < (uint)KQ_HC_MIX) {
    const device float4* f4 = (const device float4*)(fn + (int64_t)m * KTOT);
    for (uint k = tid; k < (uint)(KTOT / 4); k += 256) {
      float4 xv = float4(x4[k]);
      float4 fv = f4[k];
      acc = fma(xv.x, fv.x, acc);
      acc = fma(xv.y, fv.y, acc);
      acc = fma(xv.z, fv.z, acc);
      acc = fma(xv.w, fv.w, acc);
    }
  } else {
    for (uint k = tid; k < (uint)(KTOT / 4); k += 256) {
      float4 xv = float4(x4[k]);
      acc = fma(xv.x, xv.x, acc);
      acc = fma(xv.y, xv.y, acc);
      acc = fma(xv.z, xv.z, acc);
      acc = fma(xv.w, xv.w, acc);
    }
  }

  threadgroup float partial[8];
  acc = simd_sum(acc);
  if (lane == 0) {
    partial[sg] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = (lane < 8) ? partial[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) {
      if (m < (uint)KQ_HC_MIX) {
        mixes_raw[row * KQ_HC_MIX + m] = v;
      } else {
        sumsq[row] = v;
      }
    }
  }
}

// Shared body of kq_hc_front_expand_reduce, also run by every threadgroup
// of the fused continuation kernel. partial is threadgroup scratch [8].
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

  using T4 = vec<T, 4>;
  const device T4* xs4 = (const device T4*)xs;
  const device T4* rr4 = (const device T4*)rr;
  device T4* h4 = (device T4*)hout;
  const device float4* f4 = (const device float4*)(fn + (int64_t)m * KTOT);

  float acc = 0.0f;
  for (uint i = 0; i < (uint)KQ_HC; ++i) {
    const float pi = post[pb + i];
    const float c0 = comb[cb + 0 * 4 + i];
    const float c1 = comb[cb + 1 * 4 + i];
    const float c2 = comb[cb + 2 * 4 + i];
    const float c3 = comb[cb + 3 * 4 + i];
    for (uint d4 = tid; d4 < D4q; d4 += 256) {
      uint k = i * D4q + d4;
      float4 xv = float4(xs4[d4]);
      float4 r0 = float4(rr4[0 * D4q + d4]);
      float4 r1 = float4(rr4[1 * D4q + d4]);
      float4 r2 = float4(rr4[2 * D4q + d4]);
      float4 r3 = float4(rr4[3 * D4q + d4]);
      float4 e =
          fma(float4(pi),
              xv,
              fma(float4(c0),
                  r0,
                  fma(float4(c1), r1, fma(float4(c2), r2, float4(c3) * r3))));
      T4 hv = T4(e);
      float4 hf = float4(hv);
      if (m < (uint)KQ_HC_MIX) {
        float4 fv = f4[k];
        acc = fma(hf.x, fv.x, acc);
        acc = fma(hf.y, fv.y, acc);
        acc = fma(hf.z, fv.z, acc);
        acc = fma(hf.w, fv.w, acc);
      } else {
        h4[k] = hv;
        acc = fma(hf.x, hf.x, acc);
        acc = fma(hf.y, hf.y, acc);
        acc = fma(hf.z, hf.z, acc);
        acc = fma(hf.w, hf.w, acc);
      }
    }
  }

  acc = simd_sum(acc);
  if (lane == 0) {
    partial[sg] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = (lane < 8) ? partial[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) {
      if (m < (uint)KQ_HC_MIX) {
        mixes_raw[row * KQ_HC_MIX + m] = v;
      } else {
        sumsq[row] = v;
      }
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
  threadgroup float partial[8];
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

// Shared body of kq_hc_sinkhorn_collapse, also run by the continuation
// threadgroup of the fused kernel. pre_shared is [KQ_HC], ssq_shared [8]
// and inv_shared [1] of threadgroup scratch.
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
    threadgroup float* pre_shared,
    threadgroup float* ssq_shared,
    threadgroup float* inv_shared) {
  const uint lane = tid % 32;
  const uint sg = tid / 32;
  const int BASE_OFF = 2 * KQ_HC;
  const float EPS = hc_eps;
  const float NEPS = norm_eps;

  const device float* mix = mixes_raw + row * KQ_HC_MIX;
  device float* post_out = post + row * KQ_HC;
  device float* comb_out = comb + row * KQ_HC * KQ_HC;

  const float factor = metal::rsqrt(sumsq[row] / (float)(KQ_HC * D) + NEPS);

  if (sg == 0) {
    const float pre_scale = scale[0] * factor;
    const float post_scale = scale[1] * factor;
    const float comb_scale = scale[2] * factor;

    const float active = (lane < (uint)KQ_HC) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, (uint)(KQ_HC - 1));

    float pre_z = mix[llane] * pre_scale + base[llane];
    float post_z = mix[KQ_HC + llane] * post_scale + base[KQ_HC + llane];
    float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

    if (lane < (uint)KQ_HC) {
      pre_shared[lane] = pre_v;
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

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float p0 = pre_shared[0];
  const float p1 = pre_shared[1];
  const float p2 = pre_shared[2];
  const float p3 = pre_shared[3];

  const device T* x_row = x + (int64_t)row * (KQ_HC * D);
  device T* out_row = collapsed + (int64_t)row * D;

  using T4 = vec<T, 4>;
  const device T4* x_row0 = (const device T4*)(x_row + 0 * D);
  const device T4* x_row1 = (const device T4*)(x_row + 1 * D);
  const device T4* x_row2 = (const device T4*)(x_row + 2 * D);
  const device T4* x_row3 = (const device T4*)(x_row + 3 * D);
  device T4* out4 = (device T4*)out_row;

  const uint D4 = (uint)D / 4;
  const uint chunks = (D4 + 255) / 256;

  float4 vals[KQ_HC_MAX_CHUNKS];
  float ssq = 0.0f;
  for (uint c = 0; c < chunks; ++c) {
    uint d4 = c * 256 + tid;
    float4 result = float4(0.0f);
    if (d4 < D4) {
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

  ssq = simd_sum(ssq);
  if (lane == 0) {
    ssq_shared[sg] = ssq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = (lane < 8) ? ssq_shared[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) {
      inv_shared[0] = metal::rsqrt(v / (float)D + NEPS);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = inv_shared[0];

  for (uint c = 0; c < chunks; ++c) {
    uint d4 = c * 256 + tid;
    if (d4 < D4) {
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
  threadgroup float pre_shared[KQ_HC];
  threadgroup float ssq_shared[8];
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
      pre_shared,
      ssq_shared,
      inv_shared);
}

// The two-dispatch front cycle as one dispatch. Every threadgroup runs the
// expand_reduce body, publishes its device writes and arrives at the
// counter; the sumsq threadgroup then continues into the collapse body
// once all KQ_HC_MIX + 1 have arrived.
//
// The wait is unbounded on purpose. Metal gives no forward-progress
// guarantee across threadgroups, so the protocol is safe only while the
// whole grid is co-resident: the grid is a fixed KQ_HC_MIX + 1 = 25
// threadgroups for a single row, which the host side enforces by
// rejecting any other row count. A bounded spin would trade an
// unschedulable grid, which is a visible hang, for a silently wrong
// result, which is worse.
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
  threadgroup float partial[8];
  threadgroup float pre_shared[KQ_HC];
  threadgroup float ssq_shared[8];
  threadgroup float inv_shared[1];

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
      0u,
      tg,
      tid,
      partial);

  // publish this threadgroup's mix dot (or h and sumsq) before arriving
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
    atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
  }

  if (tg != (uint)KQ_HC_MIX) {
    return;
  }

  if (tid == 0) {
    uint v = atomic_load_explicit(arrive, memory_order_relaxed);
    while (v < (uint)(KQ_HC_MIX + 1)) {
      v = atomic_load_explicit(arrive, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
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
      0u,
      tid,
      pre_shared,
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
