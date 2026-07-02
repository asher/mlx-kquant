// Fused residual + RMSNorm glue kernels, decode-shaped (one threadgroup per
// activation row). Transformer layer glue (norms, residual adds, per-layer
// scalars) is many tiny dependent dispatches at decode; each kernel here
// collapses one recurring pattern into a single dispatch:
//
//   kq_add_rmsnorm:  out = (residual + rms_norm(h, w)) [* scale] -- the
//     post-norm residual pattern (norm then add), with an optional [1]
//     scalar epilogue (gemma-4's layer_scalar).
//   kq_rmsnorm_multi3:  out_i = rms_norm(x, w_i) for i in 0..2 -- three
//     norms of the SAME tensor share one mean-square reduction (gemma-4
//     MoE: pre_feedforward, router norm, pre_feedforward_2 all read the
//     post-attention hidden state).
//   kq_rmsnorm2_add:  out = rms_norm(a, wa) + rms_norm(b, wb) -- two
//     independent norms plus the branch merge (gemma-4 MoE: h1 + h2).
//
// All math in f32 (accumulate, normalize, add), one round to T at the write;
// rms_norm(x, w) = w * x * rsqrt(mean(x^2) + eps), matching
// mx::fast::rms_norm semantics. Grid: (T, 1, 1) threadgroups; the host sizes
// the threadgroup to ceil(D / 4) so each thread reads 4 contiguous elements
// once into registers (the stock rms_norm single-row pattern -- a scalar
// strided two-pass loop measures ~3x the exec time and inverts the E2E win).
// Rows wider than 4 * 1024 fall back to a strided two-pass loop.

#define KQ_NORM_NREADS 4
#define KQ_NORM_MAX_NSG 32

template <typename T>
[[kernel]] void kq_add_rmsnorm(
    const device T* h [[buffer(0)]],
    const device T* residual [[buffer(1)]],
    const device T* w [[buffer(2)]],
    const device T* lscale [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& D [[buffer(5)]],
    const constant float& eps [[buffer(6)]],
    const constant int& HAS_SCALE [[buffer(7)]],
    uint tid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[KQ_NORM_MAX_NSG];
  threadgroup float stat[1];

  const device T* hrow = h + (int64_t)tid * D;
  const device T* rrow = residual + (int64_t)tid * D;
  device T* orow = out + (int64_t)tid * D;
  const int nsg = (ntg + 31) / 32;
  const bool cached = D <= int(ntg) * KQ_NORM_NREADS;
  const int base = lid * KQ_NORM_NREADS;

  float v[KQ_NORM_NREADS] = {0, 0, 0, 0};
  float ss = 0;
  if (cached) {
    if (base + KQ_NORM_NREADS <= D) {
      for (int j = 0; j < KQ_NORM_NREADS; j++) {
        v[j] = float(hrow[base + j]);
        ss += v[j] * v[j];
      }
    } else {
      for (int j = 0; base + j < D; j++) {
        v[j] = float(hrow[base + j]);
        ss += v[j] * v[j];
      }
    }
  } else {
    for (int i = lid; i < D; i += ntg) {
      const float x = float(hrow[i]);
      ss += x * x;
    }
  }
  ss = simd_sum(ss);
  if (simd_lid == 0) {
    red[simd_gid] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float g = 0;
    for (int i = 0; i < nsg; i++) {
      g += red[i];
    }
    stat[0] = g;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = metal::rsqrt(stat[0] / float(D) + eps);
  const float sc = HAS_SCALE ? float(lscale[0]) : 1.0f;

  if (cached) {
    for (int j = 0; j < KQ_NORM_NREADS && base + j < D; j++) {
      const float o = float(rrow[base + j]) + float(w[base + j]) * v[j] * inv;
      orow[base + j] = static_cast<T>(o * sc);
    }
  } else {
    for (int i = lid; i < D; i += ntg) {
      const float o = float(rrow[i]) + float(w[i]) * float(hrow[i]) * inv;
      orow[i] = static_cast<T>(o * sc);
    }
  }
}

template <typename T>
[[kernel]] void kq_rmsnorm_multi3(
    const device T* x [[buffer(0)]],
    const device T* w0 [[buffer(1)]],
    const device T* w1 [[buffer(2)]],
    const device T* w2 [[buffer(3)]],
    device T* out0 [[buffer(4)]],
    device T* out1 [[buffer(5)]],
    device T* out2 [[buffer(6)]],
    const constant int& D [[buffer(7)]],
    const constant float& eps [[buffer(8)]],
    uint tid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[KQ_NORM_MAX_NSG];
  threadgroup float stat[1];

  const device T* xrow = x + (int64_t)tid * D;
  device T* o0 = out0 + (int64_t)tid * D;
  device T* o1 = out1 + (int64_t)tid * D;
  device T* o2 = out2 + (int64_t)tid * D;
  const int nsg = (ntg + 31) / 32;
  const bool cached = D <= int(ntg) * KQ_NORM_NREADS;
  const int base = lid * KQ_NORM_NREADS;

  float v[KQ_NORM_NREADS] = {0, 0, 0, 0};
  float ss = 0;
  if (cached) {
    for (int j = 0; j < KQ_NORM_NREADS && base + j < D; j++) {
      v[j] = float(xrow[base + j]);
      ss += v[j] * v[j];
    }
  } else {
    for (int i = lid; i < D; i += ntg) {
      const float xv = float(xrow[i]);
      ss += xv * xv;
    }
  }
  ss = simd_sum(ss);
  if (simd_lid == 0) {
    red[simd_gid] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float g = 0;
    for (int i = 0; i < nsg; i++) {
      g += red[i];
    }
    stat[0] = g;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = metal::rsqrt(stat[0] / float(D) + eps);

  if (cached) {
    for (int j = 0; j < KQ_NORM_NREADS && base + j < D; j++) {
      const float n = v[j] * inv;
      o0[base + j] = static_cast<T>(float(w0[base + j]) * n);
      o1[base + j] = static_cast<T>(float(w1[base + j]) * n);
      o2[base + j] = static_cast<T>(float(w2[base + j]) * n);
    }
  } else {
    for (int i = lid; i < D; i += ntg) {
      const float n = float(xrow[i]) * inv;
      o0[i] = static_cast<T>(float(w0[i]) * n);
      o1[i] = static_cast<T>(float(w1[i]) * n);
      o2[i] = static_cast<T>(float(w2[i]) * n);
    }
  }
}

template <typename T>
[[kernel]] void kq_rmsnorm2_add(
    const device T* a [[buffer(0)]],
    const device T* wa [[buffer(1)]],
    const device T* b [[buffer(2)]],
    const device T* wb [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& D [[buffer(5)]],
    const constant float& eps [[buffer(6)]],
    uint tid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[2 * KQ_NORM_MAX_NSG];
  threadgroup float stat[2];

  const device T* arow = a + (int64_t)tid * D;
  const device T* brow = b + (int64_t)tid * D;
  device T* orow = out + (int64_t)tid * D;
  const int nsg = (ntg + 31) / 32;
  const bool cached = D <= int(ntg) * KQ_NORM_NREADS;
  const int base = lid * KQ_NORM_NREADS;

  float va[KQ_NORM_NREADS] = {0, 0, 0, 0};
  float vb[KQ_NORM_NREADS] = {0, 0, 0, 0};
  float sa = 0;
  float sb = 0;
  if (cached) {
    for (int j = 0; j < KQ_NORM_NREADS && base + j < D; j++) {
      va[j] = float(arow[base + j]);
      vb[j] = float(brow[base + j]);
      sa += va[j] * va[j];
      sb += vb[j] * vb[j];
    }
  } else {
    for (int i = lid; i < D; i += ntg) {
      const float xa = float(arow[i]);
      const float xb = float(brow[i]);
      sa += xa * xa;
      sb += xb * xb;
    }
  }
  sa = simd_sum(sa);
  sb = simd_sum(sb);
  if (simd_lid == 0) {
    red[simd_gid] = sa;
    red[KQ_NORM_MAX_NSG + simd_gid] = sb;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float ga = 0;
    float gb = 0;
    for (int i = 0; i < nsg; i++) {
      ga += red[i];
      gb += red[KQ_NORM_MAX_NSG + i];
    }
    stat[0] = ga;
    stat[1] = gb;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inva = metal::rsqrt(stat[0] / float(D) + eps);
  const float invb = metal::rsqrt(stat[1] / float(D) + eps);

  if (cached) {
    for (int j = 0; j < KQ_NORM_NREADS && base + j < D; j++) {
      const float o = float(wa[base + j]) * va[j] * inva +
          float(wb[base + j]) * vb[j] * invb;
      orow[base + j] = static_cast<T>(o);
    }
  } else {
    for (int i = lid; i < D; i += ntg) {
      const float o = float(wa[i]) * float(arow[i]) * inva +
          float(wb[i]) * float(brow[i]) * invb;
      orow[i] = static_cast<T>(o);
    }
  }
}
