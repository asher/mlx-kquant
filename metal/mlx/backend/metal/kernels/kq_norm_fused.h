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
// mx::fast::rms_norm semantics. Grid: (T, 1, 1) threadgroups of (256, 1, 1);
// rows are launch-bound at decode so scalar strided loads are fine.

#define KQ_NORM_NT 256
#define KQ_NORM_NSG (KQ_NORM_NT / 32)

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
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[KQ_NORM_NSG];
  threadgroup float stat[1];

  const device T* hrow = h + (int64_t)tid * D;
  const device T* rrow = residual + (int64_t)tid * D;
  device T* orow = out + (int64_t)tid * D;

  float ss = 0;
  for (int i = lid; i < D; i += KQ_NORM_NT) {
    const float v = float(hrow[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (simd_lid == 0) {
    red[simd_gid] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float g = 0;
    for (int i = 0; i < KQ_NORM_NSG; i++) {
      g += red[i];
    }
    stat[0] = g;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = metal::rsqrt(stat[0] / float(D) + eps);
  const float sc = HAS_SCALE ? float(lscale[0]) : 1.0f;

  for (int i = lid; i < D; i += KQ_NORM_NT) {
    const float v = float(rrow[i]) + float(w[i]) * float(hrow[i]) * inv;
    orow[i] = static_cast<T>(v * sc);
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
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[KQ_NORM_NSG];
  threadgroup float stat[1];

  const device T* xrow = x + (int64_t)tid * D;
  device T* o0 = out0 + (int64_t)tid * D;
  device T* o1 = out1 + (int64_t)tid * D;
  device T* o2 = out2 + (int64_t)tid * D;

  float ss = 0;
  for (int i = lid; i < D; i += KQ_NORM_NT) {
    const float v = float(xrow[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (simd_lid == 0) {
    red[simd_gid] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float g = 0;
    for (int i = 0; i < KQ_NORM_NSG; i++) {
      g += red[i];
    }
    stat[0] = g;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = metal::rsqrt(stat[0] / float(D) + eps);

  for (int i = lid; i < D; i += KQ_NORM_NT) {
    const float v = float(xrow[i]) * inv;
    o0[i] = static_cast<T>(float(w0[i]) * v);
    o1[i] = static_cast<T>(float(w1[i]) * v);
    o2[i] = static_cast<T>(float(w2[i]) * v);
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
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[2 * KQ_NORM_NSG];
  threadgroup float stat[2];

  const device T* arow = a + (int64_t)tid * D;
  const device T* brow = b + (int64_t)tid * D;
  device T* orow = out + (int64_t)tid * D;

  float sa = 0;
  float sb = 0;
  for (int i = lid; i < D; i += KQ_NORM_NT) {
    const float va = float(arow[i]);
    const float vb = float(brow[i]);
    sa += va * va;
    sb += vb * vb;
  }
  sa = simd_sum(sa);
  sb = simd_sum(sb);
  if (simd_lid == 0) {
    red[simd_gid] = sa;
    red[KQ_NORM_NSG + simd_gid] = sb;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float ga = 0;
    float gb = 0;
    for (int i = 0; i < KQ_NORM_NSG; i++) {
      ga += red[i];
      gb += red[KQ_NORM_NSG + i];
    }
    stat[0] = ga;
    stat[1] = gb;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inva = metal::rsqrt(stat[0] / float(D) + eps);
  const float invb = metal::rsqrt(stat[1] / float(D) + eps);

  for (int i = lid; i < D; i += KQ_NORM_NT) {
    const float v = float(wa[i]) * float(arow[i]) * inva +
        float(wb[i]) * float(brow[i]) * invb;
    orow[i] = static_cast<T>(v);
  }
}
