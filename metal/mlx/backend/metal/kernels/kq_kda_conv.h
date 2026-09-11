// Fused causal short convolution for gated-delta (KDA, GDN) prefill:
//   y[t, c] = act(sum_j w[c, j] * in[t - (K - 1) + j, c])
// over the depthwise taps of a [B, T, C] activation with the K - 1 carried
// rows in front (state [B, K - 1, C]), silu on the sum, and an optional
// l2 norm over each D-channel head with a folded scale:
//   y = scale * y * rsqrt(mean_D(y^2) + eps)      (scale != 0)
// followed by the tail rows for the next call (state_out = the last K - 1
// rows of the virtual [state; x] sequence). Everything is f32 from the
// loads to one round at the write; the eager composition (concat, conv1d,
// silu, rms_norm, multiply) rounds to T after each step.
//
// One simdgroup per (token, head): lane l covers channels
// head * D + l * NPT .. + NPT - 1, so the norm is one simd_sum. Eight
// simdgroups per threadgroup; the grid covers B * T * (C / D) rows.

#define KQ_KDA_CONV_MAX_K 8

template <typename T, int NPT>
[[kernel]] void kq_kda_conv(
    const device T* x [[buffer(0)]],
    const device T* state [[buffer(1)]],
    const device T* w [[buffer(2)]],
    device T* y [[buffer(3)]],
    device T* state_out [[buffer(4)]],
    const constant int& T_len [[buffer(5)]],
    const constant int& C [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant float& scale [[buffer(8)]],
    const constant float& eps [[buffer(9)]],
    const constant int& nrows [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int D = 32 * NPT;
  const int NS = K - 1;
  const int heads = C / D;
  // Rows are (b, t, head) in that order, nrows = B * T * heads; the grid
  // may overrun the last group and a whole simdgroup shares one row. One
  // token per simdgroup: a two-token layout sharing the tap rows and
  // vectorized loads both measured slower on M5 (the kernel is bound by
  // the per-simdgroup latency, not the loads).
  const int rid = int(tgid) * 8 + int(simd_gid);
  if (rid >= nrows) {
    return;
  }
  const int bt = rid / heads;
  const int t = bt % T_len;
  const int b = bt / T_len;
  const int head = rid % heads;
  const int c0 = head * D + int(lane) * NPT;

  const device T* xb = x + size_t(b) * T_len * C;
  const device T* sb = state + size_t(b) * NS * C;

  float acc[NPT];
  for (int i = 0; i < NPT; ++i) {
    acc[i] = 0.0f;
  }
  for (int j = 0; j < K && j < KQ_KDA_CONV_MAX_K; ++j) {
    const int m = t - NS + j;
    const device T* row =
        (m < 0) ? sb + size_t(NS + m) * C + c0 : xb + size_t(m) * C + c0;
    for (int i = 0; i < NPT; ++i) {
      acc[i] = metal::fma(float(w[(c0 + i) * K + j]), float(row[i]), acc[i]);
    }
  }
  float ss = 0.0f;
  for (int i = 0; i < NPT; ++i) {
    acc[i] = acc[i] / (1.0f + metal::exp(-acc[i]));
    ss = metal::fma(acc[i], acc[i], ss);
  }
  if (scale != 0.0f) {
    ss = simd_sum(ss);
    const float f = metal::rsqrt(ss / float(D) + eps) * scale;
    for (int i = 0; i < NPT; ++i) {
      acc[i] *= f;
    }
  }
  device T* yrow = y + size_t(bt) * C + c0;
  for (int i = 0; i < NPT; ++i) {
    yrow[i] = T(acc[i]);
  }

  // Tail rows for the next call: row j of state_out is virtual row
  // T - NS + j, from x when that is a token of this call and from the
  // carried state otherwise (T < NS).
  device T* sob = state_out + size_t(b) * NS * C;
  const int jt = t - (T_len - NS);
  if (jt >= 0) {
    const device T* row = xb + size_t(t) * C + c0;
    for (int i = 0; i < NPT; ++i) {
      sob[size_t(jt) * C + c0 + i] = row[i];
    }
  }
  if (t == 0) {
    for (int j = 0; j < NS - T_len; ++j) {
      const device T* row = sb + size_t(T_len + j) * C + c0;
      for (int i = 0; i < NPT; ++i) {
        sob[size_t(j) * C + c0 + i] = row[i];
      }
    }
  }
}
