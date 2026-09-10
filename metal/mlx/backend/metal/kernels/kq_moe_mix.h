// Fused unsort and score mix for sorted-prefill MoE:
//   out[t, :] = sum_s scores[t, s] * y[inv_order[t * k + s], :]
// where y holds the expert outputs in routing-sorted row order and
// inv_order maps each (token, slot) pair to its sorted row. One dispatch
// replaces the gather back to token order, the score multiply and the sum
// over slots. One threadgroup per output row; thread i covers columns
// i * 4 + j * 4 * threads (8-byte vector loads and stores), f32
// accumulation, one round at the write. N must be a multiple of 4.

template <typename T>
[[kernel]] void kq_moe_mix(
    const device T* y [[buffer(0)]],
    const device uint32_t* inv_order [[buffer(1)]],
    const device float* scores [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& N [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]) {
  using v4 = metal::vec<T, 4>;
  const int t = int(tgid);
  const device uint32_t* io = inv_order + size_t(t) * K;
  const device float* sc = scores + size_t(t) * K;
  device T* orow = out + size_t(t) * N;
  for (int c = int(tid) * 4; c < N; c += int(ntg) * 4) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int s = 0; s < K; ++s) {
      const float w = sc[s];
      const device v4* src =
          reinterpret_cast<const device v4*>(y + size_t(io[s]) * N + c);
      const v4 r = *src;
      acc0 = metal::fma(w, float(r.x), acc0);
      acc1 = metal::fma(w, float(r.y), acc1);
      acc2 = metal::fma(w, float(r.z), acc2);
      acc3 = metal::fma(w, float(r.w), acc3);
    }
    *reinterpret_cast<device v4*>(orow + c) =
        v4(T(acc0), T(acc1), T(acc2), T(acc3));
  }
}
