// Fused output gate for gated-delta layers (KDA, GDN):
//   out = rms_norm(x, w, eps) * sigmoid(gate)
// over rows of D = 32 * NPT elements, one simdgroup per row (lane l
// covers elements l * NPT .. + NPT - 1, the mean square is one simd_sum),
// eight rows per threadgroup. All math in f32, one round at the write.

template <typename T, int NPT>
[[kernel]] void kq_rmsnorm_gate(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    const device T* gate [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant float& eps [[buffer(4)]],
    const constant int& nrows [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int D = 32 * NPT;
  const int rid = int(tgid) * 8 + int(simd_gid);
  if (rid >= nrows) {
    return;
  }
  const int c0 = int(lane) * NPT;
  const device T* xr = x + size_t(rid) * D + c0;
  const device T* gr = gate + size_t(rid) * D + c0;
  const device T* wr = w + c0;
  using v4 = metal::vec<T, 4>;
  using v2 = metal::vec<T, 2>;
  float xv[NPT], gv[NPT], wv[NPT];
  if (NPT == 2) {
    const v2 a = *reinterpret_cast<const device v2*>(xr);
    const v2 g = *reinterpret_cast<const device v2*>(gr);
    const v2 ww = *reinterpret_cast<const device v2*>(wr);
    xv[0] = float(a.x);
    xv[1] = float(a.y);
    gv[0] = float(g.x);
    gv[1] = float(g.y);
    wv[0] = float(ww.x);
    wv[1] = float(ww.y);
  } else {
    for (int i = 0; i < NPT; i += 4) {
      const v4 a = *reinterpret_cast<const device v4*>(xr + i);
      const v4 g = *reinterpret_cast<const device v4*>(gr + i);
      const v4 ww = *reinterpret_cast<const device v4*>(wr + i);
      xv[i] = float(a.x);
      xv[i + 1] = float(a.y);
      xv[i + 2] = float(a.z);
      xv[i + 3] = float(a.w);
      gv[i] = float(g.x);
      gv[i + 1] = float(g.y);
      gv[i + 2] = float(g.z);
      gv[i + 3] = float(g.w);
      wv[i] = float(ww.x);
      wv[i + 1] = float(ww.y);
      wv[i + 2] = float(ww.z);
      wv[i + 3] = float(ww.w);
    }
  }
  float ss = 0.0f;
  for (int i = 0; i < NPT; ++i) {
    ss = metal::fma(xv[i], xv[i], ss);
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(D) + eps);
  float o[NPT];
  for (int i = 0; i < NPT; ++i) {
    o[i] = xv[i] * inv * wv[i] / (1.0f + metal::exp(-gv[i]));
  }
  device T* orow = out + size_t(rid) * D + c0;
  if (NPT == 2) {
    *reinterpret_cast<device v2*>(orow) = v2(T(o[0]), T(o[1]));
  } else {
    for (int i = 0; i < NPT; i += 4) {
      *reinterpret_cast<device v4*>(orow + i) =
          v4(T(o[i]), T(o[i + 1]), T(o[i + 2]), T(o[i + 3]));
    }
  }
}
