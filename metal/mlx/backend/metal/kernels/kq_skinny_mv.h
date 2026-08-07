// Skinny matmul: y = x @ w.T for token widths 1..16 against a small-N,
// large-K weight (w is [N, K], nn.Linear layout). MLX's steel GEMM leaves
// the GEMV fast path at M >= 2 and runs these shapes 3-10x slower than
// their bytes (router gates, indexer weight projections, hyper-connection
// mixes at speculative verify widths); this kernel keeps the GEMV shape:
// one simdgroup per output column streams the w row coalesced (float4 per
// lane) while holding all M row accumulators in registers, f32 accumulate,
// one round to OT at the write.
//
// Grid: (ceil(N / KQ_SKINNY_NSG), T, 1) threadgroups of 32 * KQ_SKINNY_NSG
// threads; T collapses the batch dims ahead of the [M, K] tail. Requires
// K % 4 == 0 and M <= KQ_SKINNY_MMAX.

#define KQ_SKINNY_NSG 8
#define KQ_SKINNY_MMAX 16

template <typename XT, typename WT, typename OT>
[[kernel]] void kq_skinny_mv(
    const device XT* x [[buffer(0)]],
    const device WT* w [[buffer(1)]],
    device OT* out [[buffer(2)]],
    const constant int& M [[buffer(3)]],
    const constant int& N [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int n = tid.x * KQ_SKINNY_NSG + simd_gid;
  if (n >= N) {
    return;
  }
  using XT4 = vec<XT, 4>;
  using WT4 = vec<WT, 4>;
  const int K4 = K / 4;
  const device WT4* wrow = (const device WT4*)(w + (int64_t)n * K);
  const device XT4* xrows = (const device XT4*)(x + (int64_t)tid.y * M * K);

  float acc[KQ_SKINNY_MMAX] = {0};
  for (int k4 = simd_lid; k4 < K4; k4 += 32) {
    const float4 wv = float4(wrow[k4]);
    for (int m = 0; m < M; m++) {
      acc[m] += metal::dot(float4(xrows[m * K4 + k4]), wv);
    }
  }
  device OT* orow = out + (int64_t)tid.y * M * N;
  for (int m = 0; m < M; m++) {
    const float v = simd_sum(acc[m]);
    if (simd_lid == 0) {
      orow[m * N + n] = static_cast<OT>(v);
    }
  }
}
