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

// The same product on simdgroup 8x8 MMA, after the shape of Splash's float
// projections (runtime/metal/kernels/shared/gguf_float.metal, Apache-2.0):
// a threadgroup takes 8 output columns of up to 8 F rows, its
// KQ_SKINNY_SPLITS simdgroups take K in parts of whole 64-input steps, and
// simdgroup f then adds fragment f's parts in simdgroup order, so a result
// depends on neither the grid nor M. MMA j of the step from input k
// multiplies inputs k + 8 i + j over its index i, so lane (fm, fn) loads
// whole runs of 8: x[row fm][k + 8 fn, + 16) as A and w[column fn, fn + 1]
// [k + 8 fm, + 8) as B, and holds out[row fm][column fn, fn + 1]. Even and
// odd MMAs accumulate apart. The weights load a step ahead.
//
// Grid: (ceil(N / 8), T, 1) threadgroups of 32 * KQ_SKINNY_SPLITS threads.
// Requires K % 8 == 0 and M <= 8 F.

#define KQ_SKINNY_SPLITS 8

// Eight consecutive values of a row as float, zeros past the row.
template <typename U>
struct KqRun8 {
  float4 lo;
  float4 hi;
  static METAL_FUNC KqRun8 load(const device U* p, bool inside) {
    using U4 = vec<U, 4>;
    if (!inside) {
      return {float4(0.0f), float4(0.0f)};
    }
    return {float4(*(const device U4*)p), float4(*(const device U4*)(p + 4))};
  }
  METAL_FUNC float operator[](int i) const thread {
    return i < 4 ? lo[i] : hi[i - 4];
  }
};

template <typename XT, typename WT, typename OT, int F>
[[kernel]] void kq_skinny_mma(
    const device XT* x [[buffer(0)]],
    const device WT* w [[buffer(1)]],
    device OT* out [[buffer(2)]],
    const constant int& M [[buffer(3)]],
    const constant int& N [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float2 parts[KQ_SKINNY_SPLITS * F * 32];
  const short qid = short(simd_lid / 4);
  const short fm = (qid & 4) + short((simd_lid / 2) % 4);
  const short fn = (qid & 2) * 2 + short(simd_lid % 2) * 2;
  const int runs = K / 8;
  const int steps = (K + 63) / 64;
  const int first = int(simd_gid) * steps / KQ_SKINNY_SPLITS;
  const int last = (int(simd_gid) + 1) * steps / KQ_SKINNY_SPLITS;
  const int col = int(tid.x) * 8 + fn;
  // Columns past N read the last column and are not written.
  const device WT* w0 = w + (int64_t)min(col, N - 1) * K;
  const device WT* w1 = w + (int64_t)min(col + 1, N - 1) * K;
  const device XT* xb = x + (int64_t)tid.y * M * K;

  simdgroup_float8x8 acc[F][2];
#pragma unroll
  for (int f = 0; f < F; ++f) {
    acc[f][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[f][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }
  KqRun8<WT> nb0;
  KqRun8<WT> nb1;
  {
    const int run = first * 8 + fm;
    nb0 = KqRun8<WT>::load(w0 + run * 8, first < last && run < runs);
    nb1 = KqRun8<WT>::load(w1 + run * 8, first < last && run < runs);
  }
  for (int step = first; step < last; ++step) {
    const KqRun8<WT> b0 = nb0;
    const KqRun8<WT> b1 = nb1;
    if (step + 1 < last) {
      const int run = (step + 1) * 8 + fm;
      nb0 = KqRun8<WT>::load(w0 + run * 8, run < runs);
      nb1 = KqRun8<WT>::load(w1 + run * 8, run < runs);
    }
    const int run = step * 8 + fn;
#pragma unroll
    for (int f = 0; f < F; ++f) {
      // continue, not break: the MMAs past a break sit under an exit the
      // compiler cannot prove uniform.
      if (f * 8 >= M) {
        continue;
      }
      const int row = f * 8 + fm;
      const device XT* xr = xb + (int64_t)min(row, M - 1) * K + run * 8;
      const KqRun8<XT> a0 = KqRun8<XT>::load(xr, row < M && run < runs);
      const KqRun8<XT> a1 = KqRun8<XT>::load(xr + 8, row < M && run + 1 < runs);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        simdgroup_float8x8 a;
        simdgroup_float8x8 b;
        a.thread_elements()[0] = a0[j];
        a.thread_elements()[1] = a1[j];
        b.thread_elements()[0] = b0[j];
        b.thread_elements()[1] = b1[j];
        simdgroup_multiply_accumulate(acc[f][j % 2], a, b, acc[f][j % 2]);
      }
    }
  }
#pragma unroll
  for (int f = 0; f < F; ++f) {
    const float2 e0 =
        float2(acc[f][0].thread_elements()[0], acc[f][0].thread_elements()[1]);
    const float2 e1 =
        float2(acc[f][1].thread_elements()[0], acc[f][1].thread_elements()[1]);
    parts[(simd_gid * F + f) * 32 + simd_lid] = e0 + e1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int f = int(simd_gid);
  const int row = f * 8 + fm;
  if (f >= F || row >= M) {
    return;
  }
  float2 total = float2(0.0f);
  for (int s = 0; s < KQ_SKINNY_SPLITS; ++s) {
    total += parts[(s * F + f) * 32 + simd_lid];
  }
  device OT* orow = out + ((int64_t)tid.y * M + row) * N;
  if (col < N) {
    orow[col] = static_cast<OT>(total.x);
  }
  if (col + 1 < N) {
    orow[col + 1] = static_cast<OT>(total.y);
  }
}
