// Fused low-rank hyper-connection kernels for the qwen4exp M<=8 decode
// route. The eager reference (gmlx qwen4_exp_model.HyperConnection):
//
//   xn    = rms_norm(h, None, eps) * gamma        h [R, 4, D]
//   lo    = silu(down(xn.flat) / 4)               down: q8_0 wire, LR rows
//   gate  = sigmoid(up(lo))                       up: q8_0 wire, 4 * D rows
//   mixed = mean_s(gate * xn)                     [R, D]
//   inj   = 2 * sigmoid(inject(xn.flat) / 4)      inject: float32 [4, 4 * D]
//
// Three dispatches replace the eager op chain:
//   kq_hc_lowrank_norm:     per-stream rms factors + the gamma product,
//                           materialized once as xn (one threadgroup per
//                           row; a per-GEMV-threadgroup recompute was
//                           measured 4x slower serialized -- 41 under-
//                           occupied threadgroups each re-reading h).
//   kq_hc_lowrank_front:    down qmv over xn with silu(x / 4) plus the
//                           float32 inject dots (qmv_fast mapping).
//   kq_hc_lowrank_epilogue: up qmv, sigmoid gate, mean over the streams.
//
// Numerics mirror the eager path: f32 accumulate everywhere; the down/up
// dots round to H (the kq f32->bf16 qmv promotion) before their
// activation, exactly as the eager KQuantLinear outputs do. T is the
// residual dtype (float32 in steady state -- the f32 inject pin promotes
// the residual at the first combine -- or half at layer 0); H is the
// activation half dtype (gamma / lo). When T is half it must equal H, and
// xn takes the eager double rounding H(H(h * inv) * gamma).

#define KQ_HCLR 4
#define KQ_HCLR_MAX_LRBLK 16

METAL_FUNC float kq_hclr_sigmoid(float x) {
  return 1.0f / (1.0f + metal::precise::exp(-x));
}

// One threadgroup of 256 per (stream, row): the stream sum of squares,
// then xn = h * inv * gamma over that stream with the eager rounding
// points.
template <typename T, typename H>
[[kernel]] void kq_hc_lowrank_norm(
    const device T* h [[buffer(0)]],
    const device H* gamma [[buffer(1)]],
    device T* xn [[buffer(2)]],
    const constant int& D [[buffer(3)]],
    const constant float& eps [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
  const uint tid = tid3.x;
  const uint lane = tid % 32;
  const uint sg = tid / 32;
  const int s = int(tg.x);
  const device T* hs = h + int64_t(tg.y) * KQ_HCLR * D + s * D;
  const device H* gs = gamma + s * D;
  device T* xs = xn + int64_t(tg.y) * KQ_HCLR * D + s * D;

  float acc = 0.0f;
  for (int k = int(tid); k < D; k += 256) {
    float v = float(hs[k]);
    acc = metal::fma(v, v, acc);
  }
  threadgroup float part[8];
  acc = simd_sum(acc);
  if (lane == 0) {
    part[sg] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Every thread reduces the 8 partials in the same order: one barrier,
  // no threadgroup scalar to publish.
  float ssq = 0.0f;
  for (int i = 0; i < 8; ++i) {
    ssq += part[i];
  }
  const float inv = metal::precise::rsqrt(ssq / float(D) + eps);
  for (int k = int(tid); k < D; k += 256) {
    if (sizeof(T) == 4) { // float32 residual
      xs[k] = T(float(hs[k]) * inv * float(gs[k]));
    } else { // eager double rounding at half
      H t = H(float(hs[k]) * inv);
      xs[k] = T(H(float(t) * float(gs[k])));
    }
  }
}

// Front: grid (LR / 8 + 1, R). Threadgroups x < LR / 8 produce 8 down rows
// each (2 simdgroups x 4 rows, values_per_thread 8 -- the qmv_fast
// mapping); the last threadgroup does the 4 float32 inject dots.
template <typename T, typename H>
[[kernel]] void kq_hc_lowrank_front(
    const device T* xn [[buffer(0)]],
    const device uint8_t* w_down [[buffer(1)]],
    const device float* w_inject [[buffer(2)]],
    device H* lo [[buffer(3)]],
    device float* inj [[buffer(4)]],
    const constant int& D [[buffer(5)]],
    const constant int& LR [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
  const uint tid = tid3.x;
  const uint lane = tid % 32;
  const uint sg = tid / 32;
  const int K = KQ_HCLR * D;
  const uint row = tg.y;
  const device T* xr = xn + int64_t(row) * K;

  const uint ndown = uint(LR) / 8;
  if (tg.x < ndown) {
    const int row_bytes = K / KQ_Q8_0_GROUP * KQ_Q8_0_BLOCK_BYTES;
    const int out0 = int(tg.x) * 8 + int(sg) * 4;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x_thread[8];
    for (int k = 0; k < K; k += 256) {
      const int kg = k + int(lane) * 8;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        x_thread[i] = float(xr[kg + i]);
      }
      const int block_id = kg / KQ_Q8_0_GROUP;
      const int within = kg - block_id * KQ_Q8_0_GROUP;
      for (int r = 0; r < 4; r++) {
        const device uint8_t* block_addr = w_down +
            int64_t(out0 + r) * row_bytes + block_id * KQ_Q8_0_BLOCK_BYTES;
        const float d = kq_q8_0_d(block_addr);
        const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;
        float partial = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          partial = metal::fma(x_thread[i], float(q_ptr[i]), partial);
        }
        result[r] = metal::fma(d, partial, result[r]);
      }
    }
    for (int r = 0; r < 4; r++) {
      result[r] = simd_sum(result[r]);
      if (lane == 0) {
        // Eager: down out rounds to H (kq qmv promotion), * 1/4 (exact),
        // sigmoid rounds to H, product rounds to H.
        const float t = float(H(result[r])) * 0.25f;
        const float sig = float(H(kq_hclr_sigmoid(t)));
        lo[int64_t(row) * LR + out0 + r] = H(t * sig);
      }
    }
  } else {
    // Inject: 4 float32 rows, 2 per simdgroup, 8 values per lane per
    // chunk (the down-row sweep, float4-pair loads).
    float result[2] = {0.0f, 0.0f};
    for (int k = 0; k < K; k += 256) {
      const int kg = k + int(lane) * 8;
      float x_thread[8];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        x_thread[i] = float(xr[kg + i]);
      }
      for (int r = 0; r < 2; r++) {
        const device float* wr = w_inject + int64_t(int(sg) * 2 + r) * K + kg;
        float partial = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          partial = metal::fma(x_thread[i], wr[i], partial);
        }
        result[r] += partial;
      }
    }
    for (int r = 0; r < 2; r++) {
      result[r] = simd_sum(result[r]);
      if (lane == 0) {
        inj[row * KQ_HCLR + int(sg) * 2 + r] =
            2.0f * kq_hclr_sigmoid(result[r] * 0.25f);
      }
    }
  }
}

// Epilogue: one lane per output (row, d), 256-thread threadgroups over
// R * D outputs. The up dot has K = LR (tiny); a simdgroup-per-output
// mapping spends most of the machine on index math and the 5-shuffle
// simd_sum after ~LR / 32 fmas per lane, which made the kernel scale
// with R even though the wire is cache-resident. A serial per-lane dot
// amortizes that overhead: each lane walks its 4 up rows (s * D + d)
// block by block with packed_char4 loads, keeping the 4 stream
// accumulators as independent chains, then gates xn and means over the
// streams. No cross-lane reduction, so batching rows cannot move a bit
// relative to single-row calls.
template <typename T, typename H>
[[kernel]] void kq_hc_lowrank_epilogue(
    const device H* lo [[buffer(0)]],
    const device uint8_t* w_up [[buffer(1)]],
    const device T* xn [[buffer(2)]],
    device T* mixed [[buffer(3)]],
    const constant int& D [[buffer(4)]],
    const constant int& LR [[buffer(5)]],
    const constant int& R [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(R) * uint(D)) {
    return;
  }
  const int d = int(gid % uint(D));
  const int r = int(gid / uint(D));
  const int K = KQ_HCLR * D;
  const int nblk = LR / 32;
  const int row_bytes = LR / KQ_Q8_0_GROUP * KQ_Q8_0_BLOCK_BYTES;

  const device H* lor = lo + int64_t(r) * LR;
  float acc[KQ_HCLR] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int it = 0; it < nblk; it++) {
    const device H* lo_blk = lor + it * 32;
    float dsc[KQ_HCLR];
    const device uint8_t* ba[KQ_HCLR];
    for (int s = 0; s < KQ_HCLR; s++) {
      ba[s] = w_up + int64_t(s * D + d) * row_bytes + it * KQ_Q8_0_BLOCK_BYTES;
      dsc[s] = kq_q8_0_d(ba[s]);
    }
    for (int j = 0; j < 8; j++) {
      float l0 = float(lo_blk[j * 4 + 0]);
      float l1 = float(lo_blk[j * 4 + 1]);
      float l2 = float(lo_blk[j * 4 + 2]);
      float l3 = float(lo_blk[j * 4 + 3]);
      for (int s = 0; s < KQ_HCLR; s++) {
        const packed_char4 q =
            *((const device packed_char4*)(kq_q8_0_q_ptr(ba[s]) + j * 4));
        acc[s] = metal::fma(
            dsc[s],
            l0 * float(q.x) + l1 * float(q.y) + l2 * float(q.z) +
                l3 * float(q.w),
            acc[s]);
      }
    }
  }
  float m = 0.0f;
  for (int s = 0; s < KQ_HCLR; s++) {
    // Eager: up out rounds to H, sigmoid rounds to H, then f32 gate * xn.
    const float g = float(H(kq_hclr_sigmoid(float(H(acc[s])))));
    m = metal::fma(g, float(xn[int64_t(r) * K + s * D + d]), m);
  }
  mixed[int64_t(r) * D + d] = T(m * 0.25f);
}
