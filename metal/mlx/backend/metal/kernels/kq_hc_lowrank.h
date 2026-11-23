// Fused low-rank hyper-connection kernels for the qwen4exp short-row
// decode route. The eager reference (gmlx qwen4_exp_model.HyperConnection):
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

// Front: grid (LR / 8 + 1, ceil(R / RB)). Threadgroups x < LR / 8 produce
// 8 down rows each (2 simdgroups x 4 rows, values_per_thread 8 -- the
// qmv_fast mapping) for RB rows of xn, decoding each weight once for them;
// the last threadgroup does the 4 float32 inject dots. Each row's sums run
// in the one-row order, so a row's bits do not depend on RB or R.
template <typename T, typename H, int RB>
[[kernel]] void kq_hc_lowrank_front(
    const device T* xn [[buffer(0)]],
    const device uint8_t* w_down [[buffer(1)]],
    const device float* w_inject [[buffer(2)]],
    device H* lo [[buffer(3)]],
    device float* inj [[buffer(4)]],
    const constant int& D [[buffer(5)]],
    const constant int& LR [[buffer(6)]],
    const constant int& R [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
  const uint tid = tid3.x;
  const uint lane = tid % 32;
  const uint sg = tid / 32;
  const int K = KQ_HCLR * D;
  const int r0 = int(tg.y) * RB;
  const int cnt = min(RB, R - r0);
  const device T* xr[RB];
#pragma unroll
  for (int v = 0; v < RB; v++) {
    xr[v] = xn + int64_t(min(r0 + v, R - 1)) * K;
  }

  const uint ndown = uint(LR) / 8;
  if (tg.x < ndown) {
    const int row_bytes = K / KQ_Q8_0_GROUP * KQ_Q8_0_BLOCK_BYTES;
    const int out0 = int(tg.x) * 8 + int(sg) * 4;
    float result[RB][4];
#pragma unroll
    for (int v = 0; v < RB; v++) {
      result[v][0] = result[v][1] = result[v][2] = result[v][3] = 0.0f;
    }
    float x_thread[RB][8];
    for (int k = 0; k < K; k += 256) {
      const int kg = k + int(lane) * 8;
#pragma unroll
      for (int v = 0; v < RB; v++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          x_thread[v][i] = float(xr[v][kg + i]);
        }
      }
      const int block_id = kg / KQ_Q8_0_GROUP;
      const int within = kg - block_id * KQ_Q8_0_GROUP;
      for (int r = 0; r < 4; r++) {
        const device uint8_t* block_addr = w_down +
            int64_t(out0 + r) * row_bytes + block_id * KQ_Q8_0_BLOCK_BYTES;
        const float d = kq_q8_0_d(block_addr);
        const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;
        float q[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          q[i] = float(q_ptr[i]);
        }
#pragma unroll
        for (int v = 0; v < RB; v++) {
          float partial = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            partial = metal::fma(x_thread[v][i], q[i], partial);
          }
          result[v][r] = metal::fma(d, partial, result[v][r]);
        }
      }
    }
#pragma unroll
    for (int v = 0; v < RB; v++) {
      for (int r = 0; r < 4; r++) {
        result[v][r] = simd_sum(result[v][r]);
        if (lane == 0 && v < cnt) {
          // Eager: down out rounds to H (kq qmv promotion), * 1/4 (exact),
          // sigmoid rounds to H, product rounds to H.
          const float t = float(H(result[v][r])) * 0.25f;
          const float sig = float(H(kq_hclr_sigmoid(t)));
          lo[int64_t(r0 + v) * LR + out0 + r] = H(t * sig);
        }
      }
    }
  } else {
    // Inject: 4 float32 rows, 2 per simdgroup, 8 values per lane per
    // chunk (the down-row sweep, float4-pair loads).
    float result[RB][2];
#pragma unroll
    for (int v = 0; v < RB; v++) {
      result[v][0] = result[v][1] = 0.0f;
    }
    for (int k = 0; k < K; k += 256) {
      const int kg = k + int(lane) * 8;
      float x_thread[RB][8];
#pragma unroll
      for (int v = 0; v < RB; v++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          x_thread[v][i] = float(xr[v][kg + i]);
        }
      }
      for (int r = 0; r < 2; r++) {
        const device float* wr = w_inject + int64_t(int(sg) * 2 + r) * K + kg;
        float wv[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          wv[i] = wr[i];
        }
#pragma unroll
        for (int v = 0; v < RB; v++) {
          float partial = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            partial = metal::fma(x_thread[v][i], wv[i], partial);
          }
          result[v][r] += partial;
        }
      }
    }
#pragma unroll
    for (int v = 0; v < RB; v++) {
      for (int r = 0; r < 2; r++) {
        result[v][r] = simd_sum(result[v][r]);
        if (lane == 0 && v < cnt) {
          inj[(r0 + v) * KQ_HCLR + int(sg) * 2 + r] =
              2.0f * kq_hclr_sigmoid(result[v][r] * 0.25f);
        }
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

// Simdgroup epilogue. Up row s * D + d is cut into chunks of four values,
// chunk c (block c / 8) goes to lane c % L of the L lanes of d, a lane adds
// fma(scale, dot4, acc) over its chunks in order with dot4 an fma chain,
// the L partial sums meet in an xor butterfly, and the four streams of a d
// gate and mean in stream order as kq_hc_lowrank_epilogue does. A row's
// sums do not depend on R or on RB.

METAL_FUNC float kq_hclr_dot4(float4 l, float4 w) {
  return metal::fma(
      l.w, w.w, metal::fma(l.z, w.z, metal::fma(l.y, w.y, l.x * w.x)));
}

METAL_FUNC float4 kq_hclr_q4(const device uint8_t* ba, int j) {
  const packed_char4 w4 =
      *((const device packed_char4*)(kq_q8_0_q_ptr(ba) + j * 4));
  return float4(float(w4.x), float(w4.y), float(w4.z), float(w4.w));
}

// Lane (d, q) takes the four stream rows of d for up to RB rows of R, so a
// weight is read and converted once for those rows. The threadgroup
// first stages its rows of lo in threadgroup memory (RB * LR values).
// Threadgroups (32, 4), grid (D / (32 / L) / 4, ceil(R / RB)).
// With FN, buffer 2 holds the streams h and xn is rebuilt as the norm kernel
// writes it, from h, inv [R, 4] (buffer 7) and gamma (buffer 8).
template <typename T, typename H, int L, int RB, bool FN = false>
[[kernel]] void kq_hc_lowrank_epilogue_mr(
    const device H* lo [[buffer(0)]],
    const device uint8_t* w_up [[buffer(1)]],
    const device T* xn [[buffer(2)]],
    device T* mixed [[buffer(3)]],
    const constant int& D [[buffer(4)]],
    const constant int& LR [[buffer(5)]],
    const constant int& R [[buffer(6)]],
    const device float* inv [[buffer(7)]],
    const device H* gamma [[buffer(8)]],
    threadgroup vec<H, 4>* los [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int DPS = 32 / L;
  const int q = int(simd_lid) % L;
  const int dd = int(simd_lid) / L;
  const int d = (int(tg.x) * 4 + int(simd_gid)) * DPS + dd;
  const int K = KQ_HCLR * D;
  const int nch = LR / 4;
  const int row_bytes = LR / KQ_Q8_0_GROUP * KQ_Q8_0_BLOCK_BYTES;
  const int64_t sstride = int64_t(D) * row_bytes;
  const device uint8_t* wrow = w_up + int64_t(d) * row_bytes;
  const int r0 = int(tg.y) * RB;
  const int cnt = min(RB, R - r0);

  for (int i = int(tid); i < cnt * nch; i += 128) {
    los[i] = *(const device vec<H, 4>*)(lo + int64_t(r0) * LR + i * 4);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[RB][KQ_HCLR];
#pragma unroll
  for (int v = 0; v < RB; v++) {
    acc[v][0] = acc[v][1] = acc[v][2] = acc[v][3] = 0.0f;
  }
  for (int c = q; c < nch; c += L) {
    float dsc[KQ_HCLR];
    float4 wq[KQ_HCLR];
#pragma unroll
    for (int s = 0; s < KQ_HCLR; s++) {
      const device uint8_t* ba =
          wrow + s * sstride + (c >> 3) * KQ_Q8_0_BLOCK_BYTES;
      dsc[s] = kq_q8_0_d(ba);
      wq[s] = kq_hclr_q4(ba, c & 7);
    }
#pragma unroll
    for (int v = 0; v < RB; v++) {
      if (v < cnt) {
        const float4 l = float4(los[v * nch + c]);
#pragma unroll
        for (int s = 0; s < KQ_HCLR; s++) {
          acc[v][s] = metal::fma(dsc[s], kq_hclr_dot4(l, wq[s]), acc[v][s]);
        }
      }
    }
  }
#pragma unroll
  for (int v = 0; v < RB; v++) {
    if (v < cnt) {
#pragma unroll
      for (int s = 0; s < KQ_HCLR; s++) {
#pragma unroll
        for (int o = 1; o < L; o *= 2) {
          acc[v][s] += simd_shuffle_xor(acc[v][s], ushort(o));
        }
      }
      if (v % L == q) {
        const int r = r0 + v;
        float m = 0.0f;
#pragma unroll
        for (int s = 0; s < KQ_HCLR; s++) {
          const float g = float(H(kq_hclr_sigmoid(float(H(acc[v][s])))));
          float x = float(xn[int64_t(r) * K + s * D + d]);
          if (FN) {
            x = float(T(x * inv[r * KQ_HCLR + s] * float(gamma[s * D + d])));
          }
          m = metal::fma(g, x, m);
        }
        mixed[int64_t(r) * D + d] = T(m * 0.25f);
      }
    }
  }
}

// Front with the norm folded in, for float32 streams: grid (LR / NO + 1,
// ceil(R / RB)), threadgroups of 4 * SPS simdgroups, SPS on each stream,
// each on every SPS-th run of 256 inputs. A simdgroup sums its share of
// the stream's squares and of NO down dots of h * gamma for RB rows; the
// threadgroup adds a stream's SPS shares in order, scales the stream's dots
// by its 1 / rms and adds the four streams in stream order. The last
// threadgroup does the 4 float32 inject dots the same way and writes inv
// [R, 4] for the epilogue. Every threadgroup sums a stream's squares in the
// same order, so all of them use the bits of inv it writes. A row's bits do
// not depend on RB or R.
template <typename H, int RB, int NO, int SPS = 1>
[[kernel]] void kq_hc_lowrank_front_fn(
    const device float* h [[buffer(0)]],
    const device H* gamma [[buffer(1)]],
    const device uint8_t* w_down [[buffer(2)]],
    const device float* w_inject [[buffer(3)]],
    device H* lo [[buffer(4)]],
    device float* inj [[buffer(5)]],
    device float* inv_out [[buffer(6)]],
    const constant int& D [[buffer(7)]],
    const constant int& LR [[buffer(8)]],
    const constant int& R [[buffer(9)]],
    const constant float& eps [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int s = int(simd_gid) / SPS;
  const int sp = int(simd_gid) % SPS;
  const int K = KQ_HCLR * D;
  const int r0 = int(tg.y) * RB;
  const int cnt = min(RB, R - r0);
  const bool down = int(tg.x) < LR / NO;
  const int out0 = int(tg.x) * NO;
  const int row_bytes = K / KQ_Q8_0_GROUP * KQ_Q8_0_BLOCK_BYTES;
  const device float* xr[RB];
#pragma unroll
  for (int v = 0; v < RB; v++) {
    xr[v] = h + int64_t(min(r0 + v, R - 1)) * K + s * D;
  }
  const device H* gs = gamma + s * D;

  float ssq[RB];
  float result[RB][NO];
#pragma unroll
  for (int v = 0; v < RB; v++) {
    ssq[v] = 0.0f;
#pragma unroll
    for (int r = 0; r < NO; r++) {
      result[v][r] = 0.0f;
    }
  }
  for (int kk = sp * 256 + int(simd_lid) * 8; kk < D; kk += 256 * SPS) {
    float g[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      g[i] = float(gs[kk + i]);
    }
    float x[RB][8];
#pragma unroll
    for (int v = 0; v < RB; v++) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const float hv = xr[v][kk + i];
        ssq[v] = metal::fma(hv, hv, ssq[v]);
        x[v][i] = hv * g[i];
      }
    }
    const int kg = s * D + kk;
    if (down) {
      const int block_id = kg / KQ_Q8_0_GROUP;
      const int within = kg - block_id * KQ_Q8_0_GROUP;
#pragma unroll
      for (int r = 0; r < NO; r++) {
        const device uint8_t* ba = w_down + int64_t(out0 + r) * row_bytes +
            block_id * KQ_Q8_0_BLOCK_BYTES;
        const float dq = kq_q8_0_d(ba);
        const device int8_t* qp = kq_q8_0_q_ptr(ba) + within;
        float q[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          q[i] = float(qp[i]);
        }
#pragma unroll
        for (int v = 0; v < RB; v++) {
          float partial = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            partial = metal::fma(x[v][i], q[i], partial);
          }
          result[v][r] = metal::fma(dq, partial, result[v][r]);
        }
      }
    } else {
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const device float* wr = w_inject + int64_t(r) * K + kg;
#pragma unroll
        for (int v = 0; v < RB; v++) {
          float partial = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            partial = metal::fma(x[v][i], wr[i], partial);
          }
          result[v][r] += partial;
        }
      }
    }
  }

  threadgroup float part[KQ_HCLR * SPS][RB][NO];
  threadgroup float sq[KQ_HCLR * SPS][RB];
#pragma unroll
  for (int v = 0; v < RB; v++) {
    ssq[v] = simd_sum(ssq[v]);
#pragma unroll
    for (int r = 0; r < NO; r++) {
      result[v][r] = simd_sum(result[v][r]);
    }
    if (simd_lid == 0) {
      sq[simd_gid][v] = ssq[v];
#pragma unroll
      for (int r = 0; r < NO; r++) {
        part[simd_gid][v][r] = result[v][r];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid != 0) {
    return;
  }
  // Lane o takes row r0 + o / NO, output o % NO (the inject rows: o % NO
  // under 4).
  for (int o = int(simd_lid); o < cnt * NO; o += 32) {
    const int v = o / NO;
    const int r = o % NO;
    if (!down && r >= 4) {
      continue;
    }
    float tot = 0.0f;
    for (int t = 0; t < KQ_HCLR; t++) {
      float ss = sq[t * SPS][v];
      float pt = part[t * SPS][v][r];
      for (int u = 1; u < SPS; u++) {
        ss += sq[t * SPS + u][v];
        pt += part[t * SPS + u][v][r];
      }
      const float iv = metal::precise::rsqrt(ss / float(D) + eps);
      tot = metal::fma(iv, pt, tot);
      if (!down && r == 0) {
        inv_out[(r0 + v) * KQ_HCLR + t] = iv;
      }
    }
    if (down) {
      const float t = float(H(tot)) * 0.25f;
      const float sig = float(H(kq_hclr_sigmoid(t)));
      lo[int64_t(r0 + v) * LR + out0 + r] = H(t * sig);
    } else {
      inj[(r0 + v) * KQ_HCLR + r] = 2.0f * kq_hclr_sigmoid(tot * 0.25f);
    }
  }
}

// Folded-norm front on simdgroup 8x8 MMA, all rows as one operand: grid
// (LR / 8 + 1), threadgroups of eight simdgroups, two per stream, each on
// half of its stream's 64-input steps. A threadgroup takes 8 down columns
// for up to 8 F rows (the last one the 4 inject rows), with the lane map of
// kq_skinny_mma: lane (fm, fn) loads h * gamma of row fm at runs fn and
// fn + 1 of the step as A, and columns fn and fn + 1 of the weight at run
// fm as B. A lane also sums the squares of the h it loads; the four lanes
// of a row and the two simdgroups of a stream add in a fixed order, so
// every threadgroup holds the same 1 / rms, which the inject threadgroup
// writes for the epilogue. A row's bits do not depend on R.
struct KqHcRun8 {
  float4 lo;
  float4 hi;
  METAL_FUNC float operator[](int i) const thread {
    return i < 4 ? lo[i] : hi[i - 4];
  }
};

template <typename U>
METAL_FUNC KqHcRun8 kq_hc_run8(const device U* p) {
  using U4 = vec<U, 4>;
  return {float4(*(const device U4*)p), float4(*(const device U4*)(p + 4))};
}

// Run `run` (8 values) of a q8_0 row, scaled.
METAL_FUNC KqHcRun8 kq_hc_q8_run8(const device uint8_t* row, int run) {
  const device uint8_t* ba = row + (run >> 2) * KQ_Q8_0_BLOCK_BYTES;
  const float d = kq_q8_0_d(ba);
  const device int8_t* q = kq_q8_0_q_ptr(ba) + (run & 3) * 8;
  const packed_char4 a = *(const device packed_char4*)q;
  const packed_char4 b = *(const device packed_char4*)(q + 4);
  return {
      d * float4(float(a.x), float(a.y), float(a.z), float(a.w)),
      d * float4(float(b.x), float(b.y), float(b.z), float(b.w))};
}

template <typename H, int F>
[[kernel]] void kq_hc_lowrank_front_mma(
    const device float* h [[buffer(0)]],
    const device H* gamma [[buffer(1)]],
    const device uint8_t* w_down [[buffer(2)]],
    const device float* w_inject [[buffer(3)]],
    device H* lo [[buffer(4)]],
    device float* inj [[buffer(5)]],
    device float* inv_out [[buffer(6)]],
    const constant int& D [[buffer(7)]],
    const constant int& LR [[buffer(8)]],
    const constant int& R [[buffer(9)]],
    const constant float& eps [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float2 parts[8 * F * 32];
  threadgroup float sq[8 * F * 32];
  const short qid = short(simd_lid / 4);
  const short fm = (qid & 4) + short((simd_lid / 2) % 4);
  const short fn = (qid & 2) * 2 + short(simd_lid % 2) * 2;
  const int K = KQ_HCLR * D;
  const int s = int(simd_gid) / 2;
  const int spd = D / 64; // steps a stream
  const int first = (int(simd_gid) & 1) * (spd / 2);
  const int last = (int(simd_gid) & 1) ? spd : spd / 2;
  const bool down = int(tg.x) < LR / 8;
  const int col = int(tg.x) * 8 + fn;
  const int row_bytes = K / KQ_Q8_0_GROUP * KQ_Q8_0_BLOCK_BYTES;
  const int srun = s * (D / 8); // the stream's first run
  const device uint8_t* q0 = w_down + int64_t(down ? col : 0) * row_bytes;
  const device uint8_t* q1 = q0 + (down ? row_bytes : 0);
  // Inject columns past 3 read row 3 and are not written.
  const device float* i0 = w_inject + int64_t(min(int(fn), 3)) * K;
  const device float* i1 = w_inject + int64_t(min(int(fn) + 1, 3)) * K;
  const device H* gs = gamma + s * D;

  simdgroup_float8x8 acc[F][2];
  float ssq[F];
#pragma unroll
  for (int f = 0; f < F; ++f) {
    acc[f][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[f][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    ssq[f] = 0.0f;
  }
  KqHcRun8 nb0;
  KqHcRun8 nb1;
  {
    const int run = srun + first * 8 + fm;
    if (down) {
      nb0 = kq_hc_q8_run8(q0, run);
      nb1 = kq_hc_q8_run8(q1, run);
    } else {
      nb0 = kq_hc_run8(i0 + run * 8);
      nb1 = kq_hc_run8(i1 + run * 8);
    }
  }
  for (int step = first; step < last; ++step) {
    const KqHcRun8 b0 = nb0;
    const KqHcRun8 b1 = nb1;
    if (step + 1 < last) {
      const int run = srun + (step + 1) * 8 + fm;
      if (down) {
        nb0 = kq_hc_q8_run8(q0, run);
        nb1 = kq_hc_q8_run8(q1, run);
      } else {
        nb0 = kq_hc_run8(i0 + run * 8);
        nb1 = kq_hc_run8(i1 + run * 8);
      }
    }
    const int k = (step * 8 + fn) * 8; // within the stream
    const KqHcRun8 g0 = kq_hc_run8(gs + k);
    const KqHcRun8 g1 = kq_hc_run8(gs + k + 8);
#pragma unroll
    for (int f = 0; f < F; ++f) {
      const int row = f * 8 + fm;
      KqHcRun8 a0 = {float4(0.0f), float4(0.0f)};
      KqHcRun8 a1 = a0;
      if (row < R) {
        const device float* xr = h + int64_t(row) * K + s * D + k;
        a0 = kq_hc_run8(xr);
        a1 = kq_hc_run8(xr + 8);
      }
      ssq[f] += metal::dot(a0.lo, a0.lo) + metal::dot(a0.hi, a0.hi) +
          metal::dot(a1.lo, a1.lo) + metal::dot(a1.hi, a1.hi);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        simdgroup_float8x8 a;
        simdgroup_float8x8 b;
        a.thread_elements()[0] = a0[j] * g0[j];
        a.thread_elements()[1] = a1[j] * g1[j];
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
    sq[(simd_gid * F + f) * 32 + simd_lid] = ssq[f];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int f = int(simd_gid);
  const int row = f * 8 + fm;
  if (f >= F || row >= R) {
    return;
  }
  // The four lanes of row fm: lane bits 0 and 3 free.
  const uint l0 = simd_lid & ~9u;
  float2 total = float2(0.0f);
  for (int t = 0; t < KQ_HCLR; ++t) {
    float ss = 0.0f;
    float2 p = float2(0.0f);
    for (int u = 0; u < 2; ++u) {
      const uint base = uint((t * 2 + u) * F + f) * 32;
      ss += sq[base + l0] + sq[base + (l0 | 1u)] + sq[base + (l0 | 8u)] +
          sq[base + (l0 | 9u)];
      p += parts[base + simd_lid];
    }
    const float iv = metal::precise::rsqrt(ss / float(D) + eps);
    total += iv * p;
    if (!down && simd_lid == l0) {
      inv_out[row * KQ_HCLR + t] = iv;
    }
  }
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    if (down) {
      const float t = float(H(total[c])) * 0.25f;
      const float sig = float(H(kq_hclr_sigmoid(t)));
      lo[int64_t(row) * LR + col + c] = H(t * sig);
    } else if (fn + c < KQ_HCLR) {
      inj[row * KQ_HCLR + fn + c] = 2.0f * kq_hclr_sigmoid(total[c] * 0.25f);
    }
  }
}

// Epilogue of the folded-norm mixer on simdgroup 8x8 MMA, all rows as one
// operand: a simdgroup takes two d for the four streams (its 8 up rows) and
// up to 8 F rows of lo, with the lane map of kq_skinny_mma. Lane (fm, fn)
// holds row fm, stream fn / 2, both d; the four streams of a row meet by
// two xor shuffles. Grid (D / 8), threadgroups of four simdgroups. Needs
// LR % 64 == 0. A row's bits do not depend on R.
template <typename H, int F>
[[kernel]] void kq_hc_lowrank_epilogue_mma(
    const device H* lo [[buffer(0)]],
    const device uint8_t* w_up [[buffer(1)]],
    const device float* h [[buffer(2)]],
    device float* mixed [[buffer(3)]],
    const constant int& D [[buffer(4)]],
    const constant int& LR [[buffer(5)]],
    const constant int& R [[buffer(6)]],
    const device float* inv [[buffer(7)]],
    const device H* gamma [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const short qid = short(simd_lid / 4);
  const short fm = (qid & 4) + short((simd_lid / 2) % 4);
  const short fn = (qid & 2) * 2 + short(simd_lid % 2) * 2;
  const int K = KQ_HCLR * D;
  const int st = fn / 2;
  const int d0 = (int(tg.x) * 4 + int(simd_gid)) * 2;
  const int row_bytes = LR / KQ_Q8_0_GROUP * KQ_Q8_0_BLOCK_BYTES;
  const device uint8_t* w0 = w_up + int64_t(st * D + d0) * row_bytes;
  const device uint8_t* w1 = w0 + row_bytes;
  const int steps = LR / 64;

  simdgroup_float8x8 acc[F][2];
#pragma unroll
  for (int f = 0; f < F; ++f) {
    acc[f][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[f][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }
  KqHcRun8 nb0 = kq_hc_q8_run8(w0, fm);
  KqHcRun8 nb1 = kq_hc_q8_run8(w1, fm);
  for (int step = 0; step < steps; ++step) {
    const KqHcRun8 b0 = nb0;
    const KqHcRun8 b1 = nb1;
    if (step + 1 < steps) {
      nb0 = kq_hc_q8_run8(w0, (step + 1) * 8 + fm);
      nb1 = kq_hc_q8_run8(w1, (step + 1) * 8 + fm);
    }
    const int k = (step * 8 + fn) * 8;
#pragma unroll
    for (int f = 0; f < F; ++f) {
      const int row = f * 8 + fm;
      KqHcRun8 a0 = {float4(0.0f), float4(0.0f)};
      KqHcRun8 a1 = a0;
      if (row < R) {
        const device H* lr = lo + int64_t(row) * LR + k;
        a0 = kq_hc_run8(lr);
        a1 = kq_hc_run8(lr + 8);
      }
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
    const int row = min(f * 8 + int(fm), R - 1);
    const float iv = inv[row * KQ_HCLR + st];
    float2 m;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float up =
          acc[f][0].thread_elements()[c] + acc[f][1].thread_elements()[c];
      const float g = float(H(kq_hclr_sigmoid(float(H(up)))));
      const int kk = st * D + d0 + c;
      m[c] = g * (h[int64_t(row) * K + kk] * iv * float(gamma[kk]));
    }
    m += simd_shuffle_xor(m, ushort(1));
    m += simd_shuffle_xor(m, ushort(8));
    if (fn == 0 && f * 8 + int(fm) < R) {
      mixed[int64_t(row) * D + d0] = m.x * 0.25f;
      mixed[int64_t(row) * D + d0 + 1] = m.y * 0.25f;
    }
  }
}
