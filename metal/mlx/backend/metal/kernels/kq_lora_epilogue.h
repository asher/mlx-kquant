// LoRA epilogue kernels: out += fac * (x @ A[e]) @ B[e], issued as the second
// dispatch of a kq matmul primitive on the output its base route just wrote.
// They read only the float output, so every codec and every base route (qmv,
// verify, split-K, NAX, qmm) carries a live GGUF LoRA adapter at zero extra
// graph ops and no per-codec kernel instantiation. Decode-shaped: one
// threadgroup per (row, N tile of KQ_LORA_TG * KQ_LORA_NPT columns).
//
//   kq_lora_epilogue_rows: one activation row per output row. x [R, K],
//     A [E, K, r], B [E, r, N], out [R, N]. The expert e of a row comes from
//     ids [R] (HAS_IDS) remapped through table (HAS_TABLE; entries < 0 skip
//     the row: a dead arena slot), else e = 0 (dense). fac [R] (HAS_FAC)
//     scales the row's delta (the per-row adapter scale; 0 skips).
//   kq_lora_mix_z + kq_lora_mix_apply: the score-mixed down gather, split
//     around the base dispatch (see below).
//
// Phase 1 forms z = x @ A[e] (r floats per slot) in f32: every thread walks
// K with stride KQ_LORA_TG, accumulating KQ_LORA_RCHUNK ranks in registers
// from A's contiguous [k, :] rows, then simd + threadgroup reduces each rank
// into threadgroup memory. Phase 2 each thread owns KQ_LORA_NPT columns
// (stride KQ_LORA_TG, so B reads coalesce) and accumulates z . B[e][:, n]
// in f32, one round to T at the read-modify-write. Skips are uniform over
// the threadgroup (they depend on the row only), so the early returns and
// the barriers inside kq_lora_form_z stay well-formed.

#define KQ_LORA_TG 1024
#define KQ_LORA_NPT 1
#define KQ_LORA_RCHUNK 16
#define KQ_LORA_UNROLL 16
#define KQ_LORA_UNROLL_R 4
#define KQ_LORA_MAX_NSG (KQ_LORA_TG / 32)
#define KQ_LORA_MAX_Z 512
#define KQ_LORA_MAX_S 16 // <= KQ_LORA_RCHUNK: shares red[]

// The mix apply kernel's slot count is a function constant (one pipeline
// per S): the slot loops unroll with static indexing and the per-slot
// arrays stay in registers.
constant int kq_lora_S [[function_constant(320)]];

#define KQ_LORA_HAS_IDS 1
#define KQ_LORA_HAS_TABLE 2
#define KQ_LORA_HAS_FAC 4

template <typename T>
inline void kq_lora_form_z(
    const device T* xrow,
    const device T* Ae,
    threadgroup float* z,
    threadgroup float* red,
    int K,
    int RANK,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  // The reduction over K is a latency chain per thread, not a bandwidth
  // problem (one row of x, r rows of A): issue KQ_LORA_UNROLL independent
  // loads per round so the chain is K / (KQ_LORA_TG * KQ_LORA_UNROLL)
  // rounds long, one or two at decode widths.
  if (RANK == 1) {
    float acc = 0.0f;
    int k = int(lid);
    for (; k + (KQ_LORA_UNROLL - 1) * KQ_LORA_TG < K;
         k += KQ_LORA_UNROLL * KQ_LORA_TG) {
      float xv[KQ_LORA_UNROLL];
      float av[KQ_LORA_UNROLL];
      for (int u = 0; u < KQ_LORA_UNROLL; u++) {
        xv[u] = float(xrow[k + u * KQ_LORA_TG]);
        av[u] = float(Ae[k + u * KQ_LORA_TG]);
      }
      for (int u = 0; u < KQ_LORA_UNROLL; u++) {
        acc += xv[u] * av[u];
      }
    }
    for (; k < K; k += KQ_LORA_TG) {
      acc += float(xrow[k]) * float(Ae[k]);
    }
    acc = simd_sum(acc);
    if (simd_lid == 0) {
      red[simd_gid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      float g = 0.0f;
      for (int i = 0; i < KQ_LORA_MAX_NSG; i++) {
        g += red[i];
      }
      z[0] = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
  }
  for (int c0 = 0; c0 < RANK; c0 += KQ_LORA_RCHUNK) {
    const int nc = min(KQ_LORA_RCHUNK, RANK - c0);
    float acc[KQ_LORA_RCHUNK];
    for (int j = 0; j < KQ_LORA_RCHUNK; j++) {
      acc[j] = 0.0f;
    }
    int k = int(lid);
    for (; k + (KQ_LORA_UNROLL_R - 1) * KQ_LORA_TG < K;
         k += KQ_LORA_UNROLL_R * KQ_LORA_TG) {
      float xv[KQ_LORA_UNROLL_R];
      for (int u = 0; u < KQ_LORA_UNROLL_R; u++) {
        xv[u] = float(xrow[k + u * KQ_LORA_TG]);
      }
      for (int u = 0; u < KQ_LORA_UNROLL_R; u++) {
        const device T* arow = Ae + (int64_t)(k + u * KQ_LORA_TG) * RANK + c0;
        for (int j = 0; j < KQ_LORA_RCHUNK; j++) {
          if (j < nc) {
            acc[j] += xv[u] * float(arow[j]);
          }
        }
      }
    }
    for (; k < K; k += KQ_LORA_TG) {
      const float xv = float(xrow[k]);
      const device T* arow = Ae + (int64_t)k * RANK + c0;
      for (int j = 0; j < KQ_LORA_RCHUNK; j++) {
        if (j < nc) {
          acc[j] += xv * float(arow[j]);
        }
      }
    }
    for (int j = 0; j < KQ_LORA_RCHUNK; j++) {
      const float v = simd_sum(acc[j]);
      if (j < nc && simd_lid == 0) {
        red[j * KQ_LORA_MAX_NSG + simd_gid] = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < uint(nc)) {
      float g = 0.0f;
      for (int i = 0; i < KQ_LORA_MAX_NSG; i++) {
        g += red[lid * KQ_LORA_MAX_NSG + i];
      }
      z[c0 + lid] = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(KQ_LORA_TG)]] void
kq_lora_epilogue_rows(
    const device T* x [[buffer(0)]],
    const device T* A [[buffer(1)]],
    const device T* B [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device uint32_t* ids [[buffer(4)]],
    const device int32_t* table [[buffer(5)]],
    const device float* fac [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& RANK [[buffer(9)]],
    const constant int& FLAGS [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float z[KQ_LORA_MAX_Z];
  threadgroup float red[KQ_LORA_MAX_NSG * KQ_LORA_RCHUNK];
  const uint lid = lid3.x;
  const int row = int(tid.y);
  int e = 0;
  if (FLAGS & KQ_LORA_HAS_IDS) {
    e = int(ids[row]);
  }
  if (FLAGS & KQ_LORA_HAS_TABLE) {
    e = table[e];
    if (e < 0) {
      return;
    }
  }
  const float f = (FLAGS & KQ_LORA_HAS_FAC) ? fac[row] : 1.0f;
  if (f == 0.0f) {
    return;
  }
  kq_lora_form_z<T>(
      x + (int64_t)row * K,
      A + (int64_t)e * K * RANK,
      z,
      red,
      K,
      RANK,
      lid,
      simd_gid,
      simd_lid);
  const device T* Be = B + (int64_t)e * RANK * N;
  device T* orow = out + (int64_t)row * N;
  const int base = int(tid.x) * (KQ_LORA_TG * KQ_LORA_NPT);
  for (int i = 0; i < KQ_LORA_NPT; i++) {
    const int n = base + int(lid) + i * KQ_LORA_TG;
    if (n >= N) {
      break;
    }
    float acc = 0.0f;
    for (int j = 0; j < RANK; j++) {
      acc += z[j] * float(Be[(int64_t)j * N + n]);
    }
    orow[n] = static_cast<T>(float(orow[n]) + f * acc);
  }
}

// Mix (score-mixed down gather) LoRA, two kernels around the base gather:
//   kq_lora_mix_z runs BEFORE the base dispatch (it reads only x and A, so
//     the encoder lets it overlap the gather): one threadgroup per routed
//     row, z[row, :] = x[row] @ A[e_row] to a float32 scratch [T*S, RANK]
//     (zeros for a dead arena slot).
//   kq_lora_mix_apply runs after it, barrier-free (no threadgroup memory):
//     out[t, n] += sum_s fac[t,s] * scores[t,s] * z[t,s,:] . B[e_s][:, n].
// The serialized cost behind the gather is one tiny dependent kernel, the
// same as the addmm of a plain-op delta, at one graph node instead of three.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(KQ_LORA_TG)]] void kq_lora_mix_z(
    const device T* x [[buffer(0)]],
    const device T* A [[buffer(1)]],
    device float* zout [[buffer(2)]],
    const device uint32_t* ids [[buffer(3)]],
    const device int32_t* table [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& RANK [[buffer(6)]],
    const constant int& FLAGS [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float z[KQ_LORA_MAX_Z];
  threadgroup float red[KQ_LORA_MAX_NSG * KQ_LORA_RCHUNK];
  const uint lid = lid3.x;
  const int row = int(tid.x);
  int e = int(ids[row]);
  if (FLAGS & KQ_LORA_HAS_TABLE) {
    e = table[e];
  }
  device float* zrow = zout + (int64_t)row * RANK;
  if (e < 0) {
    for (int j = int(lid); j < RANK; j += KQ_LORA_TG) {
      zrow[j] = 0.0f;
    }
    return;
  }
  kq_lora_form_z<T>(
      x + (int64_t)row * K,
      A + (int64_t)e * K * RANK,
      z,
      red,
      K,
      RANK,
      lid,
      simd_gid,
      simd_lid);
  for (int j = int(lid); j < RANK; j += KQ_LORA_TG) {
    zrow[j] = z[j];
  }
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(KQ_LORA_TG)]] void
kq_lora_mix_apply(
    const device float* z [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* out [[buffer(2)]],
    const device uint32_t* ids [[buffer(3)]],
    const device int32_t* table [[buffer(4)]],
    const device float* fac [[buffer(5)]],
    const device float* scores [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& RANK [[buffer(8)]],
    const constant int& FLAGS [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  const uint lid = lid3.x;
  const int t = int(tid.y);
  int es[KQ_LORA_MAX_S];
  float ws[KQ_LORA_MAX_S];
  bool any = false;
  for (int s = 0; s < kq_lora_S; s++) {
    const int64_t r = (int64_t)t * kq_lora_S + s;
    int e = int(ids[r]);
    if (FLAGS & KQ_LORA_HAS_TABLE) {
      e = table[e];
    }
    float w = scores[r];
    if (FLAGS & KQ_LORA_HAS_FAC) {
      w *= fac[r];
    }
    if (e < 0) {
      // Dead arena slot: keep the address valid, zero the contribution.
      e = 0;
      w = 0.0f;
    }
    es[s] = e;
    ws[s] = w;
    any = any || (w != 0.0f);
  }
  if (!any) {
    return;
  }
  const device float* zt = z + (int64_t)t * kq_lora_S * RANK;
  device T* orow = out + (int64_t)t * N;
  const int base = int(tid.x) * (KQ_LORA_TG * KQ_LORA_NPT);
  for (int i = 0; i < KQ_LORA_NPT; i++) {
    const int n = base + int(lid) + i * KQ_LORA_TG;
    if (n >= N) {
      break;
    }
    float acc = 0.0f;
    for (int s = 0; s < kq_lora_S; s++) {
      const device T* Be = B + (int64_t)es[s] * RANK * N;
      float a = 0.0f;
      for (int j = 0; j < RANK; j++) {
        a += zt[s * RANK + j] * float(Be[(int64_t)j * N + n]);
      }
      acc += ws[s] * a;
    }
    orow[n] = static_cast<T>(float(orow[n]) + acc);
  }
}

// Densify a strided LoRA operand into a row-major temporary (host side
// kq_lora_view_gpu). Element-size templated: 2-byte (half/bfloat) and 4-byte
// (float/uint32/int32) operands share one kernel each.
struct KqLoraDensifyArgs {
  int shape[8];
  int64_t strides[8];
  int ndim;
  int64_t size;
};

template <typename T>
[[kernel]] void kq_lora_densify(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant KqLoraDensifyArgs& args [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= args.size) {
    return;
  }
  int64_t rem = gid;
  int64_t loc = 0;
  for (int d = args.ndim - 1; d >= 0; --d) {
    int64_t q = rem / args.shape[d];
    loc += (rem - q * args.shape[d]) * args.strides[d];
    rem = q;
  }
  out[gid] = in[loc];
}
