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
//     Row counts above KQ_LORA_SPLIT_MAX_ROWS (host side) take this kernel.
//   kq_lora_rows_z + kq_lora_rows_apply: the same rows semantics split
//     around the base dispatch for up to KQ_LORA_SPLIT_MAX_ROWS rows (see
//     below).
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

// Split rows LoRA, two kernels around the base route of a dense op:
//   kq_lora_rows_z runs BEFORE the base dispatch. It reads only x and A, so
//     the encoder overlaps it with the base matvec. One threadgroup per
//     (K chunk, block of RB rows) writes the partial
//     zpart[row, chunk, :] = x[row, chunk] @ A[e][chunk, :] in f32.
//   kq_lora_rows_apply runs after the base route. One threadgroup per
//     (N tile of 32 * CPT columns, block of RB rows) sums the chunk
//     partials into z and adds fac * z . B[e][:, n] to each live row of out.
// In a row block, the first live row of each entry loads that entry's A or
// B values and every row of the entry reuses them, so a block reads each
// entry once. Skipped rows write no partial, are never read by the apply
// and keep the base output. Skips are uniform per threadgroup, so every
// barrier is reached by all threads.
#define KQ_LORA_SPLIT_TG 256
#define KQ_LORA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#define KQ_LORA_Z_UNROLL 8
#define KQ_LORA_APPLY_NSG (KQ_LORA_SPLIT_TG / 32)

inline int kq_lora_row_entry(
    int row,
    const device uint32_t* ids,
    const device int32_t* table,
    const device float* fac,
    int FLAGS,
    thread float& f) {
  int e = (FLAGS & KQ_LORA_HAS_IDS) ? int(ids[row]) : 0;
  if (FLAGS & KQ_LORA_HAS_TABLE) {
    e = table[e];
  }
  f = (FLAGS & KQ_LORA_HAS_FAC) ? fac[row] : 1.0f;
  return (e < 0 || f == 0.0f) ? -1 : e;
}

// Entries of a block of RB rows starting at r0 (-1 = skip) and whether each
// row is the first live row of its entry. Returns whether any row is live.
template <int RB>
inline bool kq_lora_block_entries(
    int r0,
    int R,
    const device uint32_t* ids,
    const device int32_t* table,
    const device float* fac,
    int FLAGS,
    thread int* es,
    thread float* fs,
    thread bool* lead) {
  bool any = false;
  KQ_LORA_UNROLL_FULL
  for (int i = 0; i < RB; i++) {
    fs[i] = 0.0f;
    es[i] = (r0 + i < R)
        ? kq_lora_row_entry(r0 + i, ids, table, fac, FLAGS, fs[i])
        : -1;
    any = any || es[i] >= 0;
  }
  KQ_LORA_UNROLL_FULL
  for (int i = 0; i < RB; i++) {
    lead[i] = es[i] >= 0;
    for (int q = 0; q < i; q++) {
      lead[i] = lead[i] && es[q] != es[i];
    }
  }
  return any;
}

// Each thread takes VW adjacent ranks of an A row (RANK % VW == 0; vector
// loads when VW is 4 or 8) and every lanes-th k of the chunk, issuing
// KQ_LORA_Z_UNROLL rows of loads per round; the k lanes meet in
// threadgroup memory.
template <typename T, int VW, int RB>
[[kernel, max_total_threads_per_threadgroup(KQ_LORA_SPLIT_TG)]] void
kq_lora_rows_z(
    const device T* x [[buffer(0)]],
    const device T* A [[buffer(1)]],
    device float* zpart [[buffer(2)]],
    const device uint32_t* ids [[buffer(3)]],
    const device int32_t* table [[buffer(4)]],
    const device float* fac [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& RANK [[buffer(7)]],
    const constant int& FLAGS [[buffer(8)]],
    const constant int& KC [[buffer(9)]],
    const constant int& R [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tgs [[threadgroups_per_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  constexpr int U = KQ_LORA_Z_UNROLL;
  threadgroup float red[KQ_LORA_SPLIT_TG * VW];
  const int lid = int(lid3.x);
  const int chunk = int(tid.x);
  const int C = int(tgs.x);
  const int r0 = int(tid.y) * RB;
  int es[RB];
  float fs[RB];
  bool lead[RB];
  if (!kq_lora_block_entries<RB>(r0, R, ids, table, fac, FLAGS, es, fs, lead)) {
    return;
  }
  const int k0 = chunk * KC;
  const int k1 = min(K, k0 + KC);
  const int JW = RANK / VW;
  const int JS = min(JW, KQ_LORA_SPLIT_TG);
  const int lanes = KQ_LORA_SPLIT_TG / JS;
  const int kl = lid / JS;
  const bool active = kl < lanes;
  for (int jb = 0; jb < JW; jb += JS) {
    const int jg = jb + lid % JS;
    const bool on = active && jg < JW;
    float acc[RB][VW];
    KQ_LORA_UNROLL_FULL
    for (int i = 0; i < RB; i++) {
      KQ_LORA_UNROLL_FULL
      for (int v = 0; v < VW; v++) {
        acc[i][v] = 0.0f;
      }
    }
    auto step = [&](int k, int nk) {
      // nk <= U consecutive lanes' k values: k, k + lanes, ...
      float xv[RB][U];
      KQ_LORA_UNROLL_FULL
      for (int i = 0; i < RB; i++) {
        KQ_LORA_UNROLL_FULL
        for (int u = 0; u < U; u++) {
          xv[i][u] = (es[i] >= 0 && u < nk)
              ? float(x[(int64_t)(r0 + i) * K + k + u * lanes])
              : 0.0f;
        }
      }
      KQ_LORA_UNROLL_FULL
      for (int i = 0; i < RB; i++) {
        if (!lead[i]) {
          continue;
        }
        float a[U][VW];
        KQ_LORA_UNROLL_FULL
        for (int u = 0; u < U; u++) {
          const device T* arow =
              A + ((int64_t)es[i] * K + k + u * lanes) * RANK + jg * VW;
          if (u >= nk) {
            KQ_LORA_UNROLL_FULL
            for (int v = 0; v < VW; v++) {
              a[u][v] = 0.0f;
            }
          } else if (VW % 4 == 0) {
            KQ_LORA_UNROLL_FULL
            for (int v = 0; v < VW; v += 4) {
              const vec<T, 4> a4 = *(const device vec<T, 4>*)(arow + v);
              KQ_LORA_UNROLL_FULL
              for (int q = 0; q < 4; q++) {
                a[u][v + q] = float(a4[q]);
              }
            }
          } else {
            KQ_LORA_UNROLL_FULL
            for (int v = 0; v < VW; v++) {
              a[u][v] = float(arow[v]);
            }
          }
        }
        KQ_LORA_UNROLL_FULL
        for (int q = i; q < RB; q++) {
          if (es[q] != es[i]) {
            continue;
          }
          KQ_LORA_UNROLL_FULL
          for (int u = 0; u < U; u++) {
            KQ_LORA_UNROLL_FULL
            for (int v = 0; v < VW; v++) {
              acc[q][v] += xv[q][u] * a[u][v];
            }
          }
        }
      }
    };
    if (on) {
      int k = k0 + kl;
      for (; k + (U - 1) * lanes < k1; k += U * lanes) {
        step(k, U);
      }
      if (k < k1) {
        step(k, (k1 - k + lanes - 1) / lanes);
      }
    }
    const int nj = min(JS, JW - jb) * VW;
    KQ_LORA_UNROLL_FULL
    for (int i = 0; i < RB; i++) {
      if (es[i] < 0) {
        continue;
      }
      KQ_LORA_UNROLL_FULL
      for (int v = 0; v < VW; v++) {
        red[lid * VW + v] = acc[i][v];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int t = lid; t < nj; t += KQ_LORA_SPLIT_TG) {
        const int g = t / VW;
        const int v = t % VW;
        float sum = 0.0f;
        for (int l = 0; l < lanes; l++) {
          sum += red[(l * JS + g) * VW + v];
        }
        zpart[((int64_t)(r0 + i) * C + chunk) * RANK + jb * VW + t] = sum;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// Each simdgroup of the apply takes every NSG-th rank and each lane CPT
// adjacent columns (vector loads when V4), so a tile is 32 * CPT columns.
// A round covers 8 ranks per simdgroup: lane l sums every fourth chunk
// partial of rank l % 8, two xor shuffles finish the sums and a broadcast
// shuffle hands each rank's z to every lane, so z needs no threadgroup
// memory and its loads fly with the B loads. The simdgroups' column sums
// meet in threadgroup memory.
template <typename T, bool V4, int RB, int CPT>
[[kernel, max_total_threads_per_threadgroup(KQ_LORA_SPLIT_TG)]] void
kq_lora_rows_apply(
    const device float* zpart [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* out [[buffer(2)]],
    const device uint32_t* ids [[buffer(3)]],
    const device int32_t* table [[buffer(4)]],
    const device float* fac [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& RANK [[buffer(7)]],
    const constant int& FLAGS [[buffer(8)]],
    const constant int& C [[buffer(9)]],
    const constant int& R [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NSG = KQ_LORA_APPLY_NSG;
  constexpr int TN = 32 * CPT;
  constexpr int U = 8;
  threadgroup float red[NSG * TN];
  const int lid = int(lid3.x);
  const int r0 = int(tid.y) * RB;
  int es[RB];
  float fs[RB];
  bool lead[RB];
  if (!kq_lora_block_entries<RB>(r0, R, ids, table, fac, FLAGS, es, fs, lead)) {
    return;
  }
  // The final pass: thread t < TN owns column t of every row.
  const int nf = int(tid.x) * TN + lid;
  float of[RB];
  KQ_LORA_UNROLL_FULL
  for (int i = 0; i < RB; i++) {
    of[i] = (lid < TN && nf < N && es[i] >= 0)
        ? float(out[(int64_t)(r0 + i) * N + nf])
        : 0.0f;
  }
  const int n0 = int(tid.x) * TN + int(lane) * CPT;
  const int ju = int(lane) % U;
  const int cg = int(lane) / U;
  float acc[RB][CPT];
  KQ_LORA_UNROLL_FULL
  for (int i = 0; i < RB; i++) {
    KQ_LORA_UNROLL_FULL
    for (int c = 0; c < CPT; c++) {
      acc[i][c] = 0.0f;
    }
  }
  for (int jb = int(sg); jb < RANK; jb += U * NSG) {
    const int jz = jb + ju * NSG;
    KQ_LORA_UNROLL_FULL
    for (int i = 0; i < RB; i++) {
      if (!lead[i]) {
        continue;
      }
      float b[U][CPT];
      KQ_LORA_UNROLL_FULL
      for (int u = 0; u < U; u++) {
        const int j = jb + u * NSG;
        const device T* p = B + ((int64_t)es[i] * RANK + j) * N + n0;
        if (V4) {
          KQ_LORA_UNROLL_FULL
          for (int c = 0; c < CPT; c += 4) {
            const vec<T, 4> v = (j < RANK && n0 < N)
                ? *(const device vec<T, 4>*)(p + c)
                : vec<T, 4>(0);
            KQ_LORA_UNROLL_FULL
            for (int q = 0; q < 4; q++) {
              b[u][c + q] = float(v[q]);
            }
          }
        } else {
          KQ_LORA_UNROLL_FULL
          for (int c = 0; c < CPT; c++) {
            b[u][c] = (j < RANK && n0 + c < N) ? float(p[c]) : 0.0f;
          }
        }
      }
      KQ_LORA_UNROLL_FULL
      for (int q = i; q < RB; q++) {
        if (es[q] != es[i]) {
          continue;
        }
        float zp = 0.0f;
        if (jz < RANK) {
          const device float* src = zpart + (int64_t)(r0 + q) * C * RANK + jz;
          for (int c = cg; c < C; c += 32 / U) {
            zp += src[c * RANK];
          }
        }
        zp += simd_shuffle_xor(zp, U);
        zp += simd_shuffle_xor(zp, 2 * U);
        zp *= fs[q];
        KQ_LORA_UNROLL_FULL
        for (int u = 0; u < U; u++) {
          const float z = simd_shuffle(zp, ushort(u));
          KQ_LORA_UNROLL_FULL
          for (int c = 0; c < CPT; c++) {
            acc[q][c] += z * b[u][c];
          }
        }
      }
    }
  }
  KQ_LORA_UNROLL_FULL
  for (int i = 0; i < RB; i++) {
    if (es[i] < 0) {
      continue;
    }
    KQ_LORA_UNROLL_FULL
    for (int c = 0; c < CPT; c++) {
      red[int(sg) * TN + int(lane) * CPT + c] = acc[i][c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < TN && nf < N) {
      float g = of[i];
      KQ_LORA_UNROLL_FULL
      for (int q = 0; q < NSG; q++) {
        g += red[q * TN + lid];
      }
      out[(int64_t)(r0 + i) * N + nf] = static_cast<T>(g);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
// (float/uint32/int32) operands share one kernel each. The 1-byte form
// copies a packed weight view whose rows are not dense (host side
// kquant_dense.cpp).
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
