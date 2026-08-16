// Vector scaled-dot-product-attention kernels for large head dims (e.g. 512)
// that stock MLX's vector allowlist {64,96,128,256} excludes, forcing a
// materialized fallback. Derived from MLX's sdpa_vector_2pass (MIT); the mask,
// sink and query-transposed paths are dropped (callers route only causal / full
// attention with a row-contiguous query). K and V are read in place via their
// head/seq strides so a strided KV-cache prefix needs no copy.
//
// Two passes: pass 1 splits the keys into `blocks` chunks, each chunk computing
// a partial online-softmax output + running max + running sum; pass 2 reduces
// the per-chunk partials into the final output. `do_causal` and `blocks` are
// Metal function constants so the key-stride loop specializes at pipeline
// build.

constant bool do_causal [[function_constant(0)]];
constant int blocks [[function_constant(1)]];
constant int gqa_splits [[function_constant(2)]];
constant bool gqa_has_sinks [[function_constant(3)]];
constant bool gqa_has_starts [[function_constant(4)]];
// Quantized KV operands (mlx affine wire, bits 8 / group 64): keys/values
// bind as packed uint32 words plus per-group scale/bias arrays, and the
// cooperative tile stage dequants into sK/sV -- everything downstream of
// the stage is unchanged. Compiled out when false.
constant bool gqa_kv_q8 [[function_constant(5)]];
constant bool gqa_write_lse [[function_constant(6)]];
// Cascade merge: a second partials set (the shared-prefix region, written by
// the fa row-tile pass in its folded [1, Hkv, B*gqa, splits2, D] layout)
// folds into the same online-softmax reduction as the primary set. Compiled
// out when false.
constant bool gqa_cascade [[function_constant(7)]];
// Page-gather decode: the key walk follows a per-(batch, kv-head) list of
// selected C-row pages (sparse top-k attention) instead of the contiguous
// [0, N) axis. Compiled out when false.
constant bool gqa_paged [[function_constant(8)]];

// PT = pass-1 partial store type. float16 inputs use float: the
// un-normalized accumulator state can exceed the fp16 ceiling. bfloat16
// keeps 16-bit stores (range-safe) and the pre-fix bandwidth.
template <typename T, typename PT, int D, int V = D>
[[kernel]] void kq_sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device PT* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // One threadgroup per (kv-head, batch, kv-block); the GQA group + query seq
  // are the threadgroup y,z dims so they share each block's K/V read.
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;

  queries += o_offset * D + simd_lid * qk_per_thread;
  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    }
    if (use_key) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(keys[j]);
      }
      score = simd_sum(score);
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<PT>(o[i]);
  }
}

template <typename T, typename PT, int D>
[[kernel]] void kq_sdpa_vector_2pass_2(
    const device PT* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  U sum_exp_score = 0;
  U max_score = Limits<U>::finite_min;

  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// Decode-time (qL == 1) GQA attention for head dims whose stock vector path
// leaves bandwidth on the table at long KV. Differs from kq_sdpa_vector_2pass
// in three ways: the key axis is split into `gqa_splits` CONTIGUOUS chunks
// (few, coarse partials; merge cost stays constant with depth); each chunk is
// streamed through threadgroup-staged K/V tiles shared by the whole GQA group
// (device reads KV once per kv-head instead of once per q-head); and scores
// run NE keys in flight per simdgroup with a log2(32/NE)-step lane reduction
// instead of a full simd_sum per key. Optional per-q-head attention sinks
// (an extra softmax logit with no value row) fold into the pass-2 denominator.
// Partials are float32.
//
// Pass-1 threadgroup: (32, gqa_factor, ceil(q_len / QPS)); one simdgroup per
// q-head x QPS-query group. Grid: (n_kv_heads, B, gqa_splits). q_len is 1 at
// decode and 2..4 at speculative-verify width: every simdgroup of the group
// shares each staged K/V tile, so device KV traffic stays one sweep per
// kv-head regardless of query count. QPS is the compile-time query count per
// simdgroup: each staged element is read from threadgroup memory once and
// dotted against QPS query slices, dividing the threadgroup-memory traffic
// (the qL>1 bottleneck) by QPS at a cost of QPS query/output register sets
// (QPS=2 at D=512 is ~140 floats/thread; QPS=4 spills). Verify queries are
// the sequence's trailing positions, causally clamped
// (key <= N - q_len + query index). Requires gqa_factor * ceil(q_len / QPS)
// <= 32 (1024-thread threadgroup) and gqa_splits <= 128 (pass-2 scratch).

template <typename T, int D, int C = 32, int NE = 4, int QPS = 1>
[[kernel]] void kq_sdpa_gqa_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& q_len [[buffer(12)]],
    const device int* starts [[buffer(13)]],
    const device T* k_scales [[buffer(14)]],
    const device T* k_biases [[buffer(15)]],
    const device T* v_scales [[buffer(16)]],
    const device T* v_biases [[buffer(17)]],
    const constant size_t& ks_head_stride [[buffer(18)]],
    const constant size_t& ks_seq_stride [[buffer(19)]],
    const constant size_t& vs_head_stride [[buffer(20)]],
    const constant size_t& vs_seq_stride [[buffer(21)]],
    const device int* pages [[buffer(22)]],
    const constant int& n_pages [[buffer(23)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int D4 = D / 4;
  constexpr int NL = 32 / NE; // lanes per in-flight key
  constexpr int DP4 = D4 / NL; // float4s per lane per key row
  using T4 = metal::vec<T, 4>;

  threadgroup T4 sK[C * D4];
  threadgroup T4 sV[C * D4];

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int split_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int nq = q_len; // runtime query count (tptg.z * QPS >= nq)
  const int qz0 = tidtg.z * QPS; // first query of this simdgroup
  const int lane = tidtg.x;
  const int tx = lane % NL;
  const int ty = lane / NL;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;

  // Contiguous, C-aligned chunk of the key axis for this threadgroup.
  const int chunk = ((N + gqa_splits * C - 1) / (gqa_splits * C)) * C;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  // Per-row key start (left-padded batched KV cache): row batch_idx attends
  // keys [row_start, N). Whole tiles below the start are skipped outright --
  // the pad region's bytes are never staged -- and a partially padded tile
  // masks per key below. A chunk entirely below the start writes an empty
  // partial (max finite_min, sum 0), which pass 2 merges at zero weight.
  int row_start = 0;
  int kt0 = k0;
  if (gqa_has_starts) {
    row_start = max(0, starts[batch_idx]);
    kt0 = max(k0, (row_start / C) * C);
  }

  // With gqa_kv_q8 the k/v buffers hold packed uint32 wire and the strides
  // arrive in WORDS (D/4 per row); otherwise they hold T elements.
  const device T* kbase =
      keys + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * k_head_stride;
  const device T* vbase =
      values + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * v_head_stride;
  const size_t kv_hb = (size_t)(batch_idx * num_kv_heads + kv_head_idx);
  const device uint32_t* kwbase =
      (const device uint32_t*)keys + kv_hb * k_head_stride;
  const device uint32_t* vwbase =
      (const device uint32_t*)values + kv_hb * v_head_stride;
  const device T* ksb = k_scales + kv_hb * ks_head_stride;
  const device T* kbb = k_biases + kv_hb * ks_head_stride;
  const device T* vsb = v_scales + kv_hb * vs_head_stride;
  const device T* vbb = v_biases + kv_hb * vs_head_stride;

  // Pre-scaled query slices for this lane's key-row columns
  // ([B, Hq, q_len, D], row-contiguous). A simdgroup past the runtime query
  // count (odd q_len at QPS=2) zero-fills; its lanes compute but never write.
  const device T4* q4 =
      (const device T4*)(queries + ((size_t)q_batch_head_idx * nq + qz0) * D);
  float4 qf[QPS][DP4];
  int lim[QPS]; // highest key each query may attend (its causal position)
  for (short p = 0; p < QPS; p++) {
    const bool active = qz0 + p < nq;
    for (short ii = 0; ii < DP4; ii++) {
      qf[p][ii] = active ? scale * float4(q4[(size_t)p * D4 + ii * NL + tx])
                         : float4(0);
    }
    lim[p] = active ? N - nq + qz0 + p : -1;
  }

  float max_score[QPS];
  float sum_exp_score[QPS];
  float4 lo[QPS][DP4];
  for (short p = 0; p < QPS; p++) {
    max_score[p] = Limits<float>::finite_min;
    sum_exp_score[p] = 0;
    for (short ii = 0; ii < DP4; ii++) {
      lo[p][ii] = 0;
    }
  }

  const int flat = (tidtg.z * gqa_factor + tidtg.y) * 32 + lane;
  const int n_threads = 32 * gqa_factor * tptg.z;

  // Tile walk: contiguous chunks of the key axis, or (paged) this
  // threadgroup's slice of the selected-page list. A page tile's tail
  // guard is N itself (pages are C-aligned windows of the full cache).
  int t0 = kt0 / C;
  int t1 = (k1 + C - 1) / C;
  const device int* prow = pages;
  if (gqa_paged) {
    const int pchunk = (n_pages + gqa_splits - 1) / gqa_splits;
    t0 = split_idx * pchunk;
    t1 = min(t0 + pchunk, n_pages);
    prow = pages + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * n_pages;
  }
  const int kend = gqa_paged ? N : k1;

  for (int t = t0; t < t1; t++) {
    const int kt = gqa_paged ? prow[t] * C : t * C;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cooperative tile load; zero-fill the tail so stale threadgroup data
    // can never reach the accumulators. The q8 path works in uint4 units
    // (16 codes): one vectorized wire load and ONE scale/bias pair per
    // unit -- 16 consecutive elements never straddle a 64-element group --
    // instead of per-word scalar traffic, which measured ~40% below the
    // fp16 kernel's bandwidth.
    if (gqa_kv_q8) {
      constexpr int DU = D / 16; // uint4 units per row
      for (int u = flat; u < C * DU; u += n_threads) {
        const int row = u / DU;
        const int c16 = u % DU;
        const int kg = kt + row;
        const int i = row * D4 + c16 * 4;
        if (kg < kend) {
          const uint4 kw =
              ((const device uint4*)(kwbase + (size_t)kg * k_seq_stride))[c16];
          const uint4 vw =
              ((const device uint4*)(vwbase + (size_t)kg * v_seq_stride))[c16];
          const size_t gs = (c16 * 16) / 64;
          const float ksc = float(ksb[(size_t)kg * ks_seq_stride + gs]);
          const float kbi = float(kbb[(size_t)kg * ks_seq_stride + gs]);
          const float vsc = float(vsb[(size_t)kg * vs_seq_stride + gs]);
          const float vbi = float(vbb[(size_t)kg * vs_seq_stride + gs]);
#pragma unroll
          for (short w = 0; w < 4; w++) {
            const uint32_t kx = w == 0 ? kw.x
                : w == 1               ? kw.y
                : w == 2               ? kw.z
                                       : kw.w;
            const uint32_t vx = w == 0 ? vw.x
                : w == 1               ? vw.y
                : w == 2               ? vw.z
                                       : vw.w;
            const float4 kq4 = float4(
                float(kx & 0xff),
                float((kx >> 8) & 0xff),
                float((kx >> 16) & 0xff),
                float(kx >> 24));
            const float4 vq4 = float4(
                float(vx & 0xff),
                float((vx >> 8) & 0xff),
                float((vx >> 16) & 0xff),
                float(vx >> 24));
            sK[i + w] = T4(kq4 * ksc + kbi);
            sV[i + w] = T4(vq4 * vsc + vbi);
          }
        } else {
#pragma unroll
          for (short w = 0; w < 4; w++) {
            sK[i + w] = T4(T(0));
            sV[i + w] = T4(T(0));
          }
        }
      }
    } else {
      for (int i = flat; i < C * D4; i += n_threads) {
        const int row = i / D4;
        const int col = i % D4;
        const int kg = kt + row;
        if (kg < kend) {
          sK[i] = ((const device T4*)(kbase + (size_t)kg * k_seq_stride))[col];
          sV[i] = ((const device T4*)(vbase + (size_t)kg * v_seq_stride))[col];
        } else {
          sK[i] = T4(T(0));
          sV[i] = T4(T(0));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores: NE keys in flight, NL lanes per key, then reduce + broadcast.
    // Each staged element is read once and dotted against all QPS queries.
    float mqk[QPS][C / NE];
    float m_tile[QPS];
    for (short p = 0; p < QPS; p++) {
      m_tile[p] = Limits<float>::finite_min;
    }
    for (short cc = 0; cc < C / NE; cc++) {
      float s[QPS];
      for (short p = 0; p < QPS; p++) {
        s[p] = 0;
      }
      for (short ii = 0; ii < DP4; ii++) {
        const float4 kk = float4(sK[(cc * NE + ty) * D4 + ii * NL + tx]);
        for (short p = 0; p < QPS; p++) {
          s[p] += dot(kk, qf[p][ii]);
        }
      }
      const int kg = kt + cc * NE + ty;
      for (short p = 0; p < QPS; p++) {
        for (short off = NL / 2; off > 0; off >>= 1) {
          s[p] += simd_shuffle_down(s[p], off);
        }
        s[p] = simd_shuffle(s[p], NL * ty);
        const bool valid =
            kg < kend && kg <= lim[p] && (!gqa_has_starts || kg >= row_start);
        mqk[p][cc] = valid ? s[p] : Limits<float>::finite_min;
        m_tile[p] = max(m_tile[p], mqk[p][cc]);
      }
    }

    // Online softmax per query; each lane sums its ty-group's keys, so the
    // simd_sum counts every key NL times. A tile with no valid key is
    // skipped outright: with the running max still finite_min,
    // exp(finite_min - finite_min) == 1 would poison the sum (happens past
    // a query's causal limit at verify width, or below row_start on a
    // left-padded row).
    float vs[QPS][C / NE];
    for (short p = 0; p < QPS; p++) {
      m_tile[p] = simd_max(m_tile[p]);
      if (m_tile[p] > Limits<float>::finite_min) {
        const float new_max = max(max_score[p], m_tile[p]);
        const float factor = fast::exp(max_score[p] - new_max);
        float vsum = 0;
        for (short cc = 0; cc < C / NE; cc++) {
          vs[p][cc] = fast::exp(mqk[p][cc] - new_max);
          vsum += vs[p][cc];
        }
        sum_exp_score[p] =
            sum_exp_score[p] * factor + simd_sum(vsum) * (1.0f / NL);
        max_score[p] = new_max;
        for (short ii = 0; ii < DP4; ii++) {
          lo[p][ii] *= factor;
        }
      } else {
        for (short cc = 0; cc < C / NE; cc++) {
          vs[p][cc] = 0;
        }
      }
    }

    for (short cc = 0; cc < C / NE; cc++) {
      for (short ii = 0; ii < DP4; ii++) {
        const float4 vv = float4(sV[(cc * NE + ty) * D4 + ii * NL + tx]);
        for (short p = 0; p < QPS; p++) {
          lo[p][ii] += vv * vs[p][cc];
        }
      }
    }
  }

  // Cross-ty reduction of the deferred output accumulators; the ty == 0 lane
  // group holds the chunk totals.
  for (short p = 0; p < QPS; p++) {
    if (qz0 + p >= nq) {
      continue;
    }
    for (short ii = 0; ii < DP4; ii++) {
      for (short off = (NE / 2) * NL; off >= NL; off >>= 1) {
        lo[p][ii][0] += simd_shuffle_down(lo[p][ii][0], off);
        lo[p][ii][1] += simd_shuffle_down(lo[p][ii][1], off);
        lo[p][ii][2] += simd_shuffle_down(lo[p][ii][2], off);
        lo[p][ii][3] += simd_shuffle_down(lo[p][ii][3], off);
      }
    }

    const size_t po =
        (((size_t)q_batch_head_idx * nq + qz0 + p) * gqa_splits + split_idx);
    if (ty == 0) {
      device float4* out4 = (device float4*)(out + po * D);
      for (short ii = 0; ii < DP4; ii++) {
        out4[ii * NL + tx] = lo[p][ii];
      }
    }
    if (lane == 0) {
      sums[po] = sum_exp_score[p];
      maxs[po] = max_score[p];
    }
  }
}

// Row-wise reduce/broadcast ops for the steel MMATile helpers below.
struct KQMaxOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return metal::max(x, y);
  }
};

struct KQSumOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x + y;
  }
};

struct KQMulOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x * y;
  }
};

struct KQExpSubOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return fast::exp2(x - y);
  }
};

// Cooperative row-major tile staging for the FA verify kernels
template <typename T, int BK, int D, int LDS, int NT>
METAL_FUNC void kq_fa_stage_rows(
    threadgroup T* dst,
    const device T* src,
    size_t seq_stride,
    int rows_valid,
    int flat_tid) {
  using T4 = metal::vec<T, 4>;
  constexpr int D4 = D / 4;
  constexpr int LDS4 = LDS / 4;
  threadgroup T4* dst4 = (threadgroup T4*)dst;
  for (int i = flat_tid; i < BK * D4; i += NT) {
    const int r = i / D4;
    const int c = i - r * D4;
    dst4[r * LDS4 + c] = r < rows_valid
        ? ((const device T4*)(src + (size_t)r * seq_stride))[c]
        : T4(T(0));
  }
}

// q8 variant: rows arrive as packed uint32 affine wire (bits 8, group 64)
// with per-group scale/bias; dequant lands in the staged tile so the MMA
// consumers are unchanged. Works in uint4 units (16 codes) -- one
// vectorized wire load and one scale/bias pair per unit, never straddling
// a 64-element group (same staging shape as the decode kernel's q8 path).
template <typename T, int BK, int D, int LDS, int NT>
METAL_FUNC void kq_fa_stage_rows_q8(
    threadgroup T* dst,
    const device uint32_t* src,
    const device T* scales,
    const device T* biases,
    size_t seq_stride_w,
    size_t sb_seq_stride,
    int rows_valid,
    int flat_tid) {
  using T4 = metal::vec<T, 4>;
  constexpr int DU = D / 16; // uint4 units per row
  constexpr int LDS4 = LDS / 4;
  threadgroup T4* dst4 = (threadgroup T4*)dst;
  for (int u = flat_tid; u < BK * DU; u += NT) {
    const int r = u / DU;
    const int c16 = u - r * DU;
    const int i = r * LDS4 + c16 * 4;
    if (r < rows_valid) {
      const uint4 w =
          ((const device uint4*)(src + (size_t)r * seq_stride_w))[c16];
      const size_t g = (c16 * 16) / 64;
      const float sc = float(scales[(size_t)r * sb_seq_stride + g]);
      const float bi = float(biases[(size_t)r * sb_seq_stride + g]);
#pragma unroll
      for (short j = 0; j < 4; j++) {
        const uint32_t x = j == 0 ? w.x : j == 1 ? w.y : j == 2 ? w.z : w.w;
        const float4 q4 = float4(
            float(x & 0xff),
            float((x >> 8) & 0xff),
            float((x >> 16) & 0xff),
            float(x >> 24));
        dst4[i + j] = T4(q4 * sc + bi);
      }
    } else {
#pragma unroll
      for (short j = 0; j < 4; j++) {
        dst4[i + j] = T4(T(0));
      }
    }
  }
}

// Simdgroup-matrix (steel MMA) speculative-verify attention, pass 1. The
// caller folds the GQA group into the query rows -- q [B, Hq, qL, D] becomes
// [B, Hkv, G*qL, D] with kv-major heads -- so the kernel sees an MHA problem
// whose n_rows = G*qL <= BQ queries fill one BQ-row tile (BQ 32 or 64; one
// 8-row fragment strip per simdgroup, so per-thread register pressure is
// BQ-independent), held in per-thread fragments (each thread owns one row of
// every 8x8 fragment, so the online-softmax row max/sum live in registers
// with no threadgroup round-trips). Grid (n_kv_heads, B, gqa_splits): each
// threadgroup streams its contiguous key chunk once through
// threadgroup-staged K/V tiles (one shared buffer, steel style), computing
// S = Q @ K^T and O += P @ V on simdgroup_matrix with float32 accumulators.
//
// Each folded row is causally clamped to key <= kL - qL + (row % qL), with qL
// a runtime buffer param. Only tiles reaching past kL - qL or the split tail
// take the mask branch; split-interior tiles run mask-free (verify rows share
// the whole prefix). Scores run in exp2 space (scale premultiplied by
// log2(e)); the P values and row sums are base-independent (2^(log2(e)*x) =
// e^x), so only the stored row max converts back to natural log and the
// partials [B, Hkv, n_rows, gqa_splits, D] merge through kq_sdpa_gqa_2pass_2
// unchanged. A tile entirely past a row's limit leaves the row's running max
// at finite_min and zeroes its P row (exp2(0) == 1 would otherwise poison the
// sum); an empty split writes (O = 0, sum = 0, max = finite_min) partials
// that merge with weight zero.
template <typename T, int D, int BQ = 32>
[[kernel]] void kq_sdpa_fa_verify_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& q_len [[buffer(12)]],
    const constant int& n_rows [[buffer(13)]],
    const device T* k_scales [[buffer(14)]],
    const device T* k_biases [[buffer(15)]],
    const device T* v_scales [[buffer(16)]],
    const device T* v_biases [[buffer(17)]],
    const constant size_t& ks_head_stride [[buffer(18)]],
    const constant size_t& ks_seq_stride [[buffer(19)]],
    const constant size_t& vs_head_stride [[buffer(20)]],
    const constant size_t& vs_seq_stride [[buffer(21)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int BK = 32; // keys staged per tile
  constexpr int kNWarps = BQ / 8; // one 8-row fragment strip per simdgroup
  constexpr short kFragSize = 8;
  constexpr short TK = BK / kFragSize;
  constexpr short TD = D / kFragSize;
  constexpr short kPad = 16 / sizeof(T);
  constexpr short LDS = D + kPad; // K and V staged row-major [BK][D + pad]

  using MMAFrag_t = mlx::steel::BaseMMAFrag<float, kFragSize, kFragSize>;

  // K and V share the buffer (steel pattern): the S matmul consumes Ks
  // before the V load overwrites it, halving the threadgroup footprint.
  threadgroup T KV_smem[BK * LDS];

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int split_idx = tid.z;
  const int num_kv_heads = tpg.x;
  const int q_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  // Contiguous, BK-aligned chunk of the key axis for this threadgroup.
  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  // With gqa_kv_q8 the k/v buffers hold packed uint32 wire and the strides
  // arrive in WORDS (D/4 per row); otherwise they hold T elements.
  const size_t kv_hb = (size_t)(batch_idx * num_kv_heads + kv_head_idx);
  const device T* kbase =
      keys + kv_hb * k_head_stride + (size_t)k0 * k_seq_stride;
  const device T* vbase =
      values + kv_hb * v_head_stride + (size_t)k0 * v_seq_stride;
  const device uint32_t* kwbase = (const device uint32_t*)keys +
      kv_hb * k_head_stride + (size_t)k0 * k_seq_stride;
  const device uint32_t* vwbase = (const device uint32_t*)values +
      kv_hb * v_head_stride + (size_t)k0 * v_seq_stride;
  const device T* ksb =
      k_scales + kv_hb * ks_head_stride + (size_t)k0 * ks_seq_stride;
  const device T* kbb =
      k_biases + kv_hb * ks_head_stride + (size_t)k0 * ks_seq_stride;
  const device T* vsb =
      v_scales + kv_hb * vs_head_stride + (size_t)k0 * vs_seq_stride;
  const device T* vbb =
      v_biases + kv_hb * vs_head_stride + (size_t)k0 * vs_seq_stride;

  // Fragment coordinates: this thread owns row (row0 + sm) and the column
  // pair at sn of every 8x8 fragment.
  const short2 sc = MMAFrag_t::get_coord(simd_lid);
  const short sm = sc.y;
  const short sn = sc.x;
  const int row = int(simd_gid) * kFragSize + sm;
  const int flat_tid = int(simd_gid) * 32 + int(simd_lid);
  // Highest key this row attends. Padding rows (row >= n_rows) compute a
  // harmless in-range limit; their partials are never written.
  const int lim = N - q_len + (row % q_len);
  const int lim_min = N - q_len; // every real row attends at least this far

  // Q tile in float32 fragments (one device read; rows past n_rows
  // zero-fill, so padding rows score 0 everywhere).
  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Qtile;
  {
    const device T* qrow =
        queries + ((size_t)q_batch_head_idx * n_rows + row) * D + sn;
    Qtile.template load_safe<T, 1, 1>(qrow, D, short2(D - sn, n_rows - row));
  }

  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Stile;
  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Ktile;
  mlx::steel::MMATile<float, 1, 1, MMAFrag_t> Vtile;
  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Otile;
  Otile.clear();

  // exp2-space online softmax (steel): scores carry scale * log2(e).
  const float scale2 = scale * M_LOG2E_F;
  float max_score = Limits<float>::finite_min;
  float sum_score = 0;

  for (int kt = k0; kt < k1; kt += BK) {
    const int krem = min(k1 - kt, BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gqa_kv_q8) {
      kq_fa_stage_rows_q8<T, BK, D, LDS, kNWarps * 32>(
          KV_smem,
          kwbase + (size_t)(kt - k0) * k_seq_stride,
          ksb + (size_t)(kt - k0) * ks_seq_stride,
          kbb + (size_t)(kt - k0) * ks_seq_stride,
          k_seq_stride,
          ks_seq_stride,
          krem,
          flat_tid);
    } else {
      kq_fa_stage_rows<T, BK, D, LDS, kNWarps * 32>(
          KV_smem,
          kbase + (size_t)(kt - k0) * k_seq_stride,
          k_seq_stride,
          krem,
          flat_tid);
    }
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q @ K^T, one 8-deep head-dim slab at a time. K is staged
    // row-major, so the K^T fragment load swaps its strides.
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Ktile.template load<T, 1, 1, 1, LDS>(
          &KV_smem[sn * LDS + dd * kFragSize + sm]);
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_t::mma(
            Stile.frag_at(0, ik),
            Qtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            Stile.frag_at(0, ik));
      }
    }

    // Scale in float32, then mask. Only the split tail or a tile reaching
    // past kL - qL can mask anything; interior tiles skip the branch.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }
    if (krem < BK || kt + BK - 1 > lim_min) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const int kg = kt + ik * kFragSize + sn;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          if (kg + jj >= k1 || kg + jj > lim) {
            Stile.frag_at(0, ik)[jj] = Limits<float>::finite_min;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gqa_kv_q8) {
      kq_fa_stage_rows_q8<T, BK, D, LDS, kNWarps * 32>(
          KV_smem,
          vwbase + (size_t)(kt - k0) * v_seq_stride,
          vsb + (size_t)(kt - k0) * vs_seq_stride,
          vbb + (size_t)(kt - k0) * vs_seq_stride,
          v_seq_stride,
          vs_seq_stride,
          krem,
          flat_tid);
    } else {
      kq_fa_stage_rows<T, BK, D, LDS, kNWarps * 32>(
          KV_smem,
          vbase + (size_t)(kt - k0) * v_seq_stride,
          v_seq_stride,
          krem,
          flat_tid);
    }

    // Online softmax on this thread's row (registers only, overlapping the
    // V load). A row with no valid key yet keeps max at finite_min and
    // zeroes its P row instead of exponentiating.
    float new_max = max_score;
    Stile.template row_reduce<KQMaxOp>(&new_max);
    if (new_max > Limits<float>::finite_min) {
      Stile.template row_bin_op<KQExpSubOp>(&new_max);
      float factor = fast::exp2(max_score - new_max);
      float tile_sum = 0;
      Stile.template row_reduce<KQSumOp>(&tile_sum);
      sum_score = sum_score * factor + tile_sum;
      max_score = new_max;
      Otile.template row_bin_op<KQMulOp>(&factor);
    } else {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P @ V
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        Vtile.template load<T, 1, 1, LDS, 1>(
            &KV_smem[(ik * kFragSize + sm) * LDS + id * kFragSize + sn]);
        MMAFrag_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }
  }

  // Unnormalized partials in the kq_sdpa_gqa_2pass_1 layout. Each row's four
  // owner threads hold the reduced row stats after the shuffle reductions;
  // the sn == 0 owner writes them, with the max converted to natural log for
  // the shared merge.
  if (row < n_rows) {
    const size_t po =
        ((size_t)q_batch_head_idx * n_rows + row) * gqa_splits + split_idx;
    device float* orow = out + po * D + sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
        orow[id * kFragSize + jj] = Otile.frag_at(0, id)[jj];
      }
    }
    if (sn == 0) {
      sums[po] = sum_score;
      maxs[po] = max_score == Limits<float>::finite_min
          ? Limits<float>::finite_min
          : max_score * M_LN2_F;
    }
  }
}

// head_dim-512 FA verify pass 1. At D=512 the per-thread Q and O fragment
// sets of the kernel above are ~256 floats (each thread owns two columns of
// every 8-wide fragment across the full head dim): certain register spill
// at 128 threads. The fold therefore also splits by HEAD-DIM HALF: 8
// simdgroups (256 threads), simdgroup sg owning row strip sg / 2 and d-half
// sg & 1. Q, O and the P @ V accumulation live entirely in the owning half,
// so the per-thread register budget matches the hd256 kernel. Only
// S = Q @ K^T spans the full head dim: each half computes a partial S over
// its 256 columns and the halves are summed through a small threadgroup
// scratch each tile (half 0 writes, half 1 adds in place -- sibling
// simdgroups own identical (row, key) elements, so the add is race-free --
// then both halves read the combined tile back and compute the row stats
// redundantly but identically, in registers). BK drops to 16 so a full-D
// K/V row tile fits the 32 KB threadgroup budget. The folded-causal mask,
// exp2-space softmax, empty-split/dead-row guards, partials layout and
// merge pass all match the hd256 kernel.
template <typename T, int D>
[[kernel]] void kq_sdpa_fa_verify_dsplit_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& q_len [[buffer(12)]],
    const constant int& n_rows [[buffer(13)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int BQ = 32; // query rows (the whole fold, zero-padded)
  constexpr int BK = 16; // keys staged per tile (full-D rows)
  constexpr int kNWarps = 8; // 4 row strips x 2 head-dim halves
  constexpr short kFragSize = 8;
  constexpr short TK = BK / kFragSize;
  constexpr short DH = D / 2; // columns per head-dim half
  constexpr short TDH = DH / kFragSize;
  constexpr short kPad = 16 / sizeof(T);
  constexpr short LDS = D + kPad; // K and V staged row-major [BK][D + pad]

  using MMAFrag_t = mlx::steel::BaseMMAFrag<float, kFragSize, kFragSize>;

  // K and V share the buffer (the S matmul consumes Ks before the V load
  // overwrites it); S_smem carries the half-partial score exchange.
  threadgroup T KV_smem[BK * LDS];
  threadgroup float S_smem[BQ * BK];

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int split_idx = tid.z;
  const int num_kv_heads = tpg.x;
  const int q_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  // Contiguous, BK-aligned chunk of the key axis for this threadgroup.
  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  const device T* kbase = keys +
      (size_t)(batch_idx * num_kv_heads + kv_head_idx) * k_head_stride +
      (size_t)k0 * k_seq_stride;
  const device T* vbase = values +
      (size_t)(batch_idx * num_kv_heads + kv_head_idx) * v_head_stride +
      (size_t)k0 * v_seq_stride;

  const short dh_half = simd_gid & 1;
  const short strip = simd_gid >> 1;
  const short2 sc = MMAFrag_t::get_coord(simd_lid);
  const short sm = sc.y;
  const short sn = sc.x;
  const int row = int(strip) * kFragSize + sm;
  const int lim = N - q_len + (row % q_len);
  const int lim_min = N - q_len; // every real row attends at least this far
  const int flat_tid = int(simd_gid) * 32 + int(simd_lid);

  // This half's Q columns in float32 fragments (one device read; rows past
  // n_rows zero-fill, so padding rows score 0 everywhere).
  mlx::steel::MMATile<float, 1, TDH, MMAFrag_t> Qtile;
  {
    const device T* qrow = queries +
        ((size_t)q_batch_head_idx * n_rows + row) * D + dh_half * DH + sn;
    Qtile.template load_safe<T, 1, 1>(qrow, D, short2(DH - sn, n_rows - row));
  }

  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Stile;
  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Ktile;
  mlx::steel::MMATile<float, 1, 1, MMAFrag_t> Vtile;
  mlx::steel::MMATile<float, 1, TDH, MMAFrag_t> Otile;
  Otile.clear();

  // exp2-space online softmax (steel): scores carry scale * log2(e).
  const float scale2 = scale * M_LOG2E_F;
  float max_score = Limits<float>::finite_min;
  float sum_score = 0;

  for (int kt = k0; kt < k1; kt += BK) {
    const int krem = min(k1 - kt, BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kq_fa_stage_rows<T, BK, D, LDS, kNWarps * 32>(
        KV_smem,
        kbase + (size_t)(kt - k0) * k_seq_stride,
        k_seq_stride,
        krem,
        flat_tid);
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Partial S over this half's 256 columns (row-major staging, swapped
    // fragment load strides).
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDH; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Ktile.template load<T, 1, 1, 1, LDS>(
          &KV_smem[sn * LDS + dh_half * DH + dd * kFragSize + sm]);
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_t::mma(
            Stile.frag_at(0, ik),
            Qtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            Stile.frag_at(0, ik));
      }
    }

    // Half-partial exchange; the scale folds into the read-back.
    if (dh_half == 0) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          S_smem[row * BK + ik * kFragSize + sn + jj] =
              Stile.frag_at(0, ik)[jj];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dh_half == 1) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          S_smem[row * BK + ik * kFragSize + sn + jj] +=
              Stile.frag_at(0, ik)[jj];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
        Stile.frag_at(0, ik)[jj] =
            S_smem[row * BK + ik * kFragSize + sn + jj] * scale2;
      }
    }

    // Mask. Only the split tail or a tile reaching past kL - qL can mask
    // anything; interior tiles skip the branch.
    if (krem < BK || kt + BK - 1 > lim_min) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const int kg = kt + ik * kFragSize + sn;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          if (kg + jj >= k1 || kg + jj > lim) {
            Stile.frag_at(0, ik)[jj] = Limits<float>::finite_min;
          }
        }
      }
    }

    // The exchange barriers already ordered every K read before this
    // overwrite; no extra barrier needed.
    kq_fa_stage_rows<T, BK, D, LDS, kNWarps * 32>(
        KV_smem,
        vbase + (size_t)(kt - k0) * v_seq_stride,
        v_seq_stride,
        krem,
        flat_tid);

    // Online softmax on this thread's row (registers only, overlapping the
    // V load); both halves compute identical stats from the combined S. A
    // row with no valid key yet keeps max at finite_min and zeroes its P
    // row instead of exponentiating.
    float new_max = max_score;
    Stile.template row_reduce<KQMaxOp>(&new_max);
    if (new_max > Limits<float>::finite_min) {
      Stile.template row_bin_op<KQExpSubOp>(&new_max);
      float factor = fast::exp2(max_score - new_max);
      float tile_sum = 0;
      Stile.template row_reduce<KQSumOp>(&tile_sum);
      sum_score = sum_score * factor + tile_sum;
      max_score = new_max;
      Otile.template row_bin_op<KQMulOp>(&factor);
    } else {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O_half += P @ V[:, half]
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDH; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        Vtile.template load<T, 1, 1, LDS, 1>(
            &KV_smem
                [(ik * kFragSize + sm) * LDS + dh_half * DH + id * kFragSize +
                 sn]);
        MMAFrag_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }
  }

  // Unnormalized partials, each half writing its own columns; the sn == 0
  // owner of half 0 writes the row stats (both halves hold identical
  // values), with the max converted to natural log for the shared merge.
  if (row < n_rows) {
    const size_t po =
        ((size_t)q_batch_head_idx * n_rows + row) * gqa_splits + split_idx;
    device float* orow = out + po * D + dh_half * DH + sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDH; id++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
        orow[id * kFragSize + jj] = Otile.frag_at(0, id)[jj];
      }
    }
    if (dh_half == 0 && sn == 0) {
      sums[po] = sum_score;
      maxs[po] = max_score == Limits<float>::finite_min
          ? Limits<float>::finite_min
          : max_score * M_LN2_F;
    }
  }
}

// Merge the per-split partials; one simdgroup per (q-head, batch, query).
// Grid z is the query axis (1 at decode; q_len at verify width). Sinks are
// a per-q-head extra logit with no value row: they raise the global max and
// add exp(sink - max) to the denominator; a query's own sink applies at
// every verify position identically.
template <typename T, int D>
[[kernel]] void kq_sdpa_gqa_2pass_2(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    const device float* sinks [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& n_q_heads [[buffer(5)]],
    device float* out_lse [[buffer(6)]],
    const device float* partials2 [[buffer(7)]],
    const device float* sums2 [[buffer(8)]],
    const device float* maxs2 [[buffer(9)]],
    const constant int& cascade_splits [[buffer(10)]],
    const constant int& cascade_gqa [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int EPT = D / 32; // output elements per lane
  const int head_idx = tid.x;
  const int batch_idx = tid.y;
  const size_t base =
      ((size_t)batch_idx * n_q_heads + head_idx) * tpg.z + tid.z;

  partials += base * gqa_splits * D;
  sums += base * gqa_splits;
  maxs += base * gqa_splits;

  // Second set: fa folded layout with the query axis innermost,
  // row = ((kv*B + b)*gqa + g)*qL + t (tpg.z = qL; decode width -> t = 0).
  int splits2 = 0;
  if (gqa_cascade) {
    const int kv = head_idx / cascade_gqa;
    const int g = head_idx % cascade_gqa;
    const size_t row =
        (((size_t)kv * tpg.y + batch_idx) * cascade_gqa + g) * tpg.z + tid.z;
    splits2 = cascade_splits;
    partials2 += row * splits2 * D;
    sums2 += row * splits2;
    maxs2 += row * splits2;
  }

  threadgroup float ws[128];
  threadgroup float ws2[128];

  float m = Limits<float>::finite_min;
  for (int s = simd_lid; s < gqa_splits; s += 32) {
    m = max(m, maxs[s]);
  }
  for (int s = simd_lid; s < splits2; s += 32) {
    m = max(m, maxs2[s]);
  }
  m = simd_max(m);
  if (gqa_has_sinks) {
    m = max(m, sinks[head_idx]);
  }

  float denom = 0;
  for (int s = simd_lid; s < gqa_splits; s += 32) {
    const float w = fast::exp(maxs[s] - m);
    ws[s] = w;
    denom += w * sums[s];
  }
  for (int s = simd_lid; s < splits2; s += 32) {
    const float w = fast::exp(maxs2[s] - m);
    ws2[s] = w;
    denom += w * sums2[s];
  }
  denom = simd_sum(denom);
  if (gqa_has_sinks) {
    denom += fast::exp(sinks[head_idx] - m);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[EPT] = {0};
  for (int s = 0; s < gqa_splits; s++) {
    const float w = ws[s];
    for (short e = 0; e < EPT; e++) {
      acc[e] += w * partials[s * D + e * 32 + simd_lid];
    }
  }
  for (int s = 0; s < splits2; s++) {
    const float w = ws2[s];
    for (short e = 0; e < EPT; e++) {
      acc[e] += w * partials2[s * D + e * 32 + simd_lid];
    }
  }
  out += base * D;
  for (short e = 0; e < EPT; e++) {
    out[e * 32 + simd_lid] = static_cast<T>(denom == 0 ? 0.0f : acc[e] / denom);
  }
  // Natural-log softmax normalizer (sinks included when present): the
  // cascade merge weight for combining disjoint key regions.
  if (gqa_write_lse && simd_lid == 0) {
    out_lse[base] = denom == 0 ? -INFINITY : (fast::log(denom) + m);
  }
}
