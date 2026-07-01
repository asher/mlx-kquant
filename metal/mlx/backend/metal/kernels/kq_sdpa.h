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

template <typename T, int D, int V = D>
[[kernel]] void kq_sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
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
    out[i] = static_cast<T>(o[i]);
  }
}

template <typename T, int D>
[[kernel]] void kq_sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
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
// Pass-1 threadgroup: (32, gqa_factor, 1); one simdgroup per q-head of the
// group. Grid: (n_kv_heads, B, gqa_splits). Requires gqa_factor <= 8 and
// gqa_splits <= 128 (pass-2 scratch).

template <typename T, int D, int C = 32, int NE = 4>
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

  const device T* kbase =
      keys + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * k_head_stride;
  const device T* vbase =
      values + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * v_head_stride;

  // Pre-scaled query slice for this lane's key-row columns.
  const device T4* q4 =
      (const device T4*)(queries + (size_t)q_batch_head_idx * D);
  float4 qf[DP4];
  for (short ii = 0; ii < DP4; ii++) {
    qf[ii] = scale * float4(q4[ii * NL + tx]);
  }

  float max_score = Limits<float>::finite_min;
  float sum_exp_score = 0;
  float4 lo[DP4];
  for (short ii = 0; ii < DP4; ii++) {
    lo[ii] = 0;
  }

  const int flat = tidtg.y * 32 + lane;
  const int n_threads = 32 * gqa_factor;

  for (int kt = k0; kt < k1; kt += C) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cooperative tile load; zero-fill the tail so stale threadgroup data
    // can never reach the accumulators.
    for (int i = flat; i < C * D4; i += n_threads) {
      const int row = i / D4;
      const int col = i % D4;
      const int kg = kt + row;
      if (kg < k1) {
        sK[i] = ((const device T4*)(kbase + (size_t)kg * k_seq_stride))[col];
        sV[i] = ((const device T4*)(vbase + (size_t)kg * v_seq_stride))[col];
      } else {
        sK[i] = T4(T(0));
        sV[i] = T4(T(0));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores: NE keys in flight, NL lanes per key, then reduce + broadcast.
    float mqk[C / NE];
    float m_tile = Limits<float>::finite_min;
    for (short cc = 0; cc < C / NE; cc++) {
      float s = 0;
      for (short ii = 0; ii < DP4; ii++) {
        s += dot(float4(sK[(cc * NE + ty) * D4 + ii * NL + tx]), qf[ii]);
      }
      for (short off = NL / 2; off > 0; off >>= 1) {
        s += simd_shuffle_down(s, off);
      }
      s = simd_shuffle(s, NL * ty);
      const bool valid = (kt + cc * NE + ty) < k1;
      mqk[cc] = valid ? s : Limits<float>::finite_min;
      m_tile = max(m_tile, mqk[cc]);
    }
    m_tile = simd_max(m_tile);

    // Online softmax; each lane sums its ty-group's keys, so the simd_sum
    // counts every key NL times.
    const float new_max = max(max_score, m_tile);
    const float factor = fast::exp(max_score - new_max);
    float vs[C / NE];
    float vsum = 0;
    for (short cc = 0; cc < C / NE; cc++) {
      vs[cc] = fast::exp(mqk[cc] - new_max);
      vsum += vs[cc];
    }
    sum_exp_score = sum_exp_score * factor + simd_sum(vsum) * (1.0f / NL);
    max_score = new_max;

    for (short ii = 0; ii < DP4; ii++) {
      lo[ii] *= factor;
    }
    for (short cc = 0; cc < C / NE; cc++) {
      for (short ii = 0; ii < DP4; ii++) {
        lo[ii] += float4(sV[(cc * NE + ty) * D4 + ii * NL + tx]) * vs[cc];
      }
    }
  }

  // Cross-ty reduction of the deferred output accumulator; the ty == 0 lane
  // group holds the chunk totals.
  for (short ii = 0; ii < DP4; ii++) {
    for (short off = (NE / 2) * NL; off >= NL; off >>= 1) {
      lo[ii][0] += simd_shuffle_down(lo[ii][0], off);
      lo[ii][1] += simd_shuffle_down(lo[ii][1], off);
      lo[ii][2] += simd_shuffle_down(lo[ii][2], off);
      lo[ii][3] += simd_shuffle_down(lo[ii][3], off);
    }
  }

  const size_t po = ((size_t)q_batch_head_idx * gqa_splits + split_idx);
  if (ty == 0) {
    device float4* out4 = (device float4*)(out + po * D);
    for (short ii = 0; ii < DP4; ii++) {
      out4[ii * NL + tx] = lo[ii];
    }
  }
  if (lane == 0) {
    sums[po] = sum_exp_score;
    maxs[po] = max_score;
  }
}

// Merge the per-split partials; one simdgroup per (q-head, batch). Sinks are
// a per-q-head extra logit with no value row: they raise the global max and
// add exp(sink - max) to the denominator.
template <typename T, int D>
[[kernel]] void kq_sdpa_gqa_2pass_2(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    const device float* sinks [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& n_q_heads [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int EPT = D / 32; // output elements per lane
  const int head_idx = tid.x;
  const int batch_idx = tid.y;
  const size_t base = (size_t)batch_idx * n_q_heads + head_idx;

  partials += base * gqa_splits * D;
  sums += base * gqa_splits;
  maxs += base * gqa_splits;

  threadgroup float ws[128];

  float m = Limits<float>::finite_min;
  for (int s = simd_lid; s < gqa_splits; s += 32) {
    m = max(m, maxs[s]);
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
  out += base * D;
  for (short e = 0; e < EPT; e++) {
    out[e * 32 + simd_lid] = static_cast<T>(denom == 0 ? 0.0f : acc[e] / denom);
  }
}
