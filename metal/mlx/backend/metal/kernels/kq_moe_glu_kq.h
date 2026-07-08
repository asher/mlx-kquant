// Fused MoE gather kernels for K-quant codecs, decode-shaped (one activation
// row per gathered expert row). K-quant counterpart of kq_moe_glu.h (mxfp4):
//
//   kq_<codec>_moe_glu_gather:  out = act(gate(x)) * up(x) in ONE dispatch --
//     both expert matvecs share each activation load and the GLU epilogue
//     replaces the gather/act/mul kernel chain. No biases (K-quant MoE
//     checkpoints in the wild carry none); ACT selects the activation.
//   kq_<codec>_gather_qmv:  a gathered matvec (the down projection), one
//     activation row per (token, expert-slot).
//   kq_<codec>_moe_glu_gather_shexp:  moe_glu_gather with the shared expert
//     folded in as one extra grid row (tid.y == R reads the 2-D shexp gate/up
//     tensors instead of the expert stack); out is [T, R+1, N].
//   kq_<codec>_gather_qmv_mix:  down projection with the routing mix folded
//     in: one threadgroup accumulates all S slots (last slot = shared expert)
//     weighted by scores[t, s] in f32, writing [T, N] directly -- no
//     [T, R, N] intermediate and no mul/sum/add glue kernels.
//
// Weights are GGUF wire bytes (n_experts, out_dims, bytes_per_row), the same
// layout KQuantSwitchLinear stores and kq.gather_qmm reads. Inner loops mirror
// the per-codec kq_<codec>_qmv_fast_impl thread mappings exactly (bit-exact
// dequant); only the row addressing (expert offset) and the epilogue differ.
//
// Grid: (N / 8, R, T) threadgroups of (32, 2, 1); R = expert slots per token,
// T = tokens. Each threadgroup computes 8 output rows (2 simdgroups x 4).
// Alignment: q6_k requires K % 256 == 0 (superblock); q8_0 uses the fast-path
// mapping and requires K % 256 == 0 as well (validated host-side).

// Activation selector (epilogue): out = act(g) * u.
#define KQ_GLU_ACT_SILU 0
#define KQ_GLU_ACT_GELU 1
#define KQ_GLU_ACT_SILU_LIMIT 2

template <int ACT>
inline float kq_glu_epilogue(float g, float u, float limit) {
  if (ACT == KQ_GLU_ACT_GELU) {
    // tanh-approx gelu (matches mlx nn.gelu_approx / gemma usage)
    const float g3 = 0.044715f * g * g * g;
    const float t = metal::precise::tanh(0.7978845608028654f * (g + g3));
    return (0.5f * g * (1.0f + t)) * u;
  }
  if (ACT == KQ_GLU_ACT_SILU_LIMIT) {
    // deepseek-v4 LimitedSwiGLU: gate clamped from above only, up clamped
    // both sides, then plain silu(g) * u (alpha 1, no +1 -- NOT gpt-oss).
    g = metal::min(g, limit);
    u = metal::clamp(u, -limit, limit);
  }
  const float sig = 1.0f / (1.0f + metal::exp(-g));
  return (g * sig) * u;
}

// ---------------------------------------------------------------------------
// q6_k (superblock 256): thread mapping mirrors kq_q6_k_qmv_fast_impl.
// ---------------------------------------------------------------------------
template <typename T, int ACT>
[[kernel]] void kq_q6_k_moe_glu_gather(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant float& limit [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  typedef float U;

  thread U yl[16];
  thread U rg[results_per_simdgroup] = {0};
  thread U ru[results_per_simdgroup] = {0};

  const int tid_lane = simd_lid / 2;
  const int ix = simd_lid % 2;
  const int ip = tid_lane / 8;
  const int il = tid_lane % 8;
  const int l0 = 4 * il;
  const int is = 8 * ip + l0 / 16;

  const int R = tpg.y;
  const int expert = indices[tid.z * R + tid.y];
  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;

  const int row_bytes = K * KQ_Q6_K_BLOCK_BYTES / KQ_Q6_K_SUPERBLOCK;
  const int nb = K / KQ_Q6_K_SUPERBLOCK;
  const int64_t row0 = (int64_t)expert * N + out_row;

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * R + tid.y) * N + out_row;

  for (int ib = ix; ib < nb; ib += 2) {
    const int x_base = ib * KQ_Q6_K_SUPERBLOCK + 128 * ip + l0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      yl[4 * l + 0] = U(x[x_base + l + 0]);
      yl[4 * l + 1] = U(x[x_base + l + 32]);
      yl[4 * l + 2] = U(x[x_base + l + 64]);
      yl[4 * l + 3] = U(x[x_base + l + 96]);
    }

#pragma unroll
    for (int side = 0; side < 2; side++) {
      const device uint8_t* w = side == 0 ? gw : uw;
      thread U* acc = side == 0 ? rg : ru;
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* sb_addr =
            w + (row0 + row) * row_bytes + ib * KQ_Q6_K_BLOCK_BYTES;
        const device uint8_t* q1 = kq_q6_k_ql_ptr(sb_addr) + 64 * ip + l0;
        const device uint8_t* q2 = q1 + 32;
        const device uint8_t* qh = kq_q6_k_qh_ptr(sb_addr) + 32 * ip + l0;
        const device int8_t* sc = kq_q6_k_scales_ptr(sb_addr) + is;

        U sums[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
        for (int l = 0; l < 4; l++) {
          const uint8_t q1l = q1[l];
          const uint8_t q2l = q2[l];
          const uint8_t qhl = qh[l];
          const int8_t v0 =
              int8_t((q1l & 0x0F) | ((qhl & 0x03) << 4)) - int8_t(32);
          const int8_t v1 =
              int8_t((q2l & 0x0F) | ((qhl & 0x0C) << 2)) - int8_t(32);
          const int8_t v2 =
              int8_t((q1l >> 4) | ((qhl & 0x30) << 0)) - int8_t(32);
          const int8_t v3 =
              int8_t((q2l >> 4) | ((qhl & 0xC0) >> 2)) - int8_t(32);
          sums[0] += yl[4 * l + 0] * U(v0);
          sums[1] += yl[4 * l + 1] * U(v1);
          sums[2] += yl[4 * l + 2] * U(v2);
          sums[3] += yl[4 * l + 3] * U(v3);
        }
        const U d = U(kq_q6_k_d(sb_addr));
        acc[row] += d *
            (sums[0] * U(sc[0]) + sums[1] * U(sc[2]) + sums[2] * U(sc[4]) +
             sums[3] * U(sc[6]));
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    U g = simd_sum(rg[row]);
    U u = simd_sum(ru[row]);
    if (simd_lid == 0) {
      out[row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
    }
  }
}

template <typename T>
[[kernel]] void kq_q6_k_gather_qmv(
    const device uint8_t* w [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  typedef float U;

  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int tid_lane = simd_lid / 2;
  const int ix = simd_lid % 2;
  const int ip = tid_lane / 8;
  const int il = tid_lane % 8;
  const int l0 = 4 * il;
  const int is = 8 * ip + l0 / 16;

  const int R = tpg.y;
  const int64_t row_idx = (int64_t)tid.z * R + tid.y;
  const int expert = indices[row_idx];
  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;

  const int row_bytes = K * KQ_Q6_K_BLOCK_BYTES / KQ_Q6_K_SUPERBLOCK;
  const int nb = K / KQ_Q6_K_SUPERBLOCK;
  const int64_t row0 = (int64_t)expert * N + out_row;

  x += row_idx * K;
  out += row_idx * N + out_row;

  for (int ib = ix; ib < nb; ib += 2) {
    const int x_base = ib * KQ_Q6_K_SUPERBLOCK + 128 * ip + l0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      yl[4 * l + 0] = U(x[x_base + l + 0]);
      yl[4 * l + 1] = U(x[x_base + l + 32]);
      yl[4 * l + 2] = U(x[x_base + l + 64]);
      yl[4 * l + 3] = U(x[x_base + l + 96]);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* sb_addr =
          w + (row0 + row) * row_bytes + ib * KQ_Q6_K_BLOCK_BYTES;
      const device uint8_t* q1 = kq_q6_k_ql_ptr(sb_addr) + 64 * ip + l0;
      const device uint8_t* q2 = q1 + 32;
      const device uint8_t* qh = kq_q6_k_qh_ptr(sb_addr) + 32 * ip + l0;
      const device int8_t* sc = kq_q6_k_scales_ptr(sb_addr) + is;

      U sums[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int l = 0; l < 4; l++) {
        const uint8_t q1l = q1[l];
        const uint8_t q2l = q2[l];
        const uint8_t qhl = qh[l];
        const int8_t v0 =
            int8_t((q1l & 0x0F) | ((qhl & 0x03) << 4)) - int8_t(32);
        const int8_t v1 =
            int8_t((q2l & 0x0F) | ((qhl & 0x0C) << 2)) - int8_t(32);
        const int8_t v2 = int8_t((q1l >> 4) | ((qhl & 0x30) << 0)) - int8_t(32);
        const int8_t v3 = int8_t((q2l >> 4) | ((qhl & 0xC0) >> 2)) - int8_t(32);
        sums[0] += yl[4 * l + 0] * U(v0);
        sums[1] += yl[4 * l + 1] * U(v1);
        sums[2] += yl[4 * l + 2] * U(v2);
        sums[3] += yl[4 * l + 3] * U(v3);
      }
      const U d = U(kq_q6_k_d(sb_addr));
      result[row] += d *
          (sums[0] * U(sc[0]) + sums[1] * U(sc[2]) + sums[2] * U(sc[4]) +
           sums[3] * U(sc[6]));
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      out[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int ACT>
[[kernel]] void kq_q6_k_moe_glu_gather_shexp(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device uint8_t* sgw [[buffer(2)]],
    const device uint8_t* suw [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uint32_t* indices [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant float& limit [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  typedef float U;

  thread U yl[16];
  thread U rg[results_per_simdgroup] = {0};
  thread U ru[results_per_simdgroup] = {0};

  const int tid_lane = simd_lid / 2;
  const int ix = simd_lid % 2;
  const int ip = tid_lane / 8;
  const int il = tid_lane % 8;
  const int l0 = 4 * il;
  const int is = 8 * ip + l0 / 16;

  const int n_route = tpg.y - 1;
  const bool shared_slot = int(tid.y) == n_route;
  const int expert = shared_slot ? 0 : int(indices[tid.z * n_route + tid.y]);
  const device uint8_t* gsrc = shared_slot ? sgw : gw;
  const device uint8_t* usrc = shared_slot ? suw : uw;
  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;

  const int row_bytes = K * KQ_Q6_K_BLOCK_BYTES / KQ_Q6_K_SUPERBLOCK;
  const int nb = K / KQ_Q6_K_SUPERBLOCK;
  const int64_t row0 = (int64_t)expert * N + out_row;

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * tpg.y + tid.y) * N + out_row;

  for (int ib = ix; ib < nb; ib += 2) {
    const int x_base = ib * KQ_Q6_K_SUPERBLOCK + 128 * ip + l0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      yl[4 * l + 0] = U(x[x_base + l + 0]);
      yl[4 * l + 1] = U(x[x_base + l + 32]);
      yl[4 * l + 2] = U(x[x_base + l + 64]);
      yl[4 * l + 3] = U(x[x_base + l + 96]);
    }

#pragma unroll
    for (int side = 0; side < 2; side++) {
      const device uint8_t* w = side == 0 ? gsrc : usrc;
      thread U* acc = side == 0 ? rg : ru;
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* sb_addr =
            w + (row0 + row) * row_bytes + ib * KQ_Q6_K_BLOCK_BYTES;
        const device uint8_t* q1 = kq_q6_k_ql_ptr(sb_addr) + 64 * ip + l0;
        const device uint8_t* q2 = q1 + 32;
        const device uint8_t* qh = kq_q6_k_qh_ptr(sb_addr) + 32 * ip + l0;
        const device int8_t* sc = kq_q6_k_scales_ptr(sb_addr) + is;

        U sums[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
        for (int l = 0; l < 4; l++) {
          const uint8_t q1l = q1[l];
          const uint8_t q2l = q2[l];
          const uint8_t qhl = qh[l];
          const int8_t v0 =
              int8_t((q1l & 0x0F) | ((qhl & 0x03) << 4)) - int8_t(32);
          const int8_t v1 =
              int8_t((q2l & 0x0F) | ((qhl & 0x0C) << 2)) - int8_t(32);
          const int8_t v2 =
              int8_t((q1l >> 4) | ((qhl & 0x30) << 0)) - int8_t(32);
          const int8_t v3 =
              int8_t((q2l >> 4) | ((qhl & 0xC0) >> 2)) - int8_t(32);
          sums[0] += yl[4 * l + 0] * U(v0);
          sums[1] += yl[4 * l + 1] * U(v1);
          sums[2] += yl[4 * l + 2] * U(v2);
          sums[3] += yl[4 * l + 3] * U(v3);
        }
        const U d = U(kq_q6_k_d(sb_addr));
        acc[row] += d *
            (sums[0] * U(sc[0]) + sums[1] * U(sc[2]) + sums[2] * U(sc[4]) +
             sums[3] * U(sc[6]));
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    U g = simd_sum(rg[row]);
    U u = simd_sum(ru[row]);
    if (simd_lid == 0) {
      out[row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
    }
  }
}

template <typename T>
[[kernel]] void kq_q6_k_gather_qmv_mix(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* sw [[buffer(1)]],
    const device T* h [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    const device float* scores [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& S [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  typedef float U;

  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int tid_lane = simd_lid / 2;
  const int ix = simd_lid % 2;
  const int ip = tid_lane / 8;
  const int il = tid_lane % 8;
  const int l0 = 4 * il;
  const int is = 8 * ip + l0 / 16;

  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;
  const int row_bytes = K * KQ_Q6_K_BLOCK_BYTES / KQ_Q6_K_SUPERBLOCK;
  const int nb = K / KQ_Q6_K_SUPERBLOCK;

  for (int slot = 0; slot < S; slot++) {
    const bool shared_slot = slot == S - 1;
    const device uint8_t* src = shared_slot ? sw : w;
    const int expert = shared_slot ? 0 : int(indices[tid.z * (S - 1) + slot]);
    const U score = U(scores[tid.z * S + slot]);
    const device T* xs = h + ((int64_t)tid.z * S + slot) * K;
    const int64_t row0 = (int64_t)expert * N + out_row;

    thread U acc[results_per_simdgroup] = {0};
    for (int ib = ix; ib < nb; ib += 2) {
      const int x_base = ib * KQ_Q6_K_SUPERBLOCK + 128 * ip + l0;
#pragma unroll
      for (int l = 0; l < 4; l++) {
        yl[4 * l + 0] = U(xs[x_base + l + 0]);
        yl[4 * l + 1] = U(xs[x_base + l + 32]);
        yl[4 * l + 2] = U(xs[x_base + l + 64]);
        yl[4 * l + 3] = U(xs[x_base + l + 96]);
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* sb_addr =
            src + (row0 + row) * row_bytes + ib * KQ_Q6_K_BLOCK_BYTES;
        const device uint8_t* q1 = kq_q6_k_ql_ptr(sb_addr) + 64 * ip + l0;
        const device uint8_t* q2 = q1 + 32;
        const device uint8_t* qh = kq_q6_k_qh_ptr(sb_addr) + 32 * ip + l0;
        const device int8_t* sc = kq_q6_k_scales_ptr(sb_addr) + is;

        U sums[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
        for (int l = 0; l < 4; l++) {
          const uint8_t q1l = q1[l];
          const uint8_t q2l = q2[l];
          const uint8_t qhl = qh[l];
          const int8_t v0 =
              int8_t((q1l & 0x0F) | ((qhl & 0x03) << 4)) - int8_t(32);
          const int8_t v1 =
              int8_t((q2l & 0x0F) | ((qhl & 0x0C) << 2)) - int8_t(32);
          const int8_t v2 =
              int8_t((q1l >> 4) | ((qhl & 0x30) << 0)) - int8_t(32);
          const int8_t v3 =
              int8_t((q2l >> 4) | ((qhl & 0xC0) >> 2)) - int8_t(32);
          sums[0] += yl[4 * l + 0] * U(v0);
          sums[1] += yl[4 * l + 1] * U(v1);
          sums[2] += yl[4 * l + 2] * U(v2);
          sums[3] += yl[4 * l + 3] * U(v3);
        }
        const U d = U(kq_q6_k_d(sb_addr));
        acc[row] += d *
            (sums[0] * U(sc[0]) + sums[1] * U(sc[2]) + sums[2] * U(sc[4]) +
             sums[3] * U(sc[6]));
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] += score * acc[row];
    }
  }

  out += (int64_t)tid.z * N + out_row;
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      out[row] = static_cast<T>(result[row]);
    }
  }
}

// ---------------------------------------------------------------------------
// q8_0 (block 32): thread mapping mirrors kq_q8_0_qmv_fast_impl.
// ---------------------------------------------------------------------------
template <typename T, int ACT>
[[kernel]] void kq_q8_0_moe_glu_gather(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant float& limit [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U rg[results_per_simdgroup] = {0};
  thread U ru[results_per_simdgroup] = {0};

  const int R = tpg.y;
  const int expert = indices[tid.z * R + tid.y];
  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;

  const int row_bytes = K * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int64_t row0 = (int64_t)expert * N + out_row;
  const int lane_k_offset = simd_lid * values_per_thread;

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * R + tid.y) * N + out_row;

  for (int k = 0; k < K; k += block_size) {
    const int k_global = k + lane_k_offset;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      x_thread[i] = U(x[k_global + i]);
    }
    const int block_id = k_global / KQ_Q8_0_GROUP;
    const int within = k_global - block_id * KQ_Q8_0_GROUP;

#pragma unroll
    for (int side = 0; side < 2; side++) {
      const device uint8_t* w = side == 0 ? gw : uw;
      thread U* acc = side == 0 ? rg : ru;
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* block_addr =
            w + (row0 + row) * row_bytes + block_id * KQ_Q8_0_BLOCK_BYTES;
        const U d = U(kq_q8_0_d(block_addr));
        const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;
        U partial = 0;
#pragma unroll
        for (int i = 0; i < values_per_thread; i++) {
          partial += x_thread[i] * U(q_ptr[i]);
        }
        acc[row] += d * partial;
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    U g = simd_sum(rg[row]);
    U u = simd_sum(ru[row]);
    if (simd_lid == 0) {
      out[row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
    }
  }
}

template <typename T>
[[kernel]] void kq_q8_0_gather_qmv(
    const device uint8_t* w [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int R = tpg.y;
  const int64_t row_idx = (int64_t)tid.z * R + tid.y;
  const int expert = indices[row_idx];
  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;

  const int row_bytes = K * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int64_t row0 = (int64_t)expert * N + out_row;
  const int lane_k_offset = simd_lid * values_per_thread;

  x += row_idx * K;
  out += row_idx * N + out_row;

  for (int k = 0; k < K; k += block_size) {
    const int k_global = k + lane_k_offset;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      x_thread[i] = U(x[k_global + i]);
    }
    const int block_id = k_global / KQ_Q8_0_GROUP;
    const int within = k_global - block_id * KQ_Q8_0_GROUP;

    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* block_addr =
          w + (row0 + row) * row_bytes + block_id * KQ_Q8_0_BLOCK_BYTES;
      const U d = U(kq_q8_0_d(block_addr));
      const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;
      U partial = 0;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        partial += x_thread[i] * U(q_ptr[i]);
      }
      result[row] += d * partial;
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      out[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int ACT>
[[kernel]] void kq_q8_0_moe_glu_gather_shexp(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device uint8_t* sgw [[buffer(2)]],
    const device uint8_t* suw [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uint32_t* indices [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant float& limit [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U rg[results_per_simdgroup] = {0};
  thread U ru[results_per_simdgroup] = {0};

  const int n_route = tpg.y - 1;
  const bool shared_slot = int(tid.y) == n_route;
  const int expert = shared_slot ? 0 : int(indices[tid.z * n_route + tid.y]);
  const device uint8_t* gsrc = shared_slot ? sgw : gw;
  const device uint8_t* usrc = shared_slot ? suw : uw;
  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;

  const int row_bytes = K * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int64_t row0 = (int64_t)expert * N + out_row;
  const int lane_k_offset = simd_lid * values_per_thread;

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * tpg.y + tid.y) * N + out_row;

  for (int k = 0; k < K; k += block_size) {
    const int k_global = k + lane_k_offset;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      x_thread[i] = U(x[k_global + i]);
    }
    const int block_id = k_global / KQ_Q8_0_GROUP;
    const int within = k_global - block_id * KQ_Q8_0_GROUP;

#pragma unroll
    for (int side = 0; side < 2; side++) {
      const device uint8_t* w = side == 0 ? gsrc : usrc;
      thread U* acc = side == 0 ? rg : ru;
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* block_addr =
            w + (row0 + row) * row_bytes + block_id * KQ_Q8_0_BLOCK_BYTES;
        const U d = U(kq_q8_0_d(block_addr));
        const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;
        U partial = 0;
#pragma unroll
        for (int i = 0; i < values_per_thread; i++) {
          partial += x_thread[i] * U(q_ptr[i]);
        }
        acc[row] += d * partial;
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    U g = simd_sum(rg[row]);
    U u = simd_sum(ru[row]);
    if (simd_lid == 0) {
      out[row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
    }
  }
}

template <typename T>
[[kernel]] void kq_q8_0_gather_qmv_mix(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* sw [[buffer(1)]],
    const device T* h [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    const device float* scores [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& S [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;
  const int row_bytes = K * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int lane_k_offset = simd_lid * values_per_thread;

  for (int slot = 0; slot < S; slot++) {
    const bool shared_slot = slot == S - 1;
    const device uint8_t* src = shared_slot ? sw : w;
    const int expert = shared_slot ? 0 : int(indices[tid.z * (S - 1) + slot]);
    const U score = U(scores[tid.z * S + slot]);
    const device T* xs = h + ((int64_t)tid.z * S + slot) * K;
    const int64_t row0 = (int64_t)expert * N + out_row;

    thread U acc[results_per_simdgroup] = {0};
    for (int k = 0; k < K; k += block_size) {
      const int k_global = k + lane_k_offset;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        x_thread[i] = U(xs[k_global + i]);
      }
      const int block_id = k_global / KQ_Q8_0_GROUP;
      const int within = k_global - block_id * KQ_Q8_0_GROUP;

      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* block_addr =
            src + (row0 + row) * row_bytes + block_id * KQ_Q8_0_BLOCK_BYTES;
        const U d = U(kq_q8_0_d(block_addr));
        const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;
        U partial = 0;
#pragma unroll
        for (int i = 0; i < values_per_thread; i++) {
          partial += x_thread[i] * U(q_ptr[i]);
        }
        acc[row] += d * partial;
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] += score * acc[row];
    }
  }

  out += (int64_t)tid.z * N + out_row;
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      out[row] = static_cast<T>(result[row]);
    }
  }
}

// ---------------------------------------------------------------------------
// Codec-matrix kernels: one generic implementation per family, templated on
// the Ext codec traits from kq_quantized*.h (superblock, block_bytes,
// deq_chunk16(block, il, reg) -> 16 weights in natural order). Thread
// mapping follows kq_mv_ext_impl, templated on the K-lane width NX: the 32
// simdgroup lanes split into NX K-lanes x (32 / NX) output rows (each thread
// owns one row); the K-reduction is a log2(NX)-step simd_shuffle_down within
// the NX-lane row group. Grid mirrors the tuned kernels above --
// (N / rows_per_tg, R, T) threadgroups of (32, 2, 1), rows_per_tg = 64 / NX
// -- so host dispatch stays codec-independent. NX = 8 matches the tuned
// mapping (8 rows per threadgroup); NX = 16 / 32 halve/quarter the rows per
// threadgroup so decode-scale launches (one token, few expert slots) fill
// the device: the default grid underfills it and the kernels sit at 57-77%
// of DRAM peak until the threadgroup count is 2-4x higher. The tuned
// q6_k/q8_0 kernels stay the uniform-codec dispatch targets at NX = 8;
// these cover the remaining codecs, wide-NX dispatches (q6_k_ext/q8_0_ext
// stems), and mixed-codec shared experts (SCodec != Codec, the UD-style
// q8_0 shexp over k-quant expert stacks).
// ---------------------------------------------------------------------------

#define KQ_EXT_NXPSG 8

// Partial dot of x (one activation row) against wire-byte weight row `row`,
// this thread's K-stripe only (chunks tx, tx + NX, ...). `luts` is the
// codec's threadgroup-staged table block (KqTgLuts; unused for table-free
// codecs).
template <typename T, typename Codec, int NX>
METAL_FUNC float kq_ext_row_partial(
    const device uint8_t* w,
    const device T* x,
    int64_t row,
    int K,
    short tx,
    const threadgroup uint8_t* luts) {
  constexpr short chpb = Codec::superblock / 16;
  const int nb = K / Codec::superblock;
  const device uint8_t* w_row = w + row * (int64_t)nb * Codec::block_bytes;
  float acc = 0.0f;
  for (int ich = tx; 16 * ich < K; ich += NX) {
    const device uint8_t* block =
        w_row + (int64_t)(ich / chpb) * Codec::block_bytes;
    float4x4 lw;
    KqTgLuts<Codec>::deq_chunk16(block, short(ich % chpb), lw, luts);
    const device T* xp = x + ich * 16;
    acc += dot(lw[0], float4(*(const device vec<T, 4>*)(xp + 0))) +
        dot(lw[1], float4(*(const device vec<T, 4>*)(xp + 4))) +
        dot(lw[2], float4(*(const device vec<T, 4>*)(xp + 8))) +
        dot(lw[3], float4(*(const device vec<T, 4>*)(xp + 12)));
  }
  return acc;
}

// GLU pair variant: gate and up rows share each activation chunk load.
template <typename T, typename Codec, int NX>
METAL_FUNC float2 kq_ext_glu_row_partial(
    const device uint8_t* gw,
    const device uint8_t* uw,
    const device T* x,
    int64_t row,
    int K,
    short tx,
    const threadgroup uint8_t* luts) {
  constexpr short chpb = Codec::superblock / 16;
  const int nb = K / Codec::superblock;
  const int64_t row_off = row * (int64_t)nb * Codec::block_bytes;
  const device uint8_t* g_row = gw + row_off;
  const device uint8_t* u_row = uw + row_off;
  float2 acc = float2(0.0f);
  for (int ich = tx; 16 * ich < K; ich += NX) {
    const int64_t boff = (int64_t)(ich / chpb) * Codec::block_bytes;
    const short cch = short(ich % chpb);
    const device T* xp = x + ich * 16;
    const float4 a0 = float4(*(const device vec<T, 4>*)(xp + 0));
    const float4 a1 = float4(*(const device vec<T, 4>*)(xp + 4));
    const float4 a2 = float4(*(const device vec<T, 4>*)(xp + 8));
    const float4 a3 = float4(*(const device vec<T, 4>*)(xp + 12));
    float4x4 lw;
    KqTgLuts<Codec>::deq_chunk16(g_row + boff, cch, lw, luts);
    acc.x += dot(lw[0], a0) + dot(lw[1], a1) + dot(lw[2], a2) + dot(lw[3], a3);
    KqTgLuts<Codec>::deq_chunk16(u_row + boff, cch, lw, luts);
    acc.y += dot(lw[0], a0) + dot(lw[1], a1) + dot(lw[2], a2) + dot(lw[3], a3);
  }
  return acc;
}

// Declares `name` and stages CodecT's decode LUTs into it (no-op, 16-byte
// stub array for table-free codecs). Gather threadgroups are (32, 2, 1).
#define KQ_EXT_STAGE_LUTS(CodecT, name)                                \
  threadgroup uint4 name##_v[(KqTgLuts<CodecT>::bytes + 15) / 16 + 1]; \
  threadgroup uint8_t* name =                                          \
      reinterpret_cast<threadgroup uint8_t*>(name##_v);                \
  if (KqTgLuts<CodecT>::bytes > 0) {                                   \
    KqTgLuts<CodecT>::stage(                                           \
        name, ushort(simd_gid * 32 + simd_lid), ushort(64));           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                   \
  }

// NX-lane K-stripe reduction (row groups are consecutive lanes; only lane
// tx == 0 of each group holds the full sum afterwards). NX = 8 keeps the
// shipped 3-step order bit-identical.
template <int NX>
METAL_FUNC float kq_ext_reduce(float v) {
  for (short off = NX / 2; off > 0; off >>= 1) {
    v += simd_shuffle_down(v, ushort(off));
  }
  return v;
}

template <typename T, typename Codec, int ACT, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_moe_glu_gather(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant float& limit [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX; // rows per simdgroup (2 simdgroups per tg)
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int expert = indices[tid.z * R + tid.y];
  const int out_row = tid.x * (2 * RPS) + int(simd_gid) * RPS + ty;

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * R + tid.y) * N;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  const float2 gu = kq_ext_glu_row_partial<T, Codec, NX>(
      gw, uw, x, (int64_t)expert * N + out_row, K, tx, kq_luts);
  const float g = kq_ext_reduce<NX>(gu.x);
  const float u = kq_ext_reduce<NX>(gu.y);
  if (tx == 0) {
    out[out_row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
  }
}

template <typename T, typename Codec, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_gather_qmv(
    const device uint8_t* w [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int64_t row_idx = (int64_t)tid.z * R + tid.y;
  const int expert = indices[row_idx];
  const int out_row = tid.x * (2 * RPS) + int(simd_gid) * RPS + ty;

  x += row_idx * K;
  out += row_idx * N;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  const float r = kq_ext_reduce<NX>(kq_ext_row_partial<T, Codec, NX>(
      w, x, (int64_t)expert * N + out_row, K, tx, kq_luts));
  if (tx == 0) {
    out[out_row] = static_cast<T>(r);
  }
}

template <
    typename T,
    typename Codec,
    typename SCodec,
    int ACT,
    int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_moe_glu_gather_shexp(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device uint8_t* sgw [[buffer(2)]],
    const device uint8_t* suw [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uint32_t* indices [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant float& limit [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int n_route = tpg.y - 1;
  const bool shared_slot = int(tid.y) == n_route;
  const int out_row = tid.x * (2 * RPS) + int(simd_gid) * RPS + ty;

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * tpg.y + tid.y) * N;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  KQ_EXT_STAGE_LUTS(SCodec, kq_sluts)
  float2 gu;
  if (shared_slot) {
    gu = kq_ext_glu_row_partial<T, SCodec, NX>(
        sgw, suw, x, (int64_t)out_row, K, tx, kq_sluts);
  } else {
    const int expert = int(indices[tid.z * n_route + tid.y]);
    gu = kq_ext_glu_row_partial<T, Codec, NX>(
        gw, uw, x, (int64_t)expert * N + out_row, K, tx, kq_luts);
  }
  const float g = kq_ext_reduce<NX>(gu.x);
  const float u = kq_ext_reduce<NX>(gu.y);
  if (tx == 0) {
    out[out_row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
  }
}

// No-shared-expert mix (gemma-style MoE: plain score-weighted sum over the
// routed slots): indices and scores are both [T, S], every slot gathers from
// the expert stack.
template <typename T, typename Codec, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_gather_qmv_mix_ns(
    const device uint8_t* w [[buffer(0)]],
    const device T* h [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    const device float* scores [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& S [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int out_row = tid.x * (2 * RPS) + int(simd_gid) * RPS + ty;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  float result = 0.0f;
  for (int slot = 0; slot < S; slot++) {
    const int expert = int(indices[tid.z * S + slot]);
    const device T* xs = h + ((int64_t)tid.z * S + slot) * K;
    result += scores[tid.z * S + slot] *
        kq_ext_row_partial<T, Codec, NX>(
                  w, xs, (int64_t)expert * N + out_row, K, tx, kq_luts);
  }
  result = kq_ext_reduce<NX>(result);
  if (tx == 0) {
    out[(int64_t)tid.z * N + out_row] = static_cast<T>(result);
  }
}

template <typename T, typename Codec, typename SCodec, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_gather_qmv_mix(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* sw [[buffer(1)]],
    const device T* h [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    const device float* scores [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& S [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int out_row = tid.x * (2 * RPS) + int(simd_gid) * RPS + ty;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  KQ_EXT_STAGE_LUTS(SCodec, kq_sluts)
  float result = 0.0f;
  for (int slot = 0; slot < S - 1; slot++) {
    const int expert = int(indices[tid.z * (S - 1) + slot]);
    const device T* xs = h + ((int64_t)tid.z * S + slot) * K;
    result += scores[tid.z * S + slot] *
        kq_ext_row_partial<T, Codec, NX>(
                  w, xs, (int64_t)expert * N + out_row, K, tx, kq_luts);
  }
  {
    const device T* xs = h + ((int64_t)tid.z * S + (S - 1)) * K;
    result += scores[tid.z * S + (S - 1)] *
        kq_ext_row_partial<T, SCodec, NX>(
                  sw, xs, (int64_t)out_row, K, tx, kq_sluts);
  }
  result = kq_ext_reduce<NX>(result);
  if (tx == 0) {
    out[(int64_t)tid.z * N + out_row] = static_cast<T>(result);
  }
}

// ---------------------------------------------------------------------------
// Router top-k (codec-independent float kernel): softmax over E logits, pick
// the top R (min-index tie-break), emit gather-ready indices [T, R] uint32 and
// mix scores [T, R + SHARED] float32 in one dispatch. When SHARED == 1,
// column E of each logits row is the shared-expert gate logit and its sigmoid
// lands in scores slot R (qwen3-next); SHARED == 0 is the plain-MoE form
// (gemma). NORM selects renormalizing the picked probabilities to sum to 1
// (norm_topk_prob; also exactly softmax-over-the-selected-logits, the gemma
// router semantics). HAS_PES scales each picked score by pes[expert]
// (gemma's learned per_expert_scale); pes may be any bound buffer when
// HAS_PES == 0. One threadgroup of 256 threads per token; E <= 1024, R <= 16
// (host-checked).
// ---------------------------------------------------------------------------
#define KQ_ROUTER_MAX_E 1024
#define KQ_ROUTER_MAX_R 16

template <typename T>
[[kernel]] void kq_moe_router_topk(
    const device T* logits [[buffer(0)]],
    device uint32_t* indices [[buffer(1)]],
    device float* scores [[buffer(2)]],
    const constant int& E [[buffer(3)]],
    const constant int& R [[buffer(4)]],
    const constant int& NORM [[buffer(5)]],
    const constant int& SHARED [[buffer(6)]],
    const device float* pes [[buffer(7)]],
    const constant int& HAS_PES [[buffer(8)]],
    uint tid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int NT = 256;
  constexpr int NSG = NT / 32;
  threadgroup float p[KQ_ROUTER_MAX_E];
  threadgroup float red_v[NSG];
  threadgroup uint red_i[NSG];
  threadgroup float stat[2];
  threadgroup float win_v[KQ_ROUTER_MAX_R];
  threadgroup uint win_i[KQ_ROUTER_MAX_R];

  const device T* lrow = logits + (int64_t)tid * (E + SHARED);

  float m = -INFINITY;
  for (int e = lid; e < E; e += NT) {
    m = metal::max(m, float(lrow[e]));
  }
  m = simd_max(m);
  if (simd_lid == 0) {
    red_v[simd_gid] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float g = red_v[0];
    for (int i = 1; i < NSG; i++) {
      g = metal::max(g, red_v[i]);
    }
    stat[0] = g;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float gmax = stat[0];

  float s = 0;
  for (int e = lid; e < E; e += NT) {
    const float v = metal::exp(float(lrow[e]) - gmax);
    p[e] = v;
    s += v;
  }
  s = simd_sum(s);
  if (simd_lid == 0) {
    red_v[simd_gid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float g = 0;
    for (int i = 0; i < NSG; i++) {
      g += red_v[i];
    }
    stat[1] = g;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float gsum = stat[1];

  for (int r = 0; r < R; r++) {
    float bv = -1.0f;
    uint bi = 0xffffffffu;
    for (int e = lid; e < E; e += NT) {
      const float v = p[e];
      if (v > bv || (v == bv && uint(e) < bi)) {
        bv = v;
        bi = uint(e);
      }
    }
    const float sv = simd_max(bv);
    uint cand = bv == sv ? bi : 0xffffffffu;
    cand = simd_min(cand);
    if (simd_lid == 0) {
      red_v[simd_gid] = sv;
      red_i[simd_gid] = cand;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      float wv = red_v[0];
      uint wi = red_i[0];
      for (int i = 1; i < NSG; i++) {
        if (red_v[i] > wv || (red_v[i] == wv && red_i[i] < wi)) {
          wv = red_v[i];
          wi = red_i[i];
        }
      }
      indices[(int64_t)tid * R + r] = wi;
      win_v[r] = wv;
      win_i[r] = wi;
      p[wi] = -1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    float ps = 0;
    for (int r = 0; r < R; r++) {
      ps += win_v[r];
    }
    const float denom = NORM ? ps : gsum;
    device float* srow = scores + (int64_t)tid * (R + SHARED);
    for (int r = 0; r < R; r++) {
      float sv = win_v[r] / denom;
      if (HAS_PES) {
        sv *= pes[win_i[r]];
      }
      srow[r] = sv;
    }
    if (SHARED) {
      srow[R] = 1.0f / (1.0f + metal::exp(-float(lrow[E])));
    }
  }
}
