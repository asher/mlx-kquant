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

template <int ACT>
inline float kq_glu_epilogue(float g, float u) {
  if (ACT == KQ_GLU_ACT_GELU) {
    // tanh-approx gelu (matches mlx nn.gelu_approx / gemma usage)
    const float g3 = 0.044715f * g * g * g;
    const float t = metal::precise::tanh(0.7978845608028654f * (g + g3));
    return (0.5f * g * (1.0f + t)) * u;
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
      out[row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u));
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
      out[row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u));
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
      out[row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u));
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
      out[row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u));
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
