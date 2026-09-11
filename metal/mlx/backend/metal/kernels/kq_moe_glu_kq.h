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
#define KQ_GLU_ACT_SWIGLU_CLAMP 3

template <int ACT>
inline float
kq_glu_epilogue(float g, float u, float limit, float alpha = 1.0f) {
  if (ACT == KQ_GLU_ACT_GELU) {
    // tanh-approx gelu (matches mlx nn.gelu_approx / gemma usage)
    const float g3 = 0.044715f * g * g * g;
    const float t = metal::precise::tanh(0.7978845608028654f * (g + g3));
    return (0.5f * g * (1.0f + t)) * u;
  }
  if (ACT == KQ_GLU_ACT_SWIGLU_CLAMP) {
    // gpt-oss clamped SwiGLU: gate clamped from above only, up clamped both
    // sides, sigmoid slope alpha, and a (u + 1) linear term -- matches the
    // packed kq_moe_glu_gather epilogue (kq_moe_glu.h).
    g = metal::min(g, limit);
    u = metal::clamp(u, -limit, limit);
    const float sig = 1.0f / (1.0f + metal::exp(-alpha * g));
    return (g * sig) * (u + 1.0f);
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

template <typename T, int results_per_simdgroup = 4>
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
  const int out_row =
      tid.x * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;

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

template <typename T, int results_per_simdgroup = 4>
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
    const constant int& SC [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  typedef float U;

  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int tid_lane = simd_lid / 2;
  const int ix = simd_lid % 2;
  const int ip = tid_lane / 8;
  const int il = tid_lane % 8;
  const int l0 = 4 * il;
  const int is = 8 * ip + l0 / 16;

  const int out_row =
      tid.x * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;
  const int row_bytes = K * KQ_Q6_K_BLOCK_BYTES / KQ_Q6_K_SUPERBLOCK;
  const int nb = K / KQ_Q6_K_SUPERBLOCK;

  for (int slot = 0; slot < S; slot++) {
    const bool shared_slot = slot == S - 1;
    const device uint8_t* src = shared_slot ? sw : w;
    const int expert = shared_slot ? 0 : int(indices[tid.z * (S - 1) + slot]);
    // SC == S - 1: the shared slot's weight is an implicit 1
    const U score = slot < SC ? U(scores[tid.z * SC + slot]) : U(1);
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

template <typename T, int results_per_simdgroup = 4>
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
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int R = tpg.y;
  const int64_t row_idx = (int64_t)tid.z * R + tid.y;
  const int expert = indices[row_idx];
  const int out_row =
      tid.x * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;

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

template <typename T, int results_per_simdgroup = 4>
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
    const constant int& SC [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int out_row =
      tid.x * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;
  const int row_bytes = K * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int lane_k_offset = simd_lid * values_per_thread;

  for (int slot = 0; slot < S; slot++) {
    const bool shared_slot = slot == S - 1;
    const device uint8_t* src = shared_slot ? sw : w;
    const int expert = shared_slot ? 0 : int(indices[tid.z * (S - 1) + slot]);
    // SC == S - 1: the shared slot's weight is an implicit 1
    const U score = slot < SC ? U(scores[tid.z * SC + slot]) : U(1);
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
// deq_chunk16(block, il, reg) -> 16 weights in natural order). Row loops
// use the deq_chunk16s form and fold the returned scale once per chunk
// dot. Thread
// mapping follows kq_mv_ext_impl, templated on the K-lane width NX: the 32
// simdgroup lanes split into NX K-lanes x (32 / NX) output rows (each thread
// owns one row); the K-reduction is a log2(NX)-step simd_shuffle_down within
// the NX-lane row group. Grid: (N / rows_per_tg, R, T) threadgroups of
// (32, SG, 1), rows_per_tg = SG * 32 / NX, SG simdgroups per threadgroup
// picked by the host (kq_moe_pick_sg; the tuned kernels above stay at 2).
// SG = 8 amortizes the per-threadgroup LUT staging and barrier over 4x the
// rows, which pays for codecs staging a large table (iq2_xs: +10% at t=1,
// +47% at t=4) and is flat-to-negative for the rest, which stay at SG = 2
// like the tuned kernels. NX = 8 matches the tuned mapping; NX = 16 / 32
// halve/quarter the rows per threadgroup so decode-scale launches fill
// the device: the default grid underfills it and the kernels sit at 57-77%
// of DRAM peak until the threadgroup count is 2-4x higher. (Fine tiling --
// fewer rows per threadgroup at constant NX, the lever that pays on the
// tuned q6_k/q8_0 and packed-mxfp4 gathers -- was measured E2E-neutral on
// these Ext kernels at both 2x and 4x threadgroups and is not instantiated
// for them.) The tuned
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
    float sc;
    KqTgLuts<Codec>::deq_chunk16s(block, short(ich % chpb), lw, luts, sc);
    const device T* xp = x + ich * 16;
    acc += sc *
        (dot(lw[0], float4(*(const device vec<T, 4>*)(xp + 0))) +
         dot(lw[1], float4(*(const device vec<T, 4>*)(xp + 4))) +
         dot(lw[2], float4(*(const device vec<T, 4>*)(xp + 8))) +
         dot(lw[3], float4(*(const device vec<T, 4>*)(xp + 12))));
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
    float sc;
    KqTgLuts<Codec>::deq_chunk16s(g_row + boff, cch, lw, luts, sc);
    acc.x += sc *
        (dot(lw[0], a0) + dot(lw[1], a1) + dot(lw[2], a2) + dot(lw[3], a3));
    KqTgLuts<Codec>::deq_chunk16s(u_row + boff, cch, lw, luts, sc);
    acc.y += sc *
        (dot(lw[0], a0) + dot(lw[1], a1) + dot(lw[2], a2) + dot(lw[3], a3));
  }
  return acc;
}

// Declares `name` and stages CodecT's decode LUTs into it (no-op, 16-byte
// stub array for table-free codecs). Gather threadgroups are (32, SG, 1)
// with SG simdgroups chosen by the host (kq_moe_pick_sg), so the staging
// stride comes from the launch.
#define KQ_EXT_STAGE_LUTS(CodecT, name)                                \
  threadgroup uint4 name##_v[(KqTgLuts<CodecT>::bytes + 15) / 16 + 1]; \
  threadgroup uint8_t* name =                                          \
      reinterpret_cast<threadgroup uint8_t*>(name##_v);                \
  if (KqTgLuts<CodecT>::bytes > 0) {                                   \
    KqTgLuts<CodecT>::stage(                                           \
        name, ushort(simd_gid * 32 + simd_lid), ushort(32 * tptg.y));  \
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
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX; // rows per simdgroup (tptg.y simdgroups per tg)
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int expert = indices[tid.z * R + tid.y];
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;

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

// Biased variants (gpt-oss experts): per-(expert, out_dim) f32 biases added
// to the accumulated dots before the epilogue / store. Only the uniform
// family carries biases; shexp/mix stay bias-free (no such checkpoints).
template <typename T, typename Codec, int ACT, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_moe_glu_gather_bias(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device float* gb [[buffer(2)]],
    const device float* ub [[buffer(3)]],
    const device T* x [[buffer(4)]],
    const device uint32_t* indices [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant float& limit [[buffer(9)]],
    const constant float& alpha [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int expert = indices[tid.z * R + tid.y];
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * R + tid.y) * N;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  const int64_t row = (int64_t)expert * N + out_row;
  // Bias loads issued before the dot-product loop so their DRAM latency
  // hides behind it instead of stalling the epilogue store.
  float gbv = 0.0f;
  float ubv = 0.0f;
  if (tx == 0) {
    gbv = gb[row];
    ubv = ub[row];
  }
  const float2 gu =
      kq_ext_glu_row_partial<T, Codec, NX>(gw, uw, x, row, K, tx, kq_luts);
  const float g = kq_ext_reduce<NX>(gu.x);
  const float u = kq_ext_reduce<NX>(gu.y);
  if (tx == 0) {
    out[out_row] =
        static_cast<T>(kq_glu_epilogue<ACT>(g + gbv, u + ubv, limit, alpha));
  }
}

template <typename T, typename Codec, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_gather_qmv_bias(
    const device uint8_t* w [[buffer(0)]],
    const device float* b [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int64_t row_idx = (int64_t)tid.z * R + tid.y;
  const int expert = indices[row_idx];
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;

  x += row_idx * K;
  out += row_idx * N;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  const int64_t row = (int64_t)expert * N + out_row;
  float bv = 0.0f;
  if (tx == 0) {
    bv = b[row];
  }
  const float r = kq_ext_reduce<NX>(
      kq_ext_row_partial<T, Codec, NX>(w, x, row, K, tx, kq_luts));
  if (tx == 0) {
    out[out_row] = static_cast<T>(r + bv);
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
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int64_t row_idx = (int64_t)tid.z * R + tid.y;
  const int expert = indices[row_idx];
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;

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
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int n_route = tpg.y - 1;
  const bool shared_slot = int(tid.y) == n_route;
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;

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
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;

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

// Slot-parallel mix_ns: same math as kq_ext_gather_qmv_mix_ns with the S
// slot dots spread across S simdgroup pairs instead of a per-thread loop.
// The loop kernel launches N / 8 threadgroups at decode (T = 1); the
// per-thread slot loop leaves the device underfilled and the solo op runs
// at ~2/3 of its cross-call-overlapped bandwidth -- chained probe calls
// recover the gap, the real serialized decode graph does not. Widening
// K-lanes (NX = 16/32) shortens per-thread chains and measured
// flat-to-negative; this mapping keeps the chunk chains at NX = 8 length
// and multiplies resident threads by S. Each simdgroup owns one
// (slot, row-half); raw lane partials stage through threadgroup memory and
// the slot-0 simdgroup pair replays the loop kernel's serial score-FMA
// chain and 3-step lane reduce, so outputs are bit-identical to it.
// Dispatch: group (32, 2 * S, 1), grid (N / (2 * RPS), 1, T); host gates
// S <= KQ_MOE_SP_MAX_S.
#define KQ_MOE_SP_MAX_S 16

template <typename T, typename Codec, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_gather_qmv_mix_ns_sp(
    const device uint8_t* w [[buffer(0)]],
    const device T* h [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    const device float* scores [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& S [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int slot = int(simd_gid >> 1);
  const short lrow = short((simd_gid & 1) * RPS) + ty;
  const int out_row = tid.x * (2 * RPS) + lrow;

  threadgroup uint4 kq_luts_v[(KqTgLuts<Codec>::bytes + 15) / 16 + 1];
  threadgroup uint8_t* kq_luts =
      reinterpret_cast<threadgroup uint8_t*>(kq_luts_v);
  if (KqTgLuts<Codec>::bytes > 0) {
    KqTgLuts<Codec>::stage(
        kq_luts, ushort(simd_gid * 32 + simd_lid), ushort(tptg.x * tptg.y));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float parts[2 * RPS][NX][KQ_MOE_SP_MAX_S];

  const int expert = int(indices[tid.z * S + slot]);
  const device T* xs = h + ((int64_t)tid.z * S + slot) * K;
  parts[lrow][tx][slot] = kq_ext_row_partial<T, Codec, NX>(
      w, xs, (int64_t)expert * N + out_row, K, tx, kq_luts);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (slot == 0) {
    float result = 0.0f;
    for (int s = 0; s < S; s++) {
      result += scores[tid.z * S + s] * parts[lrow][tx][s];
    }
    result = kq_ext_reduce<NX>(result);
    if (tx == 0) {
      out[(int64_t)tid.z * N + out_row] = static_cast<T>(result);
    }
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
    const constant int& SC [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  KQ_EXT_STAGE_LUTS(SCodec, kq_sluts)
  float result = 0.0f;
  for (int slot = 0; slot < S - 1; slot++) {
    const int expert = int(indices[tid.z * (S - 1) + slot]);
    const device T* xs = h + ((int64_t)tid.z * S + slot) * K;
    result += scores[tid.z * SC + slot] *
        kq_ext_row_partial<T, Codec, NX>(
                  w, xs, (int64_t)expert * N + out_row, K, tx, kq_luts);
  }
  {
    // SC == S - 1: the shared slot's weight is an implicit 1
    const device T* xs = h + ((int64_t)tid.z * S + (S - 1)) * K;
    result += (SC == S ? scores[tid.z * SC + (S - 1)] : 1.0f) *
        kq_ext_row_partial<T, SCodec, NX>(
                  sw, xs, (int64_t)out_row, K, tx, kq_sluts);
  }
  result = kq_ext_reduce<NX>(result);
  if (tx == 0) {
    out[(int64_t)tid.z * N + out_row] = static_cast<T>(result);
  }
}

// ---------------------------------------------------------------------------
// Dedupe gathers for the MTP verify widths (T = 2..8 rows per step). The
// rows of a verify block route to overlapping expert sets (GLM-5.3-Flash
// measured 3.65 of 8 shared between two consecutive rows), and the plain
// gathers dequantize a shared expert once per (row, slot) pair. These
// kernels pair the (row, slot) pairs that route to one expert: the first
// pair in row-major order owns the expert, dequantizes its weight row
// once and dots it against both pairs' activation rows; the second pair's
// threadgroup returns before staging. A third pair on the same expert
// starts a new owner (pairs only: two accumulators keep the register
// budget of the single-row kernels). Per row the chunk accumulation and
// the reduce keep the single-row order, so the GLU outputs are
// bit-identical to the plain kernels. The shared expert pairs rows the
// same way (row 0 with row 1, row 2 with row 3, ...).
//   moe_glu_gather_dd / moe_glu_gather_shexp_dd: grid (N-tiles, slots, T).
//   gather_qmv_mix_ns_dd / gather_qmv_mix_dd: row pairs on the down
//     projection (kq_ext_gather_qmv_mix_pair), the loop kernel's launch
//     over row pairs, bit-identical to the loop kernels.
// ---------------------------------------------------------------------------

// One threadgroup LUT buffer sized for the larger of two codecs; the
// branch that runs stages its own tables (branch conditions are
// threadgroup-uniform, so the barrier inside is safe).
#define KQ_EXT_DECL_LUTS2(CodecA, CodecB, name)            \
  threadgroup uint4 name##_v                               \
      [((KqTgLuts<CodecA>::bytes > KqTgLuts<CodecB>::bytes \
             ? KqTgLuts<CodecA>::bytes                     \
             : KqTgLuts<CodecB>::bytes) +                  \
        15) /                                              \
           16 +                                            \
       1];                                                 \
  threadgroup uint8_t* name = reinterpret_cast<threadgroup uint8_t*>(name##_v);
#define KQ_EXT_STAGE_INTO(CodecT, name)                               \
  if (KqTgLuts<CodecT>::bytes > 0) {                                  \
    KqTgLuts<CodecT>::stage(                                          \
        name, ushort(simd_gid * 32 + simd_lid), ushort(32 * tptg.y)); \
    threadgroup_barrier(mem_flags::mem_threadgroup);                  \
  }

// Pair ownership: pair p owns its expert when an even number of earlier
// pairs route to the same expert; it then also serves the next such pair.
METAL_FUNC int
kq_dd_partner(const device uint32_t* indices, int p, int P, uint32_t e) {
  int rank = 0;
  for (int q = 0; q < p; ++q) {
    rank += int(indices[q] == e);
  }
  if (rank & 1) {
    return -2; // served by the previous pair
  }
  for (int q = p + 1; q < P; ++q) {
    if (indices[q] == e) {
      return q;
    }
  }
  return -1; // sole owner
}

template <typename T, typename Codec, int NX>
METAL_FUNC float2 kq_ext_row_partial_pair(
    const device uint8_t* w,
    const device T* x0,
    const device T* x1,
    int64_t row,
    int K,
    short tx,
    const threadgroup uint8_t* luts) {
  constexpr short chpb = Codec::superblock / 16;
  const int nb = K / Codec::superblock;
  const device uint8_t* w_row = w + row * (int64_t)nb * Codec::block_bytes;
  float2 acc = float2(0.0f);
  for (int ich = tx; 16 * ich < K; ich += NX) {
    const device uint8_t* block =
        w_row + (int64_t)(ich / chpb) * Codec::block_bytes;
    float4x4 lw;
    float sc;
    KqTgLuts<Codec>::deq_chunk16s(block, short(ich % chpb), lw, luts, sc);
    const device T* xp = x0 + ich * 16;
    acc.x += sc *
        (dot(lw[0], float4(*(const device vec<T, 4>*)(xp + 0))) +
         dot(lw[1], float4(*(const device vec<T, 4>*)(xp + 4))) +
         dot(lw[2], float4(*(const device vec<T, 4>*)(xp + 8))) +
         dot(lw[3], float4(*(const device vec<T, 4>*)(xp + 12))));
    xp = x1 + ich * 16;
    acc.y += sc *
        (dot(lw[0], float4(*(const device vec<T, 4>*)(xp + 0))) +
         dot(lw[1], float4(*(const device vec<T, 4>*)(xp + 4))) +
         dot(lw[2], float4(*(const device vec<T, 4>*)(xp + 8))) +
         dot(lw[3], float4(*(const device vec<T, 4>*)(xp + 12))));
  }
  return acc;
}

// GLU pair form: (gate, up) for row x0 in .xy and for row x1 in .zw.
template <typename T, typename Codec, int NX>
METAL_FUNC float4 kq_ext_glu_row_partial_pair(
    const device uint8_t* gw,
    const device uint8_t* uw,
    const device T* x0,
    const device T* x1,
    int64_t row,
    int K,
    short tx,
    const threadgroup uint8_t* luts) {
  constexpr short chpb = Codec::superblock / 16;
  const int nb = K / Codec::superblock;
  const int64_t row_off = row * (int64_t)nb * Codec::block_bytes;
  const device uint8_t* g_row = gw + row_off;
  const device uint8_t* u_row = uw + row_off;
  float4 acc = float4(0.0f);
  for (int ich = tx; 16 * ich < K; ich += NX) {
    const int64_t boff = (int64_t)(ich / chpb) * Codec::block_bytes;
    const short cch = short(ich % chpb);
    const device T* xp = x0 + ich * 16;
    const float4 a0 = float4(*(const device vec<T, 4>*)(xp + 0));
    const float4 a1 = float4(*(const device vec<T, 4>*)(xp + 4));
    const float4 a2 = float4(*(const device vec<T, 4>*)(xp + 8));
    const float4 a3 = float4(*(const device vec<T, 4>*)(xp + 12));
    const device T* yp = x1 + ich * 16;
    float4x4 lw;
    float sc;
    KqTgLuts<Codec>::deq_chunk16s(g_row + boff, cch, lw, luts, sc);
    acc.x += sc *
        (dot(lw[0], a0) + dot(lw[1], a1) + dot(lw[2], a2) + dot(lw[3], a3));
    acc.z += sc *
        (dot(lw[0], float4(*(const device vec<T, 4>*)(yp + 0))) +
         dot(lw[1], float4(*(const device vec<T, 4>*)(yp + 4))) +
         dot(lw[2], float4(*(const device vec<T, 4>*)(yp + 8))) +
         dot(lw[3], float4(*(const device vec<T, 4>*)(yp + 12))));
    KqTgLuts<Codec>::deq_chunk16s(u_row + boff, cch, lw, luts, sc);
    acc.y += sc *
        (dot(lw[0], a0) + dot(lw[1], a1) + dot(lw[2], a2) + dot(lw[3], a3));
    acc.w += sc *
        (dot(lw[0], float4(*(const device vec<T, 4>*)(yp + 0))) +
         dot(lw[1], float4(*(const device vec<T, 4>*)(yp + 4))) +
         dot(lw[2], float4(*(const device vec<T, 4>*)(yp + 8))) +
         dot(lw[3], float4(*(const device vec<T, 4>*)(yp + 12))));
  }
  return acc;
}

template <typename T, typename Codec, int ACT, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_moe_glu_gather_dd(
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
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int p = int(tid.z) * R + int(tid.y);
  const uint32_t expert = indices[p];
  const int q = kq_dd_partner(indices, p, R * int(tpg.z), expert);
  if (q == -2) {
    return;
  }
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;
  const int64_t wrow = (int64_t)expert * N + out_row;
  x += (int64_t)tid.z * K;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  if (q < 0) {
    const float2 gu =
        kq_ext_glu_row_partial<T, Codec, NX>(gw, uw, x, wrow, K, tx, kq_luts);
    const float g = kq_ext_reduce<NX>(gu.x);
    const float u = kq_ext_reduce<NX>(gu.y);
    if (tx == 0) {
      out[(int64_t)p * N + out_row] =
          static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
    }
    return;
  }
  const device T* x1 = x + (int64_t)(q / R - int(tid.z)) * K;
  const float4 gu = kq_ext_glu_row_partial_pair<T, Codec, NX>(
      gw, uw, x, x1, wrow, K, tx, kq_luts);
  const float g0 = kq_ext_reduce<NX>(gu.x);
  const float u0 = kq_ext_reduce<NX>(gu.y);
  const float g1 = kq_ext_reduce<NX>(gu.z);
  const float u1 = kq_ext_reduce<NX>(gu.w);
  if (tx == 0) {
    out[(int64_t)p * N + out_row] =
        static_cast<T>(kq_glu_epilogue<ACT>(g0, u0, limit));
    out[(int64_t)q * N + out_row] =
        static_cast<T>(kq_glu_epilogue<ACT>(g1, u1, limit));
  }
}

template <
    typename T,
    typename Codec,
    typename SCodec,
    int ACT,
    int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_moe_glu_gather_shexp_dd(
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
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int n_route = tpg.y - 1;
  const int T_ = int(tpg.z);
  const bool shared_slot = int(tid.y) == n_route;
  // Output rows are (row, slot) over tpg.y slots; partner q is the other
  // output row served by this threadgroup (-1: none, -2: served elsewhere).
  const int p = int(tid.z) * int(tpg.y) + int(tid.y);
  int q = -1;
  int q_row = 0;
  uint32_t expert = 0;
  if (shared_slot) {
    if (int(tid.z) & 1) {
      return;
    }
    if (int(tid.z) + 1 < T_) {
      q_row = int(tid.z) + 1;
      q = q_row * int(tpg.y) + n_route;
    }
  } else {
    const int pr = int(tid.z) * n_route + int(tid.y);
    expert = indices[pr];
    const int qr = kq_dd_partner(indices, pr, n_route * T_, expert);
    if (qr == -2) {
      return;
    }
    if (qr >= 0) {
      q_row = qr / n_route;
      q = q_row * int(tpg.y) + (qr % n_route);
    }
  }
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;
  x += (int64_t)tid.z * K;

  const device T* x1 = q < 0 ? x : x + (int64_t)(q_row - int(tid.z)) * K;
  float4 gu = float4(0.0f);
  KQ_EXT_DECL_LUTS2(Codec, SCodec, kq_luts)
  if (shared_slot) {
    KQ_EXT_STAGE_INTO(SCodec, kq_luts)
    if (q < 0) {
      gu.xy = kq_ext_glu_row_partial<T, SCodec, NX>(
          sgw, suw, x, (int64_t)out_row, K, tx, kq_luts);
    } else {
      gu = kq_ext_glu_row_partial_pair<T, SCodec, NX>(
          sgw, suw, x, x1, (int64_t)out_row, K, tx, kq_luts);
    }
  } else {
    KQ_EXT_STAGE_INTO(Codec, kq_luts)
    const int64_t wrow = (int64_t)expert * N + out_row;
    if (q < 0) {
      gu.xy =
          kq_ext_glu_row_partial<T, Codec, NX>(gw, uw, x, wrow, K, tx, kq_luts);
    } else {
      gu = kq_ext_glu_row_partial_pair<T, Codec, NX>(
          gw, uw, x, x1, wrow, K, tx, kq_luts);
    }
  }
  const float g0 = kq_ext_reduce<NX>(gu.x);
  const float u0 = kq_ext_reduce<NX>(gu.y);
  const float g1 = kq_ext_reduce<NX>(gu.z);
  const float u1 = kq_ext_reduce<NX>(gu.w);
  if (tx == 0) {
    out[(int64_t)p * N + out_row] =
        static_cast<T>(kq_glu_epilogue<ACT>(g0, u0, limit));
    if (q >= 0) {
      out[(int64_t)q * N + out_row] =
          static_cast<T>(kq_glu_epilogue<ACT>(g1, u1, limit));
    }
  }
}

// Row-pair mix (mix_ns and the shexp-slot mix): the loop kernel's launch
// with each simdgroup serving rows (2z, 2z+1). Row 2z's slots run in slot
// order; a slot whose expert row 2z+1 also routes dots both activation
// rows against one dequantized weight chunk and parks the second row's
// lane partial by its slot. Row 2z+1's unshared slots follow, and its
// chain then runs in its own slot order, so both rows are bit-identical
// to the loop kernel. A row that routes one expert twice pairs its k-th
// copy with the other row's k-th copy. With SHEXP the last slot reads
// `sw` and pairs the two rows unconditionally.
// Dispatch: the loop kernel's group and grid with z = ceil(T / 2); host
// gates S <= KQ_MOE_SP_MAX_S.
template <typename T, typename Codec, typename SCodec, bool SHEXP, int NX>
[[kernel]] void kq_ext_gather_qmv_mix_pair(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* sw [[buffer(1)]],
    const device T* h [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    const device float* scores [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& S [[buffer(8)]],
    const constant int& SC [[buffer(9)]],
    const constant int& T_ [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;
  const int ra = 2 * int(tid.z);
  const int rb = ra + 1;
  const bool has_b = rb < T_;
  const int n_route = SHEXP ? S - 1 : S;

  KQ_EXT_STAGE_LUTS(Codec, kq_luts)
  threadgroup uint4
      kq_sluts_v[(SHEXP ? (KqTgLuts<SCodec>::bytes + 15) / 16 : 0) + 1];
  threadgroup uint8_t* kq_sluts =
      reinterpret_cast<threadgroup uint8_t*>(kq_sluts_v);
  if (SHEXP && KqTgLuts<SCodec>::bytes > 0) {
    KqTgLuts<SCodec>::stage(
        kq_sluts, ushort(simd_gid * 32 + simd_lid), ushort(32 * tptg.y));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float pb[KQ_MOE_SP_MAX_S + 1];
  uint claimed = 0;
  float result = 0.0f;
  for (int slot = 0; slot < n_route; slot++) {
    const int expert = int(indices[ra * n_route + slot]);
    const device T* xa = h + ((int64_t)ra * S + slot) * K;
    const int64_t wrow = (int64_t)expert * N + out_row;
    int sb = -1;
    if (has_b) {
      int k = 0;
      for (int u = 0; u < slot; ++u) {
        k += int(int(indices[ra * n_route + u]) == expert);
      }
      for (int u = 0; u < n_route; ++u) {
        if (int(indices[rb * n_route + u]) == expert) {
          if (k == 0) {
            sb = u;
            break;
          }
          --k;
        }
      }
    }
    if (sb >= 0) {
      const device T* xb = h + ((int64_t)rb * S + sb) * K;
      const float2 p = kq_ext_row_partial_pair<T, Codec, NX>(
          w, xa, xb, wrow, K, tx, kq_luts);
      result += scores[ra * SC + slot] * p.x;
      pb[sb] = p.y;
      claimed |= 1u << sb;
    } else {
      result += scores[ra * SC + slot] *
          kq_ext_row_partial<T, Codec, NX>(w, xa, wrow, K, tx, kq_luts);
    }
  }
  if (SHEXP) {
    // SC == S - 1: the shared slot's weight is an implicit 1
    const device T* xa = h + ((int64_t)ra * S + n_route) * K;
    const float wa = SC == S ? scores[ra * SC + n_route] : 1.0f;
    if (has_b) {
      const device T* xb = h + ((int64_t)rb * S + n_route) * K;
      const float2 p = kq_ext_row_partial_pair<T, SCodec, NX>(
          sw, xa, xb, (int64_t)out_row, K, tx, kq_sluts);
      result += wa * p.x;
      pb[n_route] = p.y;
    } else {
      result += wa *
          kq_ext_row_partial<T, SCodec, NX>(
                    sw, xa, (int64_t)out_row, K, tx, kq_sluts);
    }
  }
  result = kq_ext_reduce<NX>(result);
  if (tx == 0) {
    out[(int64_t)ra * N + out_row] = static_cast<T>(result);
  }
  if (!has_b) {
    return;
  }
  for (int slot = 0; slot < n_route; slot++) {
    if ((claimed >> slot) & 1u) {
      continue;
    }
    const int expert = int(indices[rb * n_route + slot]);
    const device T* xb = h + ((int64_t)rb * S + slot) * K;
    pb[slot] = kq_ext_row_partial<T, Codec, NX>(
        w, xb, (int64_t)expert * N + out_row, K, tx, kq_luts);
  }
  result = 0.0f;
  for (int slot = 0; slot < n_route; slot++) {
    result += scores[rb * SC + slot] * pb[slot];
  }
  if (SHEXP) {
    result += (SC == S ? scores[rb * SC + n_route] : 1.0f) * pb[n_route];
  }
  result = kq_ext_reduce<NX>(result);
  if (tx == 0) {
    out[(int64_t)rb * N + out_row] = static_cast<T>(result);
  }
}

// ---------------------------------------------------------------------------
// Router top-k (codec-independent float kernel): score E logits, pick the
// top R (min-index tie-break), emit gather-ready indices [T, R] uint32 and
// mix scores [T, R + SHARED] float32 in one dispatch. SCORING == 0 is f32
// softmax; SCORING == 1 is sqrt(softplus(x)) (deepseek-v4), which has no
// global normalizer and requires NORM (renorm adds the model's 1e-20 guard
// since scores may be exactly 0); SCORING == 2 is sigmoid (deepseek-v3 /
// glm5 noaux-tc routing), also per-expert, renorm with the same guard when
// NORM. HAS_BIAS ranks selection by
// score + bias[expert] (e_score_correction_bias) while emitted scores stay
// unbiased. When SHARED == 1, column E of each logits row is the
// shared-expert gate logit and its sigmoid lands in scores slot R
// (qwen3-next); SHARED == 0 is the plain-MoE form (gemma). NORM selects
// renormalizing the picked scores to sum to 1 (norm_topk_prob; for softmax
// also exactly softmax-over-the-selected-logits, the gemma router
// semantics). HAS_PES scales each picked score by pes[expert] (gemma's
// learned per_expert_scale); SCALE is a uniform multiplier on emitted
// routed scores (deepseek-v4 routed_scaling_factor). pes/bias may be any
// bound buffer when HAS_PES/HAS_BIAS == 0. One threadgroup of 256 threads
// per token; E <= 1024, R <= 16 (host-checked).
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
    const device float* bias [[buffer(9)]],
    const constant int& HAS_BIAS [[buffer(10)]],
    const constant int& SCORING [[buffer(11)]],
    const constant float& SCALE [[buffer(12)]],
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

  // SCORING is threadgroup-uniform, so barriers inside each arm are safe.
  float gsum = 1.0f;
  if (SCORING == 1) {
    for (int e = lid; e < E; e += NT) {
      const float x = float(lrow[e]);
      // Stable softplus: max(x, 0) + log1p(exp(-|x|)); series for tiny z
      // where log(1 + z) loses bits.
      const float z = metal::exp(-metal::abs(x));
      const float l1p =
          (z < 1e-4f) ? metal::fma(-0.5f * z, z, z) : metal::log(1.0f + z);
      p[e] = metal::sqrt(metal::max(x, 0.0f) + l1p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else if (SCORING == 2) {
    for (int e = lid; e < E; e += NT) {
      p[e] = 1.0f / (1.0f + metal::exp(-float(lrow[e])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
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
    gsum = stat[1];
  }

  for (int r = 0; r < R; r++) {
    // -inf init/sentinel: biased ranking values can be negative.
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (int e = lid; e < E; e += NT) {
      float v = p[e];
      if (HAS_BIAS) {
        v += bias[e];
      }
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
      // Emitted score is the unbiased p[wi] (== wv when HAS_BIAS == 0).
      win_v[r] = p[wi];
      win_i[r] = wi;
      p[wi] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    float ps = 0;
    for (int r = 0; r < R; r++) {
      ps += win_v[r];
    }
    const float denom = NORM ? (SCORING != 0 ? ps + 1e-20f : ps) : gsum;
    device float* srow = scores + (int64_t)tid * (R + SHARED);
    for (int r = 0; r < R; r++) {
      float sv = win_v[r] / denom;
      if (HAS_PES) {
        sv *= pes[win_i[r]];
      }
      srow[r] = sv * SCALE;
    }
    if (SHARED) {
      srow[R] = 1.0f / (1.0f + metal::exp(-float(lrow[E])));
    }
  }
}

// ---------------------------------------------------------------------------
// Half-dot Ext gathers (KQ_MOE_HALF=1, grid codecs only; see KqTgLutsH).
// Same thread mapping and launch shape as the Ext kernels above. Three
// changes per chunk: the grid entry comes back as half4 pairs (no u64 byte
// unpack), the activation row is read from a half copy staged once per
// threadgroup in dynamic threadgroup memory (buffer threadgroup(0), K halves,
// host-sized), and the four 4-wide dots run in half with a float sum per
// chunk before the chunk scale. Expert rows only: the shared-expert slot in
// the shexp / mix kernels keeps the float path over the device activation
// row, so the shexp numerics are unchanged. Outputs differ from the float
// kernels at the half rounding level (about 1e-3 relative on the fused GLU
// output at K=4096, against a bf16 floor near 9e-4).
// ---------------------------------------------------------------------------

#define KQ_EXT_STAGE_LUTS_H(CodecT, name)                               \
  threadgroup uint4 name##_v[(KqTgLutsH<CodecT>::bytes + 15) / 16 + 1]; \
  threadgroup uint8_t* name =                                           \
      reinterpret_cast<threadgroup uint8_t*>(name##_v);                 \
  if (KqTgLutsH<CodecT>::bytes > 0) {                                   \
    KqTgLutsH<CodecT>::stage(                                           \
        name, ushort(simd_gid * 32 + simd_lid), ushort(32 * tptg.y));   \
  }

// Stages one K-wide activation row as half4 (caller barriers).
template <typename T>
METAL_FUNC void kq_ext_stage_xh(
    threadgroup half4* xs,
    const device T* x,
    int K,
    ushort lin,
    ushort n_threads) {
  const device vec<T, 4>* x4 = reinterpret_cast<const device vec<T, 4>*>(x);
  for (int i = lin; i < K / 4; i += n_threads) {
    xs[i] = half4(x4[i]);
  }
}

template <typename Codec, int NX>
METAL_FUNC float kq_ext_row_partial_h(
    const device uint8_t* w,
    const threadgroup half4* xs,
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
    half4x4 lw;
    float sc;
    KqTgLutsH<Codec>::deq_chunk16h(block, short(ich % chpb), lw, luts, sc);
    const threadgroup half4* xp = xs + ich * 4;
    acc += sc *
        (float(dot(lw[0], xp[0])) + float(dot(lw[1], xp[1])) +
         float(dot(lw[2], xp[2])) + float(dot(lw[3], xp[3])));
  }
  return acc;
}

template <typename Codec, int NX>
METAL_FUNC float2 kq_ext_glu_row_partial_h(
    const device uint8_t* gw,
    const device uint8_t* uw,
    const threadgroup half4* xs,
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
    const threadgroup half4* xp = xs + ich * 4;
    const half4 a0 = xp[0];
    const half4 a1 = xp[1];
    const half4 a2 = xp[2];
    const half4 a3 = xp[3];
    half4x4 lw;
    float sc;
    KqTgLutsH<Codec>::deq_chunk16h(g_row + boff, cch, lw, luts, sc);
    acc.x += sc *
        (float(dot(lw[0], a0)) + float(dot(lw[1], a1)) + float(dot(lw[2], a2)) +
         float(dot(lw[3], a3)));
    KqTgLutsH<Codec>::deq_chunk16h(u_row + boff, cch, lw, luts, sc);
    acc.y += sc *
        (float(dot(lw[0], a0)) + float(dot(lw[1], a1)) + float(dot(lw[2], a2)) +
         float(dot(lw[3], a3)));
  }
  return acc;
}

template <typename T, typename Codec, int ACT, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_moe_glu_gather_h(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant float& limit [[buffer(7)]],
    threadgroup half4* xs [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int expert = indices[tid.z * R + tid.y];
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;
  const ushort lin = ushort(simd_gid * 32 + simd_lid);
  const ushort nth = ushort(32 * tptg.y);

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * R + tid.y) * N;

  KQ_EXT_STAGE_LUTS_H(Codec, kq_luts)
  kq_ext_stage_xh<T>(xs, x, K, lin, nth);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float2 gu = kq_ext_glu_row_partial_h<Codec, NX>(
      gw, uw, xs, (int64_t)expert * N + out_row, K, tx, kq_luts);
  const float g = kq_ext_reduce<NX>(gu.x);
  const float u = kq_ext_reduce<NX>(gu.y);
  if (tx == 0) {
    out[out_row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
  }
}

template <typename T, typename Codec, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_gather_qmv_h(
    const device uint8_t* w [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    threadgroup half4* xs [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int R = tpg.y;
  const int64_t row_idx = (int64_t)tid.z * R + tid.y;
  const int expert = indices[row_idx];
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;
  const ushort lin = ushort(simd_gid * 32 + simd_lid);
  const ushort nth = ushort(32 * tptg.y);

  x += row_idx * K;
  out += row_idx * N;

  KQ_EXT_STAGE_LUTS_H(Codec, kq_luts)
  kq_ext_stage_xh<T>(xs, x, K, lin, nth);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = kq_ext_reduce<NX>(kq_ext_row_partial_h<Codec, NX>(
      w, xs, (int64_t)expert * N + out_row, K, tx, kq_luts));
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
[[kernel]] void kq_ext_moe_glu_gather_shexp_h(
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
    threadgroup half4* xs [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int n_route = tpg.y - 1;
  const bool shared_slot = int(tid.y) == n_route;
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;
  const ushort lin = ushort(simd_gid * 32 + simd_lid);
  const ushort nth = ushort(32 * tptg.y);

  x += (int64_t)tid.z * K;
  out += ((int64_t)tid.z * tpg.y + tid.y) * N;

  KQ_EXT_STAGE_LUTS_H(Codec, kq_luts)
  KQ_EXT_STAGE_LUTS(SCodec, kq_sluts)
  float2 gu;
  if (shared_slot) {
    // Shared expert: float path, device activations (numerics unchanged).
    gu = kq_ext_glu_row_partial<T, SCodec, NX>(
        sgw, suw, x, (int64_t)out_row, K, tx, kq_sluts);
  } else {
    kq_ext_stage_xh<T>(xs, x, K, lin, nth);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int expert = int(indices[tid.z * n_route + tid.y]);
    gu = kq_ext_glu_row_partial_h<Codec, NX>(
        gw, uw, xs, (int64_t)expert * N + out_row, K, tx, kq_luts);
  }
  const float g = kq_ext_reduce<NX>(gu.x);
  const float u = kq_ext_reduce<NX>(gu.y);
  if (tx == 0) {
    out[out_row] = static_cast<T>(kq_glu_epilogue<ACT>(g, u, limit));
  }
}

// Score-mixed down gather with the shared-expert slot folded in: the routed
// slots restage their h row per slot (two barriers each) and dot in half;
// the shared slot stays on the float path.
template <typename T, typename Codec, typename SCodec, int NX = KQ_EXT_NXPSG>
[[kernel]] void kq_ext_gather_qmv_mix_h(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* sw [[buffer(1)]],
    const device T* h [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    const device float* scores [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& S [[buffer(8)]],
    const constant int& SC [[buffer(9)]],
    threadgroup half4* xs [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const short ty = short(simd_lid / NX);
  const int out_row = tid.x * int(tptg.y) * RPS + int(simd_gid) * RPS + ty;
  const ushort lin = ushort(simd_gid * 32 + simd_lid);
  const ushort nth = ushort(32 * tptg.y);

  KQ_EXT_STAGE_LUTS_H(Codec, kq_luts)
  KQ_EXT_STAGE_LUTS(SCodec, kq_sluts)
  float result = 0.0f;
  for (int slot = 0; slot < S - 1; slot++) {
    const int expert = int(indices[tid.z * (S - 1) + slot]);
    const device T* hs = h + ((int64_t)tid.z * S + slot) * K;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kq_ext_stage_xh<T>(xs, hs, K, lin, nth);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    result += scores[tid.z * SC + slot] *
        kq_ext_row_partial_h<Codec, NX>(
                  w, xs, (int64_t)expert * N + out_row, K, tx, kq_luts);
  }
  {
    const device T* hs = h + ((int64_t)tid.z * S + (S - 1)) * K;
    result += (SC == S ? scores[tid.z * SC + (S - 1)] : 1.0f) *
        kq_ext_row_partial<T, SCodec, NX>(
                  sw, hs, (int64_t)out_row, K, tx, kq_sluts);
  }
  result = kq_ext_reduce<NX>(result);
  if (tx == 0) {
    out[(int64_t)tid.z * N + out_row] = static_cast<T>(result);
  }
}
