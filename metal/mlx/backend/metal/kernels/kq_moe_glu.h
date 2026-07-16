// Fused MoE gather kernels for MLX native-fp codecs (mxfp4), decode-shaped
// (one activation row per gathered expert row). Three kernels:
//
//   kq_moe_glu_gather:  out = glu(gate(x), up(x)) in ONE dispatch -- both
//     expert matvecs share each activation load, biases are fused, and the
//     clamped-SwiGLU epilogue replaces the gather/add/clip/sigmoid/mul chain
//     (out = (min(g, limit) * sigmoid(alpha * g)) * (clip(u, +-limit) + 1)).
//   kq_gather_qmv_bias:  a gathered matvec with the expert bias fused.
//   kq_gather_qmv_mix_bias:  gather_qmv_bias with the routing mix folded in
//     (out[t] = sum_s scores[t,s] * (W[e_s] @ x[t,s] + b[e_s]), all slots
//     accumulated in f32), replacing gather + (y * scores).sum(-2).
//
// All read MLX's packed layout (uint32 nibbles + uint8 E8M0 group scales,
// group_size 32, bits 4). The inner loop runs unguarded full blocks
// (32 lanes x 16 values) plus one lane-guarded tail block, so any K % 32 == 0
// runs at full speed -- stock's fast gather requires K % 512 == 0, which e.g.
// K = 2880 fails, dropping every such gather onto the guarded-slow variant.
//
// Grid: (N / 8, R, T) threadgroups of (32, 2, 1); R = expert slots per token,
// T = tokens (mix_bias: (N / 8, 1, T), the slot loop runs inside). Each
// threadgroup computes 8 output rows (2 simdgroups x 4). The qmv kernels
// take results_per_simdgroup: the "_fine" variants run 1 (2 rows per
// threadgroup, grid N / 2 -> 4x threadgroups) to fill the device on starved
// decode grids; per-lane work and simd_sum order are unchanged, so fine
// output is bit-identical.
//
// The load / E8M0-scale / fp4-dot primitives are self-contained (kq_ prefix):
// the stock fp_quantized.h copies collide with this repo's quantized_utils.h
// shims, and the decode must stay bit-identical to MLX's mode="mxfp4" path
// (OCP E2M1 value table, one E8M0 scale per 32 values).

constant float kq_fp4_e2m1_lut[16] = {
    0.0f,
    0.5f,
    1.0f,
    1.5f,
    2.0f,
    3.0f,
    4.0f,
    6.0f,
    -0.0f,
    -0.5f,
    -1.0f,
    -1.5f,
    -2.0f,
    -3.0f,
    -4.0f,
    -6.0f};

// 2^(e-127) built from bits, matching MLX's fp8_e8m0 float conversion:
// metal::exp2 is fast-math and lands ulps off on some compiler versions.
// e == 0 is the f32 subnormal 2^-127.
inline float kq_e8m0_scale(uint8_t e) {
  return as_type<float>(e == 0 ? 0x00400000u : uint(e) << 23);
}

template <typename T, typename U, int VPT>
inline void kq_load_vector(const device T* x, thread U* xt) {
  for (int i = 0; i < VPT; i++) {
    xt[i] = static_cast<U>(x[i]);
  }
}

template <typename U, int VPT>
inline U kq_fp4_qdot(const device uint8_t* w, const thread U* xt, U scale) {
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < VPT / 4; i++) {
    accum += xt[4 * i + 0] * kq_fp4_e2m1_lut[(ws[i] >> 0) & 0xF] +
        xt[4 * i + 1] * kq_fp4_e2m1_lut[(ws[i] >> 4) & 0xF] +
        xt[4 * i + 2] * kq_fp4_e2m1_lut[(ws[i] >> 8) & 0xF] +
        xt[4 * i + 3] * kq_fp4_e2m1_lut[(ws[i] >> 12) & 0xF];
  }
  return scale * accum;
}

template <typename T>
[[kernel]] void kq_moe_glu_gather(
    const device uint32_t* gw [[buffer(0)]],
    const device uint8_t* gs [[buffer(1)]],
    const device float* gb [[buffer(2)]],
    const device uint32_t* uw [[buffer(3)]],
    const device uint8_t* us [[buffer(4)]],
    const device float* ub [[buffer(5)]],
    const device T* x [[buffer(6)]],
    const device uint32_t* indices [[buffer(7)]],
    device T* out [[buffer(8)]],
    const constant int& K [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant float& alpha [[buffer(11)]],
    const constant float& limit [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  const int R = tpg.y;
  const int expert = indices[tid.z * R + tid.y];
  const int out_row = tid.x * 8 + simd_gid * results_per_simdgroup;

  const int w_row_bytes = K / 2; // 4-bit nibbles
  const int g_row = K / 32; // one E8M0 scale per 32 values

  const size_t row0 = (size_t)expert * N + out_row;
  const device uint8_t* wsg =
      (const device uint8_t*)gw + row0 * w_row_bytes + simd_lid * 8;
  const device uint8_t* wsu =
      (const device uint8_t*)uw + row0 * w_row_bytes + simd_lid * 8;
  const device uint8_t* slg = gs + row0 * g_row + simd_lid / 2;
  const device uint8_t* slu = us + row0 * g_row + simd_lid / 2;
  x += (size_t)tid.z * K + simd_lid * values_per_thread;

  thread U x_thread[values_per_thread];
  thread U rg[results_per_simdgroup] = {0};
  thread U ru[results_per_simdgroup] = {0};

  const int n_full = K / block_size;
  const int rem = K % block_size;

  for (int b = 0; b < n_full; b++) {
    kq_load_vector<T, U, values_per_thread>(x, x_thread);
    for (int r = 0; r < results_per_simdgroup; r++) {
      U sg_ = kq_e8m0_scale(slg[r * g_row]);
      U su_ = kq_e8m0_scale(slu[r * g_row]);
      rg[r] += kq_fp4_qdot<U, values_per_thread>(
          wsg + r * w_row_bytes, x_thread, sg_);
      ru[r] += kq_fp4_qdot<U, values_per_thread>(
          wsu + r * w_row_bytes, x_thread, su_);
    }
    wsg += block_size / 2;
    wsu += block_size / 2;
    slg += block_size / 32;
    slu += block_size / 32;
    x += block_size;
  }
  // Lane-guarded tail: rem is a multiple of 32, so any lane whose 16 values
  // start inside it is fully in range.
  if (rem > 0 && int(simd_lid) * values_per_thread < rem) {
    kq_load_vector<T, U, values_per_thread>(x, x_thread);
    for (int r = 0; r < results_per_simdgroup; r++) {
      U sg_ = kq_e8m0_scale(slg[r * g_row]);
      U su_ = kq_e8m0_scale(slu[r * g_row]);
      rg[r] += kq_fp4_qdot<U, values_per_thread>(
          wsg + r * w_row_bytes, x_thread, sg_);
      ru[r] += kq_fp4_qdot<U, values_per_thread>(
          wsu + r * w_row_bytes, x_thread, su_);
    }
  }

  out += ((size_t)tid.z * R + tid.y) * N + out_row;
  for (int r = 0; r < results_per_simdgroup; r++) {
    U g = simd_sum(rg[r]);
    U u = simd_sum(ru[r]);
    if (simd_lid == 0) {
      g += gb[row0 + r];
      u += ub[row0 + r];
      g = min(g, (U)limit);
      u = metal::clamp(u, -(U)limit, (U)limit);
      const U sig = 1.0f / (1.0f + metal::exp(-alpha * g));
      out[r] = static_cast<T>((g * sig) * (u + 1.0f));
    }
  }
}

template <typename T, int results_per_simdgroup = 4>
[[kernel]] void kq_gather_qmv_bias(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* s [[buffer(1)]],
    const device float* bias [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  const int R = tpg.y;
  const size_t row_idx = (size_t)tid.z * R + tid.y;
  const int expert = indices[row_idx];
  const int out_row =
      tid.x * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;

  const int w_row_bytes = K / 2;
  const int g_row = K / 32;

  const size_t row0 = (size_t)expert * N + out_row;
  const device uint8_t* ws =
      (const device uint8_t*)w + row0 * w_row_bytes + simd_lid * 8;
  const device uint8_t* sl = s + row0 * g_row + simd_lid / 2;
  x += row_idx * K + simd_lid * values_per_thread;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int n_full = K / block_size;
  const int rem = K % block_size;

  for (int b = 0; b < n_full; b++) {
    kq_load_vector<T, U, values_per_thread>(x, x_thread);
    for (int r = 0; r < results_per_simdgroup; r++) {
      U sc = kq_e8m0_scale(sl[r * g_row]);
      result[r] +=
          kq_fp4_qdot<U, values_per_thread>(ws + r * w_row_bytes, x_thread, sc);
    }
    ws += block_size / 2;
    sl += block_size / 32;
    x += block_size;
  }
  if (rem > 0 && int(simd_lid) * values_per_thread < rem) {
    kq_load_vector<T, U, values_per_thread>(x, x_thread);
    for (int r = 0; r < results_per_simdgroup; r++) {
      U sc = kq_e8m0_scale(sl[r * g_row]);
      result[r] +=
          kq_fp4_qdot<U, values_per_thread>(ws + r * w_row_bytes, x_thread, sc);
    }
  }

  out += row_idx * N + out_row;
  for (int r = 0; r < results_per_simdgroup; r++) {
    U v = simd_sum(result[r]);
    if (simd_lid == 0) {
      out[r] = static_cast<T>(v + bias[row0 + r]);
    }
  }
}

template <typename T, int results_per_simdgroup = 4>
[[kernel]] void kq_gather_qmv_mix_bias(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* s [[buffer(1)]],
    const device float* bias [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    const device float* scores [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& S [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * 32;
  typedef float U;

  const int out_row =
      tid.x * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;
  const int w_row_bytes = K / 2;
  const int g_row = K / 32;
  const int n_full = K / block_size;
  const int rem = K % block_size;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  for (int slot = 0; slot < S; slot++) {
    const size_t row_idx = (size_t)tid.z * S + slot;
    const int expert = indices[row_idx];
    const U score = U(scores[row_idx]);
    const size_t row0 = (size_t)expert * N + out_row;
    const device uint8_t* ws =
        (const device uint8_t*)w + row0 * w_row_bytes + simd_lid * 8;
    const device uint8_t* sl = s + row0 * g_row + simd_lid / 2;
    const device T* xs = x + row_idx * K + simd_lid * values_per_thread;

    thread U acc[results_per_simdgroup] = {0};
    for (int b = 0; b < n_full; b++) {
      kq_load_vector<T, U, values_per_thread>(xs, x_thread);
      for (int r = 0; r < results_per_simdgroup; r++) {
        U sc = kq_e8m0_scale(sl[r * g_row]);
        acc[r] += kq_fp4_qdot<U, values_per_thread>(
            ws + r * w_row_bytes, x_thread, sc);
      }
      ws += block_size / 2;
      sl += block_size / 32;
      xs += block_size;
    }
    if (rem > 0 && int(simd_lid) * values_per_thread < rem) {
      kq_load_vector<T, U, values_per_thread>(xs, x_thread);
      for (int r = 0; r < results_per_simdgroup; r++) {
        U sc = kq_e8m0_scale(sl[r * g_row]);
        acc[r] += kq_fp4_qdot<U, values_per_thread>(
            ws + r * w_row_bytes, x_thread, sc);
      }
    }
    for (int r = 0; r < results_per_simdgroup; r++) {
      result[r] += score * acc[r];
    }
    // Bias enters once per slot; lane 0 carries it into the final simd_sum.
    if (simd_lid == 0) {
      for (int r = 0; r < results_per_simdgroup; r++) {
        result[r] += score * bias[row0 + r];
      }
    }
  }

  out += (size_t)tid.z * N + out_row;
  for (int r = 0; r < results_per_simdgroup; r++) {
    U v = simd_sum(result[r]);
    if (simd_lid == 0) {
      out[r] = static_cast<T>(v);
    }
  }
}
