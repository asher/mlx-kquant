// Fused MoE gathers on the tuned qmv_fast inner loops ("qf" kernels).
//
// The Ext kernels (kq_moe_glu_kq.h) decode 16-weight chunks through generic
// codec traits. For the codecs below that decode costs more than the
// per-codec qmv_fast loops of kq_quantized*.h. The qf kernels run those
// loops unchanged: each call computes one simdgroup's rows of one expert
// matrix and returns the reduced row sums (the impl's kRows form). Gate and
// up are two calls on the same rows, the score-mixed down loops or spreads
// the slot calls, and the GLU epilogue, biases, output scales and mix
// weights apply as in the Ext kernels.
//
// Launch: threadgroups of (32, 2, 1), each computing 2 * rps output rows,
// grid (N / (2 * rps), slots, T). The slot-parallel mixes take (32, 2 * S, 1)
// with one simdgroup pair per slot, (32, 2 * (S - 1), 1) when the shared slot
// came from the dense route. The host keeps N % (2 * rps) == 0 and K
// a whole number of blocks (of 256 for a q8_0 shared expert). Compiled in
// kq_moe_qf.metal and, with output scales, kq_moe_qf_os.metal, after
// kq_moe_glu_kq.h for its epilogue and output-scale hooks.

// One codec's qmv_fast loop as a row core: rows<T, R>(w, x, ...) writes the
// sums of rows [2 R tile + R sg, + R) of the matrix at w against x to res,
// on every lane.
#define KQ_QF_CORE(NAME, FN, GS, BITS, RPS, BB)                  \
  struct NAME {                                                  \
    MLX_MTL_CONST int rps = RPS;                                 \
    static METAL_FUNC int64_t row_bytes(int K) {                 \
      return (int64_t)(K / GS) * BB;                             \
    }                                                            \
    template <typename T, int R>                                 \
    static METAL_FUNC void rows(                                 \
        const device uint8_t* w,                                 \
        const device T* x,                                       \
        const constant int& K,                                   \
        const constant int& N,                                   \
        uint tile,                                               \
        uint sg,                                                 \
        uint lid,                                                \
        thread float* res) {                                     \
      FN<T, GS, BITS, R, true>(                                  \
          w, x, nullptr, K, N, uint3(0, tile, 0), sg, lid, res); \
    }                                                            \
  };

// A core whose loop has the two-row form (kVM == 2): rows2 also dots x1.
#define KQ_QF_CORE_PAIR(NAME, FN, GS, BITS, RPS, BB)                  \
  struct NAME {                                                       \
    MLX_MTL_CONST int rps = RPS;                                      \
    static METAL_FUNC int64_t row_bytes(int K) {                      \
      return (int64_t)(K / GS) * BB;                                  \
    }                                                                 \
    template <typename T, int R>                                      \
    static METAL_FUNC void rows(                                      \
        const device uint8_t* w,                                      \
        const device T* x,                                            \
        const constant int& K,                                        \
        const constant int& N,                                        \
        uint tile,                                                    \
        uint sg,                                                      \
        uint lid,                                                     \
        thread float* res) {                                          \
      FN<T, GS, BITS, R, true>(                                       \
          w, x, nullptr, K, N, uint3(0, tile, 0), sg, lid, res);      \
    }                                                                 \
    template <typename T, int R>                                      \
    static METAL_FUNC void rows2(                                     \
        const device uint8_t* w,                                      \
        const device T* x0,                                           \
        const device T* x1,                                           \
        const constant int& K,                                        \
        const constant int& N,                                        \
        uint tile,                                                    \
        uint sg,                                                      \
        uint lid,                                                     \
        thread float* res) {                                          \
      FN<T, GS, BITS, R, true, 2>(                                    \
          w, x0, nullptr, K, N, uint3(0, tile, 0), sg, lid, res, x1); \
    }                                                                 \
  };

// clang-format off
KQ_QF_CORE(KqQfQ4_0, kq_q4_0_qmv_fast_impl, KQ_Q4_0_GROUP, 4, 4, KQ_Q4_0_BLOCK_BYTES)
KQ_QF_CORE(KqQfQ4_1, kq_q4_1_qmv_fast_impl, KQ_Q4_1_GROUP, 4, 4, KQ_Q4_1_BLOCK_BYTES)
KQ_QF_CORE_PAIR(KqQfQ5_0, kq_q5_0_qmv_fast_impl, KQ_Q5_0_GROUP, 5, 4, KQ_Q5_0_BLOCK_BYTES)
KQ_QF_CORE_PAIR(KqQfQ5_1, kq_q5_1_qmv_fast_impl, KQ_Q5_1_GROUP, 5, 4, KQ_Q5_1_BLOCK_BYTES)
KQ_QF_CORE_PAIR(KqQfIq4_nl, kq_iq4_nl_qmv_impl, KQ_IQ4_NL_GROUP, 4, 4, KQ_IQ4_NL_BLOCK_BYTES)
KQ_QF_CORE(KqQfQ2_K, kq_q2_k_qmv_fast_impl, KQ_Q2_K_SUPERBLOCK, 2, 2, KQ_Q2_K_BLOCK_BYTES)
KQ_QF_CORE(KqQfQ3_K, kq_q3_k_qmv_fast_impl, KQ_Q3_K_SUPERBLOCK, 3, 2, KQ_Q3_K_BLOCK_BYTES)
KQ_QF_CORE(KqQfQ4_K, kq_q4_k_qmv_fast_impl, KQ_Q4_K_SUPERBLOCK, 4, 2, KQ_Q4_K_BLOCK_BYTES)
KQ_QF_CORE(KqQfQ5_K, kq_q5_k_qmv_fast_impl, KQ_Q5_K_SUPERBLOCK, 5, 2, KQ_Q5_K_BLOCK_BYTES)
KQ_QF_CORE(KqQfQ6_K, kq_q6_k_qmv_fast_impl, KQ_Q6_K_SUPERBLOCK, 6, 4, KQ_Q6_K_BLOCK_BYTES)
KQ_QF_CORE_PAIR(KqQfIq4_xs, kq_iq4_xs_qmv_impl, KQ_IQ4_XS_SUPERBLOCK, 4, 2, KQ_IQ4_XS_BLOCK_BYTES)
KQ_QF_CORE_PAIR(KqQfMxfp4, kq_mxfp4_qmv_impl, KQ_MXFP4_GROUP, 4, 4, KQ_MXFP4_BLOCK_BYTES)
KQ_QF_CORE_PAIR(KqQfQ2_0, kq_q2_0_qmv_impl, KQ_Q2_0_SUPERBLOCK, 2, 2, KQ_Q2_0_BLOCK_BYTES)
// clang-format on

// q8_0 and nvfp4 take one more argument ahead of the capture; nvfp4 pairs.
struct KqQfQ8_0 {
  MLX_MTL_CONST int rps = 4;
  static METAL_FUNC int64_t row_bytes(int K) {
    return (int64_t)(K / KQ_Q8_0_GROUP) * KQ_Q8_0_BLOCK_BYTES;
  }
  template <typename T, int R>
  static METAL_FUNC void rows(
      const device uint8_t* w,
      const device T* x,
      const constant int& K,
      const constant int& N,
      uint tile,
      uint sg,
      uint lid,
      thread float* res) {
    kq_q8_0_qmv_fast_impl<T, KQ_Q8_0_GROUP, 8, R, true>(
        w, x, nullptr, K, N, uint3(0, tile, 0), sg, lid, nullptr, res);
  }
};

// The output scale applies in the qf kernels (KQ_MOE_MUL), not the impl.
struct KqQfNvfp4 {
  MLX_MTL_CONST int rps = 4;
  static METAL_FUNC int64_t row_bytes(int K) {
    return (int64_t)(K / KQ_NVFP4_SUPERBLOCK) * KQ_NVFP4_BLOCK_BYTES;
  }
  template <typename T, int R>
  static METAL_FUNC void rows(
      const device uint8_t* w,
      const device T* x,
      const constant int& K,
      const constant int& N,
      uint tile,
      uint sg,
      uint lid,
      thread float* res) {
    kq_nvfp4_qmv_impl<T, KQ_NVFP4_SUPERBLOCK, 4, R, false, true>(
        w, x, nullptr, K, N, uint3(0, tile, 0), sg, lid, nullptr, res);
  }
  template <typename T, int R>
  static METAL_FUNC void rows2(
      const device uint8_t* w,
      const device T* x0,
      const device T* x1,
      const constant int& K,
      const constant int& N,
      uint tile,
      uint sg,
      uint lid,
      thread float* res) {
    kq_nvfp4_qmv_impl<T, KQ_NVFP4_SUPERBLOCK, 4, R, false, true, 2>(
        w, x0, nullptr, K, N, uint3(0, tile, 0), sg, lid, nullptr, res, x1);
  }
};

// out = act(gate(x)) * up(x) per routed slot; grid (N / (2 rps), R, T).
template <typename T, typename C, int ACT>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_moe_glu_gather)(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant float& limit [[buffer(7)]],
    KQ_GLU_ALPHA_ARG,
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_GLU) {
  constexpr int RPS = C::rps;
  const int R = tpg.y;
  const uint expert = indices[tid.z * R + tid.y];
  const int64_t off = (int64_t)expert * N * C::row_bytes(K);
  x += (int64_t)tid.z * K;
  float g[RPS] = {0};
  float u[RPS] = {0};
  C::template rows<T, RPS>(gw + off, x, K, N, tid.x, simd_gid, simd_lid, g);
  C::template rows<T, RPS>(uw + off, x, K, N, tid.x, simd_gid, simd_lid, u);
  if (simd_lid == 0) {
    const int row0 = tid.x * 2 * RPS + simd_gid * RPS;
    out += ((int64_t)tid.z * R + tid.y) * N + row0;
    for (int i = 0; i < RPS; i++) {
      KQ_MOE_DO(
          const float sg_ = kq_moe_row_scale(out_scale, expert, N, row0 + i);
          const float su_ = kq_moe_row_scale(up_scale, expert, N, row0 + i);)
      out[i] = static_cast<T>(kq_glu_epilogue<ACT>(
          KQ_MOE_MUL(g[i], sg_), KQ_MOE_MUL(u[i], su_), limit, KQ_GLU_ALPHA));
    }
  }
}

// Dedupe GLU for the MTP verify widths (the pairing of
// kq_ext_moe_glu_gather_dd): of two (row, slot) pairs on one expert, the
// first decodes the expert's rows once for both activation rows (the core's
// rows2) and the second returns. Each row's sums are the one-row loop's, so
// the outputs are bit-identical to kq_qf_moe_glu_gather; same launch.
template <typename T, typename C, int ACT>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_moe_glu_gather_dd)(
    const device uint8_t* gw [[buffer(0)]],
    const device uint8_t* uw [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant float& limit [[buffer(7)]],
    KQ_GLU_ALPHA_ARG,
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_GLU) {
  constexpr int RPS = C::rps;
  const int R = tpg.y;
  const int p = tid.z * R + tid.y;
  const uint expert = indices[p];
  const int q = kq_dd_partner(indices, p, R * int(tpg.z), expert);
  if (q == -2) {
    return;
  }
  const int64_t off = (int64_t)expert * N * C::row_bytes(K);
  const device T* xp = x + (int64_t)tid.z * K;
  float g[2 * RPS] = {0};
  float u[2 * RPS] = {0};
  if (q < 0) {
    C::template rows<T, RPS>(gw + off, xp, K, N, tid.x, simd_gid, simd_lid, g);
    C::template rows<T, RPS>(uw + off, xp, K, N, tid.x, simd_gid, simd_lid, u);
  } else {
    const device T* xq = x + (int64_t)(q / R) * K;
    C::template rows2<T, RPS>(
        gw + off, xp, xq, K, N, tid.x, simd_gid, simd_lid, g);
    C::template rows2<T, RPS>(
        uw + off, xp, xq, K, N, tid.x, simd_gid, simd_lid, u);
  }
  if (simd_lid == 0) {
    const int row0 = tid.x * 2 * RPS + simd_gid * RPS;
    for (int m = 0; m < (q < 0 ? 1 : 2); m++) {
      device T* o = out + (int64_t)(m == 0 ? p : q) * N + row0;
      for (int i = 0; i < RPS; i++) {
        KQ_MOE_DO(
            const float sg_ = kq_moe_row_scale(out_scale, expert, N, row0 + i);
            const float su_ = kq_moe_row_scale(up_scale, expert, N, row0 + i);)
        o[i] = static_cast<T>(kq_glu_epilogue<ACT>(
            KQ_MOE_MUL(g[m * RPS + i], sg_),
            KQ_MOE_MUL(u[m * RPS + i], su_),
            limit,
            KQ_GLU_ALPHA));
      }
    }
  }
}

// Biased GLU (gpt-oss experts): f32 biases per (expert, row) ahead of the
// clamped-SwiGLU epilogue.
template <typename T, typename C, int ACT>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_moe_glu_gather_bias)(
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
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_GLU) {
  constexpr int RPS = C::rps;
  const int R = tpg.y;
  const uint expert = indices[tid.z * R + tid.y];
  const int64_t off = (int64_t)expert * N * C::row_bytes(K);
  x += (int64_t)tid.z * K;
  float g[RPS] = {0};
  float u[RPS] = {0};
  C::template rows<T, RPS>(gw + off, x, K, N, tid.x, simd_gid, simd_lid, g);
  C::template rows<T, RPS>(uw + off, x, K, N, tid.x, simd_gid, simd_lid, u);
  if (simd_lid == 0) {
    const int row0 = tid.x * 2 * RPS + simd_gid * RPS;
    const int64_t brow = (int64_t)expert * N + row0;
    out += ((int64_t)tid.z * R + tid.y) * N + row0;
    for (int i = 0; i < RPS; i++) {
      KQ_MOE_DO(
          const float sg_ = kq_moe_row_scale(out_scale, expert, N, row0 + i);
          const float su_ = kq_moe_row_scale(up_scale, expert, N, row0 + i);)
      out[i] = static_cast<T>(kq_glu_epilogue<ACT>(
          KQ_MOE_MUL(g[i], sg_) + gb[brow + i],
          KQ_MOE_MUL(u[i], su_) + ub[brow + i],
          limit,
          alpha));
    }
  }
}

// The down projection, unmixed: one activation row per (token, slot), out
// [T, R, N]; grid (N / (2 rps), R, T).
template <typename T, typename C>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_gather_qmv)(
    const device uint8_t* w [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_ROUTED) {
  constexpr int RPS = C::rps;
  const int64_t row_idx = (int64_t)tid.z * tpg.y + tid.y;
  const uint expert = indices[row_idx];
  float r[RPS] = {0};
  C::template rows<T, RPS>(
      w + (int64_t)expert * N * C::row_bytes(K),
      x + row_idx * K,
      K,
      N,
      tid.x,
      simd_gid,
      simd_lid,
      r);
  if (simd_lid == 0) {
    const int row0 = tid.x * 2 * RPS + simd_gid * RPS;
    out += row_idx * N + row0;
    for (int i = 0; i < RPS; i++) {
      out[i] = static_cast<T>(
          KQ_MOE_MUL(r[i], kq_moe_row_scale(out_scale, expert, N, row0 + i)));
    }
  }
}

template <typename T, typename C>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_gather_qmv_bias)(
    const device uint8_t* w [[buffer(0)]],
    const device float* b [[buffer(1)]],
    const device T* x [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_ROUTED) {
  constexpr int RPS = C::rps;
  const int64_t row_idx = (int64_t)tid.z * tpg.y + tid.y;
  const uint expert = indices[row_idx];
  float r[RPS] = {0};
  C::template rows<T, RPS>(
      w + (int64_t)expert * N * C::row_bytes(K),
      x + row_idx * K,
      K,
      N,
      tid.x,
      simd_gid,
      simd_lid,
      r);
  if (simd_lid == 0) {
    const int row0 = tid.x * 2 * RPS + simd_gid * RPS;
    const int64_t brow = (int64_t)expert * N + row0;
    out += row_idx * N + row0;
    for (int i = 0; i < RPS; i++) {
      out[i] = static_cast<T>(
          KQ_MOE_MUL(r[i], kq_moe_row_scale(out_scale, expert, N, row0 + i)) +
          b[brow + i]);
    }
  }
}

// Score-mixed down, no shared expert: out[t] = sum_s scores[t, s] *
// (h[t, s] @ D_e(s)); grid (N / (2 rps), 1, T).
template <typename T, typename C>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_gather_qmv_mix_ns)(
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
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_ROUTED) {
  constexpr int RPS = C::rps;
  const int row0 = tid.x * 2 * RPS + simd_gid * RPS;
  const int64_t rb = C::row_bytes(K);
  float acc[RPS] = {0};
  for (int slot = 0; slot < S; slot++) {
    const uint expert = indices[tid.z * S + slot];
    float r[RPS] = {0};
    C::template rows<T, RPS>(
        w + (int64_t)expert * N * rb,
        h + ((int64_t)tid.z * S + slot) * K,
        K,
        N,
        tid.x,
        simd_gid,
        simd_lid,
        r);
    const float sc = scores[tid.z * S + slot];
    for (int i = 0; i < RPS; i++) {
      acc[i] +=
          KQ_MOE_MUL(sc, kq_moe_row_scale(out_scale, expert, N, row0 + i)) *
          r[i];
    }
  }
  if (simd_lid == 0) {
    out += (int64_t)tid.z * N + row0;
    for (int i = 0; i < RPS; i++) {
      out[i] = static_cast<T>(acc[i]);
    }
  }
}

// Slot-parallel kq_qf_gather_qmv_mix_ns: simdgroup pair p computes slot p,
// and the first pair sums the slots in the loop kernel's order. Group
// (32, 2 S, 1), grid (N / (2 rps), 1, T); the host keeps S <=
// KQ_MOE_SP_MAX_S.
template <typename T, typename C>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_gather_qmv_mix_ns_sp)(
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
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_ROUTED) {
  constexpr int RPS = C::rps;
  const int slot = int(simd_gid >> 1);
  const uint half_ = simd_gid & 1;
  threadgroup float parts[KQ_MOE_SP_MAX_S][2 * RPS];
  {
    const uint expert = indices[tid.z * S + slot];
    float r[RPS] = {0};
    C::template rows<T, RPS>(
        w + (int64_t)expert * N * C::row_bytes(K),
        h + ((int64_t)tid.z * S + slot) * K,
        K,
        N,
        tid.x,
        half_,
        simd_lid,
        r);
    if (simd_lid == 0) {
      for (int i = 0; i < RPS; i++) {
        parts[slot][half_ * RPS + i] = r[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (slot == 0 && simd_lid < uint(RPS)) {
    const int lr = half_ * RPS + simd_lid;
    const int row = tid.x * 2 * RPS + lr;
    float acc = 0.0f;
    for (int s = 0; s < S; s++) {
      acc += KQ_MOE_MUL(
                 scores[tid.z * S + s],
                 kq_moe_row_scale(out_scale, indices[tid.z * S + s], N, row)) *
          parts[s][lr];
    }
    out[(int64_t)tid.z * N + row] = static_cast<T>(acc);
  }
}

// GLU with the shared expert as NS extra slots: slot R + j reads rows
// [j N, (j + 1) N) of the 2-D shared gate and up tensors. out [T, R + NS, N];
// grid (N / (2 rps), R + NS, T).
template <typename T, typename C, typename SC, int ACT>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_moe_glu_gather_shexp)(
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
    KQ_GLU_ALPHA_ARG,
    const constant int& NS [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_SHEXP_GLU) {
  constexpr int RPS = C::rps;
  const int n_route = int(tpg.y) - NS;
  const bool shared_slot = int(tid.y) >= n_route;
  // The dense route computed the shared slots (kq_moe_shexp_glu_pre).
  if (kq_shexp_pre && shared_slot) {
    return;
  }
  const int row0 = tid.x * 2 * RPS + simd_gid * RPS;
  x += (int64_t)tid.z * K;
  float g[RPS] = {0};
  float u[RPS] = {0};
  uint expert = 0;
  if (shared_slot) {
    const int64_t off = (int64_t)(int(tid.y) - n_route) * N * SC::row_bytes(K);
    SC::template rows<T, RPS>(sgw + off, x, K, N, tid.x, simd_gid, simd_lid, g);
    SC::template rows<T, RPS>(suw + off, x, K, N, tid.x, simd_gid, simd_lid, u);
  } else {
    expert = indices[tid.z * n_route + tid.y];
    const int64_t off = (int64_t)expert * N * C::row_bytes(K);
    C::template rows<T, RPS>(gw + off, x, K, N, tid.x, simd_gid, simd_lid, g);
    C::template rows<T, RPS>(uw + off, x, K, N, tid.x, simd_gid, simd_lid, u);
  }
  if (simd_lid == 0) {
    out += ((int64_t)tid.z * tpg.y + tid.y) * N + row0;
    for (int i = 0; i < RPS; i++) {
      KQ_MOE_DO(const float sg_ = shared_slot
                    ? kq_moe_shexp_scale(shexp_scale)
                    : kq_moe_row_scale(out_scale, expert, N, row0 + i);
                const float su_ = shared_slot
                    ? kq_moe_shexp_scale(shexp_up_scale)
                    : kq_moe_row_scale(up_scale, expert, N, row0 + i);)
      out[i] = static_cast<T>(kq_glu_epilogue<ACT>(
          KQ_MOE_MUL(g[i], sg_), KQ_MOE_MUL(u[i], su_), limit, KQ_GLU_ALPHA));
    }
  }
}

// Score-mixed down with one shared slot (NS == 1; wider shared experts
// take the Ext kernels): slots [0, S - 1) routed, slot S - 1 the shared
// down rows. SC == S - 1 gives the shared slot an implicit weight of 1.
// grid (N / (2 rps), 1, T).
template <typename T, typename C, typename SCd>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_gather_qmv_mix)(
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
    KQ_SHEXP_PRE_ARG,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_MIX) {
  constexpr int RPS = C::rps;
  const int row0 = tid.x * 2 * RPS + simd_gid * RPS;
  const int n_route = S - 1;
  const int64_t rb = C::row_bytes(K);
  float acc[RPS] = {0};
  for (int slot = 0; slot < n_route; slot++) {
    const uint expert = indices[tid.z * n_route + slot];
    float r[RPS] = {0};
    C::template rows<T, RPS>(
        w + (int64_t)expert * N * rb,
        h + ((int64_t)tid.z * S + slot) * K,
        K,
        N,
        tid.x,
        simd_gid,
        simd_lid,
        r);
    const float sc = scores[tid.z * SC + slot];
    for (int i = 0; i < RPS; i++) {
      acc[i] +=
          KQ_MOE_MUL(sc, kq_moe_row_scale(out_scale, expert, N, row0 + i)) *
          r[i];
    }
  }
  float r[RPS] = {0};
  if (kq_shexp_pre) {
    for (int i = 0; i < RPS; i++) {
      r[i] = float(shexp_pre[(int64_t)tid.z * N + row0 + i]);
    }
  } else {
    SCd::template rows<T, RPS>(
        sw,
        h + ((int64_t)tid.z * S + n_route) * K,
        K,
        N,
        tid.x,
        simd_gid,
        simd_lid,
        r);
  }
  const float ssc = KQ_MOE_MUL(
      (SC == S ? scores[tid.z * SC + n_route] : 1.0f),
      kq_moe_shexp_scale(shexp_scale));
  for (int i = 0; i < RPS; i++) {
    acc[i] += ssc * r[i];
  }
  if (simd_lid == 0) {
    out += (int64_t)tid.z * N + row0;
    for (int i = 0; i < RPS; i++) {
      out[i] = static_cast<T>(acc[i]);
    }
  }
}

// Slot-parallel kq_qf_gather_qmv_mix: pair p < S - 1 computes routed slot
// p, pair S - 1 the shared slot. Group (32, 2 S, 1), or (32, 2 (S - 1), 1)
// with kq_shexp_pre; the host keeps the pairs <= KQ_MOE_SP_MAX_S.
template <typename T, typename C, typename SCd>
[[kernel]] void KQ_MOE_KERNEL(kq_qf_gather_qmv_mix_sp)(
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
    KQ_SHEXP_PRE_ARG,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] KQ_MOE_ARGS_MIX) {
  constexpr int RPS = C::rps;
  const int pair = int(simd_gid >> 1);
  const uint half_ = simd_gid & 1;
  const int n_route = S - 1;
  threadgroup float parts[KQ_MOE_SP_MAX_S][2 * RPS];
  {
    float r[RPS] = {0};
    const device T* xs = h + ((int64_t)tid.z * S + pair) * K;
    if (pair < n_route) {
      const uint expert = indices[tid.z * n_route + pair];
      C::template rows<T, RPS>(
          w + (int64_t)expert * N * C::row_bytes(K),
          xs,
          K,
          N,
          tid.x,
          half_,
          simd_lid,
          r);
    } else {
      SCd::template rows<T, RPS>(sw, xs, K, N, tid.x, half_, simd_lid, r);
    }
    if (simd_lid == 0) {
      for (int i = 0; i < RPS; i++) {
        parts[pair][half_ * RPS + i] = r[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (pair == 0 && simd_lid < uint(RPS)) {
    const int lr = half_ * RPS + simd_lid;
    const int row = tid.x * 2 * RPS + lr;
    float acc = 0.0f;
    for (int s = 0; s < n_route; s++) {
      acc += KQ_MOE_MUL(
                 scores[tid.z * SC + s],
                 kq_moe_row_scale(
                     out_scale, indices[tid.z * n_route + s], N, row)) *
          parts[s][lr];
    }
    acc += KQ_MOE_MUL(
               (SC == S ? scores[tid.z * SC + n_route] : 1.0f),
               kq_moe_shexp_scale(shexp_scale)) *
        (kq_shexp_pre ? float(shexp_pre[(int64_t)tid.z * N + row])
                      : parts[n_route][lr]);
    out[(int64_t)tid.z * N + row] = static_cast<T>(acc);
  }
}

// Kernel names: "kq_<codec>_qf_<op>" and, with a shared expert,
// "kq_<codec>_qf_sx_<shared codec>_<op>".
// clang-format off
#define instantiate_kq_qf_type(codec, core, type)                             \
  instantiate_kernel("kq_" #codec "_qf_moe_glu_gather_silu_" #type,           \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather), type, core, KQ_GLU_ACT_SILU)        \
  instantiate_kernel("kq_" #codec "_qf_moe_glu_gather_gelu_" #type,           \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather), type, core, KQ_GLU_ACT_GELU)        \
  instantiate_kernel("kq_" #codec "_qf_moe_glu_gather_silu_limit_" #type,     \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather), type, core, KQ_GLU_ACT_SILU_LIMIT)  \
  instantiate_kernel("kq_" #codec "_qf_gather_qmv_" #type,                    \
      KQ_MOE_KERNEL(kq_qf_gather_qmv), type, core)                             \
  instantiate_kernel("kq_" #codec "_qf_gather_qmv_mix_ns_" #type,             \
      KQ_MOE_KERNEL(kq_qf_gather_qmv_mix_ns), type, core)                      \
  instantiate_kernel("kq_" #codec "_qf_gather_qmv_mix_ns_sp_" #type,          \
      KQ_MOE_KERNEL(kq_qf_gather_qmv_mix_ns_sp), type, core)

#define instantiate_kq_qf_sx_type(codec, core, scodec, score, type)           \
  instantiate_kernel(                                                         \
      "kq_" #codec "_qf_sx_" #scodec "_moe_glu_gather_shexp_silu_" #type,     \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather_shexp), type, core, score,           \
      KQ_GLU_ACT_SILU)                                                         \
  instantiate_kernel(                                                         \
      "kq_" #codec "_qf_sx_" #scodec "_moe_glu_gather_shexp_gelu_" #type,     \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather_shexp), type, core, score,           \
      KQ_GLU_ACT_GELU)                                                         \
  instantiate_kernel("kq_" #codec "_qf_sx_" #scodec "_gather_qmv_mix_" #type, \
      KQ_MOE_KERNEL(kq_qf_gather_qmv_mix), type, core, score)                  \
  instantiate_kernel(                                                         \
      "kq_" #codec "_qf_sx_" #scodec "_gather_qmv_mix_sp_" #type,             \
      KQ_MOE_KERNEL(kq_qf_gather_qmv_mix_sp), type, core, score)

#define instantiate_kq_qf_bias_type(codec, core, type)                        \
  instantiate_kernel(                                                         \
      "kq_" #codec "_qf_moe_glu_gather_bias_swiglu_clamp_" #type,             \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather_bias), type, core,                   \
      KQ_GLU_ACT_SWIGLU_CLAMP)                                                 \
  instantiate_kernel("kq_" #codec "_qf_gather_qmv_bias_" #type,               \
      KQ_MOE_KERNEL(kq_qf_gather_qmv_bias), type, core)

#define instantiate_kq_qf_type_all(codec, core, type)                         \
  instantiate_kq_qf_type(codec, core, type)                                   \
  instantiate_kq_qf_sx_type(codec, core, codec, core, type)                   \
  instantiate_kq_qf_sx_type(codec, core, q5_k, KqQfQ5_K, type)                \
  instantiate_kq_qf_sx_type(codec, core, q6_k, KqQfQ6_K, type)                \
  instantiate_kq_qf_sx_type(codec, core, q8_0, KqQfQ8_0, type)

#define instantiate_kq_qf_dd_type(codec, core, type)                          \
  instantiate_kernel("kq_" #codec "_qf_moe_glu_gather_dd_silu_" #type,        \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather_dd), type, core, KQ_GLU_ACT_SILU)     \
  instantiate_kernel("kq_" #codec "_qf_moe_glu_gather_dd_gelu_" #type,        \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather_dd), type, core, KQ_GLU_ACT_GELU)     \
  instantiate_kernel("kq_" #codec "_qf_moe_glu_gather_dd_silu_limit_" #type,  \
      KQ_MOE_KERNEL(kq_qf_moe_glu_gather_dd), type, core, KQ_GLU_ACT_SILU_LIMIT)

#define instantiate_kq_qf_dd(codec, core)                                     \
  instantiate_kq_qf_dd_type(codec, core, bfloat16_t)                          \
  instantiate_kq_qf_dd_type(codec, core, float16_t)

#define instantiate_kq_qf(codec, core)                                        \
  instantiate_kq_qf_type_all(codec, core, bfloat16_t)                         \
  instantiate_kq_qf_type_all(codec, core, float16_t)

#define instantiate_kq_qf_bias(codec, core)                                   \
  instantiate_kq_qf_bias_type(codec, core, bfloat16_t)                        \
  instantiate_kq_qf_bias_type(codec, core, float16_t)
// clang-format on
