#pragma once

// Low-rank hyper-connection kernels for weights of any codec
// (kq_hc_lowrank.h has the q8_0 forms and the reference). A weight is read
// through its Ext codec trait (kq_quantized*.h), and a float weight through
// KqHcFloat, a trait of 16-value blocks. A row's sums do not depend on R:
// the front and inject run the four-row loop at every R, and the epilogue
// takes the order of kq_hc_lowrank_epilogue_mr with 16-value chunks.

// Float weights as a codec: blocks of 16 values, no scale.
template <typename W>
struct KqHcFloat {
  MLX_MTL_CONST int superblock = 16;
  MLX_MTL_CONST int block_bytes = 16 * sizeof(W);
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short, thread float4x4& reg) {
    const device vec<W, 4>* p = (const device vec<W, 4>*)block;
    reg[0] = float4(p[0]);
    reg[1] = float4(p[1]);
    reg[2] = float4(p[2]);
    reg[3] = float4(p[3]);
  }
};

// Down rows: lo[r, j] = H(t * H(sigmoid(t))) with t = H(dot) / 4. Grid
// (LR / (SG * 32 / NX), ceil(R / 4)), threadgroups (32, SG): a simdgroup
// takes 32 / NX down rows, NX lanes a row, four activation rows.
template <typename T, typename H, typename Codec, int NX>
[[kernel]] void kq_hc_front_ext(
    const device T* xn [[buffer(0)]],
    const device uint8_t* w [[buffer(1)]],
    device H* lo [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& LR [[buffer(4)]],
    const constant int& R [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 32 / NX;
  const short tx = short(simd_lid % NX);
  const int j =
      (int(tid.x) * int(tptg.y) + int(simd_gid)) * RPS + int(simd_lid / NX);
  const int r0 = int(tid.y) * 4;
  const int cnt = min(4, R - r0);
  KQ_EXT_DECL_LUTS2(Codec, Codec, luts)
  KQ_EXT_STAGE_INTO(Codec, luts)
  const int4 xo = min(int4(r0) + int4(0, 1, 2, 3), int4(R - 1)) * K;
  const float4 acc = kq_ext_row_partial_quad<T, Codec, NX>(
      w, xn + xo.x, xn + xo.y, xn + xo.z, xn + xo.w, j, K, tx, luts);
  for (int i = 0; i < cnt; ++i) {
    const float v = kq_ext_reduce<NX>(acc[i]);
    if (tx == 0) {
      const float t = float(H(v)) * 0.25f;
      const float sig = float(H(kq_hclr_sigmoid(t)));
      lo[int64_t(r0 + i) * LR + j] = H(t * sig);
    }
  }
}

// Inject rows: inj[r, s] = I(2 * I(sigmoid(I(dot) / 4))). Grid (1,
// ceil(R / 4)), threadgroups (32, 4): simdgroup s takes inject row s, all
// 32 lanes on it, four activation rows.
template <typename T, typename I, typename Codec>
[[kernel]] void kq_hc_inject_ext(
    const device T* xn [[buffer(0)]],
    const device uint8_t* w [[buffer(1)]],
    device I* inj [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& R [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const short tx = short(simd_lid);
  const int s = int(simd_gid);
  const int r0 = int(tid.y) * 4;
  const int cnt = min(4, R - r0);
  KQ_EXT_DECL_LUTS2(Codec, Codec, luts)
  KQ_EXT_STAGE_INTO(Codec, luts)
  const int4 xo = min(int4(r0) + int4(0, 1, 2, 3), int4(R - 1)) * K;
  const float4 acc = kq_ext_row_partial_quad<T, Codec, 32>(
      w, xn + xo.x, xn + xo.y, xn + xo.z, xn + xo.w, s, K, tx, luts);
  for (int i = 0; i < cnt; ++i) {
    const float v = kq_ext_reduce<32>(acc[i]);
    if (tx == 0) {
      const float t = float(I(v)) * 0.25f;
      inj[int64_t(r0 + i) * KQ_HCLR + s] =
          I(2.0f * float(I(kq_hclr_sigmoid(t))));
    }
  }
}

// Up rows of a codec whose blocks divide LR: lane (d, q) takes chunks q,
// q + L, ... of the four stream rows of d for up to four rows of R, and the
// gate and mean are kq_hc_lowrank_epilogue_mr's. Grid (D / (32 / L) / 4,
// ceil(R / 4)), threadgroups (32, 4), lo staged as LR * min(R, 4) values.
template <typename T, typename H, typename Codec, int L>
[[kernel]] void kq_hc_epilogue_ext(
    const device H* lo [[buffer(0)]],
    const device uint8_t* w_up [[buffer(1)]],
    const device T* xn [[buffer(2)]],
    device T* mixed [[buffer(3)]],
    const constant int& D [[buffer(4)]],
    const constant int& LR [[buffer(5)]],
    const constant int& R [[buffer(6)]],
    threadgroup vec<H, 4>* los [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int RB = 4;
  constexpr int DPS = 32 / L;
  constexpr short chpb = Codec::superblock / 16;
  const int q = int(simd_lid) % L;
  const int dd = int(simd_lid) / L;
  const int d = (int(tg.x) * 4 + int(simd_gid)) * DPS + dd;
  const int K = KQ_HCLR * D;
  const int nch = LR / 16;
  const int64_t row_bytes =
      int64_t(LR / Codec::superblock) * Codec::block_bytes;
  const int64_t sstride = int64_t(D) * row_bytes;
  const device uint8_t* wrow = w_up + int64_t(d) * row_bytes;
  const int r0 = int(tg.y) * RB;
  const int cnt = min(RB, R - r0);
  const int lv = LR / 4;

  for (int i = int(tid); i < cnt * lv; i += 128) {
    los[i] = *(const device vec<H, 4>*)(lo + int64_t(r0) * LR + i * 4);
  }
  KQ_EXT_DECL_LUTS2(Codec, Codec, luts)
  if (KqTgLuts<Codec>::bytes > 0) {
    KqTgLuts<Codec>::stage(
        luts, ushort(simd_gid * 32 + simd_lid), ushort(32 * tptg.y));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[RB][KQ_HCLR];
#pragma unroll
  for (int v = 0; v < RB; v++) {
    acc[v][0] = acc[v][1] = acc[v][2] = acc[v][3] = 0.0f;
  }
  for (int c = q; c < nch; c += L) {
#pragma unroll
    for (int s = 0; s < KQ_HCLR; s++) {
      float4x4 lw;
      float sc;
      KqTgLuts<Codec>::deq_chunk16s(
          wrow + s * sstride + int64_t(c / chpb) * Codec::block_bytes,
          short(c % chpb),
          lw,
          luts,
          sc);
#pragma unroll
      for (int v = 0; v < RB; v++) {
        if (v < cnt) {
          const threadgroup vec<H, 4>* lp = los + v * lv + c * 4;
          acc[v][s] += sc *
              (dot(lw[0], float4(lp[0])) + dot(lw[1], float4(lp[1])) +
               dot(lw[2], float4(lp[2])) + dot(lw[3], float4(lp[3])));
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
          m = metal::fma(g, float(xn[int64_t(r) * K + s * D + d]), m);
        }
        mixed[int64_t(r) * D + d] = T(m * 0.25f);
      }
    }
  }
}
