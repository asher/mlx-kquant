// Native micro-scaling float formats as defined by ggml (llama.cpp, MIT) -
// see mlx_kquant/licenses/llama.cpp-LICENSE.
//
// MXFP4: 17 bytes/32 weights. [uint8 E8M0 d][uint8 qs[16]], two-halves
//   nibbles: w[j] = d * E2M1[qs[j] & 0xF], w[j+16] = d * E2M1[qs[j] >> 4].
// NVFP4: 36 bytes/64 weights. [uint8 UE4M3 sd[4]][uint8 qs[32]], four
//   16-value groups; group g = sd[g] scaling qs[8g..8g+8) two-halves.
// MXFP8: 33 bytes/32 weights. [uint8 E8M0 e][uint8 E4M3 qs[32]],
//   w[j] = e4m3(qs[j]) * 2^(e - 127). Gather-only: dequantize kernels only.
//
// Real-value E2M1 codes + full scales (the Metal convention; the CPU
// kernels use ggml's doubled-int8 LUT + half scales - same products, never
// mixed within one kernel).

MLX_MTL_CONST int KQ_MXFP4_GROUP = 32;
MLX_MTL_CONST int KQ_MXFP4_BLOCK_BYTES = 17;
MLX_MTL_CONST int KQ_MXFP4_QS_OFFSET = 1;

MLX_MTL_CONST int KQ_NVFP4_SUPERBLOCK = 64;
MLX_MTL_CONST int KQ_NVFP4_BLOCK_BYTES = 36;
MLX_MTL_CONST int KQ_NVFP4_QS_OFFSET = 4;

MLX_MTL_CONST int KQ_MXFP8_GROUP = 32;
MLX_MTL_CONST int KQ_MXFP8_BLOCK_BYTES = 33;
MLX_MTL_CONST int KQ_MXFP8_QS_OFFSET = 1;

// E2M1 code c = s e1 e0 m to its value without a table, as verify_nax
// builds it: e1 e0 m in half bits 11..9 is the value times 2^-14 (a half
// subnormal for e = 0), the sign goes to bit 15, and the product with 2^14
// is exact in half. Code 8 is -0.0 like the table it replaces.
inline float kq_fp_e2m1_value(uint c) {
  const half h = as_type<half>(ushort(((c & 7u) << 9) | ((c & 8u) << 12)));
  return float(h * 16384.0h);
}

// Two E2M1 codes, at bits 0..3 and 16..19 of n, to their values through one
// half2. The masks select the two codes, so callers pass a word holding a
// byte at bits 0..7 and one at 16..23, shifted right by 4 for the high
// nibbles.
inline float2 kq_fp_e2m1_pair(uint n) {
  const half2 h =
      as_type<half2>(((n & 0x00070007u) << 9) | ((n & 0x00080008u) << 12));
  return float2(h * half2(16384.0h));
}

// 2^(e-127) built from bits: metal::exp2 is fast-math and lands ulps off on
// some compiler versions. e == 0 is the f32 subnormal 2^-127.
inline float kq_fp_e8m0_scale(uint8_t e) {
  return as_type<float>(e == 0 ? 0x00400000u : uint(e) << 23);
}

// Unsigned E4M3 (bias 7); 0x00 and 0x7F (the NaN encoding) decode to 0.
// Normals assembled from bits (biased f32 exponent e + 120, mantissa m << 20)
// because metal::exp2 is fast-math; the subnormal product is exact as
// written. Two selects, no branch: e == 0 picks the subnormal (0 for v == 0),
// then the NaN code is zeroed. Bit 7 is ignored, so 0xFF decodes like 0x7F
// would without the NaN rule, as ggml does. S returns the scale times 2^S
// at the same cost.
template <int S = 0>
inline float kq_fp_ue4m3_scale(uint8_t v) {
  const uint e = (v >> 3) & 0xF;
  const uint m = v & 0x7;
  const float norm = as_type<float>((e + 120u + S) << 23 | m << 20);
  const float sub = float(m) * (0.001953125f * float(1 << S)); /* 2^(S-9) */
  const float r = e ? norm : sub;
  return (v == 0x7F) ? 0.0f : r;
}

// ---------------------------------------------------------------------------
// MXFP4 (structure cloned from IQ4_NL: same two-halves 16-byte qs, the fp16
// scale replaced by the E8M0 byte and the int LUT by the E2M1 decode)
// ---------------------------------------------------------------------------

template <typename T>
METAL_FUNC void kq_mxfp4_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_MXFP4_GROUP;
  const int within = gid % KQ_MXFP4_GROUP;
  const device uint8_t* block_addr =
      w + static_cast<int64_t>(block_id) * KQ_MXFP4_BLOCK_BYTES;
  const float d = kq_fp_e8m0_scale(block_addr[0]);
  const device uint8_t* qs = block_addr + KQ_MXFP4_QS_OFFSET;
  const int nib =
      (within < 16) ? (int(qs[within]) & 0x0F) : (int(qs[within - 16]) >> 4);
  out[gid] = T(d * kq_fp_e2m1_value(uint(nib)));
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_mxfp4_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(group_size == KQ_MXFP4_GROUP, "MXFP4 requires gs=32");
  static_assert(bits == 4, "MXFP4 requires bits=4");
  kq_mxfp4_dequantize_impl<T>(w, out, num_weights, gid);
}

// Natural-order chunk [il*16, il*16+16) of one block; il in [0,1].
METAL_FUNC void kq_mxfp4_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const float d = kq_fp_e8m0_scale(block[0]);
  const device uint8_t* qs = block + KQ_MXFP4_QS_OFFSET;
  const int shift = (il & 1) ? 4 : 0;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    const float2 v =
        kq_fp_e2m1_pair((uint(qs[i]) | (uint(qs[i + 1]) << 16)) >> shift);
    reg[i / 4][i % 4] = d * v.x;
    reg[i / 4][i % 4 + 1] = d * v.y;
  }
}

struct KqMxfp4Ext {
  MLX_MTL_CONST int superblock = KQ_MXFP4_GROUP;
  MLX_MTL_CONST int block_bytes = KQ_MXFP4_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_mxfp4_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_mxfp4_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqMxfp4Ext, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// MXFP4 mat-vec: one impl for both qmv and qmv_fast. A simdgroup pass
// covers 16 blocks, two lanes per block; a lane owns 8 qs bytes, so its 16
// weights are the low and the high nibbles of those bytes, and its x values
// come in as four float4 reused across the row loop.
template <typename T, int group_size, int bits, int results_per_simdgroup = 4>
METAL_FUNC void kq_mxfp4_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_MXFP4_GROUP, "MXFP4 requires gs=32");
  static_assert(bits == 4, "MXFP4 requires bits=4");
  constexpr int num_simdgroups = 2;
  constexpr int blocks_per_pass = 16;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes = in_vec_size * KQ_MXFP4_BLOCK_BYTES / KQ_MXFP4_GROUP;
  const int nb = in_vec_size / KQ_MXFP4_GROUP;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int ix = simd_lid / 2;
  const int it = simd_lid % 2;
  U result[results_per_simdgroup] = {0};
  for (int ib = ix; ib < nb; ib += blocks_per_pass) {
    const device T* xb = x + ib * KQ_MXFP4_GROUP + it * 8;
    const float4 yl0(xb[0], xb[1], xb[2], xb[3]);
    const float4 yl1(xb[16], xb[17], xb[18], xb[19]);
    const float4 yl2(xb[4], xb[5], xb[6], xb[7]);
    const float4 yl3(xb[20], xb[21], xb[22], xb[23]);
    // static trip count; the clamped tail row is dropped at the store
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* blk = w +
          static_cast<int64_t>(min(out_row + row, out_vec_size - 1)) *
              row_bytes +
          ib * KQ_MXFP4_BLOCK_BYTES;
      const U d = U(kq_fp_e8m0_scale(blk[0]));
      const device uint8_t* q = blk + KQ_MXFP4_QS_OFFSET + 8 * it;
      const uint w02 = uint(q[0]) | (uint(q[2]) << 16);
      const uint w13 = uint(q[1]) | (uint(q[3]) << 16);
      const uint w46 = uint(q[4]) | (uint(q[6]) << 16);
      const uint w57 = uint(q[5]) | (uint(q[7]) << 16);
      const float2 l02 = kq_fp_e2m1_pair(w02), l13 = kq_fp_e2m1_pair(w13);
      const float2 h02 = kq_fp_e2m1_pair(w02 >> 4);
      const float2 h13 = kq_fp_e2m1_pair(w13 >> 4);
      const float2 l46 = kq_fp_e2m1_pair(w46), l57 = kq_fp_e2m1_pair(w57);
      const float2 h46 = kq_fp_e2m1_pair(w46 >> 4);
      const float2 h57 = kq_fp_e2m1_pair(w57 >> 4);
      float4 acc = yl0 * float4(l02.x, l13.x, l02.y, l13.y);
      acc += yl1 * float4(h02.x, h13.x, h02.y, h13.y);
      acc += yl2 * float4(l46.x, l57.x, l46.y, l57.y);
      acc += yl3 * float4(h46.x, h57.x, h46.y, h57.y);
      result[row] += d * ((acc.x + acc.y) + (acc.z + acc.w));
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 4>
METAL_FUNC void kq_mxfp4_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_mxfp4_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqMxfp4BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_MXFP4_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_MXFP4_BLOCK_BYTES;
  static_assert(BCOLS == weights_per_block, "MXFP4 loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  MLX_MTL_CONST short half_block = weights_per_block / 2;
  static_assert(n_reads >= 2 && n_reads % 2 == 0, "MXFP4 needs even n_reads.");
  static_assert(n_reads % 4 == 0, "MXFP4 decodes byte pairs.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj_byte;
  threadgroup T* dst;
  const device uint8_t* src;

  KqMxfp4BlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int /* col_in_block */ = 0) thread
      : src_ld(src_ld_),
        row_bytes(src_ld_* bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? bytes_per_block
                : BROWS*(src_ld_* bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id* SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const thread {
    const float d = kq_fp_e8m0_scale(src[0]);
    const device uint8_t* qs = src + KQ_MXFP4_QS_OFFSET + bj_byte;
#pragma unroll
    for (short i = 0; i + 1 < bytes_per_thread; i += 2) {
      const uint w = uint(qs[i]) | (uint(qs[i + 1]) << 16);
      const float2 lo = kq_fp_e2m1_pair(w);
      const float2 hi = kq_fp_e2m1_pair(w >> 4);
      dst[i] = T(d * lo.x);
      dst[i + 1] = T(d * lo.y);
      dst[half_block + i] = T(d * hi.x);
      dst[half_block + i + 1] = T(d * hi.y);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        dst[i] = T(0);
        dst[half_block + i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() thread {
    src += tile_stride;
  }
};

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    bool batched,
    int BM = 64,
    int BN = 64,
    int WM = 2,
    int WN = 2>
[[kernel]] void kq_mxfp4_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_MXFP4_GROUP, "MXFP4 requires gs=32");
  static_assert(bits == 4, "MXFP4 requires bits=4");
  constexpr int BK = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqMxfp4BlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    int small_bm = 0,
    typename U = T>
[[kernel]] void kq_mxfp4_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device U* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_MXFP4_GROUP, "MXFP4 requires gs=32");
  static_assert(bits == 4, "MXFP4 requires bits=4");
  constexpr int BM = small_bm ? small_bm : 32;
  constexpr int BK = 32, BN = small_bm ? 64 : 32;
  constexpr int WM = BM == 8 ? 1 : 2, WN = 4 / WM;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqMxfp4BlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN, U>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_mxfp4_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_MXFP4_GROUP, "MXFP4 requires gs=32");
  static_assert(bits == 4, "MXFP4 requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqMxfp4BlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_mxfp4_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_mxfp4_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 8 in the default), quadrupling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_mxfp4_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& /* x_batch_ndims */,
    const constant int* /* x_shape */,
    const constant int64_t* /* x_strides */,
    const constant int& /* w_batch_ndims */,
    const constant int* /* w_shape */,
    const constant int64_t* /* w_strides */,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_mxfp4_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_mxfp4_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_mxfp4_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_mxfp4_qmv_fast_fine): 2 output rows per threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_mxfp4_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& /* x_batch_ndims */,
    const constant int* /* x_shape */,
    const constant int64_t* /* x_strides */,
    const constant int& /* w_batch_ndims */,
    const constant int* /* w_shape */,
    const constant int64_t* /* w_strides */,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_mxfp4_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ---------------------------------------------------------------------------
// NVFP4: 36-byte superblock of 64 weights = four 16-value groups, each with
// its own UE4M3 sub-scale and two-halves nibbles within the group's 8 bytes.
// weight t: g = t/16, j = t%16, byte = qs[8g + j%8], nibble j<8 ? lo : hi.
// ---------------------------------------------------------------------------

template <typename T>
METAL_FUNC void kq_nvfp4_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_NVFP4_SUPERBLOCK;
  const int within = gid % KQ_NVFP4_SUPERBLOCK;
  const device uint8_t* block_addr =
      w + static_cast<int64_t>(block_id) * KQ_NVFP4_BLOCK_BYTES;
  const int g = within / 16;
  const int j = within % 16;
  const float d = kq_fp_ue4m3_scale(block_addr[g]);
  const uint8_t b = block_addr[KQ_NVFP4_QS_OFFSET + 8 * g + (j % 8)];
  const int nib = (j < 8) ? (b & 0x0F) : (b >> 4);
  out[gid] = T(d * kq_fp_e2m1_value(uint(nib)));
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_nvfp4_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(group_size == KQ_NVFP4_SUPERBLOCK, "NVFP4 requires gs=64");
  static_assert(bits == 4, "NVFP4 requires bits=4");
  kq_nvfp4_dequantize_impl<T>(w, out, num_weights, gid);
}

// A group's 8 qs bytes as two words: the 36-byte superblock keeps them
// 4-byte aligned.
METAL_FUNC uint2 kq_nvfp4_group_qs(const device uint8_t* block, int g) {
  const device uint32_t* q =
      (const device uint32_t*)(block + KQ_NVFP4_QS_OFFSET + 8 * g);
  return uint2(q[0], q[1]);
}

// E2M1 values of nibble `shift` (0 low, 4 high) of each byte of `q`: bytes
// 0 and 2 through one half2, 1 and 3 through the other.
METAL_FUNC float4 kq_nvfp4_nibbles4(uint32_t q, uint shift) {
  const float2 a = kq_fp_e2m1_pair(q >> shift);
  const float2 b = kq_fp_e2m1_pair(q >> (8 + shift));
  return float4(a.x, b.x, a.y, b.y);
}

// Natural-order chunk [il*16, il*16+16) == sub-scale group il; il in [0,4).
METAL_FUNC void kq_nvfp4_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const float d = kq_fp_ue4m3_scale(block[il]);
  const uint2 q = kq_nvfp4_group_qs(block, il);
  reg[0] = d * kq_nvfp4_nibbles4(q.x, 0);
  reg[1] = d * kq_nvfp4_nibbles4(q.y, 0);
  reg[2] = d * kq_nvfp4_nibbles4(q.x, 4);
  reg[3] = d * kq_nvfp4_nibbles4(q.y, 4);
}

struct KqNvfp4Ext {
  MLX_MTL_CONST int superblock = KQ_NVFP4_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_NVFP4_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_nvfp4_deq_chunk16(block, il, reg);
  }
};

// NVFP4 codes to half, four at a time. kq_nvfp4_nibs keeps the low
// nibble of each byte of w, or the high one, with its sign moved to bit 6
// and e1 e0 m left in bits 0..2. kq_nvfp4_pair moves bytes 0 and 2, or 1
// and 3, to half bits 15 and 11..9, which is the E2M1 value times 2^-14.
// Times a UE4M3 scale times 2^7 (61440 at most) the value stays exact in
// half, at 2^-7 of the weight.
METAL_FUNC uint kq_nvfp4_nibs(uint w, bool hi) {
  return hi ? (((w >> 4) & 0x07070707u) | ((w >> 1) & 0x40404040u))
            : ((w & 0x07070707u) | ((w << 3) & 0x40404040u));
}
METAL_FUNC half2 kq_nvfp4_pair(uint l, bool odd) {
  return as_type<half2>(
      odd ? ((l & 0xFF00FF00u) << 1) : ((l & 0x00FF00FFu) << 9));
}

// Register-resident MMA verify (kq_verify_mma.h). Lane pair L owns
// 16-weight group L: its scale byte and qs bytes 8L..8L+7 as two words.
// Fragment f takes word (f >> 1) & 1, its low nibbles for f < 4 and its
// high ones after, and pair f & 1 of kq_nvfp4_pair: weights 4 (f >> 1) +
// (f & 1) and that plus 2 of the group, times the group scale times 2^7.
// Each weight is exact in half at 2^-7 of its value, and out_mul restores
// the factor at the store.
struct KqNvfp4Mma {
  static constant constexpr int block_k = KQ_NVFP4_SUPERBLOCK;
  static constant constexpr int block_bytes = KQ_NVFP4_BLOCK_BYTES;
  static constant constexpr bool codes_scaled = true;
  static constant constexpr float out_mul = 128.0f;
  static METAL_FUNC int perm(int f, int col) {
    return 16 * (col / 2) + 4 * (f >> 1) + (f & 1) + 2 * (col & 1);
  }
  template <int NT, typename X, typename A>
  static METAL_FUNC void block(
      thread const KqVmmaRows<NT>& rows,
      int boff,
      short L,
      short fm,
      const threadgroup X* xb,
      thread simdgroup_matrix<A, 8, 8> (&acc)[NT]) {
    uint2 q[NT];
    half d[NT];
    for (short t = 0; t < NT; ++t) {
      const device uint8_t* bp = rows.p[t] + boff;
      d[t] = half(kq_fp_ue4m3_scale<7>(bp[L]));
      q[t] = kq_nvfp4_group_qs(bp, L);
    }
    for (short f = 0; f < 8; ++f) {
      half2 a[NT];
      for (short t = 0; t < NT; ++t) {
        const uint wv = (f & 2) ? q[t].y : q[t].x;
        a[t] = kq_nvfp4_pair(kq_nvfp4_nibs(wv, f >= 4), f & 1) * d[t];
      }
      kq_vmma_step<NT>(xb + 8 * perm(f, fm), a, acc);
    }
  }
};

KQ_DEFINE_VERIFY_MMA_KERNEL_OS(nvfp4, KqNvfp4Mma, 2, 1)

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_nvfp4_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    KQ_OUT_SCALE_ARG) {
  kq_mv_ext_impl<T, KqNvfp4Ext, r1ptg, nsg, nxpsg, true>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg, out_scale);
}

// NVFP4 mat-vec on the MXFP4 lane layout: lane `simd_lid` owns sub-scale
// group simd_lid % 4 of superblock simd_lid / 4, 8 superblocks per pass; x is
// read once into four float4 and reused across the unrolled rows.
// kOutScale entries take the output scale (kq_out_scale.h).
template <
    typename T,
    int group_size,
    int bits,
    int results_per_simdgroup = 4,
    bool kOutScale = false>
METAL_FUNC void kq_nvfp4_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    const device float* out_scale = nullptr) {
  static_assert(group_size == KQ_NVFP4_SUPERBLOCK, "NVFP4 requires gs=64");
  static_assert(bits == 4, "NVFP4 requires bits=4");
  constexpr int num_simdgroups = 2;
  constexpr int blocks_per_pass = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_NVFP4_BLOCK_BYTES / KQ_NVFP4_SUPERBLOCK;
  const int nb = in_vec_size / KQ_NVFP4_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int ix = simd_lid / 4;
  const int g = simd_lid % 4;
  U result[results_per_simdgroup] = {0};
  for (int ib = ix; ib < nb; ib += blocks_per_pass) {
    const device T* xb = x + ib * KQ_NVFP4_SUPERBLOCK + 16 * g;
    const float4 yl0(xb[0], xb[1], xb[2], xb[3]);
    const float4 yl1(xb[4], xb[5], xb[6], xb[7]);
    const float4 yl2(xb[8], xb[9], xb[10], xb[11]);
    const float4 yl3(xb[12], xb[13], xb[14], xb[15]);
    // static trip count; the clamped tail row is dropped at the store
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* blk = w +
          static_cast<int64_t>(min(out_row + row, out_vec_size - 1)) *
              row_bytes +
          ib * KQ_NVFP4_BLOCK_BYTES;
      const U d = U(kq_fp_ue4m3_scale(blk[g]));
      const uint2 q = kq_nvfp4_group_qs(blk, g);
      float4 acc = yl0 * kq_nvfp4_nibbles4(q.x, 0);
      acc += yl1 * kq_nvfp4_nibbles4(q.y, 0);
      acc += yl2 * kq_nvfp4_nibbles4(q.x, 4);
      acc += yl3 * kq_nvfp4_nibbles4(q.y, 4);
      result[row] += d * ((acc.x + acc.y) + (acc.z + acc.w));
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      if constexpr (kOutScale) {
        if (kq_has_out_scale) {
          r *= kq_out_scale_col(out_scale, out_row + row);
        }
      }
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <
    typename T,
    int group_size,
    int bits,
    int results_per_simdgroup = 4,
    bool kOutScale = false>
METAL_FUNC void kq_nvfp4_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    const device float* out_scale = nullptr) {
  kq_nvfp4_qmv_impl<T, group_size, bits, results_per_simdgroup, kOutScale>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid, out_scale);
}

// Sub-block loader (q4_k pattern): BCOLS==32 K-tiles walk the two 32-weight
// halves of each 64-weight superblock before advancing the byte pointer.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNvfp4BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_NVFP4_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_NVFP4_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;

  static_assert(
      BCOLS == sub_block_size,
      "NVFP4 loader requires BCOLS == 32 (one half-superblock per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_sub_block_idx;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device uint8_t* src;
  short sub_block_idx;

  KqNvfp4BlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0) thread
      : src_ld(src_ld_),
        row_bytes(src_ld_* bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS*(src_ld_* bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id* SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const thread {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    if constexpr (reduction_dim == 1 && (n_reads == 16 || n_reads == 8)) {
      // The thread's run is one group (16) or one nibble half of it (8):
      // one scale and the group's two qs words.
      const short t0 = sb * sub_block_size + bj;
      const short g = t0 / 16;
      const float d = kq_fp_ue4m3_scale(src[g]);
      const uint2 q = kq_nvfp4_group_qs(src, g);
      const uint shift = (n_reads == 8 && (t0 % 16) != 0) ? 4 : 0;
      float4 v[n_reads / 4];
      v[0] = d * kq_nvfp4_nibbles4(q.x, shift);
      v[1] = d * kq_nvfp4_nibbles4(q.y, shift);
      if constexpr (n_reads == 16) {
        v[2] = d * kq_nvfp4_nibbles4(q.x, 4);
        v[3] = d * kq_nvfp4_nibbles4(q.y, 4);
      }
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(v[i / 4][i % 4]);
      }
    } else {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const short t = sb * sub_block_size + bj + i; // weight in superblock
        const short g = t / 16;
        const short j = t % 16;
        const float d = kq_fp_ue4m3_scale(src[g]);
        const uint8_t b = src[KQ_NVFP4_QS_OFFSET + 8 * g + (j % 8)];
        const int nib = (j < 8) ? (b & 0x0F) : (b >> 4);
        dst[i] = T(d * kq_fp_e2m1_value(uint(nib)));
      }
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() thread {
    if (reduction_dim == 1) {
      sub_block_idx++;
      if (sub_block_idx == sub_blocks_per_block) {
        sub_block_idx = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    bool batched,
    int BM = 64,
    int BN = 64,
    int WM = 2,
    int WN = 2>
[[kernel]] void kq_nvfp4_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    KQ_OUT_SCALE_ARG) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_NVFP4_SUPERBLOCK, "NVFP4 requires gs=64");
  static_assert(bits == 4, "NVFP4 requires bits=4");
  constexpr int BK = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqNvfp4BlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN, T, true>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid, out_scale);
}

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    int small_bm = 0,
    typename U = T>
[[kernel]] void kq_nvfp4_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device U* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_NVFP4_SUPERBLOCK, "NVFP4 requires gs=64");
  static_assert(bits == 4, "NVFP4 requires bits=4");
  constexpr int BM = small_bm ? small_bm : 32;
  constexpr int BK = 32, BN = small_bm ? 64 : 32;
  constexpr int WM = BM == 8 ? 1 : 2, WN = 4 / WM;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqNvfp4BlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN, U>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_nvfp4_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_NVFP4_SUPERBLOCK, "NVFP4 requires gs=64");
  static_assert(bits == 4, "NVFP4 requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqNvfp4BlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_nvfp4_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    KQ_OUT_SCALE_ARG) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_nvfp4_qmv_fast_impl<T, group_size, bits, 4, true>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid, out_scale);
}

// Finer-tiled qmv_fast (see kq_mxfp4_qmv_fast_fine): 2 output rows per
// threadgroup, 4x the threadgroups for the same N. Bit-exact vs the default.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_nvfp4_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& /* x_batch_ndims */,
    const constant int* /* x_shape */,
    const constant int64_t* /* x_strides */,
    const constant int& /* w_batch_ndims */,
    const constant int* /* w_shape */,
    const constant int64_t* /* w_strides */,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    KQ_OUT_SCALE_ARG) {
  kq_nvfp4_qmv_fast_impl<T, group_size, bits, 1, true>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid, out_scale);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_nvfp4_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    KQ_OUT_SCALE_ARG) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_nvfp4_qmv_impl<T, group_size, bits, 4, true>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid, out_scale);
}

// Finer-tiled qmv (see kq_nvfp4_qmv_fast_fine): 2 output rows per threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_nvfp4_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& /* x_batch_ndims */,
    const constant int* /* x_shape */,
    const constant int64_t* /* x_strides */,
    const constant int& /* w_batch_ndims */,
    const constant int* /* w_shape */,
    const constant int64_t* /* w_strides */,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    KQ_OUT_SCALE_ARG) {
  kq_nvfp4_qmv_impl<T, group_size, bits, 1, true>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid, out_scale);
}

// ---------------------------------------------------------------------------
// MXFP8 (OCP MX, E4M3 "fn" codes): a lookup-table codec, read by row
// ---------------------------------------------------------------------------

// e4m3(q) * 2^(e - 127) assembled as f32 bits. The code is a 4-bit
// significand times a power of two, so the product is a normal, an exact
// subnormal (the smallest is 2^-136) or an overflow to inf, and integer ops
// give it without relying on the float unit's denormal handling. E4M3 has
// no infinity: codes 0x7F and 0xFF are NaN, and so is the scale e == 0xFF.
inline float kq_fp_mxfp8_value(uint8_t e, uint8_t q) {
  const uint sign = uint(q & 0x80) << 24;
  if (e == 0xFF || (q & 0x7F) == 0x7F) {
    return as_type<float>(sign | 0x7FC00000u);
  }
  const uint ex = (q >> 3) & 0xF;
  const uint man = q & 0x7;
  // Normal codes are (8 + man) * 2^(ex - 10), subnormal codes man * 2^-9.
  const uint sig = ex == 0 ? man : 8 + man;
  if (sig == 0) {
    return as_type<float>(sign);
  }
  const int k = (ex == 0 ? -9 : int(ex) - 10) + int(e) - 127;
  const int p = 31 - int(clz(sig)); // leading significand bit, 0..3
  const int b = p + k + 127; // biased f32 exponent
  uint bits;
  if (b >= 255) {
    bits = 0x7F800000u;
  } else if (b >= 1) {
    bits = uint(b) << 23 | ((sig << (23 - p)) & 0x7FFFFFu);
  } else {
    bits = sig << (k + 149);
  }
  return as_type<float>(sign | bits);
}

template <typename T>
METAL_FUNC void kq_mxfp8_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_MXFP8_GROUP;
  const int within = gid % KQ_MXFP8_GROUP;
  const device uint8_t* block_addr =
      w + static_cast<int64_t>(block_id) * KQ_MXFP8_BLOCK_BYTES;
  out[gid] = T(kq_fp_mxfp8_value(
      block_addr[0], block_addr[KQ_MXFP8_QS_OFFSET + within]));
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_mxfp8_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(group_size == KQ_MXFP8_GROUP, "MXFP8 requires gs=32");
  static_assert(bits == 8, "MXFP8 requires bits=8");
  kq_mxfp8_dequantize_impl<T>(w, out, num_weights, gid);
}
