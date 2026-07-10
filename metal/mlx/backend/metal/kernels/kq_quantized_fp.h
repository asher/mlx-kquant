// Native micro-scaling float formats as defined by ggml (llama.cpp, MIT) -
// see mlx_kquant/licenses/llama.cpp-LICENSE.
//
// MXFP4: 17 bytes/32 weights. [uint8 E8M0 d][uint8 qs[16]], two-halves
//   nibbles: w[j] = d * E2M1[qs[j] & 0xF], w[j+16] = d * E2M1[qs[j] >> 4].
// NVFP4: 36 bytes/64 weights. [uint8 UE4M3 sd[4]][uint8 qs[32]], four
//   16-value groups; group g = sd[g] scaling qs[8g..8g+8) two-halves.
//
// Real-value E2M1 LUT + full scales (the Metal convention; the CPU kernels
// use ggml's doubled-int8 LUT + half scales - same products, never mixed
// within one kernel).

MLX_MTL_CONST int KQ_MXFP4_GROUP = 32;
MLX_MTL_CONST int KQ_MXFP4_BLOCK_BYTES = 17;
MLX_MTL_CONST int KQ_MXFP4_QS_OFFSET = 1;

MLX_MTL_CONST int KQ_NVFP4_SUPERBLOCK = 64;
MLX_MTL_CONST int KQ_NVFP4_BLOCK_BYTES = 36;
MLX_MTL_CONST int KQ_NVFP4_QS_OFFSET = 4;

constant float kq_fp_e2m1_lut[16] = {
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

// 2^(e-127) built from bits: metal::exp2 is fast-math and lands ulps off on
// some compiler versions. e == 0 is the f32 subnormal 2^-127.
inline float kq_fp_e8m0_scale(uint8_t e) {
  return as_type<float>(e == 0 ? 0x00400000u : uint(e) << 23);
}

// Unsigned E4M3 (bias 7); 0x00 and 0x7F (the NaN encoding) decode to 0.
// Normals assembled from bits (biased f32 exponent e + 120, mantissa m << 20)
// for the same fast-math reason; the subnormal product is exact as written.
inline float kq_fp_ue4m3_scale(uint8_t v) {
  if (v == 0 || v == 0x7F) {
    return 0.0f;
  }
  const uint e = (v >> 3) & 0xF;
  const uint m = v & 0x7;
  return e == 0 ? float(m) * 0.001953125f /* 2^-9 */
                : as_type<float>((e + 120u) << 23 | m << 20);
}

// ---------------------------------------------------------------------------
// MXFP4 (structure cloned from IQ4_NL: same two-halves 16-byte qs, the fp16
// scale replaced by the E8M0 byte and the int LUT by the E2M1 float LUT)
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
  out[gid] = T(d * kq_fp_e2m1_lut[nib]);
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
  for (int i = 0; i < 16; ++i) {
    const int nib = (int(qs[i]) >> shift) & 0x0F;
    reg[i / 4][i % 4] = d * kq_fp_e2m1_lut[nib];
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

// MXFP4 mat-vec: one impl for both qmv and qmv_fast (the LUT decode is cheap,
// so there's no separate aligned fast path). Lane `simd_lid` owns weight
// `lane` of every 32-block; simd_sum reduces the 32 lanes.
template <typename T, int group_size, int bits>
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
  constexpr int results_per_simdgroup = 4; // bn = 8
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
  const bool is_high = simd_lid >= 16;
  const int byteidx = is_high ? int(simd_lid) - 16 : int(simd_lid);
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    const U xv = U(x[ib * KQ_MXFP4_GROUP + simd_lid]);
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* blk = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_MXFP4_BLOCK_BYTES;
      const U d = U(kq_fp_e8m0_scale(blk[0]));
      const uint8_t b = blk[KQ_MXFP4_QS_OFFSET + byteidx];
      const int nib = is_high ? (b >> 4) : (b & 0x0F);
      result[row] += d * U(kq_fp_e2m1_lut[nib]) * xv;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_mxfp4_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_mxfp4_qmv_impl<T, group_size, bits>(
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
      int /* col_in_block */ = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const float d = kq_fp_e8m0_scale(src[0]);
    const device uint8_t* qs = src + KQ_MXFP4_QS_OFFSET + bj_byte;
#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t b = qs[i];
      dst[i] = T(d * kq_fp_e2m1_lut[b & 0x0F]);
      dst[half_block + i] = T(d * kq_fp_e2m1_lut[b >> 4]);
    }
  }

  void load_safe(short2 src_tile_dim) const {
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

  void next() {
    src += tile_stride;
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
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
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqMxfp4BlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_mxfp4_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
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
  constexpr int BM = 32, BK = 32, BN = 32;
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
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
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
  out[gid] = T(d * kq_fp_e2m1_lut[nib]);
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

// Natural-order chunk [il*16, il*16+16) == sub-scale group il; il in [0,4).
METAL_FUNC void kq_nvfp4_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const float d = kq_fp_ue4m3_scale(block[il]);
  const device uint8_t* qs = block + KQ_NVFP4_QS_OFFSET + 8 * il;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const uint8_t b = qs[i % 8];
    const int nib = (i < 8) ? (b & 0x0F) : (b >> 4);
    reg[i / 4][i % 4] = d * kq_fp_e2m1_lut[nib];
  }
}

struct KqNvfp4Ext {
  MLX_MTL_CONST int superblock = KQ_NVFP4_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_NVFP4_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_nvfp4_deq_chunk16(block, il, reg);
  }
};

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
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqNvfp4Ext, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// NVFP4 mat-vec: lane `simd_lid` owns weights `lane` and `lane+32` of every
// 64-superblock (groups 0-1 and 2-3 respectively); simd_sum reduces.
template <typename T, int group_size, int bits>
METAL_FUNC void kq_nvfp4_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_NVFP4_SUPERBLOCK, "NVFP4 requires gs=64");
  static_assert(bits == 4, "NVFP4 requires bits=4");
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4; // bn = 8
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
  const int g0 = int(simd_lid) / 16; // 0 or 1
  const int j = int(simd_lid) % 16;
  const bool is_high = j >= 8;
  const int byteidx = 8 * g0 + (j % 8);
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    const U xv0 = U(x[ib * KQ_NVFP4_SUPERBLOCK + simd_lid]);
    const U xv1 = U(x[ib * KQ_NVFP4_SUPERBLOCK + 32 + simd_lid]);
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* blk = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_NVFP4_BLOCK_BYTES;
      const U d0 = U(kq_fp_ue4m3_scale(blk[g0]));
      const U d1 = U(kq_fp_ue4m3_scale(blk[g0 + 2]));
      const uint8_t b0 = blk[KQ_NVFP4_QS_OFFSET + byteidx];
      const uint8_t b1 = blk[KQ_NVFP4_QS_OFFSET + 16 + byteidx];
      const int nib0 = is_high ? (b0 >> 4) : (b0 & 0x0F);
      const int nib1 = is_high ? (b1 >> 4) : (b1 & 0x0F);
      result[row] += d0 * U(kq_fp_e2m1_lut[nib0]) * xv0 +
          d1 * U(kq_fp_e2m1_lut[nib1]) * xv1;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_nvfp4_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_nvfp4_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
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
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const short t = sb * sub_block_size + bj + i; // weight in superblock
      const short g = t / 16;
      const short j = t % 16;
      const float d = kq_fp_ue4m3_scale(src[g]);
      const uint8_t b = src[KQ_NVFP4_QS_OFFSET + 8 * g + (j % 8)];
      const int nib = (j < 8) ? (b & 0x0F) : (b >> 4);
      dst[i] = T(d * kq_fp_e2m1_lut[nib]);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
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
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqNvfp4BlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_nvfp4_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
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
  constexpr int BM = 32, BK = 32, BN = 32;
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
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
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
  kq_nvfp4_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
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
  kq_nvfp4_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}
