// STQ1_0 (llama.cpp PR #22836): 42-byte block [qs[32]][sign[8]][fp16 d].
// Group g = (chunk g/16, gloc g%16) covers weights chunk*64 + gloc + p*16, so
// 16 contiguous weights are plane p = il%4 of chunk il/4. The 42-byte block
// stride is 2 mod 4, so only half/ushort loads are alignment-safe.

MLX_MTL_CONST int KQ_STQ1_0_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_STQ1_0_BLOCK_BYTES = 42;
MLX_MTL_CONST int KQ_STQ1_0_QS_OFFSET = 0;
MLX_MTL_CONST int KQ_STQ1_0_SIGN_OFFSET = 32;
MLX_MTL_CONST int KQ_STQ1_0_D_OFFSET = 40;

// Packs the codebook bytes of four consecutive groups: q4 holds their slot
// nibbles (low first), sbits their sign bits at 0..3.
METAL_FUNC uint kq_stq1_0_cbw(ushort q4, ushort sbits) {
  uint cbw = 0;
#pragma unroll
  for (short j = 0; j < 4; ++j) {
    cbw |= uint(stq1_0_codebook
                    [(((sbits >> j) & 1) << 4) | ((q4 >> (4 * j)) & 0xF)])
        << (8 * j);
  }
  return cbw;
}

METAL_FUNC float4 kq_stq1_0_peel(uint cbw, short p) {
  const uint v = (cbw >> (2 * p)) & 0x03030303u;
  return float4(as_type<uchar4>(v)) - 1.0f;
}

template <typename T>
METAL_FUNC void kq_stq1_0_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_STQ1_0_SUPERBLOCK;
  const int within = gid - sb_id * KQ_STQ1_0_SUPERBLOCK;
  const short chunk = within >> 6;
  const short p = (within >> 4) & 3;
  const short gloc = within & 15;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_STQ1_0_BLOCK_BYTES;
  const float d = float(*(const device half*)(sb + KQ_STQ1_0_D_OFFSET));
  const short g = chunk * 16 + gloc;
  const short slot = (sb[g >> 1] >> (4 * (g & 1))) & 0xF;
  const short sbit = (sb[KQ_STQ1_0_SIGN_OFFSET + (g >> 3)] >> (g & 7)) & 1;
  const uint8_t cb = stq1_0_codebook[(sbit << 4) | slot];
  out[gid] = T(d * float(short((cb >> (2 * p)) & 3) - 1));
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_stq1_0_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_STQ1_0_SUPERBLOCK, "STQ1_0 kernel requires gs=256");
  static_assert(bits == 1, "STQ1_0 kernel requires bits=1");
  kq_stq1_0_dequantize_impl<T>(w, out, num_weights, gid);
}

inline void kq_stq1_0_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const short chunk = il >> 2;
  const short p = il & 3;
  const float d = float(*(const device half*)(block + KQ_STQ1_0_D_OFFSET));
  const device ushort* qsw =
      reinterpret_cast<const device ushort*>(block) + 4 * chunk;
  const ushort sg =
      *(reinterpret_cast<const device ushort*>(block + KQ_STQ1_0_SIGN_OFFSET) +
        chunk);
#pragma unroll
  for (short k = 0; k < 4; ++k) {
    const uint cbw = kq_stq1_0_cbw(qsw[k], (sg >> (4 * k)) & 0xF);
    reg[k] = kq_stq1_0_peel(cbw, p) * d;
  }
}

struct KqStq1_0Ext {
  MLX_MTL_CONST int superblock = KQ_STQ1_0_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_STQ1_0_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_stq1_0_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_stq1_0_mv_ext(
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
  kq_mv_ext_impl<T, KqStq1_0Ext, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// deq_chunk16s returns scale = d with raw {-1,0,+1} lanes; the hoisted scale
// makes the fused path tolerance-equal to the scaled decode, not bit-equal.
template <>
struct KqTgLuts<KqStq1_0Ext> {
  MLX_MTL_CONST int bytes = 32;
  static METAL_FUNC void
  stage(threadgroup uint8_t* dst, ushort lin, ushort n_threads) {
    threadgroup uint32_t* d32 = reinterpret_cast<threadgroup uint32_t*>(dst);
    const constant uint32_t* cb32 =
        reinterpret_cast<const constant uint32_t*>(stq1_0_codebook);
    for (int i = lin; i < 8; i += n_threads) {
      d32[i] = cb32[i];
    }
  }
  static METAL_FUNC void deq_chunk16s(
      const device uint8_t* block,
      short il,
      thread float4x4& reg,
      const threadgroup uint8_t* luts,
      thread float& scale) {
    const short chunk = il >> 2;
    const short p = il & 3;
    scale = float(*(const device half*)(block + KQ_STQ1_0_D_OFFSET));
    const device ushort* qsw =
        reinterpret_cast<const device ushort*>(block) + 4 * chunk;
    const ushort sg = *(
        reinterpret_cast<const device ushort*>(block + KQ_STQ1_0_SIGN_OFFSET) +
        chunk);
#pragma unroll
    for (short k = 0; k < 4; ++k) {
      const ushort q4 = qsw[k];
      uint cbw = 0;
#pragma unroll
      for (short j = 0; j < 4; ++j) {
        cbw |= uint(luts
                        [(((sg >> (4 * k + j)) & 1) << 4) |
                         ((q4 >> (4 * j)) & 0xF)])
            << (8 * j);
      }
      reg[k] = kq_stq1_0_peel(cbw, p);
    }
  }
  static METAL_FUNC void deq_chunk16(
      const device uint8_t* block,
      short il,
      thread float4x4& reg,
      const threadgroup uint8_t* luts) {
    float scale;
    deq_chunk16s(block, il, reg, luts, scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      reg[i] *= scale;
    }
  }
};

// ===================== STQ1_0 matmul / gather / qmv =====================

// Lane simd_lid owns weights [simd_lid*8, +8): chunk lid/8, plane (lid/2)%4,
// half lid%2.
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_stq1_0_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_STQ1_0_SUPERBLOCK, "STQ1_0 requires gs=256");
  static_assert(bits == 1, "STQ1_0 requires bits=1");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_STQ1_0_BLOCK_BYTES / KQ_STQ1_0_SUPERBLOCK;
  const int nb = in_vec_size / KQ_STQ1_0_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const short chunk = simd_lid >> 3;
  const short p = (simd_lid >> 1) & 3;
  const short h = simd_lid & 1;
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_STQ1_0_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_STQ1_0_BLOCK_BYTES;
      const U d = U(float(*(const device half*)(sb + KQ_STQ1_0_D_OFFSET)));
      const device ushort* qsw =
          reinterpret_cast<const device ushort*>(sb) + 4 * chunk;
      const ushort sg =
          *(reinterpret_cast<const device ushort*>(sb + KQ_STQ1_0_SIGN_OFFSET) +
            chunk);
      const uint cbw0 = kq_stq1_0_cbw(qsw[2 * h], (sg >> (8 * h)) & 0xF);
      const uint cbw1 =
          kq_stq1_0_cbw(qsw[2 * h + 1], (sg >> (8 * h + 4)) & 0xF);
      const float4 w0 = kq_stq1_0_peel(cbw0, p);
      const float4 w1 = kq_stq1_0_peel(cbw1, p);
      U partial = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        partial += xt[j] * U(w0[j]);
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        partial += xt[4 + j] * U(w1[j]);
      }
      result[row] += d * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_stq1_0_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_stq1_0_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqStq1_0BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_STQ1_0_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_STQ1_0_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "STQ1_0 loader requires BCOLS==32.");
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

  KqStq1_0BlockLoader(
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
    // A 32-weight sub-block is planes {0,1} or {2,3} of chunk sb/2; it never
    // straddles a chunk.
    static_assert(n_reads % 8 == 0, "vector loader needs whole 8-runs");
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const short chunk = sb >> 1;
    const float d = float(*(const device half*)(src + KQ_STQ1_0_D_OFFSET));
    const device ushort* qsw =
        reinterpret_cast<const device ushort*>(src) + 4 * chunk;
    const ushort sg =
        *(reinterpret_cast<const device ushort*>(src + KQ_STQ1_0_SIGN_OFFSET) +
          chunk);
#pragma unroll
    for (short t = 0; t < n_reads / 8; ++t) {
      const short o = (sb & 1) * 32 + bj + 8 * t;
      const short p = o >> 4;
      const short h = (o >> 3) & 1;
      const uint cbw0 = kq_stq1_0_cbw(qsw[2 * h], (sg >> (8 * h)) & 0xF);
      const uint cbw1 =
          kq_stq1_0_cbw(qsw[2 * h + 1], (sg >> (8 * h + 4)) & 0xF);
      *(threadgroup vec<T, 4>*)(dst + 8 * t) =
          vec<T, 4>(kq_stq1_0_peel(cbw0, p) * d);
      *(threadgroup vec<T, 4>*)(dst + 8 * t + 4) =
          vec<T, 4>(kq_stq1_0_peel(cbw1, p) * d);
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_stq1_0_qmm_t(
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
  static_assert(group_size == KQ_STQ1_0_SUPERBLOCK, "STQ1_0 requires gs=256");
  static_assert(bits == 1, "STQ1_0 requires bits=1");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqStq1_0BlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    int small_bm = 0>
[[kernel]] void kq_stq1_0_qmm_t_splitk(
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
  static_assert(group_size == KQ_STQ1_0_SUPERBLOCK, "STQ1_0 requires gs=256");
  static_assert(bits == 1, "STQ1_0 requires bits=1");
  constexpr int BM = small_bm ? small_bm : 32;
  constexpr int BK = 32, BN = small_bm ? 64 : 32;
  constexpr int WM = BM == 8 ? 1 : 2, WN = 4 / WM;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqStq1_0BlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(
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
[[kernel]] void kq_stq1_0_qmm_n(
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
  static_assert(group_size == KQ_STQ1_0_SUPERBLOCK, "STQ1_0 requires gs=256");
  static_assert(bits == 1, "STQ1_0 requires bits=1");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqStq1_0BlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_stq1_0_qmv_fast(
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
  kq_stq1_0_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer tiling: 1 result per simdgroup, 2 output rows per threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_stq1_0_qmv_fast_fine(
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
  kq_stq1_0_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_stq1_0_qmv(
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
  kq_stq1_0_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_stq1_0_qmv_fine(
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
  kq_stq1_0_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}
