// K-quant superblock decode math derives from ggml (llama.cpp, MIT); the
// qmv / qmm kernel structure (tiling, simdgroup layout) is adapted from MLX's
// quantized kernels (MIT). See mlx_kquant/licenses/.
#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/kq_quantized_iq_tables.h"

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

struct kq_empty {};

template <
    typename T,
    typename LoaderW,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void kq_qmm_t_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const int K_eff,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be >= SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be a multiple of SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      /*transpose_a=*/false,
      /*transpose_b=*/true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = w;

  x += y_row * static_cast<int64_t>(K);
  wl += static_cast<int64_t>(y_col) * K_w;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  LoaderW loader_w(wl, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename T,
    typename LoaderW,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void kq_qmm_n_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be >= SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be a multiple of SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      /*transpose_a=*/false,
      /*transpose_b=*/false,
      BK_padded,
      BN_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;

  auto wl = w;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += (y_col / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  LoaderW loader_w(
      wl, N, Ws, simd_gid, simd_lid, y_col % LoaderW::weights_per_block);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, num_els));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, BM));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename T>
METAL_FUNC void kq_adjust_matrix_offsets(
    const device T*& x,
    const device uint8_t*& w,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
  } else {
    w += elem_to_loc(w_idx, w_shape, w_strides, w_batch_ndims);
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void kq_adjust_matrix_offsets(
    const device T*& x,
    const device uint8_t*& w,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
  } else {
    w += elem_to_loc(w_idx, w_shape, w_strides, w_batch_ndims);
  }
  y += tid.z * output_stride;
}

// ---------------------------------------------------------------------------
// Flat-with-M verify mat-vec (port of ggml-metal kernel_mul_mv_ext_q4x4),
// codec-agnostic. The per-row qmv puts M on grid_dims.x, so each of the M rows
// runs its own threadgroup that re-reads the weight tile - M-x the weight
// traffic for the same answer, and under GPU saturation (the verify forward's
// real condition) the M dots stay exposed. This decomposition stays flat with
// M: the 32 simdgroup lanes are nxpsg along K x nypsg output rows, so nypsg
// output rows compute in parallel per simdgroup, EACH thread owns ONE row and
// holds only r1ptg(=M) float accumulators, and the K-reduction is an
// nxpsg-lane simd_shuffle_down. The weight row streams once and is dotted
// against all M activation columns - same weight DRAM traffic as M=1. r1ptg is
// the compile-time draft width (== runtime vm); dispatched for M in [2,12].
// Bit-exact (fp-noise) vs verify_qmv and per-row qmv.
//
// Codec is a traits struct giving super-block geometry (superblock,
// block_bytes) and deq_chunk16(block, il, reg): dequantize the 16 contiguous
// weights of chunk il into a float4x4 in natural order [il*16, il*16+16). The
// ggml dequantize_* permutation of il yields exactly that natural order, so the
// contiguous activation read below pairs each weight with its activation.
// Non-batched (B == 1) only; x is row-contiguous [vm, K], y is [vm, N].
// Derived from llama.cpp ggml-metal.metal (MIT); attribution in licenses/.
template <typename T, typename Codec, short r1ptg, short nsg, short nxpsg>
METAL_FUNC void kq_mv_ext_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    uint3 tgpig,
    ushort tiisg,
    ushort sgitg) {
  constexpr short nypsg = 32 / nxpsg; // output rows per simdgroup
  constexpr short chpb = Codec::superblock / 16; // 16-weight chunks per block
  const short tx = tiisg % nxpsg; // K position within the row group
  const short ty = tiisg / nxpsg; // which of nypsg rows this thread owns

  const int i01 = tgpig.x * (nypsg * nsg) + nypsg * sgitg + ty; // output row
  const int i11 = tgpig.y * r1ptg; // first activation column (grid.y==1 -> 0)

  const int nb = in_vec_size / Codec::superblock;
  const int row_bytes = nb * Codec::block_bytes;
  // Clamp OOB rows to row 0 for a valid read; the store is masked below.
  const device uint8_t* w_row =
      (i01 < out_vec_size) ? w + static_cast<int64_t>(i01) * row_bytes : w;

  const device T* y_col[r1ptg];
#pragma unroll
  for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
    y_col[ir1] = x + static_cast<int64_t>(i11 + ir1) * in_vec_size + tx * 16;
  }

  float sumf[r1ptg];
#pragma unroll
  for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
    sumf[ir1] = 0.0f;
  }

  for (int ich = tx; 16 * ich < in_vec_size; ich += nxpsg) {
    const int ib = ich / chpb; // super-block index
    const short cch = ich % chpb; // chunk within super-block
    const device uint8_t* block =
        w_row + static_cast<int64_t>(ib) * Codec::block_bytes;
    float4x4 lx;
    Codec::deq_chunk16(block, cch, lx);
#pragma unroll
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
      const device T* yp = y_col[ir1];
      const float4 a0 = float4(*(const device vec<T, 4>*)(yp + 0));
      const float4 a1 = float4(*(const device vec<T, 4>*)(yp + 4));
      const float4 a2 = float4(*(const device vec<T, 4>*)(yp + 8));
      const float4 a3 = float4(*(const device vec<T, 4>*)(yp + 12));
      sumf[ir1] +=
          dot(lx[0], a0) + dot(lx[1], a1) + dot(lx[2], a2) + dot(lx[3], a3);
      y_col[ir1] += nxpsg * 16;
    }
  }

#pragma unroll
  for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
    if (nxpsg >= 32) {
      sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
    }
    if (nxpsg >= 16) {
      sumf[ir1] += simd_shuffle_down(sumf[ir1], 8);
    }
    if (nxpsg >= 8) {
      sumf[ir1] += simd_shuffle_down(sumf[ir1], 4);
    }
    if (nxpsg >= 4) {
      sumf[ir1] += simd_shuffle_down(sumf[ir1], 2);
    }
    if (nxpsg >= 2) {
      sumf[ir1] += simd_shuffle_down(sumf[ir1], 1);
    }
  }

  if (tx == 0 && i01 < out_vec_size) {
#pragma unroll
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
      y[static_cast<int64_t>(i11 + ir1) * out_vec_size + i01] =
          static_cast<T>(sumf[ir1]);
    }
  }
}

// Q8_0: 34 bytes/32 weights. [fp16 d][int8 q[32]]. w[i] = d * q[i].

MLX_MTL_CONST int KQ_Q8_0_GROUP = 32;
MLX_MTL_CONST int KQ_Q8_0_BLOCK_BYTES = 34;
MLX_MTL_CONST int KQ_Q8_0_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q8_0_Q_OFFSET = 2;

inline float kq_q8_0_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q8_0_D_OFFSET));
}

inline const device int8_t* kq_q8_0_q_ptr(const device uint8_t* block_addr) {
  return (const device int8_t*)(block_addr + KQ_Q8_0_Q_OFFSET);
}

template <typename T>
METAL_FUNC void kq_q8_0_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_Q8_0_GROUP;
  const int within = gid % KQ_Q8_0_GROUP;
  const device uint8_t* block_addr =
      w + static_cast<int64_t>(block_id) * KQ_Q8_0_BLOCK_BYTES;
  const float d = kq_q8_0_d(block_addr);
  const int8_t q = kq_q8_0_q_ptr(block_addr)[within];
  out[gid] = T(d * float(q));
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q8_0_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    const device T* bias = nullptr) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int lane_k_offset = simd_lid * values_per_thread;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x + k + lane_k_offset, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* row_base =
          w + static_cast<int64_t>(row_idx) * row_bytes;

      const int k_global = k + lane_k_offset;
      const int block_id = k_global / KQ_Q8_0_GROUP;
      const int within = k_global - block_id * KQ_Q8_0_GROUP;
      const device uint8_t* block_addr =
          row_base + block_id * KQ_Q8_0_BLOCK_BYTES;

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
      U out_val = result[row];
      if (bias != nullptr) {
        out_val += U(bias[out_row + row]);
      }
      y[out_row + row] = static_cast<T>(out_val);
    }
  }
}

// Verify-shaped qmv (see kq_q6_k_verify_qmv_impl). Reads each weight block once
// per output row and dots it against all `vm` activation rows. Non-batched
// only. results_per_simdgroup is templated: the default (4) packs 8 output rows
// per threadgroup; a finer value (1 -> 2 rows/tg) multiplies the threadgroup
// count for the same N, restoring GPU occupancy when each row's weight is small
// enough to stay L2-resident (so amortizing the weight read saves little DRAM
// traffic and the lost occupancy isn't repaid). Bit-exact across values: each
// output row's per-lane sequential K-fold + simd_sum is identical regardless of
// how rows are partitioned across threadgroups.
template <typename T, int group_size, int bits, int results_per_simdgroup = 4>
METAL_FUNC void kq_q8_0_verify_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& vm,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");

  constexpr int num_simdgroups = 2;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int MAX_VM = 8;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U wq[values_per_thread];
  thread U result[MAX_VM][results_per_simdgroup];
#pragma unroll
  for (int m = 0; m < MAX_VM; m++) {
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[m][row] = U(0);
    }
  }

  const int row_bytes = in_vec_size * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int lane_k_offset = simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* row_base =
          w + static_cast<int64_t>(row_idx) * row_bytes;
      const int k_global = k + lane_k_offset;
      const int block_id = k_global / KQ_Q8_0_GROUP;
      const int within = k_global - block_id * KQ_Q8_0_GROUP;
      const device uint8_t* block_addr =
          row_base + block_id * KQ_Q8_0_BLOCK_BYTES;
      const U d = U(kq_q8_0_d(block_addr));
      const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        wq[i] = U(q_ptr[i]); // dequantize weights once per output row
      }

      for (int m = 0; m < vm; m++) {
        load_vector<T, U, values_per_thread>(
            x + m * in_vec_size + k + lane_k_offset, x_thread);
        U partial = 0;
#pragma unroll
        for (int i = 0; i < values_per_thread; i++) {
          partial += x_thread[i] * wq[i];
        }
        result[m][row] += d * partial;
      }
    }
  }

  for (int m = 0; m < vm; m++) {
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      U r = simd_sum(result[m][row]);
      if (simd_lid == 0) {
        y[m * out_vec_size + out_row + row] = static_cast<T>(r);
      }
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q8_0_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid,
    const device T* bias = nullptr) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int lane_k_offset = simd_lid * values_per_thread;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int k_remaining = in_vec_size - k - lane_k_offset;
    if (k_remaining >= values_per_thread) {
      load_vector<T, U, values_per_thread>(x + k + lane_k_offset, x_thread);
    } else if (k_remaining > 0) {
      load_vector_safe<T, U, values_per_thread>(
          x + k + lane_k_offset, x_thread, k_remaining);
    } else {
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        x_thread[i] = 0;
      }
    }

    const int n_inner = k_remaining >= values_per_thread
        ? values_per_thread
        : (k_remaining > 0 ? k_remaining : 0);

    if (n_inner == 0) {
      continue;
    }

    const int k_global = k + lane_k_offset;
    const int block_id = k_global / KQ_Q8_0_GROUP;
    const int within = k_global - block_id * KQ_Q8_0_GROUP;

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* row_base =
          w + static_cast<int64_t>(row_idx) * row_bytes;
      const device uint8_t* block_addr =
          row_base + block_id * KQ_Q8_0_BLOCK_BYTES;

      const U d = U(kq_q8_0_d(block_addr));
      const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;

      U partial = 0;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        if (i < n_inner) {
          partial += x_thread[i] * U(q_ptr[i]);
        }
      }
      result[row] += d * partial;
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      U out_val = result[row];
      if (bias != nullptr) {
        out_val += U(bias[out_row + row]);
      }
      y[out_row + row] = static_cast<T>(out_val);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqQ8_0BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q8_0_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q8_0_BLOCK_BYTES;

  static_assert(
      BCOLS == weights_per_block,
      "Q8_0 loader requires BCOLS == 32 (one block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;

  KqQ8_0BlockLoader(
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
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const float d = float(*(const device half*)src);
    const device int8_t* q =
        (const device int8_t*)(src + KQ_Q8_0_Q_OFFSET + bj);
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      dst[i] = T(d * float(q[i]));
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
    const float d = float(*(const device half*)src);
    const device int8_t* q =
        (const device int8_t*)(src + KQ_Q8_0_Q_OFFSET + bj);
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      dst[i] = T(d * float(q[i]));
    }
  }

  void next() {
    src += tile_stride;
  }
};

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_q8_0_qmm_t(
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
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ8_0BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_q8_0_qmm_t_splitk(
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
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ8_0BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;

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
[[kernel]] void kq_q8_0_qmm_n(
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
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ8_0BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q8_0_qmv_fast(
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
  kq_q8_0_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Bias-fused variant of kq_q8_0_qmv_fast: decode-only (B=1) fast path for a
// KQuantLinear whose GGUF weight carries a real linear bias (e.g. gpt-oss
// attention QKVO). Fuses the post-matmul add into this kernel's own output
// write instead of a separate elementwise dispatch. Non-batched only -- the
// batched (MoE-style) axis isn't wired for this variant; KQuantLinear never
// batches, so it isn't needed here.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q8_0_qmv_fast_bias(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const device T* bias,
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
  kq_q8_0_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid, bias);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q8_0_verify_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& vm,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_q8_0_verify_qmv_impl<T, group_size, bits, 4>(
      w, x, y, in_vec_size, out_vec_size, vm, tid, simd_gid, simd_lid);
}

// Finer-tiled variant: 1 result per simdgroup -> 2 output rows per threadgroup
// (vs 8 in the default), quadrupling the threadgroup count for the same N. Used
// when occupancy is the bottleneck (small per-row weight that stays
// L2-resident). Bit-exact vs the default variant and vs per-row qmv.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q8_0_verify_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& vm,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_q8_0_verify_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, vm, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q8_0_qmv(
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
  kq_q8_0_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Bias-fused variant of kq_q8_0_qmv -- see kq_q8_0_qmv_fast_bias above.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q8_0_qmv_bias(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const device T* bias,
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
  kq_q8_0_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid, bias);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q8_0_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 kernel requires bits=8");
  kq_q8_0_dequantize_impl<T>(w, out, num_weights, gid);
}

// q8_0 flat-with-M verify mat-vec: kq_mv_ext_impl (see above) + chunk dequant.
// Direct port of ggml-metal dequantize_q8_0: the 16 contiguous int8 weights of
// chunk il, natural order [il*16, il*16+16). A q8_0 block is 32 weights, so a
// 16-weight chunk is half a block (il in [0,1]).
inline void kq_q8_0_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const float d = kq_q8_0_d(block);
  const device int8_t* qs = kq_q8_0_q_ptr(block) + 16 * il;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    reg[i / 4][i % 4] = float(qs[i]) * d;
  }
}

struct KqQ8_0Ext {
  MLX_MTL_CONST int superblock = KQ_Q8_0_GROUP;
  MLX_MTL_CONST int block_bytes = KQ_Q8_0_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_q8_0_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_q8_0_mv_ext(
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
  kq_mv_ext_impl<T, KqQ8_0Ext, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

#include "mlx/backend/metal/kernels/kq_quantized_legacy.h"

// Q5_1: 24 bytes/32 weights. [fp16 d][fp16 m][uint32 qh][uint8 qs[16]].
// q5 = (low4 | high_bit<<4); w[i] = d * q5[i] + m.

MLX_MTL_CONST int KQ_Q5_1_GROUP = 32;
MLX_MTL_CONST int KQ_Q5_1_BLOCK_BYTES = 24;
MLX_MTL_CONST int KQ_Q5_1_D_OFFSET = 0;
MLX_MTL_CONST int KQ_Q5_1_M_OFFSET = 2;
MLX_MTL_CONST int KQ_Q5_1_QH_OFFSET = 4;
MLX_MTL_CONST int KQ_Q5_1_QS_OFFSET = 8;

inline float kq_q5_1_d(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q5_1_D_OFFSET));
}
inline float kq_q5_1_m(const device uint8_t* block_addr) {
  return float(*(const device half*)(block_addr + KQ_Q5_1_M_OFFSET));
}
inline uint32_t kq_q5_1_qh(const device uint8_t* block_addr) {
  return *(const device uint32_t*)(block_addr + KQ_Q5_1_QH_OFFSET);
}
inline const device uint8_t* kq_q5_1_qs_ptr(const device uint8_t* block_addr) {
  return block_addr + KQ_Q5_1_QS_OFFSET;
}

template <typename T>
METAL_FUNC void kq_q5_1_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int block_id = gid / KQ_Q5_1_GROUP;
  const int within = gid % KQ_Q5_1_GROUP;
  const device uint8_t* block_addr =
      w + static_cast<int64_t>(block_id) * KQ_Q5_1_BLOCK_BYTES;
  const float d = kq_q5_1_d(block_addr);
  const float m = kq_q5_1_m(block_addr);
  const uint32_t qh = kq_q5_1_qh(block_addr);
  const device uint8_t* qs = kq_q5_1_qs_ptr(block_addr);
  const uint32_t hi = ((qh >> within) << 4) & 0x10u;
  const uint8_t lo =
      (within < 16) ? (qs[within] & 0x0Fu) : (qs[within - 16] >> 4);
  const float q5 = float(uint32_t(lo) | hi);
  out[gid] = T(d * q5 + m);
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q5_1_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int row_bytes = in_vec_size * KQ_Q5_1_BLOCK_BYTES / KQ_Q5_1_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q5_1_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q5_1_GROUP + il;
    U sumy = U(0);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const U a0 = U(x[x_base + i + 0]);
      const U a1 = U(x[x_base + i + 1]);
      const U b0 = U(x[x_base + i + 16]);
      const U b1 = U(x[x_base + i + 17]);
      sumy += a0 + a1 + b0 + b1;
      yl[i + 0] = a0;
      yl[i + 1] = a1 * (U(1) / U(256));
      yl[i + 8] = b0 * (U(1) / U(16));
      yl[i + 9] = b1 * (U(1) / U(4096));
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr = w +
          static_cast<int64_t>(row_idx) * row_bytes + ib * KQ_Q5_1_BLOCK_BYTES;
      const U d = U(kq_q5_1_d(block_addr));
      const U m = U(kq_q5_1_m(block_addr));
      const uint32_t qh = kq_q5_1_qh(block_addr);
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q5_1_qs_ptr(block_addr)) +
          il / 2;

      U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        acc[0] += yl[i + 0] *
            U((qi & 0x000F) | (((qh >> (i + 0 + il)) << 4) & 0x00010));
        acc[1] += yl[i + 1] *
            U((qi & 0x0F00) | (((qh >> (i + 1 + il)) << 12) & 0x01000));
        acc[2] += yl[i + 8] *
            U((qi & 0x00F0) | (((qh >> (i + 0 + il + 16)) << 8) & 0x00100));
        acc[3] += yl[i + 9] *
            U((qi & 0xF000) | (((qh >> (i + 1 + il + 16)) << 16) & 0x10000));
      }
      result[row] += d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}

// Verify-shaped qmv (see kq_q6_k_verify_qmv_impl). The masked weight values
// (incl. the 5th bit from qh) are cached once per output row; the m-loop
// rebuilds the cheap per-position activation scaling and dots, so the dominant
// per-row weight read is amortized while the math stays bit-for-bit the
// qmv_fast path.
template <typename T, int group_size, int bits>
METAL_FUNC void kq_q5_1_verify_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& vm,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;
  constexpr int MAX_VM = 8;

  typedef float U;
  thread U wm[16]; // masked weight value per yl position, cached per output row
  thread U yl[16];
  thread U result[MAX_VM][results_per_simdgroup];
#pragma unroll
  for (int m = 0; m < MAX_VM; m++) {
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[m][row] = U(0);
    }
  }

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int row_bytes = in_vec_size * KQ_Q5_1_BLOCK_BYTES / KQ_Q5_1_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int nb = in_vec_size / KQ_Q5_1_GROUP;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q5_1_GROUP + il;

    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr = w +
          static_cast<int64_t>(row_idx) * row_bytes + ib * KQ_Q5_1_BLOCK_BYTES;
      const U d = U(kq_q5_1_d(block_addr));
      const U mm = U(kq_q5_1_m(block_addr));
      const uint32_t qh = kq_q5_1_qh(block_addr);
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q5_1_qs_ptr(block_addr)) +
          il / 2;

      // --- unpack this output row's masked weights once ---
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        wm[i + 0] = U((qi & 0x000F) | (((qh >> (i + 0 + il)) << 4) & 0x00010));
        wm[i + 1] = U((qi & 0x0F00) | (((qh >> (i + 1 + il)) << 12) & 0x01000));
        wm[i + 8] =
            U((qi & 0x00F0) | (((qh >> (i + 0 + il + 16)) << 8) & 0x00100));
        wm[i + 9] =
            U((qi & 0xF000) | (((qh >> (i + 1 + il + 16)) << 16) & 0x10000));
      }

      // --- per activation row: rebuild scaled yl + dot ---
      for (int m = 0; m < vm; m++) {
        const device T* xm = x + m * in_vec_size + x_base;
        U sumy = U(0);
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const U a0 = U(xm[i + 0]);
          const U a1 = U(xm[i + 1]);
          const U b0 = U(xm[i + 16]);
          const U b1 = U(xm[i + 17]);
          sumy += a0 + a1 + b0 + b1;
          yl[i + 0] = a0;
          yl[i + 1] = a1 * (U(1) / U(256));
          yl[i + 8] = b0 * (U(1) / U(16));
          yl[i + 9] = b1 * (U(1) / U(4096));
        }
        U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          acc[0] += yl[i + 0] * wm[i + 0];
          acc[1] += yl[i + 1] * wm[i + 1];
          acc[2] += yl[i + 8] * wm[i + 8];
          acc[3] += yl[i + 9] * wm[i + 9];
        }
        result[m][row] += d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * mm;
      }
    }
  }

  for (int m = 0; m < vm; m++) {
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      U r = simd_sum(result[m][row]);
      if (simd_lid == 0) {
        y[m * out_vec_size + out_row + row] = static_cast<T>(r);
      }
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void kq_q5_1_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int block_stride = 16;

  typedef float U;
  thread U yl[16];
  thread U result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q5_1_BLOCK_BYTES / KQ_Q5_1_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }
  const int max_row = min(out_vec_size, out_row + results_per_simdgroup);
  const int active_rows = max_row - out_row;

  const int ix = simd_lid / 2;
  const int il = (simd_lid % 2) * 8;

  const int nb = in_vec_size / KQ_Q5_1_GROUP;

  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;

  for (int ib = ix; ib < nb; ib += block_stride) {
    const int x_base = ib * KQ_Q5_1_GROUP + il;
    U sumy = U(0);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const U a0 = U(x[x_base + i + 0]);
      const U a1 = U(x[x_base + i + 1]);
      const U b0 = U(x[x_base + i + 16]);
      const U b1 = U(x[x_base + i + 17]);
      sumy += a0 + a1 + b0 + b1;
      yl[i + 0] = a0;
      yl[i + 1] = a1 * (U(1) / U(256));
      yl[i + 8] = b0 * (U(1) / U(16));
      yl[i + 9] = b1 * (U(1) / U(4096));
    }

    for (int row = 0; row < active_rows; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* block_addr = w +
          static_cast<int64_t>(row_idx) * row_bytes + ib * KQ_Q5_1_BLOCK_BYTES;
      const U d = U(kq_q5_1_d(block_addr));
      const U m = U(kq_q5_1_m(block_addr));
      const uint32_t qh = kq_q5_1_qh(block_addr);
      const device uint16_t* qs =
          reinterpret_cast<const device uint16_t*>(kq_q5_1_qs_ptr(block_addr)) +
          il / 2;

      U acc[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const uint16_t qi = qs[i / 2];
        acc[0] += yl[i + 0] *
            U((qi & 0x000F) | (((qh >> (i + 0 + il)) << 4) & 0x00010));
        acc[1] += yl[i + 1] *
            U((qi & 0x0F00) | (((qh >> (i + 1 + il)) << 12) & 0x01000));
        acc[2] += yl[i + 8] *
            U((qi & 0x00F0) | (((qh >> (i + 0 + il + 16)) << 8) & 0x00100));
        acc[3] += yl[i + 9] *
            U((qi & 0xF000) | (((qh >> (i + 1 + il + 16)) << 16) & 0x10000));
      }
      result[row] += d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqQ5_1BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q5_1_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q5_1_BLOCK_BYTES;

  static_assert(
      BCOLS == weights_per_block,
      "Q5_1 loader requires BCOLS == 32 (one block per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  MLX_MTL_CONST short half_block = weights_per_block / 2;
  static_assert(n_reads >= 2 && n_reads % 2 == 0, "Q5_1 needs even n_reads.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;

  KqQ5_1BlockLoader(
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
    const float d = float(*(const device half*)(src + KQ_Q5_1_D_OFFSET));
    const float m = float(*(const device half*)(src + KQ_Q5_1_M_OFFSET));
    const uint32_t qh = *(const device uint32_t*)(src + KQ_Q5_1_QH_OFFSET);
    const device uint8_t* qs = src + KQ_Q5_1_QS_OFFSET + bj_byte;
    static_assert(
        bytes_per_thread == 4 || bytes_per_thread == 8,
        "Q5_1 ALU vector load supports bytes_per_thread=4 or 8 (uint).");
    uint8_t qs_b[bytes_per_thread];
#pragma unroll
    for (short v = 0; v < bytes_per_thread / 4; v++) {
      const uint qs_v = *reinterpret_cast<const device uint*>(qs + v * 4);
      *reinterpret_cast<thread uint*>(&qs_b[v * 4]) = qs_v;
    }
#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t b = qs_b[i];
      const int j_lo = bj_byte + i;
      const int j_hi = bj_byte + half_block + i;
      const uint32_t hi_lo = ((qh >> j_lo) << 4) & 0x10u;
      const uint32_t hi_hi = ((qh >> j_hi) << 4) & 0x10u;
      const float q5_lo = float(uint32_t(b & 0x0F) | hi_lo);
      const float q5_hi = float(uint32_t(b >> 4) | hi_hi);
      dst[i] = T(d * q5_lo + m);
      dst[half_block + i] = T(d * q5_hi + m);
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
[[kernel]] void kq_q5_1_qmm_t(
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
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ5_1BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_q5_1_qmm_t_splitk(
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
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW = KqQ5_1BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;

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
[[kernel]] void kq_q5_1_qmm_n(
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
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW = KqQ5_1BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_1_qmv_fast(
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
  kq_q5_1_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_1_verify_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& vm,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_q5_1_verify_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, vm, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_1_qmv(
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
  kq_q5_1_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q5_1_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 kernel requires bits=5");
  kq_q5_1_dequantize_impl<T>(w, out, num_weights, gid);
}

// q5_1 flat-with-M verify mat-vec: kq_mv_ext_impl (see above) + chunk dequant.
// Mirrors kq_q5_1_dequantize_impl (natural order [il*16, il*16+16)); il in
// [0,1]
// -> within base 0 then 16. 5th bit from qh bit `within`. w[i] = d * q5 + m,
// q5 = lo4 | (qh_bit << 4).
inline void kq_q5_1_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const float d = kq_q5_1_d(block);
  const float m = kq_q5_1_m(block);
  const uint32_t qh = kq_q5_1_qh(block);
  const device uint8_t* qs = kq_q5_1_qs_ptr(block);
  const int base = (il & 1) * 16;
  const int shift = (il & 1) ? 4 : 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int within = base + i;
    const uint32_t hi = ((qh >> within) << 4) & 0x10u;
    const int lo = (int(qs[i]) >> shift) & 0x0F;
    const float q5 = float(lo | int(hi));
    reg[i / 4][i % 4] = d * q5 + m;
  }
}

struct KqQ5_1Ext {
  MLX_MTL_CONST int superblock = KQ_Q5_1_GROUP;
  MLX_MTL_CONST int block_bytes = KQ_Q5_1_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_q5_1_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_q5_1_mv_ext(
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
  kq_mv_ext_impl<T, KqQ5_1Ext, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

inline void kq_get_scale_min_k4(
    int j,
    const device uint8_t* q,
    thread uint8_t& d_out,
    thread uint8_t& m_out) {
  const int j_lo = j & 3;
  const bool j_high = (j & 4) != 0;
  const uint8_t a = q[j_lo];
  const uint8_t b = q[j_lo + 4];
  const uint8_t c = q[j_lo + 8];
  const uint8_t d_low = a & 0x3F;
  const uint8_t m_low = b & 0x3F;
  const uint8_t d_high = (c & 0x0F) | ((a >> 6) << 4);
  const uint8_t m_high = (c >> 4) | ((b >> 6) << 4);
  d_out = j_high ? d_high : d_low;
  m_out = j_high ? m_high : m_low;
}

// ggml get_scale_min_k4_just2: returns {scale, min} (6-bit each) for sub-block
// pair (j, k) of a q4_k/q5_k 12-byte scales array. Used by the mv_ext chunk
// dequant; direct port of ggml-metal.
inline uchar2 kq_get_scale_min_k4_just2(int j, int k, const device uint8_t* q) {
  return j < 4 ? uchar2(q[j + 0 + k] & 63, q[j + 4 + k] & 63)
               : uchar2(
                     (q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2),
                     (q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2));
}

#include "mlx/backend/metal/kernels/kq_quantized_kquants.h"

#include "mlx/backend/metal/kernels/kq_quantized_iq.h"

#define KQUANT_DEFINE_GATHER_KERNELS(CODEC, LOADER)                   \
  template <typename T, int group_size, int bits>                     \
  [[kernel]] void kq_##CODEC##_gather_qmv_fast(                       \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      const device uint32_t* lhs_indices,                             \
      const device uint32_t* rhs_indices,                             \
      device T* y,                                                    \
      const constant int& in_vec_size,                                \
      const constant int& out_vec_size,                               \
      const constant int& x_batch_ndims,                              \
      const constant int* x_shape,                                    \
      const constant int64_t* x_strides,                              \
      const constant int& w_batch_ndims,                              \
      const constant int* w_shape,                                    \
      const constant int64_t* w_strides,                              \
      const constant int64_t* /* s_strides */,                        \
      const constant int& batch_ndims,                                \
      const constant int* batch_shape,                                \
      const constant int64_t* lhs_strides,                            \
      const constant int64_t* rhs_strides,                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    int M = x_shape[x_batch_ndims];                                   \
    kq_adjust_matrix_offsets<T>(                                      \
        x,                                                            \
        w,                                                            \
        lhs_indices,                                                  \
        rhs_indices,                                                  \
        y,                                                            \
        out_vec_size * M,                                             \
        batch_ndims,                                                  \
        batch_shape,                                                  \
        lhs_strides,                                                  \
        rhs_strides,                                                  \
        x_batch_ndims,                                                \
        x_shape,                                                      \
        x_strides,                                                    \
        w_batch_ndims,                                                \
        w_shape,                                                      \
        w_strides,                                                    \
        tid);                                                         \
    kq_##CODEC##_qmv_fast_impl<T, group_size, bits>(                  \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid); \
  }                                                                   \
                                                                      \
  template <typename T, int group_size, int bits>                     \
  [[kernel]] void kq_##CODEC##_gather_qmv(                            \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      const device uint32_t* lhs_indices,                             \
      const device uint32_t* rhs_indices,                             \
      device T* y,                                                    \
      const constant int& in_vec_size,                                \
      const constant int& out_vec_size,                               \
      const constant int& x_batch_ndims,                              \
      const constant int* x_shape,                                    \
      const constant int64_t* x_strides,                              \
      const constant int& w_batch_ndims,                              \
      const constant int* w_shape,                                    \
      const constant int64_t* w_strides,                              \
      const constant int64_t* /* s_strides */,                        \
      const constant int& batch_ndims,                                \
      const constant int* batch_shape,                                \
      const constant int64_t* lhs_strides,                            \
      const constant int64_t* rhs_strides,                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    int M = x_shape[x_batch_ndims];                                   \
    kq_adjust_matrix_offsets<T>(                                      \
        x,                                                            \
        w,                                                            \
        lhs_indices,                                                  \
        rhs_indices,                                                  \
        y,                                                            \
        out_vec_size * M,                                             \
        batch_ndims,                                                  \
        batch_shape,                                                  \
        lhs_strides,                                                  \
        rhs_strides,                                                  \
        x_batch_ndims,                                                \
        x_shape,                                                      \
        x_strides,                                                    \
        w_batch_ndims,                                                \
        w_shape,                                                      \
        w_strides,                                                    \
        tid);                                                         \
    kq_##CODEC##_qmv_impl<T, group_size, bits>(                       \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid); \
  }                                                                   \
                                                                      \
  template <typename T, int group_size, int bits, bool aligned_N>     \
  [[kernel]] void kq_##CODEC##_gather_qmm_t(                          \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      const device uint32_t* lhs_indices,                             \
      const device uint32_t* rhs_indices,                             \
      device T* y,                                                    \
      const constant int& K,                                          \
      const constant int& N,                                          \
      const constant int& M,                                          \
      const constant int& x_batch_ndims,                              \
      const constant int* x_shape,                                    \
      const constant int64_t* x_strides,                              \
      const constant int& w_batch_ndims,                              \
      const constant int* w_shape,                                    \
      const constant int64_t* w_strides,                              \
      const constant int64_t* /* s_strides */,                        \
      const constant int& batch_ndims,                                \
      const constant int* batch_shape,                                \
      const constant int64_t* lhs_strides,                            \
      const constant int64_t* rhs_strides,                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint lid [[thread_index_in_threadgroup]],                       \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    kq_adjust_matrix_offsets<T>(                                      \
        x,                                                            \
        w,                                                            \
        lhs_indices,                                                  \
        rhs_indices,                                                  \
        y,                                                            \
        M * N,                                                        \
        batch_ndims,                                                  \
        batch_shape,                                                  \
        lhs_strides,                                                  \
        rhs_strides,                                                  \
        x_batch_ndims,                                                \
        x_shape,                                                      \
        x_strides,                                                    \
        w_batch_ndims,                                                \
        w_shape,                                                      \
        w_strides,                                                    \
        tid);                                                         \
    constexpr int BM = 32, BK = 32, BN = 32;                          \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                  \
    threadgroup T Xs[BM * BK_padded];                                 \
    threadgroup T Ws[BN * BK_padded];                                 \
    using LoaderW = LOADER<                                           \
        T,                                                            \
        BN,                                                           \
        BK,                                                           \
        BK_padded,                                                    \
        /*reduction_dim=*/1,                                          \
        /*tgp_size=*/2 * 2 * SIMD_SIZE>;                              \
    kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(                 \
        w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);   \
  }                                                                   \
                                                                      \
  template <typename T, int group_size, int bits>                     \
  [[kernel]] void kq_##CODEC##_gather_qmm_n(                          \
      const device uint8_t* w,                                        \
      const device uint8_t* /* scales */,                             \
      const device T* x,                                              \
      const device uint32_t* lhs_indices,                             \
      const device uint32_t* rhs_indices,                             \
      device T* y,                                                    \
      const constant int& K,                                          \
      const constant int& N,                                          \
      const constant int& M,                                          \
      const constant int& x_batch_ndims,                              \
      const constant int* x_shape,                                    \
      const constant int64_t* x_strides,                              \
      const constant int& w_batch_ndims,                              \
      const constant int* w_shape,                                    \
      const constant int64_t* w_strides,                              \
      const constant int64_t* /* s_strides */,                        \
      const constant int& batch_ndims,                                \
      const constant int* batch_shape,                                \
      const constant int64_t* lhs_strides,                            \
      const constant int64_t* rhs_strides,                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint lid [[thread_index_in_threadgroup]],                       \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    kq_adjust_matrix_offsets<T>(                                      \
        x,                                                            \
        w,                                                            \
        lhs_indices,                                                  \
        rhs_indices,                                                  \
        y,                                                            \
        M * N,                                                        \
        batch_ndims,                                                  \
        batch_shape,                                                  \
        lhs_strides,                                                  \
        rhs_strides,                                                  \
        x_batch_ndims,                                                \
        x_shape,                                                      \
        x_strides,                                                    \
        w_batch_ndims,                                                \
        w_shape,                                                      \
        w_strides,                                                    \
        tid);                                                         \
    constexpr int BM = 32, BK = 32, BN = 32;                          \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                  \
    constexpr int BN_padded = (BN + 16 / sizeof(T));                  \
    threadgroup T Xs[BM * BK_padded];                                 \
    threadgroup T Ws[BK * BN_padded];                                 \
    using LoaderW = LOADER<                                           \
        T,                                                            \
        BK,                                                           \
        BN,                                                           \
        BN_padded,                                                    \
        /*reduction_dim=*/0,                                          \
        /*tgp_size=*/2 * 2 * SIMD_SIZE>;                              \
    kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(                            \
        w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);      \
  }

KQUANT_DEFINE_GATHER_KERNELS(q8_0, KqQ8_0BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q4_0, KqQ4_0BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q4_1, KqQ4_1BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q5_0, KqQ5_0BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q5_1, KqQ5_1BlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q4_k, KqQ4_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q5_k, KqQ5_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q6_k, KqQ6_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q3_k, KqQ3_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(q2_k, KqQ2_KBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq4_nl, KqIq4_nlBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq4_xs, KqIq4_xsBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq3_xxs, KqIq3_xxsBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq3_s, KqIq3_sBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq2_xxs, KqIq2_xxsBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq2_xs, KqIq2_xsBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq2_s, KqIq2_sBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq1_s, KqIq1_sBlockLoader)
KQUANT_DEFINE_GATHER_KERNELS(iq1_m, KqIq1_mBlockLoader)

#undef KQUANT_DEFINE_GATHER_KERNELS

// Sorted-rhs gather GEMM (ALU/steel path), the non-NAX counterpart of
// kq_gather_qmm_rhs_nax_tgp_impl. x rows arrive sorted by expert (the
// SwitchGLU prefill layout): each BM output tile walks its row segments of
// equal expert index and runs one steel-mma GEMM per segment against that
// expert's transposed weight matrix, storing only the segment's row slice.
// Without this kernel the sorted-prefill batch decomposes into per-row
// gather tiles that fill 1/BM of every simdgroup matmul. Transpose (nt)
// form only; K must be a multiple of BK (every kquant block size is).
template <
    typename T,
    typename LoaderW,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 32,
    const int BN = 64>
METAL_FUNC void kq_gather_qmm_rhs_impl(
    const device T* x,
    const device uint8_t* w,
    const device uint32_t* indices,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& M,
    const constant int& N,
    const constant int& K,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be >= SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be a multiple of SIMD_SIZE");

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      /*transpose_a=*/false,
      /*transpose_b=*/true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const size_t stride_w = size_t(N) * K_w;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  x += size_t(y_row) * K;
  y += size_t(y_row) * N + y_col;
  auto wl = w + size_t(y_col) * K_w;

  const short num_els = short(min(BM, M - y_row));
  const short num_outs = aligned_N ? short(BN) : short(min(BN, N - y_col));

  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  short n = 0;
  while (n < num_els) {
    n++;
    const short offset = offset_next;
    const uint32_t index = index_next;
    offset_next = num_els;
    for (; n < num_els; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    mma_t mma_op(simd_gid, simd_lid);
    loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
    LoaderW loader_w(wl + index * stride_w, K, Ws, simd_gid, simd_lid);

    if (num_els == BM) {
      if (aligned_N || num_outs == BN) {
        for (int k = 0; k < K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_x.load_unsafe();
          loader_w.load_unsafe();
          threadgroup_barrier(mem_flags::mem_threadgroup);
          mma_op.mma(Xs, Ws);
          loader_x.next();
          loader_w.next();
        }
      } else {
        for (int k = 0; k < K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_x.load_unsafe();
          loader_w.load_safe(short2(BK, num_outs));
          threadgroup_barrier(mem_flags::mem_threadgroup);
          mma_op.mma(Xs, Ws);
          loader_x.next();
          loader_w.next();
        }
      }
    } else {
      if (aligned_N || num_outs == BN) {
        for (int k = 0; k < K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_x.load_safe(short2(BK, num_els));
          loader_w.load_unsafe();
          threadgroup_barrier(mem_flags::mem_threadgroup);
          mma_op.mma(Xs, Ws);
          loader_x.next();
          loader_w.next();
        }
      } else {
        for (int k = 0; k < K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_x.load_safe(short2(BK, num_els));
          loader_w.load_safe(short2(BK, num_outs));
          threadgroup_barrier(mem_flags::mem_threadgroup);
          mma_op.mma(Xs, Ws);
          loader_x.next();
          loader_w.next();
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (offset == 0 && offset_next == BM && (aligned_N || num_outs == BN)) {
      mma_op.store_result(y, N);
    } else {
      mma_op.store_result_slice(
          y, N, short2(0, offset), short2(num_outs, offset_next));
    }
  }
}

// BM is a template parameter (64/32/16): sorted prefill batches with few
// rows per expert fragment each row tile into per-expert segments that each
// pay a full-tile mma K-loop, so utilization is ~1/segments-per-tile. The
// dispatch picks the largest BM not much above the batch's rows-per-expert.
#define KQ_DEFINE_GATHER_QMM_RHS(CODEC, LOADER)                       \
  template <typename T, int group_size, int bits, bool aligned_N,     \
            int BM>                                                   \
  [[kernel]] void kq_##CODEC##_gather_qmm_rhs(                        \
      const device T* x [[buffer(0)]],                                \
      const device uint8_t* w [[buffer(1)]],                          \
      const device uint8_t* scales [[buffer(2)]],                     \
      const device uint32_t* indices [[buffer(3)]],                   \
      device T* y [[buffer(4)]],                                      \
      const constant int& M [[buffer(5)]],                            \
      const constant int& N [[buffer(6)]],                            \
      const constant int& K [[buffer(7)]],                            \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    constexpr int BK = 32, BN = 64;                                   \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                  \
    using LoaderW = LOADER<                                           \
        T,                                                            \
        BN,                                                           \
        BK,                                                           \
        BK_padded,                                                    \
        /*reduction_dim=*/1,                                          \
        /*tgp_size=*/2 * 2 * SIMD_SIZE>;                              \
    static_assert(                                                    \
        group_size == LoaderW::weights_per_block,                     \
        #CODEC " gather_qmm_rhs requires group_size == block size");  \
    threadgroup T Xs[BM * BK_padded];                                 \
    threadgroup T Ws[BN * BK_padded];                                 \
    kq_gather_qmm_rhs_impl<T, LoaderW, aligned_N, BM, BK, BN>(        \
        x, w, indices, y, Xs, Ws, M, N, K, tid, simd_gid, simd_lid);  \
  }

KQ_DEFINE_GATHER_QMM_RHS(q8_0, KqQ8_0BlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q4_0, KqQ4_0BlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q4_1, KqQ4_1BlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q5_0, KqQ5_0BlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q5_1, KqQ5_1BlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q4_k, KqQ4_KBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q5_k, KqQ5_KBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q6_k, KqQ6_KBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q3_k, KqQ3_KBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(q2_k, KqQ2_KBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq4_nl, KqIq4_nlBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq4_xs, KqIq4_xsBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq3_xxs, KqIq3_xxsBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq3_s, KqIq3_sBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq2_xxs, KqIq2_xxsBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq2_xs, KqIq2_xsBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq2_s, KqIq2_sBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq1_s, KqIq1_sBlockLoader)
KQ_DEFINE_GATHER_QMM_RHS(iq1_m, KqIq1_mBlockLoader)

#undef KQ_DEFINE_GATHER_QMM_RHS
