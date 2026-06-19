// Single-pass NAX qmm built on MLX's steel-GEMM / NAX kernel framework
// (MIT) - see mlx_kquant/licenses/mlx-LICENSE.
#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/kq_quantized.h"

using namespace metal;
using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

template <
    typename T,
    typename LoaderW,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void kq_qmm_t_nax_tgp_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = w;

  x += y_row * static_cast<int64_t>(K);
  wl += static_cast<int64_t>(y_col) * K_w;
  y += y_row * static_cast<int64_t>(N) + y_col;

  LoaderW loader_w(wl, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  // Clamp to SM/SN in int before narrowing to short - a bare short() cast of
  // the raw remaining-row/col span overflows once M (or N) exceeds 32767,
  // corrupting the leading tiles of a single large matmul.
  const short sgp_sm = short(min(int(SM), M - (y_row + tm)));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : short(min(int(SN), N - (y_col + tn)));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    typename LoaderW,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void kq_qmm_n_nax_tgp_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;
  (void)M;

  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  auto wl = w;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += (y_col / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += y_row * static_cast<int64_t>(N) + y_col;

  LoaderW loader_w(
      wl, N, Ws, simd_gid, simd_lid, y_col % LoaderW::weights_per_block);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  const short ldb_tgp = BN_padded;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = false;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  // Dispatch gates NAX entry on K%BK==0.
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TK, TN> Btile;

      volatile int compiler_barrier;

      Atile.load(x + kk1, K);
      Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * ldb_tgp);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    x += BK;
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  Dtile.store(y + tm * N + tn, N);
}

// Q8_0: 34 bytes per 32-weight block. [fp16 d][int8 q[32]]

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ8_0BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q8_0_GROUP; // 32
  MLX_MTL_CONST int bytes_per_block = KQ_Q8_0_BLOCK_BYTES; // 34

  static_assert(
      BCOLS % weights_per_block == 0,
      "Q8_0 NAX loader requires BCOLS to be a multiple of 32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(
      n_reads <= weights_per_block,
      "Q8_0 NAX loader: n_reads must not exceed 32 (one block per thread).");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;

  KqNaxQ8_0BlockLoader(
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
                ? (BCOLS / weights_per_block) * bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const short block_idx = bj / weights_per_block;
    const short within = bj % weights_per_block;
    const device uint8_t* block_addr = src + block_idx * bytes_per_block;
    const float d = float(*(const device half*)(block_addr + KQ_Q8_0_D_OFFSET));
    const device int8_t* q =
        (const device int8_t*)(block_addr + KQ_Q8_0_Q_OFFSET + within);
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
    load_unsafe();
  }

  void next() {
    src += tile_stride;
  }
};

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    bool batched,
    int BM,
    int BN,
    int WM,
    int WN>
[[kernel]] void kq_q8_0_qmm_t_nax(
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
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 NAX kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 NAX kernel requires bits=8");

  constexpr int BK = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];

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

  using LoaderW = KqNaxQ8_0BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/WM * WN * SIMD_SIZE>;
  kq_qmm_t_nax_tgp_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(
      w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q8_0_qmm_n_nax(
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
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 NAX kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 NAX kernel requires bits=8");

  constexpr int BM = 64, BK = 64, BN = 64, WM = 2, WN = 2;
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Ws[BK * BN_padded];

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

  using LoaderW = KqNaxQ8_0BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/WM * WN * SIMD_SIZE>;
  kq_qmm_n_nax_tgp_impl<T, LoaderW, BM, BK, BN, WM, WN>(
      w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    int BM,
    int BN,
    int WM,
    int WN>
[[kernel]] void kq_q8_0_gather_qmm_t_nax(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
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
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 NAX kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 NAX kernel requires bits=8");

  constexpr int BK = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];

  kq_adjust_matrix_offsets<T>(
      x,
      w,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      tid);

  using LoaderW = KqNaxQ8_0BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/WM * WN * SIMD_SIZE>;
  kq_qmm_t_nax_tgp_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(
      w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q8_0_gather_qmm_n_nax(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
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
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_Q8_0_GROUP, "Q8_0 NAX kernel requires group_size=32");
  static_assert(bits == 8, "Q8_0 NAX kernel requires bits=8");

  constexpr int BM = 64, BK = 64, BN = 64, WM = 2, WN = 2;
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Ws[BK * BN_padded];

  kq_adjust_matrix_offsets<T>(
      x,
      w,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      tid);

  using LoaderW = KqNaxQ8_0BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/WM * WN * SIMD_SIZE>;
  kq_qmm_n_nax_tgp_impl<T, LoaderW, BM, BK, BN, WM, WN>(
      w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

// Q5_1: 24 bytes per 32-weight block. [fp16 d][fp16 m][qh[4]][qs[16]]

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ5_1BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q5_1_GROUP; // 32
  MLX_MTL_CONST int bytes_per_block = KQ_Q5_1_BLOCK_BYTES; // 24

  static_assert(
      BCOLS % weights_per_block == 0,
      "Q5_1 NAX loader requires BCOLS to be a multiple of 32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(
      n_reads <= weights_per_block,
      "Q5_1 NAX loader: n_reads must not exceed 32 (one block per thread).");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;

  KqNaxQ5_1BlockLoader(
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
                ? (BCOLS / weights_per_block) * bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const short block_idx = bj / weights_per_block;
    const short within = bj % weights_per_block;
    const device uint8_t* block_addr = src + block_idx * bytes_per_block;
    const float d = float(*(const device half*)(block_addr + KQ_Q5_1_D_OFFSET));
    const float m = float(*(const device half*)(block_addr + KQ_Q5_1_M_OFFSET));
    const uint32_t qh =
        *(const device uint32_t*)(block_addr + KQ_Q5_1_QH_OFFSET);
    const device uint8_t* qs = block_addr + KQ_Q5_1_QS_OFFSET;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int j = within + i;
      const uint32_t hi = ((qh >> j) << 4) & 0x10u;
      const uint8_t lo = (j < 16) ? (qs[j] & 0x0Fu) : (qs[j - 16] >> 4);
      const float q5 = float(uint32_t(lo) | hi);
      dst[i] = T(d * q5 + m);
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
    src += tile_stride;
  }
};

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    bool batched,
    int BM,
    int BN,
    int WM,
    int WN>
[[kernel]] void kq_q5_1_qmm_t_nax(
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
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 NAX kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 NAX kernel requires bits=5");

  constexpr int BK = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];

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

  using LoaderW = KqNaxQ5_1BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/WM * WN * SIMD_SIZE>;
  kq_qmm_t_nax_tgp_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(
      w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_q5_1_qmm_n_nax(
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
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 NAX kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 NAX kernel requires bits=5");

  constexpr int BM = 64, BK = 64, BN = 64, WM = 2, WN = 2;
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Ws[BK * BN_padded];

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

  using LoaderW = KqNaxQ5_1BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/WM * WN * SIMD_SIZE>;
  kq_qmm_n_nax_tgp_impl<T, LoaderW, BM, BK, BN, WM, WN>(
      w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    int BM,
    int BN,
    int WM,
    int WN>
[[kernel]] void kq_q5_1_gather_qmm_t_nax(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
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
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 NAX kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 NAX kernel requires bits=5");

  constexpr int BK = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];

  kq_adjust_matrix_offsets<T>(
      x,
      w,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      tid);

  using LoaderW = KqNaxQ5_1BlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      /*reduction_dim=*/1,
      /*tgp_size=*/WM * WN * SIMD_SIZE>;
  kq_qmm_t_nax_tgp_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(
      w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_q5_1_gather_qmm_n_nax(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
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
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_Q5_1_GROUP, "Q5_1 NAX kernel requires group_size=32");
  static_assert(bits == 5, "Q5_1 NAX kernel requires bits=5");

  constexpr int BM = 64, BK = 64, BN = 64, WM = 2, WN = 2;
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Ws[BK * BN_padded];

  kq_adjust_matrix_offsets<T>(
      x,
      w,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      tid);

  using LoaderW = KqNaxQ5_1BlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      /*reduction_dim=*/0,
      /*tgp_size=*/WM * WN * SIMD_SIZE>;
  kq_qmm_n_nax_tgp_impl<T, LoaderW, BM, BK, BN, WM, WN>(
      w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ4_0BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q4_0_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q4_0_BLOCK_BYTES;

  static_assert(
      BCOLS % weights_per_block == 0,
      "Q4_0 NAX loader requires BCOLS to be a multiple of 32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(
      n_reads <= weights_per_block,
      "Q4_0 NAX loader: n_reads must not exceed 32 (one block per thread).");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;

  KqNaxQ4_0BlockLoader(
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
                ? (BCOLS / weights_per_block) * bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const short block_idx = bj / weights_per_block;
    const short within = bj % weights_per_block;
    const device uint8_t* block_addr = src + block_idx * bytes_per_block;
    const float d = float(*(const device half*)(block_addr + KQ_Q4_0_D_OFFSET));
    const device uint8_t* qs = block_addr + KQ_Q4_0_QS_OFFSET;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int j = within + i;
      const int x = (j < 16) ? (qs[j] & 0x0Fu) : (qs[j - 16] >> 4);
      dst[i] = T(d * float(int(x) - 8));
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
    src += tile_stride;
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ4_1BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q4_1_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q4_1_BLOCK_BYTES;

  static_assert(
      BCOLS % weights_per_block == 0,
      "Q4_1 NAX loader requires BCOLS to be a multiple of 32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(
      n_reads <= weights_per_block,
      "Q4_1 NAX loader: n_reads must not exceed 32 (one block per thread).");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;

  KqNaxQ4_1BlockLoader(
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
                ? (BCOLS / weights_per_block) * bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const short block_idx = bj / weights_per_block;
    const short within = bj % weights_per_block;
    const device uint8_t* block_addr = src + block_idx * bytes_per_block;
    const float d = float(*(const device half*)(block_addr + KQ_Q4_1_D_OFFSET));
    const float m = float(*(const device half*)(block_addr + KQ_Q4_1_M_OFFSET));
    const device uint8_t* qs = block_addr + KQ_Q4_1_QS_OFFSET;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int j = within + i;
      const int x = (j < 16) ? (qs[j] & 0x0Fu) : (qs[j - 16] >> 4);
      dst[i] = T(d * float(x) + m);
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
    src += tile_stride;
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ5_0BlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q5_0_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_Q5_0_BLOCK_BYTES;

  static_assert(
      BCOLS % weights_per_block == 0,
      "Q5_0 NAX loader requires BCOLS to be a multiple of 32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(
      n_reads <= weights_per_block,
      "Q5_0 NAX loader: n_reads must not exceed 32 (one block per thread).");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;

  KqNaxQ5_0BlockLoader(
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
                ? (BCOLS / weights_per_block) * bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const short block_idx = bj / weights_per_block;
    const short within = bj % weights_per_block;
    const device uint8_t* block_addr = src + block_idx * bytes_per_block;
    const float d = float(*(const device half*)(block_addr + KQ_Q5_0_D_OFFSET));
    const device uint8_t* qh_p = block_addr + KQ_Q5_0_QH_OFFSET;
    const uint32_t qh = uint32_t(qh_p[0]) | (uint32_t(qh_p[1]) << 8) |
        (uint32_t(qh_p[2]) << 16) | (uint32_t(qh_p[3]) << 24);
    const device uint8_t* qs = block_addr + KQ_Q5_0_QS_OFFSET;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int j = within + i;
      const uint32_t hi = ((qh >> j) << 4) & 0x10u;
      const uint8_t lo = (j < 16) ? (qs[j] & 0x0Fu) : (qs[j - 16] >> 4);
      const float q5 = float(uint32_t(lo) | hi);
      dst[i] = T(d * (q5 - 16.0f));
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
    src += tile_stride;
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ4_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q4_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q4_K_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;

  static_assert(
      BCOLS == 64,
      "Q4_K NAX loader requires BCOLS == 64 (two sub-blocks per K-tile).");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  static_assert(
      n_reads == sub_block_size,
      "Q4_K NAX loader expects n_reads == 32 (half pair per thread).");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_sb_base;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;
  short sb_base;

  KqNaxQ4_KBlockLoader(
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
        fixed_sb_base(reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sb_base(0) {}

  void load_unsafe() const {
    const short pair_base = (reduction_dim == 0) ? fixed_sb_base : sb_base;

    const float d = float(*(const device half*)(src + KQ_Q4_K_D_OFFSET));
    const float dmin = float(*(const device half*)(src + KQ_Q4_K_DMIN_OFFSET));
    const device uint8_t* scales12 = src + KQ_Q4_K_SCALES_OFFSET;

    uint8_t sc6_lo, mn6_lo, sc6_hi, mn6_hi;
    kq_get_scale_min_k4(pair_base + 0, scales12, sc6_lo, mn6_lo);
    kq_get_scale_min_k4(pair_base + 1, scales12, sc6_hi, mn6_hi);
    const float eff_scale_lo = d * float(sc6_lo);
    const float eff_min_lo = dmin * float(mn6_lo);
    const float eff_scale_hi = d * float(sc6_hi);
    const float eff_min_hi = dmin * float(mn6_hi);

    const short pair = pair_base / 2;
    const device uint8_t* qs = src + KQ_Q4_K_QS_OFFSET + pair * 32 + bj_byte;

    static_assert(
        bytes_per_thread == 16,
        "Q4_K NAX vector load assumes bytes_per_thread == 16 (uint4).");
    const uint4 qs_v = *reinterpret_cast<const device uint4*>(qs);
    const thread uint8_t* qs_b = reinterpret_cast<const thread uint8_t*>(&qs_v);

#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t b = qs_b[i];
      dst[i] = T(eff_scale_lo * float(b & 0x0F) - eff_min_lo);
      dst[sub_block_size + i] = T(eff_scale_hi * float(b >> 4) - eff_min_hi);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        dst[i] = T(0);
        dst[sub_block_size + i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      sb_base += 2;
      if (sb_base == sub_blocks_per_block) {
        sb_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// Q5_K: 176 bytes per 256-weight super-block. Q4_K layout + qh[32] high bits.

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ5_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q5_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q5_K_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;

  static_assert(BCOLS == 64, "Q5_K NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  static_assert(n_reads == sub_block_size, "Q5_K NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_sb_base;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;
  short sb_base;
  // qh cached on sb_base==0; reduction_dim==0 reads per-call instead.
  struct Caches {
    uint8_t qh_cache[bytes_per_thread];
  };
  metal::conditional_t<reduction_dim == 1, Caches, kq_empty> cached;

  KqNaxQ5_KBlockLoader(
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
        fixed_sb_base(reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sb_base(0) {}

  void load_unsafe() {
    static_assert(
        bytes_per_thread == 16,
        "Q5_K NAX vector load assumes bytes_per_thread == 16 (uint4).");

    if constexpr (reduction_dim == 1) {
      const short pair_base = sb_base;

      const float d = float(*(const device half*)(src + KQ_Q5_K_D_OFFSET));
      const float dmin =
          float(*(const device half*)(src + KQ_Q5_K_DMIN_OFFSET));
      const device uint8_t* scales12 = src + KQ_Q5_K_SCALES_OFFSET;

      uint8_t sc6_lo, mn6_lo, sc6_hi, mn6_hi;
      kq_get_scale_min_k4(pair_base + 0, scales12, sc6_lo, mn6_lo);
      kq_get_scale_min_k4(pair_base + 1, scales12, sc6_hi, mn6_hi);
      const float eff_scale_lo = d * float(sc6_lo);
      const float eff_min_lo = dmin * float(mn6_lo);
      const float eff_scale_hi = d * float(sc6_hi);
      const float eff_min_hi = dmin * float(mn6_hi);

      const short pair = pair_base / 2;
      const device uint8_t* qs = src + KQ_Q5_K_QS_OFFSET + pair * 32 + bj_byte;
      const device uint8_t* qh = src + KQ_Q5_K_QH_OFFSET + bj_byte;

      const uint4 qs_v = *reinterpret_cast<const device uint4*>(qs);
      const thread uint8_t* qs_b =
          reinterpret_cast<const thread uint8_t*>(&qs_v);

      if (sb_base == 0) {
        const uint4 qh_v = *reinterpret_cast<const device uint4*>(qh);
        const thread uint8_t* qh_b =
            reinterpret_cast<const thread uint8_t*>(&qh_v);
#pragma unroll
        for (short i = 0; i < bytes_per_thread; i++) {
          cached.qh_cache[i] = qh_b[i];
        }
      }

#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        const uint8_t b = qs_b[i];
        const uint8_t h = cached.qh_cache[i];
        const uint8_t q4_lo = b & 0x0F;
        const uint8_t q4_hi = b >> 4;
        const uint8_t hi_lo = (h >> pair_base) & 1u;
        const uint8_t hi_hi = (h >> (pair_base + 1)) & 1u;
        const uint8_t q5_lo = q4_lo | (hi_lo << 4);
        const uint8_t q5_hi = q4_hi | (hi_hi << 4);
        dst[i] = T(eff_scale_lo * float(q5_lo) - eff_min_lo);
        dst[sub_block_size + i] = T(eff_scale_hi * float(q5_hi) - eff_min_hi);
      }
      return;
    }

    const short pair_base = fixed_sb_base;

    const float d = float(*(const device half*)(src + KQ_Q5_K_D_OFFSET));
    const float dmin = float(*(const device half*)(src + KQ_Q5_K_DMIN_OFFSET));
    const device uint8_t* scales12 = src + KQ_Q5_K_SCALES_OFFSET;

    uint8_t sc6_lo, mn6_lo, sc6_hi, mn6_hi;
    kq_get_scale_min_k4(pair_base + 0, scales12, sc6_lo, mn6_lo);
    kq_get_scale_min_k4(pair_base + 1, scales12, sc6_hi, mn6_hi);
    const float eff_scale_lo = d * float(sc6_lo);
    const float eff_min_lo = dmin * float(mn6_lo);
    const float eff_scale_hi = d * float(sc6_hi);
    const float eff_min_hi = dmin * float(mn6_hi);

    const short pair = pair_base / 2;
    const device uint8_t* qs = src + KQ_Q5_K_QS_OFFSET + pair * 32 + bj_byte;
    const device uint8_t* qh = src + KQ_Q5_K_QH_OFFSET + bj_byte;

    const uint4 qs_v = *reinterpret_cast<const device uint4*>(qs);
    const thread uint8_t* qs_b = reinterpret_cast<const thread uint8_t*>(&qs_v);

    const uint4 qh_v = *reinterpret_cast<const device uint4*>(qh);
    const thread uint8_t* qh_b = reinterpret_cast<const thread uint8_t*>(&qh_v);

#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t b = qs_b[i];
      const uint8_t h = qh_b[i];
      const uint8_t q4_lo = b & 0x0F;
      const uint8_t q4_hi = b >> 4;
      const uint8_t hi_lo = (h >> pair_base) & 1u;
      const uint8_t hi_hi = (h >> (pair_base + 1)) & 1u;
      const uint8_t q5_lo = q4_lo | (hi_lo << 4);
      const uint8_t q5_hi = q4_hi | (hi_hi << 4);
      dst[i] = T(eff_scale_lo * float(q5_lo) - eff_min_lo);
      dst[sub_block_size + i] = T(eff_scale_hi * float(q5_hi) - eff_min_hi);
    }
  }

  void load_safe(short2 src_tile_dim) {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        dst[i] = T(0);
        dst[sub_block_size + i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      sb_base += 2;
      if (sb_base == sub_blocks_per_block) {
        sb_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// Q6_K: 210 bytes per 256-weight super-block. Reversed field order:
//   [ql[128]][qh[64]][int8 scales[16]][fp16 d]

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ6_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q6_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q6_K_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "Q6_K NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "Q6_K NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;
  // Pair-cache: kt_base & 2 == 0 computes both pairs; reduction_dim==0 has no
  // cache.
  struct Caches {
    T cached[n_reads];
  };
  metal::conditional_t<reduction_dim == 1, Caches, kq_empty> cached;

  KqNaxQ6_KBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() {
    if constexpr (reduction_dim == 1) {
      if (kt_base & 2) {
#pragma unroll
        for (short i = 0; i < n_reads; i++) {
          dst[i] = cached.cached[i];
        }
        return;
      }

      const short base = kt_base;
      const short kt = base + (bj / k_tile_size);
      const short half_idx = kt / 4;
      const short quadrant = kt - half_idx * 4;

      const float d = float(*(const device half*)(src + KQ_Q6_K_D_OFFSET));
      const device int8_t* scales =
          (const device int8_t*)(src + KQ_Q6_K_SCALES_OFFSET);

      const device uint8_t* ql_base =
          src + KQ_Q6_K_QL_OFFSET + half_idx * 64 + (quadrant & 1) * 32;
      const device uint8_t* qh_base = src + KQ_Q6_K_QH_OFFSET + half_idx * 32;

      const float es_lo_a = d * float(scales[kt * 2 + 0]);
      const float es_hi_a = d * float(scales[kt * 2 + 1]);
      const float es_lo_b = d * float(scales[(kt + 2) * 2 + 0]);
      const float es_hi_b = d * float(scales[(kt + 2) * 2 + 1]);
      const short qh_shift_a = quadrant * 2;
      const short qh_shift_b = qh_shift_a + 4;

#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        const uint8_t ql_byte = ql_base[i];
        const uint8_t h = qh_base[i];
        const float es_a = (i >= 16) ? es_hi_a : es_lo_a;
        const float es_b = (i >= 16) ? es_hi_b : es_lo_b;
        const uint8_t low4_a = ql_byte & 0x0F;
        const uint8_t low4_b = ql_byte >> 4;
        const uint8_t high2_a = (uint8_t)((h >> qh_shift_a) & 0x03);
        const uint8_t high2_b = (uint8_t)((h >> qh_shift_b) & 0x03);
        const int8_t q6_a = (int8_t)(low4_a | (high2_a << 4)) - (int8_t)32;
        const int8_t q6_b = (int8_t)(low4_b | (high2_b << 4)) - (int8_t)32;
        dst[i] = T(es_a * float(q6_a));
        cached.cached[i] = T(es_b * float(q6_b));
      }
      return;
    }

    const short base = fixed_kt_base;
    const short kt = base + (bj / k_tile_size);
    const short half_idx = kt / 4;
    const short quadrant = kt - half_idx * 4;

    const float d = float(*(const device half*)(src + KQ_Q6_K_D_OFFSET));
    const device int8_t* scales =
        (const device int8_t*)(src + KQ_Q6_K_SCALES_OFFSET);

    const device uint8_t* ql_base =
        src + KQ_Q6_K_QL_OFFSET + half_idx * 64 + (quadrant & 1) * 32;
    const device uint8_t* qh_base = src + KQ_Q6_K_QH_OFFSET + half_idx * 32;

    const bool is_high_nibble = (quadrant >= 2);
    const short qh_shift = quadrant * 2;
    const float eff_scale_lo = d * float(scales[kt * 2 + 0]);
    const float eff_scale_hi = d * float(scales[kt * 2 + 1]);

#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const float eff_scale = (i >= 16) ? eff_scale_hi : eff_scale_lo;
      const uint8_t low4 =
          is_high_nibble ? (ql_base[i] >> 4) : (ql_base[i] & 0x0F);
      const uint8_t high2 = (uint8_t)((qh_base[i] >> qh_shift) & 0x03);
      const int8_t q6 = (int8_t)(low4 | (high2 << 4)) - (int8_t)32;
      dst[i] = T(eff_scale * float(q6));
    }
  }

  void load_safe(short2 src_tile_dim) {
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// Q3_K: 110 bytes per 256-weight super-block.
// [hmask[32]][qs[64]][scales[12]][fp16 d]

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ3_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q3_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q3_K_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "Q3_K NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  static_assert(n_reads == k_tile_size, "Q3_K NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;
  // Pair-cache + hmask cache; reduction_dim==0 has no register storage.
  struct Caches {
    T cached[n_reads];
    uint8_t hmask_cache[bytes_per_thread];
  };
  metal::conditional_t<reduction_dim == 1, Caches, kq_empty> cached;

  KqNaxQ3_KBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() {
    if constexpr (reduction_dim == 1) {
      if (kt_base & 2) {
#pragma unroll
        for (short i = 0; i < bytes_per_thread; i++) {
          dst[i] = cached.cached[i];
          dst[k_tile_size + i] = cached.cached[bytes_per_thread + i];
        }
        return;
      }

      const short base = kt_base;
      const short outer_half = base / 4;
      const short scale_off = (bj_byte >= 16) ? 1 : 0;

      const float d = float(*(const device half*)(src + KQ_Q3_K_D_OFFSET));
      const device uint8_t* qs =
          src + KQ_Q3_K_QS_OFFSET + outer_half * 32 + bj_byte;
      const device uint8_t* hm = src + KQ_Q3_K_HMASK_OFFSET + bj_byte;

      const float es_a = d *
          float((int)kq_q3_k_unpack_scale(
                    base * 2 + scale_off, src + KQ_Q3_K_SCALES_OFFSET) -
                32);
      const float es_b = d *
          float((int)kq_q3_k_unpack_scale(
                    (base + 1) * 2 + scale_off, src + KQ_Q3_K_SCALES_OFFSET) -
                32);
      const float es_c = d *
          float((int)kq_q3_k_unpack_scale(
                    (base + 2) * 2 + scale_off, src + KQ_Q3_K_SCALES_OFFSET) -
                32);
      const float es_d = d *
          float((int)kq_q3_k_unpack_scale(
                    (base + 3) * 2 + scale_off, src + KQ_Q3_K_SCALES_OFFSET) -
                32);

      const short shift_a = (base & 3) * 2;
      const short shift_b = ((base + 1) & 3) * 2;
      const short shift_c = ((base + 2) & 3) * 2;
      const short shift_d = ((base + 3) & 3) * 2;

      if (kt_base == 0) {
#pragma unroll
        for (short i = 0; i < bytes_per_thread; i++) {
          cached.hmask_cache[i] = hm[i];
        }
      }

#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        const uint8_t q = qs[i];
        const uint8_t h = cached.hmask_cache[i];
        const uint8_t q2_a = (q >> shift_a) & 0x03;
        const uint8_t q2_b = (q >> shift_b) & 0x03;
        const uint8_t q2_c = (q >> shift_c) & 0x03;
        const uint8_t q2_d = (q >> shift_d) & 0x03;
        const int q3_a = (int)q2_a - (((h >> base) & 1) ? 0 : 4);
        const int q3_b = (int)q2_b - (((h >> (base + 1)) & 1) ? 0 : 4);
        const int q3_c = (int)q2_c - (((h >> (base + 2)) & 1) ? 0 : 4);
        const int q3_d = (int)q2_d - (((h >> (base + 3)) & 1) ? 0 : 4);
        dst[i] = T(es_a * float(q3_a));
        dst[k_tile_size + i] = T(es_b * float(q3_b));
        cached.cached[i] = T(es_c * float(q3_c));
        cached.cached[bytes_per_thread + i] = T(es_d * float(q3_d));
      }
      return;
    }

    const short base = fixed_kt_base;
    const short outer_half = base / 4;
    const short scale_off = (bj_byte >= 16) ? 1 : 0;

    const float d = float(*(const device half*)(src + KQ_Q3_K_D_OFFSET));
    const device uint8_t* qs =
        src + KQ_Q3_K_QS_OFFSET + outer_half * 32 + bj_byte;
    const device uint8_t* hm = src + KQ_Q3_K_HMASK_OFFSET + bj_byte;

    const short kt = base;
    const float es = d *
        float((int)kq_q3_k_unpack_scale(
                  kt * 2 + scale_off, src + KQ_Q3_K_SCALES_OFFSET) -
              32);
    const short shift = (kt & 3) * 2;
    const short hbit = kt;

    const float es_b = d *
        float((int)kq_q3_k_unpack_scale(
                  (kt + 1) * 2 + scale_off, src + KQ_Q3_K_SCALES_OFFSET) -
              32);
    const short shift_b = ((kt + 1) & 3) * 2;
    const short hbit_b = kt + 1;

#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t q = qs[i];
      const uint8_t h = hm[i];
      const uint8_t q2_a = (q >> shift) & 0x03;
      const uint8_t q2_b = (q >> shift_b) & 0x03;
      const int q3_a = (int)q2_a - (((h >> hbit) & 1) ? 0 : 4);
      const int q3_b = (int)q2_b - (((h >> hbit_b) & 1) ? 0 : 4);
      dst[i] = T(es * float(q3_a));
      dst[k_tile_size + i] = T(es_b * float(q3_b));
    }
  }

  void load_safe(short2 src_tile_dim) {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        dst[i] = T(0);
        dst[k_tile_size + i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// Q2_K: 84 bytes per 256-weight super-block. [scales[16]][qs[64]][fp16 d][fp16
// dmin]

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxQ2_KBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_Q2_K_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_Q2_K_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "Q2_K NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  MLX_MTL_CONST short bytes_per_thread = n_reads / 2;
  static_assert(n_reads == k_tile_size, "Q2_K NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj_byte;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;
  struct Caches {
    T cached[n_reads];
  };
  metal::conditional_t<reduction_dim == 1, Caches, kq_empty> cached;

  KqNaxQ2_KBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj_byte((thread_idx % TCOLS) * bytes_per_thread),
        dst(dst_ + bi * dst_ld + bj_byte),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() {
    if constexpr (reduction_dim == 1) {
      if (kt_base & 2) {
#pragma unroll
        for (short i = 0; i < bytes_per_thread; i++) {
          dst[i] = cached.cached[i];
          dst[k_tile_size + i] = cached.cached[bytes_per_thread + i];
        }
        return;
      }

      const short base = kt_base;
      const short outer_half = base / 4;
      const short scale_off = (bj_byte >= 16) ? 1 : 0;

      const float d = float(*(const device half*)(src + KQ_Q2_K_D_OFFSET));
      const float dmin =
          float(*(const device half*)(src + KQ_Q2_K_DMIN_OFFSET));
      const device uint8_t* qs =
          src + KQ_Q2_K_QS_OFFSET + outer_half * 32 + bj_byte;

      static_assert(
          bytes_per_thread == 16,
          "Q2_K NAX vector load assumes bytes_per_thread == 16.");
      uint8_t qs_b[bytes_per_thread];
#pragma unroll
      for (short v = 0; v < bytes_per_thread / 4; v++) {
        const uint qs_v = *reinterpret_cast<const device uint*>(qs + v * 4);
        *reinterpret_cast<thread uint*>(&qs_b[v * 4]) = qs_v;
      }

      const uint8_t sc_a = src[KQ_Q2_K_SCALES_OFFSET + base * 2 + scale_off];
      const uint8_t sc_b =
          src[KQ_Q2_K_SCALES_OFFSET + (base + 1) * 2 + scale_off];
      const uint8_t sc_c =
          src[KQ_Q2_K_SCALES_OFFSET + (base + 2) * 2 + scale_off];
      const uint8_t sc_d =
          src[KQ_Q2_K_SCALES_OFFSET + (base + 3) * 2 + scale_off];
      const float es_a = d * float(sc_a & 0x0F);
      const float em_a = dmin * float(sc_a >> 4);
      const float es_b = d * float(sc_b & 0x0F);
      const float em_b = dmin * float(sc_b >> 4);
      const float es_c = d * float(sc_c & 0x0F);
      const float em_c = dmin * float(sc_c >> 4);
      const float es_d = d * float(sc_d & 0x0F);
      const float em_d = dmin * float(sc_d >> 4);

      const short shift_a = (base & 3) * 2;
      const short shift_b = ((base + 1) & 3) * 2;
      const short shift_c = ((base + 2) & 3) * 2;
      const short shift_d = ((base + 3) & 3) * 2;

#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        const uint8_t q = qs_b[i];
        const uint8_t q2_a = (q >> shift_a) & 0x03;
        const uint8_t q2_b = (q >> shift_b) & 0x03;
        const uint8_t q2_c = (q >> shift_c) & 0x03;
        const uint8_t q2_d = (q >> shift_d) & 0x03;
        dst[i] = T(es_a * float(q2_a) - em_a);
        dst[k_tile_size + i] = T(es_b * float(q2_b) - em_b);
        cached.cached[i] = T(es_c * float(q2_c) - em_c);
        cached.cached[bytes_per_thread + i] = T(es_d * float(q2_d) - em_d);
      }
      return;
    }

    const short base = fixed_kt_base;
    const short outer_half = base / 4;
    const short scale_off = (bj_byte >= 16) ? 1 : 0;

    const float d = float(*(const device half*)(src + KQ_Q2_K_D_OFFSET));
    const float dmin = float(*(const device half*)(src + KQ_Q2_K_DMIN_OFFSET));
    const device uint8_t* qs =
        src + KQ_Q2_K_QS_OFFSET + outer_half * 32 + bj_byte;

    static_assert(
        bytes_per_thread == 16,
        "Q2_K NAX vector load assumes bytes_per_thread == 16.");
    uint8_t qs_b[bytes_per_thread];
#pragma unroll
    for (short v = 0; v < bytes_per_thread / 4; v++) {
      const uint qs_v = *reinterpret_cast<const device uint*>(qs + v * 4);
      *reinterpret_cast<thread uint*>(&qs_b[v * 4]) = qs_v;
    }

    const short kt = base;
    const uint8_t sc_a = src[KQ_Q2_K_SCALES_OFFSET + kt * 2 + scale_off];
    const uint8_t sc_b = src[KQ_Q2_K_SCALES_OFFSET + (kt + 1) * 2 + scale_off];
    const float es_a = d * float(sc_a & 0x0F);
    const float em_a = dmin * float(sc_a >> 4);
    const float es_b = d * float(sc_b & 0x0F);
    const float em_b = dmin * float(sc_b >> 4);
    const short shift_a = (kt & 3) * 2;
    const short shift_b = ((kt + 1) & 3) * 2;

#pragma unroll
    for (short i = 0; i < bytes_per_thread; i++) {
      const uint8_t q = qs_b[i];
      const uint8_t q2_a = (q >> shift_a) & 0x03;
      const uint8_t q2_b = (q >> shift_b) & 0x03;
      dst[i] = T(es_a * float(q2_a) - em_a);
      dst[k_tile_size + i] = T(es_b * float(q2_b) - em_b);
    }
  }

  void load_safe(short2 src_tile_dim) {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < bytes_per_thread; i++) {
        dst[i] = T(0);
        dst[k_tile_size + i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
    if (reduction_dim == 1) {
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

#define KQ_NAX_DEFINE_KERNELS(codec, GROUP_CONST, bits_val, LOADER)       \
  template <                                                              \
      typename T,                                                         \
      int group_size,                                                     \
      int bits,                                                           \
      bool aligned_N,                                                     \
      bool batched,                                                       \
      int BM,                                                             \
      int BN,                                                             \
      int WM,                                                             \
      int WN>                                                             \
  [[kernel]] void kq_##codec##_qmm_t_nax(                                 \
      const device uint8_t* w,                                            \
      const device uint8_t* /* scales */,                                 \
      const device T* x,                                                  \
      device T* y,                                                        \
      const constant int& K,                                              \
      const constant int& N,                                              \
      const constant int& M,                                              \
      const constant int& x_batch_ndims,                                  \
      const constant int* x_shape,                                        \
      const constant int64_t* x_strides,                                  \
      const constant int& w_batch_ndims,                                  \
      const constant int* w_shape,                                        \
      const constant int64_t* w_strides,                                  \
      const constant int64_t* /* s_strides */,                            \
      uint3 tid [[threadgroup_position_in_grid]],                         \
      uint lid [[thread_index_in_threadgroup]],                           \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
      uint simd_lid [[thread_index_in_simdgroup]]) {                      \
    static_assert(                                                        \
        group_size == GROUP_CONST,                                        \
        #codec " NAX kernel requires group_size=" #GROUP_CONST);          \
    static_assert(                                                        \
        bits == bits_val, #codec " NAX kernel requires bits=" #bits_val); \
    constexpr int BK = 64;                                                \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                      \
    threadgroup T Ws[BN * BK_padded];                                     \
    if constexpr (batched) {                                              \
      kq_adjust_matrix_offsets<T>(                                        \
          x,                                                              \
          w,                                                              \
          y,                                                              \
          M * N,                                                          \
          x_batch_ndims,                                                  \
          x_shape,                                                        \
          x_strides,                                                      \
          w_batch_ndims,                                                  \
          w_shape,                                                        \
          w_strides,                                                      \
          tid);                                                           \
    }                                                                     \
    using LoaderW = LOADER<                                               \
        T,                                                                \
        BN,                                                               \
        BK,                                                               \
        BK_padded,                                                        \
        /*reduction_dim=*/1,                                              \
        /*tgp_size=*/WM * WN * SIMD_SIZE>;                                \
    kq_qmm_t_nax_tgp_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(     \
        w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);              \
  }                                                                       \
                                                                          \
  template <typename T, int group_size, int bits, bool batched>           \
  [[kernel]] void kq_##codec##_qmm_n_nax(                                 \
      const device uint8_t* w,                                            \
      const device uint8_t* /* scales */,                                 \
      const device T* x,                                                  \
      device T* y,                                                        \
      const constant int& K,                                              \
      const constant int& N,                                              \
      const constant int& M,                                              \
      const constant int& x_batch_ndims,                                  \
      const constant int* x_shape,                                        \
      const constant int64_t* x_strides,                                  \
      const constant int& w_batch_ndims,                                  \
      const constant int* w_shape,                                        \
      const constant int64_t* w_strides,                                  \
      const constant int64_t* /* s_strides */,                            \
      uint3 tid [[threadgroup_position_in_grid]],                         \
      uint lid [[thread_index_in_threadgroup]],                           \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
      uint simd_lid [[thread_index_in_simdgroup]]) {                      \
    static_assert(                                                        \
        group_size == GROUP_CONST,                                        \
        #codec " NAX kernel requires group_size=" #GROUP_CONST);          \
    static_assert(                                                        \
        bits == bits_val, #codec " NAX kernel requires bits=" #bits_val); \
    constexpr int BM = 64, BK = 64, BN = 64, WM = 2, WN = 2;              \
    constexpr int BN_padded = (BN + 16 / sizeof(T));                      \
    threadgroup T Ws[BK * BN_padded];                                     \
    if constexpr (batched) {                                              \
      kq_adjust_matrix_offsets<T>(                                        \
          x,                                                              \
          w,                                                              \
          y,                                                              \
          M * N,                                                          \
          x_batch_ndims,                                                  \
          x_shape,                                                        \
          x_strides,                                                      \
          w_batch_ndims,                                                  \
          w_shape,                                                        \
          w_strides,                                                      \
          tid);                                                           \
    }                                                                     \
    using LoaderW = LOADER<                                               \
        T,                                                                \
        BK,                                                               \
        BN,                                                               \
        BN_padded,                                                        \
        /*reduction_dim=*/0,                                              \
        /*tgp_size=*/WM * WN * SIMD_SIZE>;                                \
    kq_qmm_n_nax_tgp_impl<T, LoaderW, BM, BK, BN, WM, WN>(                \
        w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);              \
  }                                                                       \
                                                                          \
  template <                                                              \
      typename T,                                                         \
      int group_size,                                                     \
      int bits,                                                           \
      bool aligned_N,                                                     \
      int BM,                                                             \
      int BN,                                                             \
      int WM,                                                             \
      int WN>                                                             \
  [[kernel]] void kq_##codec##_gather_qmm_t_nax(                          \
      const device uint8_t* w,                                            \
      const device uint8_t* /* scales */,                                 \
      const device T* x,                                                  \
      const device uint32_t* lhs_indices,                                 \
      const device uint32_t* rhs_indices,                                 \
      device T* y,                                                        \
      const constant int& K,                                              \
      const constant int& N,                                              \
      const constant int& M,                                              \
      const constant int& x_batch_ndims,                                  \
      const constant int* x_shape,                                        \
      const constant int64_t* x_strides,                                  \
      const constant int& w_batch_ndims,                                  \
      const constant int* w_shape,                                        \
      const constant int64_t* w_strides,                                  \
      const constant int64_t* /* s_strides */,                            \
      const constant int& batch_ndims,                                    \
      const constant int* batch_shape,                                    \
      const constant int64_t* lhs_strides,                                \
      const constant int64_t* rhs_strides,                                \
      uint3 tid [[threadgroup_position_in_grid]],                         \
      uint lid [[thread_index_in_threadgroup]],                           \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
      uint simd_lid [[thread_index_in_simdgroup]]) {                      \
    static_assert(                                                        \
        group_size == GROUP_CONST,                                        \
        #codec " NAX kernel requires group_size=" #GROUP_CONST);          \
    static_assert(                                                        \
        bits == bits_val, #codec " NAX kernel requires bits=" #bits_val); \
    constexpr int BK = 64;                                                \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                      \
    threadgroup T Ws[BN * BK_padded];                                     \
    kq_adjust_matrix_offsets<T>(                                          \
        x,                                                                \
        w,                                                                \
        lhs_indices,                                                      \
        rhs_indices,                                                      \
        y,                                                                \
        M * N,                                                            \
        batch_ndims,                                                      \
        batch_shape,                                                      \
        lhs_strides,                                                      \
        rhs_strides,                                                      \
        x_batch_ndims,                                                    \
        x_shape,                                                          \
        x_strides,                                                        \
        w_batch_ndims,                                                    \
        w_shape,                                                          \
        w_strides,                                                        \
        tid);                                                             \
    using LoaderW = LOADER<                                               \
        T,                                                                \
        BN,                                                               \
        BK,                                                               \
        BK_padded,                                                        \
        /*reduction_dim=*/1,                                              \
        /*tgp_size=*/WM * WN * SIMD_SIZE>;                                \
    kq_qmm_t_nax_tgp_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(     \
        w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);              \
  }                                                                       \
                                                                          \
  template <typename T, int group_size, int bits>                         \
  [[kernel]] void kq_##codec##_gather_qmm_n_nax(                          \
      const device uint8_t* w,                                            \
      const device uint8_t* /* scales */,                                 \
      const device T* x,                                                  \
      const device uint32_t* lhs_indices,                                 \
      const device uint32_t* rhs_indices,                                 \
      device T* y,                                                        \
      const constant int& K,                                              \
      const constant int& N,                                              \
      const constant int& M,                                              \
      const constant int& x_batch_ndims,                                  \
      const constant int* x_shape,                                        \
      const constant int64_t* x_strides,                                  \
      const constant int& w_batch_ndims,                                  \
      const constant int* w_shape,                                        \
      const constant int64_t* w_strides,                                  \
      const constant int64_t* /* s_strides */,                            \
      const constant int& batch_ndims,                                    \
      const constant int* batch_shape,                                    \
      const constant int64_t* lhs_strides,                                \
      const constant int64_t* rhs_strides,                                \
      uint3 tid [[threadgroup_position_in_grid]],                         \
      uint lid [[thread_index_in_threadgroup]],                           \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
      uint simd_lid [[thread_index_in_simdgroup]]) {                      \
    static_assert(                                                        \
        group_size == GROUP_CONST,                                        \
        #codec " NAX kernel requires group_size=" #GROUP_CONST);          \
    static_assert(                                                        \
        bits == bits_val, #codec " NAX kernel requires bits=" #bits_val); \
    constexpr int BM = 64, BK = 64, BN = 64, WM = 2, WN = 2;              \
    constexpr int BN_padded = (BN + 16 / sizeof(T));                      \
    threadgroup T Ws[BK * BN_padded];                                     \
    kq_adjust_matrix_offsets<T>(                                          \
        x,                                                                \
        w,                                                                \
        lhs_indices,                                                      \
        rhs_indices,                                                      \
        y,                                                                \
        M * N,                                                            \
        batch_ndims,                                                      \
        batch_shape,                                                      \
        lhs_strides,                                                      \
        rhs_strides,                                                      \
        x_batch_ndims,                                                    \
        x_shape,                                                          \
        x_strides,                                                        \
        w_batch_ndims,                                                    \
        w_shape,                                                          \
        w_strides,                                                        \
        tid);                                                             \
    using LoaderW = LOADER<                                               \
        T,                                                                \
        BK,                                                               \
        BN,                                                               \
        BN_padded,                                                        \
        /*reduction_dim=*/0,                                              \
        /*tgp_size=*/WM * WN * SIMD_SIZE>;                                \
    kq_qmm_n_nax_tgp_impl<T, LoaderW, BM, BK, BN, WM, WN>(                \
        w, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);              \
  }

// ---- IQ-codec NAX block loaders (load-only; ALU dequant re-tiled to BK=64)
// ---- iq4_nl (32-weight group): structurally KqNaxQ4_0BlockLoader with the
// non-linear kvalues_iq4nl codebook replacing the linear (x-8) dequant. Same
// 18-byte block (fp16 d @0, packed nibbles @2) and interleaved-nibble layout
// (weight j<16 -> qs[j]&0xF, weight 16+i -> qs[i]>>4).
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq4_nlBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ4_NL_GROUP;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ4_NL_BLOCK_BYTES;

  static_assert(
      BCOLS % weights_per_block == 0,
      "IQ4_NL NAX loader requires BCOLS to be a multiple of 32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(
      n_reads <= weights_per_block,
      "IQ4_NL NAX loader: n_reads must not exceed 32 (one block per thread).");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;

  KqNaxIq4_nlBlockLoader(
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
                ? (BCOLS / weights_per_block) * bytes_per_block
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)) {}

  void load_unsafe() const {
    const short block_idx = bj / weights_per_block;
    const short within = bj % weights_per_block;
    const device uint8_t* block_addr = src + block_idx * bytes_per_block;
    const float d = float(*(const device half*)block_addr);
    const device uint8_t* qs = block_addr + KQ_IQ4_NL_QS_OFFSET;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int j = within + i;
      const int x = (j < 16) ? (qs[j] & 0x0Fu) : (qs[j - 16] >> 4);
      dst[i] = T(d * float(kvalues_iq4nl[x]));
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
    src += tile_stride;
  }
};

// Superblock IQ NAX loaders. All eight model on KqNaxQ6_KBlockLoader: a 64-wide
// K-tile spans two 32-weight sub-blocks (BCOLS==64, n_reads==32, bj in {0,32}),
// the thread owns one whole sub-block selected by sb = base + bj/32, and next()
// strides two sub-blocks. The Q6_K pair-cache is dropped (correctness first;
// the redundant superblock-header read is an N5 profiling concern). The
// per-weight dequant body is a verbatim transcription of the validated ALU
// KqIq*BlockLoader (kq_quantized.h) with the within-sub-block index p == i.
// Grid/sign tables come in via kq_quantized.h.

// iq4_xs: superblock nibble decode (kvalues_iq4nl), 6-bit per-sub-block scale.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq4_xsBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ4_XS_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ4_XS_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "iq4_xs NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "iq4_xs NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;

  KqNaxIq4_xsBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() const {
    const short sb =
        (reduction_dim == 1 ? kt_base : fixed_kt_base) + bj / k_tile_size;
    const float d = float(*(const device half*)src);
    const uint16_t scales_h = uint16_t(src[2]) | (uint16_t(src[3]) << 8);
    const device uint8_t* scales_l = src + KQ_IQ4_XS_SCALESL_OFFSET;
    const int ls = ((scales_l[sb / 2] >> (4 * (sb & 1))) & 0xf) |
        (((scales_h >> (2 * sb)) & 3) << 4);
    const float dl = d * float(ls - 32);
    const device uint8_t* qs = src + KQ_IQ4_XS_QS_OFFSET + sb * 16;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = i;
      const bool is_high = p >= 16;
      const int b = is_high ? (p - 16) : p;
      const int nib = is_high ? (qs[b] >> 4) : (qs[b] & 0xf);
      dst[i] = T(dl * float(kvalues_iq4nl[nib]));
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// iq3_xxs: 3-bit grid (iq3xxs_grid) + ksigns_iq2xs signs, gas-packed scale.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq3_xxsBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ3_XXS_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ3_XXS_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "iq3_xxs NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "iq3_xxs NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;

  KqNaxIq3_xxsBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() const {
    const short sb =
        (reduction_dim == 1 ? kt_base : fixed_kt_base) + bj / k_tile_size;
    const float d = float(*(const device half*)src);
    const device uint8_t* qs = src + KQ_IQ3_XXS_QS_OFFSET + sb * 8;
    const device uint8_t* gas = src + KQ_IQ3_XXS_GAS_OFFSET + sb * 4;
    const uint aux32 = uint(gas[0]) | (uint(gas[1]) << 8) |
        (uint(gas[2]) << 16) | (uint(gas[3]) << 24);
    const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = i;
      const int l = p / 8;
      const int jpos = p % 8;
      const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
      const uint g = iq3xxs_grid[jpos < 4 ? qs[2 * l] : qs[2 * l + 1]];
      const int jj = jpos < 4 ? jpos : jpos - 4;
      const float gb = float((g >> (8 * jj)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[jpos]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// iq3_s: 3-bit grid (iq3s_grid) with qh high bit + per-byte signs.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq3_sBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ3_S_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ3_S_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "iq3_s NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "iq3_s NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;

  KqNaxIq3_sBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() const {
    const short sb =
        (reduction_dim == 1 ? kt_base : fixed_kt_base) + bj / k_tile_size;
    const float d = float(*(const device half*)src);
    const device uint8_t* scales = src + KQ_IQ3_S_SCALES_OFFSET;
    const float db =
        d * float(1 + 2 * ((scales[sb / 2] >> (4 * (sb & 1))) & 0xf));
    const uint qh = src[KQ_IQ3_S_QH_OFFSET + sb];
    const device uint8_t* qs = src + KQ_IQ3_S_QS_OFFSET + sb * 8;
    const device uint8_t* sg = src + KQ_IQ3_S_SIGNS_OFFSET + sb * 4;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = i;
      const int l = p / 8;
      const int jpos = p % 8;
      const uint8_t signs = sg[l];
      const uint idx = (jpos < 4)
          ? (qs[2 * l] | ((qh << (8 - 2 * l)) & 256))
          : (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256));
      const uint g = iq3s_grid[idx];
      const int jj = jpos < 4 ? jpos : jpos - 4;
      const float gb = float((g >> (8 * jj)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[jpos]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// iq2_xxs: 2-bit grid (iq2xxs_grid uint64) + ksigns, signbits-packed scale.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq2_xxsBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ2_XXS_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ2_XXS_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "iq2_xxs NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "iq2_xxs NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;

  KqNaxIq2_xxsBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() const {
    const short sb =
        (reduction_dim == 1 ? kt_base : fixed_kt_base) + bj / k_tile_size;
    const float d = float(*(const device half*)src);
    const device uint8_t* qs = src + KQ_IQ2_XXS_QS_OFFSET + sb * 8;
    const uint signbits = uint(qs[4]) | (uint(qs[5]) << 8) |
        (uint(qs[6]) << 16) | (uint(qs[7]) << 24);
    const float db = d * (0.5f + float(signbits >> 28)) * 0.25f;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = i;
      const int l = p / 8;
      const int j = p % 8;
      const uint8_t signs = ksigns_iq2xs[(signbits >> (7 * l)) & 127];
      const uint64_t g = iq2xxs_grid[qs[l]];
      const float gb = float((g >> (8 * j)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// iq2_xs: 2-bit grid (iq2xs_grid) with 9-bit packed index + ksigns, nibble sc.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq2_xsBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ2_XS_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ2_XS_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "iq2_xs NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "iq2_xs NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;

  KqNaxIq2_xsBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() const {
    const short sb =
        (reduction_dim == 1 ? kt_base : fixed_kt_base) + bj / k_tile_size;
    const float d = float(*(const device half*)src);
    const device uint8_t* qs = src + KQ_IQ2_XS_QS_OFFSET + sb * 8;
    const uint8_t sc = src[KQ_IQ2_XS_SCALES_OFFSET + sb];
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = i;
      const int l = p / 8;
      const int j = p % 8;
      const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
      const float db = d * (0.5f + float(sc_nib)) * 0.25f;
      const device uint8_t* qp = qs + l * 2;
      const uint q = uint(qp[0]) | (uint(qp[1]) << 8);
      const uint8_t signs = ksigns_iq2xs[q >> 9];
      const uint64_t g = iq2xs_grid[q & 511];
      const float gb = float((g >> (8 * j)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// iq2_s: 2-bit grid (iq2s_grid) with qh high bits + per-byte signs, nibble sc.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq2_sBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ2_S_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ2_S_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "iq2_s NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "iq2_s NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;

  KqNaxIq2_sBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() const {
    const short sb =
        (reduction_dim == 1 ? kt_base : fixed_kt_base) + bj / k_tile_size;
    const float d = float(*(const device half*)src);
    const device uint8_t* qs = src + KQ_IQ2_S_QS_OFFSET + sb * 4;
    const device uint8_t* sg = src + KQ_IQ2_S_SIGNS_OFFSET + sb * 4;
    const uint qh = src[KQ_IQ2_S_QH_OFFSET + sb];
    const uint8_t sc = src[KQ_IQ2_S_SCALES_OFFSET + sb];
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = i;
      const int l = p / 8;
      const int j = p % 8;
      const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
      const float db = d * (0.5f + float(sc_nib)) * 0.25f;
      const uint idx = qs[l] | ((qh << (8 - 2 * l)) & 0x300);
      const uint8_t signs = sg[l];
      const uint64_t g = iq2s_grid[idx];
      const float gb = float((g >> (8 * j)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// iq1_s: 1-bit grid (iq1s_grid) with qh-derived scale + per-block delta sign.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq1_sBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ1_S_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ1_S_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "iq1_s NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "iq1_s NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;

  KqNaxIq1_sBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() const {
    const short sb =
        (reduction_dim == 1 ? kt_base : fixed_kt_base) + bj / k_tile_size;
    const float d = float(*(const device half*)src);
    const device uint8_t* qhp = src + KQ_IQ1_S_QH_OFFSET + sb * 2;
    const uint qh = uint(qhp[0]) | (uint(qhp[1]) << 8);
    const device uint8_t* qs = src + KQ_IQ1_S_QS_OFFSET + sb * 4;
    const float dl = d * float(2 * int((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = i;
      const int l = p / 8;
      const int j = p % 8;
      const uint idx = uint(qs[l]) | (((qh >> (3 * l)) & 7) << 8);
      const int8_t gv =
          as_type<int8_t>(uint8_t((iq1s_grid[idx] >> (8 * j)) & 0xff));
      dst[i] = T(dl * (float(gv) + delta));
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

// iq1_m: 1-bit grid (iq1s_grid) with reconstructed d, per-half scale + delta.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqNaxIq1_mBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ1_M_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ1_M_BLOCK_BYTES;
  MLX_MTL_CONST int k_tile_size = 32;
  MLX_MTL_CONST int k_tiles_per_block = weights_per_block / k_tile_size;

  static_assert(BCOLS == 64, "iq1_m NAX loader requires BCOLS == 64.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");

  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == k_tile_size, "iq1_m NAX expects n_reads == 32.");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_kt_base;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  short kt_base;

  KqNaxIq1_mBlockLoader(
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
        fixed_kt_base(reduction_dim == 0 ? (col_in_block / k_tile_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        kt_base(0) {}

  void load_unsafe() const {
    const short sb =
        (reduction_dim == 1 ? kt_base : fixed_kt_base) + bj / k_tile_size;
    const device uint8_t* scp = src + KQ_IQ1_M_SCALES_OFFSET;
    const ushort sc0 = ushort(scp[0]) | (ushort(scp[1]) << 8);
    const ushort sc1 = ushort(scp[2]) | (ushort(scp[3]) << 8);
    const ushort sc2 = ushort(scp[4]) | (ushort(scp[5]) << 8);
    const ushort sc3 = ushort(scp[6]) | (ushort(scp[7]) << 8);
    const ushort scale_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
        ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    const float d = float(as_type<half>(scale_u16));
    const device uint8_t* swp = scp + (sb / 2) * 2;
    const uint sc_word = uint(swp[0]) | (uint(swp[1]) << 8);
    const device uint8_t* qhp = src + KQ_IQ1_M_QH_OFFSET + sb * 2;
    const device uint8_t* qs = src + KQ_IQ1_M_QS_OFFSET + sb * 4;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = i;
      const int l = p / 8;
      const int j = p % 8;
      const int shift = 6 * (sb & 1) + ((l < 2) ? 0 : 3);
      const float dl = d * float(2 * int((sc_word >> shift) & 7) + 1);
      const uint8_t qh = qhp[l / 2];
      const int hshift = (l & 1) ? 4 : 8;
      const uint idx = uint(qs[l]) | ((uint(qh) << hshift) & 0x700);
      const float delta = (qh & ((l & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
      const int8_t gv =
          as_type<int8_t>(uint8_t((iq1s_grid[idx] >> (8 * j)) & 0xff));
      dst[i] = T(dl * (float(gv) + delta));
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
      kt_base += 2;
      if (kt_base == k_tiles_per_block) {
        kt_base = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

KQ_NAX_DEFINE_KERNELS(iq4_nl, 32, 4, KqNaxIq4_nlBlockLoader)
KQ_NAX_DEFINE_KERNELS(iq4_xs, 256, 4, KqNaxIq4_xsBlockLoader)
KQ_NAX_DEFINE_KERNELS(iq3_xxs, 256, 3, KqNaxIq3_xxsBlockLoader)
KQ_NAX_DEFINE_KERNELS(iq3_s, 256, 3, KqNaxIq3_sBlockLoader)
KQ_NAX_DEFINE_KERNELS(iq2_xxs, 256, 2, KqNaxIq2_xxsBlockLoader)
KQ_NAX_DEFINE_KERNELS(iq2_xs, 256, 2, KqNaxIq2_xsBlockLoader)
KQ_NAX_DEFINE_KERNELS(iq2_s, 256, 2, KqNaxIq2_sBlockLoader)
KQ_NAX_DEFINE_KERNELS(iq1_s, 256, 1, KqNaxIq1_sBlockLoader)
KQ_NAX_DEFINE_KERNELS(iq1_m, 256, 1, KqNaxIq1_mBlockLoader)
KQ_NAX_DEFINE_KERNELS(q4_0, 32, 4, KqNaxQ4_0BlockLoader)
KQ_NAX_DEFINE_KERNELS(q4_1, 32, 4, KqNaxQ4_1BlockLoader)
KQ_NAX_DEFINE_KERNELS(q5_0, 32, 5, KqNaxQ5_0BlockLoader)
KQ_NAX_DEFINE_KERNELS(q4_k, 256, 4, KqNaxQ4_KBlockLoader)
KQ_NAX_DEFINE_KERNELS(q5_k, 256, 5, KqNaxQ5_KBlockLoader)
KQ_NAX_DEFINE_KERNELS(q6_k, 256, 6, KqNaxQ6_KBlockLoader)
KQ_NAX_DEFINE_KERNELS(q3_k, 256, 3, KqNaxQ3_KBlockLoader)
KQ_NAX_DEFINE_KERNELS(q2_k, 256, 2, KqNaxQ2_KBlockLoader)

template <
    typename T,
    typename LoaderW,
    bool transpose,
    int BM = 64,
    int BN = 64,
    int BK = 64,
    int WM = 2,
    int WN = 2>
METAL_FUNC void kq_gather_qmm_rhs_nax_tgp_impl(
    const device T* x,
    const device uint8_t* w,
    const device uint32_t* indices,
    device T* y,
    const constant int& M,
    const constant int& N,
    const constant int& K,
    threadgroup T* Ws,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int N_w = (N / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? size_t(N) * K_w : size_t(K) * N_w;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  auto wl = w;
  x += y_row_long * static_cast<size_t>(K);
  y += y_row_long * static_cast<size_t>(N) + y_col_long;
  if (transpose) {
    wl += y_col_long * K_w;
  } else {
    wl += (y_col_long / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  }

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  // Clamp to SM/SN (small) in int BEFORE narrowing to short. Casting the raw
  // (M - (y_row + tm)) term first overflows short once M > 32767, poisoning the
  // tail-tile row count for every tile whose remaining-row span exceeds
  // SHRT_MAX and silently corrupting their output (mirrors the min-then-cast
  // order used for tgp_bm/tgp_bn above).
  const short sgp_sm =
      align_M ? SM : short(min(int(SM), max(0, (M - (y_row + tm)))));
  const short sgp_sn =
      align_N ? SN : short(min(int(SN), max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

    thread LoaderW loader_w(
        wl + index * stride_w,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id,
        transpose ? 0 : int(y_col_long % LoaderW::weights_per_block));

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            // Prevents the Metal compiler from reordering loads across
            // iterations.
            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(xn + kk1, K);
            } else {
              Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            }

            if constexpr (transpose) {
              Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }

          xn += BK;
          loader_w.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            // Prevents the Metal compiler from reordering loads across
            // iterations.
            volatile int compiler_barrier;

            const short psk = min(int(SK), max(0, (BK - kk1)));
            Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

            if constexpr (transpose) {
              Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
        const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));

        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }
}

#define KQ_NAX_DEFINE_GATHER_RHS(codec, GROUP_CONST, bits_val, LOADER)         \
  template <                                                                   \
      typename T,                                                              \
      int group_size,                                                          \
      int bits,                                                                \
      int BM,                                                                  \
      int BN,                                                                  \
      int BK,                                                                  \
      int WM,                                                                  \
      int WN,                                                                  \
      bool transpose>                                                          \
  [[kernel]] void kq_##codec##_gather_qmm_rhs_nax(                             \
      const device T* x [[buffer(0)]],                                         \
      const device uint8_t* w [[buffer(1)]],                                   \
      const device uint8_t* scales [[buffer(2)]],                              \
      const device uint32_t* indices [[buffer(3)]],                            \
      device T* y [[buffer(4)]],                                               \
      const constant int& M [[buffer(5)]],                                     \
      const constant int& N [[buffer(6)]],                                     \
      const constant int& K [[buffer(7)]],                                     \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                   \
      uint simd_lane_id [[thread_index_in_simdgroup]]) {                       \
    static_assert(                                                             \
        group_size == GROUP_CONST,                                             \
        #codec " NAX kernel requires group_size=" #GROUP_CONST);               \
    static_assert(                                                             \
        bits == bits_val, #codec " NAX kernel requires bits=" #bits_val);      \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                           \
    constexpr int BN_padded = (BN + 16 / sizeof(T));                           \
    threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];             \
    using LoaderW = LOADER<                                                    \
        T,                                                                     \
        transpose ? BN : BK,                                                   \
        transpose ? BK : BN,                                                   \
        transpose ? BK_padded : BN_padded,                                     \
        /*reduction_dim=*/(transpose ? 1 : 0),                                 \
        /*tgp_size=*/WM * WN * SIMD_SIZE>;                                     \
    kq_gather_qmm_rhs_nax_tgp_impl<T, LoaderW, transpose, BM, BN, BK, WM, WN>( \
        x, w, indices, y, M, N, K, Ws, tid, simd_group_id, simd_lane_id);      \
  }

KQ_NAX_DEFINE_GATHER_RHS(q8_0, 32, 8, KqNaxQ8_0BlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q5_1, 32, 5, KqNaxQ5_1BlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq4_nl, 32, 4, KqNaxIq4_nlBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq4_xs, 256, 4, KqNaxIq4_xsBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq3_xxs, 256, 3, KqNaxIq3_xxsBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq3_s, 256, 3, KqNaxIq3_sBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq2_xxs, 256, 2, KqNaxIq2_xxsBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq2_xs, 256, 2, KqNaxIq2_xsBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq2_s, 256, 2, KqNaxIq2_sBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq1_s, 256, 1, KqNaxIq1_sBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(iq1_m, 256, 1, KqNaxIq1_mBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q4_0, 32, 4, KqNaxQ4_0BlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q4_1, 32, 4, KqNaxQ4_1BlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q5_0, 32, 5, KqNaxQ5_0BlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q4_k, 256, 4, KqNaxQ4_KBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q5_k, 256, 5, KqNaxQ5_KBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q6_k, 256, 6, KqNaxQ6_KBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q3_k, 256, 3, KqNaxQ3_KBlockLoader)
KQ_NAX_DEFINE_GATHER_RHS(q2_k, 256, 2, KqNaxQ2_KBlockLoader)

#undef KQ_NAX_DEFINE_KERNELS
#undef KQ_NAX_DEFINE_GATHER_RHS
