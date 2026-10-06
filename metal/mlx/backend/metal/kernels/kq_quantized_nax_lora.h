// NAX qmm_t kernel with a low-rank term in the same reduction, built on the
// tile and loaders of kq_quantized_nax.h. Separate entry points, so the
// plain qmm_t kernels keep their code.
//
//   y = x @ W^T + z @ B
//
// x [M, K], W a quantized weight [N, K], z [M, R] and B [R, N] in the
// dtype of x. A threadgroup owns one [BM, BN] tile of y. It walks the rank
// of its column band in steps of 32, with the A tile loading z and the B
// tile loading B straight from their buffers, and then walks K as the
// plain kernel does. Both walks add into one f32 accumulator, which is
// rounded once at the store, so y is written once and never read.
#include "mlx/backend/metal/kernels/kq_quantized_nax.h"

// The walk of a band's rank, 32 ranks a step, into the accumulator of a
// simdgroup. sg_cols is how many of the simdgroup's columns lie in the
// band, and sg_rows how many of its rows lie in M. A simdgroup with no
// column in the band runs no step, so no value of z reaches its columns.
// Past the band's rank the A tile and the B tile load zero, so no z column
// and no B row of another band enters the sum, whatever it holds. Past the
// band's end the B tile loads zero. The host runs this kernel only for
// bands that end on a simdgroup's column tile or at N, so a simdgroup holds
// no stored column outside its band.
template <
    typename T,
    const bool aligned_M,
    const short TM,
    const short TN,
    const short TK>
METAL_FUNC void kq_qmm_t_nax_lora_tail(
    thread NAXTile<float, TM, TN>& Dtile,
    const device T* z,
    const device T* b,
    const int R,
    const int N,
    const int band_rank,
    const short sg_cols,
    const short sg_rows) {
  constexpr short SK = TK * 16;
  constexpr short SN = TN * 16;
  STEEL_PRAGMA_NO_UNROLL
  for (int r0 = 0; sg_cols > 0 && r0 < band_rank; r0 += SK) {
    NAXTile<T, TM, TK> Atile;
    NAXTile<T, TK, TN> Btile;

    volatile int compiler_barrier;

    const short zk = short(min(int(SK), band_rank - r0));
    if constexpr (aligned_M) {
      if (zk == SK) {
        Atile.load(z + r0, R);
      } else {
        Atile.load_safe(z + r0, R, short2(zk, sg_rows));
      }
    } else {
      Atile.load_safe(z + r0, R, short2(zk, sg_rows));
    }

    const device T* br = b + r0 * static_cast<int64_t>(N);
    if (zk == SK && sg_cols == SN) {
      Btile.load(br, N);
    } else {
      Btile.load_safe(br, N, short2(sg_cols, zk));
    }

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<false>{});

    (void)compiler_barrier;
  }
}

// band_z, band_rank and band_cols describe the column band of the
// threadgroup's tile: the first z column, the rank, and how many of the
// tile's columns lie in the band. A tile with no column in a band has
// band_cols at or below 0 and runs the plain walk alone.
template <
    typename T,
    typename LoaderW,
    const bool aligned_N,
    const int BM,
    const int BK,
    const int BN,
    const int WM,
    const int WN>
METAL_FUNC void kq_qmm_t_nax_lora_impl(
    const device uint8_t* w,
    const device T* x,
    const device T* z,
    const device T* b,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& R,
    const int band_z,
    const int band_rank,
    const int band_cols,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK % 32 == 0, "BK must be a multiple of the MMA K step");

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  x += y_row * static_cast<int64_t>(K);
  w += static_cast<int64_t>(y_col) * K_w;
  y += y_row * static_cast<int64_t>(N) + y_col;
  z += y_row * static_cast<int64_t>(R) + band_z;
  b += band_z * static_cast<int64_t>(N) + y_col;

  LoaderW loader_w(w, K, Ws, simd_gid, simd_lid);

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

  // Spans are clamped in int before narrowing: M and N pass 32767.
  const short sgp_sm = short(min(int(SM), M - (y_row + tm)));
  const bool is_unaligned_sm = (M - y_row) < BM;
  // A simdgroup row at or beyond M keeps its share of the weight decode
  // and every barrier, and skips the A loads, the MMAs and the store.
  const bool sg_live = tm < M - y_row;

  const short sgp_sn = aligned_N ? SN : short(min(int(SN), N - (y_col + tn)));
  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;
  z += tm * R;
  // Columns of the simdgroup inside the band.
  const short sg_cols = short(clamp(band_cols - int(tn), 0, int(SN)));

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      // The low-rank term first, then the K walk and the store, in the
      // order of the plain kernel.
      if (sg_live) {
        kq_qmm_t_nax_lora_tail<T, kAlignedM.value, TM, TN, TK>(
            Dtile, z, b + tn, R, N, band_rank, sg_cols, sgp_sm);
      }
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; sg_live && kk1 < BK; kk1 += SK) {
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

      if (!sg_live) {
        return;
      }

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

// Kernel of one codec. LOADER64 decodes a K tile of 64 and LOADER128 a K
// tile of 128; a codec with one K tile names its loader twice. bands holds
// n_bands rows of (col_start, col_end, z_start, rank), disjoint and in
// column order, with every col_start on a column tile. The tile takes the
// last band that starts at or before it: past that band's end band_cols is
// at or below 0.
#define KQ_NAX_DEFINE_LORA_KERNEL(codec, LOADER64, LOADER128)             \
  template <                                                              \
      typename T,                                                         \
      bool aligned_N,                                                     \
      int BM,                                                             \
      int BN,                                                             \
      int BK,                                                             \
      int WM,                                                             \
      int WN>                                                             \
  [[kernel]] void kq_##codec##_qmm_t_nax_lora(                            \
      const device uint8_t* w,                                            \
      const device T* x,                                                  \
      const device T* z,                                                  \
      const device T* b,                                                  \
      device T* y,                                                        \
      const constant int& K,                                              \
      const constant int& N,                                              \
      const constant int& M,                                              \
      const constant int& R,                                              \
      const constant int& n_bands,                                        \
      const constant int* bands,                                          \
      uint3 tid [[threadgroup_position_in_grid]],                         \
      uint3 tgp_grid [[threadgroups_per_grid]],                           \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
      uint simd_lid [[thread_index_in_simdgroup]]) {                      \
    tid = kq_swizzle_tid(tid, tgp_grid, N, BN);                           \
    if (int(tid.y) * BM >= M) {                                           \
      return;                                                             \
    }                                                                     \
    const int y_col = int(tid.x) * BN;                                    \
    int band_z = 0;                                                       \
    int band_rank = 0;                                                    \
    int band_cols = 0;                                                    \
    for (int i = 0; i < n_bands; i++) {                                   \
      if (y_col >= bands[4 * i]) {                                        \
        band_z = bands[4 * i + 2];                                        \
        band_rank = bands[4 * i + 3];                                     \
        band_cols = min(BN, bands[4 * i + 1] - y_col);                    \
      }                                                                   \
    }                                                                     \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                      \
    threadgroup T Ws[BN * BK_padded];                                     \
    if constexpr (BK == 64) {                                             \
      using LoaderW = LOADER64<                                           \
          T,                                                              \
          BN,                                                             \
          64,                                                             \
          BK_padded,                                                      \
          /*reduction_dim=*/1,                                            \
          /*tgp_size=*/WM * WN * SIMD_SIZE>;                              \
      kq_qmm_t_nax_lora_impl<T, LoaderW, aligned_N, BM, 64, BN, WM, WN>(  \
          w,                                                              \
          x,                                                              \
          z,                                                              \
          b,                                                              \
          y,                                                              \
          Ws,                                                             \
          K,                                                              \
          N,                                                              \
          M,                                                              \
          R,                                                              \
          band_z,                                                         \
          band_rank,                                                      \
          band_cols,                                                      \
          tid,                                                            \
          simd_gid,                                                       \
          simd_lid);                                                      \
    } else {                                                              \
      using LoaderW = LOADER128<                                          \
          T,                                                              \
          BN,                                                             \
          128,                                                            \
          BK_padded,                                                      \
          /*reduction_dim=*/1,                                            \
          /*tgp_size=*/WM * WN * SIMD_SIZE>;                              \
      kq_qmm_t_nax_lora_impl<T, LoaderW, aligned_N, BM, 128, BN, WM, WN>( \
          w,                                                              \
          x,                                                              \
          z,                                                              \
          b,                                                              \
          y,                                                              \
          Ws,                                                             \
          K,                                                              \
          N,                                                              \
          M,                                                              \
          R,                                                              \
          band_z,                                                         \
          band_rank,                                                      \
          band_cols,                                                      \
          tid,                                                            \
          simd_gid,                                                       \
          simd_lid);                                                      \
    }                                                                     \
  }

KQ_NAX_DEFINE_LORA_KERNEL(q4_k, KqNaxQ4_KBlockLoader, KqNaxQ4_KHalfLoader)
KQ_NAX_DEFINE_LORA_KERNEL(q5_k, KqNaxQ5_KBlockLoader, KqNaxQ5_KHalfLoader)
KQ_NAX_DEFINE_LORA_KERNEL(q6_k, KqNaxQ6_KBlockLoader, KqNaxQ6_KHalfLoader)
KQ_NAX_DEFINE_LORA_KERNEL(q8_0, KqNaxQ8_0BlockLoader, KqNaxQ8_0BlockLoader)
#undef KQ_NAX_DEFINE_LORA_KERNEL
