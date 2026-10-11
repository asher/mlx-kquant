// NAX qmm_t kernel with a residual epilogue, built on the tile and loaders
// of kq_quantized_nax.h. Separate entry points, so the plain qmm_t kernels
// keep their code.
//
//   y = r + g * (x @ W^T)
//
// x [B, M, K], W a quantized weight [N, K], r and y [B, M, N], g [B, N] or
// [N] (g_stride N or 0), all in the dtype of x. With gated false there is
// no g and y = r + x @ W^T. A threadgroup owns one [BM, BN] tile of y in
// batch tid.z. It walks K as the plain kernel does into one f32
// accumulator, then reads its tile of r and its columns of g and stores
// r + g * acc, rounded once.
#include "mlx/backend/metal/kernels/kq_quantized_nax.h"

template <
    typename T,
    typename LoaderW,
    const bool aligned_N,
    const bool gated,
    const int BM,
    const int BK,
    const int BN,
    const int WM,
    const int WN>
METAL_FUNC void kq_qmm_t_nax_resid_impl(
    const device uint8_t* w,
    const device T* x,
    const device T* r,
    const device T* g,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& g_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK % 32 == 0, "BK must be a multiple of the MMA K step");

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const int64_t batch = tid.z;

  x += (batch * M + y_row) * static_cast<int64_t>(K);
  w += static_cast<int64_t>(y_col) * K_w;
  y += (batch * M + y_row) * static_cast<int64_t>(N) + y_col;
  r += (batch * M + y_row) * static_cast<int64_t>(N) + y_col;
  if constexpr (gated) {
    g += batch * g_stride + y_col;
  }

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

      const device T* rs = r + tm * N + tn;
      device T* ys = y + tm * N + tn;
      const bool full = (kAlignedM.value && kAlignedN.value) ||
          (kAlignedM.value && sgp_sn == SN);

      // The residual tile, and the gate as a tile whose rows all read
      // the same columns (a row stride of 0).
      NAXTile<float, TM, TN> Rtile;
      if (full) {
        Rtile.load(rs, N);
      } else {
        Rtile.load_safe(rs, N, short2(sgp_sn, sgp_sm));
      }
      constexpr short NE = decltype(Dtile)::kElemsPerFrag;
      constexpr short NF = decltype(Dtile)::kNumFrags;
      if constexpr (gated) {
        NAXTile<float, TM, TN> Gtile;
        if (full) {
          Gtile.load(g + tn, 0);
        } else {
          Gtile.load_safe(g + tn, 0, short2(sgp_sn, sgp_sm));
        }
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < NF; ++f) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < NE; ++e) {
            Dtile.val_frags[f][e] = Rtile.val_frags[f][e] +
                Gtile.val_frags[f][e] * Dtile.val_frags[f][e];
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < NF; ++f) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < NE; ++e) {
            Dtile.val_frags[f][e] += Rtile.val_frags[f][e];
          }
        }
      }

      if (full) {
        Dtile.store(ys, N);
      } else {
        Dtile.store_safe(ys, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

// Kernel of one codec. LOADER64 decodes a K tile of 64 and LOADER128 a K
// tile of 128; a codec with one K tile names its loader twice.
#define KQ_NAX_DEFINE_RESID_KERNEL(codec, LOADER64, LOADER128)                \
  template <                                                                  \
      typename T,                                                             \
      bool aligned_N,                                                         \
      bool gated,                                                             \
      int BM,                                                                 \
      int BN,                                                                 \
      int BK,                                                                 \
      int WM,                                                                 \
      int WN>                                                                 \
  [[kernel]] void kq_##codec##_qmm_t_nax_resid(                               \
      const device uint8_t* w,                                                \
      const device T* x,                                                      \
      const device T* r,                                                      \
      const device T* g,                                                      \
      device T* y,                                                            \
      const constant int& K,                                                  \
      const constant int& N,                                                  \
      const constant int& M,                                                  \
      const constant int& g_stride,                                           \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint3 tgp_grid [[threadgroups_per_grid]],                               \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]) {                          \
    tid = kq_swizzle_tid(tid, tgp_grid, N, BN);                               \
    if (int(tid.y) * BM >= M) {                                               \
      return;                                                                 \
    }                                                                         \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                          \
    threadgroup T Ws[BN * BK_padded];                                         \
    if constexpr (BK == 64) {                                                 \
      using LoaderW = LOADER64<                                               \
          T,                                                                  \
          BN,                                                                 \
          64,                                                                 \
          BK_padded,                                                          \
          /*reduction_dim=*/1,                                                \
          /*tgp_size=*/WM * WN * SIMD_SIZE>;                                  \
      kq_qmm_t_nax_resid_impl<                                                \
          T,                                                                  \
          LoaderW,                                                            \
          aligned_N,                                                          \
          gated,                                                              \
          BM,                                                                 \
          64,                                                                 \
          BN,                                                                 \
          WM,                                                                 \
          WN>(w, x, r, g, y, Ws, K, N, M, g_stride, tid, simd_gid, simd_lid); \
    } else {                                                                  \
      using LoaderW = LOADER128<                                              \
          T,                                                                  \
          BN,                                                                 \
          128,                                                                \
          BK_padded,                                                          \
          /*reduction_dim=*/1,                                                \
          /*tgp_size=*/WM * WN * SIMD_SIZE>;                                  \
      kq_qmm_t_nax_resid_impl<                                                \
          T,                                                                  \
          LoaderW,                                                            \
          aligned_N,                                                          \
          gated,                                                              \
          BM,                                                                 \
          128,                                                                \
          BN,                                                                 \
          WM,                                                                 \
          WN>(w, x, r, g, y, Ws, K, N, M, g_stride, tid, simd_gid, simd_lid); \
    }                                                                         \
  }

KQ_NAX_DEFINE_RESID_KERNEL(q4_k, KqNaxQ4_KBlockLoader, KqNaxQ4_KHalfLoader)
KQ_NAX_DEFINE_RESID_KERNEL(q5_k, KqNaxQ5_KBlockLoader, KqNaxQ5_KHalfLoader)
KQ_NAX_DEFINE_RESID_KERNEL(q6_k, KqNaxQ6_KBlockLoader, KqNaxQ6_KHalfLoader)
KQ_NAX_DEFINE_RESID_KERNEL(q8_0, KqNaxQ8_0BlockLoader, KqNaxQ8_0BlockLoader)
#undef KQ_NAX_DEFINE_RESID_KERNEL
