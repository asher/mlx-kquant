// NAX qmm_t kernels with an epilogue, built on the tile and loaders of
// kq_quantized_nax.h. Separate entry points, so the plain qmm_t kernels
// keep their code.
//
// GLU product: y = act(x @ Wg^T) * (x @ Wu^T) for two quantized weights of
// one codec and one shape. A threadgroup owns one [BM, BN] output tile of
// y, the only output buffer. The activation and the product run in f32.
//
// Two forms. The two-walk form (kq_qmm_t_nax_glu_walks_impl) is the one
// the op runs. The one-walk form (kq_qmm_t_nax_glu_impl) keeps both sums
// live and measured slower; it builds as a probe kernel only.
#include "mlx/backend/metal/kernels/kq_quantized_nax.h"

// 0 = silu, 1 = gelu (tanh form).
constant int kq_epi_act [[function_constant(360)]];
// Clamp to the finite half range before the store (float16 outputs).
constant bool kq_epi_saturate [[function_constant(361)]];

METAL_FUNC float kq_epi_activation(float g) {
  if (kq_epi_act == 1) {
    // 0.5 g (1 + tanh(sqrt(2 / pi) (g + 0.044715 g^3)))
    const float u = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (1.0f + precise::tanh(u));
  }
  return g / (1.0f + precise::exp(-g));
}

// GLU product in one K walk with two f32 tiles, rounded once at the store.
// dual = true keeps the two decoded weight tiles in two Ws buffers: one
// barrier pair per K step serves both weights, and each A tile is loaded
// once for both products. dual = false decodes the two weights one after
// the other into one Ws buffer. WS_LD is the row stride of a tile in Ws:
// BK plus the pad of the plain kernels, or BK alone where two padded
// buffers would pass the 32 KB of threadgroup memory.
template <
    typename T,
    typename LoaderW,
    const bool aligned_N,
    const int BM,
    const int BK,
    const int BN,
    const int WM,
    const int WN,
    const bool dual,
    const int WS_LD = (BK + 16 / sizeof(T))>
METAL_FUNC void kq_qmm_t_nax_glu_impl(
    const device uint8_t* wg,
    const device uint8_t* wu,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK % 32 == 0, "BK must be a multiple of the MMA K step");

  // Row stride of a decoded weight tile in Ws.
  constexpr int BK_padded = WS_LD;
  constexpr int WS_STRIDE = BN * BK_padded;

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  x += y_row * static_cast<int64_t>(K);
  wg += static_cast<int64_t>(y_col) * K_w;
  wu += static_cast<int64_t>(y_col) * K_w;
  y += y_row * static_cast<int64_t>(N) + y_col;

  threadgroup T* Wg = Ws;
  threadgroup T* Wu = dual ? Ws + WS_STRIDE : Ws;
  LoaderW loader_g(wg, K, Wg, simd_gid, simd_lid);
  LoaderW loader_u(wu, K, Wu, simd_gid, simd_lid);

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
  // A simdgroup row at or beyond M decodes its share of the weights and
  // keeps every barrier, and skips the A loads, the MMAs and the store.
  const bool sg_live = tm < M - y_row;

  const short sgp_sn = aligned_N ? SN : short(min(int(SN), N - (y_col + tn)));
  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  NAXTile<float, TM, TN> Dg;
  NAXTile<float, TM, TN> Du;
  Dg.clear();
  Du.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_g.load_unsafe();
        } else {
          loader_g.load_safe(short2(BK, tgp_bn));
        }
        if constexpr (dual) {
          if constexpr (kAlignedN.value) {
            loader_u.load_unsafe();
          } else {
            loader_u.load_safe(short2(BK, tgp_bn));
          }
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

          Btile.template load<T, BK_padded, 1>(Wg + tn * BK_padded + kk1);
          tile_matmad_nax(
              Dg,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          if constexpr (dual) {
            Btile.template load<T, BK_padded, 1>(Wu + tn * BK_padded + kk1);
            tile_matmad_nax(
                Du,
                Atile,
                metal::bool_constant<transpose_a>{},
                Btile,
                metal::bool_constant<transpose_b>{});
          }

          (void)compiler_barrier;
        }

        if constexpr (!dual) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_u.load_unsafe();
          } else {
            loader_u.load_safe(short2(BK, tgp_bn));
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

            Btile.template load<T, BK_padded, 1>(Wu + tn * BK_padded + kk1);
            tile_matmad_nax(
                Du,
                Atile,
                metal::bool_constant<transpose_a>{},
                Btile,
                metal::bool_constant<transpose_b>{});

            (void)compiler_barrier;
          }
        }

        x += BK;
        loader_g.next();
        loader_u.next();
      }

      if (!sg_live) {
        return;
      }

      constexpr short NE = decltype(Dg)::kElemsPerFrag;
      constexpr short NF = decltype(Dg)::kNumFrags;
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < NF; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < NE; ++e) {
          float v = kq_epi_activation(Dg.val_frags[f][e]) * Du.val_frags[f][e];
          if (kq_epi_saturate) {
            v = clamp(v, -65504.0f, 65504.0f);
          }
          Dg.val_frags[f][e] = v;
        }
      }

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dg.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dg.store(y + tm * N + tn, N);
      } else {
        Dg.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

// GLU product in two K walks with one f32 tile: the gate walk stores
// act(x @ Wg^T) to y in the output dtype, the up walk reads that tile back
// and stores the product. One accumulator tile is live at a time, as in
// the plain kernels. With the saturating store the activation is clamped
// to the finite half range before the product, and the product after it.
template <
    typename T,
    typename LoaderW,
    const bool aligned_N,
    const int BM,
    const int BK,
    const int BN,
    const int WM,
    const int WN>
METAL_FUNC void kq_qmm_t_nax_glu_walks_impl(
    const device uint8_t* wg,
    const device uint8_t* wu,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK % 32 == 0, "BK must be a multiple of the MMA K step");

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  const int K_w = (K / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  x += y_row * static_cast<int64_t>(K);
  wg += static_cast<int64_t>(y_col) * K_w;
  wu += static_cast<int64_t>(y_col) * K_w;
  y += y_row * static_cast<int64_t>(N) + y_col;

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
  const bool sg_live = tm < M - y_row;

  const short sgp_sn = aligned_N ? SN : short(min(int(SN), N - (y_col + tn)));
  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  x += tm * K;
  y += tm * N + tn;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      STEEL_PRAGMA_NO_UNROLL
      for (int pass = 0; pass < 2; pass++) {
        LoaderW loader_w(pass == 0 ? wg : wu, K, Ws, simd_gid, simd_lid);
        const device T* xk = x;
        NAXTile<float, TM, TN> Dtile;
        Dtile.clear();

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
              Atile.load(xk + kk1, K);
            } else {
              Atile.load_safe(xk + kk1, K, short2(SK, sgp_sm));
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

          xk += BK;
          loader_w.next();
        }

        if (!sg_live) {
          continue;
        }

        constexpr short NE = decltype(Dtile)::kElemsPerFrag;
        constexpr short NF = decltype(Dtile)::kNumFrags;
        if (pass == 0) {
          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < NF; ++f) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < NE; ++e) {
              float v = kq_epi_activation(Dtile.val_frags[f][e]);
              if (kq_epi_saturate) {
                v = clamp(v, -65504.0f, 65504.0f);
              }
              Dtile.val_frags[f][e] = v;
            }
          }
        } else {
          NAXTile<float, TM, TN> Ctile;
          if constexpr (kAlignedM.value && kAlignedN.value) {
            Ctile.load(y, N);
          } else if (kAlignedM.value && sgp_sn == SN) {
            Ctile.load(y, N);
          } else {
            Ctile.load_safe(y, N, short2(sgp_sn, sgp_sm));
          }
          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < NF; ++f) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < NE; ++e) {
              float v = Ctile.val_frags[f][e] * Dtile.val_frags[f][e];
              if (kq_epi_saturate) {
                v = clamp(v, -65504.0f, 65504.0f);
              }
              Dtile.val_frags[f][e] = v;
            }
          }
        }

        if constexpr (kAlignedM.value && kAlignedN.value) {
          Dtile.store(y, N);
        } else if (kAlignedM.value && sgp_sn == SN) {
          Dtile.store(y, N);
        } else {
          Dtile.store_safe(y, N, short2(sgp_sn, sgp_sm));
        }
      }
    });
  });
}

#define KQ_NAX_DEFINE_GLU_WALKS_KERNEL(codec, LOADER64, LOADER128)             \
  template <                                                                   \
      typename T,                                                              \
      bool aligned_N,                                                          \
      int BM,                                                                  \
      int BN,                                                                  \
      int BK,                                                                  \
      int WM,                                                                  \
      int WN>                                                                  \
  [[kernel]] void kq_##codec##_qmm_t_nax_glu_walks(                            \
      const device uint8_t* wg,                                                \
      const device uint8_t* wu,                                                \
      const device T* x,                                                       \
      device T* y,                                                             \
      const constant int& K,                                                   \
      const constant int& N,                                                   \
      const constant int& M,                                                   \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint3 tgp_grid [[threadgroups_per_grid]],                                \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]) {                           \
    tid = kq_swizzle_tid(tid, tgp_grid, N, BN);                                \
    if (int(tid.y) * BM >= M) {                                                \
      return;                                                                  \
    }                                                                          \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                           \
    threadgroup T Ws[BN * BK_padded];                                          \
    if constexpr (BK == 64) {                                                  \
      using LoaderW = LOADER64<                                                \
          T,                                                                   \
          BN,                                                                  \
          64,                                                                  \
          BK_padded,                                                           \
          /*reduction_dim=*/1,                                                 \
          /*tgp_size=*/WM * WN * SIMD_SIZE>;                                   \
      kq_qmm_t_nax_glu_walks_impl<T, LoaderW, aligned_N, BM, 64, BN, WM, WN>(  \
          wg, wu, x, y, Ws, K, N, M, tid, simd_gid, simd_lid);                 \
    } else {                                                                   \
      using LoaderW = LOADER128<                                               \
          T,                                                                   \
          BN,                                                                  \
          128,                                                                 \
          BK_padded,                                                           \
          /*reduction_dim=*/1,                                                 \
          /*tgp_size=*/WM * WN * SIMD_SIZE>;                                   \
      kq_qmm_t_nax_glu_walks_impl<T, LoaderW, aligned_N, BM, 128, BN, WM, WN>( \
          wg, wu, x, y, Ws, K, N, M, tid, simd_gid, simd_lid);                 \
    }                                                                          \
  }

// GLU kernel of one codec. LOADER64 decodes a K tile of 64 and LOADER128 a
// K tile of 128. BK 64 keeps two Ws buffers; BK 128 keeps one, since two
// would pass 32 KB.
#define KQ_NAX_DEFINE_GLU_KERNEL(codec, LOADER64, LOADER128)                  \
  template <                                                                  \
      typename T,                                                             \
      bool aligned_N,                                                         \
      int BM,                                                                 \
      int BN,                                                                 \
      int BK,                                                                 \
      int WM,                                                                 \
      int WN>                                                                 \
  [[kernel]] void kq_##codec##_qmm_t_nax_glu(                                 \
      const device uint8_t* wg,                                               \
      const device uint8_t* wu,                                               \
      const device T* x,                                                      \
      device T* y,                                                            \
      const constant int& K,                                                  \
      const constant int& N,                                                  \
      const constant int& M,                                                  \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint3 tgp_grid [[threadgroups_per_grid]],                               \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]) {                          \
    tid = kq_swizzle_tid(tid, tgp_grid, N, BN);                               \
    if (int(tid.y) * BM >= M) {                                               \
      return;                                                                 \
    }                                                                         \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                          \
    constexpr bool dual = BK == 64;                                           \
    threadgroup T Ws[(dual ? 2 : 1) * BN * BK_padded];                        \
    if constexpr (BK == 64) {                                                 \
      using LoaderW = LOADER64<                                               \
          T,                                                                  \
          BN,                                                                 \
          64,                                                                 \
          BK_padded,                                                          \
          /*reduction_dim=*/1,                                                \
          /*tgp_size=*/WM * WN * SIMD_SIZE>;                                  \
      kq_qmm_t_nax_glu_impl<T, LoaderW, aligned_N, BM, 64, BN, WM, WN, true>( \
          wg, wu, x, y, Ws, K, N, M, tid, simd_gid, simd_lid);                \
    } else {                                                                  \
      using LoaderW = LOADER128<                                              \
          T,                                                                  \
          BN,                                                                 \
          128,                                                                \
          BK_padded,                                                          \
          /*reduction_dim=*/1,                                                \
          /*tgp_size=*/WM * WN * SIMD_SIZE>;                                  \
      kq_qmm_t_nax_glu_impl<                                                  \
          T,                                                                  \
          LoaderW,                                                            \
          aligned_N,                                                          \
          BM,                                                                 \
          128,                                                                \
          BN,                                                                 \
          WM,                                                                 \
          WN,                                                                 \
          false>(wg, wu, x, y, Ws, K, N, M, tid, simd_gid, simd_lid);         \
    }                                                                         \
  }

// GLU kernel on a K tile of 128 with two unpadded Ws buffers, 32 KB at
// BN 64 and a 2-byte T.
#define KQ_NAX_DEFINE_GLU_D128_KERNEL(codec, LOADER128)                 \
  template <typename T, bool aligned_N, int BM, int BN, int WM, int WN> \
  [[kernel]] void kq_##codec##_qmm_t_nax_glu_d128(                      \
      const device uint8_t* wg,                                         \
      const device uint8_t* wu,                                         \
      const device T* x,                                                \
      device T* y,                                                      \
      const constant int& K,                                            \
      const constant int& N,                                            \
      const constant int& M,                                            \
      uint3 tid [[threadgroup_position_in_grid]],                       \
      uint3 tgp_grid [[threadgroups_per_grid]],                         \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                 \
      uint simd_lid [[thread_index_in_simdgroup]]) {                    \
    tid = kq_swizzle_tid(tid, tgp_grid, N, BN);                         \
    if (int(tid.y) * BM >= M) {                                         \
      return;                                                           \
    }                                                                   \
    static_assert(2 * BN * 128 * sizeof(T) <= 32768, "Ws over 32 KB");  \
    threadgroup T Ws[2 * BN * 128];                                     \
    using LoaderW = LOADER128<                                          \
        T,                                                              \
        BN,                                                             \
        128,                                                            \
        128,                                                            \
        /*reduction_dim=*/1,                                            \
        /*tgp_size=*/WM * WN * SIMD_SIZE>;                              \
    kq_qmm_t_nax_glu_impl<                                              \
        T,                                                              \
        LoaderW,                                                        \
        aligned_N,                                                      \
        BM,                                                             \
        128,                                                            \
        BN,                                                             \
        WM,                                                             \
        WN,                                                             \
        true,                                                           \
        128>(wg, wu, x, y, Ws, K, N, M, tid, simd_gid, simd_lid);       \
  }

KQ_NAX_DEFINE_GLU_KERNEL(q4_k, KqNaxQ4_KBlockLoader, KqNaxQ4_KHalfLoader)
KQ_NAX_DEFINE_GLU_D128_KERNEL(q4_k, KqNaxQ4_KHalfLoader)
KQ_NAX_DEFINE_GLU_WALKS_KERNEL(q4_k, KqNaxQ4_KBlockLoader, KqNaxQ4_KHalfLoader)
#undef KQ_NAX_DEFINE_GLU_WALKS_KERNEL
#undef KQ_NAX_DEFINE_GLU_KERNEL
#undef KQ_NAX_DEFINE_GLU_D128_KERNEL
