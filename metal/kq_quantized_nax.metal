// clang-format off
// Kernel instantiations; derived-code attribution lives in the included
// kq_*.h headers and mlx_kquant/licenses/.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/kq_quantized_nax.h"

#define instantiate_kquant_nax_qmm_t(                                                      \
    type, gs, bits, aligned_N, batched, bm, bn, wm, wn, codec)                             \
  instantiate_kernel(                                                                      \
      "kquant_" #codec "_qmm_t_nax_" #type "_gs_" #gs "_b_" #bits                          \
          "_bm" #bm "_bn" #bn "_bk64_wm" #wm "_wn" #wn                                     \
          "_alN_" #aligned_N "_batch_" #batched,                                           \
      kq_ ## codec ## _qmm_t_nax,                                                          \
      type,                                                                                \
      gs,                                                                                  \
      bits,                                                                                \
      aligned_N,                                                                           \
      batched,                                                                             \
      bm,                                                                                  \
      bn,                                                                                  \
      wm,                                                                                  \
      wn)

#define instantiate_kquant_nax_qmm_n(type, gs, bits, batched, codec)                       \
  instantiate_kernel(                                                                      \
      "kquant_" #codec "_qmm_n_nax_" #type "_gs_" #gs "_b_" #bits                          \
          "_bm64_bn64_bk64_wm2_wn2_batch_" #batched,                                       \
      kq_ ## codec ## _qmm_n_nax,                                                          \
      type,                                                                                \
      gs,                                                                                  \
      bits,                                                                                \
      batched)

#define instantiate_kquant_nax_gather_qmm_t(                                               \
    type, gs, bits, aligned_N, bm, bn, wm, wn, codec)                                      \
  instantiate_kernel(                                                                      \
      "kquant_" #codec "_gather_qmm_t_nax_" #type "_gs_" #gs "_b_" #bits                   \
          "_bm" #bm "_bn" #bn "_bk64_wm" #wm "_wn" #wn "_alN_" #aligned_N,                 \
      kq_ ## codec ## _gather_qmm_t_nax,                                                   \
      type,                                                                                \
      gs,                                                                                  \
      bits,                                                                                \
      aligned_N,                                                                           \
      bm,                                                                                  \
      bn,                                                                                  \
      wm,                                                                                  \
      wn)

#define instantiate_kquant_nax_gather_qmm_n(type, gs, bits, codec)                         \
  instantiate_kernel(                                                                      \
      "kquant_" #codec "_gather_qmm_n_nax_" #type "_gs_" #gs "_b_" #bits                   \
          "_bm64_bn64_bk64_wm2_wn2",                                                       \
      kq_ ## codec ## _gather_qmm_n_nax,                                                   \
      type,                                                                                \
      gs,                                                                                  \
      bits)

#define instantiate_kquant_nax_gather_qmm_rhs(                                             \
    type, gs, bits, transpose, suffix, bm, bn, wm, wn, codec)                              \
  instantiate_kernel(                                                                      \
      "kquant_" #codec "_gather_qmm_rhs_nax_" #suffix "_" #type                            \
          "_gs_" #gs "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_64_wm_" #wm "_wn_" #wn,        \
      kq_ ## codec ## _gather_qmm_rhs_nax,                                                 \
      type,                                                                                \
      gs,                                                                                  \
      bits,                                                                                \
      bm,                                                                                  \
      bn,                                                                                  \
      64,                                                                                  \
      wm,                                                                                  \
      wn,                                                                                  \
      transpose)

#define instantiate_kquant_nax_codec_for_type(codec, type, gs, bits)                        \
  instantiate_kquant_nax_qmm_t(type, gs, bits, true,  1,  64,  64, 2, 2, codec)            \
  instantiate_kquant_nax_qmm_t(type, gs, bits, true,  0,  64,  64, 2, 2, codec)            \
  instantiate_kquant_nax_qmm_t(type, gs, bits, false, 1,  64,  64, 2, 2, codec)            \
  instantiate_kquant_nax_qmm_t(type, gs, bits, false, 0,  64,  64, 2, 2, codec)            \
  instantiate_kquant_nax_qmm_n(type, gs, bits, 1, codec)                                   \
  instantiate_kquant_nax_qmm_n(type, gs, bits, 0, codec)                                   \
  instantiate_kquant_nax_gather_qmm_t(type, gs, bits, true,   64,  64, 2, 2, codec)        \
  instantiate_kquant_nax_gather_qmm_t(type, gs, bits, false,  64,  64, 2, 2, codec)        \
  instantiate_kquant_nax_gather_qmm_n(type, gs, bits, codec)                               \
  instantiate_kquant_nax_gather_qmm_rhs(type, gs, bits, true,  nt, 64, 64, 2, 2, codec)    \
  instantiate_kquant_nax_gather_qmm_rhs(type, gs, bits, false, nn, 64, 64, 2, 2, codec)

#define instantiate_kquant_nax_codec(codec, gs, bits)                                      \
  instantiate_kquant_nax_codec_for_type(codec, float,       gs, bits)                      \
  instantiate_kquant_nax_codec_for_type(codec, float16_t,   gs, bits)                      \
  instantiate_kquant_nax_codec_for_type(codec, bfloat16_t,  gs, bits)

// Small-BM qmm_t tile for the batch-decode M range: BM=64 wastes up to 75%
// of MMA issues on row padding at M<=32 (MMA measured ~48% of kernel time).
// BM=32 (wm2 wn2 -> TM=1, TN=2) is the smallest tile the fragment pairing
// in tile_matmad_nax and the loaders' n_reads==32 contract allow: BM=16
// forces TM=TN=1, where NEITHER matmad branch exists and the kernel is
// silently empty. q6_k first.
// BM=32 wm2/wn2 (TM=1, TN=2, paired-N matmad). NOTE two rejected shapes:
// BM=16 forces TM=TN=1 where NEITHER tile_matmad_nax pairing branch exists
// (silently empty kernel); BM=32 wm1/wn4 (TM=2, TN=1) takes the paired-M
// branch, which produces wrong numerics with (ta=false, tb=true) -- only
// visible at M>16 where the second row fragment is non-zero. The
// mixed-alignment penalty this shape would have fixed is handled by the
// TG-uniform alignment branch in kq_qmm_t_nax_tgp_impl instead.
#define instantiate_kquant_nax_qmm_t_smallbm(codec, type, gs, bits)          \
  instantiate_kquant_nax_qmm_t(type, gs, bits, true,  1, 32, 64, 2, 2, codec) \
  instantiate_kquant_nax_qmm_t(type, gs, bits, true,  0, 32, 64, 2, 2, codec) \
  instantiate_kquant_nax_qmm_t(type, gs, bits, false, 1, 32, 64, 2, 2, codec) \
  instantiate_kquant_nax_qmm_t(type, gs, bits, false, 0, 32, 64, 2, 2, codec)
#define instantiate_kquant_nax_smallbm(codec, gs, bits)                      \
  instantiate_kquant_nax_qmm_t_smallbm(codec, float,      gs, bits)          \
  instantiate_kquant_nax_qmm_t_smallbm(codec, float16_t,  gs, bits)          \
  instantiate_kquant_nax_qmm_t_smallbm(codec, bfloat16_t, gs, bits)
instantiate_kquant_nax_smallbm(q6_k, 256, 6)
instantiate_kquant_nax_smallbm(q8_0, 32, 8)
instantiate_kquant_nax_smallbm(q4_k, 256, 4)
instantiate_kquant_nax_smallbm(q5_k, 256, 5)
instantiate_kquant_nax_smallbm(q3_k, 256, 3)
instantiate_kquant_nax_smallbm(q2_k, 256, 2)
instantiate_kquant_nax_smallbm(q5_1, 32, 5)
instantiate_kquant_nax_smallbm(q4_0, 32, 4)
instantiate_kquant_nax_smallbm(q4_1, 32, 4)
instantiate_kquant_nax_smallbm(q5_0, 32, 5)
instantiate_kquant_nax_smallbm(iq4_nl, 32, 4)
instantiate_kquant_nax_smallbm(iq4_xs, 256, 4)
instantiate_kquant_nax_smallbm(iq3_xxs, 256, 3)
instantiate_kquant_nax_smallbm(iq3_s, 256, 3)
instantiate_kquant_nax_smallbm(iq2_xxs, 256, 2)
instantiate_kquant_nax_smallbm(iq2_xs, 256, 2)
instantiate_kquant_nax_smallbm(iq2_s, 256, 2)
instantiate_kquant_nax_smallbm(iq1_s, 256, 1)
instantiate_kquant_nax_smallbm(iq1_m, 256, 1)

instantiate_kquant_nax_codec(q8_0, 32, 8)
instantiate_kquant_nax_codec(q5_1, 32, 5)
instantiate_kquant_nax_codec(q4_0, 32, 4)
instantiate_kquant_nax_codec(q4_1, 32, 4)
instantiate_kquant_nax_codec(q5_0, 32, 5)
instantiate_kquant_nax_codec(q4_k, 256, 4)
instantiate_kquant_nax_codec(q5_k, 256, 5)
instantiate_kquant_nax_codec(q6_k, 256, 6)
instantiate_kquant_nax_codec(q3_k, 256, 3)
instantiate_kquant_nax_codec(q2_k, 256, 2)
instantiate_kquant_nax_codec(iq4_nl, 32, 4)
instantiate_kquant_nax_codec(iq4_xs, 256, 4)
instantiate_kquant_nax_codec(iq3_xxs, 256, 3)
instantiate_kquant_nax_codec(iq3_s, 256, 3)
instantiate_kquant_nax_codec(iq2_xxs, 256, 2)
instantiate_kquant_nax_codec(iq2_xs, 256, 2)
instantiate_kquant_nax_codec(iq2_s, 256, 2)
instantiate_kquant_nax_codec(iq1_s, 256, 1)
instantiate_kquant_nax_codec(iq1_m, 256, 1)
    // clang-format on
