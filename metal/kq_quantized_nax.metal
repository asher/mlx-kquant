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
// No float x variant, same reachability argument as the db64 macro.
#define instantiate_kquant_nax_smallbm(codec, gs, bits)                      \
  instantiate_kquant_nax_qmm_t_smallbm(codec, float16_t,  gs, bits)          \
  instantiate_kquant_nax_qmm_t_smallbm(codec, bfloat16_t, gs, bits)

// Double-buffered BM=64 qmm_t, name-suffixed _db: dispatched by the host
// solely for the M33-64 decode band (kq_smallbm_policy db64 + KQ_NAX_DB64).
// As a blanket BM=64 default the doubled Ws cut occupancy (M96+ -3-15%,
// prefill -3-7%); band-gating keeps the +7-17% M33-64 decode win without
// the tax. Same tile shape as the stock bm64 kernel; only kWsBufs differs.
#define instantiate_kquant_nax_qmm_t_db(                                     \
    type, gs, bits, aligned_N, batched, codec)                               \
  instantiate_kernel(                                                        \
      "kquant_" #codec "_qmm_t_nax_" #type "_gs_" #gs "_b_" #bits            \
          "_bm64_bn64_bk64_wm2_wn2"                                          \
          "_alN_" #aligned_N "_batch_" #batched "_db",                       \
      kq_ ## codec ## _qmm_t_nax,                                            \
      type, gs, bits, aligned_N, batched, 64, 64, 2, 2, true)
// No float x variant: kquant_ops promotes float32 activations to bf16
// before the primitive and the qmm_nax route is gated on
// x.dtype() != float32, so a float lookup can never happen. Codecs are
// the five whose policy can enable the band (db64_min_n > 0); probing a
// crossover on another codec now needs its instantiation restored and a
// rebuild, not just KQ_NAX_DB64=1 (the host availability-gates on the
// policy so the force lever cannot look up a missing kernel).
#define instantiate_kquant_nax_qmm_t_db64_for_type(codec, type, gs, bits)    \
  instantiate_kquant_nax_qmm_t_db(type, gs, bits, true,  1, codec)           \
  instantiate_kquant_nax_qmm_t_db(type, gs, bits, true,  0, codec)           \
  instantiate_kquant_nax_qmm_t_db(type, gs, bits, false, 1, codec)           \
  instantiate_kquant_nax_qmm_t_db(type, gs, bits, false, 0, codec)
#define instantiate_kquant_nax_db64(codec, gs, bits)                         \
  instantiate_kquant_nax_qmm_t_db64_for_type(codec, float16_t,  gs, bits)    \
  instantiate_kquant_nax_qmm_t_db64_for_type(codec, bfloat16_t, gs, bits)
instantiate_kquant_nax_db64(q6_k, 256, 6)
instantiate_kquant_nax_db64(q8_0, 32, 8)
instantiate_kquant_nax_db64(q5_1, 32, 5)
instantiate_kquant_nax_db64(q4_1, 32, 4)
instantiate_kquant_nax_db64(q5_0, 32, 5)
// BM=128 qmm_t tile (wm2 wn2 -> TM=4, TN=2, paired-N matmad; generic in
// TM so no pairing-branch trap). The M>64 regime is bound by
// per-threadgroup work (loader dequant + MMA issue, real DRAM ~half
// peak); BM=128 halves the row-tile count and with it total loader
// dequant per unique weight. A loads straight from device, so the cost
// is Dtile registers: 8 accumulator fragments vs 4. Dispatched only when
// ceil(M/64) is even and M >= 193 (padding parity with BM=64; q6_k ABBA
// at three shapes: +9-25% M224-512, prefill M512-2048 +8-10% hot,
// +31-40% cold). KQ_NAX_BM128=0 kills.
#define instantiate_kquant_nax_qmm_t_bm128(codec, type, gs, bits)             \
  instantiate_kquant_nax_qmm_t(type, gs, bits, true,  1, 128, 64, 2, 2, codec) \
  instantiate_kquant_nax_qmm_t(type, gs, bits, true,  0, 128, 64, 2, 2, codec) \
  instantiate_kquant_nax_qmm_t(type, gs, bits, false, 1, 128, 64, 2, 2, codec) \
  instantiate_kquant_nax_qmm_t(type, gs, bits, false, 0, 128, 64, 2, 2, codec)
// No float x variant, same reachability argument as the db64 macro.
#define instantiate_kquant_nax_bm128(codec, gs, bits)                        \
  instantiate_kquant_nax_qmm_t_bm128(codec, float16_t,  gs, bits)            \
  instantiate_kquant_nax_qmm_t_bm128(codec, bfloat16_t, gs, bits)
instantiate_kquant_nax_bm128(q6_k, 256, 6)
instantiate_kquant_nax_bm128(q8_0, 32, 8)
instantiate_kquant_nax_bm128(q4_k, 256, 4)
instantiate_kquant_nax_bm128(q5_k, 256, 5)
instantiate_kquant_nax_bm128(q3_k, 256, 3)
instantiate_kquant_nax_bm128(q2_k, 256, 2)
instantiate_kquant_nax_bm128(q5_1, 32, 5)
instantiate_kquant_nax_bm128(q4_0, 32, 4)
instantiate_kquant_nax_bm128(q4_1, 32, 4)
instantiate_kquant_nax_bm128(q5_0, 32, 5)
instantiate_kquant_nax_bm128(iq4_nl, 32, 4)
instantiate_kquant_nax_bm128(iq4_xs, 256, 4)
instantiate_kquant_nax_bm128(iq3_xxs, 256, 3)
instantiate_kquant_nax_bm128(iq3_s, 256, 3)
instantiate_kquant_nax_bm128(iq2_xxs, 256, 2)
instantiate_kquant_nax_bm128(iq2_xs, 256, 2)
instantiate_kquant_nax_bm128(iq2_s, 256, 2)
instantiate_kquant_nax_bm128(iq1_s, 256, 1)
instantiate_kquant_nax_bm128(iq1_m, 256, 1)

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
