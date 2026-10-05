// clang-format off
// Probe tiles of the epilogue kernels, built with KQ_PROBE_KERNELS=ON only:
// the one-walk GLU form and the two-walk tiles the op does not run.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/kq_quantized_nax_epi.h"

#define instantiate_kquant_nax_glu(codec, type, al, bm, bk)                  \
  instantiate_kernel(                                                        \
      "kquant_" #codec "_qmm_t_nax_glu_" #type                               \
          "_bm" #bm "_bn64_bk" #bk "_wm2_wn2_alN_" #al,                      \
      kq_##codec##_qmm_t_nax_glu,                                            \
      type,                                                                  \
      al,                                                                    \
      bm,                                                                    \
      64,                                                                    \
      bk,                                                                    \
      2,                                                                     \
      2)
#define instantiate_kquant_nax_glu_d128(codec, type, al, bm)                 \
  instantiate_kernel(                                                        \
      "kquant_" #codec "_qmm_t_nax_glu_" #type                               \
          "_bm" #bm "_bn64_bk128d_wm2_wn2_alN_" #al,                         \
      kq_##codec##_qmm_t_nax_glu_d128,                                       \
      type,                                                                  \
      al,                                                                    \
      bm,                                                                    \
      64,                                                                    \
      2,                                                                     \
      2)
#define instantiate_kquant_nax_glu_d128_bm(codec, bm)                        \
  instantiate_kquant_nax_glu_d128(codec, float16_t,  true,  bm)              \
  instantiate_kquant_nax_glu_d128(codec, float16_t,  false, bm)              \
  instantiate_kquant_nax_glu_d128(codec, bfloat16_t, true,  bm)              \
  instantiate_kquant_nax_glu_d128(codec, bfloat16_t, false, bm)
#define instantiate_kquant_nax_glu_walks(codec, type, al, bm, bk)            \
  instantiate_kernel(                                                        \
      "kquant_" #codec "_qmm_t_nax_glu_" #type                               \
          "_bm" #bm "_bn64_bk" #bk "w_wm2_wn2_alN_" #al,                     \
      kq_##codec##_qmm_t_nax_glu_walks,                                      \
      type,                                                                  \
      al,                                                                    \
      bm,                                                                    \
      64,                                                                    \
      bk,                                                                    \
      2,                                                                     \
      2)
#define instantiate_kquant_nax_glu_walks_tile(codec, bm, bk)                 \
  instantiate_kquant_nax_glu_walks(codec, float16_t,  true,  bm, bk)         \
  instantiate_kquant_nax_glu_walks(codec, float16_t,  false, bm, bk)         \
  instantiate_kquant_nax_glu_walks(codec, bfloat16_t, true,  bm, bk)         \
  instantiate_kquant_nax_glu_walks(codec, bfloat16_t, false, bm, bk)
#define instantiate_kquant_nax_glu_tile(codec, bm, bk)                       \
  instantiate_kquant_nax_glu(codec, float16_t,  true,  bm, bk)               \
  instantiate_kquant_nax_glu(codec, float16_t,  false, bm, bk)               \
  instantiate_kquant_nax_glu(codec, bfloat16_t, true,  bm, bk)               \
  instantiate_kquant_nax_glu(codec, bfloat16_t, false, bm, bk)
#define instantiate_kquant_nax_glu_probe_codec(codec)                        \
  instantiate_kquant_nax_glu_tile(codec, 64,  64)                            \
  instantiate_kquant_nax_glu_tile(codec, 64,  128)                           \
  instantiate_kquant_nax_glu_tile(codec, 128, 64)                            \
  instantiate_kquant_nax_glu_tile(codec, 128, 128)                           \
  instantiate_kquant_nax_glu_d128_bm(codec, 64)                              \
  instantiate_kquant_nax_glu_d128_bm(codec, 128)                             \
  instantiate_kquant_nax_glu_walks_tile(codec, 64,  64)                      \
  instantiate_kquant_nax_glu_walks_tile(codec, 64,  128)                     \
  instantiate_kquant_nax_glu_walks_tile(codec, 128, 64)

instantiate_kquant_nax_glu_probe_codec(q4_k)
    // clang-format on
