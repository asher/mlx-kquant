// clang-format off
// NAX qmm_t kernels with an epilogue; see kq_quantized_nax_epi.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/kq_quantized_nax_epi.h"

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

instantiate_kquant_nax_glu_walks_tile(q4_k, 128, 128)
instantiate_kquant_nax_glu_walks_tile(q5_k, 128, 128)
instantiate_kquant_nax_glu_walks_tile(q6_k, 128, 128)
// q8_0 on bfloat16 only: on float16 it measured slower than the unfused ops.
instantiate_kquant_nax_glu_walks(q8_0, bfloat16_t, true,  128, 64)
instantiate_kquant_nax_glu_walks(q8_0, bfloat16_t, false, 128, 64)
    // clang-format on
