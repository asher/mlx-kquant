// clang-format off
// NAX qmm_t kernels with a low-rank term; see kq_quantized_nax_lora.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/kq_quantized_nax_lora.h"

#define instantiate_kquant_nax_lora(codec, type, al, bm, bk)                 \
  instantiate_kernel(                                                        \
      "kquant_" #codec "_qmm_t_nax_lora_" #type                              \
          "_bm" #bm "_bn64_bk" #bk "_wm2_wn2_alN_" #al,                      \
      kq_##codec##_qmm_t_nax_lora,                                           \
      type,                                                                  \
      al,                                                                    \
      bm,                                                                    \
      64,                                                                    \
      bk,                                                                    \
      2,                                                                     \
      2)

instantiate_kquant_nax_lora(q4_k, bfloat16_t, true,  128, 128)
instantiate_kquant_nax_lora(q4_k, bfloat16_t, false, 128, 128)
    // clang-format on
