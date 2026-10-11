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

#define instantiate_kquant_nax_lora_codec(codec, bk)                         \
  instantiate_kquant_nax_lora(codec, bfloat16_t, true,  128, bk)             \
  instantiate_kquant_nax_lora(codec, bfloat16_t, false, 128, bk)
instantiate_kquant_nax_lora_codec(q4_k, 128)
instantiate_kquant_nax_lora_codec(q5_k, 128)
instantiate_kquant_nax_lora_codec(q6_k, 128)
instantiate_kquant_nax_lora_codec(q8_0, 64)
    // clang-format on
