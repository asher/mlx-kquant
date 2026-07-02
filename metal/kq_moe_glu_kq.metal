// clang-format off
// K-quant fused MoE gather kernel instantiations; see kq_moe_glu_kq.h.
// Same include prelude as kq_quantized.metal (the per-codec dequant helpers
// live in kq_quantized*.h).
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"

#define instantiate_kq_moe_glu_kq(codec, type)                        \
  instantiate_kernel(                                                 \
      "kq_" #codec "_moe_glu_gather_silu_" #type,                     \
      kq_ ## codec ## _moe_glu_gather,                                \
      type,                                                           \
      KQ_GLU_ACT_SILU)                                                \
  instantiate_kernel(                                                 \
      "kq_" #codec "_moe_glu_gather_gelu_" #type,                     \
      kq_ ## codec ## _moe_glu_gather,                                \
      type,                                                           \
      KQ_GLU_ACT_GELU)                                                \
  instantiate_kernel(                                                 \
      "kq_" #codec "_gather_qmv_" #type,                              \
      kq_ ## codec ## _gather_qmv,                                    \
      type)                                                           \
  instantiate_kernel(                                                 \
      "kq_" #codec "_moe_glu_gather_shexp_silu_" #type,               \
      kq_ ## codec ## _moe_glu_gather_shexp,                          \
      type,                                                           \
      KQ_GLU_ACT_SILU)                                                \
  instantiate_kernel(                                                 \
      "kq_" #codec "_moe_glu_gather_shexp_gelu_" #type,               \
      kq_ ## codec ## _moe_glu_gather_shexp,                          \
      type,                                                           \
      KQ_GLU_ACT_GELU)                                                \
  instantiate_kernel(                                                 \
      "kq_" #codec "_gather_qmv_mix_" #type,                          \
      kq_ ## codec ## _gather_qmv_mix,                                \
      type)

instantiate_kq_moe_glu_kq(q6_k, bfloat16_t)
instantiate_kq_moe_glu_kq(q6_k, float16_t)
instantiate_kq_moe_glu_kq(q8_0, bfloat16_t)
instantiate_kq_moe_glu_kq(q8_0, float16_t)
    // clang-format on
