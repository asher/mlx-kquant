// clang-format off
// Fused MoE gather kernel instantiations (mxfp4 packed layout). Self-contained
// load/dot primitives; see kq_moe_glu.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_moe_glu.h"

#define instantiate_kq_moe_glu(type)                                  \
  instantiate_kernel(                                                 \
      "kq_moe_glu_gather_" #type,                                     \
      kq_moe_glu_gather,                                              \
      type)                                                           \
  instantiate_kernel(                                                 \
      "kq_gather_qmv_bias_" #type,                                    \
      kq_gather_qmv_bias,                                             \
      type)

instantiate_kq_moe_glu(bfloat16_t)
instantiate_kq_moe_glu(float16_t)
    // clang-format on
