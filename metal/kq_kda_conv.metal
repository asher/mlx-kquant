// clang-format off
// Fused causal short-conv instantiations; see kq_kda_conv.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_kda_conv.h"

#define instantiate_kq_kda_conv(type, npt) \
  instantiate_kernel("kq_kda_conv_" #type "_" #npt, kq_kda_conv, type, npt)

instantiate_kq_kda_conv(bfloat16_t, 2)
instantiate_kq_kda_conv(bfloat16_t, 4)
instantiate_kq_kda_conv(bfloat16_t, 8)
instantiate_kq_kda_conv(float16_t, 2)
instantiate_kq_kda_conv(float16_t, 4)
instantiate_kq_kda_conv(float16_t, 8)
