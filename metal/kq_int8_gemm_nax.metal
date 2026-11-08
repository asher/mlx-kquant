// clang-format off
// int8 x int8 matmul instantiations; see kq_int8_gemm_nax.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_int8_gemm_nax.h"

#define instantiate_int8_mm(tname, type)                                     \
  instantiate_kernel("kq_int8_mm_nax_" #tname, kq_int8_mm_nax, type)         \
  instantiate_kernel("kq_int8_mm_grouped_nax_" #tname,                       \
                     kq_int8_mm_grouped_nax, type)

instantiate_int8_mm(bfloat16_t, bfloat16_t)
instantiate_int8_mm(float16_t, float16_t)
instantiate_int8_mm(float, float)
    // clang-format on
