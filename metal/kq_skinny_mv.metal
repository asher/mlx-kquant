// clang-format off
// Skinny matmul kernel instantiations; see kq_skinny_mv.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_skinny_mv.h"

instantiate_kernel("kq_skinny_mv_float16_t_float16_t", kq_skinny_mv, float16_t, float16_t, float16_t)
instantiate_kernel("kq_skinny_mv_bfloat16_t_bfloat16_t", kq_skinny_mv, bfloat16_t, bfloat16_t, bfloat16_t)
instantiate_kernel("kq_skinny_mv_float_float", kq_skinny_mv, float, float, float)
instantiate_kernel("kq_skinny_mv_float16_t_float", kq_skinny_mv, float16_t, float, float)
instantiate_kernel("kq_skinny_mv_bfloat16_t_float", kq_skinny_mv, bfloat16_t, float, float)
    // clang-format on
