// clang-format off
// Skinny matmul kernel instantiations; see kq_skinny_mv.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_skinny_mv.h"

instantiate_kernel("kq_skinny_mv_float16_t_float16_t", kq_skinny_mv, float16_t, float16_t, float16_t)
instantiate_kernel("kq_skinny_mv_bfloat16_t_bfloat16_t", kq_skinny_mv, bfloat16_t, bfloat16_t, bfloat16_t)
instantiate_kernel("kq_skinny_mv_float_float", kq_skinny_mv, float, float, float)
instantiate_kernel("kq_skinny_mv_float16_t_float", kq_skinny_mv, float16_t, float, float)
instantiate_kernel("kq_skinny_mv_bfloat16_t_float", kq_skinny_mv, bfloat16_t, float, float)
instantiate_kernel("kq_skinny_mv_float_float16_t", kq_skinny_mv, float, float16_t, float)
instantiate_kernel("kq_skinny_mv_float_bfloat16_t", kq_skinny_mv, float, bfloat16_t, float)
instantiate_kernel("kq_skinny_mma_float16_t_float16_t_f1", kq_skinny_mma, float16_t, float16_t, float16_t, 1)
instantiate_kernel("kq_skinny_mma_bfloat16_t_bfloat16_t_f1", kq_skinny_mma, bfloat16_t, bfloat16_t, bfloat16_t, 1)
instantiate_kernel("kq_skinny_mma_float_float_f1", kq_skinny_mma, float, float, float, 1)
instantiate_kernel("kq_skinny_mma_float16_t_float_f1", kq_skinny_mma, float16_t, float, float, 1)
instantiate_kernel("kq_skinny_mma_bfloat16_t_float_f1", kq_skinny_mma, bfloat16_t, float, float, 1)
instantiate_kernel("kq_skinny_mma_float_float16_t_f1", kq_skinny_mma, float, float16_t, float, 1)
instantiate_kernel("kq_skinny_mma_float_bfloat16_t_f1", kq_skinny_mma, float, bfloat16_t, float, 1)
instantiate_kernel("kq_skinny_mma_float16_t_float16_t_f2", kq_skinny_mma, float16_t, float16_t, float16_t, 2)
instantiate_kernel("kq_skinny_mma_bfloat16_t_bfloat16_t_f2", kq_skinny_mma, bfloat16_t, bfloat16_t, bfloat16_t, 2)
instantiate_kernel("kq_skinny_mma_float_float_f2", kq_skinny_mma, float, float, float, 2)
instantiate_kernel("kq_skinny_mma_float16_t_float_f2", kq_skinny_mma, float16_t, float, float, 2)
instantiate_kernel("kq_skinny_mma_bfloat16_t_float_f2", kq_skinny_mma, bfloat16_t, float, float, 2)
instantiate_kernel("kq_skinny_mma_float_float16_t_f2", kq_skinny_mma, float, float16_t, float, 2)
instantiate_kernel("kq_skinny_mma_float_bfloat16_t_f2", kq_skinny_mma, float, bfloat16_t, float, 2)
    // clang-format on
