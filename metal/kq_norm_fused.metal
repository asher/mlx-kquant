// clang-format off
// Fused residual + RMSNorm glue kernel instantiations; see kq_norm_fused.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_norm_fused.h"

instantiate_kernel("kq_add_rmsnorm_bfloat16_t", kq_add_rmsnorm, bfloat16_t)
instantiate_kernel("kq_add_rmsnorm_float16_t", kq_add_rmsnorm, float16_t)
instantiate_kernel("kq_rmsnorm_multi3_bfloat16_t", kq_rmsnorm_multi3, bfloat16_t)
instantiate_kernel("kq_rmsnorm_multi3_float16_t", kq_rmsnorm_multi3, float16_t)
instantiate_kernel("kq_rmsnorm2_add_bfloat16_t", kq_rmsnorm2_add, bfloat16_t)
instantiate_kernel("kq_rmsnorm2_add_float16_t", kq_rmsnorm2_add, float16_t)
    // clang-format on
