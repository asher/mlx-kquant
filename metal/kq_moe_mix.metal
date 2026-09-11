// clang-format off
// Fused MoE unsort and mix instantiations; see kq_moe_mix.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_moe_mix.h"

instantiate_kernel("kq_moe_mix_bfloat16_t", kq_moe_mix, bfloat16_t)
instantiate_kernel("kq_moe_mix_float16_t", kq_moe_mix, float16_t)
