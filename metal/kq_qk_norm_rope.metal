// clang-format off
// Fused per-head RMSNorm and RoPE instantiations; see kq_qk_norm_rope.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_qk_norm_rope.h"

#define instantiate_kq_qk_norm_rope(type, d) \
  instantiate_kernel("kq_qk_norm_rope_" #type "_" #d, kq_qk_norm_rope, type, d)

instantiate_kq_qk_norm_rope(bfloat16_t, 64)
instantiate_kq_qk_norm_rope(bfloat16_t, 96)
instantiate_kq_qk_norm_rope(bfloat16_t, 128)
instantiate_kq_qk_norm_rope(bfloat16_t, 256)
instantiate_kq_qk_norm_rope(float16_t, 64)
instantiate_kq_qk_norm_rope(float16_t, 96)
instantiate_kq_qk_norm_rope(float16_t, 128)
instantiate_kq_qk_norm_rope(float16_t, 256)
    // clang-format on
