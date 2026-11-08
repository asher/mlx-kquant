// clang-format off
// Attention with int8 Q K^T instantiations; see kq_sdpa_qk8_nax.h.
// Adapted from MLX (MIT, see mlx_kquant/licenses/mlx-LICENSE).
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_sdpa_qk8_nax.h"

instantiate_kernel("kq_sdpa_qk8_rows_bfloat16_t", kq_sdpa_qk8_rows, bfloat16_t)
instantiate_kernel("kq_sdpa_qk8_nax_bfloat16_t", kq_sdpa_qk8_nax, bfloat16_t)
