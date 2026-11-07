// clang-format off
// MiniMax-H3 VSA block-sparse attention instantiations; see kq_vsa_nax.h.
// Ported from FastVideo (Apache-2.0, see mlx_kquant/licenses/FastVideo-LICENSE)
// and adapted there from MLX (MIT, see mlx_kquant/licenses/mlx-LICENSE).
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_vsa_nax.h"

#define instantiate_kq_vsa_nax(type, tile, bq)                         \
  instantiate_kernel(                                                  \
      "kq_vsa_nax_" #type "_" #tile "_bq" #bq,                         \
      kq_vsa_nax,                                                      \
      type, tile, bq, false)                                           \
  instantiate_kernel(                                                  \
      "kq_vsa_nax_gated_" #type "_" #tile "_bq" #bq,                   \
      kq_vsa_nax,                                                      \
      type, tile, bq, true)

instantiate_kq_vsa_nax(bfloat16_t, 64, 16)
instantiate_kq_vsa_nax(bfloat16_t, 64, 32)
instantiate_kq_vsa_nax(bfloat16_t, 64, 64)
