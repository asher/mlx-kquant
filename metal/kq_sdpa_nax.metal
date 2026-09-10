// clang-format off
// Tensor-op index-gathered attention instantiations; see kq_sdpa_nax.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_sdpa_nax.h"

#define instantiate_kq_sdpa_fa_indexed_nax(type)                       \
  instantiate_kernel(                                                  \
      "kq_sdpa_fa_indexed_nax_2pass_1_" #type,                         \
      kq_sdpa_fa_indexed_nax_2pass_1,                                  \
      type)

instantiate_kq_sdpa_fa_indexed_nax(bfloat16_t)
instantiate_kq_sdpa_fa_indexed_nax(float16_t)
