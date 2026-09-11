// clang-format off
// Tensor-op chunked KDA prefill instantiations; see kq_kda_chunk_nax.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_kda_chunk_nax.h"

#define instantiate_kq_kda_chunk_nax(type) \
  instantiate_kernel("kq_kda_chunk_nax_" #type, kq_kda_chunk_nax, type, false) \
  instantiate_kernel("kq_kda_chunk_nax_gated_" #type, kq_kda_chunk_nax, type, true)

instantiate_kq_kda_chunk_nax(bfloat16_t)
instantiate_kq_kda_chunk_nax(float16_t)
