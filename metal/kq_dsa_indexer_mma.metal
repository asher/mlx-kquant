// clang-format off
// Tensor-op lightning-indexer instantiations; see kq_dsa_indexer_mma.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_dsa_indexer_mma.h"

#define instantiate_kq_dsa_indexer_score_mma(tname, dtype) \
  instantiate_kernel(                                      \
      "kq_dsa_indexer_score_mma_" #tname,                 \
      kq_dsa_indexer_score_mma, dtype)

#define instantiate_kq_dsa_indexer_score_i8mx(tname, dtype) \
  instantiate_kernel(                                       \
      "kq_dsa_indexer_score_i8mx_" #tname,                 \
      kq_dsa_indexer_score_i8mx, dtype)

instantiate_kq_dsa_indexer_score_mma(float16_t, half);
instantiate_kq_dsa_indexer_score_mma(bfloat16_t, bfloat16_t);

instantiate_kq_dsa_indexer_score_i8mx(float16_t, half);
instantiate_kq_dsa_indexer_score_i8mx(bfloat16_t, bfloat16_t);
// clang-format on
