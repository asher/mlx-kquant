// clang-format off
// Fused indexer QAT round-trip instantiations; see kq_dsa_qat.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_dsa_qat.h"

#define instantiate_kq_dsa_indexer_qat(tname, dtype)                    \
  instantiate_kernel("kq_dsa_indexer_qat_" #tname, kq_dsa_indexer_qat, dtype)

instantiate_kq_dsa_indexer_qat(float16_t, half);
instantiate_kq_dsa_indexer_qat(bfloat16_t, bfloat16_t);
instantiate_kq_dsa_indexer_qat(float, float);

#define instantiate_kq_dsa_indexer_qat_quant(tname, dtype) \
  instantiate_kernel(                                      \
      "kq_dsa_indexer_qat_quant_" #tname, kq_dsa_indexer_qat_quant, dtype)

instantiate_kq_dsa_indexer_qat_quant(float16_t, half);
instantiate_kq_dsa_indexer_qat_quant(bfloat16_t, bfloat16_t);
instantiate_kq_dsa_indexer_qat_quant(float, float);
// clang-format on

// clang-format off
#define instantiate_kq_dsa_kv_qat(tname, dtype)                         \
  instantiate_kernel("kq_dsa_kv_qat_" #tname, kq_dsa_kv_qat, dtype)

instantiate_kq_dsa_kv_qat(float16_t, half);
instantiate_kq_dsa_kv_qat(bfloat16_t, bfloat16_t);
instantiate_kq_dsa_kv_qat(float, float);
// clang-format on
