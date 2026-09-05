// clang-format off
// KVarN quantize/dequant instantiations; see kq_kvarn.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_kvarn.h"

#define instantiate_kq_kvarn_quantize(tname, dtype)                     \
  instantiate_kernel("kq_kvarn_quantize_" #tname, kq_kvarn_quantize, dtype)

instantiate_kq_kvarn_quantize(float16_t, half);

#define instantiate_kq_kvarn_dequant(tname, dtype)                      \
  instantiate_kernel("kq_kvarn_dequant_" #tname, kq_kvarn_dequant, dtype)

instantiate_kq_kvarn_dequant(float16_t, half);
instantiate_kq_kvarn_dequant(bfloat16_t, bfloat16_t);
// clang-format on
