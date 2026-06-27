// clang-format off
// Vector SDPA kernel instantiations for large head dims. Derived-code
// attribution lives in kq_sdpa.h and mlx_kquant/licenses/.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_sdpa.h"

#define instantiate_kq_sdpa(type, D)                                  \
  instantiate_kernel(                                                 \
      "kq_sdpa_vector_2pass_1_" #type "_" #D,                         \
      kq_sdpa_vector_2pass_1,                                         \
      type,                                                           \
      D)                                                              \
  instantiate_kernel(                                                 \
      "kq_sdpa_vector_2pass_2_" #type "_" #D,                         \
      kq_sdpa_vector_2pass_2,                                         \
      type,                                                           \
      D)

instantiate_kq_sdpa(bfloat16_t, 256)
instantiate_kq_sdpa(bfloat16_t, 512)
instantiate_kq_sdpa(float16_t, 256)
instantiate_kq_sdpa(float16_t, 512)
    // clang-format on
