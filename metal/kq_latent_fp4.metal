// clang-format off
// FP4 latent pack / unpack instantiations; see kq_latent_fp4.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_latent_fp4.h"

#define instantiate_kq_latent_fp4(tname, dtype)                                \
  instantiate_kernel("kq_latent_fp4_pack_" #tname, kq_latent_fp4_pack, dtype)  \
  instantiate_kernel("kq_latent_fp4_unpack_" #tname, kq_latent_fp4_unpack, dtype)

instantiate_kq_latent_fp4(float16_t, half);
instantiate_kq_latent_fp4(bfloat16_t, bfloat16_t);
instantiate_kq_latent_fp4(float, float);
// clang-format on
