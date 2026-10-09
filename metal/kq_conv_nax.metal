// clang-format off
// NAX 3x3 convolution as an implicit GEMM; see kq_conv_nax.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/kq_conv_nax.h"

#define instantiate_kq_conv3x3_nax(type, bm, bn, wm, wn)                      \
  instantiate_kernel(                                                        \
      "kq_conv3x3_nax_" #type "_bm" #bm "_bn" #bn "_wm" #wm "_wn" #wn,       \
      kq_conv3x3_nax,                                                        \
      type,                                                                  \
      bm,                                                                    \
      bn,                                                                    \
      wm,                                                                    \
      wn)

instantiate_kq_conv3x3_nax(bfloat16_t, 128, 128, 4, 4)
instantiate_kq_conv3x3_nax(bfloat16_t, 128, 64, 4, 2)
instantiate_kq_conv3x3_nax(bfloat16_t, 64, 128, 2, 4)
instantiate_kq_conv3x3_nax(bfloat16_t, 64, 64, 2, 2)
    // clang-format on
