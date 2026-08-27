// clang-format off
// Low-rank hyper-connection kernel instantiations; see kq_hc_lowrank.h.
// Same include prelude as kq_quantized.metal (the q8_0 wire helpers live
// in kq_quantized.h).
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_hc_lowrank.h"

#define instantiate_kq_hc_lowrank(tname, itype, hname, htype)               \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_norm_" #tname "_" #hname,                              \
      kq_hc_lowrank_norm, itype, htype)                                     \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_front_" #tname "_" #hname,                             \
      kq_hc_lowrank_front, itype, htype)                                    \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_" #tname "_" #hname,                          \
      kq_hc_lowrank_epilogue, itype, htype)

instantiate_kq_hc_lowrank(float, float, bfloat16_t, bfloat16_t)
instantiate_kq_hc_lowrank(float, float, float16_t, float16_t)
instantiate_kq_hc_lowrank(bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t)
instantiate_kq_hc_lowrank(float16_t, float16_t, float16_t, float16_t)
    // clang-format on
