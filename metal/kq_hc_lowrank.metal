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
      "kq_hc_lowrank_front_r1_" #tname "_" #hname,                          \
      kq_hc_lowrank_front, itype, htype, 1)                                 \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_front_r2_" #tname "_" #hname,                          \
      kq_hc_lowrank_front, itype, htype, 2)                                 \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_front_r4_" #tname "_" #hname,                          \
      kq_hc_lowrank_front, itype, htype, 4)                                 \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_" #tname "_" #hname,                          \
      kq_hc_lowrank_epilogue, itype, htype)                                 \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr4_r1_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 4, 1) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr4_r2_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 4, 2) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr4_r4_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 4, 4) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr4_r8_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 4, 8) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr8_r1_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 8, 1) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr8_r2_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 8, 2) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr8_r4_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 8, 4) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr8_r8_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 8, 8) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr16_r1_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 16, 1) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr16_r2_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 16, 2) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr16_r4_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 16, 4) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mr16_r8_" #tname "_" #hname,                   \
      kq_hc_lowrank_epilogue_mr, itype, htype, 16, 8)

#define instantiate_kq_hc_lowrank_fn_epi(hname, htype, l, rb)            \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_fn_mr" #l "_r" #rb "_float_" #hname,          \
      kq_hc_lowrank_epilogue_mr, float, htype, l, rb, true)

#define instantiate_kq_hc_lowrank_fn_front(hname, htype, rb, no)         \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_front_fn_r" #rb "_o" #no "_" #hname,                   \
      kq_hc_lowrank_front_fn, htype, rb, no)

#define instantiate_kq_hc_lowrank_fn_front_s(hname, htype, rb, sps)      \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_front_fn_r" #rb "_o4_s" #sps "_" #hname,                \
      kq_hc_lowrank_front_fn, htype, rb, 4, sps)

#define instantiate_kq_hc_lowrank_fn(hname, htype)                         \
  instantiate_kq_hc_lowrank_fn_front_s(hname, htype, 1, 2)                 \
  instantiate_kq_hc_lowrank_fn_front_s(hname, htype, 2, 2)                 \
  instantiate_kq_hc_lowrank_fn_front_s(hname, htype, 1, 4)                 \
  instantiate_kq_hc_lowrank_fn_front_s(hname, htype, 2, 4)                 \
  instantiate_kq_hc_lowrank_fn_front(hname, htype, 1, 4)                   \
  instantiate_kq_hc_lowrank_fn_front(hname, htype, 2, 4)                   \
  instantiate_kq_hc_lowrank_fn_front(hname, htype, 4, 4)                   \
  instantiate_kq_hc_lowrank_fn_front(hname, htype, 8, 4)                   \
  instantiate_kq_hc_lowrank_fn_front(hname, htype, 1, 8)                   \
  instantiate_kq_hc_lowrank_fn_front(hname, htype, 2, 8)                   \
  instantiate_kq_hc_lowrank_fn_front(hname, htype, 4, 8)                   \
  instantiate_kq_hc_lowrank_fn_front(hname, htype, 8, 8)                   \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 4, 1)                     \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 4, 2)                     \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 4, 4)                     \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 4, 8)                     \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 8, 1)                     \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 8, 2)                     \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 8, 4)                     \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 8, 8)                     \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 16, 1)                    \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 16, 2)                    \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 16, 4)                    \
  instantiate_kq_hc_lowrank_fn_epi(hname, htype, 16, 8)

#define instantiate_kq_hc_lowrank_mma(hname, htype)                        \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_front_mma_f1_" #hname, kq_hc_lowrank_front_mma, htype, 1) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_front_mma_f2_" #hname, kq_hc_lowrank_front_mma, htype, 2) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mma_f1_" #hname, kq_hc_lowrank_epilogue_mma, htype, 1) \
  instantiate_kernel(                                                       \
      "kq_hc_lowrank_epilogue_mma_f2_" #hname, kq_hc_lowrank_epilogue_mma, htype, 2)

instantiate_kq_hc_lowrank_mma(bfloat16_t, bfloat16_t)
instantiate_kq_hc_lowrank_mma(float16_t, float16_t)
instantiate_kq_hc_lowrank_fn(bfloat16_t, bfloat16_t)
instantiate_kq_hc_lowrank_fn(float16_t, float16_t)
instantiate_kq_hc_lowrank(float, float, bfloat16_t, bfloat16_t)
instantiate_kq_hc_lowrank(float, float, float16_t, float16_t)
instantiate_kq_hc_lowrank(bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t)
instantiate_kq_hc_lowrank(float16_t, float16_t, float16_t, float16_t)
    // clang-format on
