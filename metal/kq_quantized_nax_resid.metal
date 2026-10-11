// clang-format off
// NAX qmm_t kernels with a residual epilogue; see kq_quantized_nax_resid.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/kq_quantized_nax_resid.h"

#define instantiate_kquant_nax_resid(codec, type, al, gt, bm, bk)            \
  instantiate_kernel(                                                        \
      "kquant_" #codec "_qmm_t_nax_resid_" #type                             \
          "_bm" #bm "_bn64_bk" #bk "_wm2_wn2_alN_" #al "_gate_" #gt,         \
      kq_##codec##_qmm_t_nax_resid,                                          \
      type,                                                                  \
      al,                                                                    \
      gt,                                                                    \
      bm,                                                                    \
      64,                                                                    \
      bk,                                                                    \
      2,                                                                     \
      2)

#define instantiate_kquant_nax_resid_bm(codec, bm, bk)                       \
  instantiate_kquant_nax_resid(codec, bfloat16_t, true,  true,  bm, bk)      \
  instantiate_kquant_nax_resid(codec, bfloat16_t, false, true,  bm, bk)      \
  instantiate_kquant_nax_resid(codec, bfloat16_t, true,  false, bm, bk)      \
  instantiate_kquant_nax_resid(codec, bfloat16_t, false, false, bm, bk)

// BM 64 for the codecs whose large-row tile has a K ceiling (q4_k and
// q8_0 above K 10240), BM 128 for every codec.
instantiate_kquant_nax_resid_bm(q4_k, 128, 128)
instantiate_kquant_nax_resid_bm(q4_k, 64, 128)
instantiate_kquant_nax_resid_bm(q5_k, 128, 128)
instantiate_kquant_nax_resid_bm(q6_k, 128, 128)
instantiate_kquant_nax_resid_bm(q8_0, 128, 64)
instantiate_kquant_nax_resid_bm(q8_0, 64, 64)
    // clang-format on
