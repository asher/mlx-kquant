// clang-format off
// Low-rank hyper-connection kernels for weights of any codec; see
// kq_hc_ext.h. The Ext codec traits come with kq_moe_glu_kq.h's includes.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"
#include "mlx/backend/metal/kernels/kq_hc_lowrank.h"
#include "mlx/backend/metal/kernels/kq_hc_ext.h"

// Kernel names "kq_hc_<part>_ext_<codec>_<dtypes>": the front and epilogue
// by residual and half dtype, the inject by residual and output dtype.
#define instantiate_kq_hc_ext_th(codec, traits, tname, itype, hname, htype)  \
  instantiate_kernel(                                                       \
      "kq_hc_front_ext_" #codec "_" #tname "_" #hname,                      \
      kq_hc_front_ext, itype, htype, traits, 16)                            \
  instantiate_kernel(                                                       \
      "kq_hc_epilogue_ext_" #codec "_" #tname "_" #hname,                   \
      kq_hc_epilogue_ext, itype, htype, traits, 4)

#define instantiate_kq_hc_ext_inj(codec, traits, tname, itype, iname, otype) \
  instantiate_kernel(                                                       \
      "kq_hc_inject_ext_" #codec "_" #tname "_" #iname,                     \
      kq_hc_inject_ext, itype, otype, traits)

#define instantiate_kq_hc_ext(codec, traits)                                   \
  instantiate_kq_hc_ext_th(codec, traits, float, float, bfloat16_t, bfloat16_t) \
  instantiate_kq_hc_ext_th(codec, traits, float, float, float16_t, float16_t)   \
  instantiate_kq_hc_ext_th(codec, traits, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t) \
  instantiate_kq_hc_ext_th(codec, traits, float16_t, float16_t, float16_t, float16_t) \
  instantiate_kq_hc_ext_inj(codec, traits, float, float, float, float)           \
  instantiate_kq_hc_ext_inj(codec, traits, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t) \
  instantiate_kq_hc_ext_inj(codec, traits, float16_t, float16_t, float16_t, float16_t)

instantiate_kq_hc_ext(q2_k, KqQ2_KExt)
instantiate_kq_hc_ext(q3_k, KqQ3_KExt)
instantiate_kq_hc_ext(q4_k, KqQ4_KExt)
instantiate_kq_hc_ext(q5_k, KqQ5_KExt)
instantiate_kq_hc_ext(q6_k, KqQ6_KExt)
instantiate_kq_hc_ext(q8_0, KqQ8_0Ext)
instantiate_kq_hc_ext(q4_0, KqQ4_0Ext)
instantiate_kq_hc_ext(q4_1, KqQ4_1Ext)
instantiate_kq_hc_ext(q5_0, KqQ5_0Ext)
instantiate_kq_hc_ext(q5_1, KqQ5_1Ext)
instantiate_kq_hc_ext(iq4_nl, KqIq4_nlExt)
instantiate_kq_hc_ext(iq4_xs, KqIq4_xsExt)
instantiate_kq_hc_ext(iq3_s, KqIq3_sExt)
instantiate_kq_hc_ext(iq3_xxs, KqIq3_xxsExt)
instantiate_kq_hc_ext(iq2_xxs, KqIq2_xxsExt)
instantiate_kq_hc_ext(iq2_xs, KqIq2_xsExt)
instantiate_kq_hc_ext(iq2_s, KqIq2_sExt)
instantiate_kq_hc_ext(iq1_s, KqIq1_sExt)
instantiate_kq_hc_ext(iq1_m, KqIq1_mExt)
instantiate_kq_hc_ext(stq1_0, KqStq1_0Ext)
instantiate_kq_hc_ext(pq2_0, KqPq2_0Ext)
instantiate_kq_hc_ext(q2_0, KqQ2_0Ext)
instantiate_kq_hc_ext(ptq1_0, KqPtq1_0Ext)
instantiate_kq_hc_ext(mxfp4, KqMxfp4Ext)
instantiate_kq_hc_ext(f32, KqHcFloat<float>)
instantiate_kq_hc_ext(f16, KqHcFloat<float16_t>)
instantiate_kq_hc_ext(bf16, KqHcFloat<bfloat16_t>)

// A float inject weight keeps the eager promotion: float32 over a half
// residual, and float32 for float16 against bfloat16.
instantiate_kq_hc_ext_inj(f32, KqHcFloat<float>, bfloat16_t, bfloat16_t, float, float)
instantiate_kq_hc_ext_inj(f32, KqHcFloat<float>, float16_t, float16_t, float, float)
instantiate_kq_hc_ext_inj(f16, KqHcFloat<float16_t>, bfloat16_t, bfloat16_t, float, float)
instantiate_kq_hc_ext_inj(bf16, KqHcFloat<bfloat16_t>, float16_t, float16_t, float, float)
    // clang-format on
