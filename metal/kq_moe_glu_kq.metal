// clang-format off
// K-quant fused MoE gather kernel instantiations; see kq_moe_glu_kq.h.
// Same include prelude as kq_quantized.metal (the per-codec dequant helpers
// live in kq_quantized*.h).
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"

#define instantiate_kq_moe_glu_kq(codec, type)                        \
  instantiate_kernel(                                                 \
      "kq_" #codec "_moe_glu_gather_silu_" #type,                     \
      kq_ ## codec ## _moe_glu_gather,                                \
      type,                                                           \
      KQ_GLU_ACT_SILU)                                                \
  instantiate_kernel(                                                 \
      "kq_" #codec "_moe_glu_gather_gelu_" #type,                     \
      kq_ ## codec ## _moe_glu_gather,                                \
      type,                                                           \
      KQ_GLU_ACT_GELU)                                                \
  instantiate_kernel(                                                 \
      "kq_" #codec "_gather_qmv_" #type,                              \
      kq_ ## codec ## _gather_qmv,                                    \
      type)                                                           \
  instantiate_kernel(                                                 \
      "kq_" #codec "_moe_glu_gather_shexp_silu_" #type,               \
      kq_ ## codec ## _moe_glu_gather_shexp,                          \
      type,                                                           \
      KQ_GLU_ACT_SILU)                                                \
  instantiate_kernel(                                                 \
      "kq_" #codec "_moe_glu_gather_shexp_gelu_" #type,               \
      kq_ ## codec ## _moe_glu_gather_shexp,                          \
      type,                                                           \
      KQ_GLU_ACT_GELU)                                                \
  instantiate_kernel(                                                 \
      "kq_" #codec "_gather_qmv_mix_" #type,                          \
      kq_ ## codec ## _gather_qmv_mix,                                \
      type)

instantiate_kq_moe_glu_kq(q6_k, bfloat16_t)
instantiate_kq_moe_glu_kq(q6_k, float16_t)
instantiate_kq_moe_glu_kq(q8_0, bfloat16_t)
instantiate_kq_moe_glu_kq(q8_0, float16_t)

// Codec matrix (generic Ext-trait kernels; q6_k/q8_0 uniform stay on the
// tuned kernels above). Names match the tuned scheme so dispatch is uniform.
#define instantiate_kq_ext_uniform(codec, traits, type)                       \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_silu_" #type,                             \
      kq_ext_moe_glu_gather, type, traits, KQ_GLU_ACT_SILU)                    \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_gelu_" #type,                             \
      kq_ext_moe_glu_gather, type, traits, KQ_GLU_ACT_GELU)                    \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_" #type,                                      \
      kq_ext_gather_qmv, type, traits)                                         \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_shexp_silu_" #type,                       \
      kq_ext_moe_glu_gather_shexp, type, traits, traits, KQ_GLU_ACT_SILU)      \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_shexp_gelu_" #type,                       \
      kq_ext_moe_glu_gather_shexp, type, traits, traits, KQ_GLU_ACT_GELU)      \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_mix_" #type,                                  \
      kq_ext_gather_qmv_mix, type, traits, traits)

// Mixed-codec shared expert: shexp tensors in scodec over codec expert
// stacks (UD-style upcast shexp). Dispatch key "kq_<codec>_sx_<scodec>_...".
#define instantiate_kq_ext_sx(codec, traits, scodec, straits, type)           \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_moe_glu_gather_shexp_silu_" #type,        \
      kq_ext_moe_glu_gather_shexp, type, traits, straits, KQ_GLU_ACT_SILU)     \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_moe_glu_gather_shexp_gelu_" #type,        \
      kq_ext_moe_glu_gather_shexp, type, traits, straits, KQ_GLU_ACT_GELU)     \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_gather_qmv_mix_" #type,                   \
      kq_ext_gather_qmv_mix, type, traits, straits)

#define instantiate_kq_ext_all(codec, traits)                                 \
  instantiate_kq_ext_uniform(codec, traits, bfloat16_t)                        \
  instantiate_kq_ext_uniform(codec, traits, float16_t)                         \
  instantiate_kq_ext_sx(codec, traits, q6_k, KqQ6_KExt, bfloat16_t)            \
  instantiate_kq_ext_sx(codec, traits, q6_k, KqQ6_KExt, float16_t)             \
  instantiate_kq_ext_sx(codec, traits, q8_0, KqQ8_0Ext, bfloat16_t)            \
  instantiate_kq_ext_sx(codec, traits, q8_0, KqQ8_0Ext, float16_t)

instantiate_kq_ext_all(q2_k, KqQ2_KExt)
instantiate_kq_ext_all(q3_k, KqQ3_KExt)
instantiate_kq_ext_all(q4_k, KqQ4_KExt)
instantiate_kq_ext_all(q5_k, KqQ5_KExt)
instantiate_kq_ext_all(q4_0, KqQ4_0Ext)
instantiate_kq_ext_all(q4_1, KqQ4_1Ext)
instantiate_kq_ext_all(q5_0, KqQ5_0Ext)
instantiate_kq_ext_all(q5_1, KqQ5_1Ext)
instantiate_kq_ext_all(iq4_nl, KqIq4_nlExt)
instantiate_kq_ext_all(iq4_xs, KqIq4_xsExt)
instantiate_kq_ext_all(iq3_s, KqIq3_sExt)
instantiate_kq_ext_all(iq3_xxs, KqIq3_xxsExt)
instantiate_kq_ext_all(iq2_xxs, KqIq2_xxsExt)
instantiate_kq_ext_all(iq2_xs, KqIq2_xsExt)
instantiate_kq_ext_all(iq2_s, KqIq2_sExt)
instantiate_kq_ext_all(iq1_s, KqIq1_sExt)
instantiate_kq_ext_all(iq1_m, KqIq1_mExt)

// q6_k/q8_0 experts with the OTHER upcast shexp codec (diagonals covered by
// the tuned kernels).
instantiate_kq_ext_sx(q6_k, KqQ6_KExt, q8_0, KqQ8_0Ext, bfloat16_t)
instantiate_kq_ext_sx(q6_k, KqQ6_KExt, q8_0, KqQ8_0Ext, float16_t)
instantiate_kq_ext_sx(q8_0, KqQ8_0Ext, q6_k, KqQ6_KExt, bfloat16_t)
instantiate_kq_ext_sx(q8_0, KqQ8_0Ext, q6_k, KqQ6_KExt, float16_t)

instantiate_kernel("kq_moe_router_topk_float", kq_moe_router_topk, float)
instantiate_kernel("kq_moe_router_topk_bfloat16_t", kq_moe_router_topk, bfloat16_t)
instantiate_kernel("kq_moe_router_topk_float16_t", kq_moe_router_topk, float16_t)
    // clang-format on
