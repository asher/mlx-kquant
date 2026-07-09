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
      "kq_" #codec "_moe_glu_gather_silu_limit_" #type,               \
      kq_ ## codec ## _moe_glu_gather,                                \
      type,                                                           \
      KQ_GLU_ACT_SILU_LIMIT)                                          \
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
// Every op is emitted at NX = 8 (plain name, tuned-equivalent grid), 16
// ("_nx16"), and 32 ("_nx32"); the wide K-lane variants trade rows per
// threadgroup for threadgroup count so decode-scale launches fill the GPU.
#define instantiate_kq_ext_uniform_nx(codec, traits, type, nx, sfx)           \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_silu" sfx "_" #type,                      \
      kq_ext_moe_glu_gather, type, traits, KQ_GLU_ACT_SILU, nx)                \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_gelu" sfx "_" #type,                      \
      kq_ext_moe_glu_gather, type, traits, KQ_GLU_ACT_GELU, nx)                \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_silu_limit" sfx "_" #type,                \
      kq_ext_moe_glu_gather, type, traits, KQ_GLU_ACT_SILU_LIMIT, nx)          \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv" sfx "_" #type,                               \
      kq_ext_gather_qmv, type, traits, nx)                                     \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_shexp_silu" sfx "_" #type,                \
      kq_ext_moe_glu_gather_shexp, type, traits, traits, KQ_GLU_ACT_SILU, nx)  \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_shexp_gelu" sfx "_" #type,                \
      kq_ext_moe_glu_gather_shexp, type, traits, traits, KQ_GLU_ACT_GELU, nx)  \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_mix" sfx "_" #type,                           \
      kq_ext_gather_qmv_mix, type, traits, traits, nx)

#define instantiate_kq_ext_uniform(codec, traits, type)                       \
  instantiate_kq_ext_uniform_nx(codec, traits, type, 8, "")                    \
  instantiate_kq_ext_uniform_nx(codec, traits, type, 16, "_nx16")              \
  instantiate_kq_ext_uniform_nx(codec, traits, type, 32, "_nx32")

// Mixed-codec shared expert: shexp tensors in scodec over codec expert
// stacks (UD-style upcast shexp). Dispatch key "kq_<codec>_sx_<scodec>_...".
#define instantiate_kq_ext_sx_nx(codec, traits, scodec, straits, type, nx, sfx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_moe_glu_gather_shexp_silu" sfx "_" #type, \
      kq_ext_moe_glu_gather_shexp, type, traits, straits, KQ_GLU_ACT_SILU, nx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_moe_glu_gather_shexp_gelu" sfx "_" #type, \
      kq_ext_moe_glu_gather_shexp, type, traits, straits, KQ_GLU_ACT_GELU, nx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_gather_qmv_mix" sfx "_" #type,            \
      kq_ext_gather_qmv_mix, type, traits, straits, nx)

#define instantiate_kq_ext_sx(codec, traits, scodec, straits, type)           \
  instantiate_kq_ext_sx_nx(codec, traits, scodec, straits, type, 8, "")        \
  instantiate_kq_ext_sx_nx(codec, traits, scodec, straits, type, 16, "_nx16")  \
  instantiate_kq_ext_sx_nx(codec, traits, scodec, straits, type, 32, "_nx32")

// No-shared-expert mix (gemma-style weighted sum; always the generic kernel,
// q6_k/q8_0 included).
#define instantiate_kq_ext_mix_ns_nx(codec, traits, nx, sfx)                  \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_mix_ns" sfx "_bfloat16_t",                    \
      kq_ext_gather_qmv_mix_ns, bfloat16_t, traits, nx)                        \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_mix_ns" sfx "_float16_t",                     \
      kq_ext_gather_qmv_mix_ns, float16_t, traits, nx)

#define instantiate_kq_ext_mix_ns(codec, traits)                              \
  instantiate_kq_ext_mix_ns_nx(codec, traits, 8, "")                           \
  instantiate_kq_ext_mix_ns_nx(codec, traits, 16, "_nx16")                     \
  instantiate_kq_ext_mix_ns_nx(codec, traits, 32, "_nx32")

// Biased experts (gpt-oss): per-(expert, out_dim) f32 biases fused into the
// GLU epilogue / qmv store. Only the clamped-SwiGLU epilogue is emitted --
// biased expert checkpoints all use it.
#define instantiate_kq_ext_bias_nx(codec, traits, type, nx, sfx)               \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_bias_swiglu_clamp" sfx "_" #type,         \
      kq_ext_moe_glu_gather_bias, type, traits, KQ_GLU_ACT_SWIGLU_CLAMP, nx)   \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_bias" sfx "_" #type,                          \
      kq_ext_gather_qmv_bias, type, traits, nx)

#define instantiate_kq_ext_bias(codec, traits, type)                           \
  instantiate_kq_ext_bias_nx(codec, traits, type, 8, "")                       \
  instantiate_kq_ext_bias_nx(codec, traits, type, 16, "_nx16")                 \
  instantiate_kq_ext_bias_nx(codec, traits, type, 32, "_nx32")

#define instantiate_kq_ext_all(codec, traits)                                 \
  instantiate_kq_ext_uniform(codec, traits, bfloat16_t)                        \
  instantiate_kq_ext_uniform(codec, traits, float16_t)                         \
  instantiate_kq_ext_mix_ns(codec, traits)                                     \
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
instantiate_kq_ext_all(mxfp4, KqMxfp4Ext)
instantiate_kq_ext_all(nvfp4, KqNvfp4Ext)
instantiate_kq_ext_bias(mxfp4, KqMxfp4Ext, bfloat16_t)
instantiate_kq_ext_bias(mxfp4, KqMxfp4Ext, float16_t)
instantiate_kq_ext_bias(nvfp4, KqNvfp4Ext, bfloat16_t)
instantiate_kq_ext_bias(nvfp4, KqNvfp4Ext, float16_t)

// q6_k/q8_0 experts with the OTHER upcast shexp codec (diagonals covered by
// the tuned kernels).
instantiate_kq_ext_sx(q6_k, KqQ6_KExt, q8_0, KqQ8_0Ext, bfloat16_t)
instantiate_kq_ext_sx(q6_k, KqQ6_KExt, q8_0, KqQ8_0Ext, float16_t)
instantiate_kq_ext_sx(q8_0, KqQ8_0Ext, q6_k, KqQ6_KExt, bfloat16_t)
instantiate_kq_ext_sx(q8_0, KqQ8_0Ext, q6_k, KqQ6_KExt, float16_t)

// q6_k/q8_0 mix_ns (generic; no tuned ns kernels exist).
instantiate_kq_ext_mix_ns(q6_k, KqQ6_KExt)
instantiate_kq_ext_mix_ns(q8_0, KqQ8_0Ext)

// Generic-uniform q6_k/q8_0 stems: the K % 256 != 0 fallback for tuned q8_0
// (32-weight blocks only need K % 32) and the wide-NX dispatch target for
// both tuned codecs (the tuned kernels are NX = 8 only).
instantiate_kq_ext_uniform(q6_k_ext, KqQ6_KExt, bfloat16_t)
instantiate_kq_ext_uniform(q6_k_ext, KqQ6_KExt, float16_t)
instantiate_kq_ext_uniform(q8_0_ext, KqQ8_0Ext, bfloat16_t)
instantiate_kq_ext_uniform(q8_0_ext, KqQ8_0Ext, float16_t)

instantiate_kernel("kq_moe_router_topk_float", kq_moe_router_topk, float)
instantiate_kernel("kq_moe_router_topk_bfloat16_t", kq_moe_router_topk, bfloat16_t)
instantiate_kernel("kq_moe_router_topk_float16_t", kq_moe_router_topk, float16_t)
    // clang-format on
