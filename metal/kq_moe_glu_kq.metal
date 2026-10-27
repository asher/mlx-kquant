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

// Fine-tiled single-stream tuned kernels: results_per_simdgroup = 1 -> 2
// output rows per (32, 2, 1) threadgroup, 4x the threadgroups. Bit-exact vs
// the default (per-lane K-chain and simd_sum order unchanged).
#define instantiate_kq_moe_glu_kq_fine(codec, type)                    \
  instantiate_kernel(                                                 \
      "kq_" #codec "_gather_qmv_fine_" #type,                         \
      kq_ ## codec ## _gather_qmv,                                    \
      type,                                                           \
      1)                                                              \
  instantiate_kernel(                                                 \
      "kq_" #codec "_gather_qmv_mix_fine_" #type,                     \
      kq_ ## codec ## _gather_qmv_mix,                                \
      type,                                                           \
      1)

instantiate_kq_moe_glu_kq_fine(q6_k, bfloat16_t)
instantiate_kq_moe_glu_kq_fine(q6_k, float16_t)
instantiate_kq_moe_glu_kq_fine(q8_0, bfloat16_t)
instantiate_kq_moe_glu_kq_fine(q8_0, float16_t)

// The generic Ext-trait instantiation macros are in kq_moe_glu_kq.h, which
// kq_moe_glu_kq_os.metal shares for the nvfp4 kernels.

instantiate_kq_ext_all(q2_k, KqQ2_KExt)
instantiate_kq_ext_all(q3_k, KqQ3_KExt)
instantiate_kq_ext_all(q4_k, KqQ4_KExt)
instantiate_kq_ext_uniform(q5_k, KqQ5_KExt, bfloat16_t)
instantiate_kq_ext_uniform(q5_k, KqQ5_KExt, float16_t)
instantiate_kq_ext_mix_ns(q5_k, KqQ5_KExt)
instantiate_kq_ext_sx(q5_k, KqQ5_KExt, q6_k, KqQ6_KExt, bfloat16_t)
instantiate_kq_ext_sx(q5_k, KqQ5_KExt, q6_k, KqQ6_KExt, float16_t)
instantiate_kq_ext_sx(q5_k, KqQ5_KExt, q8_0, KqQ8_0Ext, bfloat16_t)
instantiate_kq_ext_sx(q5_k, KqQ5_KExt, q8_0, KqQ8_0Ext, float16_t)
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
instantiate_kq_ext_all(stq1_0, KqStq1_0Ext)
instantiate_kq_ext_all(pq2_0, KqPq2_0Ext)
instantiate_kq_ext_all(q2_0, KqQ2_0Ext)
instantiate_kq_ext_all(ptq1_0, KqPtq1_0Ext)
instantiate_kq_ext_all(mxfp4, KqMxfp4Ext)
instantiate_kq_ext_bias(mxfp4, KqMxfp4Ext, bfloat16_t)
instantiate_kq_ext_bias(mxfp4, KqMxfp4Ext, float16_t)

// q6_k/q8_0 experts with the OTHER upcast shexp codec (diagonals covered by
// the tuned kernels).
instantiate_kq_ext_sx(q6_k, KqQ6_KExt, q8_0, KqQ8_0Ext, bfloat16_t)
instantiate_kq_ext_sx(q6_k, KqQ6_KExt, q8_0, KqQ8_0Ext, float16_t)
instantiate_kq_ext_sx(q8_0, KqQ8_0Ext, q6_k, KqQ6_KExt, bfloat16_t)
instantiate_kq_ext_sx(q8_0, KqQ8_0Ext, q6_k, KqQ6_KExt, float16_t)
instantiate_kq_ext_sx_q5k(q6_k, KqQ6_KExt, bfloat16_t)
instantiate_kq_ext_sx_q5k(q6_k, KqQ6_KExt, float16_t)
instantiate_kq_ext_sx_q5k(q8_0, KqQ8_0Ext, bfloat16_t)
instantiate_kq_ext_sx_q5k(q8_0, KqQ8_0Ext, float16_t)

// Split-codec shared expert (kq_ext_moe_glu_gather_shexp2): the per-tensor
// codec picks of the low-bit i-quant builds, where the shared expert's gate
// and up are each one of iq4_xs / iq3_s / q6_k over iq2_s / iq2_xxs / iq1_m
// expert stacks. Dispatch key "kq_<codec>_sx2_<gate>_<up>_...".
#define instantiate_kq_ext_sx2_nx(codec, traits, sg, sgt, su, sut, type, nx, sfx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx2_" #sg "_" #su "_moe_glu_gather_shexp_silu" sfx "_" #type, \
      kq_ext_moe_glu_gather_shexp2, type, traits, sgt, sut, KQ_GLU_ACT_SILU, nx)

#define instantiate_kq_ext_sx2(codec, traits, sg, sgt, su, sut)               \
  instantiate_kq_ext_sx2_nx(codec, traits, sg, sgt, su, sut, bfloat16_t, 8, "") \
  instantiate_kq_ext_sx2_nx(codec, traits, sg, sgt, su, sut, bfloat16_t, 16, "_nx16") \
  instantiate_kq_ext_sx2_nx(codec, traits, sg, sgt, su, sut, bfloat16_t, 32, "_nx32") \
  instantiate_kq_ext_sx2_nx(codec, traits, sg, sgt, su, sut, float16_t, 8, "") \
  instantiate_kq_ext_sx2_nx(codec, traits, sg, sgt, su, sut, float16_t, 16, "_nx16") \
  instantiate_kq_ext_sx2_nx(codec, traits, sg, sgt, su, sut, float16_t, 32, "_nx32")

// q6_k / q6_k is the upcast pair above.
#define instantiate_kq_ext_sx2_lowbit(codec, traits)                          \
  instantiate_kq_ext_sx2(codec, traits, iq4_xs, KqIq4_xsExt, iq4_xs, KqIq4_xsExt) \
  instantiate_kq_ext_sx2(codec, traits, iq4_xs, KqIq4_xsExt, iq3_s, KqIq3_sExt) \
  instantiate_kq_ext_sx2(codec, traits, iq4_xs, KqIq4_xsExt, q6_k, KqQ6_KExt)  \
  instantiate_kq_ext_sx2(codec, traits, iq3_s, KqIq3_sExt, iq4_xs, KqIq4_xsExt) \
  instantiate_kq_ext_sx2(codec, traits, iq3_s, KqIq3_sExt, iq3_s, KqIq3_sExt)  \
  instantiate_kq_ext_sx2(codec, traits, iq3_s, KqIq3_sExt, q6_k, KqQ6_KExt)    \
  instantiate_kq_ext_sx2(codec, traits, q6_k, KqQ6_KExt, iq4_xs, KqIq4_xsExt)  \
  instantiate_kq_ext_sx2(codec, traits, q6_k, KqQ6_KExt, iq3_s, KqIq3_sExt)

instantiate_kq_ext_sx2_lowbit(iq2_s, KqIq2_sExt)
instantiate_kq_ext_sx2_lowbit(iq2_xxs, KqIq2_xxsExt)
instantiate_kq_ext_sx2_lowbit(iq1_m, KqIq1_mExt)

// Down mix for the same builds: q2_0 expert stacks with an iq4_nl shared
// expert. Mix kernels only; the pair has no GLU gather.
#define instantiate_kq_ext_sx_mix_nx(codec, traits, scodec, straits, type, nx, sfx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_gather_qmv_mix" sfx "_" #type,            \
      kq_ext_gather_qmv_mix, type, traits, straits, nx)                        \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_gather_qmv_mix_dd" sfx "_" #type,         \
      kq_ext_gather_qmv_mix_pair, type, traits, straits, true, nx)

#define instantiate_kq_ext_sx_mix(codec, traits, scodec, straits, type)       \
  instantiate_kq_ext_sx_mix_nx(codec, traits, scodec, straits, type, 8, "")    \
  instantiate_kq_ext_sx_mix_nx(codec, traits, scodec, straits, type, 16, "_nx16") \
  instantiate_kq_ext_sx_mix_nx(codec, traits, scodec, straits, type, 32, "_nx32") \
  instantiate_kq_ext_mix_sp(#codec "_sx_" #scodec, traits, straits, type)

instantiate_kq_ext_sx_mix(q2_0, KqQ2_0Ext, iq4_nl, KqIq4_nlExt, bfloat16_t)
instantiate_kq_ext_sx_mix(q2_0, KqQ2_0Ext, iq4_nl, KqIq4_nlExt, float16_t)

// 32-weight decode step (KqExtC32) for the uniform gathers at decode.
#define instantiate_kq_ext_c32_nx(codec, traits, type, nx, sfx)               \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_silu_c32" sfx "_" #type,                  \
      kq_ext_moe_glu_gather, type, traits, KQ_GLU_ACT_SILU, nx, true)          \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_gelu_c32" sfx "_" #type,                  \
      kq_ext_moe_glu_gather, type, traits, KQ_GLU_ACT_GELU, nx, true)          \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_silu_limit_c32" sfx "_" #type,            \
      kq_ext_moe_glu_gather, type, traits, KQ_GLU_ACT_SILU_LIMIT, nx, true)    \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_c32" sfx "_" #type,                           \
      kq_ext_gather_qmv, type, traits, nx, true)

// Shared-expert GLU and shared-expert mix with the shexp in scodec (the
// codec itself or an upcast q5_k, q6_k or q8_0); stem is "codec" or
// "codec_sx_scodec".
#define instantiate_kq_ext_c32_sx_nx(stem, traits, straits, type, nx, sfx)    \
  instantiate_kernel(                                                         \
      "kq_" stem "_moe_glu_gather_shexp_silu_c32" sfx "_" #type,              \
      kq_ext_moe_glu_gather_shexp, type, traits, straits, KQ_GLU_ACT_SILU, nx, \
      true)                                                                    \
  instantiate_kernel(                                                         \
      "kq_" stem "_moe_glu_gather_shexp_gelu_c32" sfx "_" #type,              \
      kq_ext_moe_glu_gather_shexp, type, traits, straits, KQ_GLU_ACT_GELU, nx, \
      true)                                                                    \
  instantiate_kernel(                                                         \
      "kq_" stem "_gather_qmv_mix_c32" sfx "_" #type,                         \
      kq_ext_gather_qmv_mix, type, traits, straits, nx, true)

#define instantiate_kq_ext_c32_sx(stem, traits, straits, type)                \
  instantiate_kq_ext_c32_sx_nx(stem, traits, straits, type, 8, "")            \
  instantiate_kq_ext_c32_sx_nx(stem, traits, straits, type, 16, "_nx16")      \
  instantiate_kernel(                                                         \
      "kq_" stem "_gather_qmv_mix_sp_c32_" #type,                             \
      kq_ext_gather_qmv_mix_sp, type, traits, straits, 8, true)

#define instantiate_kq_ext_c32_type(codec, traits, type)                      \
  instantiate_kq_ext_c32_nx(codec, traits, type, 8, "")                        \
  instantiate_kq_ext_c32_nx(codec, traits, type, 16, "_nx16")                  \
  instantiate_kernel(                                                         \
      "kq_" #codec "_qmv_c32_" #type, kq_ext_qmv_c32, type, traits, 8)         \
  instantiate_kernel(                                                         \
      "kq_" #codec "_qmv_c32_nx16_" #type, kq_ext_qmv_c32, type, traits, 16)   \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_mix_ns_c32_" #type,                           \
      kq_ext_gather_qmv_mix_ns, type, traits, 8, true)                         \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_mix_ns_c32_nx16_" #type,                      \
      kq_ext_gather_qmv_mix_ns, type, traits, 16, true)                        \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_mix_ns_sp_c32_" #type,                        \
      kq_ext_gather_qmv_mix_ns_sp, type, traits, 8, true)                      \
  instantiate_kq_ext_c32_sx(#codec, traits, traits, type)                      \
  instantiate_kq_ext_c32_sx(#codec "_sx_q5_k", traits, KqQ5_KExt, type)        \
  instantiate_kq_ext_c32_sx(#codec "_sx_q6_k", traits, KqQ6_KExt, type)        \
  instantiate_kq_ext_c32_sx(#codec "_sx_q8_0", traits, KqQ8_0Ext, type)

#define instantiate_kq_ext_c32(codec, traits)                                 \
  instantiate_kq_ext_c32_type(codec, traits, bfloat16_t)                       \
  instantiate_kq_ext_c32_type(codec, traits, float16_t)

instantiate_kq_ext_c32(iq2_s, KqIq2_sExt)
instantiate_kq_ext_c32(iq2_xxs, KqIq2_xxsExt)
instantiate_kq_ext_c32(iq2_xs, KqIq2_xsExt)
instantiate_kq_ext_c32(iq3_xxs, KqIq3_xxsExt)
instantiate_kq_ext_c32(iq3_s, KqIq3_sExt)
instantiate_kq_ext_c32(iq1_s, KqIq1_sExt)
instantiate_kq_ext_c32(iq1_m, KqIq1_mExt)

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

// Half-dot variants (KQ_MOE_HALF=1) for the grid codecs whose float decode
// is ALU-bound at decode widths: iq2_xs, iq2_xxs, iq3_xxs. Names insert
// "_h" after the op stem. nx 8 and 16 only (the host caps the pick).
#define instantiate_kq_ext_half_nx(codec, traits, type, nx, sfx)              \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_h_silu" sfx "_" #type,                    \
      kq_ext_moe_glu_gather_h, type, traits, KQ_GLU_ACT_SILU, nx)              \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_h_gelu" sfx "_" #type,                    \
      kq_ext_moe_glu_gather_h, type, traits, KQ_GLU_ACT_GELU, nx)              \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_h_silu_limit" sfx "_" #type,              \
      kq_ext_moe_glu_gather_h, type, traits, KQ_GLU_ACT_SILU_LIMIT, nx)        \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_h" sfx "_" #type,                             \
      kq_ext_gather_qmv_h, type, traits, nx)                                   \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_shexp_h_silu" sfx "_" #type,              \
      kq_ext_moe_glu_gather_shexp_h, type, traits, traits, KQ_GLU_ACT_SILU, nx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_moe_glu_gather_shexp_h_gelu" sfx "_" #type,              \
      kq_ext_moe_glu_gather_shexp_h, type, traits, traits, KQ_GLU_ACT_GELU, nx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_gather_qmv_mix_h" sfx "_" #type,                         \
      kq_ext_gather_qmv_mix_h, type, traits, traits, nx)

#define instantiate_kq_ext_half_sx_nx(codec, traits, scodec, straits, type, nx, sfx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_moe_glu_gather_shexp_h_silu" sfx "_" #type, \
      kq_ext_moe_glu_gather_shexp_h, type, traits, straits, KQ_GLU_ACT_SILU, nx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_moe_glu_gather_shexp_h_gelu" sfx "_" #type, \
      kq_ext_moe_glu_gather_shexp_h, type, traits, straits, KQ_GLU_ACT_GELU, nx) \
  instantiate_kernel(                                                         \
      "kq_" #codec "_sx_" #scodec "_gather_qmv_mix_h" sfx "_" #type,          \
      kq_ext_gather_qmv_mix_h, type, traits, straits, nx)

#define instantiate_kq_ext_half_type(codec, traits, type)                     \
  instantiate_kq_ext_half_nx(codec, traits, type, 8, "")                       \
  instantiate_kq_ext_half_nx(codec, traits, type, 16, "_nx16")                 \
  instantiate_kq_ext_half_sx_nx(codec, traits, q5_k, KqQ5_KExt, type, 8, "")   \
  instantiate_kq_ext_half_sx_nx(codec, traits, q5_k, KqQ5_KExt, type, 16, "_nx16") \
  instantiate_kq_ext_half_sx_nx(codec, traits, q6_k, KqQ6_KExt, type, 8, "")   \
  instantiate_kq_ext_half_sx_nx(codec, traits, q6_k, KqQ6_KExt, type, 16, "_nx16") \
  instantiate_kq_ext_half_sx_nx(codec, traits, q8_0, KqQ8_0Ext, type, 8, "")   \
  instantiate_kq_ext_half_sx_nx(codec, traits, q8_0, KqQ8_0Ext, type, 16, "_nx16")

#define instantiate_kq_ext_half_all(codec, traits)                            \
  instantiate_kq_ext_half_type(codec, traits, bfloat16_t)                      \
  instantiate_kq_ext_half_type(codec, traits, float16_t)

instantiate_kq_ext_half_all(iq2_xs, KqIq2_xsExt)
instantiate_kq_ext_half_all(iq2_xxs, KqIq2_xxsExt)
instantiate_kq_ext_half_all(iq3_xxs, KqIq3_xxsExt)

instantiate_kernel("kq_moe_router_topk_float", kq_moe_router_topk, float)
instantiate_kernel("kq_moe_router_topk_bfloat16_t", kq_moe_router_topk, bfloat16_t)
instantiate_kernel("kq_moe_router_topk_float16_t", kq_moe_router_topk, float16_t)
    // clang-format on
