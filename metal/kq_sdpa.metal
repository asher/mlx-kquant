// clang-format off
// Vector SDPA kernel instantiations for large head dims. Derived-code
// attribution lives in kq_sdpa.h and mlx_kquant/licenses/.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/loader.h"
#include "mlx/backend/metal/kernels/steel/attn/mma.h"
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

#define instantiate_kq_sdpa_gqa(type, D, C)                           \
  instantiate_kernel(                                                 \
      "kq_sdpa_gqa_2pass_1_" #type "_" #D "_c" #C,                    \
      kq_sdpa_gqa_2pass_1,                                            \
      type,                                                           \
      D,                                                              \
      C)

// NE (keys in flight per simdgroup) override: wide head dims halve NE so the
// per-lane query/output register slices stay at <= 64 floats.
#define instantiate_kq_sdpa_gqa_ne(type, D, C, NE)                    \
  instantiate_kernel(                                                 \
      "kq_sdpa_gqa_2pass_1_" #type "_" #D "_c" #C,                    \
      kq_sdpa_gqa_2pass_1,                                            \
      type,                                                           \
      D,                                                              \
      C,                                                              \
      NE)

#define instantiate_kq_sdpa_gqa_merge(type, D)                        \
  instantiate_kernel(                                                 \
      "kq_sdpa_gqa_2pass_2_" #type "_" #D,                            \
      kq_sdpa_gqa_2pass_2,                                            \
      type,                                                           \
      D)

// Tile C is bounded by threadgroup memory (2 * C * D * sizeof(T4)/4 bytes;
// 32 KB limit): D=64/128 stage 32 or 16 keys, D=256 stages 16 or 8.
instantiate_kq_sdpa_gqa(bfloat16_t, 64, 32)
instantiate_kq_sdpa_gqa(bfloat16_t, 64, 16)
instantiate_kq_sdpa_gqa(float16_t, 64, 32)
instantiate_kq_sdpa_gqa(float16_t, 64, 16)
instantiate_kq_sdpa_gqa(bfloat16_t, 128, 32)
instantiate_kq_sdpa_gqa(bfloat16_t, 128, 16)
instantiate_kq_sdpa_gqa(float16_t, 128, 32)
instantiate_kq_sdpa_gqa(float16_t, 128, 16)
instantiate_kq_sdpa_gqa(bfloat16_t, 256, 16)
instantiate_kq_sdpa_gqa(bfloat16_t, 256, 8)
instantiate_kq_sdpa_gqa(float16_t, 256, 16)
instantiate_kq_sdpa_gqa(float16_t, 256, 8)
instantiate_kq_sdpa_gqa_ne(bfloat16_t, 512, 8, 2)
instantiate_kq_sdpa_gqa_ne(float16_t, 512, 8, 2)

// Verify-width (qL 2..4) pair variants: two queries per simdgroup, so each
// staged element read feeds two dots (threadgroup-memory traffic halves) at
// two query/output register sets per thread. NE mirrors the qL==1 choice.
#define instantiate_kq_sdpa_gqa_p2(type, D, C, NE)                    \
  instantiate_kernel(                                                 \
      "kq_sdpa_gqa_2pass_1_" #type "_" #D "_c" #C "_p2",              \
      kq_sdpa_gqa_2pass_1,                                            \
      type,                                                           \
      D,                                                              \
      C,                                                              \
      NE,                                                             \
      2)

instantiate_kq_sdpa_gqa_p2(bfloat16_t, 512, 8, 2)
instantiate_kq_sdpa_gqa_p2(float16_t, 512, 8, 2)
instantiate_kq_sdpa_gqa_p2(bfloat16_t, 256, 16, 4)
instantiate_kq_sdpa_gqa_p2(float16_t, 256, 16, 4)
instantiate_kq_sdpa_gqa_p2(bfloat16_t, 128, 32, 4)
instantiate_kq_sdpa_gqa_p2(float16_t, 128, 32, 4)
instantiate_kq_sdpa_gqa_p2(bfloat16_t, 64, 32, 4)
instantiate_kq_sdpa_gqa_p2(float16_t, 64, 32, 4)

// Simdgroup-matrix FA verify pass 1 (folded GQA, one 32-row Q tile); the
// merge reuses kq_sdpa_gqa_2pass_2. head_dim 256 only for now: at 512 the
// Q + O fragment sets are ~256 floats/thread, so a 512 instantiation needs a
// BD-chunked (two-sweep) output accumulator first.
#define instantiate_kq_sdpa_fa_verify(type, D)                         \
  instantiate_kernel(                                                  \
      "kq_sdpa_fa_verify_2pass_1_" #type "_" #D,                       \
      kq_sdpa_fa_verify_2pass_1,                                       \
      type,                                                            \
      D)

instantiate_kq_sdpa_fa_verify(bfloat16_t, 256)
instantiate_kq_sdpa_fa_verify(float16_t, 256)

instantiate_kq_sdpa_gqa_merge(bfloat16_t, 64)
instantiate_kq_sdpa_gqa_merge(float16_t, 64)
instantiate_kq_sdpa_gqa_merge(bfloat16_t, 128)
instantiate_kq_sdpa_gqa_merge(float16_t, 128)
instantiate_kq_sdpa_gqa_merge(bfloat16_t, 256)
instantiate_kq_sdpa_gqa_merge(float16_t, 256)
instantiate_kq_sdpa_gqa_merge(bfloat16_t, 512)
instantiate_kq_sdpa_gqa_merge(float16_t, 512)
    // clang-format on
