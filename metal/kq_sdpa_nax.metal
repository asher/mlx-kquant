// clang-format off
// Tensor-op index-gathered attention instantiations; see kq_sdpa_nax.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_sdpa_nax.h"

#define instantiate_kq_sdpa_fa_indexed_nax(type)                       \
  instantiate_kernel(                                                  \
      "kq_sdpa_fa_indexed_nax_2pass_1_" #type,                         \
      kq_sdpa_fa_indexed_nax_2pass_1,                                  \
      type)

instantiate_kq_sdpa_fa_indexed_nax(bfloat16_t)
instantiate_kq_sdpa_fa_indexed_nax(float16_t)

#define instantiate_kq_sdpa_gqa_nax(type)                              \
  instantiate_kernel(                                                  \
      "kq_sdpa_gqa_nax_2pass_1_" #type,                                \
      kq_sdpa_gqa_nax_2pass_1,                                         \
      type)

instantiate_kq_sdpa_gqa_nax(bfloat16_t)
instantiate_kq_sdpa_gqa_nax(float16_t)

#define instantiate_kq_sdpa_fa_verify_nax(type, D, WM, WN)             \
  instantiate_kernel(                                                  \
      "kq_sdpa_fa_verify_2pass_1_nax_" #type "_" #D "_wm" #WM "_wn" #WN, \
      kq_sdpa_fa_verify_2pass_1_nax,                                   \
      type, D, WM, WN)

#define instantiate_kq_sdpa_fa_verify_nax_rows(type, D, WN)            \
  instantiate_kq_sdpa_fa_verify_nax(type, D, 1, WN)                    \
  instantiate_kq_sdpa_fa_verify_nax(type, D, 2, WN)                    \
  instantiate_kq_sdpa_fa_verify_nax(type, D, 3, WN)                    \
  instantiate_kq_sdpa_fa_verify_nax(type, D, 4, WN)

instantiate_kq_sdpa_fa_verify_nax_rows(bfloat16_t, 256, 2)
instantiate_kq_sdpa_fa_verify_nax_rows(float16_t, 256, 2)
instantiate_kq_sdpa_fa_verify_nax_rows(bfloat16_t, 256, 4)
instantiate_kq_sdpa_fa_verify_nax_rows(float16_t, 256, 4)
instantiate_kq_sdpa_fa_verify_nax_rows(bfloat16_t, 128, 1)
instantiate_kq_sdpa_fa_verify_nax_rows(float16_t, 128, 1)
instantiate_kq_sdpa_fa_verify_nax_rows(bfloat16_t, 128, 2)
instantiate_kq_sdpa_fa_verify_nax_rows(float16_t, 128, 2)

// sdpa_decode_gqa pass 1 at head_dim 128 under KQ_GQA_NAX_SPLIT_P=1: the
// fa_verify kernel, one row group, two d groups, P @ V on split P. The
// default runs the fa_verify instantiation itself.
#define instantiate_kq_sdpa_gqa_nax_128(type)                          \
  instantiate_kernel(                                                  \
      "kq_sdpa_gqa_nax_2pass_1_" #type "_128",                         \
      kq_sdpa_fa_verify_2pass_1_nax,                                   \
      type, 128, 1, 2, true)

instantiate_kq_sdpa_gqa_nax_128(bfloat16_t)
instantiate_kq_sdpa_gqa_nax_128(float16_t)

// Sliding-window prefill attention; see kq_sdpa_window_nax.h.
#include "mlx/backend/metal/kernels/kq_sdpa_window_nax.h"

#define instantiate_kq_sdpa_window_nax(type, D)                        \
  instantiate_kernel(                                                  \
      "kq_sdpa_window_nax_" #type "_" #D,                              \
      kq_sdpa_window_nax,                                              \
      type, D)

instantiate_kq_sdpa_window_nax(bfloat16_t, 64)
instantiate_kq_sdpa_window_nax(float16_t, 64)
instantiate_kq_sdpa_window_nax(bfloat16_t, 128)
instantiate_kq_sdpa_window_nax(float16_t, 128)
instantiate_kq_sdpa_window_nax(bfloat16_t, 256)
instantiate_kq_sdpa_window_nax(float16_t, 256)
