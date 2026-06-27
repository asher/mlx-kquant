// clang-format off
// Kernel instantiations; derived-code attribution lives in the included
// kq_*.h headers and mlx_kquant/licenses/.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"

#define instantiate_kquant_batched(func, type, gs, bits, batched, codec)  \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_" #func "_" #type "_gs_" #gs "_b_" #bits         \
          "_batch_" #batched,                                             \
      kq_ ## codec ## _ ## func,                                          \
      type,                                                               \
      gs,                                                                 \
      bits,                                                               \
      batched)

#define instantiate_kquant_qmm_t(type, gs, bits, aligned_N, batched, codec) \
  instantiate_kernel(                                                       \
      "kquant_" #codec "_qmm_t_" #type "_gs_" #gs "_b_" #bits               \
          "_alN_" #aligned_N "_batch_" #batched,                            \
      kq_ ## codec ## _qmm_t,                                               \
      type,                                                                 \
      gs,                                                                   \
      bits,                                                                 \
      aligned_N,                                                            \
      batched)

#define instantiate_kquant_qmm_t_splitk(type, gs, bits, aligned_N, codec) \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_qmm_t_splitk_" #type "_gs_" #gs "_b_" #bits      \
          "_alN_" #aligned_N,                                             \
      kq_ ## codec ## _qmm_t_splitk,                                      \
      type,                                                               \
      gs,                                                                 \
      bits,                                                               \
      aligned_N)

#define instantiate_kquant_qmm_n(type, gs, bits, batched, codec)        \
  instantiate_kernel(                                                   \
      "kquant_" #codec "_qmm_n_" #type "_gs_" #gs "_b_" #bits           \
          "_batch_" #batched,                                           \
      kq_ ## codec ## _qmm_n,                                           \
      type,                                                             \
      gs,                                                               \
      bits,                                                             \
      batched)

#define instantiate_kquant_gather_qmv(func, type, gs, bits, codec)        \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_" #func "_" #type "_gs_" #gs "_b_" #bits,        \
      kq_ ## codec ## _ ## func,                                          \
      type,                                                               \
      gs,                                                                 \
      bits)

#define instantiate_kquant_gather_qmm_t(type, gs, bits, aligned_N, codec) \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_gather_qmm_t_" #type "_gs_" #gs "_b_" #bits      \
          "_alN_" #aligned_N,                                             \
      kq_ ## codec ## _gather_qmm_t,                                      \
      type,                                                               \
      gs,                                                                 \
      bits,                                                               \
      aligned_N)

#define instantiate_kquant_gather_qmm_n(type, gs, bits, codec)            \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_gather_qmm_n_" #type "_gs_" #gs "_b_" #bits,     \
      kq_ ## codec ## _gather_qmm_n,                                      \
      type,                                                               \
      gs,                                                                 \
      bits)

#define instantiate_kquant_dequantize(type, gs, bits, codec)              \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_dequantize_" #type "_gs_" #gs "_b_" #bits,       \
      kq_ ## codec ## _dequantize,                                        \
      type,                                                               \
      gs,                                                                 \
      bits)

#define instantiate_kquant_q8_0_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 32, 8, 0, q8_0)          \
  instantiate_kquant_batched(verify_qmv_fine, type, 32, 8, 0, q8_0)     \
  instantiate_kquant_batched(qmv_fast, type, 32, 8, 0, q8_0)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 8, 1, q8_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 8, 0, q8_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 8, 1, q8_0)            \
  instantiate_kquant_qmm_t(type, 32, 8, true, 0, q8_0)                  \
  instantiate_kquant_qmm_t(type, 32, 8, true, 1, q8_0)                  \
  instantiate_kquant_qmm_t(type, 32, 8, false, 0, q8_0)                 \
  instantiate_kquant_qmm_t(type, 32, 8, false, 1, q8_0)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 8, true, q8_0)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 8, false, q8_0)             \
  instantiate_kquant_qmm_n(type, 32, 8, 0, q8_0)                        \
  instantiate_kquant_qmm_n(type, 32, 8, 1, q8_0)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 8, q8_0)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 8, q8_0)     \
  instantiate_kquant_gather_qmm_t(type, 32, 8, true, q8_0)              \
  instantiate_kquant_gather_qmm_t(type, 32, 8, false, q8_0)             \
  instantiate_kquant_gather_qmm_n(type, 32, 8, q8_0)                    \
  instantiate_kquant_dequantize(type, 32, 8, q8_0)

instantiate_kquant_q8_0_for_type(float)
instantiate_kquant_q8_0_for_type(bfloat16_t)
instantiate_kquant_q8_0_for_type(float16_t)

#define instantiate_kquant_q5_1_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 32, 5, 0, q5_1)          \
  instantiate_kquant_batched(qmv_fast, type, 32, 5, 0, q5_1)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 5, 1, q5_1)            \
  instantiate_kquant_batched(qmv,      type, 32, 5, 0, q5_1)            \
  instantiate_kquant_batched(qmv,      type, 32, 5, 1, q5_1)            \
  instantiate_kquant_qmm_t(type, 32, 5, true, 0, q5_1)                  \
  instantiate_kquant_qmm_t(type, 32, 5, true, 1, q5_1)                  \
  instantiate_kquant_qmm_t(type, 32, 5, false, 0, q5_1)                 \
  instantiate_kquant_qmm_t(type, 32, 5, false, 1, q5_1)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 5, true, q5_1)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 5, false, q5_1)             \
  instantiate_kquant_qmm_n(type, 32, 5, 0, q5_1)                        \
  instantiate_kquant_qmm_n(type, 32, 5, 1, q5_1)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 5, q5_1)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 5, q5_1)     \
  instantiate_kquant_gather_qmm_t(type, 32, 5, true, q5_1)              \
  instantiate_kquant_gather_qmm_t(type, 32, 5, false, q5_1)             \
  instantiate_kquant_gather_qmm_n(type, 32, 5, q5_1)                    \
  instantiate_kquant_dequantize(type, 32, 5, q5_1)

instantiate_kquant_q5_1_for_type(float)
instantiate_kquant_q5_1_for_type(bfloat16_t)
instantiate_kquant_q5_1_for_type(float16_t)

#define instantiate_kquant_q4_0_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 32, 4, 0, q4_0)          \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 0, q4_0)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 1, q4_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 4, 0, q4_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 4, 1, q4_0)            \
  instantiate_kquant_qmm_t(type, 32, 4, true, 0, q4_0)                  \
  instantiate_kquant_qmm_t(type, 32, 4, true, 1, q4_0)                  \
  instantiate_kquant_qmm_t(type, 32, 4, false, 0, q4_0)                 \
  instantiate_kquant_qmm_t(type, 32, 4, false, 1, q4_0)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, true, q4_0)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, false, q4_0)             \
  instantiate_kquant_qmm_n(type, 32, 4, 0, q4_0)                        \
  instantiate_kquant_qmm_n(type, 32, 4, 1, q4_0)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 4, q4_0)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 4, q4_0)     \
  instantiate_kquant_gather_qmm_t(type, 32, 4, true, q4_0)              \
  instantiate_kquant_gather_qmm_t(type, 32, 4, false, q4_0)             \
  instantiate_kquant_gather_qmm_n(type, 32, 4, q4_0)                    \
  instantiate_kquant_dequantize(type, 32, 4, q4_0)

instantiate_kquant_q4_0_for_type(float)
instantiate_kquant_q4_0_for_type(bfloat16_t)
instantiate_kquant_q4_0_for_type(float16_t)

#define instantiate_kquant_q4_1_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 32, 4, 0, q4_1)          \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 0, q4_1)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 1, q4_1)            \
  instantiate_kquant_batched(qmv,      type, 32, 4, 0, q4_1)            \
  instantiate_kquant_batched(qmv,      type, 32, 4, 1, q4_1)            \
  instantiate_kquant_qmm_t(type, 32, 4, true, 0, q4_1)                  \
  instantiate_kquant_qmm_t(type, 32, 4, true, 1, q4_1)                  \
  instantiate_kquant_qmm_t(type, 32, 4, false, 0, q4_1)                 \
  instantiate_kquant_qmm_t(type, 32, 4, false, 1, q4_1)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, true, q4_1)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, false, q4_1)             \
  instantiate_kquant_qmm_n(type, 32, 4, 0, q4_1)                        \
  instantiate_kquant_qmm_n(type, 32, 4, 1, q4_1)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 4, q4_1)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 4, q4_1)     \
  instantiate_kquant_gather_qmm_t(type, 32, 4, true, q4_1)              \
  instantiate_kquant_gather_qmm_t(type, 32, 4, false, q4_1)             \
  instantiate_kquant_gather_qmm_n(type, 32, 4, q4_1)                    \
  instantiate_kquant_dequantize(type, 32, 4, q4_1)

instantiate_kquant_q4_1_for_type(float)
instantiate_kquant_q4_1_for_type(bfloat16_t)
instantiate_kquant_q4_1_for_type(float16_t)

#define instantiate_kquant_q5_0_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 32, 5, 0, q5_0)          \
  instantiate_kquant_batched(qmv_fast, type, 32, 5, 0, q5_0)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 5, 1, q5_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 5, 0, q5_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 5, 1, q5_0)            \
  instantiate_kquant_qmm_t(type, 32, 5, true, 0, q5_0)                  \
  instantiate_kquant_qmm_t(type, 32, 5, true, 1, q5_0)                  \
  instantiate_kquant_qmm_t(type, 32, 5, false, 0, q5_0)                 \
  instantiate_kquant_qmm_t(type, 32, 5, false, 1, q5_0)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 5, true, q5_0)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 5, false, q5_0)             \
  instantiate_kquant_qmm_n(type, 32, 5, 0, q5_0)                        \
  instantiate_kquant_qmm_n(type, 32, 5, 1, q5_0)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 5, q5_0)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 5, q5_0)     \
  instantiate_kquant_gather_qmm_t(type, 32, 5, true, q5_0)              \
  instantiate_kquant_gather_qmm_t(type, 32, 5, false, q5_0)             \
  instantiate_kquant_gather_qmm_n(type, 32, 5, q5_0)                    \
  instantiate_kquant_dequantize(type, 32, 5, q5_0)

instantiate_kquant_q5_0_for_type(float)
instantiate_kquant_q5_0_for_type(bfloat16_t)
instantiate_kquant_q5_0_for_type(float16_t)

#define instantiate_kquant_q4_k_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 256, 4, 0, q4_k)          \
  instantiate_kquant_batched(qmv_fast, type, 256, 4, 0, q4_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 4, 1, q4_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 4, 0, q4_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 4, 1, q4_k)            \
  instantiate_kquant_qmm_t(type, 256, 4, true, 0, q4_k)                  \
  instantiate_kquant_qmm_t(type, 256, 4, true, 1, q4_k)                  \
  instantiate_kquant_qmm_t(type, 256, 4, false, 0, q4_k)                 \
  instantiate_kquant_qmm_t(type, 256, 4, false, 1, q4_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 4, true, q4_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 4, false, q4_k)             \
  instantiate_kquant_qmm_n(type, 256, 4, 0, q4_k)                        \
  instantiate_kquant_qmm_n(type, 256, 4, 1, q4_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 4, q4_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 4, q4_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 4, true, q4_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 4, false, q4_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 4, q4_k)                    \
  instantiate_kquant_dequantize(type, 256, 4, q4_k)

instantiate_kquant_q4_k_for_type(float)
instantiate_kquant_q4_k_for_type(bfloat16_t)
instantiate_kquant_q4_k_for_type(float16_t)

#define instantiate_kquant_q5_k_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 256, 5, 0, q5_k)          \
  instantiate_kquant_batched(qmv_fast, type, 256, 5, 0, q5_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 5, 1, q5_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 5, 0, q5_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 5, 1, q5_k)            \
  instantiate_kquant_qmm_t(type, 256, 5, true, 0, q5_k)                  \
  instantiate_kquant_qmm_t(type, 256, 5, true, 1, q5_k)                  \
  instantiate_kquant_qmm_t(type, 256, 5, false, 0, q5_k)                 \
  instantiate_kquant_qmm_t(type, 256, 5, false, 1, q5_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 5, true, q5_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 5, false, q5_k)             \
  instantiate_kquant_qmm_n(type, 256, 5, 0, q5_k)                        \
  instantiate_kquant_qmm_n(type, 256, 5, 1, q5_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 5, q5_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 5, q5_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 5, true, q5_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 5, false, q5_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 5, q5_k)                    \
  instantiate_kquant_dequantize(type, 256, 5, q5_k)

instantiate_kquant_q5_k_for_type(float)
instantiate_kquant_q5_k_for_type(bfloat16_t)
instantiate_kquant_q5_k_for_type(float16_t)

#define instantiate_kquant_q6_k_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 256, 6, 0, q6_k)          \
  instantiate_kquant_batched(verify_qmv_fine, type, 256, 6, 0, q6_k)     \
  instantiate_kquant_batched(qmv_fast, type, 256, 6, 0, q6_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 6, 1, q6_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 6, 0, q6_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 6, 1, q6_k)            \
  instantiate_kquant_qmm_t(type, 256, 6, true, 0, q6_k)                  \
  instantiate_kquant_qmm_t(type, 256, 6, true, 1, q6_k)                  \
  instantiate_kquant_qmm_t(type, 256, 6, false, 0, q6_k)                 \
  instantiate_kquant_qmm_t(type, 256, 6, false, 1, q6_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 6, true, q6_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 6, false, q6_k)             \
  instantiate_kquant_qmm_n(type, 256, 6, 0, q6_k)                        \
  instantiate_kquant_qmm_n(type, 256, 6, 1, q6_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 6, q6_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 6, q6_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 6, true, q6_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 6, false, q6_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 6, q6_k)                    \
  instantiate_kquant_dequantize(type, 256, 6, q6_k)

instantiate_kquant_q6_k_for_type(float)
instantiate_kquant_q6_k_for_type(bfloat16_t)
instantiate_kquant_q6_k_for_type(float16_t)

// Flat-with-M verify mat-vec (port of ggml mul_mv_ext_q4x4). nsg=2, nxpsg=8;
// r1ptg = M templated for M in [2,8] (one kernel per draft width). Codec-generic
// (kq_mv_ext_impl); one deq_chunk16 per codec. q8_0 is a 32-weight block, the
// K-quants are 256.
#define instantiate_mv_ext(codec, type, gs, bits, m)                    \
  instantiate_kernel(                                                   \
      "kquant_" #codec "_mv_ext_" #type "_gs_" #gs "_b_" #bits "_m" #m, \
      kq_ ## codec ## _mv_ext, type, m, 2, 8)
#define instantiate_mv_ext_for_type(codec, gs, bits, type)             \
  instantiate_mv_ext(codec, type, gs, bits, 2)                          \
  instantiate_mv_ext(codec, type, gs, bits, 3)                          \
  instantiate_mv_ext(codec, type, gs, bits, 4)                          \
  instantiate_mv_ext(codec, type, gs, bits, 5)                          \
  instantiate_mv_ext(codec, type, gs, bits, 6)                          \
  instantiate_mv_ext(codec, type, gs, bits, 7)                          \
  instantiate_mv_ext(codec, type, gs, bits, 8)
#define instantiate_mv_ext_all(codec, gs, bits)                        \
  instantiate_mv_ext_for_type(codec, gs, bits, float)                   \
  instantiate_mv_ext_for_type(codec, gs, bits, bfloat16_t)              \
  instantiate_mv_ext_for_type(codec, gs, bits, float16_t)
instantiate_mv_ext_all(q8_0, 32, 8)
instantiate_mv_ext_all(q6_k, 256, 6)

#define instantiate_kquant_q3_k_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 256, 3, 0, q3_k)          \
  instantiate_kquant_batched(qmv_fast, type, 256, 3, 0, q3_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 3, 1, q3_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 3, 0, q3_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 3, 1, q3_k)            \
  instantiate_kquant_qmm_t(type, 256, 3, true, 0, q3_k)                  \
  instantiate_kquant_qmm_t(type, 256, 3, true, 1, q3_k)                  \
  instantiate_kquant_qmm_t(type, 256, 3, false, 0, q3_k)                 \
  instantiate_kquant_qmm_t(type, 256, 3, false, 1, q3_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 3, true, q3_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 3, false, q3_k)             \
  instantiate_kquant_qmm_n(type, 256, 3, 0, q3_k)                        \
  instantiate_kquant_qmm_n(type, 256, 3, 1, q3_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 3, q3_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 3, q3_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 3, true, q3_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 3, false, q3_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 3, q3_k)                    \
  instantiate_kquant_dequantize(type, 256, 3, q3_k)

instantiate_kquant_q3_k_for_type(float)
instantiate_kquant_q3_k_for_type(bfloat16_t)
instantiate_kquant_q3_k_for_type(float16_t)

#define instantiate_kquant_q2_k_for_type(type)                          \
  instantiate_kquant_batched(verify_qmv, type, 256, 2, 0, q2_k)          \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 0, q2_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 1, q2_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 2, 0, q2_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 2, 1, q2_k)            \
  instantiate_kquant_qmm_t(type, 256, 2, true, 0, q2_k)                  \
  instantiate_kquant_qmm_t(type, 256, 2, true, 1, q2_k)                  \
  instantiate_kquant_qmm_t(type, 256, 2, false, 0, q2_k)                 \
  instantiate_kquant_qmm_t(type, 256, 2, false, 1, q2_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, true, q2_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, false, q2_k)             \
  instantiate_kquant_qmm_n(type, 256, 2, 0, q2_k)                        \
  instantiate_kquant_qmm_n(type, 256, 2, 1, q2_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 2, q2_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 2, q2_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 2, true, q2_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 2, false, q2_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 2, q2_k)                    \
  instantiate_kquant_dequantize(type, 256, 2, q2_k)

instantiate_kquant_q2_k_for_type(float)
instantiate_kquant_q2_k_for_type(bfloat16_t)
instantiate_kquant_q2_k_for_type(float16_t)

// IQ codecs are decode-only (load community GGUFs); they ship ALU-only (no NAX)
// and without verify_qmv (the small-batch path falls back to qmv, bit-exact).
#define instantiate_kquant_iq4_nl_for_type(type)                     \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 0, iq4_nl)       \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 1, iq4_nl)       \
  instantiate_kquant_batched(qmv,      type, 32, 4, 0, iq4_nl)       \
  instantiate_kquant_batched(qmv,      type, 32, 4, 1, iq4_nl)       \
  instantiate_kquant_qmm_t(type, 32, 4, true, 0, iq4_nl)             \
  instantiate_kquant_qmm_t(type, 32, 4, true, 1, iq4_nl)             \
  instantiate_kquant_qmm_t(type, 32, 4, false, 0, iq4_nl)            \
  instantiate_kquant_qmm_t(type, 32, 4, false, 1, iq4_nl)            \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, true, iq4_nl)         \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, false, iq4_nl)        \
  instantiate_kquant_qmm_n(type, 32, 4, 0, iq4_nl)                   \
  instantiate_kquant_qmm_n(type, 32, 4, 1, iq4_nl)                   \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 4, iq4_nl) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 4, iq4_nl) \
  instantiate_kquant_gather_qmm_t(type, 32, 4, true, iq4_nl)         \
  instantiate_kquant_gather_qmm_t(type, 32, 4, false, iq4_nl)        \
  instantiate_kquant_gather_qmm_n(type, 32, 4, iq4_nl)               \
  instantiate_kquant_dequantize(type, 32, 4, iq4_nl)

instantiate_kquant_iq4_nl_for_type(float)
instantiate_kquant_iq4_nl_for_type(bfloat16_t)
instantiate_kquant_iq4_nl_for_type(float16_t)

#define instantiate_kquant_iq4_xs_for_type(type)                     \
  instantiate_kquant_batched(qmv_fast, type, 256, 4, 0, iq4_xs)      \
  instantiate_kquant_batched(qmv_fast, type, 256, 4, 1, iq4_xs)      \
  instantiate_kquant_batched(qmv,      type, 256, 4, 0, iq4_xs)      \
  instantiate_kquant_batched(qmv,      type, 256, 4, 1, iq4_xs)      \
  instantiate_kquant_qmm_t(type, 256, 4, true, 0, iq4_xs)            \
  instantiate_kquant_qmm_t(type, 256, 4, true, 1, iq4_xs)           \
  instantiate_kquant_qmm_t(type, 256, 4, false, 0, iq4_xs)          \
  instantiate_kquant_qmm_t(type, 256, 4, false, 1, iq4_xs)          \
  instantiate_kquant_qmm_t_splitk(type, 256, 4, true, iq4_xs)        \
  instantiate_kquant_qmm_t_splitk(type, 256, 4, false, iq4_xs)       \
  instantiate_kquant_qmm_n(type, 256, 4, 0, iq4_xs)                  \
  instantiate_kquant_qmm_n(type, 256, 4, 1, iq4_xs)                  \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 4, iq4_xs) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 4, iq4_xs) \
  instantiate_kquant_gather_qmm_t(type, 256, 4, true, iq4_xs)        \
  instantiate_kquant_gather_qmm_t(type, 256, 4, false, iq4_xs)       \
  instantiate_kquant_gather_qmm_n(type, 256, 4, iq4_xs)              \
  instantiate_kquant_dequantize(type, 256, 4, iq4_xs)

instantiate_kquant_iq4_xs_for_type(float)
instantiate_kquant_iq4_xs_for_type(bfloat16_t)
instantiate_kquant_iq4_xs_for_type(float16_t)

#define instantiate_kquant_iq3_xxs_for_type(type)                    \
  instantiate_kquant_batched(qmv_fast, type, 256, 3, 0, iq3_xxs)     \
  instantiate_kquant_batched(qmv_fast, type, 256, 3, 1, iq3_xxs)     \
  instantiate_kquant_batched(qmv,      type, 256, 3, 0, iq3_xxs)     \
  instantiate_kquant_batched(qmv,      type, 256, 3, 1, iq3_xxs)     \
  instantiate_kquant_qmm_t(type, 256, 3, true, 0, iq3_xxs)           \
  instantiate_kquant_qmm_t(type, 256, 3, true, 1, iq3_xxs)           \
  instantiate_kquant_qmm_t(type, 256, 3, false, 0, iq3_xxs)          \
  instantiate_kquant_qmm_t(type, 256, 3, false, 1, iq3_xxs)          \
  instantiate_kquant_qmm_t_splitk(type, 256, 3, true, iq3_xxs)       \
  instantiate_kquant_qmm_t_splitk(type, 256, 3, false, iq3_xxs)      \
  instantiate_kquant_qmm_n(type, 256, 3, 0, iq3_xxs)                 \
  instantiate_kquant_qmm_n(type, 256, 3, 1, iq3_xxs)                 \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 3, iq3_xxs) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 3, iq3_xxs) \
  instantiate_kquant_gather_qmm_t(type, 256, 3, true, iq3_xxs)       \
  instantiate_kquant_gather_qmm_t(type, 256, 3, false, iq3_xxs)      \
  instantiate_kquant_gather_qmm_n(type, 256, 3, iq3_xxs)             \
  instantiate_kquant_dequantize(type, 256, 3, iq3_xxs)

instantiate_kquant_iq3_xxs_for_type(float)
instantiate_kquant_iq3_xxs_for_type(bfloat16_t)
instantiate_kquant_iq3_xxs_for_type(float16_t)

#define instantiate_kquant_iq3_s_for_type(type)                      \
  instantiate_kquant_batched(qmv_fast, type, 256, 3, 0, iq3_s)       \
  instantiate_kquant_batched(qmv_fast, type, 256, 3, 1, iq3_s)       \
  instantiate_kquant_batched(qmv,      type, 256, 3, 0, iq3_s)       \
  instantiate_kquant_batched(qmv,      type, 256, 3, 1, iq3_s)       \
  instantiate_kquant_qmm_t(type, 256, 3, true, 0, iq3_s)             \
  instantiate_kquant_qmm_t(type, 256, 3, true, 1, iq3_s)             \
  instantiate_kquant_qmm_t(type, 256, 3, false, 0, iq3_s)            \
  instantiate_kquant_qmm_t(type, 256, 3, false, 1, iq3_s)            \
  instantiate_kquant_qmm_t_splitk(type, 256, 3, true, iq3_s)         \
  instantiate_kquant_qmm_t_splitk(type, 256, 3, false, iq3_s)        \
  instantiate_kquant_qmm_n(type, 256, 3, 0, iq3_s)                   \
  instantiate_kquant_qmm_n(type, 256, 3, 1, iq3_s)                   \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 3, iq3_s) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 3, iq3_s) \
  instantiate_kquant_gather_qmm_t(type, 256, 3, true, iq3_s)         \
  instantiate_kquant_gather_qmm_t(type, 256, 3, false, iq3_s)        \
  instantiate_kquant_gather_qmm_n(type, 256, 3, iq3_s)              \
  instantiate_kquant_dequantize(type, 256, 3, iq3_s)

instantiate_kquant_iq3_s_for_type(float)
instantiate_kquant_iq3_s_for_type(bfloat16_t)
instantiate_kquant_iq3_s_for_type(float16_t)

#define instantiate_kquant_iq2_xxs_for_type(type)                    \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 0, iq2_xxs)     \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 1, iq2_xxs)     \
  instantiate_kquant_batched(qmv,      type, 256, 2, 0, iq2_xxs)     \
  instantiate_kquant_batched(qmv,      type, 256, 2, 1, iq2_xxs)     \
  instantiate_kquant_qmm_t(type, 256, 2, true, 0, iq2_xxs)           \
  instantiate_kquant_qmm_t(type, 256, 2, true, 1, iq2_xxs)           \
  instantiate_kquant_qmm_t(type, 256, 2, false, 0, iq2_xxs)          \
  instantiate_kquant_qmm_t(type, 256, 2, false, 1, iq2_xxs)          \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, true, iq2_xxs)       \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, false, iq2_xxs)      \
  instantiate_kquant_qmm_n(type, 256, 2, 0, iq2_xxs)                 \
  instantiate_kquant_qmm_n(type, 256, 2, 1, iq2_xxs)                 \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 2, iq2_xxs) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 2, iq2_xxs) \
  instantiate_kquant_gather_qmm_t(type, 256, 2, true, iq2_xxs)       \
  instantiate_kquant_gather_qmm_t(type, 256, 2, false, iq2_xxs)      \
  instantiate_kquant_gather_qmm_n(type, 256, 2, iq2_xxs)             \
  instantiate_kquant_dequantize(type, 256, 2, iq2_xxs)

instantiate_kquant_iq2_xxs_for_type(float)
instantiate_kquant_iq2_xxs_for_type(bfloat16_t)
instantiate_kquant_iq2_xxs_for_type(float16_t)

#define instantiate_kquant_iq2_xs_for_type(type)                     \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 0, iq2_xs)      \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 1, iq2_xs)      \
  instantiate_kquant_batched(qmv,      type, 256, 2, 0, iq2_xs)      \
  instantiate_kquant_batched(qmv,      type, 256, 2, 1, iq2_xs)      \
  instantiate_kquant_qmm_t(type, 256, 2, true, 0, iq2_xs)            \
  instantiate_kquant_qmm_t(type, 256, 2, true, 1, iq2_xs)            \
  instantiate_kquant_qmm_t(type, 256, 2, false, 0, iq2_xs)           \
  instantiate_kquant_qmm_t(type, 256, 2, false, 1, iq2_xs)           \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, true, iq2_xs)        \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, false, iq2_xs)       \
  instantiate_kquant_qmm_n(type, 256, 2, 0, iq2_xs)                  \
  instantiate_kquant_qmm_n(type, 256, 2, 1, iq2_xs)                  \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 2, iq2_xs) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 2, iq2_xs) \
  instantiate_kquant_gather_qmm_t(type, 256, 2, true, iq2_xs)        \
  instantiate_kquant_gather_qmm_t(type, 256, 2, false, iq2_xs)       \
  instantiate_kquant_gather_qmm_n(type, 256, 2, iq2_xs)              \
  instantiate_kquant_dequantize(type, 256, 2, iq2_xs)

instantiate_kquant_iq2_xs_for_type(float)
instantiate_kquant_iq2_xs_for_type(bfloat16_t)
instantiate_kquant_iq2_xs_for_type(float16_t)

#define instantiate_kquant_iq2_s_for_type(type)                      \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 0, iq2_s)       \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 1, iq2_s)       \
  instantiate_kquant_batched(qmv,      type, 256, 2, 0, iq2_s)       \
  instantiate_kquant_batched(qmv,      type, 256, 2, 1, iq2_s)       \
  instantiate_kquant_qmm_t(type, 256, 2, true, 0, iq2_s)             \
  instantiate_kquant_qmm_t(type, 256, 2, true, 1, iq2_s)             \
  instantiate_kquant_qmm_t(type, 256, 2, false, 0, iq2_s)            \
  instantiate_kquant_qmm_t(type, 256, 2, false, 1, iq2_s)            \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, true, iq2_s)         \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, false, iq2_s)        \
  instantiate_kquant_qmm_n(type, 256, 2, 0, iq2_s)                   \
  instantiate_kquant_qmm_n(type, 256, 2, 1, iq2_s)                   \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 2, iq2_s) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 2, iq2_s) \
  instantiate_kquant_gather_qmm_t(type, 256, 2, true, iq2_s)         \
  instantiate_kquant_gather_qmm_t(type, 256, 2, false, iq2_s)        \
  instantiate_kquant_gather_qmm_n(type, 256, 2, iq2_s)               \
  instantiate_kquant_dequantize(type, 256, 2, iq2_s)

instantiate_kquant_iq2_s_for_type(float)
instantiate_kquant_iq2_s_for_type(bfloat16_t)
instantiate_kquant_iq2_s_for_type(float16_t)

#define instantiate_kquant_iq1_s_for_type(type)                      \
  instantiate_kquant_batched(qmv_fast, type, 256, 1, 0, iq1_s)       \
  instantiate_kquant_batched(qmv_fast, type, 256, 1, 1, iq1_s)       \
  instantiate_kquant_batched(qmv,      type, 256, 1, 0, iq1_s)       \
  instantiate_kquant_batched(qmv,      type, 256, 1, 1, iq1_s)       \
  instantiate_kquant_qmm_t(type, 256, 1, true, 0, iq1_s)             \
  instantiate_kquant_qmm_t(type, 256, 1, true, 1, iq1_s)            \
  instantiate_kquant_qmm_t(type, 256, 1, false, 0, iq1_s)            \
  instantiate_kquant_qmm_t(type, 256, 1, false, 1, iq1_s)            \
  instantiate_kquant_qmm_t_splitk(type, 256, 1, true, iq1_s)         \
  instantiate_kquant_qmm_t_splitk(type, 256, 1, false, iq1_s)        \
  instantiate_kquant_qmm_n(type, 256, 1, 0, iq1_s)                   \
  instantiate_kquant_qmm_n(type, 256, 1, 1, iq1_s)                   \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 1, iq1_s) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 1, iq1_s) \
  instantiate_kquant_gather_qmm_t(type, 256, 1, true, iq1_s)         \
  instantiate_kquant_gather_qmm_t(type, 256, 1, false, iq1_s)        \
  instantiate_kquant_gather_qmm_n(type, 256, 1, iq1_s)               \
  instantiate_kquant_dequantize(type, 256, 1, iq1_s)
instantiate_kquant_iq1_s_for_type(float)
instantiate_kquant_iq1_s_for_type(bfloat16_t)
instantiate_kquant_iq1_s_for_type(float16_t)

#define instantiate_kquant_iq1_m_for_type(type)                      \
  instantiate_kquant_batched(qmv_fast, type, 256, 1, 0, iq1_m)       \
  instantiate_kquant_batched(qmv_fast, type, 256, 1, 1, iq1_m)       \
  instantiate_kquant_batched(qmv,      type, 256, 1, 0, iq1_m)       \
  instantiate_kquant_batched(qmv,      type, 256, 1, 1, iq1_m)       \
  instantiate_kquant_qmm_t(type, 256, 1, true, 0, iq1_m)             \
  instantiate_kquant_qmm_t(type, 256, 1, true, 1, iq1_m)            \
  instantiate_kquant_qmm_t(type, 256, 1, false, 0, iq1_m)            \
  instantiate_kquant_qmm_t(type, 256, 1, false, 1, iq1_m)            \
  instantiate_kquant_qmm_t_splitk(type, 256, 1, true, iq1_m)         \
  instantiate_kquant_qmm_t_splitk(type, 256, 1, false, iq1_m)        \
  instantiate_kquant_qmm_n(type, 256, 1, 0, iq1_m)                   \
  instantiate_kquant_qmm_n(type, 256, 1, 1, iq1_m)                   \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 1, iq1_m) \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 1, iq1_m) \
  instantiate_kquant_gather_qmm_t(type, 256, 1, true, iq1_m)         \
  instantiate_kquant_gather_qmm_t(type, 256, 1, false, iq1_m)        \
  instantiate_kquant_gather_qmm_n(type, 256, 1, iq1_m)               \
  instantiate_kquant_dequantize(type, 256, 1, iq1_m)
instantiate_kquant_iq1_m_for_type(float)
instantiate_kquant_iq1_m_for_type(bfloat16_t)
instantiate_kquant_iq1_m_for_type(float16_t)
    // clang-format on
