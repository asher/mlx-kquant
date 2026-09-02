// clang-format off
// DeepSeek-V4-Flash lightning-indexer instantiations; see kq_dsa_indexer.h.
// omlx is Apache-2.0: see mlx_kquant/licenses/omlx-LICENSE.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/kq_dsa_indexer.h"

#define instantiate_kq_dsa_indexer_score(tname, dtype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                      \
      "kq_dsa_indexer_score_" #tname                                       \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                  \
      kq_dsa_indexer_score, dtype, bm, bn, bk, wm, wn)

#define instantiate_kq_dsa_topk_indices(tname, dtype, topk, threads) \
  instantiate_kernel(                                                \
      "kq_dsa_topk_indices_" #tname "_topk" #topk "_t" #threads,    \
      kq_dsa_topk_indices_16bit,                                     \
      dtype,                                                         \
      uint,                                                          \
      topk,                                                          \
      threads)

instantiate_kq_dsa_indexer_score(float16_t, half, 64, 64, 16, 2, 2);
instantiate_kq_dsa_indexer_score(bfloat16_t, bfloat16_t, 64, 64, 16, 2, 2);

instantiate_kq_dsa_topk_indices(float16_t, half, 2048, 1024);
instantiate_kq_dsa_topk_indices(bfloat16_t, bfloat16_t, 2048, 1024);
instantiate_kq_dsa_topk_indices(float16_t, half, 512, 1024);
instantiate_kq_dsa_topk_indices(bfloat16_t, bfloat16_t, 512, 1024);

#define instantiate_kq_dsa_indexer_score_decode(tname, dtype, ql) \
  instantiate_kernel(                                             \
      "kq_dsa_indexer_score_decode_" #tname "_ql" #ql,           \
      kq_dsa_indexer_score_decode, dtype, ql)

instantiate_kq_dsa_indexer_score_decode(float16_t, half, 1);
instantiate_kq_dsa_indexer_score_decode(float16_t, half, 2);
instantiate_kq_dsa_indexer_score_decode(float16_t, half, 3);
instantiate_kq_dsa_indexer_score_decode(float16_t, half, 4);
instantiate_kq_dsa_indexer_score_decode(bfloat16_t, bfloat16_t, 1);
instantiate_kq_dsa_indexer_score_decode(bfloat16_t, bfloat16_t, 2);
instantiate_kq_dsa_indexer_score_decode(bfloat16_t, bfloat16_t, 3);
instantiate_kq_dsa_indexer_score_decode(bfloat16_t, bfloat16_t, 4);

// 4-head band (qwen4exp QSA indexer: 4 heads of dim 128, uniform weights)
#define instantiate_kq_dsa_indexer_score_decode_h4(tname, dtype, ql) \
  instantiate_kernel(                                                \
      "kq_dsa_indexer_score_decode_h4_" #tname "_ql" #ql,           \
      kq_dsa_indexer_score_decode, dtype, ql, 4)

instantiate_kq_dsa_indexer_score_decode_h4(float16_t, half, 1);
instantiate_kq_dsa_indexer_score_decode_h4(float16_t, half, 2);
instantiate_kq_dsa_indexer_score_decode_h4(float16_t, half, 3);
instantiate_kq_dsa_indexer_score_decode_h4(float16_t, half, 4);
instantiate_kq_dsa_indexer_score_decode_h4(bfloat16_t, bfloat16_t, 1);
instantiate_kq_dsa_indexer_score_decode_h4(bfloat16_t, bfloat16_t, 2);
instantiate_kq_dsa_indexer_score_decode_h4(bfloat16_t, bfloat16_t, 3);
instantiate_kq_dsa_indexer_score_decode_h4(bfloat16_t, bfloat16_t, 4);

// 32-head band (glm5 pooled indexer) and the fp32-head-weight arms.
#define instantiate_kq_dsa_indexer_score_decode_h32(tname, dtype, ql) \
  instantiate_kernel(                                                 \
      "kq_dsa_indexer_score_decode_h32_" #tname "_ql" #ql,           \
      kq_dsa_indexer_score_decode, dtype, ql, 32)

instantiate_kq_dsa_indexer_score_decode_h32(float16_t, half, 1);
instantiate_kq_dsa_indexer_score_decode_h32(float16_t, half, 2);
instantiate_kq_dsa_indexer_score_decode_h32(float16_t, half, 3);
instantiate_kq_dsa_indexer_score_decode_h32(float16_t, half, 4);
instantiate_kq_dsa_indexer_score_decode_h32(bfloat16_t, bfloat16_t, 1);
instantiate_kq_dsa_indexer_score_decode_h32(bfloat16_t, bfloat16_t, 2);
instantiate_kq_dsa_indexer_score_decode_h32(bfloat16_t, bfloat16_t, 3);
instantiate_kq_dsa_indexer_score_decode_h32(bfloat16_t, bfloat16_t, 4);

#define instantiate_kq_dsa_indexer_score_decode_wf(tname, dtype, ql, h) \
  instantiate_kernel(                                                   \
      "kq_dsa_indexer_score_decode_h" #h "_wf_" #tname "_ql" #ql,      \
      kq_dsa_indexer_score_decode, dtype, ql, h, 128, 4, 8, float)

#define instantiate_kq_dsa_indexer_score_decode_wf_all(tname, dtype, h) \
  instantiate_kq_dsa_indexer_score_decode_wf(tname, dtype, 1, h);       \
  instantiate_kq_dsa_indexer_score_decode_wf(tname, dtype, 2, h);       \
  instantiate_kq_dsa_indexer_score_decode_wf(tname, dtype, 3, h);       \
  instantiate_kq_dsa_indexer_score_decode_wf(tname, dtype, 4, h)

instantiate_kq_dsa_indexer_score_decode_wf_all(float16_t, half, 4);
instantiate_kq_dsa_indexer_score_decode_wf_all(float16_t, half, 32);
instantiate_kq_dsa_indexer_score_decode_wf_all(float16_t, half, 64);
instantiate_kq_dsa_indexer_score_decode_wf_all(bfloat16_t, bfloat16_t, 4);
instantiate_kq_dsa_indexer_score_decode_wf_all(bfloat16_t, bfloat16_t, 32);
instantiate_kq_dsa_indexer_score_decode_wf_all(bfloat16_t, bfloat16_t, 64);
// clang-format on
