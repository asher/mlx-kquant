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

// Decode scorer: heads 4/32/64, 16-bit or fp32 head weights.
#define instantiate_kq_dsa_indexer_score_decode(tname, dtype, ql, h, hs, wt, ws) \
  instantiate_kernel(                                                               \
      "kq_dsa_indexer_score_decode_" hs ws #tname "_ql" #ql,                        \
      kq_dsa_indexer_score_decode, dtype, ql, h, 128, 8, 1, wt)                     \
  instantiate_kernel(                                                               \
      "kq_dsa_indexer_score_decode_" hs ws #tname "_ql" #ql "_cand",                \
      kq_dsa_indexer_score_decode, dtype, ql, h, 128, 4, 1, wt, true)

#define instantiate_kq_dsa_indexer_score_decode_all(tname, dtype, h, hs)  \
  instantiate_kq_dsa_indexer_score_decode(tname, dtype, 1, h, hs, dtype, ""); \
  instantiate_kq_dsa_indexer_score_decode(tname, dtype, 2, h, hs, dtype, ""); \
  instantiate_kq_dsa_indexer_score_decode(tname, dtype, 3, h, hs, dtype, ""); \
  instantiate_kq_dsa_indexer_score_decode(tname, dtype, 4, h, hs, dtype, ""); \
  instantiate_kq_dsa_indexer_score_decode(tname, dtype, 1, h, "h" #h "_", float, "wf_"); \
  instantiate_kq_dsa_indexer_score_decode(tname, dtype, 2, h, "h" #h "_", float, "wf_"); \
  instantiate_kq_dsa_indexer_score_decode(tname, dtype, 3, h, "h" #h "_", float, "wf_"); \
  instantiate_kq_dsa_indexer_score_decode(tname, dtype, 4, h, "h" #h "_", float, "wf_")

instantiate_kq_dsa_indexer_score_decode_all(float16_t, half, 64, "");
instantiate_kq_dsa_indexer_score_decode_all(float16_t, half, 32, "h32_");
instantiate_kq_dsa_indexer_score_decode_all(float16_t, half, 4, "h4_");
instantiate_kq_dsa_indexer_score_decode_all(bfloat16_t, bfloat16_t, 64, "");
instantiate_kq_dsa_indexer_score_decode_all(bfloat16_t, bfloat16_t, 32, "h32_");
instantiate_kq_dsa_indexer_score_decode_all(bfloat16_t, bfloat16_t, 4, "h4_");

// clang-format on
