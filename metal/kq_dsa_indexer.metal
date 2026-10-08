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

instantiate_kq_dsa_topk_indices(float16_t, half, 2048, 512);
instantiate_kq_dsa_topk_indices(bfloat16_t, bfloat16_t, 2048, 512);
instantiate_kq_dsa_topk_indices(float16_t, half, 512, 512);
instantiate_kq_dsa_topk_indices(bfloat16_t, bfloat16_t, 512, 512);

// Split select: any topk, one row over `chunks` threadgroups.
#define instantiate_kq_dsa_topk_split(tname, dtype, chunks, threads)           \
  instantiate_kernel(                                                          \
      "kq_dsa_topk_hist_hi_" #tname "_c" #chunks "_t" #threads,                \
      kq_dsa_topk_hist_hi, dtype, chunks, threads)                             \
  instantiate_kernel(                                                          \
      "kq_dsa_topk_hist_lo_" #tname "_c" #chunks "_t" #threads,                \
      kq_dsa_topk_hist_lo, dtype, chunks, threads)                             \
  instantiate_kernel(                                                          \
      "kq_dsa_topk_emit_" #tname "_c" #chunks "_t" #threads,                   \
      kq_dsa_topk_emit, dtype, uint, chunks, threads)

#define instantiate_kq_dsa_topk_split_all(tname, dtype) \
  instantiate_kq_dsa_topk_split(tname, dtype, 2, 512)   \
  instantiate_kq_dsa_topk_split(tname, dtype, 4, 512)   \
  instantiate_kq_dsa_topk_split(tname, dtype, 8, 512)   \
  instantiate_kq_dsa_topk_split(tname, dtype, 16, 512)

instantiate_kq_dsa_topk_split_all(float16_t, half)
instantiate_kq_dsa_topk_split_all(bfloat16_t, bfloat16_t)

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

// Two query rows per tile, 4 heads only.
#define instantiate_kq_dsa_indexer_score_decode_pair(tname, dtype, ql, wt, ws) \
  instantiate_kernel(                                                           \
      "kq_dsa_indexer_score_decode_h4_" ws #tname "_ql" #ql "_pair",            \
      kq_dsa_indexer_score_decode, dtype, ql, 4, 128, 8, 1, wt, false, true)

#define instantiate_kq_dsa_indexer_score_decode_pair_all(tname, dtype)       \
  instantiate_kq_dsa_indexer_score_decode_pair(tname, dtype, 2, dtype, "");  \
  instantiate_kq_dsa_indexer_score_decode_pair(tname, dtype, 3, dtype, "");  \
  instantiate_kq_dsa_indexer_score_decode_pair(tname, dtype, 4, dtype, "");  \
  instantiate_kq_dsa_indexer_score_decode_pair(tname, dtype, 2, float, "wf_"); \
  instantiate_kq_dsa_indexer_score_decode_pair(tname, dtype, 3, float, "wf_"); \
  instantiate_kq_dsa_indexer_score_decode_pair(tname, dtype, 4, float, "wf_")

instantiate_kq_dsa_indexer_score_decode_pair_all(float16_t, half);
instantiate_kq_dsa_indexer_score_decode_pair_all(bfloat16_t, bfloat16_t);

instantiate_kq_dsa_indexer_score_decode_all(float16_t, half, 64, "");
instantiate_kq_dsa_indexer_score_decode_all(float16_t, half, 32, "h32_");
instantiate_kq_dsa_indexer_score_decode_all(float16_t, half, 4, "h4_");
instantiate_kq_dsa_indexer_score_decode_all(bfloat16_t, bfloat16_t, 64, "");
instantiate_kq_dsa_indexer_score_decode_all(bfloat16_t, bfloat16_t, 32, "h32_");
instantiate_kq_dsa_indexer_score_decode_all(bfloat16_t, bfloat16_t, 4, "h4_");

// clang-format on
