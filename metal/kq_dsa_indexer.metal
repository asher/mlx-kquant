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
// clang-format on
