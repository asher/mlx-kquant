// clang-format off
// DeepSeek-V4-Flash sparse attention instantiations; see kq_dsa_sparse_attn.h.
// omlx is Apache-2.0: see mlx_kquant/licenses/omlx-LICENSE.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
#include "mlx/backend/metal/kernels/kq_dsa_sparse_attn.h"

#define instantiate_kq_dsa_sparse_attention(tname, dtype, bk, dc, h, d, wm) \
  instantiate_kernel(                                                       \
      "kq_dsa_sparse_attention_" #tname "_bk" #bk "_dc" #dc "_h" #h        \
      "_d" #d "_wm" #wm,                                                    \
      kq_dsa_sparse_attention,                                              \
      dtype,                                                                \
      bk,                                                                   \
      dc,                                                                   \
      h,                                                                    \
      d,                                                                    \
      wm,                                                                   \
      uint,                                                                 \
      float)

instantiate_kq_dsa_sparse_attention(float16_t, half, 256, 32, 64, 512, 8);
instantiate_kq_dsa_sparse_attention(bfloat16_t, bfloat16_t, 256, 32, 64, 512, 8);
instantiate_kq_dsa_sparse_attention(float16_t, half, 128, 32, 64, 512, 8);
instantiate_kq_dsa_sparse_attention(bfloat16_t, bfloat16_t, 128, 32, 64, 512, 8);

#define instantiate_kq_dsa_sparse_attention_split(tname, dtype, bk, dc, h, d, wm) \
  instantiate_kernel(                                                             \
      "kq_dsa_sparse_attention_split_" #tname "_bk" #bk "_dc" #dc "_h" #h         \
      "_d" #d "_wm" #wm,                                                          \
      kq_dsa_sparse_attention_split,                                              \
      dtype,                                                                      \
      bk,                                                                         \
      dc,                                                                         \
      h,                                                                          \
      d,                                                                          \
      wm,                                                                         \
      uint,                                                                       \
      float)

#define instantiate_kq_dsa_sparse_attention_merge(tname, dtype, d) \
  instantiate_kernel(                                              \
      "kq_dsa_sparse_attention_merge_" #tname "_d" #d,             \
      kq_dsa_sparse_attention_merge,                               \
      dtype,                                                       \
      d)

instantiate_kq_dsa_sparse_attention_split(float16_t, half, 256, 32, 64, 512, 8);
instantiate_kq_dsa_sparse_attention_split(bfloat16_t, bfloat16_t, 256, 32, 64, 512, 8);
instantiate_kq_dsa_sparse_attention_split(float16_t, half, 128, 32, 64, 512, 8);
instantiate_kq_dsa_sparse_attention_split(bfloat16_t, bfloat16_t, 128, 32, 64, 512, 8);
instantiate_kq_dsa_sparse_attention_merge(float16_t, half, 512);
instantiate_kq_dsa_sparse_attention_merge(bfloat16_t, bfloat16_t, 512);
// clang-format on
