// clang-format off
// Sparse decode attention instantiations; see kq_sdpa_sparse_decode.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_sdpa_sparse_decode.h"

#define instantiate_kq_sdpa_sparse_decode_split(tname, dtype, iname, itype, d) \
  instantiate_kernel(                                                          \
      "kq_sdpa_sparse_decode_split_" #tname "_" #iname "_d" #d,               \
      kq_sdpa_sparse_decode_split, dtype, itype, d)

#define instantiate_kq_sdpa_sparse_decode_merge(tname, dtype, d) \
  instantiate_kernel(                                            \
      "kq_sdpa_sparse_decode_merge_" #tname "_d" #d,             \
      kq_sdpa_sparse_decode_merge, dtype, d)

#define instantiate_kq_sdpa_sparse_decode_d(d)                                  \
  instantiate_kq_sdpa_sparse_decode_split(float16_t, half, i32, int32_t, d)    \
  instantiate_kq_sdpa_sparse_decode_split(float16_t, half, u32, uint32_t, d)   \
  instantiate_kq_sdpa_sparse_decode_split(bfloat16_t, bfloat16_t, i32, int32_t, d)  \
  instantiate_kq_sdpa_sparse_decode_split(bfloat16_t, bfloat16_t, u32, uint32_t, d) \
  instantiate_kq_sdpa_sparse_decode_merge(float16_t, half, d)                  \
  instantiate_kq_sdpa_sparse_decode_merge(bfloat16_t, bfloat16_t, d)

instantiate_kq_sdpa_sparse_decode_d(128)
instantiate_kq_sdpa_sparse_decode_d(256)
instantiate_kq_sdpa_sparse_decode_d(512)
    // clang-format on
