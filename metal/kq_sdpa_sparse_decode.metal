// clang-format off
// Sparse decode attention instantiations; see kq_sdpa_sparse_decode.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_sdpa_sparse_decode.h"

#define instantiate_kq_sdpa_sparse_decode_split(tname, dtype, iname, itype, d, hg) \
  instantiate_kernel(                                                          \
      "kq_sdpa_sparse_decode_split_" #tname "_" #iname "_d" #d "_hg" #hg,     \
      kq_sdpa_sparse_decode_split, dtype, itype, d, hg, false)                 \
  instantiate_kernel(                                                          \
      "kq_sdpa_sparse_decode_split_" #tname "_" #iname "_d" #d "_hg" #hg "_pk", \
      kq_sdpa_sparse_decode_split, dtype, itype, d, hg, true)

#define instantiate_kq_sdpa_sparse_decode_merge(tname, dtype, d) \
  instantiate_kernel(                                            \
      "kq_sdpa_sparse_decode_merge_" #tname "_d" #d,             \
      kq_sdpa_sparse_decode_merge, dtype, d)

#define instantiate_kq_sdpa_sparse_decode_hg(d, hg)                                     \
  instantiate_kq_sdpa_sparse_decode_split(float16_t, half, i32, int32_t, d, hg)          \
  instantiate_kq_sdpa_sparse_decode_split(float16_t, half, u32, uint32_t, d, hg)         \
  instantiate_kq_sdpa_sparse_decode_split(bfloat16_t, bfloat16_t, i32, int32_t, d, hg)   \
  instantiate_kq_sdpa_sparse_decode_split(bfloat16_t, bfloat16_t, u32, uint32_t, d, hg)

#define instantiate_kq_sdpa_sparse_decode_d(d)                                  \
  instantiate_kq_sdpa_sparse_decode_hg(d, 8)                                   \
  instantiate_kq_sdpa_sparse_decode_hg(d, 16)                                  \
  instantiate_kq_sdpa_sparse_decode_hg(d, 32)                                  \
  instantiate_kq_sdpa_sparse_decode_hg(d, 64)                                  \
  instantiate_kq_sdpa_sparse_decode_merge(float16_t, half, d)                  \
  instantiate_kq_sdpa_sparse_decode_merge(bfloat16_t, bfloat16_t, d)

instantiate_kq_sdpa_sparse_decode_d(128)
instantiate_kq_sdpa_sparse_decode_d(256)
instantiate_kq_sdpa_sparse_decode_d(512)

// Prefill form: (head group, D slices, keys per block) configurations. The
// host default must be instantiated at every head dim; the rest are the
// KQ_SDPA_SPARSE_PREFILL_CFG A/B set at head dim 512.
#define instantiate_kq_sdpa_sparse_prefill(tname, dtype, iname, itype, d, hg, ds, kb) \
  instantiate_kernel(                                                               \
      "kq_sdpa_sparse_prefill_" #tname "_" #iname "_d" #d "_hg" #hg "_ds" #ds "_kb" #kb, \
      kq_sdpa_sparse_prefill, dtype, itype, d, hg, ds, kb, false)                   \
  instantiate_kernel(                                                               \
      "kq_sdpa_sparse_prefill_" #tname "_" #iname "_d" #d "_hg" #hg "_ds" #ds "_kb" #kb "_pk", \
      kq_sdpa_sparse_prefill, dtype, itype, d, hg, ds, kb, true)

#define instantiate_kq_sdpa_sparse_prefill_cfg(d, hg, ds, kb)                                   \
  instantiate_kq_sdpa_sparse_prefill(float16_t, half, i32, int32_t, d, hg, ds, kb)               \
  instantiate_kq_sdpa_sparse_prefill(float16_t, half, u32, uint32_t, d, hg, ds, kb)              \
  instantiate_kq_sdpa_sparse_prefill(bfloat16_t, bfloat16_t, i32, int32_t, d, hg, ds, kb)        \
  instantiate_kq_sdpa_sparse_prefill(bfloat16_t, bfloat16_t, u32, uint32_t, d, hg, ds, kb)

instantiate_kq_sdpa_sparse_prefill_cfg(128, 16, 4, 8)
instantiate_kq_sdpa_sparse_prefill_cfg(256, 16, 4, 8)
instantiate_kq_sdpa_sparse_prefill_cfg(512, 16, 4, 8)
instantiate_kq_sdpa_sparse_prefill_cfg(512, 32, 4, 8)
instantiate_kq_sdpa_sparse_prefill_cfg(512, 16, 2, 8)
    // clang-format on
