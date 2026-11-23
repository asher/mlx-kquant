// clang-format off
// Wide-row expert store kernels (moe_store_seg); see kq_moe_store_seg.h.
// Field planes: q5_0, iq4_nl, iq3_xxs, iq4_xs. Generic format: the codecs
// whose NAX block loader walks a block by 32-weight tiles. Every other
// store codec keeps the tile kernels of kq_moe_store.metal at every width.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/kq_quantized_nax.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"
#include "mlx/backend/metal/kernels/kq_moe_tile.h"
#define KQ_MOE_STORE_NO_PACK
#include "mlx/backend/metal/kernels/unary_ops.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/kq_moe_store_seg.h"

// 72 = BK + 16 / sizeof(T), the slab's padded row of a 2-byte T.
#define instantiate_kq_st_seg_t(codec, type)                                  \
  instantiate_kernel(                                                         \
      "kq_store_seg_" #codec "_" #type "_bm_64", kq_st_seg, type,             \
      KqStSegL_##codec<type>, 64, 2)                                          \
  instantiate_kernel(                                                         \
      "kq_store_seg_" #codec "_" #type "_bm_128", kq_st_seg, type,            \
      KqStSegL_##codec<type>, 128, 4)
#define instantiate_kq_st_seg(codec)                                          \
  instantiate_kq_st_seg_t(codec, bfloat16_t)                                  \
  instantiate_kq_st_seg_t(codec, float16_t)
#define instantiate_kq_st_seg_split(codec, traits)                            \
  template <typename T>                                                       \
  using KqStSegL_##codec = KqStSegSplitLoader<T, traits, 72>;                 \
  instantiate_kq_st_seg(codec)
#define instantiate_kq_st_seg_block(codec, inner)                             \
  template <typename T>                                                       \
  using KqStSegL_##codec =                                                    \
      KqStSegBlockLoader<T, inner<T, 64, 64, 72, 1, 128>>;                    \
  instantiate_kq_st_seg(codec)

instantiate_kq_st_seg_split(q5_0, KqQ5_0Ext)
instantiate_kq_st_seg_split(iq4_nl, KqIq4_nlExt)
instantiate_kq_st_seg_split(iq3_xxs, KqIq3_xxsExt)
instantiate_kq_st_seg_split(iq4_xs, KqIq4_xsExt)

instantiate_kq_st_seg_block(iq2_s, KqNaxIq2_sBlockLoader)
instantiate_kq_st_seg_block(q2_0, KqNaxQ2_0BlockLoader)
instantiate_kq_st_seg_block(iq3_s, KqNaxIq3_sBlockLoader)
instantiate_kq_st_seg_block(iq2_xxs, KqNaxIq2_xxsBlockLoader)
instantiate_kq_st_seg_block(iq2_xs, KqNaxIq2_xsBlockLoader)
instantiate_kq_st_seg_block(iq1_s, KqNaxIq1_sBlockLoader)
instantiate_kq_st_seg_block(iq1_m, KqNaxIq1_mBlockLoader)
instantiate_kq_st_seg_block(stq1_0, KqNaxStq1_0BlockLoader)
instantiate_kq_st_seg_block(pq2_0, KqNaxPq2_0BlockLoader)
instantiate_kq_st_seg_block(ptq1_0, KqNaxPtq1_0BlockLoader)

// Gate and up in one pass (moe_store_seg_pair): o0 both products, o1 the
// SwiGLU of the two.
#define instantiate_kq_st_seg_pair_t(codec, type, bm, wm, out)                \
  instantiate_kernel(                                                         \
      "kq_store_seg_pair_" #codec "_" #type "_bm_" #bm "_o" #out,             \
      kq_st_seg_pair, type, KqStSegL_##codec<type>, bm, wm, out)
#define instantiate_kq_st_seg_pair(codec)                                     \
  instantiate_kq_st_seg_pair_t(codec, bfloat16_t, 64, 2, 0)                   \
  instantiate_kq_st_seg_pair_t(codec, bfloat16_t, 128, 4, 0)                  \
  instantiate_kq_st_seg_pair_t(codec, bfloat16_t, 64, 2, 1)                   \
  instantiate_kq_st_seg_pair_t(codec, bfloat16_t, 128, 4, 1)                  \
  instantiate_kq_st_seg_pair_t(codec, float16_t, 64, 2, 0)                    \
  instantiate_kq_st_seg_pair_t(codec, float16_t, 128, 4, 0)                   \
  instantiate_kq_st_seg_pair_t(codec, float16_t, 64, 2, 1)                    \
  instantiate_kq_st_seg_pair_t(codec, float16_t, 128, 4, 1)
instantiate_kq_st_seg_pair(iq3_xxs)
instantiate_kq_st_seg_pair(iq4_xs)
instantiate_kq_st_seg_pair(iq2_s)
