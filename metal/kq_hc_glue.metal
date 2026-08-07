// clang-format off
// Hyper-connection M=1 glue kernel instantiations; see kq_hc_glue.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_hc_glue.h"

instantiate_kernel("kq_hc_front_reduce_bfloat16_t", kq_hc_front_reduce, bfloat16_t)
instantiate_kernel("kq_hc_front_reduce_float16_t", kq_hc_front_reduce, float16_t)
instantiate_kernel("kq_hc_front_expand_reduce_bfloat16_t", kq_hc_front_expand_reduce, bfloat16_t)
instantiate_kernel("kq_hc_front_expand_reduce_float16_t", kq_hc_front_expand_reduce, float16_t)
instantiate_kernel("kq_hc_sinkhorn_collapse_bfloat16_t", kq_hc_sinkhorn_collapse, bfloat16_t)
instantiate_kernel("kq_hc_sinkhorn_collapse_float16_t", kq_hc_sinkhorn_collapse, float16_t)
instantiate_kernel("kq_hc_expand_bfloat16_t", kq_hc_expand, bfloat16_t)
instantiate_kernel("kq_hc_expand_float16_t", kq_hc_expand, float16_t)
    // clang-format on
