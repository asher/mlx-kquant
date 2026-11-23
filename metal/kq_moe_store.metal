// clang-format off
// Expert store kernels of the fused MoE ops; see kq_moe_store.h.
// q5_0, iq4_nl, iq3_xxs and iq4_xs have formats with a plane a block field;
// every other codec of the tiled kernels has the generic format (a plane of
// whole blocks). nvfp4 has no store and keeps the in-place kernels.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"
#include "mlx/backend/metal/kernels/kq_moe_qf.h"
#include "mlx/backend/metal/kernels/kq_moe_tile.h"
#include "mlx/backend/metal/kernels/kq_moe_store.h"

instantiate_kernel("kq_moe_store_plan_bfloat16_t", kq_moe_store_plan, bfloat16_t)
instantiate_kernel("kq_moe_store_plan_float16_t", kq_moe_store_plan, float16_t)
instantiate_kernel("kq_moe_store_combine_bfloat16_t", kq_moe_store_combine, bfloat16_t)
instantiate_kernel("kq_moe_store_combine_float16_t", kq_moe_store_combine, float16_t)

instantiate_kq_st_all(q5_0, KqQ5_0Ext)
instantiate_kq_st_all(iq4_nl, KqIq4_nlExt)
instantiate_kq_st_all(iq3_xxs, KqIq3_xxsExt)
instantiate_kq_st_all(iq4_xs, KqIq4_xsExt)
instantiate_kq_st_all(iq2_s, KqIq2_sExt)
instantiate_kq_st_all(q2_0, KqQ2_0Ext)
instantiate_kq_st_all(q2_k, KqQ2_KExt)
instantiate_kq_st_all(q3_k, KqQ3_KExt)
instantiate_kq_st_all(q4_k, KqQ4_KExt)
instantiate_kq_st_wide(q5_k, KqQ5_KExt)
instantiate_kq_st_wide(q6_k, KqQ6_KExt)
instantiate_kq_st_wide(q8_0, KqQ8_0Ext)
instantiate_kq_st_all(q4_0, KqQ4_0Ext)
instantiate_kq_st_all(q4_1, KqQ4_1Ext)
instantiate_kq_st_all(q5_1, KqQ5_1Ext)
instantiate_kq_st_all(iq3_s, KqIq3_sExt)
instantiate_kq_st_all(iq2_xxs, KqIq2_xxsExt)
instantiate_kq_st_all(iq2_xs, KqIq2_xsExt)
instantiate_kq_st_all(iq1_s, KqIq1_sExt)
instantiate_kq_st_all(iq1_m, KqIq1_mExt)
instantiate_kq_st_all(stq1_0, KqStq1_0Ext)
instantiate_kq_st_all(pq2_0, KqPq2_0Ext)
instantiate_kq_st_all(ptq1_0, KqPtq1_0Ext)
instantiate_kq_st_all(mxfp4, KqMxfp4Ext)

// The pair-grid forms (the four formats with a plane a block field) and the
// in-place pair form on the planes (iq3_xxs).
instantiate_kq_st_pg_all(q5_0, KqQ5_0Ext)
instantiate_kq_st_pg_all(iq4_nl, KqIq4_nlExt)
instantiate_kq_st_pg_all(iq3_xxs, KqIq3_xxsExt)
instantiate_kq_st_pg_all(iq4_xs, KqIq4_xsExt)
instantiate_kq_st_dd_all()
    // clang-format on
