// clang-format off
// Tiled kernels of the fused MoE ops; see kq_moe_tile.h.
// Every codec pair of the grouped kernels but nvfp4, which takes output
// scales and keeps the grouped kernels.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"
#include "mlx/backend/metal/kernels/kq_moe_tile.h"

instantiate_kernel("kq_moe_tile_plan_bfloat16_t", kq_moe_tile_plan, bfloat16_t)
instantiate_kernel("kq_moe_tile_plan_float16_t", kq_moe_tile_plan, float16_t)
instantiate_kernel("kq_moe_tile_combine_bfloat16_t", kq_moe_tile_combine, bfloat16_t)
instantiate_kernel("kq_moe_tile_combine_float16_t", kq_moe_tile_combine, float16_t)

instantiate_kq_tl_all(q2_k, KqQ2_KExt)
instantiate_kq_tl_all(q3_k, KqQ3_KExt)
instantiate_kq_tl_all(q4_k, KqQ4_KExt)
instantiate_kq_tl_pair("q5_k_sx_q5_k", KqQ5_KExt, KqQ5_KExt)
instantiate_kq_tl_pair("q5_k_sx_q6_k", KqQ5_KExt, KqQ6_KExt)
instantiate_kq_tl_pair("q5_k_sx_q8_0", KqQ5_KExt, KqQ8_0Ext)
instantiate_kq_tl_all(q4_0, KqQ4_0Ext)
instantiate_kq_tl_all(q4_1, KqQ4_1Ext)
instantiate_kq_tl_all(q5_0, KqQ5_0Ext)
instantiate_kq_tl_all(q5_1, KqQ5_1Ext)
instantiate_kq_tl_all(iq4_nl, KqIq4_nlExt)
instantiate_kq_tl_all(iq4_xs, KqIq4_xsExt)
instantiate_kq_tl_all(iq3_s, KqIq3_sExt)
instantiate_kq_tl_all(iq3_xxs, KqIq3_xxsExt)
instantiate_kq_tl_all(iq2_xxs, KqIq2_xxsExt)
instantiate_kq_tl_all(iq2_xs, KqIq2_xsExt)
instantiate_kq_tl_all(iq2_s, KqIq2_sExt)
instantiate_kq_tl_all(iq1_s, KqIq1_sExt)
instantiate_kq_tl_all(iq1_m, KqIq1_mExt)
instantiate_kq_tl_all(stq1_0, KqStq1_0Ext)
instantiate_kq_tl_all(pq2_0, KqPq2_0Ext)
instantiate_kq_tl_all(q2_0, KqQ2_0Ext)
instantiate_kq_tl_all(ptq1_0, KqPtq1_0Ext)
instantiate_kq_tl_all(mxfp4, KqMxfp4Ext)
instantiate_kq_tl_pair("q6_k_sx_q8_0", KqQ6_KExt, KqQ8_0Ext)
instantiate_kq_tl_pair("q8_0_sx_q6_k", KqQ8_0Ext, KqQ6_KExt)
instantiate_kq_tl_pair("q6_k_sx_q5_k", KqQ6_KExt, KqQ5_KExt)
instantiate_kq_tl_pair("q8_0_sx_q5_k", KqQ8_0Ext, KqQ5_KExt)
instantiate_kq_tl_pair("q2_0_sx_iq4_nl", KqQ2_0Ext, KqIq4_nlExt)
instantiate_kq_tl_pair("q6_k_ext_sx_q6_k_ext", KqQ6_KExt, KqQ6_KExt)
instantiate_kq_tl_pair("q8_0_ext_sx_q8_0_ext", KqQ8_0Ext, KqQ8_0Ext)
    // clang-format on
