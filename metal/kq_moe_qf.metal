// clang-format off
// Fused MoE gathers on the tuned qmv_fast loops; see kq_moe_qf.h. The nvfp4
// kernels, which take output scales, are in kq_moe_qf_os.metal.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"
#include "mlx/backend/metal/kernels/kq_moe_qf.h"

instantiate_kq_qf(q2_k, KqQfQ2_K)
instantiate_kq_qf(q3_k, KqQfQ3_K)
instantiate_kq_qf(q4_k, KqQfQ4_K)
instantiate_kq_qf(iq4_xs, KqQfIq4_xs)
instantiate_kq_qf_dd(iq4_xs, KqQfIq4_xs)
instantiate_kq_qf(q4_0, KqQfQ4_0)
instantiate_kq_qf(q4_1, KqQfQ4_1)
instantiate_kq_qf(q5_0, KqQfQ5_0)
instantiate_kq_qf_dd(q5_0, KqQfQ5_0)
instantiate_kq_qf(q5_1, KqQfQ5_1)
instantiate_kq_qf_dd(q5_1, KqQfQ5_1)
instantiate_kq_qf(iq4_nl, KqQfIq4_nl)
instantiate_kq_qf_dd(iq4_nl, KqQfIq4_nl)
instantiate_kq_qf(q2_0, KqQfQ2_0)
instantiate_kq_qf_dd(q2_0, KqQfQ2_0)
instantiate_kq_qf(mxfp4, KqQfMxfp4)
instantiate_kq_qf_dd(mxfp4, KqQfMxfp4)
instantiate_kq_qf_bias(mxfp4, KqQfMxfp4)
    // clang-format on
