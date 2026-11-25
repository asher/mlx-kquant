// clang-format off
// The nvfp4 fused MoE gather kernels, which take output scales
// (kq_moe_glu_kq.h, KQ_MOE_OS). Every other codec's kernels are in
// kq_moe_glu_kq.metal.
#define KQ_MOE_OS 1

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"

instantiate_kq_moe_grp_common()
instantiate_kq_ext_all(nvfp4, KqNvfp4Ext)
instantiate_kq_ext_bias(nvfp4, KqNvfp4Ext, kq_x16)
    // clang-format on
