// clang-format off
// The nvfp4 qf kernels, which take output scales (kq_moe_glu_kq.h,
// KQ_MOE_OS). Every other qf codec is in kq_moe_qf.metal.
#define KQ_MOE_OS 1

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_moe_glu_kq.h"
#include "mlx/backend/metal/kernels/kq_moe_qf.h"

instantiate_kq_qf(nvfp4, KqQfNvfp4)
instantiate_kq_qf_dd(nvfp4, KqQfNvfp4)
instantiate_kq_qf_bias(nvfp4, KqQfNvfp4)
    // clang-format on
