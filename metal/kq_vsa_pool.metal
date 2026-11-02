// clang-format off
// The tile sums of vsa_attention's pools: MLX's own column-sum kernel,
// col_reduce_looped (reduction/reduce_col.h, MIT, see
// mlx_kquant/licenses/mlx-LICENSE), instantiated with a bfloat16 input and a
// float32 sum. MLX runs the same template at float32 for
// x.astype(float32).reshape(tiles, 64, H, D).sum(1); the bfloat16 to float32
// widening on load is exact, so each tile adds the same values in the same
// order. kquant_vsa.cpp encodes MLX's arguments for that reduce.
#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduction/reduce_col.h"

instantiate_kernel(
    "kq_vsa_tile_sums_bfloat16",
    col_reduce_looped,
    bfloat16_t, float, Sum<float>, int, 1, 32, 32)
