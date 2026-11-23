// clang-format off
// Dense row-class kernels (kq_dense_rows.h); derived-code attribution lives
// in the included kq_*.h headers and mlx_kquant/licenses/.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_dense_rows.h"

#define instantiate_kquant_rows(name)                                  \
  instantiate_kernel("kquant_" #name "_rows_bfloat16_t",              \
      kq_ ## name ## _rows, bfloat16_t)                               \
  instantiate_kernel("kquant_" #name "_rows_float16_t",               \
      kq_ ## name ## _rows, float16_t)

instantiate_kquant_rows(pq2_0)
instantiate_kquant_rows(q2_0)
instantiate_kquant_rows(nvfp4)
instantiate_kquant_rows(nvfp4_sb)
instantiate_kquant_rows(q4_0)
instantiate_kquant_rows(q4_0_sb)
instantiate_kquant_rows(q4_1)
instantiate_kquant_rows(q4_1_sb)
instantiate_kquant_rows(q8_0)
instantiate_kquant_rows(q4_k)
instantiate_kquant_rows(q5_k)
instantiate_kquant_rows(q6_k)
instantiate_kquant_rows(q3_k)
instantiate_kquant_rows(q2_k)
instantiate_kquant_rows(iq4_xs)
instantiate_kquant_rows(iq4_nl)
instantiate_kquant_rows(q2_k_gen)
instantiate_kquant_rows(q3_k_gen)
instantiate_kquant_rows(q4_k_gen)
instantiate_kquant_rows(q5_k_gen)
instantiate_kquant_rows(q6_k_gen)
instantiate_kquant_rows(q4_0_gen)
instantiate_kquant_rows(q4_1_gen)
instantiate_kquant_rows(q5_0_gen)
instantiate_kquant_rows(q5_1_gen)
instantiate_kquant_rows(q8_0_gen)
instantiate_kquant_rows(iq4_nl_gen)
instantiate_kquant_rows(iq4_xs_gen)
instantiate_kquant_rows(iq3_s_gen)
instantiate_kquant_rows(iq3_xxs_gen)
instantiate_kquant_rows(iq2_xxs_gen)
instantiate_kquant_rows(iq2_xs_gen)
instantiate_kquant_rows(iq2_s_gen)
instantiate_kquant_rows(iq1_s_gen)
instantiate_kquant_rows(iq1_m_gen)
instantiate_kquant_rows(stq1_0_gen)
instantiate_kquant_rows(pq2_0_gen)
instantiate_kquant_rows(q2_0_gen)
instantiate_kquant_rows(ptq1_0_gen)
instantiate_kquant_rows(mxfp4_gen)
instantiate_kquant_rows(nvfp4_gen)
instantiate_kquant_rows(q4_0_gen1)
instantiate_kquant_rows(q4_1_gen1)
instantiate_kquant_rows(q5_0_gen1)
instantiate_kquant_rows(q5_1_gen1)
instantiate_kquant_rows(q8_0_gen1)
instantiate_kquant_rows(iq4_nl_gen1)
instantiate_kquant_rows(q2_0_gen1)
instantiate_kquant_rows(mxfp4_gen1)
instantiate_kquant_rows(nvfp4_gen1)
    // clang-format on
