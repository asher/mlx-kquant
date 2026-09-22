// clang-format off
// Signed block Walsh-Hadamard rotation instantiations; see kq_hadamard.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_hadamard.h"

#define instantiate_kq_hadamard(type, n) \
  instantiate_kernel("kq_hadamard_" #type "_" #n, kq_hadamard, n, KQ_HADAMARD_NT, type) \
  instantiate_kernel("kq_glu_hadamard_silu_" #type "_" #n, kq_glu_hadamard, n, KQ_HADAMARD_NT, type, 0) \
  instantiate_kernel("kq_glu_hadamard_sigmoid_" #type "_" #n, kq_glu_hadamard, n, KQ_HADAMARD_NT, type, 1)

#define instantiate_kq_hadamard_all(type) \
  instantiate_kq_hadamard(type, 256)      \
  instantiate_kq_hadamard(type, 512)      \
  instantiate_kq_hadamard(type, 1024)     \
  instantiate_kq_hadamard(type, 2048)     \
  instantiate_kq_hadamard(type, 4096)

instantiate_kq_hadamard_all(float)
instantiate_kq_hadamard_all(float16_t)
instantiate_kq_hadamard_all(bfloat16_t)
    // clang-format on
