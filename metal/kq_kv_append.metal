// clang-format off
// KV cache append instantiations per element size; see kq_kv_append.h.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_kv_append.h"

instantiate_kernel("kq_kv_append_1", kq_kv_append, uint8_t)
instantiate_kernel("kq_kv_append_2", kq_kv_append, uint16_t)
instantiate_kernel("kq_kv_append_4", kq_kv_append, uint32_t)
    // clang-format on
