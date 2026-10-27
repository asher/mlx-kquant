// clang-format off
// LoRA epilogue kernel instantiations; see kq_lora_epilogue.h. Codec
// independent: rows, split rows, mix z and mix apply kernels per activation
// dtype.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_lora_epilogue.h"

instantiate_kernel("kq_lora_epilogue_rows_bfloat16_t", kq_lora_epilogue_rows, bfloat16_t)
instantiate_kernel("kq_lora_epilogue_rows_float16_t", kq_lora_epilogue_rows, float16_t)
instantiate_kernel("kq_lora_rows_z_bfloat16_t_v8", kq_lora_rows_z, bfloat16_t, 8, 4)
instantiate_kernel("kq_lora_rows_z_bfloat16_t_v4", kq_lora_rows_z, bfloat16_t, 4, 4)
instantiate_kernel("kq_lora_rows_z_bfloat16_t_v1", kq_lora_rows_z, bfloat16_t, 1, 4)
instantiate_kernel("kq_lora_rows_z_float16_t_v8", kq_lora_rows_z, float16_t, 8, 4)
instantiate_kernel("kq_lora_rows_z_float16_t_v4", kq_lora_rows_z, float16_t, 4, 4)
instantiate_kernel("kq_lora_rows_z_float16_t_v1", kq_lora_rows_z, float16_t, 1, 4)
instantiate_kernel("kq_lora_rows_apply_bfloat16_t_v4", kq_lora_rows_apply, bfloat16_t, true, 4, 4)
instantiate_kernel("kq_lora_rows_apply_bfloat16_t_v1", kq_lora_rows_apply, bfloat16_t, false, 4, 4)
instantiate_kernel("kq_lora_rows_apply_float16_t_v4", kq_lora_rows_apply, float16_t, true, 4, 4)
instantiate_kernel("kq_lora_rows_apply_float16_t_v1", kq_lora_rows_apply, float16_t, false, 4, 4)
instantiate_kernel("kq_lora_mix_z_bfloat16_t", kq_lora_mix_z, bfloat16_t)
instantiate_kernel("kq_lora_mix_z_float16_t", kq_lora_mix_z, float16_t)
instantiate_kernel("kq_lora_mix_apply_bfloat16_t", kq_lora_mix_apply, bfloat16_t)
instantiate_kernel("kq_lora_mix_apply_float16_t", kq_lora_mix_apply, float16_t)
instantiate_kernel("kq_lora_densify_1", kq_lora_densify, uint8_t)
instantiate_kernel("kq_lora_densify_2", kq_lora_densify, uint16_t)
instantiate_kernel("kq_lora_densify_4", kq_lora_densify, uint32_t)
    // clang-format on
