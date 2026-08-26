// clang-format off
// LoRA epilogue kernel instantiations; see kq_lora_epilogue.h. Codec
// independent: rows, mix z and mix apply kernels per activation dtype.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/kq_lora_epilogue.h"

instantiate_kernel("kq_lora_epilogue_rows_bfloat16_t", kq_lora_epilogue_rows, bfloat16_t)
instantiate_kernel("kq_lora_epilogue_rows_float16_t", kq_lora_epilogue_rows, float16_t)
instantiate_kernel("kq_lora_mix_z_bfloat16_t", kq_lora_mix_z, bfloat16_t)
instantiate_kernel("kq_lora_mix_z_float16_t", kq_lora_mix_z, float16_t)
instantiate_kernel("kq_lora_mix_apply_bfloat16_t", kq_lora_mix_apply, bfloat16_t)
instantiate_kernel("kq_lora_mix_apply_float16_t", kq_lora_mix_apply, float16_t)
instantiate_kernel("kq_lora_densify_2", kq_lora_densify, uint16_t)
instantiate_kernel("kq_lora_densify_4", kq_lora_densify, uint32_t)
    // clang-format on
