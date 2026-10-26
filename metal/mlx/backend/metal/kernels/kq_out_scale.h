// Output scale epilogue: y = s * (W x), applied to the float accumulator
// before the one store in the activation dtype. A GGUF nvfp4 tensor converted
// from a ModelOpt checkpoint carries this as a per-tensor (or per-expert)
// `.scale`.
//
// The mode is function constant 380. A pipeline fetched without it has mode
// 0: no scale buffer and no multiply, the kernel as it was. kq_get_kernel
// passes mode 0 to every function that declares the constant.
//   1  scalar                   s[0]
//   2  per column               s[n]
//   3  per expert               s[e]
//   4  per expert and column    s[e * N + n]
// A gather entry resolves the expert with kq_out_scale_at, after which modes
// 3 and 4 read as 1 and 2. The buffer is bound at KQ_OUT_SCALE_SLOT in every
// family.
#pragma once

#include <metal_stdlib>

constant int kq_out_scale_fc [[function_constant(380)]];
constant int kq_out_scale_mode =
    is_function_constant_defined(kq_out_scale_fc) ? kq_out_scale_fc : 0;
constant bool kq_has_out_scale = kq_out_scale_mode != 0;
constant bool kq_out_scale_cols =
    kq_out_scale_mode == 2 || kq_out_scale_mode == 4;

#define KQ_OUT_SCALE_SLOT 30

// Scale for output column n.
inline float kq_out_scale_col(const device float* s, int n) {
  return kq_out_scale_cols ? s[n] : s[0];
}

// The scale pointer of expert e, for a weight stack of n_cols columns.
inline const device float*
kq_out_scale_at(const device float* s, uint e, int n_cols) {
  return kq_out_scale_mode == 3 ? s + e
      : kq_out_scale_mode == 4  ? s + size_t(e) * uint(n_cols)
                                : s;
}

// The kernel argument of an entry that takes the scale. It is bound only
// when the mode is not 0.
#define KQ_OUT_SCALE_ARG        \
  const device float* out_scale \
      [[buffer(KQ_OUT_SCALE_SLOT), function_constant(kq_has_out_scale)]]

// For the shared kernel macros: OS 1 adds the argument and passes it to the
// impl (KQ_OS_AT: at expert e of n columns), OS 0 leaves the kernel as it
// was.
#define KQ_OS_ARG_1 , KQ_OUT_SCALE_ARG
#define KQ_OS_ARG_0
#define KQ_OS_PTR_1 , out_scale
#define KQ_OS_PTR_0
#define KQ_OS_AT_1(e, n) , kq_out_scale_at(out_scale, e, n)
#define KQ_OS_AT_0(e, n)
