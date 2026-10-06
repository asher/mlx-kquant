// Dense matrix rows for weight operands, decided at eval.
//
// The matmul kernels walk the last two dims of a packed weight as dense
// rows. An op builder can check that only on an array that has memory: an
// unevaluated view reports the default dense strides at build time, so a
// lazy w[0::2] looks dense and the kernel then reads the rows in between.
// kq_matrix_dense keeps the build-time check for an array with memory and
// gives any other array a KQuantMatrixDense node. The node aliases an input
// whose matrix rows are dense, whatever the stride of its leading dims and
// the size of its buffer, and copies any other input row-major.
// kq_row_major is the same for an operand that a kernel reads as one
// row-major run, such as the bytes dequantize decodes: there the node
// aliases only an input that is row-major in all of its dims.
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// Dtypes and ranks the copy kernels take (kq_lora_densify_<itemsize>).
bool dense_copy_ok(const mx::array& a) {
  const size_t isz = a.itemsize();
  return (isz == 1 || isz == 2 || isz == 4) && a.ndim() <= 8;
}

} // namespace

mx::array kq_matrix_dense(const mx::array& x, const mx::Stream& s) {
  if (x.status() != mx::array::Status::unscheduled) {
    return kq_matrix_rows_dense(x) ? x : mx::contiguous(x, false, s);
  }
  // One element or none has no layout to get wrong.
  if (x.size() <= 1) {
    return x;
  }
  if (!dense_copy_ok(x)) {
    return mx::contiguous(x, false, s);
  }
  return mx::array(
      x.shape(), x.dtype(), std::make_shared<KQuantMatrixDense>(s), {x});
}

mx::array kq_row_major(const mx::array& x, const mx::Stream& s) {
  if (x.status() != mx::array::Status::unscheduled) {
    return x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  }
  if (x.size() <= 1) {
    return x;
  }
  if (!dense_copy_ok(x)) {
    return mx::contiguous(x, false, s);
  }
  return mx::array(
      x.shape(), x.dtype(), std::make_shared<KQuantMatrixDense>(s, true), {x});
}

bool KQuantMatrixDense::passes(const mx::array& in) const {
  if (in.size() == 0) {
    return true;
  }
  return whole_ ? in.flags().row_contiguous : kq_matrix_rows_dense(in);
}

#ifdef _METAL_

namespace {

// Copies a (any strides) into out (row-major, allocated) with one thread
// per element.
void dense_copy_gpu(const mx::array& a, mx::array& out, const mx::Stream& s) {
  if (!dense_copy_ok(a)) {
    throw std::invalid_argument(
        "[mlx_kquant] a strided weight view must have a 1-, 2- or 4-byte "
        "dtype and at most 8 dims.");
  }
  // The copy kernel indexes its threads with 32 bits.
  if (a.size() > UINT32_MAX) {
    throw std::invalid_argument(
        "[mlx_kquant] a weight view of more than 4294967295 elements whose "
        "rows are not dense cannot be copied at eval. Evaluate a contiguous "
        "copy of it first, for example with mx.contiguous(w).");
  }
  auto& d = mx::metal::device(s.device);
  auto& ce = mx::metal::get_command_encoder(s);
  // KqLoraDensifyArgs in kq_lora_epilogue.h.
  struct {
    int shape[8];
    int64_t strides[8];
    int ndim;
    int64_t size;
  } args{};
  for (int i = 0; i < static_cast<int>(a.ndim()); i++) {
    args.shape[i] = a.shape(i);
    args.strides[i] = a.strides()[i];
  }
  args.ndim = static_cast<int>(a.ndim());
  args.size = static_cast<int64_t>(a.size());
  auto kernel =
      kq_get_kernel(d, "kq_lora_densify_" + std::to_string(a.itemsize()));
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(a, 0);
  ce.set_output_array(out, 1);
  ce.set_bytes(args, 2);
  const size_t tg = a.size() < 1024 ? a.size() : 1024;
  ce.dispatch_threads(MTL::Size(a.size(), 1, 1), MTL::Size(tg, 1, 1));
}

} // namespace

mx::array kq_matrix_dense_gpu(const mx::array& a, const mx::Stream& s) {
  if (a.size() == 0 || kq_matrix_rows_dense(a)) {
    return a;
  }
  mx::array t(a.shape(), a.dtype(), nullptr, {});
  t.set_data(mx::allocator::malloc(t.nbytes()));
  dense_copy_gpu(a, t, s);
  mx::metal::get_command_encoder(s).add_temporary(t);
  return t;
}

void KQuantMatrixDense::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& in = inputs[0];
  auto& out = outputs[0];
  if (passes(in)) {
    out.copy_shared_buffer(in);
    return;
  }
  out.set_data(mx::allocator::malloc(out.nbytes()));
  dense_copy_gpu(in, out, stream());
}

#else // !_METAL_

mx::array kq_matrix_dense_gpu(const mx::array&, const mx::Stream&) {
  throw std::runtime_error("[mlx_kquant] GPU eval requires a Metal build.");
}

void KQuantMatrixDense::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant] GPU eval requires a Metal build.");
}

#endif // _METAL_

void KQuantMatrixDense::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& in = inputs[0];
  auto& out = outputs[0];
  if (passes(in)) {
    out.copy_shared_buffer(in);
    return;
  }
  out.set_data(mx::allocator::malloc(out.nbytes()));
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([src = mx::array::unsafe_weak_copy(in),
                    dst = mx::array::unsafe_weak_copy(out)]() mutable {
    const size_t isz = src.itemsize();
    const char* ip = src.data<char>();
    char* op = dst.data<char>();
    const int64_t n = static_cast<int64_t>(src.size());
    for (int64_t i = 0; i < n; i++) {
      const int64_t loc = elem_to_loc64(i, src.shape(), src.strides());
      std::memcpy(op + i * isz, ip + loc * static_cast<int64_t>(isz), isz);
    }
  });
}

} // namespace mlx_kquant
