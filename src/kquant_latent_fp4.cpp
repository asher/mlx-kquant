// KQLatentFp4Pack / KQLatentFp4Unpack primitives: FP4 rows at rest for the
// sparse attention pool (see kq_latent_fp4.h). Groups of 16 values, one
// E2M1 nibble each plus an E4M3 scale byte per group; rows on ds4's latent
// QAT grid pack exactly. Inference-only (no CPU eval).
#include <sstream>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h" // contiguous
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

void KQLatentFp4Pack::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scales = outputs[1];
  codes.set_data(mx::allocator::malloc(codes.nbytes()));
  scales.set_data(mx::allocator::malloc(scales.nbytes()));

  const int groups = int(x.size() / 16);
  const std::string kname = "kq_latent_fp4_pack_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname, kname, {});
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  ce.set_input_array(x, 0);
  ce.set_output_array(codes, 1);
  ce.set_output_array(scales, 2);
  ce.set_bytes(groups, 3);

  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims((groups + 255) / 256, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQLatentFp4Unpack::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& codes = inputs[0];
  const auto& scales = inputs[1];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const int groups = int(scales.size());
  const std::string kname =
      "kq_latent_fp4_unpack_" + kq_type_string(out.dtype());
  auto kernel = kq_get_kernel(d, kname, kname, {});
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  ce.set_input_array(codes, 0);
  ce.set_input_array(scales, 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(groups, 3);

  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims((groups + 255) / 256, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void KQLatentFp4Pack::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.latent_fp4_pack] requires a Metal build.");
}

void KQLatentFp4Unpack::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.latent_fp4_unpack] requires a Metal build.");
}

#endif

void KQLatentFp4Pack::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.latent_fp4_pack] has no CPU implementation.");
}

std::vector<mx::Shape> KQLatentFp4Pack::output_shapes(
    const std::vector<mx::array>& inputs) {
  auto codes_shape = inputs[0].shape();
  codes_shape.back() /= 2;
  auto scales_shape = inputs[0].shape();
  scales_shape.back() /= 16;
  return {std::move(codes_shape), std::move(scales_shape)};
}

bool KQLatentFp4Pack::is_equivalent(const mx::Primitive&) const {
  return true;
}

void KQLatentFp4Unpack::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.latent_fp4_unpack] has no CPU implementation.");
}

std::vector<mx::Shape> KQLatentFp4Unpack::output_shapes(
    const std::vector<mx::array>& inputs) {
  auto shape = inputs[0].shape();
  shape.back() *= 2;
  return {std::move(shape)};
}

bool KQLatentFp4Unpack::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQLatentFp4Unpack&>(other);
  return dtype_ == o.dtype_;
}

std::vector<mx::array> latent_fp4_pack(mx::array x, mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() < 1 || x.shape(-1) % 16 != 0 || x.shape(-1) == 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.latent_fp4_pack] expected a trailing dim that is a "
        << "positive multiple of 16, got shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != mx::float16 && x.dtype() != mx::bfloat16 &&
      x.dtype() != mx::float32) {
    std::ostringstream msg;
    msg << "[mlx_kquant.latent_fp4_pack] expected fp16/bf16/fp32 input, got "
        << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  auto xc = mx::contiguous(x, false, s);
  auto codes_shape = xc.shape();
  codes_shape.back() /= 2;
  auto scales_shape = xc.shape();
  scales_shape.back() /= 16;
  std::vector<mx::Shape> shapes = {
      std::move(codes_shape), std::move(scales_shape)};
  std::vector<mx::Dtype> dtypes = {mx::uint8, mx::uint8};
  return mx::array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<KQLatentFp4Pack>(s),
      {std::move(xc)});
}

mx::array latent_fp4_unpack(
    mx::array codes,
    mx::array scales,
    mx::Dtype dtype,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (codes.dtype() != mx::uint8 || scales.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.latent_fp4_unpack] codes and scales must be uint8.");
  }
  if (codes.ndim() < 1 || scales.ndim() != codes.ndim() ||
      codes.shape(-1) % 8 != 0 || codes.shape(-1) == 0 ||
      scales.shape(-1) != codes.shape(-1) / 8) {
    std::ostringstream msg;
    msg << "[mlx_kquant.latent_fp4_unpack] expected codes [..., D / 2] and "
        << "scales [..., D / 16] with D a multiple of 16, got " << codes.shape()
        << " and " << scales.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  for (int i = 0; i + 1 < codes.ndim(); ++i) {
    if (codes.shape(i) != scales.shape(i)) {
      std::ostringstream msg;
      msg << "[mlx_kquant.latent_fp4_unpack] codes and scales disagree on "
          << "the leading dims: " << codes.shape() << " vs " << scales.shape()
          << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  if (dtype != mx::float16 && dtype != mx::bfloat16 && dtype != mx::float32) {
    std::ostringstream msg;
    msg << "[mlx_kquant.latent_fp4_unpack] dtype must be fp16/bf16/fp32, got "
        << dtype << ".";
    throw std::invalid_argument(msg.str());
  }
  auto cc = mx::contiguous(codes, false, s);
  auto sc = mx::contiguous(scales, false, s);
  auto shape = cc.shape();
  shape.back() *= 2;
  return mx::array(
      std::move(shape),
      dtype,
      std::make_shared<KQLatentFp4Unpack>(s, dtype),
      {std::move(cc), std::move(sc)});
}

} // namespace mlx_kquant
