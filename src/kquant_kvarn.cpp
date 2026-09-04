// KVarN KV-cache group quantize/dequant primitives (see kvarn_quantize
// and kvarn_dequant in kquant.h). Kernel bodies in kq_kvarn.h; oracle in
// tests/kvarn_ref.py. Inference-only (no CPU eval).
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string, kq_kvarn_bits_ok

#include "../metal/mlx/backend/metal/kernels/kq_kvarn_params.h"

#include "mlx/ops.h" // contiguous
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

int kvarn_record_words(int bits) {
  return KQ_KVARN_SLICE_WORDS_PER_BIT * bits;
}

} // namespace

#ifdef _METAL_

void KQKvarnQuantize::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& axes = outputs[1];
  codes.set_data(mx::allocator::malloc(codes.nbytes()));
  axes.set_data(mx::allocator::malloc(axes.nbytes()));

  const int n_tiles = int(x.size() / (KQ_KVARN_GROUP * KQ_KVARN_GROUP));
  const int vside = vside_ ? 1 : 0;

  const std::string kname = "kq_kvarn_quantize_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname, kname, {});
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  ce.set_input_array(x, 0);
  ce.set_output_array(codes, 1);
  ce.set_output_array(axes, 2);
  ce.set_bytes(bits_, 3);
  ce.set_bytes(vside, 4);
  ce.set_bytes(iterations_, 5);

  MTL::Size group_dims(KQ_KVARN_GROUP, 1, 1);
  MTL::Size grid_dims(n_tiles, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQKvarnDequant::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& codes = inputs[0];
  const auto& axes = inputs[1];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const int n_tiles = int(codes.size() / kvarn_record_words(bits_));
  const int vside = vside_ ? 1 : 0;

  const std::string kname = "kq_kvarn_dequant_" + kq_type_string(out.dtype());
  auto kernel = kq_get_kernel(d, kname, kname, {});
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  ce.set_input_array(codes, 0);
  ce.set_input_array(axes, 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(bits_, 3);
  ce.set_bytes(vside, 4);

  MTL::Size group_dims(KQ_KVARN_GROUP, 1, 1);
  MTL::Size grid_dims(n_tiles, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQKvarnQuantize::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.kvarn_quantize] requires a Metal build.");
}

void KQKvarnDequant::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.kvarn_dequant] requires a Metal build.");
}

#endif

void KQKvarnQuantize::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.kvarn_quantize] has no CPU implementation.");
}

std::vector<mx::Shape> KQKvarnQuantize::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& shape = inputs[0].shape();
  mx::Shape lead(shape.begin(), shape.end() - 2);
  const int groups = shape[shape.size() - 2] / KQ_KVARN_GROUP;
  mx::Shape codes_shape = lead;
  codes_shape.push_back(groups);
  codes_shape.push_back(KQ_KVARN_SLICE_WORDS_PER_BIT * bits_);
  mx::Shape axes_shape = std::move(lead);
  axes_shape.push_back(groups);
  axes_shape.push_back(3);
  axes_shape.push_back(KQ_KVARN_GROUP);
  return {std::move(codes_shape), std::move(axes_shape)};
}

bool KQKvarnQuantize::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQKvarnQuantize&>(other);
  return bits_ == o.bits_ && vside_ == o.vside_ && iterations_ == o.iterations_;
}

void KQKvarnDequant::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.kvarn_dequant] has no CPU implementation.");
}

std::vector<mx::Shape> KQKvarnDequant::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& shape = inputs[0].shape();
  mx::Shape out(shape.begin(), shape.end() - 1);
  const int groups = out.back();
  out.back() = groups * KQ_KVARN_GROUP;
  out.push_back(KQ_KVARN_GROUP);
  return {std::move(out)};
}

bool KQKvarnDequant::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQKvarnDequant&>(other);
  return bits_ == o.bits_ && vside_ == o.vside_;
}

std::vector<mx::array> kvarn_quantize(
    mx::array x,
    int bits,
    bool vside,
    int iterations,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (!kq_kvarn_bits_ok(bits)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.kvarn_quantize] bits must be one of 2/3/4/5/6/8, "
        << "got " << bits << ".";
    throw std::invalid_argument(msg.str());
  }
  if (iterations < 0 || iterations > 64) {
    std::ostringstream msg;
    msg << "[mlx_kquant.kvarn_quantize] iterations must be in [0, 64], got "
        << iterations << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.ndim() < 2 || x.shape(-1) != KQ_KVARN_GROUP ||
      x.shape(-2) % KQ_KVARN_GROUP != 0 || x.shape(-2) == 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.kvarn_quantize] x must be [..., T, 128] with "
        << "T a positive multiple of 128, got shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != mx::float16) {
    std::ostringstream msg;
    msg << "[mlx_kquant.kvarn_quantize] the rotated stage must be float16, "
        << "got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto xc = mx::contiguous(x, false, s);
  auto prim = std::make_shared<KQKvarnQuantize>(s, bits, vside, iterations);
  auto shapes = prim->output_shapes({xc});
  std::vector<mx::Dtype> dtypes = {mx::uint32, mx::float16};
  return mx::array::make_arrays(
      std::move(shapes), dtypes, std::move(prim), {std::move(xc)});
}

mx::array kvarn_dequant(
    mx::array codes,
    mx::array axes,
    int bits,
    bool vside,
    mx::Dtype out_dtype,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (!kq_kvarn_bits_ok(bits)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.kvarn_dequant] bits must be one of 2/3/4/5/6/8, got "
        << bits << ".";
    throw std::invalid_argument(msg.str());
  }
  if (codes.dtype() != mx::uint32 || codes.ndim() < 2 ||
      codes.shape(-1) != KQ_KVARN_SLICE_WORDS_PER_BIT * bits) {
    std::ostringstream msg;
    msg << "[mlx_kquant.kvarn_dequant] codes must be uint32 [..., G, "
        << KQ_KVARN_SLICE_WORDS_PER_BIT * bits << "] for bits = " << bits
        << ", got " << codes.dtype() << " " << codes.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (axes.dtype() != mx::float16 || axes.ndim() != codes.ndim() + 1 ||
      axes.shape(-1) != KQ_KVARN_GROUP || axes.shape(-2) != 3) {
    std::ostringstream msg;
    msg << "[mlx_kquant.kvarn_dequant] axes must be float16 [..., G, 3, "
        << KQ_KVARN_GROUP << "], got " << axes.dtype() << " " << axes.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  for (int i = 0; i + 1 < int(codes.ndim()); ++i) {
    if (axes.shape(i) != codes.shape(i)) {
      std::ostringstream msg;
      msg << "[mlx_kquant.kvarn_dequant] codes/axes leading shapes differ: "
          << codes.shape() << " vs " << axes.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  if (out_dtype != mx::float16 && out_dtype != mx::bfloat16) {
    std::ostringstream msg;
    msg << "[mlx_kquant.kvarn_dequant] out_dtype must be float16 or "
        << "bfloat16, got " << out_dtype << ".";
    throw std::invalid_argument(msg.str());
  }

  auto cc = mx::contiguous(codes, false, s);
  auto ac = mx::contiguous(axes, false, s);
  auto prim = std::make_shared<KQKvarnDequant>(s, bits, vside);
  auto shapes = prim->output_shapes({cc});
  return mx::array(
      std::move(shapes[0]),
      out_dtype,
      std::move(prim),
      {std::move(cc), std::move(ac)});
}

} // namespace mlx_kquant
