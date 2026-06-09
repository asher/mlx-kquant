// KQuantDequantize primitive: GGUF K-quant wire bytes -> float array.
// GPU dispatch ports mlx/backend/metal/quantized.cpp:1974-2013 (the kquant
// dequantize arm of fast::Quantize::eval_gpu), with two extension-specific
// changes:
//   * the kernel is fetched from OUR bundled mlx_kquant.metallib (the fork's
//     kernels live in mlx-core's default metallib), via d.get_kernel(name,
//     lib);
//   * row-contiguity is guaranteed by the op (mx::contiguous), since the fork's
//     contiguous_copy_gpu is not exported.
#include <sstream>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel (cached library handle)
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

std::string kq_type_string(mx::Dtype d) {
  if (d == mx::float32) {
    return "float";
  }
  if (d == mx::float16) {
    return "float16_t";
  }
  if (d == mx::bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "[mlx_kquant] Unsupported output dtype " << d
      << " (expected float32, float16, or bfloat16).";
  throw std::invalid_argument(msg.str());
}

std::vector<mx::Shape> KQuantDequantize::output_shapes(
    const std::vector<mx::array>& inputs) {
  const KQuantCodec* codec = codec_by_name(kquant_type_);
  // codec is guaranteed non-null: validated when the op constructed this node.
  auto shape = inputs[0].shape();
  shape.back() =
      (inputs[0].shape(-1) / codec->bytes_per_block) * codec->weights_per_block;
  return {shape};
}

bool KQuantDequantize::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantDequantize&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_;
}

void KQuantDequantize::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] dequantize has no CPU implementation yet; run on the GPU "
      "stream (the default device).");
}

#ifdef _METAL_

void KQuantDequantize::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& w = inputs[0]; // uint8 wire bytes, row-contiguous (ensured by the op)
  auto& scales = inputs[1]; // vestigial placeholder, ignored by the kernel
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& compute_encoder = mx::metal::get_command_encoder(s);

  uint32_t num_weights = static_cast<uint32_t>(out.size());

  // kquant_<codec>_dequantize_<type>_gs_<gs>_b_<bits>
  // (kq_quantized.metal:76-78)
  std::string kname = kq_kname_prefix(kquant_type_) + "dequantize_" +
      kq_type_string(out.dtype()) + "_gs_" + std::to_string(group_size_) +
      "_b_" + std::to_string(bits_);

  auto kernel = kq_get_kernel(d, kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_bytes(num_weights, 3);

  NS::UInteger tg = kernel->maxTotalThreadsPerThreadgroup();
  if (tg > static_cast<NS::UInteger>(num_weights)) {
    tg = num_weights;
  }
  MTL::Size group_dims = MTL::Size(tg, 1, 1);
  MTL::Size grid_dims = MTL::Size(num_weights, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

#else // Metal not available

void KQuantDequantize::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] dequantize has no GPU implementation.");
}

#endif

} // namespace mlx_kquant
