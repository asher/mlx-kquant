// KQuantQuantize primitive: encode a float weight tensor into GGUF K-quant wire
// bytes. The GPU path fetches the encode kernel from the bundled metallib via
// kq_get_kernel; the op guarantees row-contiguity before dispatch, and
// kernel-name type tokens come from kq_type_string. The CPU path covers the
// flat codecs (kquant_cpu_encode.h); the K-quant codecs are GPU-only for now
// and the CPU dispatch throws for them.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_cpu_encode.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/types/half_types.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h" // concatenate
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

std::vector<mx::Shape> KQuantQuantize::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& w = inputs[0];
  const KQuantCodec* codec = codec_by_name(kquant_type_);
  auto wq_shape = w.shape();
  wq_shape.back() =
      (w.shape(-1) / codec->weights_per_block) * codec->bytes_per_block;
  mx::Shape s_shape = {1};
  return {std::move(wq_shape), std::move(s_shape)};
}

bool KQuantQuantize::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQuantize&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_;
}

void KQuantQuantize::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs[0] = w (float, row-contiguous via the op); inputs[1] = imatrix (opt,
  // float32, row-contiguous via the op).
  auto& w = inputs[0];
  auto& out = outputs[0]; // wq (uint8)
  auto& scales_placeholder = outputs[1]; // vestigial [1] uint8
  out.set_data(mx::allocator::malloc(out.nbytes()));
  scales_placeholder.set_data(
      mx::allocator::malloc(scales_placeholder.nbytes()));

  bool has_imatrix = inputs.size() >= 2;
  std::size_t num_weights = static_cast<std::size_t>(w.size());
  std::size_t K = static_cast<std::size_t>(w.shape(-1));
  // Capture a real array for the imatrix slot even when absent (dummy = w,
  // never read), so the lambda has a valid weak copy to hold.
  const mx::array& imat_src = has_imatrix ? inputs[1] : w;

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(w);
  if (has_imatrix) {
    encoder.set_input_array(inputs[1]);
  }
  encoder.set_output_array(out);
  encoder.dispatch([w = mx::array::unsafe_weak_copy(w),
                    imat = mx::array::unsafe_weak_copy(imat_src),
                    out = mx::array::unsafe_weak_copy(out),
                    has_imatrix,
                    num_weights,
                    K,
                    kquant_type = kquant_type_]() mutable {
    const float* imatrix = has_imatrix ? imat.data<float>() : nullptr;
    uint8_t* outp = out.data<uint8_t>();
    auto dt = w.dtype();
    if (dt == mx::float32) {
      kquant_quantize_dispatch(
          w.data<float>(), outp, num_weights, kquant_type, imatrix, K);
    } else if (dt == mx::float16) {
      kquant_quantize_dispatch(
          w.data<mx::float16_t>(), outp, num_weights, kquant_type, imatrix, K);
    } else if (dt == mx::bfloat16) {
      kquant_quantize_dispatch(
          w.data<mx::bfloat16_t>(), outp, num_weights, kquant_type, imatrix, K);
    } else {
      throw std::runtime_error(
          "[mlx_kquant] quantize: only float32/float16/bfloat16 inputs are "
          "supported.");
    }
  });
}

#ifdef _METAL_

void KQuantQuantize::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0]; // wq (uint8)
  auto& scales_placeholder = outputs[1]; // vestigial [1] uint8
  out.set_data(mx::allocator::malloc(out.nbytes()));
  scales_placeholder.set_data(
      mx::allocator::malloc(scales_placeholder.nbytes()));

  // inputs[0] = w (float, row-contiguous via the op); inputs[1] = imatrix (opt,
  // float32, row-contiguous via the op).
  const auto& w_contig = inputs[0];
  const KQuantCodec* codec = codec_by_name(kquant_type_);
  uint32_t num_blocks =
      static_cast<uint32_t>(out.size() / codec->bytes_per_block);

  auto& ce = mx::metal::get_command_encoder(s);

  int c = 0;
  ce.set_input_array(w_contig, c++); // 0: w
  ce.set_output_array(out, c++); // 1: wq
  ce.set_bytes(num_blocks, c++); // 2

  uint32_t has_imatrix = 0;
  uint32_t K = static_cast<uint32_t>(w_contig.shape(-1));
  if (inputs.size() >= 2) {
    ce.set_input_array(inputs[1], 3); // 3: imatrix (float32)
    has_imatrix = 1;
  } else {
    ce.set_input_array(w_contig, 3); // 3: dummy (unread when has_imatrix==0)
  }
  ce.set_bytes(has_imatrix, 4);
  ce.set_bytes(K, 5);

  std::string type_string = kq_type_string(w_contig.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type_) + "quantize_",
      type_string,
      "_gs_",
      group_size_,
      "_b_",
      bits_);

  auto kernel = kq_get_kernel(d, kname);
  ce.set_compute_pipeline_state(kernel);

  // K-codecs (gs >= 256): one 256-thread threadgroup per super-block.
  // Flat codecs (gs == 32): one thread per block.
  if (group_size_ >= 256) {
    MTL::Size group_dims(256, 1, 1);
    MTL::Size grid_dims(num_blocks, 1, 1);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  } else {
    NS::UInteger tg = kernel->maxTotalThreadsPerThreadgroup();
    if (tg > static_cast<NS::UInteger>(num_blocks)) {
      tg = num_blocks;
    }
    MTL::Size group_dims(tg, 1, 1);
    MTL::Size grid_dims(num_blocks, 1, 1);
    ce.dispatch_threads(grid_dims, group_dims);
  }
}

#else

void KQuantQuantize::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] quantize (encode) has no GPU implementation.");
}

#endif // _METAL_

} // namespace mlx_kquant
