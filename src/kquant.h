// Public C++ surface for mlx-kquant: the kq.* ops and their Primitive
// subclasses. M0 shipped the toolchain self-checks; M1 adds dequantize.
// quantized_matmul / gather_qmm / quantize land in later milestones.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace mlx_kquant {

// ----------------------------- toolchain self-checks (M0) -----------------------------

// Directory containing the bundled mlx_kquant.metallib (colocated with _ext.so).
std::string metallib_dir();

// Load the bundled metallib via MLX's Metal device. Throws on failure.
// Returns false only on a non-Metal build.
bool metallib_loads();

// ----------------------------- ops -----------------------------

// Dequantize GGUF K-quant wire bytes `w` (uint8, last dim a multiple of the
// codec's bytes_per_block) into a float array. `scales` is a vestigial
// placeholder — K-quant scales live inside `w`; the argument exists only for
// buffer-signature parity with the Metal kernel and is otherwise ignored.
// group_size/bits are derived from `kquant_type`. Default output dtype is
// float16 (matches mlx-core's kquant dequantize default).
mx::array dequantize(
    const mx::array& w,
    const mx::array& scales,
    const std::string& kquant_type,
    std::optional<mx::Dtype> dtype = std::nullopt,
    mx::StreamOrDevice s = {});

// Quantized matmul: x (float) @ dequant(w). `w` is uint8 K-quant wire bytes,
// `scales` a vestigial placeholder (see dequantize). transpose=true means
// `w` is laid out [N, K] (the row-major weight convention). Output dtype is
// x.dtype(), except float32 x is promoted to bfloat16 (matches mlx-core).
// group_size/bits are derived from `kquant_type`.
mx::array quantized_matmul(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    bool transpose = true,
    mx::StreamOrDevice s = {});

// Gather (mixture-of-experts) quantized matmul: for each output row, select an
// expert weight matrix via `rhs_indices` and an x row via `lhs_indices`, then
// matmul. `w` is uint8 K-quant wire bytes shaped (n_experts, N, bytes_per_row);
// `scales` a vestigial placeholder (see dequantize). Either index defaults to a
// plain arange when omitted. `sorted_indices` enables an optimized path when the
// (defaulted) indices are sorted. Output dtype follows quantized_matmul.
// group_size/bits are derived from `kquant_type`.
mx::array gather_qmm(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    std::optional<mx::array> lhs_indices = std::nullopt,
    std::optional<mx::array> rhs_indices = std::nullopt,
    bool transpose = true,
    bool sorted_indices = false,
    mx::StreamOrDevice s = {});

// Quantize (encode) a float weight tensor `w` into GGUF K-quant wire bytes.
// Returns {wq (uint8), scales_placeholder (uint8, shape [1])}: K-quant scales
// live inline in `wq`, so the second output is a vestigial placeholder kept for
// signature parity with dequantize/quantized_matmul. Optional `imatrix` (a 1-D
// float32 importance vector of length K = w.shape(-1)) steers the encoder.
// group_size/bits are derived from `kquant_type`. GPU-only. The last dim of `w`
// must be a multiple of the codec's weights_per_block.
std::vector<mx::array> quantize(
    const mx::array& w,
    const std::string& kquant_type,
    const std::optional<mx::array>& imatrix = std::nullopt,
    mx::StreamOrDevice s = {});

// ----------------------------- primitives -----------------------------

// Dequantize a single uint8 K-quant wire-byte tensor. Inference-only:
// jvp/vjp/vmap inherit the base-class throwing defaults.
class KQuantDequantize : public mx::Primitive {
 public:
  explicit KQuantDequantize(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantDequantize";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
};

// Quantized matmul. Inference-only: jvp/vjp/vmap inherit the base-class
// throwing defaults. Split-k perf paths (qmm_splitk / qvm_split_k) are gated
// off — they depend on the unexported strided_reduce_general_dispatch; plain
// qmm/qvm produce identical results with less parallelism.
class KQuantMatmul : public mx::Primitive {
 public:
  explicit KQuantMatmul(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits,
      bool transpose)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits),
        transpose_(transpose) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantMatmul";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
  bool transpose_;
};

// Gather (MoE) quantized matmul. Inference-only: jvp/vjp/vmap inherit the
// base-class throwing defaults. The gather_qmm_rhs fast path (the only
// function-constant kernel) is deferred — it requires right_sorted_ == true,
// i.e. rhs_indices defaulted + sorted_indices, which the lab never does (it
// always passes rhs_indices explicitly). The dispatch falls through to
// gather_qmm / gather_qmv, which is correctness-preserving for all inputs.
class KQuantGatherQMM : public mx::Primitive {
 public:
  explicit KQuantGatherQMM(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits,
      bool transpose,
      bool left_sorted,
      bool right_sorted)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits),
        transpose_(transpose),
        left_sorted_(left_sorted),
        right_sorted_(right_sorted) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantGatherQMM";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
  bool transpose_;
  bool left_sorted_;
  bool right_sorted_;
};

// Encode a float weight tensor into K-quant wire bytes. Multi-output primitive:
// outputs[0] = wq (uint8), outputs[1] = scales placeholder (uint8, shape [1]).
// Whether an imatrix is used is derived from inputs.size() (1 = w only, 2 = w +
// imatrix), matching the fork. Inference-only: jvp/vjp/vmap throw. GPU-only.
class KQuantQuantize : public mx::Primitive {
 public:
  explicit KQuantQuantize(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantQuantize";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
};

} // namespace mlx_kquant
