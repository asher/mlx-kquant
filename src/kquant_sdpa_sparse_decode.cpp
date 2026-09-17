// KQSdpaSparseDecode: decode attention over a window plus index-listed pool
// rows (K == V), split over keys with a renormalizing merge. See
// kq_sdpa_sparse_decode.h. Inference-only (no CPU eval).
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "../metal/mlx/backend/metal/kernels/kq_sdpa_sparse_decode_params.h"
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {
constexpr int kHeadGroup = 8; // heads per split threadgroup
constexpr int kKeyBlock = 8; // keys per staged block
constexpr int kMaxSplits = 32;
// Queries per call: a batch dimension of the grid, bounded by the f32
// partials [B, L, splits, Hp, D] the merge reads back.
constexpr int kMaxQueries = 4096;
} // namespace

#ifdef _METAL_

void KQSdpaSparseDecode::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& q = inputs[0];
  const auto& win = inputs[1];
  const auto& pool = inputs[2];
  const auto& idx = inputs[3];
  size_t next = 4;
  const mx::array* sinks = has_sinks_ ? &inputs[next++] : nullptr;
  const mx::array* wmask = has_win_mask_ ? &inputs[next++] : nullptr;
  const mx::array* smask = has_sel_mask_ ? &inputs[next++] : nullptr;
  auto& o = outputs[0];
  o.set_data(mx::allocator::malloc(o.nbytes()));

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int L = q.shape(2);
  const int D = q.shape(3);
  const int W = win.shape(2);
  const int P = pool.shape(1);
  const int N = idx.shape(2);
  const int total = W + N;

  // A decode step spreads its few queries over 8-head groups; a prefill
  // block has queries to spare, so 16 heads share each staged key block.
  // KQ_SDPA_SPARSE_HG=8|16|32|64 forces one (A/B lever).
  int hg = L > 1 ? 2 * kHeadGroup : kHeadGroup;
  {
    static const char* env = std::getenv("KQ_SDPA_SPARSE_HG");
    const int v = env ? std::atoi(env) : 0;
    if (v == 8 || v == 16 || v == 32 || v == 64) {
      hg = v;
    }
  }
  // Enough threadgroups to cover the GPU: head groups x (B * L) x splits.
  // Splits hold whole 8-key blocks, so the count can land under the ask.
  const int hgroups = (H + hg - 1) / hg;
  const int Hp = hgroups * hg;
  int n_splits = splits_;
  if (n_splits <= 0) {
    static const char* env = std::getenv("KQ_SDPA_SPARSE_SPLITS");
    n_splits = env ? std::atoi(env) : 0;
  }
  if (n_splits <= 0) {
    const int groups = hgroups * B * L;
    n_splits = (160 + groups - 1) / groups;
  }
  n_splits = std::max(1, std::min({n_splits, kMaxSplits, std::max(1, total)}));
  int keys_per_split = (total + n_splits - 1) / n_splits;
  keys_per_split = ((keys_per_split + kKeyBlock - 1) / kKeyBlock) * kKeyBlock;
  n_splits = std::max(1, (total + keys_per_split - 1) / keys_per_split);

  auto bcast = [](const mx::array& a, int axis) -> int64_t {
    return a.shape(axis) == 1 ? 0 : a.strides(axis);
  };
  KQSdpaSparseDecodeParams params{};
  params.B = B;
  params.H = H;
  params.Hp = Hp;
  params.L = L;
  params.W = W;
  params.P = P;
  params.N = N;
  params.n_splits = n_splits;
  params.keys_per_split = keys_per_split;
  params.has_sinks = sinks != nullptr;
  params.has_win_mask = wmask != nullptr;
  params.has_sel_mask = smask != nullptr;
  params.direct = n_splits == 1;
  params.scale_log2 = scale_ * 1.4426950408889634f;
  params.q_strides[0] = q.strides(0);
  params.q_strides[1] = q.strides(1);
  params.q_strides[2] = q.strides(2);
  params.win_strides[0] = bcast(win, 0);
  params.win_strides[1] = win.strides(2);
  params.pool_strides[0] = bcast(pool, 0);
  params.pool_strides[1] = pool.strides(1);
  params.idx_strides[0] = bcast(idx, 0);
  params.idx_strides[1] = idx.strides(1);
  if (wmask) {
    params.win_mask_strides[0] = bcast(*wmask, 0);
    params.win_mask_strides[1] = wmask->strides(1);
  }
  if (smask) {
    params.sel_mask_strides[0] = bcast(*smask, 0);
    params.sel_mask_strides[1] = smask->strides(1);
  }
  params.o_strides[0] = o.strides(0);
  params.o_strides[1] = o.strides(1);
  params.o_strides[2] = o.strides(2);

  auto& ce = mx::metal::get_command_encoder(s);
  const std::string ts = kq_type_string(q.dtype());
  const std::string is = idx.dtype() == mx::int32 ? "i32" : "u32";
  const std::string dtag = "_d" + std::to_string(D);
  auto split_kernel = kq_get_kernel(
      d,
      "kq_sdpa_sparse_decode_split_" + ts + "_" + is + dtag + "_hg" +
          std::to_string(hg));
  ce.set_compute_pipeline_state(split_kernel);
  ce.set_input_array(q, 0);
  ce.set_input_array(win, 1);
  ce.set_input_array(pool, 2);
  ce.set_input_array(idx, 3);
  // Absent optionals bind a placeholder the kernel never reads.
  ce.set_input_array(wmask ? *wmask : idx, 4);
  ce.set_input_array(smask ? *smask : idx, 5);
  ce.set_bytes(params, 9);
  ce.set_input_array(sinks ? *sinks : q, 10);
  ce.set_output_array(o, 11);
  if (params.direct) {
    // No partials: the split kernel writes O and never touches 6-8.
    ce.set_input_array(q, 6);
    ce.set_input_array(q, 7);
    ce.set_input_array(q, 8);
    ce.dispatch_threadgroups(
        MTL::Size(n_splits, hgroups, B * L), MTL::Size(hg / 4 * 32, 1, 1));
    return;
  }

  mx::array oacc({B, L, n_splits, Hp, D}, mx::float32, nullptr, {});
  mx::array ms({B, L, n_splits, Hp}, mx::float32, nullptr, {});
  mx::array ls({B, L, n_splits, Hp}, mx::float32, nullptr, {});
  oacc.set_data(mx::allocator::malloc(oacc.nbytes()));
  ms.set_data(mx::allocator::malloc(ms.nbytes()));
  ls.set_data(mx::allocator::malloc(ls.nbytes()));
  ce.add_temporary(oacc);
  ce.add_temporary(ms);
  ce.add_temporary(ls);
  ce.set_output_array(oacc, 6);
  ce.set_output_array(ms, 7);
  ce.set_output_array(ls, 8);
  ce.dispatch_threadgroups(
      MTL::Size(n_splits, hgroups, B * L), MTL::Size(hg / 4 * 32, 1, 1));

  auto merge_kernel =
      kq_get_kernel(d, "kq_sdpa_sparse_decode_merge_" + ts + dtag);
  ce.set_compute_pipeline_state(merge_kernel);
  ce.set_input_array(oacc, 0);
  ce.set_input_array(ms, 1);
  ce.set_input_array(ls, 2);
  ce.set_input_array(sinks ? *sinks : q, 3);
  ce.set_output_array(o, 4);
  ce.set_bytes(params, 5);
  ce.dispatch_threadgroups(MTL::Size(H, L, B), MTL::Size(D / 4, 1, 1));
}

#else // !_METAL_

void KQSdpaSparseDecode::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_sparse_decode] requires a Metal build.");
}

#endif

void KQSdpaSparseDecode::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_sparse_decode] has no CPU implementation.");
}

std::vector<mx::Shape> KQSdpaSparseDecode::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQSdpaSparseDecode::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQSdpaSparseDecode&>(other);
  return scale_ == o.scale_ && splits_ == o.splits_ &&
      has_sinks_ == o.has_sinks_ && has_win_mask_ == o.has_win_mask_ &&
      has_sel_mask_ == o.has_sel_mask_;
}

mx::array sdpa_sparse_decode(
    mx::array q,
    mx::array window,
    mx::array pool,
    mx::array idx,
    float scale,
    const std::optional<mx::array>& sinks,
    const std::optional<mx::array>& win_mask,
    const std::optional<mx::array>& sel_mask,
    int splits,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.sdpa_sparse_decode]";

  if (q.ndim() != 4 || window.ndim() != 4 || pool.ndim() != 3 ||
      idx.ndim() != 3) {
    std::ostringstream msg;
    msg << op << " expected q [B, H, L, D], window [B, 1, W, D], pool "
        << "[B, P, D], idx [B, L, N]; got " << q.shape() << ", "
        << window.shape() << ", " << pool.shape() << ", " << idx.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  const int B = q.shape(0);
  const int H = q.shape(1);
  const int L = q.shape(2);
  const int D = q.shape(3);
  if (D != 128 && D != 256 && D != 512) {
    throw std::invalid_argument(
        std::string(op) + " head_dim must be 128, 256 or 512.");
  }
  if (window.shape(1) != 1 || window.shape(3) != D || pool.shape(2) != D ||
      (window.shape(0) != B && window.shape(0) != 1) ||
      (pool.shape(0) != B && pool.shape(0) != 1) ||
      (idx.shape(0) != B && idx.shape(0) != 1) || idx.shape(1) != L) {
    std::ostringstream msg;
    msg << op << " incompatible shapes: " << q.shape() << ", " << window.shape()
        << ", " << pool.shape() << ", " << idx.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (idx.dtype() != mx::int32 && idx.dtype() != mx::uint32) {
    throw std::invalid_argument(
        std::string(op) + " idx must be int32 or uint32.");
  }
  if (L < 1 || L > kMaxQueries) {
    throw std::invalid_argument(
        std::string(op) + " needs 1 <= L <= " + std::to_string(kMaxQueries) +
        ".");
  }
  if (window.shape(2) + idx.shape(2) < 1) {
    throw std::invalid_argument(std::string(op) + " needs at least one key.");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " q must be float16 or bfloat16.");
  }
  if (splits < 0 || splits > kMaxSplits) {
    throw std::invalid_argument(
        std::string(op) + " splits must be in [0, 32].");
  }

  // Unit stride on the feature axis: force contiguity rather than trusting
  // pre-eval flags (Contiguous donates when the input already is).
  auto q_c = mx::contiguous(q, false, s);
  auto win_c = mx::contiguous(mx::astype(window, dt, s), false, s);
  auto pool_c = mx::contiguous(mx::astype(pool, dt, s), false, s);
  auto idx_c = mx::contiguous(idx, false, s);
  std::vector<mx::array> inputs = {
      std::move(q_c), std::move(win_c), std::move(pool_c), std::move(idx_c)};

  if (sinks.has_value()) {
    if (sinks->size() != static_cast<size_t>(H)) {
      throw std::invalid_argument(
          std::string(op) + " sinks must have H elements.");
    }
    inputs.push_back(mx::contiguous(
        mx::astype(mx::reshape(*sinks, {H}, s), dt, s), false, s));
  }
  auto mask_in = [&](const mx::array& m, int X, const char* what) {
    if (m.dtype() != mx::bool_) {
      throw std::invalid_argument(
          std::string(op) + " " + what + " must be bool.");
    }
    const int64_t n = static_cast<int64_t>(m.size());
    if (n != int64_t(L) * X && n != int64_t(B) * L * X) {
      std::ostringstream msg;
      msg << op << " " << what << " must be [L, " << X << "] or [B, L, " << X
          << "]; got " << m.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    const int mb = n == int64_t(L) * X ? 1 : B;
    return mx::contiguous(mx::reshape(m, {mb, L, X}, s), false, s);
  };
  if (win_mask.has_value()) {
    inputs.push_back(mask_in(*win_mask, window.shape(2), "win_mask"));
  }
  if (sel_mask.has_value()) {
    inputs.push_back(mask_in(*sel_mask, idx.shape(2), "sel_mask"));
  }

  return mx::array(
      q.shape(),
      dt,
      std::make_shared<KQSdpaSparseDecode>(
          s,
          scale,
          splits,
          sinks.has_value(),
          win_mask.has_value(),
          sel_mask.has_value()),
      std::move(inputs));
}

} // namespace mlx_kquant
