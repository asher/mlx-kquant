// KQSdpaSparseDecode: decode attention over a window plus index-listed pool
// rows (K == V), split over keys with a renormalizing merge. See
// kq_sdpa_sparse_decode.h. Inference-only (no CPU eval).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

// A window [B, 1, S, D] or pool [B, P, D] as [B, rows * D]. The kernels
// address rows D apart from a per-batch base.
mx::array rows_flat(const mx::array& a, const char* op, mx::StreamOrDevice s) {
  const int nd = a.ndim();
  const int64_t rows = a.shape(nd - 2);
  const int64_t flat = rows * a.shape(nd - 1);
  if (flat > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        std::string(op) +
        " window or pool exceeds INT32_MAX elements per batch.");
  }
  return mx::reshape(a, {a.shape(0), static_cast<int>(flat)}, s);
}
} // namespace

#ifdef _METAL_

void KQSdpaSparseDecode::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& ce = mx::metal::get_command_encoder(s);

  const auto& q = inputs[0];
  const auto& win = inputs[1]; // [B|1, W * D], rows D apart
  const auto& pool = inputs[2]; // [B|1, P * D] (packed: [B|1, P * D / 2] codes)
  const auto& idx = inputs[3];
  size_t next = 4;
  const mx::array* pscales = packed_ ? &inputs[next++] : nullptr;
  const mx::array* sinks = has_sinks_ ? &inputs[next++] : nullptr;
  const mx::array* wmask = has_win_mask_ ? &inputs[next++] : nullptr;
  const mx::array* smask = has_sel_mask_ ? &inputs[next++] : nullptr;
  auto& o = outputs[0];
  o.set_data(mx::allocator::malloc(o.nbytes()));

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int L = q.shape(2);
  const int D = q.shape(3);
  const int W = win.shape(1) / D;
  const int P = packed_ ? pool.shape(1) * 2 / D : pool.shape(1) / D;
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
  params.win_strides[1] = D;
  // Packed batch stride in T units: the kernel halves it for the code
  // bytes and takes a sixteenth for the scale bytes.
  params.pool_strides[0] = bcast(pool, 0) * (packed_ ? 2 : 1);
  params.pool_strides[1] = D;
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

  const std::string ts = kq_type_string(q.dtype());
  const std::string is = idx.dtype() == mx::int32 ? "i32" : "u32";
  const std::string dtag = "_d" + std::to_string(D);
  auto split_kernel = kq_get_kernel(
      d,
      "kq_sdpa_sparse_decode_split_" + ts + "_" + is + dtag + "_hg" +
          std::to_string(hg) + (packed_ ? "_pk" : ""));
  // Register-heavy pipeline: some GPUs cap it below the dispatch width, and
  // Metal turns an oversized dispatch into silent garbage, not an error.
  const size_t tg = size_t(hg / 4) * 32;
  if (tg > split_kernel->maxTotalThreadsPerThreadgroup()) {
    throw std::runtime_error(
        "[mlx_kquant.sdpa_sparse_decode] threadgroup of " + std::to_string(tg) +
        " threads exceeds this GPU's pipeline limit (" +
        std::to_string(split_kernel->maxTotalThreadsPerThreadgroup()) + ").");
  }
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
  ce.set_input_array(pscales ? *pscales : idx, 12);
  if (params.direct) {
    // No partials: the split kernel writes O and never touches 6-8.
    ce.set_input_array(q, 6);
    ce.set_input_array(q, 7);
    ce.set_input_array(q, 8);
    ce.dispatch_threadgroups(
        MTL::Size(n_splits, hgroups, B * L), MTL::Size(tg, 1, 1));
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
      MTL::Size(n_splits, hgroups, B * L), MTL::Size(tg, 1, 1));

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
      has_sel_mask_ == o.has_sel_mask_ && packed_ == o.packed_;
}

namespace {

// The pool's row width in values: D for a T pool, 2 * codes width for a
// packed one (whose scales must then be [B|1, P, D / 16] uint8).
void check_pool(
    const char* op,
    const mx::array& pool,
    const std::optional<mx::array>& scales,
    int D) {
  if (!scales.has_value()) {
    if (pool.shape(2) != D) {
      std::ostringstream msg;
      msg << op << " pool rows must be " << D << " wide; got " << pool.shape()
          << ".";
      throw std::invalid_argument(msg.str());
    }
    return;
  }
  if (pool.dtype() != mx::uint8 || scales->dtype() != mx::uint8 ||
      scales->ndim() != 3 || pool.shape(2) != D / 2 ||
      scales->shape(2) != D / 16 || scales->shape(0) != pool.shape(0) ||
      scales->shape(1) != pool.shape(1)) {
    std::ostringstream msg;
    msg << op << " a packed pool is uint8 codes [B, P, " << D / 2
        << "] with uint8 pool_scales [B, P, " << D / 16 << "]; got "
        << pool.shape() << " " << pool.dtype() << " and " << scales->shape()
        << " " << scales->dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

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
    const std::optional<mx::array>& pool_scales,
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
  check_pool(op, pool, pool_scales, D);
  if (window.shape(1) != 1 || window.shape(3) != D ||
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

  // q and idx: force contiguity rather than trusting pre-eval flags
  // (Contiguous donates when the input already is). The window and pool
  // go through a reshape to [B, rows * D] instead: a cache's prefix slice
  // is a row-contiguous view, which Reshape keeps as a view at eval while
  // Contiguous would copy it (its buffer is larger than the view), and
  // any other layout Reshape copies.
  auto q_c = mx::contiguous(q, false, s);
  auto win_c = rows_flat(mx::astype(window, dt, s), op, s);
  auto pool_c = rows_flat(
      pool_scales.has_value() ? pool : mx::astype(pool, dt, s), op, s);
  auto idx_c = mx::contiguous(idx, false, s);
  std::vector<mx::array> inputs = {
      std::move(q_c), std::move(win_c), std::move(pool_c), std::move(idx_c)};
  if (pool_scales.has_value()) {
    inputs.push_back(rows_flat(*pool_scales, op, s));
  }

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
          sel_mask.has_value(),
          pool_scales.has_value()),
      std::move(inputs));
}

// ---------------------------------------------------------------------------
// Prefill form: one threadgroup per (head group, query), the band derived
// from the query position, no key split.

namespace {
struct PrefillCfg {
  int hg; // heads per threadgroup
  int ds; // simdgroups per 8-head subgroup (D slices)
  int kb; // keys per staged block
};
// The first entry is the default and is instantiated at every head dim;
// the rest exist at head dim 512 only (KQ_SDPA_SPARSE_PREFILL_CFG=hg,ds,kb).
constexpr PrefillCfg kPrefillCfgs[] = {
    {16, 4, 8},
    {32, 4, 8},
    {16, 2, 8},
};

PrefillCfg prefill_cfg(int D) {
  PrefillCfg cfg = kPrefillCfgs[0];
  const char* env = std::getenv("KQ_SDPA_SPARSE_PREFILL_CFG");
  if (env == nullptr || D != 512) {
    return cfg;
  }
  int hg = 0, ds = 0, kb = 0;
  if (std::sscanf(env, "%d,%d,%d", &hg, &ds, &kb) != 3) {
    return cfg;
  }
  for (const auto& c : kPrefillCfgs) {
    if (c.hg == hg && c.ds == ds && c.kb == kb) {
      return c;
    }
  }
  return cfg;
}
} // namespace

#ifdef _METAL_

void KQSdpaSparsePrefill::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& ce = mx::metal::get_command_encoder(s);

  const auto& q = inputs[0];
  const auto& win = inputs[1]; // [B|1, S * D], rows D apart
  const auto& pool = inputs[2]; // [B|1, P * D] (packed: [B|1, P * D / 2] codes)
  const auto& idx = inputs[3];
  size_t next = 4;
  const mx::array* pscales = packed_ ? &inputs[next++] : nullptr;
  const mx::array* sinks = has_sinks_ ? &inputs[next++] : nullptr;
  const mx::array* smask = has_sel_mask_ ? &inputs[next++] : nullptr;
  auto& o = outputs[0];
  o.set_data(mx::allocator::malloc(o.nbytes()));

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int L = q.shape(2);
  const int D = q.shape(3);
  const int S = win.shape(1) / D;
  const int P = packed_ ? pool.shape(1) * 2 / D : pool.shape(1) / D;
  const int N = idx.shape(2);
  const PrefillCfg cfg = prefill_cfg(D);
  const int hgroups = (H + cfg.hg - 1) / cfg.hg;

  auto bcast = [](const mx::array& a, int axis) -> int64_t {
    return a.shape(axis) == 1 ? 0 : a.strides(axis);
  };
  KQSdpaSparsePrefillParams params{};
  params.B = B;
  params.H = H;
  params.L = L;
  params.S = S;
  params.P = P;
  params.N = N;
  params.band = band_;
  params.koff = S - L;
  params.has_sinks = sinks != nullptr;
  params.has_sel_mask = smask != nullptr;
  params.scale_log2 = scale_ * 1.4426950408889634f;
  params.q_strides[0] = q.strides(0);
  params.q_strides[1] = q.strides(1);
  params.q_strides[2] = q.strides(2);
  params.win_strides[0] = bcast(win, 0);
  params.win_strides[1] = D;
  params.pool_strides[0] = bcast(pool, 0) * (packed_ ? 2 : 1);
  params.pool_strides[1] = D;
  params.idx_strides[0] = bcast(idx, 0);
  params.idx_strides[1] = idx.strides(1);
  if (smask) {
    params.sel_mask_strides[0] = bcast(*smask, 0);
    params.sel_mask_strides[1] = smask->strides(1);
  }
  params.o_strides[0] = o.strides(0);
  params.o_strides[1] = o.strides(1);
  params.o_strides[2] = o.strides(2);

  const std::string ts = kq_type_string(q.dtype());
  const std::string is = idx.dtype() == mx::int32 ? "i32" : "u32";
  auto kernel = kq_get_kernel(
      d,
      "kq_sdpa_sparse_prefill_" + ts + "_" + is + "_d" + std::to_string(D) +
          "_hg" + std::to_string(cfg.hg) + "_ds" + std::to_string(cfg.ds) +
          "_kb" + std::to_string(cfg.kb) + (packed_ ? "_pk" : ""));
  // Register-heavy pipeline: some GPUs cap it below the dispatch width, and
  // Metal turns an oversized dispatch into silent garbage, not an error.
  const size_t tg = size_t(cfg.hg / 8) * cfg.ds * 32;
  if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
    throw std::runtime_error(
        "[mlx_kquant.sdpa_sparse_prefill] threadgroup of " +
        std::to_string(tg) + " threads exceeds this GPU's pipeline limit (" +
        std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
  }
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(q, 0);
  ce.set_input_array(win, 1);
  ce.set_input_array(pool, 2);
  ce.set_input_array(idx, 3);
  // Absent optionals bind a placeholder the kernel never reads.
  ce.set_input_array(smask ? *smask : idx, 4);
  ce.set_input_array(sinks ? *sinks : q, 5);
  ce.set_output_array(o, 6);
  ce.set_bytes(params, 7);
  ce.set_input_array(pscales ? *pscales : idx, 8);
  ce.dispatch_threadgroups(MTL::Size(1, hgroups, B * L), MTL::Size(tg, 1, 1));
}

#else // !_METAL_

void KQSdpaSparsePrefill::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_sparse_prefill] requires a Metal build.");
}

#endif

void KQSdpaSparsePrefill::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_sparse_prefill] has no CPU implementation.");
}

std::vector<mx::Shape> KQSdpaSparsePrefill::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQSdpaSparsePrefill::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQSdpaSparsePrefill&>(other);
  return scale_ == o.scale_ && band_ == o.band_ && has_sinks_ == o.has_sinks_ &&
      has_sel_mask_ == o.has_sel_mask_ && packed_ == o.packed_;
}

mx::array sdpa_sparse_prefill(
    mx::array q,
    mx::array window,
    mx::array pool,
    mx::array idx,
    float scale,
    int band,
    const std::optional<mx::array>& sinks,
    const std::optional<mx::array>& sel_mask,
    const std::optional<mx::array>& pool_scales,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.sdpa_sparse_prefill]";

  if (q.ndim() != 4 || window.ndim() != 4 || pool.ndim() != 3 ||
      idx.ndim() != 3) {
    std::ostringstream msg;
    msg << op << " expected q [B, H, L, D], window [B, 1, S, D], pool "
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
  check_pool(op, pool, pool_scales, D);
  if (window.shape(1) != 1 || window.shape(3) != D ||
      (window.shape(0) != B && window.shape(0) != 1) ||
      (pool.shape(0) != B && pool.shape(0) != 1) ||
      (idx.shape(0) != B && idx.shape(0) != 1) || idx.shape(1) != L) {
    std::ostringstream msg;
    msg << op << " incompatible shapes: " << q.shape() << ", " << window.shape()
        << ", " << pool.shape() << ", " << idx.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (L < 1 || window.shape(2) < L) {
    std::ostringstream msg;
    msg << op << " window needs at least L rows (the queries' own); got "
        << window.shape() << " for L = " << L << ".";
    throw std::invalid_argument(msg.str());
  }
  if (band < 1) {
    throw std::invalid_argument(std::string(op) + " band must be >= 1.");
  }
  if (idx.dtype() != mx::int32 && idx.dtype() != mx::uint32) {
    throw std::invalid_argument(
        std::string(op) + " idx must be int32 or uint32.");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " q must be float16 or bfloat16.");
  }

  // As in sdpa_sparse_decode: the window and pool pass as row views.
  auto q_c = mx::contiguous(q, false, s);
  auto win_c = rows_flat(mx::astype(window, dt, s), op, s);
  auto pool_c = rows_flat(
      pool_scales.has_value() ? pool : mx::astype(pool, dt, s), op, s);
  auto idx_c = mx::contiguous(idx, false, s);
  std::vector<mx::array> inputs = {
      std::move(q_c), std::move(win_c), std::move(pool_c), std::move(idx_c)};
  if (pool_scales.has_value()) {
    inputs.push_back(rows_flat(*pool_scales, op, s));
  }

  if (sinks.has_value()) {
    if (sinks->size() != static_cast<size_t>(H)) {
      throw std::invalid_argument(
          std::string(op) + " sinks must have H elements.");
    }
    inputs.push_back(mx::contiguous(
        mx::astype(mx::reshape(*sinks, {H}, s), dt, s), false, s));
  }
  if (sel_mask.has_value()) {
    const int N = idx.shape(2);
    if (sel_mask->dtype() != mx::bool_) {
      throw std::invalid_argument(std::string(op) + " sel_mask must be bool.");
    }
    const int64_t n = static_cast<int64_t>(sel_mask->size());
    if (n != int64_t(L) * N && n != int64_t(B) * L * N) {
      std::ostringstream msg;
      msg << op << " sel_mask must be [L, " << N << "] or [B, L, " << N
          << "]; got " << sel_mask->shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    const int mb = n == int64_t(L) * N ? 1 : B;
    inputs.push_back(
        mx::contiguous(mx::reshape(*sel_mask, {mb, L, N}, s), false, s));
  }

  return mx::array(
      q.shape(),
      dt,
      std::make_shared<KQSdpaSparsePrefill>(
          s,
          scale,
          band,
          sinks.has_value(),
          sel_mask.has_value(),
          pool_scales.has_value()),
      std::move(inputs));
}

} // namespace mlx_kquant
