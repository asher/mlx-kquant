// LoRA epilogue host side: operand validation for the op layer and the
// Metal dispatch for eval_gpu. See kquant_lora_epilogue.h.

#include <sstream>
#include <stdexcept>

#include "kquant_internal.h" // kq_type_string
#include "kquant_lora_epilogue.h"

#ifdef _METAL_
#include "kquant_metal_internal.h"
#endif

namespace mlx_kquant {

namespace {

mx::array kq_lora_contig(const mx::array& a, mx::StreamOrDevice s) {
  return a.flags().row_contiguous ? a : mx::contiguous(a, false, s);
}

std::string kq_shape_str(const mx::Shape& sh) {
  std::ostringstream m;
  m << "[";
  for (size_t i = 0; i < sh.size(); i++) {
    m << (i ? ", " : "") << sh[i];
  }
  m << "]";
  return m.str();
}

} // namespace

int kq_lora_prep(
    const char* op,
    const std::optional<mx::array>& a,
    const std::optional<mx::array>& b,
    const std::optional<mx::array>& ids,
    const std::optional<mx::array>& table,
    const std::optional<mx::array>& fac,
    mx::Dtype dtype,
    int K,
    int N,
    int n_experts,
    const mx::Shape& rows,
    int slots,
    mx::StreamOrDevice s,
    std::vector<mx::array>& inputs) {
  const std::string o(op);
  if (!a.has_value() && !b.has_value()) {
    if (ids.has_value() || table.has_value() || fac.has_value()) {
      throw std::invalid_argument(
          o + " lora_ids/lora_table/lora_rows need lora_a and lora_b.");
    }
    return 0;
  }
  if (!a.has_value() || !b.has_value()) {
    throw std::invalid_argument(o + " lora_a and lora_b go together.");
  }
  if (dtype != mx::float16 && dtype != mx::bfloat16) {
    throw std::invalid_argument(
        o + " the LoRA epilogue runs at float16/bfloat16 only.");
  }
  if (a->dtype() != dtype || b->dtype() != dtype) {
    std::ostringstream m;
    m << o << " lora_a and lora_b must carry the activation dtype (" << dtype
      << ").";
    throw std::invalid_argument(m.str());
  }
  const int nd = n_experts > 0 ? 3 : 2;
  if (a->ndim() != nd || b->ndim() != nd) {
    std::ostringstream m;
    m << o << " lora_a must be " << (nd == 3 ? "[E, K, r]" : "[K, r]")
      << " and lora_b " << (nd == 3 ? "[E, r, N]" : "[r, N]") << ".";
    throw std::invalid_argument(m.str());
  }
  if (n_experts > 0 && (a->shape(0) != n_experts || b->shape(0) != n_experts)) {
    std::ostringstream m;
    m << o << " lora_a/lora_b leading dim must equal the expert count ("
      << n_experts << "), got " << a->shape(0) << "/" << b->shape(0) << ".";
    throw std::invalid_argument(m.str());
  }
  const int rank = a->shape(-1);
  if (a->shape(-2) != K || b->shape(-2) != rank || b->shape(-1) != N) {
    std::ostringstream m;
    m << o << " LoRA shapes do not compose: lora_a [.., " << a->shape(-2)
      << ", " << rank << "], lora_b [.., " << b->shape(-2) << ", "
      << b->shape(-1) << "] against K=" << K << ", N=" << N << ".";
    throw std::invalid_argument(m.str());
  }
  if (rank < 1 || rank * slots > KQ_LORA_MAX_Z || slots > KQ_LORA_MAX_S) {
    std::ostringstream m;
    m << o << " LoRA rank " << rank << " x " << slots
      << " slots exceeds the epilogue's " << KQ_LORA_MAX_Z
      << "-float intermediate (or " << KQ_LORA_MAX_S << " slots).";
    throw std::invalid_argument(m.str());
  }
  int flags = KQ_LORA_PRESENT;
  inputs.push_back(kq_lora_contig(*a, s));
  inputs.push_back(kq_lora_contig(*b, s));
  if (ids.has_value()) {
    if (n_experts == 0) {
      throw std::invalid_argument(o + " lora_ids apply to gathered ops only.");
    }
    if (ids->shape() != rows) {
      throw std::invalid_argument(
          o + " lora_ids must match the routing indices' shape.");
    }
    flags |= KQ_LORA_HAS_IDS;
    inputs.push_back(kq_lora_contig(mx::astype(*ids, mx::uint32, s), s));
  }
  if (table.has_value()) {
    if (n_experts == 0) {
      throw std::invalid_argument(
          o + " lora_table applies to gathered ops only.");
    }
    if (table->ndim() != 1) {
      throw std::invalid_argument(
          o + " lora_table must be 1-D (slot -> expert, < 0 = dead).");
    }
    flags |= KQ_LORA_HAS_TABLE;
    inputs.push_back(kq_lora_contig(mx::astype(*table, mx::int32, s), s));
  }
  if (fac.has_value()) {
    if (fac->shape() != rows) {
      std::ostringstream m;
      m << o << " lora_rows must be shaped " << kq_shape_str(rows) << ", got "
        << kq_shape_str(fac->shape()) << ".";
      throw std::invalid_argument(m.str());
    }
    flags |= KQ_LORA_HAS_FAC;
    inputs.push_back(kq_lora_contig(mx::astype(*fac, mx::float32, s), s));
  }
  return flags;
}

KqLoraView
kq_lora_view(const std::vector<mx::array>& inputs, size_t base, int flags) {
  KqLoraView v;
  v.flags = flags;
  if (!(flags & KQ_LORA_PRESENT)) {
    return v;
  }
  size_t i = base;
  v.a = &inputs[i++];
  v.b = &inputs[i++];
  if (flags & KQ_LORA_HAS_IDS) {
    v.ids = &inputs[i++];
  }
  if (flags & KQ_LORA_HAS_TABLE) {
    v.table = &inputs[i++];
  }
  if (flags & KQ_LORA_HAS_FAC) {
    v.fac = &inputs[i++];
  }
  v.rank = v.a->shape(-1);
  return v;
}

#ifdef _METAL_

namespace {
constexpr int kq_lora_tg = 1024; // KQ_LORA_TG
constexpr int kq_lora_tile = kq_lora_tg * 1; // KQ_LORA_TG * KQ_LORA_NPT
} // namespace

void kq_lora_epilogue_rows_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& x,
    const KqLoraView& v,
    mx::array& out,
    int R,
    int K,
    int N) {
  std::string kname = "kq_lora_epilogue_rows_" + kq_type_string(out.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(*v.a, 1);
  ce.set_input_array(*v.b, 2);
  // Re-binding out as an output after the base dispatch inserts the
  // encoder barrier that orders the epilogue behind it.
  ce.set_output_array(out, 3);
  // Absent operands bind x as a placeholder; the kernel never reads them.
  ce.set_input_array(v.ids ? *v.ids : x, 4);
  ce.set_input_array(v.table ? *v.table : x, 5);
  ce.set_input_array(v.fac ? *v.fac : x, 6);
  const int rank = v.rank;
  const int flags = v.flags & KQ_LORA_KERNEL_FLAGS;
  ce.set_bytes(K, 7);
  ce.set_bytes(N, 8);
  ce.set_bytes(rank, 9);
  ce.set_bytes(flags, 10);
  MTL::Size group_dims(kq_lora_tg, 1, 1);
  MTL::Size grid_dims((N + kq_lora_tile - 1) / kq_lora_tile, R, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

mx::array kq_lora_mix_z_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& x,
    const KqLoraView& v,
    int T,
    int S,
    int K) {
  const int rank = v.rank;
  mx::array z({T * S, rank}, mx::float32, nullptr, {});
  z.set_data(mx::allocator::malloc(z.nbytes()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(z);
  std::string kname = "kq_lora_mix_z_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(*v.a, 1);
  ce.set_output_array(z, 2);
  ce.set_input_array(*v.ids, 3);
  ce.set_input_array(v.table ? *v.table : x, 4);
  const int flags = v.flags & KQ_LORA_KERNEL_FLAGS;
  ce.set_bytes(K, 5);
  ce.set_bytes(rank, 6);
  ce.set_bytes(flags, 7);
  MTL::Size group_dims(kq_lora_tg, 1, 1);
  MTL::Size grid_dims(T * S, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
  return z;
}

void kq_lora_mix_apply_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& z,
    const mx::array& scores,
    const KqLoraView& v,
    mx::array& out,
    int T,
    int S,
    int N) {
  std::string kname = "kq_lora_mix_apply_" + kq_type_string(out.dtype());
  // S is a function constant: one pipeline per slot count.
  int s_const = S;
  mx::metal::MTLFCList func_consts = {
      {&s_const, MTL::DataType::DataTypeInt, 320},
  };
  const std::string hash_name = kname + "_s" + std::to_string(S);
  auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(z, 0);
  ce.set_input_array(*v.b, 1);
  // Re-binding out as an output after the base dispatch inserts the
  // encoder barrier that orders the apply behind it (and behind z).
  ce.set_output_array(out, 2);
  ce.set_input_array(*v.ids, 3);
  ce.set_input_array(v.table ? *v.table : z, 4);
  ce.set_input_array(v.fac ? *v.fac : z, 5);
  ce.set_input_array(scores, 6);
  const int rank = v.rank;
  const int flags = v.flags & KQ_LORA_KERNEL_FLAGS;
  ce.set_bytes(N, 7);
  ce.set_bytes(rank, 8);
  ce.set_bytes(flags, 9);
  MTL::Size group_dims(kq_lora_tg, 1, 1);
  MTL::Size grid_dims((N + kq_lora_tile - 1) / kq_lora_tile, T, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#endif // _METAL_

} // namespace mlx_kquant
