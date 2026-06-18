// KQuantGatherQMM primitive: mixture-of-experts quantized matmul. The GPU path
// dispatches the gather leaf kernels (gather_qmm / gather_qmm_nax / gather_qmv)
// from the bundled metallib via kq_get_kernel; the op guarantees row-contiguity
// before dispatch, kernel-name type tokens come from kq_type_string, NAX
// availability is probed via kq_is_nax_available, and kquant carries no biases
// so no bias buffer is plumbed through.
//
// The gather_qmm_rhs fast path is implemented here as gather_qmm_rhs_nax - the
// only function-constant kernel (align_M/N/K at constant ids 200/201/202). It
// requires right_sorted_ == true, which holds when lhs_indices is defaulted AND
// sorted_indices is requested. mlx-lm's SwitchGLU sorts tokens by expert
// (do_sort when indices.size>=64) and passes rhs_indices only, so
// right_sorted_ == do_sort: MoE PREFILL takes this sorted per-expert GEMM
// (~=6-8x faster than B separate gather_qmv vector-matmuls), while decode
// (top_k<64 -> no sort -> B<16) falls through to gather_qmv.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_cpu_decode.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h" // elem_to_loc
#include "mlx/backend/cpu/encoder.h"
#include "mlx/types/half_types.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // shared dispatch helpers
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h" // concatenate
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

namespace {

using mx::Stream;
// array / Device / CommandEncoder and the shared dispatch helpers come from
// kquant_metal_internal.h.

// NAX (tensor-core) GEMM dispatch for the gathered (MoE) matmul (no biases).
void gather_qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int B = out.size() / M / N;
  int wm = 2, wn = 2, bm = 64, bn = 64, bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % bn == 0;
  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (transpose ? "gather_qmm_t_nax_" : "gather_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(lhs_indices, c++);
  ce.set_input_array(rhs_indices, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  c = add_strides_and_shapes(ce, false, x, w, scales, c);
  add_gather_strides_and_shapes(ce, lhs_indices, rhs_indices, c);

  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Tiled gathered (MoE) quantized GEMM dispatch (no biases).
void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  // enable_tf32 dropped (always short-circuited; op promotes f32 x -> bf16).
  if (kq_is_nax_available() && transpose && (K % 64 == 0) &&
      (x.dtype() != mx::float32) && codec_has_nax(kquant_type)) {
    return gather_qmm_nax(
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        out,
        transpose,
        group_size,
        bits,
        M,
        N,
        K,
        d,
        s,
        kquant_type);
  }

  int B = out.size() / M / N;
  int wm = 2, wn = 2, bm = 32, bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % 32 == 0;
  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (transpose ? "gather_qmm_t_" : "gather_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(lhs_indices, c++);
  ce.set_input_array(rhs_indices, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  c = add_strides_and_shapes(ce, false, x, w, scales, c);
  add_gather_strides_and_shapes(ce, lhs_indices, rhs_indices, c);

  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Gathered (MoE) matrix-times-vector quantized kernel dispatch (no biases).
void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int B = out.size() / M / N;
  int bn = kquant_qmv_bn(kquant_type);
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string type_string = kq_type_string(x.dtype());
  int k_align = qmv_fast_k_align();
  bool fast = (N % bn == 0) && (K % k_align == 0);
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (fast ? "gather_qmv_fast_" : "gather_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(lhs_indices, c++);
  ce.set_input_array(rhs_indices, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  c = add_strides_and_shapes(ce, false, x, w, scales, c);
  add_gather_strides_and_shapes(ce, lhs_indices, rhs_indices, c);

  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Sorted-rhs fast path (NAX-only, no biases): x rows are pre-sorted by expert
// (lhs_indices defaulted), so a single batched GEMM walks contiguous per-expert
// row blocks, switching the weight matrix per row-block from the sorted
// `indices`. M here is the TOTAL row count (x.size()/K), NOT x.shape(-2)==1.
// Unlike the other gather leaves this passes no index strides - it requires
// row-contiguous x / indices (the caller guards this) and bakes align_M/N/K
// into func consts 200/201/202. kquant has no non-NAX gather_qmm_rhs kernel, so
// this is only ever reached when the NAX gate holds.
void gather_qmm_rhs_nax(
    const array& x,
    const array& w,
    const array& scales,
    const array& indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int bm = 64, bn = 64, bk = 64, wm = 2, wn = 2;
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (transpose ? "gather_qmm_rhs_nax_nt_" : "gather_qmm_rhs_nax_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  mx::metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  std::string hash_name;
  hash_name.reserve(128);
  mx::concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  int c = 0;
  ce.set_input_array(x, c++);
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(indices, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(M, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(K, c++);

  ce.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

#endif // _METAL_

std::vector<mx::Shape> KQuantGatherQMM::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& lhs_indices = inputs[3];
  const KQuantCodec* codec = codec_by_name(kquant_type_);
  int weights_per_row =
      (w.shape(-1) / codec->bytes_per_block) * codec->weights_per_block;
  int N = transpose_ ? w.shape(-2) : weights_per_row;
  auto shape = lhs_indices.shape();
  shape.push_back(x.shape(-2));
  shape.push_back(N);
  return {shape};
}

bool KQuantGatherQMM::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMM&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_ && transpose_ == o.transpose_ &&
      left_sorted_ == o.left_sorted_ && right_sorted_ == o.right_sorted_;
}

void KQuantGatherQMM::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: x, w (uint8), scales placeholder (ignored), lhs_indices,
  // rhs_indices. x / w matrix-contiguous by the op; indices walked via
  // elem_to_loc. One output row [M, N] per lhs_indices element.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& lhs_indices = inputs[3];
  const auto& rhs_indices = inputs[4];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);
  encoder.dispatch([out = mx::array::unsafe_weak_copy(out),
                    x = mx::array::unsafe_weak_copy(x),
                    w = mx::array::unsafe_weak_copy(w),
                    lhs_indices = mx::array::unsafe_weak_copy(lhs_indices),
                    rhs_indices = mx::array::unsafe_weak_copy(rhs_indices),
                    transpose_ = transpose_,
                    kquant_type = kquant_type_]() mutable {
    int K = x.shape(-1);
    int M = x.shape(-2);
    int N = out.shape(-1);
    std::size_t w_els = static_cast<std::size_t>(w.shape(-1)) * w.shape(-2);
    const uint32_t* lhs_ptr = lhs_indices.data<uint32_t>();
    const uint32_t* rhs_ptr = rhs_indices.data<uint32_t>();
    int n_rows = static_cast<int>(lhs_indices.size());

    // Group the index entries by expert (w_idx) so each expert's wire bytes
    // are decoded ONCE per call instead of once per entry. In sorted MoE
    // prefill there is one entry per (token, expert) pair, so the naive
    // entry-at-a-time loop re-dequantizes every expert per token - the
    // dominant cost by far. Grouped, all of an expert's rows are packed into
    // one [Rg*M, K] activation block and run as a single qmm (which threads
    // internally and picks fused-GEMV or dequant-once+GEMM by row count).
    struct Entry {
      int i;
      int x_idx;
      int w_idx;
    };
    std::vector<Entry> entries(n_rows);
    for (int i = 0; i < n_rows; i++) {
      entries[i].i = i;
      entries[i].x_idx = static_cast<int>(lhs_ptr[mx::elem_to_loc(
          i, lhs_indices.shape(), lhs_indices.strides())]);
      entries[i].w_idx = static_cast<int>(rhs_ptr[mx::elem_to_loc(
          i, rhs_indices.shape(), rhs_indices.strides())]);
    }
    std::stable_sort(
        entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
          return a.w_idx < b.w_idx;
        });

    auto gather_loop = [&](auto* tag) {
      using T = std::remove_pointer_t<decltype(tag)>;
      const std::size_t row_els = static_cast<std::size_t>(M) * K;
      const std::size_t out_row_els = static_cast<std::size_t>(M) * N;

      // Group boundaries [first, last) over the expert-sorted entries.
      std::vector<std::pair<std::size_t, std::size_t>> groups;
      {
        std::size_t ga = 0;
        while (ga < entries.size()) {
          std::size_t gb = ga;
          while (gb < entries.size() &&
                 entries[gb].w_idx == entries[ga].w_idx) {
            gb++;
          }
          groups.emplace_back(ga, gb);
          ga = gb;
        }
      }
      auto group_w = [&](std::size_t first) {
        return w.data<uint8_t>() +
            mx::elem_to_loc(
                   entries[first].w_idx * w_els, w.shape(), w.strides());
      };

      // Decode-shape consolidation: when every per-expert group is small
      // enough for the fused GEMV, run the whole call as ONE parallel job
      // over all (expert, output-row) work items via kquant_qmm_cpu_batch.
      // The per-group qmm-call-per-expert path below would pay a thread-pool
      // wake/teardown per (expert, matrix) on ~MBs of work each - at MoE
      // decode that's the dominant dispatch overhead.
      // Per-group dedupe plan: duplicate (x_idx, w_idx) entries - the same
      // activation rows against the same expert - compute once and fan out
      // by memcpy at scatter time, so repeated expert indices save FLOPs as
      // well as weight traffic. Admission stays gated on the RAW entry
      // count (not the unique count): widening it would shift groups from
      // the dequant-once GEMM path onto the fused GEMV - a numerically
      // valid but different accumulation order that would break the
      // bit-stability of existing call shapes.
      struct GroupPlan {
        std::vector<int> uniq; // unique x_idx, first-seen order
        std::vector<int> slot; // per group entry -> index into uniq
      };
      std::vector<GroupPlan> plans;
      bool all_small = transpose_;
      if (all_small) {
        for (const auto& g : groups) {
          if ((g.second - g.first) * static_cast<std::size_t>(M) >
              static_cast<std::size_t>(kQmvFusedMaxM)) {
            all_small = false;
            break;
          }
        }
      }
      if (all_small) {
        plans.resize(groups.size());
        for (std::size_t gi = 0; gi < groups.size(); gi++) {
          const auto [ga, gb] = groups[gi];
          GroupPlan& plan = plans[gi];
          plan.slot.reserve(gb - ga);
          for (std::size_t r = ga; r < gb; r++) {
            const int xi = entries[r].x_idx;
            std::size_t u = 0;
            while (u < plan.uniq.size() && plan.uniq[u] != xi) {
              u++;
            }
            if (u == plan.uniq.size()) {
              plan.uniq.push_back(xi);
            }
            plan.slot.push_back(static_cast<int>(u));
          }
        }
      }
      if (all_small) {
        std::vector<KQmvTask<T>> tasks(groups.size());
        std::vector<std::vector<T>> xg_packs; // keep packed copies alive
        std::vector<std::vector<T>> og_packs;
        for (std::size_t gi = 0; gi < groups.size(); gi++) {
          const auto [ga, gb] = groups[gi];
          const std::size_t rg = gb - ga;
          const std::size_t ru = plans[gi].uniq.size();
          KQmvTask<T>& task = tasks[gi];
          task.w = group_w(ga);
          task.m = static_cast<int>(ru) * M;
          if (rg == 1) {
            task.out =
                out.data<T>() + static_cast<std::size_t>(entries[ga].i) * M * N;
            task.x = x.data<T>() +
                mx::elem_to_loc(
                         entries[ga].x_idx * M * K, x.shape(), x.strides());
          } else if (ru == 1) {
            // One unique row: read x in place, fan the output out below.
            og_packs.emplace_back(out_row_els);
            task.x = x.data<T>() +
                mx::elem_to_loc(
                         plans[gi].uniq[0] * M * K, x.shape(), x.strides());
            task.out = og_packs.back().data();
          } else {
            xg_packs.emplace_back(ru * row_els);
            og_packs.emplace_back(ru * out_row_els);
            for (std::size_t u = 0; u < ru; u++) {
              std::memcpy(
                  xg_packs.back().data() + u * row_els,
                  x.data<T>() +
                      mx::elem_to_loc(
                          plans[gi].uniq[u] * M * K, x.shape(), x.strides()),
                  row_els * sizeof(T));
            }
            task.x = xg_packs.back().data();
            task.out = og_packs.back().data();
          }
        }
        kquant_qmm_cpu_batch<T>(
            tasks.data(), static_cast<int>(tasks.size()), N, K, kquant_type);
        // Scatter packed outputs to entry rows (duplicates share a slot).
        std::size_t pi = 0;
        for (std::size_t gi = 0; gi < groups.size(); gi++) {
          const auto [ga, gb] = groups[gi];
          const std::size_t rg = gb - ga;
          if (rg == 1) {
            continue;
          }
          for (std::size_t r = 0; r < rg; r++) {
            std::memcpy(
                out.data<T>() +
                    static_cast<std::size_t>(entries[ga + r].i) * M * N,
                og_packs[pi].data() +
                    static_cast<std::size_t>(plans[gi].slot[r]) * out_row_els,
                out_row_els * sizeof(T));
          }
          pi++;
        }
        return;
      }

      std::vector<T> xg;
      std::vector<T> og;
      for (const auto& [a, b] : groups) {
        const std::size_t rg = b - a;
        const uint8_t* wp = group_w(a);
        if (rg == 1) {
          // Single entry for this expert (the decode shape): no packing.
          kquant_qmm_cpu<T>(
              out.data<T>() + static_cast<std::size_t>(entries[a].i) * M * N,
              x.data<T>() +
                  mx::elem_to_loc(
                      entries[a].x_idx * M * K, x.shape(), x.strides()),
              wp,
              M,
              N,
              K,
              transpose_,
              kquant_type);
        } else {
          xg.resize(rg * row_els);
          og.resize(rg * out_row_els);
          for (std::size_t r = 0; r < rg; r++) {
            std::memcpy(
                xg.data() + r * row_els,
                x.data<T>() +
                    mx::elem_to_loc(
                        entries[a + r].x_idx * M * K, x.shape(), x.strides()),
                row_els * sizeof(T));
          }
          kquant_qmm_cpu<T>(
              og.data(),
              xg.data(),
              wp,
              static_cast<int>(rg) * M,
              N,
              K,
              transpose_,
              kquant_type);
          for (std::size_t r = 0; r < rg; r++) {
            std::memcpy(
                out.data<T>() +
                    static_cast<std::size_t>(entries[a + r].i) * M * N,
                og.data() + r * out_row_els,
                out_row_els * sizeof(T));
          }
        }
      }
    };
    auto dt = x.dtype();
    if (dt == mx::float32) {
      gather_loop(static_cast<float*>(nullptr));
    } else if (dt == mx::float16) {
      gather_loop(static_cast<mx::float16_t*>(nullptr));
    } else if (dt == mx::bfloat16) {
      gather_loop(static_cast<mx::bfloat16_t*>(nullptr));
    } else {
      throw std::runtime_error(
          "[mlx_kquant] gather_qmm: only float32/float16/bfloat16 inputs are "
          "supported.");
    }
  });
}

std::vector<mx::array> KQuantGatherQMM::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  // primals = {x, w (wire bytes), scales placeholder, lhs_indices,
  // rhs_indices}. Only the gradient wrt x is defined: gather the cotangent
  // against the experts with the transpose flipped, then scatter-add each
  // output row's gradient back onto its source x row via lhs_indices. When the
  // indices are sorted and there is one x row per output row, the gather alone
  // already lands the gradient in place. The quantized base is frozen (the LoRA
  // use case), so the weight/scale branches throw; the gradient wrt the indices
  // is undefined.
  std::vector<mx::array> vjps;
  const auto& x = primals[0];
  const auto& w = primals[1];
  const auto& scales = primals[2];
  const auto& lhs_indices = primals[3];
  const auto& rhs_indices = primals[4];
  const auto& cotan = cotangents[0];

  int M = cotan.shape(-2);
  int K = x.shape(-1);
  bool sorted = left_sorted_ || right_sorted_;
  bool no_broadcast =
      rhs_indices.size() * static_cast<size_t>(M) * static_cast<size_t>(K) ==
      x.size();

  for (auto arg : argnums) {
    if (arg == 0) {
      auto g = gather_qmm(
          cotan,
          w,
          scales,
          kquant_type_,
          std::nullopt,
          rhs_indices,
          !transpose_,
          sorted,
          stream());
      if (sorted && no_broadcast) {
        vjps.push_back(g);
      } else {
        vjps.push_back(mx::reshape(
            mx::scatter_add(
                mx::flatten(mx::zeros_like(x, stream()), 0, -3, stream()),
                lhs_indices,
                mx::expand_dims(g, -3, stream()),
                0,
                stream()),
            x.shape(),
            stream()));
      }
    } else if (arg == 1) {
      throw std::invalid_argument(
          "[mlx_kquant] gather_qmm vjp: no gradient wrt the quantized weights "
          "(the kquant base is frozen).");
    } else if (arg == 2) {
      throw std::invalid_argument(
          "[mlx_kquant] gather_qmm vjp: no gradient wrt scales.");
    } else {
      throw std::invalid_argument(
          "[mlx_kquant] gather_qmm vjp: cannot compute the gradient wrt the "
          "indices.");
    }
  }
  return vjps;
}

#ifdef _METAL_

void KQuantGatherQMM::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // inputs are row-contiguous (ensured by the op):
  //   x (float), w (uint8), scales, lhs_indices (uint32), rhs_indices (uint32).
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];
  const auto& lhs_indices = inputs[3];
  const auto& rhs_indices = inputs[4];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;

  // Sorted-rhs fast path. When the expert
  // (rhs) indices are sorted and lhs was defaulted (right_sorted_), and the
  // batch is large enough to amortize a per-expert GEMM, route to the NAX
  // gather_qmm_rhs kernel instead of B separate gather_qmv vector-matmuls.
  // kquant has no non-NAX rhs kernel, so the NAX gate must hold. We
  // additionally require x and the rhs indices to already be row-contiguous
  // with one x row per output row (x.size()/K == B): the op keeps broadcast
  // index strides for the strided leaves, whereas this kernel takes no strides.
  // The SwitchGLU sort path satisfies all of this; any case that does not falls
  // through to the (correct, slower) gather_qmv / gather_qmm leaves below.
  bool kquant_rhs_ok = kq_is_nax_available() && transpose_ && (K % 64 == 0) &&
      (x.dtype() != mx::float32) && codec_has_nax(kquant_type_);
  if (M == 1 && B >= 16 && right_sorted_ && (B / E >= 4) && kquant_rhs_ok &&
      x.flags().row_contiguous && rhs_indices.flags().row_contiguous &&
      (x.size() / K == static_cast<size_t>(B))) {
    gather_qmm_rhs_nax(
        x,
        w,
        scales,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        /*M=*/static_cast<int>(x.size() / K),
        N,
        K,
        d,
        s,
        kquant_type_);
    return;
  }

  if (M >= vector_limit) {
    gather_qmm(
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        kquant_type_);
    return;
  }

  if (transpose_) {
    gather_qmv(
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        kquant_type_);
    return;
  }

  // KQuant has no dedicated gather_qvm kernel; route through gather_qmm_n.
  gather_qmm(
      x,
      w,
      scales,
      lhs_indices,
      rhs_indices,
      out,
      transpose_,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      kquant_type_);
}

#else

void KQuantGatherQMM::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] gather_qmm has no GPU implementation.");
}

#endif // _METAL_

} // namespace mlx_kquant
