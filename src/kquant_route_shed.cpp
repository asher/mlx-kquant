// Routed-expert slot remap + residency shed for streamed MoE decode (the
// gpu-dispatch autonomous-token front end). One tiny dispatch per MoE layer
// replaces the per-layer host round-trip: the kernel remaps routed expert ids
// to arena slots via a GPU-resident slot table, sheds every non-resident
// expert (a lazy graph cannot demand-read disk), renormalizes the kept gate
// weights mass-preserving, and reports the misses so the host can prestage
// between tokens. The CPU eval mirrors the kernel's f32 arithmetic and
// ordering exactly (parity harness in tests/test_route_shed.py).
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h"

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

constexpr int KQ_ROUTE_SHED_MAX_R = 64;

} // namespace

#ifdef _METAL_

void KQuantRouteShed::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }

  const auto& indices = inputs[0];
  const auto& scores = inputs[1];
  const auto& table = inputs[2];

  int R = indices.shape(-1);
  int T = int(indices.size() / R);
  int E = table.shape(0);

  auto kernel = kq_get_kernel(d, "kq_route_shed");
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(indices, 0);
  ce.set_input_array(scores, 1);
  ce.set_input_array(table, 2);
  ce.set_output_array(outputs[0], 3);
  ce.set_output_array(outputs[1], 4);
  ce.set_output_array(outputs[2], 5);
  ce.set_output_array(outputs[3], 6);
  ce.set_bytes(R, 7);
  ce.set_bytes(E, 8);
  ce.set_bytes(T, 9);
  // One thread per token row: R <= 64 entries of serial work, launch cost
  // dominates. Grid = T threads in groups of 32.
  int tg = T < 32 ? T : 32;
  MTL::Size group_dims(tg, 1, 1);
  MTL::Size grid_dims((T + tg - 1) / tg, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantRouteShed::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.route_shed] GPU eval requires a Metal build.");
}

#endif // _METAL_

void KQuantRouteShed::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }

  auto& encoder = mx::cpu::get_command_encoder(stream());
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  for (auto& out : outputs) {
    encoder.set_output_array(out);
  }
  encoder.dispatch(
      [indices = mx::array::unsafe_weak_copy(inputs[0]),
       scores = mx::array::unsafe_weak_copy(inputs[1]),
       table = mx::array::unsafe_weak_copy(inputs[2]),
       slots = mx::array::unsafe_weak_copy(outputs[0]),
       mix = mx::array::unsafe_weak_copy(outputs[1]),
       miss_ids = mx::array::unsafe_weak_copy(outputs[2]),
       miss_scores = mx::array::unsafe_weak_copy(outputs[3])]() mutable {
        const int R = indices.shape(-1);
        const int64_t T = indices.size() / R;
        const int E = table.shape(0);
        const uint32_t* idp = indices.data<uint32_t>();
        const float* scp = scores.data<float>();
        const int32_t* tbp = table.data<int32_t>();
        uint32_t* slp = slots.data<uint32_t>();
        float* mxp = mix.data<float>();
        int32_t* mip = miss_ids.data<int32_t>();
        float* msp = miss_scores.data<float>();

        for (int64_t t = 0; t < T; t++) {
          const uint32_t* id = idp + t * R;
          const float* sc = scp + t * R;
          uint32_t* sl = slp + t * R;
          float* mo = mxp + t * R;
          int32_t* mi = mip + t * R;
          float* ms = msp + t * R;

          // Pass 1: residency, mass sums, first kept slot. Ascending-r f32
          // accumulation order is part of the op contract (GPU parity).
          int32_t slot_of[KQ_ROUTE_SHED_MAX_R];
          float s_all = 0.0f;
          float s_kept = 0.0f;
          int32_t first_kept = 0;
          bool have_kept = false;
          for (int r = 0; r < R; r++) {
            const int32_t e = int32_t(id[r]);
            const int32_t slot = (e >= 0 && e < E) ? tbp[e] : -1;
            slot_of[r] = slot;
            const float sv = sc[r];
            s_all += sv;
            if (slot >= 0) {
              s_kept += sv;
              if (!have_kept) {
                first_kept = slot;
                have_kept = true;
              }
            }
          }
          const float renorm = s_kept > 0.0f ? s_all / s_kept : 0.0f;

          // Pass 2: slots + mix; collect misses.
          int n_miss = 0;
          for (int r = 0; r < R; r++) {
            if (slot_of[r] >= 0) {
              sl[r] = uint32_t(slot_of[r]);
              mo[r] = sc[r] * renorm;
            } else {
              sl[r] = uint32_t(first_kept);
              mo[r] = 0.0f;
              mi[n_miss] = int32_t(id[r]);
              ms[n_miss] = sc[r];
              n_miss++;
            }
          }
          // Misses front-packed in descending score order (prestage priority);
          // stable insertion sort, matching the kernel.
          for (int i = 1; i < n_miss; i++) {
            const int32_t vi = mi[i];
            const float vs = ms[i];
            int j = i - 1;
            while (j >= 0 && ms[j] < vs) {
              mi[j + 1] = mi[j];
              ms[j + 1] = ms[j];
              j--;
            }
            mi[j + 1] = vi;
            ms[j + 1] = vs;
          }
          for (int r = n_miss; r < R; r++) {
            mi[r] = -1;
            ms[r] = 0.0f;
          }
        }
      });
}

std::vector<mx::Shape> KQuantRouteShed::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& shape = inputs[0].shape();
  return {shape, shape, shape, shape};
}

std::vector<mx::array> route_shed(
    mx::array indices,
    mx::array scores,
    mx::array slot_table,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.route_shed]";
  if (indices.ndim() < 1 || indices.shape(-1) < 1) {
    throw std::invalid_argument(
        std::string(op) + " indices must have a last axis.");
  }
  if (indices.shape(-1) > KQ_ROUTE_SHED_MAX_R) {
    throw std::invalid_argument(
        std::string(op) + " routed width R must be <= " +
        std::to_string(KQ_ROUTE_SHED_MAX_R) + ".");
  }
  if (indices.dtype() != mx::uint32) {
    throw std::invalid_argument(std::string(op) + " indices must be uint32.");
  }
  if (scores.shape() != indices.shape()) {
    throw std::invalid_argument(
        std::string(op) + " scores must match indices in shape.");
  }
  if (scores.dtype() != mx::float32) {
    throw std::invalid_argument(std::string(op) + " scores must be float32.");
  }
  if (slot_table.ndim() != 1 || slot_table.shape(0) < 1) {
    throw std::invalid_argument(
        std::string(op) + " slot_table must be 1-D [n_experts].");
  }
  if (slot_table.dtype() != mx::int32) {
    throw std::invalid_argument(std::string(op) + " slot_table must be int32.");
  }

  auto contig = [&](const mx::array& a) {
    return a.flags().row_contiguous ? a : mx::contiguous(a, false, s);
  };
  const auto& shape = indices.shape();
  return mx::array::make_arrays(
      {shape, shape, shape, shape},
      {mx::uint32, mx::float32, mx::int32, mx::float32},
      std::make_shared<KQuantRouteShed>(s),
      {contig(indices), contig(scores), contig(slot_table)});
}

} // namespace mlx_kquant
