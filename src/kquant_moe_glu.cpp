// Fused MoE gather ops. Two families sharing the decode shape (one activation
// row per gathered expert row), all inference-only (no CPU eval):
//   mxfp4 packed layout: moe_glu_gather (gate + up + biases + clamped-SwiGLU
//     in one dispatch) and gather_qmv_bias.
//   K-quant wire bytes: moe_glu_gather_kq (gate + up + GLU epilogue;
//     optional biases with act "swiglu_clamp", mxfp4/nvfp4 only; act
//     "swiglu_oai" rides the silu kernels under out-scale bit 4) and
//     gather_qmv_kq, per-codec kernels (full codec matrix).
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_codec.h" // codec_by_name
#include "kquant_internal.h" // kq_type_string

#include "kquant_lora_epilogue.h"

#include "../metal/mlx/backend/metal/kernels/kq_glu_pairs.h"

#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// Tuned q6_k/q8_0 kernels stride K in 256-wide steps; the generic Ext-trait
// kernels only need whole blocks. q6_k is structurally K % 256 == 0; q8_0 at
// K % 256 != 0 dispatches the generic "q8_0_ext" instantiations.
inline std::string kq_gather_stem(const std::string& t, int K) {
  if (t == "q8_0" && K % 256 != 0) {
    return "q8_0_ext";
  }
  return t;
}

// Decode-scale launches underfill the GPU at the default mapping (8 K-lanes
// x 8 output rows per threadgroup); widening the K-lane split (NX = 16/32
// lanes per output row) multiplies threadgroups and shortens each K-chain.
// Measured at t = 1 (gemma-a4b q4_k K=2816, qwen3-next q6_k K=2048): only
// the two-stream GLU gathers (gate+up / shexp) profit -- sharing each
// activation chunk across two weight streams keeps per-thread work high
// enough that the extra threadgroups come free (nx16 +9% / +30% incl. the
// tuned-q6_k swap; nx32 mixed, so the auto pick caps at 16). Single-stream
// gathers (qmv / mix) are flat-to-negative under widening at every
// measured shape and stay NX = 8; mix_ns widens for the codecs listed in
// kq_moe_mix_ns_wide. KQ_MOE_NX=8|16|32 forces a width for ALL ops (A/B
// and tests); rows = output rows across the whole dispatch.
// The threadgroup cap covers MTP-verify widths (t = 2..8): at qwen4exp
// geometry (E=512 top-10, K=2560 N=640, iq3_xxs/iq4_xs + q8_0 shexp) nx16
// beats nx8 by 3.6-9.8% at every t in 2..8, still positive at the 7040-tg
// t=8 point, so the cap sits at 8192 tgs; true prefill grids stay NX = 8.
// The width KQ_MOE_NX forces, or 0. Re-read per call only when the variable
// exists at all (interleaved A/B flips it in-process); unset costs one
// static check.
inline int kq_moe_nx_env() {
  static const bool has_env = kq_getenv("KQ_MOE_NX") != nullptr;
  if (has_env) {
    const char* e = kq_getenv("KQ_MOE_NX");
    const int v = e == nullptr ? 0 : std::atoi(e);
    if (v == 8 || v == 16 || v == 32) {
      return v;
    }
  }
  return 0;
}

// The KQ_MOE_NX setting that forces a width, or nullptr. A kernel family
// without the forced width (the q5_k shared-expert kernels stop at nx 16)
// then raises naming the setting, not a bare load error (kq_get_kernel).
inline const char* kq_moe_nx_setting() {
  switch (kq_moe_nx_env()) {
    case 8:
      return "KQ_MOE_NX=8";
    case 16:
      return "KQ_MOE_NX=16";
    case 32:
      return "KQ_MOE_NX=32";
    default:
      return nullptr;
  }
}

inline int kq_moe_pick_nx(int64_t rows, int K, bool wide_ok) {
  if (const int v = kq_moe_nx_env()) {
    return v;
  }
  if (wide_ok && K / 16 >= 32 && (rows * 8) / 64 < 8192) {
    return 16;
  }
  return 8;
}

// Score-mixed down gather (mix_ns) at decode widths: the slot-parallel
// launch spreads S slots over simdgroup pairs, so every threadgroup covers
// only 8 output rows and per-row setup dominates the short K chains. The
// nx16 launch (16 K-lanes per row, 2 simdgroups) beats it for the cheap-
// decode 4-5 bpw codecs and iq3_xxs (M5 Max, E=128..288 top-8, K=2048 and
// 4096, N=4096, chained-dispatch timing: q4_k +2-5%, q4_0 +2-9%, q5_0
// +2-9%, iq3_xxs +1.5-3%; same-order independent-dispatch timing +9-25%).
// q2_k / q3_k (scale unpack heavy) lose 8-14%, q6_k / q8_0 are flat, the
// iq2 grid codecs and q5_k lose in the chained regime, and every codec is
// flat-to-negative at t >= 4 or K < 2048, so those keep the slot-parallel
// form. Where c32 is on, iq3_xxs at t = 1 takes the c32 slot-parallel
// kernel instead (M5 Max: 7-9% less time than this pick). q4_1 tracks q4_0
// on an M3 Max (E=256 top-8, K=2048 and 4096, N=4096, chained, four runs):
// 8-34% less time at t = 2 for both, and flat to mixed at t = 1 for both.
// KQ_MOE_NX still forces a width.
inline bool kq_moe_mix_ns_wide(const std::string& codec, int T, int K) {
  if (T > 2 || K < 2048) {
    return false;
  }
  return codec == "q4_k" || codec == "q4_0" || codec == "q4_1" ||
      codec == "q5_0" || codec == "iq3_xxs";
}

// Dedupe gathers for the MTP verify widths (KQ_MOE_DEDUP, default on; 0
// disables, read live once set): rows of a verify block route to
// overlapping expert sets, and the "_dd" kernels pair the (row, slot)
// pairs that share an expert so each expert is dequantized once per pair
// of rows. The GLU gathers keep the per-pair grid and write both rows
// from the owner; the down projection runs the loop kernel's launch over
// row pairs. Outputs are bit-identical to the per-row kernels. The
// half-dot kernels have no dedupe form; an explicit KQ_MOE_HALF=1 keeps
// its kernels at every width and dedupe stands down.
inline bool kq_moe_dedup(int T, int max_t) {
  if (T < 2 || T > max_t) {
    return false;
  }
  static const bool has_env = kq_getenv("KQ_MOE_DEDUP") != nullptr;
  return !has_env || kq_env_flag("KQ_MOE_DEDUP", 1) != 0;
}
constexpr int KQ_DD_MAX_T = 8;

// Grouped gathers (KQ_MOE_GROUP): at 2 to 16 rows the "_grp" kernels read an
// expert once per eight (row, slot) pairs that route to it, where the pair
// kernels read it once per two. Each kernel finds its pairs' runs from the
// indices, so no table goes ahead of it. Every codec and shared-expert codec of
// the fused gathers has them (kq_moe_group_form). KQ_MOE_GROUP_MIN_T and
// KQ_MOE_GROUP_MIX_MIN_T set the fewest rows of the grouped GLU and of the
// grouped score-mixed down (read per call). On M5 Max the grouped forms
// lose at 3 rows, tie at 4 and win from 5 (Qwen3.8-Flash-Next verify
// forward). They are on by default on GPUs with the tensor-op units, the
// GPUs they were timed on; KQ_MOE_GROUP=1 turns them on elsewhere and 0 off
// (read live once set). The kernels themselves run on any Metal GPU.
constexpr int KQ_GRP_MAX_T = 16;
constexpr int KQ_GRP_GLU_MIN_T = 5;
constexpr int KQ_GRP_MIX_MIN_T = 5;
constexpr int KQ_GRP_DENSE_OFF_MIN_T = 7;
// Output tiles a grouped score-mixed down threadgroup takes, so that one
// search for its run serves them (KQ_MOE_GROUP_MIX_NT); a row's bits do not
// depend on it.
constexpr int KQ_GRP_MIX_NT = 2;
inline bool kq_moe_group(int T) {
  if (T < 2 || T > KQ_GRP_MAX_T) {
    return false;
  }
  static const bool has_env = kq_getenv("KQ_MOE_GROUP") != nullptr;
  if (has_env) {
    return kq_env_flag("KQ_MOE_GROUP", 1) != 0;
  }
#ifdef _METAL_
  return kq_is_nax_available();
#else
  return false;
#endif
}

inline bool kq_moe_group_glu(int T) {
  return kq_moe_group(T) &&
      T >= kq_env_lever_from("KQ_MOE_GROUP_MIN_T", KQ_GRP_GLU_MIN_T, 2);
}

// A forced KQ_MOE_MIX_FORM names one of the ungrouped forms, so the grouped
// down stands aside for it.
inline bool kq_moe_group_mix(int T) {
  static const bool has_form = kq_getenv("KQ_MOE_MIX_FORM") != nullptr;
  if (has_form) {
    const char* e = kq_getenv("KQ_MOE_MIX_FORM");
    const std::string v = e == nullptr ? "" : e;
    if (v == "loop" || v == "sp" || v == "dd") {
      return false;
    }
  }
  return kq_moe_group(T) &&
      T >= kq_env_lever_from("KQ_MOE_GROUP_MIX_MIN_T", KQ_GRP_MIX_MIN_T, 2);
}

// K-lanes per row of the grouped kernels, which have 8 and 16.
inline int kq_moe_group_nx(int64_t rows, int K, bool wide_ok) {
  const int nx = kq_moe_pick_nx(rows, K, wide_ok);
  return nx > 16 ? 16 : nx;
}

inline const char* kq_nx_suffix(int nx) {
  return nx == 32 ? "_nx32" : (nx == 16 ? "_nx16" : "");
}

// Simdgroups per threadgroup for the generic Ext gathers (the kernels read
// the launch shape; the tuned q6_k/q8_0 kernels are fixed at 2). Every
// threadgroup stages its codec's decode tables into threadgroup memory and
// barriers before the dot loop; with 2 simdgroups that cost lands every 8
// rows. Codecs with a large staged table (iq2_xs: 4 KB grid + signs) win
// from amortizing it over 8 simdgroups: measured interleaved at E=288
// N=2048 K=4096 gate/up, +10% at t=1 and +47% at t=4 (both metrics min of 4
// rounds). Codecs with <= 2 KB tables or none (iq2_xxs, iq3_s, iq3_xxs,
// q2_k, q4_k, iq1_m) measure flat to -5% at 8, so they stay at 2. Falls
// back to the largest power of two that divides N's row tiling.
// KQ_MOE_SG=2|4|8 forces a value for every Ext dispatch (A/B and tests);
// read live once set.
// 32 * sg threads must fit the pipeline, not just the device. iq2_xs asks
// for 8 simdgroups where every other codec asks 2, and that kernel is the
// most register-hungry of the family, so a GPU can support fewer threads
// for it than the launch requests. An oversized dispatch is undefined, and
// a paravirtual device answers it by hanging the command buffer rather
// than returning garbage. Halve down the same ladder KQ_MOE_SG walks;
// N stays divisible because the picker already made it so at the larger
// value. KQ_MOE_NX_LOG=1 prints the cap beside the kernel name.
template <typename K>
inline int kq_moe_fit_sg(K kernel, int sg) {
  const int cap = int(kernel->maxTotalThreadsPerThreadgroup());
  while (sg > 1 && 32 * sg > cap) {
    sg /= 2;
  }
  return sg;
}

inline int kq_moe_pick_sg(const std::string& codec, int N, int nx) {
  const int rps = 32 / nx;
  int sg = codec == "iq2_xs" ? 8 : 2;
  static const bool has_env = kq_getenv("KQ_MOE_SG") != nullptr;
  if (has_env) {
    const char* e = kq_getenv("KQ_MOE_SG");
    const int v = e == nullptr ? 0 : std::atoi(e);
    if (v == 2 || v == 4 || v == 8) {
      sg = v;
    }
  }
  while (sg > 2 && N % (sg * rps) != 0) {
    sg /= 2;
  }
  return sg;
}

// Slot-parallel mix_ns (_sp): the S slot dots spread across S simdgroup
// pairs, multiplying resident threads by S without shortening per-thread
// K-chains (the widening lever that measured flat-to-negative on the
// single-stream gathers). Outputs are bit-identical to the loop kernel.
// Solo dispatch -5% at the V4-Flash down shape (the loop kernel's launch
// ramp is occupancy-shy); E2E -0.1% pipelined / -0.5% naive, direction
// consistent across 8 runs x 2 loop regimes x both ABA orders. Default: on
// when the coarse grid underfills the device (decode-scale launches);
// prefill-scale grids keep the loop kernel. KQ_MOE_SP=1/0 forces on/off;
// read live once set so in-process A/Bs can flip arms.
inline bool kq_moe_sp(int64_t coarse_tgs, int S) {
  if (S < 2 || S > 16) {
    return false; // threadgroup is 64 * S threads; 1024 cap => S <= 16
  }
  static const bool has_env = kq_getenv("KQ_MOE_SP") != nullptr;
  const int v = has_env ? kq_env_flag("KQ_MOE_SP", -1) : -1;
  return v == -1 ? coarse_tgs < 2048 : v != 0;
}

// Threads a slot-parallel launch may use: the pipeline's limit, or a lower
// KQ_MOE_SP_THREAD_CAP (test lever, read per call). A slot-parallel launch
// past it (64 threads per simdgroup pair) takes the loop kernel.
template <typename K>
inline int kq_moe_sp_cap(K kernel) {
  const int cap = int(kernel->maxTotalThreadsPerThreadgroup());
  const int v = kq_env_lever_from("KQ_MOE_SP_THREAD_CAP", cap, 32);
  return v < cap ? v : cap;
}

// Pieces of each shared row in the score-mixed down with a shared expert
// (kq_ext_gather_qmv_mix_sp gives each piece a simdgroup pair). The loop
// and pair kernels sum the same pieces in the same order, so the forms
// stay bit-identical to each other. Two pieces at one row, where the shared
// row is the longest slot of a grid with few threadgroups; one at two rows
// and more, where the grid fills the GPU and pieces only add work. One on
// the half-dot kernels and where the shared codec is the routed one (rows
// of equal work, and the tuned q6_k and q8_0 loop kernels take no pieces).
// At most the shared row's block count, and the count that keeps the
// slot-parallel launch within 16 simdgroup pairs. KQ_MOE_SX_SPLIT=<n>
// forces n pieces at any row count, within the same bounds (read per
// call).
inline int kq_moe_sx_split(
    const std::string& t,
    const std::string& st,
    int K,
    int n_route,
    int NS,
    bool use_half,
    int T) {
  const KQuantCodec* sc = codec_by_name(st);
  if (use_half || NS < 1 || sc == nullptr || st == t) {
    return 1;
  }
  const int cap = std::min(K / sc->weights_per_block, (16 - n_route) / NS);
  if (cap <= 1) {
    return 1;
  }
  const int forced = kq_env_lever_from("KQ_MOE_SX_SPLIT", 0, 1);
  const int sp = forced > 0 ? forced : (T == 1 ? 2 : 1);
  return std::max(1, std::min(sp, cap));
}

// A shared expert k times as wide as a routed expert runs as k shared
// slots. KQ_MOE_SHEXP_MULTI=0 declines k > 1 (read per call).
inline bool kq_moe_shexp_multi() {
  return kq_env_flag("KQ_MOE_SHEXP_MULTI", 1) != 0;
}

// Form of a score-mixed down gather (mix and mix_ns) where more than one
// applies: the loop kernel, the slot-parallel kernel or the dedupe row-pair
// kernel. The pair kernel runs the loop launch over half the rows, so on a
// grid that underfills the device it loses to the slot-parallel kernel
// until the two rows of a pair share about two thirds of their experts.
// The slot-parallel kernel goes first where it applies, the pair kernel
// serves the wider grids. Fit: benchmarks/bench_moe_mix_forms.py.
// KQ_MOE_MIX_FORM=loop|sp|dd forces a form where it applies (probe lever,
// read live once set).
enum { KQ_MIX_LOOP = 0, KQ_MIX_SP = 1, KQ_MIX_DD = 2 };

inline int kq_moe_mix_form(bool sp_ok, bool dd_ok) {
  static const bool has_env = kq_getenv("KQ_MOE_MIX_FORM") != nullptr;
  if (has_env) {
    const char* e = kq_getenv("KQ_MOE_MIX_FORM");
    const std::string v = e == nullptr ? "" : e;
    if (v == "loop") {
      return KQ_MIX_LOOP;
    }
    if (v == "sp") {
      return sp_ok ? KQ_MIX_SP : KQ_MIX_LOOP;
    }
    if (v == "dd") {
      return dd_ok ? KQ_MIX_DD : KQ_MIX_LOOP;
    }
  }
  return sp_ok ? KQ_MIX_SP : dd_ok ? KQ_MIX_DD : KQ_MIX_LOOP;
}

// KQ_MOE_NX_LOG=1: print each fused-MoE kernel name once (dispatch audit).
inline void kq_moe_log_kname(const std::string& kname, int cap = 0) {
  static const bool log = kq_getenv("KQ_MOE_NX_LOG") != nullptr;
  if (!log) {
    return;
  }
  static std::set<std::string> seen;
  if (seen.insert(kname).second) {
    if (cap > 0) {
      std::fprintf(
          stderr, "[kq_moe] %s (pipeline max %d)\n", kname.c_str(), cap);
    } else {
      std::fprintf(stderr, "[kq_moe] %s\n", kname.c_str());
    }
  }
}

// Output scales of the fused MoE ops (kq_out_scale.h): bits 0 to 2 are the
// mode of the expert stacks' scales, bit 3 says the shared expert has one.
// The scales are the op's last inputs, the routed ones first, gate before
// up in a GLU. Bit 4 is act swiglu_oai on a GLU, which takes no input: its
// alpha is bound as bytes. Bit 5 says the op computed its shared expert on
// the dense route (kq_moe_shexp_dense): the products sit just ahead of the
// scales, a GLU's gate and up and a mix's down.
constexpr int KQ_MOE_SHEXP_SCALE = 8;
constexpr int KQ_MOE_SWIGLU_OAI = 16;
constexpr int KQ_MOE_SHEXP_PRE = 32;
// Bit 6 is act situ on a routed GLU: it rides the silu kernels like
// swiglu_oai, with beta as the limit and linear_beta bound as alpha.
constexpr int KQ_MOE_SITU = 64;
constexpr int KQ_MOE_GLU_ALPHA = KQ_MOE_SWIGLU_OAI | KQ_MOE_SITU;

inline int kq_moe_scale_inputs(int bits, bool glu) {
  const int per = glu ? 2 : 1;
  return ((bits & 7) ? per : 0) + ((bits & KQ_MOE_SHEXP_SCALE) ? per : 0);
}

// Index of the first dense-route product (KQ_MOE_SHEXP_PRE).
inline size_t
kq_moe_pre_input(const std::vector<mx::array>& inputs, int bits, bool glu) {
  return inputs.size() - kq_moe_scale_inputs(bits, glu) - (glu ? 2 : 1);
}

#ifdef _METAL_
inline void kq_moe_bind_scales(
    mx::metal::CommandEncoder& ce,
    const std::vector<mx::array>& inputs,
    int bits,
    bool glu) {
  size_t i = inputs.size() - kq_moe_scale_inputs(bits, glu);
  if (bits & 7) {
    ce.set_input_array(inputs[i++], kq_out_scale_slot);
    if (glu) {
      ce.set_input_array(inputs[i++], kq_up_scale_slot);
    }
  }
  if (bits & KQ_MOE_SHEXP_SCALE) {
    ce.set_input_array(inputs[i++], kq_shexp_scale_slot);
    if (glu) {
      ce.set_input_array(inputs[i++], kq_shexp_up_scale_slot);
    }
  }
  // A GLU's kernels skip the shared slots; a mix's read the down product.
  if ((bits & KQ_MOE_SHEXP_PRE) && !glu) {
    ce.set_input_array(
        inputs[kq_moe_pre_input(inputs, bits, false)], kq_shexp_pre_slot);
  }
}

// Row-pair dedupe down projection (mix_ns and the shexp-slot mix): the
// loop kernel's launch over row pairs; S is capped by the partial store.
// NS is the shared slot count (unused by mix_ns).
inline void kq_gather_qmv_pair_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const std::string& kname,
    const mx::array& w,
    const mx::array& sw,
    const mx::array& x,
    const mx::array& indices,
    const mx::array& scores,
    mx::array& out,
    int T,
    int S,
    int N,
    int K,
    int nx,
    int sg,
    int NS,
    int SP,
    const std::vector<mx::array>& inputs,
    int os_bits) {
  auto kernel = kq_get_kernel_os(d, kname, os_bits, kq_moe_nx_setting());
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(sw, 1);
  ce.set_input_array(x, 2);
  ce.set_input_array(indices, 3);
  ce.set_input_array(scores, 4);
  ce.set_output_array(out, 5);
  ce.set_bytes(K, 6);
  ce.set_bytes(N, 7);
  ce.set_bytes(S, 8);
  const int SC = scores.shape(1);
  ce.set_bytes(SC, 9);
  ce.set_bytes(T, 10);
  ce.set_bytes(NS, 11);
  ce.set_bytes(SP, 12);
  kq_moe_bind_scales(ce, inputs, os_bits, false);
  const int fsg = kq_moe_fit_sg(kernel, sg);
  MTL::Size group_dims(32, fsg, 1);
  MTL::Size grid_dims(N / (fsg * 32 / nx), 1, (T + 1) / 2);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}
#endif // _METAL_

// Wide-NX variants exist only on the generic Ext kernels; tuned q6_k/q8_0
// uniform dispatches reroute to their "_ext" stems when a wide NX is picked.
inline std::string kq_gather_stem_nx(const std::string& t, int K, int nx) {
  std::string stem = kq_gather_stem(t, K);
  if (nx > 8 && (stem == "q6_k" || stem == "q8_0")) {
    stem += "_ext";
  }
  return stem;
}

// Fine tiling for the single-stream gathers: fewer output rows per
// threadgroup at the unchanged 8-lane K split. Instantiated where it
// measured an E2E win -- the tuned q6_k/q8_0 kernels and the packed-mxfp4
// bias kernels, results_per_simdgroup 4 -> 1 (2 rows per threadgroup, 4x
// threadgroups) -- and dropped for the Ext codec matrix, where both the 2x
// (one-simdgroup) and 4x (NX 16 one-simdgroup) variants were E2E-neutral.
// Unlike wide NX this multiplies work-in-flight without shortening per-lane
// K-chains, and output stays bit-exact. Default: on when the coarse grid
// underfills the device (decode-scale launches); callers only consult this
// at NX = 8, so a forced KQ_MOE_NX=16/32 wins. KQ_GATHER_FINE=1/0 forces
// fine/coarse; read live once set (KQ_MOE_NX pattern above) so interleaved
// in-process A/Bs can flip arms.
// Half-dot Ext gathers (KQ_MOE_HALF=1, default off): the grid codecs whose
// float decode is ALU-bound at decode widths (iq2_xs, iq2_xxs, iq3_xxs) run
// the "_h" kernels, which stage the grid as half4 pairs, stage the
// activation row as half in dynamic threadgroup memory (K halves), and dot
// each 16-weight chunk in float. Expert slots only; shared experts keep the
// float path. Read live once set (KQ_MOE_NX pattern) so interleaved
// in-process A/Bs can flip arms. K is capped so the staged row plus the
// largest half table (iq2_xs, 8.3 KB) stays inside the 32 KB threadgroup
// budget. An activation past the half range (65504) turns to inf in the
// staged row.
inline bool kq_moe_half(const std::string& codec, int K) {
  static const bool has_env = kq_getenv("KQ_MOE_HALF") != nullptr;
  if (!has_env) {
    return false;
  }
  const char* e = kq_getenv("KQ_MOE_HALF");
  if (e == nullptr || std::atoi(e) != 1) {
    return false;
  }
  if (K > 8192 || K % 16 != 0) {
    return false;
  }
  return codec == "iq2_xs" || codec == "iq2_xxs" || codec == "iq3_xxs";
}

// Simdgroups per threadgroup for the half-dot kernels: the staged
// activation row (K halves) and the half grid are amortized over the
// threadgroup's rows, so every codec runs at 8 (KQ_MOE_SG still overrides).
inline int kq_moe_pick_sg_h(int N, int nx) {
  const int rps = 32 / nx;
  int sg = 8;
  static const bool has_env = kq_getenv("KQ_MOE_SG") != nullptr;
  if (has_env) {
    const char* e = kq_getenv("KQ_MOE_SG");
    const int v = e == nullptr ? 0 : std::atoi(e);
    if (v == 2 || v == 4 || v == 8) {
      sg = v;
    }
  }
  while (sg > 2 && N % (sg * rps) != 0) {
    sg /= 2;
  }
  return sg;
}

// Codecs with a 32-weight decode step (KqExtC32): the grid codecs.
inline bool kq_moe_has_c32(const std::string& codec) {
  return codec == "iq2_s" || codec == "iq2_xxs" || codec == "iq2_xs" ||
      codec == "iq3_xxs" || codec == "iq3_s" || codec == "iq1_s" ||
      codec == "iq1_m";
}

// 32-weight decode step ("_c32" kernels, KqExtC32) for the grid-codec
// gathers: by default at t = 1, where this GPU's class takes it for the
// codec (moe_c32_on). KQ_MOE_C32=0 forces the 16-weight kernels and =1 the
// c32 kernels at every t. Read per call. The c32 kernels have NX 8 and 16
// forms, so a forced KQ_MOE_NX=32 keeps the 16-weight kernels. The dedupe
// and half-dot forms have no c32 kernels and keep the 16-weight step.
inline bool kq_moe_c32(const std::string& codec, int T) {
  if (!kq_moe_has_c32(codec) || kq_moe_nx_env() == 32) {
    return false;
  }
  const int v = kq_env_flag("KQ_MOE_C32", -1);
  if (v != -1) {
    return v != 0;
  }
#ifdef _METAL_
  static const std::string arch =
      mx::metal::device(mx::Device::gpu).get_architecture();
  return T == 1 && moe_c32_on(arch, codec);
#else
  return false;
#endif
}

// A shared expert in the routed codec or an upcast q5_k, q6_k or q8_0: the
// pairs with c32 shexp and mix kernels.
inline bool kq_moe_c32_shexp_ok(
    const std::string& codec,
    const std::string& shexp) {
  return shexp == codec || shexp == "q5_k" || shexp == "q6_k" ||
      shexp == "q8_0";
}

// The kernel stem of a routed GLU with its gate stack in t and its up stack
// in u: t itself, or "<t>_up_<u>" for a split pair (kq_glu_pairs.h). The
// form picks that key on the stem see no codec of their own in a pair's,
// so a pair takes the Ext forms (it has no qf or half-dot kernels); its c32
// step and simdgroup count come from its two codecs (kq_moe_c32_glu,
// kq_moe_pick_sg_glu).
inline std::string kq_glu_stem(const std::string& t, const std::string& u) {
  return u == t ? t : t + "_up_" + u;
}

// The c32 GLU gathers where the gate or the up codec takes them; a codec
// without the c32 step decodes its two 16-weight chunks in that loop.
inline bool kq_moe_c32_glu(const std::string& t, const std::string& u, int T) {
  return kq_moe_c32(t, T) || (u != t && kq_moe_c32(u, T));
}

inline int
kq_moe_pick_sg_glu(const std::string& t, const std::string& u, int N, int nx) {
  const int sg = kq_moe_pick_sg(t, N, nx);
  return u == t ? sg : std::max(sg, kq_moe_pick_sg(u, N, nx));
}

// K-lanes per row of the c32 GLU gathers: a forced KQ_MOE_NX of 8 or 16,
// else this GPU's class width (moe_c32_nx).
inline int kq_moe_c32_nx() {
  const int v = kq_moe_nx_env();
  if (v == 8 || v == 16) {
    return v;
  }
#ifdef _METAL_
  static const int here =
      moe_c32_nx(mx::metal::device(mx::Device::gpu).get_architecture());
  return here;
#else
  return 8;
#endif
}

// K-lanes per row of the c32 down gathers (plain, mix and mix_ns): a forced
// KQ_MOE_NX of 8 or 16, else 8. On an M5 Max the c32 down gathers of the
// grid codecs took 0 to 9% less time at 8 than at 16 (E=32..64, top 8,
// K=1536 and 4096).
inline int kq_moe_c32_down_nx() {
  const int v = kq_moe_nx_env();
  return v == 16 ? 16 : 8;
}

inline bool kq_gather_fine(int64_t coarse_tgs) {
  static const bool has_env = kq_getenv("KQ_GATHER_FINE") != nullptr;
  const int v = has_env ? kq_env_flag("KQ_GATHER_FINE", -1) : -1;
  return v == -1 ? coarse_tgs < 2048 : v != 0;
}

// The qf kernels (kq_moe_qf.h) run the fused MoE gathers on the tuned
// qmv_fast loops, for the codecs whose Ext decode is slower than those
// loops. Rows per simdgroup of the codec's loop, or 0 without qf kernels.
inline int kq_qf_rps(const std::string& t) {
  if (t == "q2_k" || t == "q3_k" || t == "q4_k" || t == "iq4_xs" ||
      t == "q2_0") {
    return 2;
  }
  if (t == "q4_0" || t == "q4_1" || t == "q5_0" || t == "q5_1" ||
      t == "iq4_nl" || t == "mxfp4" || t == "nvfp4") {
    return 4;
  }
  return 0;
}

// A qf codec takes the qf kernels at every T, so a row's routed output
// does not depend on the batch it is in (the shared expert does, from
// KQ_MOE_SHEXP_DENSE tokens). They need whole row tiles and whole blocks
// of K. KQ_MOE_QF=0 keeps the Ext kernels (A/B and tests, read live once
// set), and so does a forced KQ_MOE_NX, which names an Ext width.
inline bool kq_moe_qf_on(const std::string& t) {
  if (kq_qf_rps(t) == 0 || kq_moe_nx_env() != 0) {
    return false;
  }
  static const bool has_env = kq_getenv("KQ_MOE_QF") != nullptr;
  return !has_env || kq_env_flag("KQ_MOE_QF", 1) != 0;
}

inline bool kq_moe_qf(const std::string& t, int N, int K) {
  const KQuantCodec* c = codec_by_name(t);
  return kq_moe_qf_on(t) && c != nullptr && N % (2 * kq_qf_rps(t)) == 0 &&
      K % c->weights_per_block == 0;
}

// qf codecs with a pair GLU (kq_qf_moe_glu_gather_dd) at the dedupe widths.
// The k-quant and q4_0 / q4_1 loops gain nothing from it or lose occupancy.
inline bool kq_qf_pairs(const std::string& t) {
  return t == "q5_0" || t == "q5_1" || t == "iq4_nl" || t == "iq4_xs" ||
      t == "mxfp4" || t == "nvfp4" || t == "q2_0";
}

// Fewest tokens at which a fused MoE op computes its shared expert on the
// dense route (quantized_matmul_multi), which reads the shared weights once
// for the call, where the fused kernels read them once per token. The fused
// kernels then skip the shared slots. On M5 Max the route loses below 4
// tokens and ties or wins from 4 on every codec. KQ_MOE_SHEXP_DENSE=<t>
// sets the width and 0 turns the route off. Read live.
constexpr int KQ_SHEXP_DENSE_MIN_T = 4;
inline bool kq_moe_shexp_dense(int T) {
  const int m = kq_env_lever("KQ_MOE_SHEXP_DENSE", KQ_SHEXP_DENSE_MIN_T);
  return m > 0 && T >= m;
}

// A shared expert the qf kernels take: the routed codec, or a q5_k, q6_k
// or q8_0 upcast at K % 256 == 0 (the q8_0 loop steps 256 weights).
inline bool kq_qf_shexp_ok(const std::string& t, const std::string& st, int K) {
  return st == t ||
      ((st == "q5_k" || st == "q6_k" || st == "q8_0") && K % 256 == 0);
}

// The grouped kernel family of a call on expert codec t with shared codec
// st ("" for a call with no shared expert). qf_shape says the plain call
// could take the qf kernels (a score-mixed down with other than one shared
// slot cannot).
//   KQ_GRP_QF: the plain call takes the qf kernels. The grouped qf kernels
//     run the same loop over four rows a pass, so a row keeps the bits of the
//     plain qf kernels at every call width.
//   KQ_GRP_NONE: the half-dot kernels (KQ_MOE_HALF) have no grouped form.
//   KQ_GRP_EXT: every other call, on the grouped Ext loops.
// KQ_MOE_GROUP_QF=0 puts every qf codec on the grouped Ext loops, whose
// rows differ from a one-row call by rounding (read per call).
constexpr int KQ_GRP_NONE = 0;
constexpr int KQ_GRP_EXT = 1;
constexpr int KQ_GRP_QF = 2;
inline int kq_moe_group_form(
    const std::string& t,
    const std::string& st,
    int N,
    int K,
    bool qf_shape = true) {
  if (kq_moe_half(t, K)) {
    return KQ_GRP_NONE;
  }
  const bool qf = qf_shape && kq_moe_qf(t, N, K) &&
      (st.empty() || kq_qf_shexp_ok(t, st, K));
  if (!qf || kq_env_lever_from("KQ_MOE_GROUP_QF", 1, 0) == 0) {
    return KQ_GRP_EXT;
  }
  return KQ_GRP_QF;
}

// "kq_<stem>_<op>_grp...": the shared codec is in the name also when it is
// the routed one, and a q6_k or q8_0 pair is the "_ext" instantiation.
inline std::string
kq_moe_group_stem(int form, const std::string& t, const std::string& st) {
  const std::string sx = st.empty() ? t : st;
  if (form == KQ_GRP_QF) {
    return t + "_qf_sx_" + sx;
  }
  if (sx == t && (t == "q6_k" || t == "q8_0")) {
    return t + "_ext_sx_" + t + "_ext";
  }
  return t + "_sx_" + sx;
}

// KQ_MOE_SHEXP_DENSE is set: the route follows it, ahead of the tiled
// kernels' own shared tiles.
inline bool kq_moe_shexp_dense_forced() {
  const char* e = kq_getenv("KQ_MOE_SHEXP_DENSE");
  return e != nullptr && e[0] != '\0';
}

// The route for an op on expert codec t with shared codec st. The grouped
// kernels read the shared expert once per eight rows (once per row on the
// qf loops), and from 7 rows the route's three dispatches cost more than
// they save beside them (Qwen3.8-Flash-Next verify forward on M5 Max), so
// the route stands down there unless KQ_MOE_SHEXP_DENSE is set.
inline bool kq_moe_shexp_dense(
    int T,
    const std::string& t,
    const std::string& st,
    int N,
    int K) {
  const char* e = kq_getenv("KQ_MOE_SHEXP_DENSE");
  if ((e == nullptr || e[0] == '\0') && T >= KQ_GRP_DENSE_OFF_MIN_T &&
      kq_moe_group_glu(T) && kq_moe_group_form(t, st, N, K) != KQ_GRP_NONE) {
    return false;
  }
  return kq_moe_shexp_dense(T);
}

// Tiled score-mixed down (kq_moe_tile.h, KQ_MOE_TILE, default on; 0 keeps
// the grouped kernels, read per call): at KQ_MOE_TILE_MIN_T to 16 rows on a
// GPU with the tensor-op units the pairs are sorted by expert into tiles of
// up to eight, and a tile reads its expert once, a lane an output column.
// The shared slots run as tiles too, so the dense route stands down. Every
// codec pair of the grouped Ext kernels has the kernel but nvfp4; a call
// with output scales, the half-dot kernels, a forced KQ_MOE_MIX_FORM and
// KQ_MOE_GROUP=0 keep their kernels. A row's bits depend on the call's
// width. On M5 Max (Qwen3.8-Flash-Next, K 640, N 2560) the tiled down took
// 146 to 149 us a layer against 154 to 165 for the grouped one at 8 rows
// and 261 against 338 at 16; with the floor at 4 the MoE block tied.
constexpr int KQ_TL_MIX_MIN_T = 8;
inline bool kq_moe_tile_ok(
    int T,
    const std::string& t,
    const std::string& st,
    int K,
    int bits) {
#ifdef _METAL_
  if (T > KQ_GRP_MAX_T || bits != 0 || K % 32 != 0) {
    return false;
  }
  if (t == "nvfp4" || st == "nvfp4" || kq_moe_half(t, K)) {
    return false;
  }
  return kq_moe_group(T) && kq_env_flag("KQ_MOE_TILE", 1) != 0 &&
      kq_is_nax_available();
#else
  (void)T;
  (void)t;
  (void)st;
  (void)K;
  (void)bits;
  return false;
#endif
}

inline bool kq_moe_tile_mix(
    int T,
    const std::string& t,
    const std::string& st,
    int N,
    int K,
    int bits) {
  static const bool has_form = kq_getenv("KQ_MOE_MIX_FORM") != nullptr;
  if (has_form) {
    const char* e = kq_getenv("KQ_MOE_MIX_FORM");
    if (e != nullptr && e[0] != '\0') {
      return false;
    }
  }
  return N % 4 == 0 &&
      T >= kq_env_lever_from("KQ_MOE_TILE_MIN_T", KQ_TL_MIX_MIN_T, 2) &&
      kq_moe_tile_ok(T, t, st, K, bits);
}

#ifdef _METAL_
// The tiled score-mixed down: kq_moe_tile_plan sorts the pairs and their
// activation rows into tile order, the tile kernel writes every pair's row
// sums [pairs, N], and kq_moe_tile_combine adds each row's slots under
// their scores. NS == 0 is the call with no shared expert (sw is then w).
inline void kq_gather_qmv_mix_tile_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const std::string& t,
    const std::string& st,
    const mx::array& w,
    const mx::array& sw,
    const mx::array& x,
    const mx::array& indices,
    const mx::array& scores,
    mx::array& out,
    int T,
    int S,
    int N,
    int K,
    int NS) {
  const int pairs = T * S;
  const std::string type = kq_type_string(x.dtype());
  // The staged tile reads eight rows from a tile's first, so the last tile
  // may read up to seven rows past the pairs; their sums are not stored.
  mx::array hg({pairs + 7, K}, x.dtype(), nullptr, {});
  mx::array tiles({pairs, 4}, mx::uint32, nullptr, {});
  mx::array place({T * (S - NS) + 1}, mx::uint32, nullptr, {});
  mx::array y({pairs, N}, x.dtype(), nullptr, {});
  hg.set_data(mx::allocator::malloc(hg.nbytes()));
  tiles.set_data(mx::allocator::malloc(tiles.nbytes()));
  place.set_data(mx::allocator::malloc(place.nbytes()));
  y.set_data(mx::allocator::malloc(y.nbytes()));
  auto& ce = mx::metal::get_command_encoder(s);

  const std::string pname = "kq_moe_tile_plan_" + type;
  auto pk = kq_get_kernel(d, pname);
  kq_moe_log_kname(pname, int(pk->maxTotalThreadsPerThreadgroup()));
  ce.set_compute_pipeline_state(pk);
  ce.set_input_array(indices, 0);
  ce.set_input_array(x, 1);
  ce.set_output_array(hg, 2);
  ce.set_output_array(tiles, 3);
  ce.set_output_array(place, 4);
  ce.set_bytes(K, 5);
  ce.set_bytes(S, 6);
  ce.set_bytes(NS, 7);
  ce.set_bytes(T, 8);
  MTL::Size pgroup(32, 1, 1);
  MTL::Size pgrid(pairs, 1, 1);
  kq_log_dispatch(pname, x, hg, pgrid, pgroup);
  ce.dispatch_threadgroups(pgrid, pgroup);

  const std::string kname = "kq_" + kq_moe_group_stem(KQ_GRP_EXT, t, st) +
      "_gather_qmv_mix_tl_" + type;
  auto kernel = kq_get_kernel(d, kname);
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(sw, 1);
  ce.set_input_array(hg, 2);
  ce.set_input_array(tiles, 3);
  ce.set_output_array(y, 5);
  ce.set_bytes(K, 6);
  ce.set_bytes(N, 7);
  ce.set_bytes(NS, 9);
  const int COLR = kq_env_lever_from("KQ_MOE_TILE_COL_ROWS", 2, 0);
  ce.set_bytes(COLR, 10);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims((N + 63) / 64, pairs, 1);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);

  const std::string cname = "kq_moe_tile_combine_" + type;
  auto ck = kq_get_kernel(d, cname);
  kq_moe_log_kname(cname, int(ck->maxTotalThreadsPerThreadgroup()));
  ce.set_compute_pipeline_state(ck);
  ce.set_input_array(y, 0);
  ce.set_input_array(scores, 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(N, 3);
  ce.set_bytes(S, 4);
  const int SC = scores.shape(1);
  ce.set_bytes(SC, 5);
  ce.set_bytes(NS, 6);
  ce.set_input_array(place, 7);
  MTL::Size cgroup(std::min(N / 4, 256), 1, 1);
  MTL::Size cgrid(N / 4, 1, T);
  kq_log_dispatch(cname, x, out, cgrid, cgroup);
  ce.dispatch_threads(cgrid, cgroup);
  ce.add_temporary(hg);
  ce.add_temporary(tiles);
  ce.add_temporary(place);
  ce.add_temporary(y);
}

// A grouped score-mixed down (form KQ_GRP_EXT or KQ_GRP_QF): the grouped
// kernel writes every (row, slot) pair's row sums to a float32 [T, S, N]
// temporary and kq_moe_mix_grp_sum adds each row's slots under their
// scores. NS == 0 is the call with no shared expert (sw is then w).
inline void kq_gather_qmv_mix_grp_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    int form,
    const std::string& t,
    const std::string& st,
    const mx::array& w,
    const mx::array& sw,
    const mx::array& x,
    const mx::array& indices,
    const mx::array& scores,
    mx::array& out,
    int T,
    int S,
    int N,
    int K,
    int NS,
    const std::vector<mx::array>& inputs,
    int os_bits) {
  const bool qf = form == KQ_GRP_QF;
  const bool pre = (os_bits & KQ_MOE_SHEXP_PRE) != 0;
  const int n_route = S - NS;
  mx::array part({T, S, N}, mx::float32, nullptr, {});
  part.set_data(mx::allocator::malloc(part.nbytes()));
  auto& ce = mx::metal::get_command_encoder(s);
  const int nx = !qf && kq_moe_nx_env() == 16 ? 16 : 8;
  const int nt = kq_env_lever_from("KQ_MOE_GROUP_MIX_NT", KQ_GRP_MIX_NT, 1);
  const std::string kname = "kq_" + kq_moe_group_stem(form, t, st) +
      "_gather_qmv_mix_grp" + kq_nx_suffix(nx) + "_" +
      kq_type_string(x.dtype());
  auto kernel = kq_get_kernel_os(d, kname, os_bits, nullptr);
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(sw, 1);
  ce.set_input_array(x, 2);
  ce.set_input_array(indices, 3);
  ce.set_output_array(part, 5);
  ce.set_bytes(K, 6);
  ce.set_bytes(N, 7);
  ce.set_bytes(S, 8);
  ce.set_bytes(NS, 9);
  ce.set_bytes(nt, 10);
  const int sg = qf ? 2 : kq_moe_fit_sg(kernel, kq_moe_pick_sg(t, N, nx));
  const int tile = qf ? 2 * kq_qf_rps(t) : sg * 32 / nx;
  MTL::Size group_dims(32, sg, 1);
  MTL::Size grid_dims((N / tile + nt - 1) / nt, pre ? n_route : S, T);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);

  // The nvfp4 kernels, which take output scales, have their own sum.
  const bool os = t == "nvfp4";
  const std::string sname = std::string("kq_moe_mix_grp_sum_") +
      (os ? "os_" : "") + kq_type_string(x.dtype());
  auto sk = kq_get_kernel_os(
      d, sname, os ? os_bits : os_bits & KQ_MOE_SHEXP_PRE, nullptr);
  kq_moe_log_kname(sname, int(sk->maxTotalThreadsPerThreadgroup()));
  ce.set_compute_pipeline_state(sk);
  ce.set_input_array(part, 0);
  ce.set_input_array(scores, 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(N, 3);
  ce.set_bytes(S, 4);
  const int SC = scores.shape(1);
  ce.set_bytes(SC, 5);
  ce.set_bytes(NS, 6);
  ce.set_input_array(indices, 7);
  kq_moe_bind_scales(ce, inputs, os_bits, false);
  MTL::Size sgroup(std::min(N, 256), 1, 1);
  MTL::Size sgrid(N, 1, T);
  kq_log_dispatch(sname, x, out, sgrid, sgroup);
  ce.dispatch_threads(sgrid, sgroup);
  ce.add_temporary(part);
}

#endif // _METAL_

// Slot-parallel or loop form of a qf score-mixed down, by the same rule as
// the Ext forms (kq_moe_sp, KQ_MOE_MIX_FORM=loop|sp).
inline bool kq_qf_mix_sp(int T, int N, int rps, int S) {
  return kq_moe_mix_form(kq_moe_sp((int64_t)T * (N / (2 * rps)), S), false) ==
      KQ_MIX_SP;
}

// Layout of the operands the MoE kernels read without strides. An
// unevaluated view reports the default dense strides to an op builder, so
// the builders decide only for an operand that has memory, and each eval
// decides for the rest before it binds a buffer.
//
// The kernels find expert e of a stack at e * N * bytes_per_row and read a
// shared-expert tensor as dense rows. A stack is too large to copy on a
// call, so a stack or a shared-expert tensor that is not row-major is
// refused, with the builder's message.
inline std::string moe_op(const char* name) {
  return std::string("[mlx_kquant.") + name + "]";
}

inline void moe_check_stack(const char* name, const mx::array& w) {
  kq_check_row_major(w, moe_op(name) + " weights");
}

inline void moe_check_shexp(const char* name, const mx::array& w) {
  kq_check_row_major(w, moe_op(name) + " shared-expert weights");
}

// The packed-mxfp4 stacks carry a companion scales tensor.
inline void
moe_check_stack(const char* name, const mx::array& w, const mx::array& sc) {
  if (!w.flags().row_contiguous || !sc.flags().row_contiguous) {
    throw std::invalid_argument(
        moe_op(name) + " weights/scales must be row-contiguous.");
  }
}

} // namespace

#ifdef _METAL_

void KQuantMoEGLU::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const auto& gw = inputs[0];
  const auto& gs = inputs[1];
  const auto& gb = inputs[2];
  const auto& uw = inputs[3];
  const auto& us = inputs[4];
  const auto& ub = inputs[5];
  const auto& indices = inputs[7];
  moe_check_stack("moe_glu_gather", gw, gs);
  moe_check_stack("moe_glu_gather", uw, us);
  // x is read as dense rows from a 16-byte boundary: any other view is
  // copied first (kq_x_aligned_gpu).
  const mx::array x = kq_x_aligned_gpu(inputs[6], s, "moe_glu_gather x");
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = gw.shape(1);
  int K = x.shape(-1);
  float alpha = alpha_;
  float limit = limit_;

  std::string kname = "kq_moe_glu_gather_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(gw, 0);
  ce.set_input_array(gs, 1);
  ce.set_input_array(gb, 2);
  ce.set_input_array(uw, 3);
  ce.set_input_array(us, 4);
  ce.set_input_array(ub, 5);
  ce.set_input_array(x, 6);
  ce.set_input_array(indices, 7);
  ce.set_output_array(out, 8);
  ce.set_bytes(K, 9);
  ce.set_bytes(N, 10);
  ce.set_bytes(alpha, 11);
  ce.set_bytes(limit, 12);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / 8, R, T);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVBias::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const auto& w = inputs[0];
  const auto& sc = inputs[1];
  const auto& b = inputs[2];
  const auto& indices = inputs[4];
  moe_check_stack("gather_qmv_bias", w, sc);
  const mx::array x = kq_x_aligned_gpu(inputs[3], s, "gather_qmv_bias x");
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);

  const bool fine = kq_gather_fine((int64_t)(N / 8) * R * T);
  std::string kname = std::string("kq_gather_qmv_bias") +
      (fine ? "_fine_" : "_") + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(sc, 1);
  ce.set_input_array(b, 2);
  ce.set_input_array(x, 3);
  ce.set_input_array(indices, 4);
  ce.set_output_array(out, 5);
  ce.set_bytes(K, 6);
  ce.set_bytes(N, 7);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / (fine ? 2 : 8), R, T);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVMixBias::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const auto& w = inputs[0];
  const auto& sc = inputs[1];
  const auto& b = inputs[2];
  const auto& indices = inputs[4];
  const auto& scores = inputs[5];
  moe_check_stack("gather_qmv_mix_bias", w, sc);
  const mx::array x = kq_x_aligned_gpu(inputs[3], s, "gather_qmv_mix_bias x");
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int T = indices.shape(0);
  int S = indices.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);

  const bool fine = kq_gather_fine((int64_t)(N / 8) * T);
  std::string kname = std::string("kq_gather_qmv_mix_bias") +
      (fine ? "_fine_" : "_") + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(sc, 1);
  ce.set_input_array(b, 2);
  ce.set_input_array(x, 3);
  ce.set_input_array(indices, 4);
  ce.set_input_array(scores, 5);
  ce.set_output_array(out, 6);
  ce.set_bytes(K, 7);
  ce.set_bytes(N, 8);
  ce.set_bytes(S, 9);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / (fine ? 2 : 8), 1, T);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantMoEGLUKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  // Biased (swiglu_clamp) layout: gw, uw, gb, ub, x, indices. The scales
  // follow.
  const size_t n_base =
      inputs.size() - kq_moe_scale_inputs(out_scale_bits_, true);
  const bool biased = n_base == 6;
  const auto& gw = inputs[0];
  const auto& uw = inputs[1];
  const auto& indices = inputs[n_base - 1];
  moe_check_stack("moe_glu_gather_kq", gw);
  moe_check_stack("moe_glu_gather_kq", uw);
  kq_check_weight_base(gw, kquant_type_, "moe_glu_gather_kq");
  kq_check_weight_base(uw, up_type_, "moe_glu_gather_kq");
  // The kernel stem: the codec, or the split pair's (one codec when equal).
  const std::string kt = kq_glu_stem(kquant_type_, up_type_);
  const mx::array x =
      kq_x_aligned_gpu(inputs[biased ? 4 : 2], s, "moe_glu_gather_kq x");
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = gw.shape(1);
  int K = x.shape(-1);

  // Grouped form: the shared-expert grouped kernel with no shared slots,
  // the expert stacks standing in for its shared operands.
  const int gform = !biased && kq_moe_group_glu(T)
      ? kq_moe_group_form(kt, "", N, K)
      : KQ_GRP_NONE;
  if (gform != KQ_GRP_NONE) {
    const bool gqf = gform == KQ_GRP_QF;
    const int nx = gqf ? 8 : kq_moe_group_nx((int64_t)N * R * T, K, true);
    // swiglu_oai and situ ride the silu kernels, with alpha bound.
    const bool galpha = (out_scale_bits_ & KQ_MOE_GLU_ALPHA) != 0;
    const std::string kname = "kq_" + kq_moe_group_stem(gform, kt, "") +
        "_moe_glu_gather_shexp_grp_" +
        (act_ == "gelu" && !galpha ? "gelu" : "silu") + kq_nx_suffix(nx) + "_" +
        kq_type_string(x.dtype());
    auto kernel = kq_get_kernel_os(d, kname, out_scale_bits_, nullptr);
    kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
    auto& ce = mx::metal::get_command_encoder(s);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(gw, 0);
    ce.set_input_array(uw, 1);
    ce.set_input_array(gw, 2);
    ce.set_input_array(uw, 3);
    ce.set_input_array(x, 4);
    ce.set_input_array(indices, 5);
    ce.set_output_array(out, 6);
    ce.set_bytes(K, 7);
    ce.set_bytes(N, 8);
    ce.set_bytes(limit_, 9);
    const int NS = 0;
    ce.set_bytes(NS, 10);
    if (galpha) {
      ce.set_bytes(alpha_, kq_glu_alpha_slot);
    }
    kq_moe_bind_scales(ce, inputs, out_scale_bits_, true);
    const int sg = gqf
        ? 2
        : kq_moe_fit_sg(
              kernel, kq_moe_pick_sg_glu(kquant_type_, up_type_, N, nx));
    const int tile = gqf ? 2 * kq_qf_rps(kt) : sg * 32 / nx;
    MTL::Size group_dims(32, sg, 1);
    MTL::Size grid_dims(N / tile, R, T);
    kq_log_dispatch(kname, x, out, grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);
    return;
  }

  // The half kernels run at nx 8: 32 rows per threadgroup amortize the
  // staged activation row and half grid (nx 16 measured 10% slower).
  const bool qf = kq_moe_qf(kt, N, K);
  const bool use_half = !qf && !biased && kq_moe_half(kt, K);
  const bool dd = (!qf || kq_qf_pairs(kt)) && !biased && !use_half &&
      kq_moe_dedup(T, KQ_DD_MAX_T);
  const bool c32 = !qf && !biased && !use_half && !dd &&
      kq_moe_c32_glu(kquant_type_, up_type_, T);
  const int nx = qf || use_half ? 8
      : c32                     ? kq_moe_c32_nx()
                                : kq_moe_pick_nx((int64_t)N * R * T, K, true);
  std::string stem = qf ? kt + "_qf" : kq_gather_stem_nx(kt, K, nx);
  if (dd && (stem == "q6_k" || stem == "q8_0")) {
    stem += "_ext"; // the dedupe kernels are Ext-only
  }
  // swiglu_oai and situ ride the silu kernels under their out-scale bits
  // (kq_glu_epilogue), with the run-time limit and alpha.
  const bool rides_silu = (out_scale_bits_ & KQ_MOE_GLU_ALPHA) != 0;
  std::string kname = "kq_" + stem + "_moe_glu_gather_" +
      (biased ? "bias_" : "") + (use_half ? "h_" : "") + (dd ? "dd_" : "") +
      (rides_silu ? std::string("silu") : act_) + (c32 ? "_c32" : "") +
      kq_nx_suffix(nx) + "_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel_os(
      d, kname, out_scale_bits_, qf ? nullptr : kq_moe_nx_setting());
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  if (use_half) {
    ce.set_threadgroup_memory_length((size_t)K * sizeof(uint16_t), 0);
  }
  ce.set_input_array(gw, 0);
  ce.set_input_array(uw, 1);
  if (biased) {
    ce.set_input_array(inputs[2], 2);
    ce.set_input_array(inputs[3], 3);
    ce.set_input_array(x, 4);
    ce.set_input_array(indices, 5);
    ce.set_output_array(out, 6);
    ce.set_bytes(K, 7);
    ce.set_bytes(N, 8);
    ce.set_bytes(limit_, 9);
    ce.set_bytes(alpha_, 10);
  } else {
    ce.set_input_array(x, 2);
    ce.set_input_array(indices, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(K, 5);
    ce.set_bytes(N, 6);
    ce.set_bytes(limit_, 7);
    if (rides_silu) {
      ce.set_bytes(alpha_, kq_glu_alpha_slot);
    }
  }
  kq_moe_bind_scales(ce, inputs, out_scale_bits_, true);
  // Tuned q6_k/q8_0 kernels (nx 8) and the qf kernels are fixed at 2
  // simdgroups.
  const bool tuned = qf || stem == "q6_k" || stem == "q8_0";
  const int sg = kq_moe_fit_sg(
      kernel,
      use_half
          ? kq_moe_pick_sg_h(N, nx)
          : (tuned ? 2 : kq_moe_pick_sg_glu(kquant_type_, up_type_, N, nx)));
  const int tile = qf ? 2 * kq_qf_rps(kt) : sg * 32 / nx;
  MTL::Size group_dims(32, sg, 1);
  MTL::Size grid_dims(N / tile, R, T);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  // Biased layout: w, b, x, indices (then the LoRA operands and the scale,
  // if any).
  const size_t n_base = inputs.size() - kq_lora_input_count(lora_flags_) -
      kq_moe_scale_inputs(out_scale_bits_, false);
  const bool biased = n_base == 4;
  const auto& w = inputs[0];
  const auto& indices = inputs[n_base - 1];
  moe_check_stack("gather_qmv_kq", w);
  kq_check_weight_base(w, kquant_type_, "gather_qmv_kq");
  const mx::array x =
      kq_x_aligned_gpu(inputs[biased ? 2 : 1], s, "gather_qmv_kq x");
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);
  std::optional<KqLoraRows> lora;
  if (lora_flags_ & KQ_LORA_PRESENT) {
    lora =
        kq_lora_rows_pre_gpu(d, s, x, inputs, n_base, lora_flags_, T * R, K, N);
  }

  const bool qf = kq_moe_qf(kquant_type_, N, K);
  const bool use_half = !qf && !biased && kq_moe_half(kquant_type_, K);
  const bool c32 = !qf && !biased && !use_half && kq_moe_c32(kquant_type_, T);
  const int nx = qf || use_half ? 8
      : c32                     ? kq_moe_c32_down_nx()
                                : kq_moe_pick_nx((int64_t)N * R * T, K, false);
  const std::string stem =
      qf ? kquant_type_ + "_qf" : kq_gather_stem_nx(kquant_type_, K, nx);
  // Fine tiling is instantiated only on the tuned q6_k/q8_0 kernels; the Ext
  // equivalents (2x and 4x threadgroup variants) measured E2E-neutral and
  // were dropped. Biased stacks are mxfp4/nvfp4 wire, i.e. never tuned.
  const bool tuned = stem == "q6_k" || stem == "q8_0";
  const bool fine =
      tuned && !biased && nx == 8 && kq_gather_fine((int64_t)(N / 8) * R * T);
  std::string kname = "kq_" + stem + "_gather_qmv" + (biased ? "_bias" : "") +
      (use_half ? "_h" : "") + (c32 ? "_c32" : "") +
      (fine ? "_fine" : kq_nx_suffix(nx)) + "_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel_os(
      d, kname, out_scale_bits_, fine || qf ? nullptr : kq_moe_nx_setting());
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  if (use_half) {
    ce.set_threadgroup_memory_length((size_t)K * sizeof(uint16_t), 0);
  }
  kq_moe_bind_scales(ce, inputs, out_scale_bits_, false);
  ce.set_input_array(w, 0);
  if (biased) {
    ce.set_input_array(inputs[1], 1);
    ce.set_input_array(x, 2);
    ce.set_input_array(indices, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(K, 5);
    ce.set_bytes(N, 6);
  } else {
    ce.set_input_array(x, 1);
    ce.set_input_array(indices, 2);
    ce.set_output_array(out, 3);
    ce.set_bytes(K, 4);
    ce.set_bytes(N, 5);
  }
  const int sg = kq_moe_fit_sg(
      kernel,
      use_half ? kq_moe_pick_sg_h(N, nx)
               : (tuned || qf ? 2 : kq_moe_pick_sg(kquant_type_, N, nx)));
  const int tile = qf ? 2 * kq_qf_rps(kquant_type_) : fine ? 2 : sg * 32 / nx;
  MTL::Size group_dims(32, sg, 1);
  MTL::Size grid_dims(N / tile, R, T);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);
  if (lora) {
    kq_lora_rows_post_gpu(d, s, x, *lora, out, T * R, K, N);
  }
}

void KQuantMoEGLUShexpKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const auto& gw = inputs[0];
  const auto& uw = inputs[1];
  const auto& sgw = inputs[2];
  const auto& suw = inputs[3];
  const auto& indices = inputs[5];
  moe_check_stack("moe_glu_gather_shexp_kq", gw);
  moe_check_stack("moe_glu_gather_shexp_kq", uw);
  moe_check_shexp("moe_glu_gather_shexp_kq", sgw);
  moe_check_shexp("moe_glu_gather_shexp_kq", suw);
  kq_check_weight_base(gw, kquant_type_, "moe_glu_gather_shexp_kq");
  kq_check_weight_base(uw, up_type_, "moe_glu_gather_shexp_kq");
  kq_check_weight_base(sgw, shexp_type_, "moe_glu_gather_shexp_kq");
  kq_check_weight_base(suw, shexp_up_type_, "moe_glu_gather_shexp_kq");
  const mx::array x =
      kq_x_aligned_gpu(inputs[4], s, "moe_glu_gather_shexp_kq x");
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = gw.shape(1);
  int K = x.shape(-1);
  // Shared slots: the shared expert is NS routed experts wide.
  const int NS = sgw.shape(0) / N;

  // The routed kernel stem: the codec, or a split pair's (kq_glu_stem).
  const std::string kt = kq_glu_stem(kquant_type_, up_type_);
  const bool upcast_pair = shexp_up_type_ == shexp_type_ &&
      (shexp_type_ == kt || shexp_type_ == "q5_k" || shexp_type_ == "q6_k" ||
       shexp_type_ == "q8_0");
  if (!upcast_pair) {
    // Gate and up of the shared expert each in its own codec: the "_sx2_"
    // instantiations, generic float path, one launch shape for any T.
    const bool grp2 = kq_moe_group_glu(T) && !kq_moe_half(kquant_type_, K);
    const int nx = grp2 ? kq_moe_group_nx((int64_t)N * (R + NS) * T, K, true)
                        : kq_moe_pick_nx((int64_t)N * (R + NS) * T, K, true);
    const std::string kname = "kq_" + kquant_type_ + "_sx2_" + shexp_type_ +
        "_" + shexp_up_type_ + "_moe_glu_gather_shexp_" +
        (grp2 ? "grp_silu" : "silu") + kq_nx_suffix(nx) + "_" +
        kq_type_string(x.dtype());
    auto kernel = kq_get_kernel_os(
        d, kname, out_scale_bits_, grp2 ? nullptr : kq_moe_nx_setting());
    kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
    auto& ce = mx::metal::get_command_encoder(s);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(gw, 0);
    ce.set_input_array(uw, 1);
    ce.set_input_array(sgw, 2);
    ce.set_input_array(suw, 3);
    ce.set_input_array(x, 4);
    ce.set_input_array(indices, 5);
    ce.set_output_array(out, 6);
    ce.set_bytes(K, 7);
    ce.set_bytes(N, 8);
    ce.set_bytes(limit_, 9);
    ce.set_bytes(NS, 10);
    if (out_scale_bits_ & KQ_MOE_GLU_ALPHA) {
      ce.set_bytes(alpha_, kq_glu_alpha_slot);
    }
    const int sg = kq_moe_fit_sg(kernel, kq_moe_pick_sg(kquant_type_, N, nx));
    MTL::Size group_dims(32, sg, 1);
    MTL::Size grid_dims(N / (sg * 32 / nx), R + NS, T);
    kq_log_dispatch(kname, x, out, grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);
    return;
  }

  // Mixed shexp codecs dispatch the generic "_sx_" instantiations. The
  // half-dot pick keys on the expert codec; the shared slot stays float.
  const int gform = kq_moe_group_glu(T)
      ? kq_moe_group_form(kt, shexp_type_, N, K)
      : KQ_GRP_NONE;
  const bool grp = gform != KQ_GRP_NONE;
  const bool grp_qf = gform == KQ_GRP_QF;
  const bool qf = grp_qf ||
      (!grp && kq_moe_qf(kt, N, K) && kq_qf_shexp_ok(kt, shexp_type_, K));
  const bool use_half = !grp && !qf && kq_moe_half(kt, K) &&
      (shexp_type_ == kt || shexp_type_ == "q5_k" || shexp_type_ == "q6_k" ||
       shexp_type_ == "q8_0");
  const bool dd = !grp && !qf && !use_half && kq_moe_dedup(T, KQ_DD_MAX_T);
  const bool c32 = !grp && !qf && !use_half && !dd &&
      kq_moe_c32_glu(kquant_type_, up_type_, T);
  const int nx = qf || use_half ? 8
      : grp ? kq_moe_group_nx((int64_t)N * (R + NS) * T, K, true)
      : c32 ? kq_moe_c32_nx()
            : kq_moe_pick_nx((int64_t)N * (R + NS) * T, K, true);
  // The qf kernels always carry the shared codec in the name, and so do
  // the grouped ones.
  std::string stem = grp  ? kq_moe_group_stem(gform, kt, shexp_type_)
      : qf                ? kt + "_qf_sx_" + shexp_type_
      : shexp_type_ == kt ? kq_gather_stem_nx(kt, K, nx)
                          : kt + "_sx_" + shexp_type_;
  if (dd && (stem == "q6_k" || stem == "q8_0")) {
    stem += "_ext";
  }
  // silu_limit and swiglu_oai ride the silu instantiations: the epilogue
  // clamps at run time when the limit is positive, and swiglu_oai sets its
  // out-scale bit (kq_glu_epilogue).
  const std::string act =
      act_ == "silu_limit" || act_ == "swiglu_oai" ? "silu" : act_;
  std::string kname = "kq_" + stem + "_moe_glu_gather_shexp_" +
      (use_half ? "h_" : "") + (dd ? "dd_" : "") + (grp ? "grp_" : "") + act +
      (c32 ? "_c32" : "") + kq_nx_suffix(nx) + "_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel_os(
      d, kname, out_scale_bits_, qf || grp ? nullptr : kq_moe_nx_setting());
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  if (use_half) {
    ce.set_threadgroup_memory_length((size_t)K * sizeof(uint16_t), 0);
  }
  ce.set_input_array(gw, 0);
  ce.set_input_array(uw, 1);
  ce.set_input_array(sgw, 2);
  ce.set_input_array(suw, 3);
  ce.set_input_array(x, 4);
  ce.set_input_array(indices, 5);
  ce.set_output_array(out, 6);
  ce.set_bytes(K, 7);
  ce.set_bytes(N, 8);
  ce.set_bytes(limit_, 9);
  ce.set_bytes(NS, 10);
  if (out_scale_bits_ & KQ_MOE_GLU_ALPHA) {
    ce.set_bytes(alpha_, kq_glu_alpha_slot);
  }
  kq_moe_bind_scales(ce, inputs, out_scale_bits_, true);
  const bool tuned = qf || stem == "q6_k" || stem == "q8_0";
  const int sg = kq_moe_fit_sg(
      kernel,
      use_half
          ? kq_moe_pick_sg_h(N, nx)
          : (tuned ? 2 : kq_moe_pick_sg_glu(kquant_type_, up_type_, N, nx)));
  const int tile = qf ? 2 * kq_qf_rps(kt) : sg * 32 / nx;
  MTL::Size group_dims(32, sg, 1);
  MTL::Size grid_dims(N / tile, R + NS, T);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);
  if (!(out_scale_bits_ & KQ_MOE_SHEXP_PRE)) {
    return;
  }
  // The shared slots from the dense route's gate and up products.
  const std::string ename = std::string("kq_moe_shexp_glu_pre_") +
      (act == "gelu" ? "gelu" : "silu") + "_" + kq_type_string(x.dtype());
  auto ek =
      kq_get_kernel_os(d, ename, out_scale_bits_ & KQ_MOE_SWIGLU_OAI, nullptr);
  kq_moe_log_kname(ename, int(ek->maxTotalThreadsPerThreadgroup()));
  ce.set_compute_pipeline_state(ek);
  const size_t pi = kq_moe_pre_input(inputs, out_scale_bits_, true);
  ce.set_input_array(inputs[pi], 0);
  ce.set_input_array(inputs[pi + 1], 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(N, 3);
  ce.set_bytes(R, 4);
  ce.set_bytes(limit_, 5);
  if (out_scale_bits_ & KQ_MOE_GLU_ALPHA) {
    ce.set_bytes(alpha_, kq_glu_alpha_slot);
  }
  MTL::Size egroup(std::min(N, 256), 1, 1);
  MTL::Size egrid(N, NS, T);
  kq_log_dispatch(ename, x, out, egrid, egroup);
  ce.dispatch_threads(egrid, egroup);
}

void KQuantGatherQMVMixKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const auto& w = inputs[0];
  const auto& sw = inputs[1];
  const auto& indices = inputs[3];
  const auto& scores = inputs[4];
  moe_check_stack("gather_qmv_mix_kq", w);
  moe_check_shexp("gather_qmv_mix_kq", sw);
  kq_check_weight_base(w, kquant_type_, "gather_qmv_mix_kq");
  kq_check_weight_base(sw, shexp_type_, "gather_qmv_mix_kq");
  const mx::array x = kq_x_aligned_gpu(inputs[2], s, "gather_qmv_mix_kq x");
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int T = x.shape(0);
  int S = x.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);
  // Shared slots: the last NS of the S slots.
  const int NS = S - indices.shape(1);

  if (kq_moe_tile_mix(T, kquant_type_, shexp_type_, N, K, out_scale_bits_)) {
    kq_gather_qmv_mix_tile_gpu(
        d,
        s,
        kquant_type_,
        shexp_type_,
        w,
        sw,
        x,
        indices,
        scores,
        out,
        T,
        S,
        N,
        K,
        NS);
    return;
  }

  // Grouped form: a routed expert is read once per eight pairs on it.
  const int gform = kq_moe_group_mix(T)
      ? kq_moe_group_form(kquant_type_, shexp_type_, N, K, NS == 1)
      : KQ_GRP_NONE;
  if (gform != KQ_GRP_NONE) {
    kq_gather_qmv_mix_grp_gpu(
        d,
        s,
        gform,
        kquant_type_,
        shexp_type_,
        w,
        sw,
        x,
        indices,
        scores,
        out,
        T,
        S,
        N,
        K,
        NS,
        inputs,
        out_scale_bits_);
    return;
  }

  // The qf kernels take one shared slot; wider shared experts keep Ext.
  if (NS == 1 && kq_moe_qf(kquant_type_, N, K) &&
      kq_qf_shexp_ok(kquant_type_, shexp_type_, K)) {
    const int rps = kq_qf_rps(kquant_type_);
    // With the dense route the shared slot has no simdgroup pair.
    const int pairs = out_scale_bits_ & KQ_MOE_SHEXP_PRE ? S - 1 : S;
    bool sp = kq_qf_mix_sp(T, N, rps, pairs);
    auto qf_name = [&](bool slot_parallel) {
      return "kq_" + kquant_type_ + "_qf_sx_" + shexp_type_ +
          "_gather_qmv_mix" + (slot_parallel ? "_sp_" : "_") +
          kq_type_string(x.dtype());
    };
    std::string kname = qf_name(sp);
    auto kernel = kq_get_kernel_os(d, kname, out_scale_bits_, nullptr);
    if (sp && 64 * pairs > kq_moe_sp_cap(kernel)) {
      sp = false;
      kname = qf_name(sp);
      kernel = kq_get_kernel_os(d, kname, out_scale_bits_, nullptr);
    }
    kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
    auto& ce = mx::metal::get_command_encoder(s);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(w, 0);
    ce.set_input_array(sw, 1);
    ce.set_input_array(x, 2);
    ce.set_input_array(indices, 3);
    ce.set_input_array(scores, 4);
    ce.set_output_array(out, 5);
    ce.set_bytes(K, 6);
    ce.set_bytes(N, 7);
    ce.set_bytes(S, 8);
    const int SC = scores.shape(1);
    ce.set_bytes(SC, 9);
    kq_moe_bind_scales(ce, inputs, out_scale_bits_, false);
    MTL::Size group_dims(32, sp ? 2 * pairs : 2, 1);
    MTL::Size grid_dims(N / (2 * rps), 1, T);
    kq_log_dispatch(kname, x, out, grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);
    return;
  }

  // Same codec-keyed widening as mix_ns. The slot-parallel form serves the
  // generic kernels at nx 8; the tuned q6_k and q8_0 kernels keep their
  // fine tier.
  const bool use_half = kq_moe_half(kquant_type_, K) &&
      (shexp_type_ == kquant_type_ || shexp_type_ == "q5_k" ||
       shexp_type_ == "q6_k" || shexp_type_ == "q8_0");
  const bool dd_ok = !use_half && S <= 16 && kq_moe_dedup(T, KQ_DD_MAX_T);
  const int SP =
      kq_moe_sx_split(kquant_type_, shexp_type_, K, S - NS, NS, use_half, T);
  const std::string sp_stem = shexp_type_ == kquant_type_
      ? kq_gather_stem(kquant_type_, K)
      : kquant_type_ + "_sx_" + shexp_type_;
  // c32 serves the loop and slot-parallel forms; the dedupe form keeps the
  // 16-weight step.
  const bool c32 = !use_half && kq_moe_c32(kquant_type_, T) &&
      kq_moe_c32_shexp_ok(kquant_type_, shexp_type_);
  const int nx_one = c32
      ? kq_moe_c32_down_nx()
      : kq_moe_pick_nx(
            (int64_t)N * T, K, kq_moe_mix_ns_wide(kquant_type_, T, K));
  const bool sp_ok = !use_half && sp_stem != "q6_k" && sp_stem != "q8_0" &&
      nx_one == 8 && kq_moe_sp((int64_t)T * (N / 8), S);
  const int form = kq_moe_mix_form(sp_ok, dd_ok);
  if (form == KQ_MIX_SP) {
    const std::string kname = "kq_" + sp_stem + "_gather_qmv_mix_sp_" +
        (c32 ? "c32_" : "") + kq_type_string(x.dtype());
    auto kernel = kq_get_kernel_os(d, kname, out_scale_bits_, nullptr);
    // 64 threads per simdgroup pair must fit the pipeline; the loop kernel
    // serves the call when they do not.
    const int pairs =
        S - NS + (out_scale_bits_ & KQ_MOE_SHEXP_PRE ? 0 : NS * SP);
    if (64 * pairs <= kq_moe_sp_cap(kernel)) {
      kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
      auto& ce = mx::metal::get_command_encoder(s);
      ce.set_compute_pipeline_state(kernel);
      ce.set_input_array(w, 0);
      ce.set_input_array(sw, 1);
      ce.set_input_array(x, 2);
      ce.set_input_array(indices, 3);
      ce.set_input_array(scores, 4);
      ce.set_output_array(out, 5);
      ce.set_bytes(K, 6);
      ce.set_bytes(N, 7);
      ce.set_bytes(S, 8);
      const int SC = scores.shape(1);
      ce.set_bytes(SC, 9);
      ce.set_bytes(NS, 10);
      ce.set_bytes(SP, 11);
      kq_moe_bind_scales(ce, inputs, out_scale_bits_, false);
      MTL::Size group_dims(32, 2 * pairs, 1);
      MTL::Size grid_dims(N / 8, 1, T);
      kq_log_dispatch(kname, x, out, grid_dims, group_dims);
      ce.dispatch_threadgroups(grid_dims, group_dims);
      return;
    }
  }
  if (form == KQ_MIX_DD) {
    // The pair launch serves two rows per grid step, so the width pick
    // sees a two-row step whatever the block width.
    const int nx = kq_moe_pick_nx(
        (int64_t)N * T, K, kq_moe_mix_ns_wide(kquant_type_, 2, K));
    std::string stem = shexp_type_ == kquant_type_
        ? kq_gather_stem_nx(kquant_type_, K, nx)
        : kquant_type_ + "_sx_" + shexp_type_;
    if (stem == "q6_k" || stem == "q8_0") {
      stem += "_ext";
    }
    kq_gather_qmv_pair_gpu(
        d,
        s,
        "kq_" + stem + "_gather_qmv_mix_dd" + kq_nx_suffix(nx) + "_" +
            kq_type_string(x.dtype()),
        w,
        sw,
        x,
        indices,
        scores,
        out,
        T,
        S,
        N,
        K,
        nx,
        kq_moe_pick_sg(kquant_type_, N, nx),
        NS,
        SP,
        inputs,
        out_scale_bits_);
    return;
  }
  int nx = nx_one;
  if (use_half && nx > 16) {
    nx = 16;
  }
  std::string stem = shexp_type_ == kquant_type_
      ? kq_gather_stem_nx(kquant_type_, K, nx)
      : kquant_type_ + "_sx_" + shexp_type_;
  // Fine tiling on the tuned uniform-codec kernels only (see gather_qmv).
  const bool tuned = stem == "q6_k" || stem == "q8_0";
  const bool fine = tuned && nx == 8 && kq_gather_fine((int64_t)(N / 8) * T);
  std::string kname = "kq_" + stem + "_gather_qmv_mix" +
      (use_half ? "_h" : "") + (c32 ? "_c32" : "") +
      (fine ? "_fine" : kq_nx_suffix(nx)) + "_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel_os(
      d, kname, out_scale_bits_, fine ? nullptr : kq_moe_nx_setting());
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  if (use_half) {
    ce.set_threadgroup_memory_length((size_t)K * sizeof(uint16_t), 0);
  }
  ce.set_input_array(w, 0);
  ce.set_input_array(sw, 1);
  ce.set_input_array(x, 2);
  ce.set_input_array(indices, 3);
  ce.set_input_array(scores, 4);
  ce.set_output_array(out, 5);
  ce.set_bytes(K, 6);
  ce.set_bytes(N, 7);
  ce.set_bytes(S, 8);
  const int SC = scores.shape(1);
  ce.set_bytes(SC, 9);
  ce.set_bytes(NS, 10);
  ce.set_bytes(SP, 11);
  kq_moe_bind_scales(ce, inputs, out_scale_bits_, false);
  const int sg = kq_moe_fit_sg(
      kernel,
      use_half ? kq_moe_pick_sg_h(N, nx)
               : (tuned ? 2 : kq_moe_pick_sg(kquant_type_, N, nx)));
  MTL::Size group_dims(32, sg, 1);
  MTL::Size grid_dims(N / (fine ? 2 : (sg * 32 / nx)), 1, T);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVMixNSKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];

  const auto& w = inputs[0];
  const auto& indices = inputs[2];
  const auto& scores = inputs[3];
  moe_check_stack("gather_qmv_mix_ns_kq", w);
  kq_check_weight_base(w, kquant_type_, "gather_qmv_mix_ns_kq");
  const mx::array x = kq_x_aligned_gpu(inputs[1], s, "gather_qmv_mix_ns_kq x");
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int T = x.shape(0);
  int S = x.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);

  // LoRA: z = x @ A[e] is dispatched first so it overlaps the gather.
  std::optional<mx::array> lora_z;
  KqLoraView lv;
  if (lora_flags_ & KQ_LORA_PRESENT) {
    lv = kq_lora_view_gpu(inputs, 4, lora_flags_, s);
    lora_z = kq_lora_mix_z_gpu(d, s, x, lv, T, S, K);
  }

  // mix_ns is generic for every codec (no tuned variants) -- plain names.
  // No fine tier: the Ext fine variants measured E2E-neutral and were
  // dropped. Decode-scale launches route to the slot-parallel variant
  // unless the codec widens (kq_moe_mix_ns_wide).
  const bool qf = kq_moe_qf(kquant_type_, N, K);
  const bool dd_ok = !qf && S <= 16 && kq_moe_dedup(T, KQ_DD_MAX_T);
  const bool c32_ok = !qf && kq_moe_c32(kquant_type_, T);
  const bool sp_ok =
      (c32_ok
           ? kq_moe_c32_down_nx()
           : kq_moe_pick_nx(
                 (int64_t)N * T, K, kq_moe_mix_ns_wide(kquant_type_, T, K))) ==
          8 &&
      kq_moe_sp((int64_t)T * (N / 8), S);
  const bool dd = kq_moe_mix_form(sp_ok, dd_ok) == KQ_MIX_DD;
  // c32 serves the loop and slot-parallel forms, not the dedupe form.
  const bool c32 = c32_ok && !dd;
  const int nx = c32
      ? kq_moe_c32_down_nx()
      : kq_moe_pick_nx(
            (int64_t)N * T, K, kq_moe_mix_ns_wide(kquant_type_, dd ? 2 : T, K));
  const int gform = kq_moe_group_mix(T)
      ? kq_moe_group_form(kquant_type_, "", N, K)
      : KQ_GRP_NONE;
  if (kq_moe_tile_mix(T, kquant_type_, kquant_type_, N, K, out_scale_bits_)) {
    kq_gather_qmv_mix_tile_gpu(
        d, s, kquant_type_, "", w, w, x, indices, scores, out, T, S, N, K, 0);
  } else if (gform != KQ_GRP_NONE) {
    kq_gather_qmv_mix_grp_gpu(
        d,
        s,
        gform,
        kquant_type_,
        "",
        w,
        w,
        x,
        indices,
        scores,
        out,
        T,
        S,
        N,
        K,
        0,
        inputs,
        out_scale_bits_);
  } else if (qf) {
    const int rps = kq_qf_rps(kquant_type_);
    bool sp = kq_qf_mix_sp(T, N, rps, S);
    auto qf_name = [&](bool slot_parallel) {
      return "kq_" + kquant_type_ + "_qf_gather_qmv_mix_ns" +
          (slot_parallel ? "_sp_" : "_") + kq_type_string(x.dtype());
    };
    std::string kname = qf_name(sp);
    auto kernel = kq_get_kernel_os(d, kname, out_scale_bits_, nullptr);
    if (sp && 64 * S > kq_moe_sp_cap(kernel)) {
      sp = false;
      kname = qf_name(sp);
      kernel = kq_get_kernel_os(d, kname, out_scale_bits_, nullptr);
    }
    kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
    auto& ce = mx::metal::get_command_encoder(s);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(w, 0);
    ce.set_input_array(x, 1);
    ce.set_input_array(indices, 2);
    ce.set_input_array(scores, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(K, 5);
    ce.set_bytes(N, 6);
    ce.set_bytes(S, 7);
    kq_moe_bind_scales(ce, inputs, out_scale_bits_, false);
    MTL::Size group_dims(32, sp ? 2 * S : 2, 1);
    MTL::Size grid_dims(N / (2 * rps), 1, T);
    kq_log_dispatch(kname, x, out, grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  } else if (dd) {
    kq_gather_qmv_pair_gpu(
        d,
        s,
        "kq_" + kquant_type_ + "_gather_qmv_mix_ns_dd" + kq_nx_suffix(nx) +
            "_" + kq_type_string(x.dtype()),
        w,
        w,
        x,
        indices,
        scores,
        out,
        T,
        S,
        N,
        K,
        nx,
        kq_moe_pick_sg(kquant_type_, N, nx),
        0,
        1,
        inputs,
        out_scale_bits_);
  } else {
    bool sp = nx == 8 && kq_moe_sp((int64_t)T * (N / 8), S) &&
        kq_moe_mix_form(true, false) == KQ_MIX_SP;
    auto ns_name = [&](bool slot_parallel) {
      return "kq_" + kquant_type_ + "_gather_qmv_mix_ns" +
          (slot_parallel ? "_sp" : "") + (c32 ? "_c32" : "") +
          (slot_parallel ? "" : kq_nx_suffix(nx)) + "_" +
          kq_type_string(x.dtype());
    };
    std::string kname = ns_name(sp);
    auto kernel =
        kq_get_kernel_os(d, kname, out_scale_bits_, kq_moe_nx_setting());
    // The slot-parallel kernel gives each slot its own simdgroup pair, so
    // its 64 * S threads cannot be halved the way the sg kernels can. When
    // the pipeline will not take that many, drop to the general kernel.
    if (sp && 64 * S > kq_moe_sp_cap(kernel)) {
      sp = false;
      kname = ns_name(sp);
      kernel = kq_get_kernel_os(d, kname, out_scale_bits_, kq_moe_nx_setting());
    }
    kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
    auto& ce = mx::metal::get_command_encoder(s);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(w, 0);
    ce.set_input_array(x, 1);
    ce.set_input_array(indices, 2);
    ce.set_input_array(scores, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(K, 5);
    ce.set_bytes(N, 6);
    ce.set_bytes(S, 7);
    kq_moe_bind_scales(ce, inputs, out_scale_bits_, false);
    const int sg =
        sp ? 2 * S : kq_moe_fit_sg(kernel, kq_moe_pick_sg(kquant_type_, N, nx));
    MTL::Size group_dims(32, sg, 1);
    MTL::Size grid_dims(N / (sp ? 8 : (sg * 32 / nx)), 1, T);
    kq_log_dispatch(kname, x, out, grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
  if (lora_z) {
    kq_lora_mix_apply_gpu(d, s, *lora_z, scores, lv, out, T, S, N);
  }
}

void KQuantMoERouterTopK::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& indices = outputs[0];
  auto& scores = outputs[1];
  // The kernel reads logits as dense rows: a view with other strides is
  // copied first.
  const mx::array logits = kq_row_major_gpu(inputs[0], s);
  indices.set_data(mx::allocator::malloc(indices.nbytes()));
  scores.set_data(mx::allocator::malloc(scores.nbytes()));

  int SHARED = shared_ ? 1 : 0;
  int HAS_PES = has_pes_ ? 1 : 0;
  int HAS_BIAS = has_bias_ ? 1 : 0;
  int SCORING = scoring_;
  float SCALE = scale_;
  int T = logits.shape(0);
  int E = logits.shape(1) - SHARED;
  int R = top_k_;
  int NORM = norm_ ? 1 : 0;
  // pes/bias must be bound buffers even when unused; logits stands in.
  const auto& pes = has_pes_ ? inputs[1] : logits;
  const auto& bias = has_bias_ ? inputs[1 + (has_pes_ ? 1 : 0)] : logits;

  std::string kname = "kq_moe_router_topk_" + kq_type_string(logits.dtype());
  auto kernel = kq_get_kernel(d, kname);
  kq_moe_log_kname(kname, int(kernel->maxTotalThreadsPerThreadgroup()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(logits, 0);
  ce.set_output_array(indices, 1);
  ce.set_output_array(scores, 2);
  ce.set_bytes(E, 3);
  ce.set_bytes(R, 4);
  ce.set_bytes(NORM, 5);
  ce.set_bytes(SHARED, 6);
  ce.set_input_array(pes, 7);
  ce.set_bytes(HAS_PES, 8);
  ce.set_input_array(bias, 9);
  ce.set_bytes(HAS_BIAS, 10);
  ce.set_bytes(SCORING, 11);
  ce.set_bytes(SCALE, 12);
  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims(T, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantMoEGLU::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.moe_glu_gather] requires Metal.");
}

void KQuantGatherQMVBias::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_bias] requires Metal.");
}

void KQuantGatherQMVMixBias::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_mix_bias] requires Metal.");
}

void KQuantMoEGLUKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.moe_glu_gather_kq] requires Metal.");
}

void KQuantGatherQMVKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_kq] requires Metal.");
}

void KQuantMoEGLUShexpKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_glu_gather_shexp_kq] requires Metal.");
}

void KQuantGatherQMVMixKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_mix_kq] requires Metal.");
}

void KQuantGatherQMVMixNSKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_mix_ns_kq] requires Metal.");
}

void KQuantMoERouterTopK::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.moe_router_topk] requires Metal.");
}

#endif

void KQuantMoEGLU::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_glu_gather] has no CPU implementation.");
}

void KQuantGatherQMVBias::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_bias] has no CPU implementation.");
}

void KQuantGatherQMVMixBias::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_mix_bias] has no CPU implementation.");
}

bool KQuantMoEGLU::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoEGLU&>(other);
  return alpha_ == o.alpha_ && limit_ == o.limit_;
}

bool KQuantGatherQMVBias::is_equivalent(const mx::Primitive&) const {
  return true;
}

bool KQuantGatherQMVMixBias::is_equivalent(const mx::Primitive&) const {
  return true;
}

void KQuantMoEGLUKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_glu_gather_kq] has no CPU implementation.");
}

void KQuantGatherQMVKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_kq] has no CPU implementation.");
}

bool KQuantMoEGLUKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoEGLUKQ&>(other);
  return kquant_type_ == o.kquant_type_ && up_type_ == o.up_type_ &&
      act_ == o.act_ && limit_ == o.limit_ && alpha_ == o.alpha_ &&
      out_scale_bits_ == o.out_scale_bits_;
}

bool KQuantGatherQMVKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVKQ&>(other);
  return kquant_type_ == o.kquant_type_ && lora_flags_ == o.lora_flags_ &&
      out_scale_bits_ == o.out_scale_bits_;
}

std::vector<mx::Shape> KQuantMoEGLUKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  // Indices are the last input before the scales in both the unbiased and
  // biased layouts.
  const auto& idx =
      inputs[inputs.size() - 1 - kq_moe_scale_inputs(out_scale_bits_, true)];
  return {{idx.shape(0), idx.shape(1), inputs[0].shape(1)}};
}

void KQuantMoEGLUShexpKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_glu_gather_shexp_kq] has no CPU implementation.");
}

void KQuantGatherQMVMixKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_mix_kq] has no CPU implementation.");
}

bool KQuantMoEGLUShexpKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoEGLUShexpKQ&>(other);
  return kquant_type_ == o.kquant_type_ && up_type_ == o.up_type_ &&
      act_ == o.act_ && shexp_type_ == o.shexp_type_ &&
      shexp_up_type_ == o.shexp_up_type_ && limit_ == o.limit_ &&
      alpha_ == o.alpha_ && out_scale_bits_ == o.out_scale_bits_;
}

bool KQuantGatherQMVMixKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVMixKQ&>(other);
  return kquant_type_ == o.kquant_type_ && shexp_type_ == o.shexp_type_ &&
      out_scale_bits_ == o.out_scale_bits_;
}

std::vector<mx::Shape> KQuantMoEGLUShexpKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  // R routed slots, then one shared slot per routed width of the shexp.
  const int NS = inputs[2].shape(0) / inputs[0].shape(1);
  return {{inputs[5].shape(0), inputs[5].shape(1) + NS, inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantGatherQMVMixKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[2].shape(0), inputs[0].shape(1)}};
}

void KQuantGatherQMVMixNSKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_mix_ns_kq] has no CPU implementation.");
}

bool KQuantGatherQMVMixNSKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVMixNSKQ&>(other);
  return kquant_type_ == o.kquant_type_ && lora_flags_ == o.lora_flags_ &&
      out_scale_bits_ == o.out_scale_bits_;
}

std::vector<mx::Shape> KQuantGatherQMVMixNSKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[1].shape(0), inputs[0].shape(1)}};
}

void KQuantMoERouterTopK::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_router_topk] has no CPU implementation.");
}

bool KQuantMoERouterTopK::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoERouterTopK&>(other);
  return top_k_ == o.top_k_ && norm_ == o.norm_ && shared_ == o.shared_ &&
      has_pes_ == o.has_pes_ && has_bias_ == o.has_bias_ &&
      scoring_ == o.scoring_ && scale_ == o.scale_;
}

std::vector<mx::Shape> KQuantMoERouterTopK::output_shapes(
    const std::vector<mx::array>& inputs) {
  int T = inputs[0].shape(0);
  return {{T, top_k_}, {T, top_k_ + (shared_ ? 1 : 0)}};
}

std::vector<mx::Shape> KQuantGatherQMVKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  // Indices are the last op input in both the unbiased and biased layouts
  // (the LoRA operands and the scale, when present, trail them).
  const auto& idx = inputs
      [inputs.size() - 1 - kq_lora_input_count(lora_flags_) -
       kq_moe_scale_inputs(out_scale_bits_, false)];
  return {{idx.shape(0), idx.shape(1), inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantMoEGLU::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[7].shape(0), inputs[7].shape(1), inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantGatherQMVBias::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[4].shape(0), inputs[4].shape(1), inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantGatherQMVMixBias::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[4].shape(0), inputs[0].shape(1)}};
}

// The generation of a Mac GPU, from applegpu_g<gen><suffix> with a two digit
// gen and a lower case suffix other than p; 0 for a phone or anything else.
static int moe_mac_gpu_gen(const std::string& arch) {
  const std::string head = "applegpu_g";
  if (arch.size() != head.size() + 3 || arch.compare(0, head.size(), head)) {
    return 0;
  }
  const char d0 = arch[head.size()];
  const char d1 = arch[head.size() + 1];
  const char suffix = arch[head.size() + 2];
  if (!std::isdigit(static_cast<unsigned char>(d0)) ||
      !std::isdigit(static_cast<unsigned char>(d1)) ||
      !std::islower(static_cast<unsigned char>(suffix)) || suffix == 'p') {
    return 0;
  }
  return (d0 - '0') * 10 + (d1 - '0');
}

// K-lanes per row of the c32 GLU gathers: 16 from gen 17 (M5), where it
// beat 8 by 2 to 6% on a Muse Spark MoE block (M5 Max, E=256, top 8,
// N=K=4096). 8 elsewhere.
int moe_c32_nx(const std::string& arch) {
  return moe_mac_gpu_gen(arch) >= 17 ? 16 : 8;
}

// iq2_s on every GPU (M3 Max and M5 Max measured). The other grid codecs
// from gen 17 (M5), measured on an M5 Max only: at E=32, top 8, N=K=4096
// the 16-weight gathers took 1 to 23% longer (iq1_s most, iq3_xxs least).
bool moe_c32_on(const std::string& arch, const std::string& codec) {
  return codec == "iq2_s" ||
      (kq_moe_has_c32(codec) && moe_mac_gpu_gen(arch) >= 17);
}

bool codec_has_c32(const std::string& codec) {
  return kq_moe_has_c32(codec);
}

// The M = 1 quantized_matmul of a grid codec through the c32 row step: from
// gen 17 (M5), measured on an M5 Max only. At K=4096, N=14336 the qmv
// kernels took 2 to 21% longer (iq2_s and iq3_s most, iq2_xxs least).
bool qmv_c32_on(const std::string& arch, const std::string& codec) {
  return moe_mac_gpu_gen(arch) >= 17 && kq_moe_has_c32(codec);
}

// Codecs with the fused kq GLU/gather kernels wired (kq_moe_glu_kq.h):
// tuned kernels for q6_k/q8_0, generic Ext-trait kernels for the rest of the
// codec matrix. Unsupported codecs must fall back to the stock path
// per-tensor (callers gate on this via ops throwing invalid_argument).
bool codec_has_moe_glu(const std::string& t) {
  return t == "q2_k" || t == "q3_k" || t == "q4_k" || t == "q5_k" ||
      t == "q6_k" || t == "q8_0" || t == "q4_0" || t == "q4_1" || t == "q5_0" ||
      t == "q5_1" || t == "iq4_nl" || t == "iq4_xs" || t == "iq3_s" ||
      t == "iq3_xxs" || t == "iq2_xxs" || t == "iq2_xs" || t == "iq2_s" ||
      t == "iq1_s" || t == "iq1_m" || t == "stq1_0" || t == "pq2_0" ||
      t == "q2_0" || t == "ptq1_0" || t == "mxfp4" || t == "nvfp4";
}

bool moe_qf_kernels(const std::string& t) {
#ifdef _METAL_
  return kq_moe_qf_on(t);
#else
  return false;
#endif
}

namespace {

// Mixed-codec shared experts with every kernel variant: shexp codec ==
// expert codec, or the UD-style q5_k / q6_k / q8_0 upcast over anything.
bool shexp_upcast_pair(
    const std::string& kquant_type,
    const std::string& shexp_type) {
  return codec_has_moe_glu(kquant_type) && codec_has_moe_glu(shexp_type) &&
      (shexp_type == kquant_type || shexp_type == "q5_k" ||
       shexp_type == "q6_k" || shexp_type == "q8_0");
}

bool lowbit_shexp_codec(const std::string& t) {
  return t == "iq4_xs" || t == "iq3_s" || t == "q6_k";
}

// A shared down slot of `width` weights starts on a whole block of `codec`.
bool shexp_slot_on_block(const std::string& codec, int64_t width) {
  const KQuantCodec* c = codec_by_name(codec);
  return c != nullptr && width > 0 && width % c->weights_per_block == 0;
}

} // namespace

bool shexp_glu_combo_has_kernel(
    const std::string& kquant_type,
    const std::string& shexp_gate_type,
    const std::string& shexp_up_type,
    int shared_slots) {
  if (shared_slots < 1 || (shared_slots > 1 && !kq_moe_shexp_multi())) {
    return false;
  }
  if (shexp_gate_type == shexp_up_type &&
      shexp_upcast_pair(kquant_type, shexp_gate_type)) {
    return true;
  }
  // The "_sx2_" family (kq_moe_glu_kq.metal): low-bit i-quant expert
  // stacks, gate and up each one of iq4_xs / iq3_s / q6_k.
  return (kquant_type == "iq2_s" || kquant_type == "iq2_xxs" ||
          kquant_type == "iq1_m") &&
      lowbit_shexp_codec(shexp_gate_type) && lowbit_shexp_codec(shexp_up_type);
}

namespace {

// A split routed GLU pair of kq_glu_pairs.h: gate in g, up in u, g != u.
bool glu_split_pair(const std::string& g, const std::string& u) {
#define KQ_GLU_PAIR_MATCH(gc, gt, uc, ut) \
  if (g == #gc && u == #uc) {             \
    return true;                          \
  }
  KQ_GLU_SPLIT_PAIRS(KQ_GLU_PAIR_MATCH)
#undef KQ_GLU_PAIR_MATCH
  return false;
}

} // namespace

bool moe_glu_pair_has_kernel(
    const std::string& gate_type,
    const std::string& up_type,
    const std::string& shexp_gate_type,
    const std::string& shexp_up_type,
    int shared_slots) {
  const std::string sg = shexp_gate_type;
  const std::string su = shexp_up_type.empty() ? sg : shexp_up_type;
  if (gate_type == up_type) {
    return sg.empty()
        ? codec_has_moe_glu(gate_type)
        : shexp_glu_combo_has_kernel(gate_type, sg, su, shared_slots);
  }
  if (!glu_split_pair(gate_type, up_type)) {
    return false;
  }
  if (sg.empty()) {
    return su.empty();
  }
  // The split pairs' shared-expert kernels: one q5_k, q6_k or q8_0 codec.
  if (shared_slots < 1 || (shared_slots > 1 && !kq_moe_shexp_multi())) {
    return false;
  }
  return su == sg && (sg == "q5_k" || sg == "q6_k" || sg == "q8_0");
}

bool shexp_mix_combo_has_kernel(
    const std::string& kquant_type,
    const std::string& shexp_type,
    int shared_slots,
    int slot_width) {
  if (shared_slots < 1 || (shared_slots > 1 && !kq_moe_shexp_multi())) {
    return false;
  }
  if (!shexp_upcast_pair(kquant_type, shexp_type) &&
      !(kquant_type == "q2_0" && shexp_type == "iq4_nl")) {
    return false;
  }
  // Each shared slot starts on a whole block of the shared codec, so a
  // multi-slot probe needs the slot width.
  if (shared_slots > 1 && slot_width <= 0) {
    return false;
  }
  return slot_width <= 0 || shexp_slot_on_block(shexp_type, slot_width);
}

bool shexp_mix_slot_parallel(
    const std::string& kquant_type,
    const std::string& shexp_type,
    int shared_slots,
    int slot_width) {
  // The tuned q6_k and q8_0 kernels keep their fine tier.
  const bool tuned = shexp_type == kquant_type &&
      (kquant_type == "q6_k" || kquant_type == "q8_0");
  return !tuned &&
      shexp_mix_combo_has_kernel(
          kquant_type, shexp_type, shared_slots, slot_width);
}

namespace {

// Shared validation for one packed-mxfp4 expert stack (w, scales, bias).
void check_expert_stack(
    const char* op,
    const mx::array& w,
    const mx::array& sc,
    const mx::array& b,
    int K) {
  if (w.ndim() != 3 || w.dtype() != mx::uint32) {
    throw std::invalid_argument(
        std::string(op) + " expert weights must be uint32 [E, N, K/8].");
  }
  if (sc.ndim() != 3 || sc.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) + " expert scales must be uint8 [E, N, K/32].");
  }
  if (w.shape(2) * 8 != K || sc.shape(2) * 32 != K) {
    throw std::invalid_argument(
        std::string(op) + " weight/scale trailing dims do not match K.");
  }
  if (w.shape(0) != sc.shape(0) || w.shape(1) != sc.shape(1)) {
    throw std::invalid_argument(
        std::string(op) + " weight/scale expert dims do not match.");
  }
  if (b.ndim() != 2 || b.shape(0) != w.shape(0) || b.shape(1) != w.shape(1)) {
    throw std::invalid_argument(std::string(op) + " bias must be [E, N].");
  }
  if (w.shape(1) % 8 != 0) {
    throw std::invalid_argument(
        std::string(op) + " N must be a multiple of 8.");
  }
  if (K % 32 != 0) {
    throw std::invalid_argument(
        std::string(op) + " K must be a multiple of 32.");
  }
  // An operand without memory is checked at eval.
  if ((kq_has_layout(w) && !w.flags().row_contiguous) ||
      (kq_has_layout(sc) && !sc.flags().row_contiguous)) {
    throw std::invalid_argument(
        std::string(op) + " weights/scales must be row-contiguous.");
  }
}

mx::array prep_bias(const mx::array& b, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(b, mx::float32, s), false, s);
}

// x or router logits for a kernel that reads dense rows. An array with
// memory is copied here when it is not row-major. An array without memory
// reports the default dense strides, so it goes in as it is and the eval
// decides (kq_row_major_gpu): x is unevaluated on nearly every call, and a
// node here would cost every one of them.
mx::array moe_dense_x(const mx::array& x, mx::StreamOrDevice s) {
  if (kq_has_layout(x) && !x.flags().row_contiguous) {
    return mx::contiguous(x, false, s);
  }
  return x;
}

mx::array prep_indices(const mx::array& idx, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(idx, mx::uint32, s), false, s);
}

// Shared validation for one K-quant wire-byte expert stack.
void check_kq_expert_stack(
    const char* op,
    const mx::array& w,
    const std::string& kquant_type,
    int K) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr || !codec_has_moe_glu(kquant_type)) {
    throw std::invalid_argument(
        std::string(op) + " codec '" + kquant_type +
        "' has no fused MoE kernel.");
  }
  if (w.ndim() != 3 || w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) +
        " expert weights must be uint8 [E, N, bytes_per_row].");
  }
  const int64_t bpr =
      (int64_t)K / codec->weights_per_block * codec->bytes_per_block;
  if (K % codec->weights_per_block != 0 || w.shape(2) != bpr) {
    throw std::invalid_argument(
        std::string(op) + " weight trailing dim does not match K for '" +
        kquant_type + "'.");
  }
  // Whole blocks suffice: the generic Ext-trait kernels stride K in
  // 16-weight chunks; the tuned q6_k/q8_0 kernels additionally need
  // K % 256 == 0 and kq_gather_stem falls back to the generic q8_0_ext
  // instantiations otherwise (q6_k's superblock IS 256).
  if (w.shape(1) % 8 != 0) {
    throw std::invalid_argument(
        std::string(op) + " N must be a multiple of 8.");
  }
  // A stack without memory is checked at eval.
  if (kq_has_layout(w)) {
    kq_check_row_major(w, std::string(op) + " weights");
  }
}

// The shared expert's gate or up tensor [k N, bytes_per_row(K)]: k shared
// slots of N rows each. Returns k.
int check_kq_shexp_rows(
    const char* op,
    const mx::array& w,
    const std::string& kquant_type,
    int K,
    int N) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        std::string(op) + " unknown shared-expert codec '" + kquant_type +
        "'.");
  }
  if (w.ndim() != 2 || w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) +
        " shared-expert weights must be uint8 [k * N, bytes_per_row].");
  }
  const int64_t bpr =
      (int64_t)K / codec->weights_per_block * codec->bytes_per_block;
  if (w.shape(1) != bpr || w.shape(0) < N || w.shape(0) % N != 0) {
    throw std::invalid_argument(
        std::string(op) +
        " shared-expert weights must be [k * N, bytes_per_row(K)] for a "
        "whole k, N the expert stack width.");
  }
  const int k = w.shape(0) / N;
  if (k > 1 && !kq_moe_shexp_multi()) {
    throw std::invalid_argument(
        std::string(op) +
        " a shared expert wider than one routed expert is off "
        "(KQ_MOE_SHEXP_MULTI=0).");
  }
  if (kq_has_layout(w)) {
    kq_check_row_major(w, std::string(op) + " shared-expert weights");
  }
  return k;
}

// The shared expert's down tensor [N, bytes_per_row(k K)]: shared slot j
// reads columns [j K, (j + 1) K) of each row, so K must be whole blocks of
// the shared codec. Returns k.
int check_kq_shexp_down(
    const char* op,
    const mx::array& w,
    const std::string& kquant_type,
    int K,
    int N) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        std::string(op) + " unknown shared-expert codec '" + kquant_type +
        "'.");
  }
  if (w.ndim() != 2 || w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) +
        " shared-expert weights must be uint8 [N, bytes_per_row].");
  }
  if (!shexp_slot_on_block(kquant_type, K)) {
    throw std::invalid_argument(
        std::string(op) + " a shared-expert slot of " + std::to_string(K) +
        " weights is not a whole number of '" + kquant_type + "' blocks (" +
        std::to_string(codec->weights_per_block) + " weights).");
  }
  const int64_t bpr =
      (int64_t)K / codec->weights_per_block * codec->bytes_per_block;
  if (w.shape(0) != N || w.shape(1) < bpr || w.shape(1) % bpr != 0) {
    throw std::invalid_argument(
        std::string(op) +
        " shared-expert weights must be [N, bytes_per_row(k * K)] for a "
        "whole k, K the expert stack's input width.");
  }
  const int k = int(w.shape(1) / bpr);
  if (k > 1 && !kq_moe_shexp_multi()) {
    throw std::invalid_argument(
        std::string(op) +
        " a shared expert wider than one routed expert is off "
        "(KQ_MOE_SHEXP_MULTI=0).");
  }
  if (kq_has_layout(w)) {
    kq_check_row_major(w, std::string(op) + " shared-expert weights");
  }
  return k;
}

mx::array prep_scores(const mx::array& sc, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(sc, mx::float32, s), false, s);
}

// The scale mode of a fused MoE op's expert stacks (N columns, E experts),
// and for a GLU the up scale's, which must match it.
int moe_scale_bits(
    const char* name,
    const std::string& kquant_type,
    int N,
    int64_t E,
    const std::optional<mx::array>& scale,
    const char* what,
    const std::optional<mx::array>* up = nullptr,
    const char* up_what = nullptr) {
  const int mode =
      kq_out_scale_mode(scale, kquant_type, true, N, E, name, what);
  if (up != nullptr &&
      kq_out_scale_mode(*up, kquant_type, true, N, E, name, up_what) != mode) {
    throw std::invalid_argument(
        moe_op(name) + " " + what + " and " + up_what +
        " go together, with one shape.");
  }
  return mode;
}

// The shared-expert scale bit of a fused MoE op: float32 (1,) on an nvfp4
// shared expert, and for a GLU with an up scale beside it.
int moe_shexp_scale_bit(
    const char* name,
    const std::string& shexp_type,
    const std::optional<mx::array>& scale,
    const char* what,
    const std::optional<mx::array>* up = nullptr,
    const char* up_what = nullptr) {
  auto check = [&](const std::optional<mx::array>& v, const char* w) {
    if (!v.has_value()) {
      return;
    }
    if (shexp_type != "nvfp4") {
      throw std::invalid_argument(
          moe_op(name) + " " + w + " is for nvfp4 shared-expert weights, " +
          "got '" + shexp_type + "'.");
    }
    if (v->dtype() != mx::float32 || v->ndim() != 1 || v->shape(0) != 1) {
      std::ostringstream msg;
      msg << moe_op(name) << " " << w << " must be float32 of shape (1,), got "
          << v->dtype() << " " << v->shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  };
  check(scale, what);
  if (up != nullptr) {
    check(*up, up_what);
    if (up->has_value() != scale.has_value()) {
      throw std::invalid_argument(
          moe_op(name) + " " + what + " and " + up_what + " go together.");
    }
  }
  return scale.has_value() ? KQ_MOE_SHEXP_SCALE : 0;
}

// Appends the given scales, last, in the order the kernels bind them.
void moe_push_scales(
    std::vector<mx::array>& inputs,
    const mx::Stream& s,
    std::initializer_list<const std::optional<mx::array>*> scales) {
  for (const auto* v : scales) {
    if (v->has_value()) {
      inputs.push_back(kq_row_major(**v, s));
    }
  }
}

} // namespace

mx::array moe_glu_gather(
    mx::array x,
    mx::array gate_w,
    mx::array gate_scales,
    mx::array gate_bias,
    mx::array up_w,
    mx::array up_scales,
    mx::array up_bias,
    mx::array indices,
    float alpha,
    float limit,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather] x must be 2-D [T, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather] indices must be [T, R].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather] x must be float16 or bfloat16.");
  }
  int K = x.shape(1);
  check_expert_stack(
      "[mlx_kquant.moe_glu_gather]", gate_w, gate_scales, gate_bias, K);
  check_expert_stack(
      "[mlx_kquant.moe_glu_gather]", up_w, up_scales, up_bias, K);
  if (gate_w.shape(0) != up_w.shape(0) || gate_w.shape(1) != up_w.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather] gate/up expert shapes must match.");
  }

  auto x_c = moe_dense_x(x, s);
  mx::Shape out_shape = {x.shape(0), indices.shape(1), gate_w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantMoEGLU>(s, alpha, limit),
      {std::move(gate_w),
       std::move(gate_scales),
       prep_bias(gate_bias, s),
       std::move(up_w),
       std::move(up_scales),
       prep_bias(up_bias, s),
       std::move(x_c),
       prep_indices(indices, s)});
}

mx::array gather_qmv_bias(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array bias,
    mx::array indices,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() != 3) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_bias] x must be 3-D [T, R, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != x.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_bias] indices must be [T, R] matching x.");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_bias] x must be float16 or bfloat16.");
  }
  int K = x.shape(2);
  check_expert_stack("[mlx_kquant.gather_qmv_bias]", w, scales, bias, K);

  auto x_c = moe_dense_x(x, s);
  mx::Shape out_shape = {x.shape(0), x.shape(1), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVBias>(s),
      {std::move(w),
       std::move(scales),
       prep_bias(bias, s),
       std::move(x_c),
       prep_indices(indices, s)});
}

mx::array gather_qmv_mix_bias(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array bias,
    mx::array indices,
    mx::array scores,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.gather_qmv_mix_bias]";
  if (x.ndim() != 3) {
    throw std::invalid_argument(std::string(op) + " x must be 3-D [T, S, K].");
  }
  int S = x.shape(1);
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != S) {
    throw std::invalid_argument(std::string(op) + " indices must be [T, S].");
  }
  if (scores.ndim() != 2 || scores.shape(0) != x.shape(0) ||
      scores.shape(1) != S) {
    throw std::invalid_argument(std::string(op) + " scores must be [T, S].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16 or bfloat16.");
  }
  int K = x.shape(2);
  check_expert_stack(op, w, scales, bias, K);

  auto x_c = moe_dense_x(x, s);
  mx::Shape out_shape = {x.shape(0), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVMixBias>(s),
      {std::move(w),
       std::move(scales),
       prep_bias(bias, s),
       std::move(x_c),
       prep_indices(indices, s),
       prep_scores(scores, s)});
}

mx::array moe_glu_gather_kq(
    mx::array x,
    mx::array gate_w,
    mx::array up_w,
    const std::string& kquant_type,
    mx::array indices,
    const std::string& act,
    float limit,
    const std::optional<mx::array>& gate_bias,
    const std::optional<mx::array>& up_bias,
    float alpha,
    const std::optional<mx::array>& gate_scale,
    const std::optional<mx::array>& up_scale,
    const std::string& up_kquant_type,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const std::string up_type =
      up_kquant_type.empty() ? kquant_type : up_kquant_type;
  if (up_type != kquant_type) {
    if (!moe_glu_pair_has_kernel(kquant_type, up_type)) {
      throw std::invalid_argument(
          "[mlx_kquant.moe_glu_gather_kq] no fused kernel for gate codec '" +
          kquant_type + "' with up codec '" + up_type + "'.");
    }
    if (gate_bias.has_value() || up_bias.has_value() ||
        gate_scale.has_value() || up_scale.has_value()) {
      throw std::invalid_argument(
          "[mlx_kquant.moe_glu_gather_kq] biases and output scales need the "
          "gate and up stacks in one codec.");
    }
  }
  if (x.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] x must be 2-D [T, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] indices must be [T, R].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] x must be float16 or bfloat16.");
  }
  if (act != "silu" && act != "gelu" && act != "silu_limit" &&
      act != "swiglu_clamp" && act != "swiglu_oai" && act != "situ") {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] act must be 'silu', 'gelu', "
        "'silu_limit', 'swiglu_clamp', 'swiglu_oai' or 'situ'.");
  }
  if (act == "situ" && (!(limit > 0.0f) || !(alpha >= 0.0f))) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] act 'situ' requires limit (beta) "
        "> 0 and alpha (linear_beta) >= 0.");
  }
  if (act == "silu_limit" && !(limit > 0.0f)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] act 'silu_limit' requires limit > 0.");
  }
  if (act == "swiglu_oai" && (!(limit > 0.0f) || !(alpha > 0.0f))) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] act 'swiglu_oai' requires limit > 0 "
        "and alpha > 0.");
  }
  const bool biased = gate_bias.has_value() || up_bias.has_value();
  if (act == "swiglu_clamp") {
    if (!gate_bias.has_value() || !up_bias.has_value() || !(limit > 0.0f) ||
        !(alpha > 0.0f)) {
      throw std::invalid_argument(
          "[mlx_kquant.moe_glu_gather_kq] act 'swiglu_clamp' requires "
          "gate_bias, up_bias, limit > 0 and alpha > 0.");
    }
    if (kquant_type != "mxfp4" && kquant_type != "nvfp4") {
      throw std::invalid_argument(
          "[mlx_kquant.moe_glu_gather_kq] biased kernels are instantiated "
          "for mxfp4/nvfp4 only.");
    }
  } else if (biased) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] gate_bias/up_bias require act "
        "'swiglu_clamp'.");
  }
  int K = x.shape(1);
  check_kq_expert_stack(
      "[mlx_kquant.moe_glu_gather_kq]", gate_w, kquant_type, K);
  check_kq_expert_stack("[mlx_kquant.moe_glu_gather_kq]", up_w, up_type, K);
  kq_check_weight_base_at_build(gate_w, kquant_type, s, "moe_glu_gather_kq");
  kq_check_weight_base_at_build(up_w, up_type, s, "moe_glu_gather_kq");
  if (gate_w.shape(0) != up_w.shape(0) || gate_w.shape(1) != up_w.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] gate/up expert shapes must match.");
  }
  if (biased) {
    for (const auto* b : {&*gate_bias, &*up_bias}) {
      if (b->ndim() != 2 || b->shape(0) != gate_w.shape(0) ||
          b->shape(1) != gate_w.shape(1)) {
        throw std::invalid_argument(
            "[mlx_kquant.moe_glu_gather_kq] biases must be [E, N] matching "
            "the expert stacks.");
      }
    }
  }
  const int os_bits = moe_scale_bits(
                          "moe_glu_gather_kq",
                          kquant_type,
                          gate_w.shape(1),
                          gate_w.shape(0),
                          gate_scale,
                          "gate_scale",
                          &up_scale,
                          "up_scale") |
      (act == "swiglu_oai" ? KQ_MOE_SWIGLU_OAI : 0) |
      (act == "situ" ? KQ_MOE_SITU : 0);

  auto x_c = moe_dense_x(x, s);
  mx::Shape out_shape = {x.shape(0), indices.shape(1), gate_w.shape(1)};
  std::vector<mx::array> op_inputs = {std::move(gate_w), std::move(up_w)};
  if (biased) {
    op_inputs.push_back(prep_bias(*gate_bias, s));
    op_inputs.push_back(prep_bias(*up_bias, s));
  }
  op_inputs.push_back(std::move(x_c));
  op_inputs.push_back(prep_indices(indices, s));
  moe_push_scales(op_inputs, s, {&gate_scale, &up_scale});
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantMoEGLUKQ>(
          s,
          kquant_type,
          act,
          (act == "silu_limit" || act == "swiglu_clamp" ||
           act == "swiglu_oai" || act == "situ")
              ? limit
              : 0.0f,
          (act == "swiglu_clamp" || act == "swiglu_oai" || act == "situ")
              ? alpha
              : 0.0f,
          os_bits,
          up_type),
      std::move(op_inputs));
}

mx::array gather_qmv_kq(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array indices,
    const std::optional<mx::array>& bias,
    const std::optional<mx::array>& lora_a,
    const std::optional<mx::array>& lora_b,
    const std::optional<mx::array>& lora_ids,
    const std::optional<mx::array>& lora_table,
    const std::optional<mx::array>& lora_rows,
    const std::optional<mx::array>& out_scale,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() != 3) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_kq] x must be 3-D [T, R, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != x.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_kq] indices must be [T, R] matching x.");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_kq] x must be float16 or bfloat16.");
  }
  int K = x.shape(2);
  check_kq_expert_stack("[mlx_kquant.gather_qmv_kq]", w, kquant_type, K);
  kq_check_weight_base_at_build(w, kquant_type, s, "gather_qmv_kq");
  if (bias.has_value()) {
    if (kquant_type != "mxfp4" && kquant_type != "nvfp4") {
      throw std::invalid_argument(
          "[mlx_kquant.gather_qmv_kq] biased kernels are instantiated for "
          "mxfp4/nvfp4 only.");
    }
    if (bias->ndim() != 2 || bias->shape(0) != w.shape(0) ||
        bias->shape(1) != w.shape(1)) {
      throw std::invalid_argument(
          "[mlx_kquant.gather_qmv_kq] bias must be [E, N] matching the "
          "expert stack.");
    }
  }
  const int os_bits = moe_scale_bits(
      "gather_qmv_kq",
      kquant_type,
      w.shape(1),
      w.shape(0),
      out_scale,
      "out_scale");

  auto x_c = moe_dense_x(x, s);
  const int E = w.shape(0);
  const int N = w.shape(1);
  mx::Shape out_shape = {x.shape(0), x.shape(1), N};
  std::vector<mx::array> op_inputs = {std::move(w)};
  if (bias.has_value()) {
    op_inputs.push_back(prep_bias(*bias, s));
  }
  op_inputs.push_back(std::move(x_c));
  op_inputs.push_back(prep_indices(indices, s));
  // The adapter's expert per row defaults to the routing indices.
  const std::optional<mx::array> ids_or =
      (lora_a.has_value() && !lora_ids.has_value())
      ? std::optional<mx::array>(indices)
      : lora_ids;
  const int lora_flags = kq_lora_prep(
      "[mlx_kquant.gather_qmv_kq]",
      lora_a,
      lora_b,
      ids_or,
      lora_table,
      lora_rows,
      dt,
      K,
      N,
      E,
      indices.shape(),
      1,
      s,
      op_inputs);
  moe_push_scales(op_inputs, s, {&out_scale});
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVKQ>(s, kquant_type, lora_flags, os_bits),
      std::move(op_inputs));
}

mx::array moe_glu_gather_shexp_kq(
    mx::array x,
    mx::array gate_w,
    mx::array up_w,
    mx::array shexp_gate_w,
    mx::array shexp_up_w,
    const std::string& kquant_type,
    mx::array indices,
    const std::string& act,
    const std::string& shexp_kquant_type,
    float limit,
    const std::string& shexp_up_kquant_type,
    const std::optional<mx::array>& gate_scale,
    const std::optional<mx::array>& up_scale,
    const std::optional<mx::array>& shexp_gate_scale,
    const std::optional<mx::array>& shexp_up_scale,
    float alpha,
    const std::string& up_kquant_type,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_glu_gather_shexp_kq]";
  const std::string up_type =
      up_kquant_type.empty() ? kquant_type : up_kquant_type;
  const bool split = up_type != kquant_type;
  const std::string shexp_type =
      shexp_kquant_type.empty() ? kquant_type : shexp_kquant_type;
  const std::string shexp_up_type =
      shexp_up_kquant_type.empty() ? shexp_type : shexp_up_kquant_type;
  if (split) {
    if (!moe_glu_pair_has_kernel(
            kquant_type, up_type, shexp_type, shexp_up_type)) {
      throw std::invalid_argument(
          std::string(op) + " no fused kernel for gate codec '" + kquant_type +
          "' with up codec '" + up_type + "' and shared-expert codecs '" +
          shexp_type + "' (gate) and '" + shexp_up_type + "' (up).");
    }
    if (gate_scale.has_value() || up_scale.has_value()) {
      throw std::invalid_argument(
          std::string(op) +
          " output scales need the gate and up stacks in one codec.");
    }
  }
  const bool upcast_pair = shexp_up_type == shexp_type &&
      (split || shexp_upcast_pair(kquant_type, shexp_type));
  if ((!split &&
       !shexp_glu_combo_has_kernel(kquant_type, shexp_type, shexp_up_type)) ||
      (!upcast_pair && act != "silu" && act != "swiglu_oai")) {
    throw std::invalid_argument(
        std::string(op) + " no fused kernel for expert codec '" + kquant_type +
        "' with shared-expert codecs '" + shexp_type + "' (gate) and '" +
        shexp_up_type + "' (up), act '" + act + "'.");
  }
  if (x.ndim() != 2) {
    throw std::invalid_argument(std::string(op) + " x must be 2-D [T, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0)) {
    throw std::invalid_argument(std::string(op) + " indices must be [T, R].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16 or bfloat16.");
  }
  if (act != "silu" && act != "gelu" && act != "silu_limit" &&
      act != "swiglu_oai") {
    throw std::invalid_argument(
        std::string(op) +
        " act must be 'silu', 'gelu', 'silu_limit' or 'swiglu_oai'.");
  }
  if (act == "silu_limit" && !(limit > 0.0f)) {
    throw std::invalid_argument(
        std::string(op) + " act 'silu_limit' requires limit > 0.");
  }
  if (act == "swiglu_oai" && (!(limit > 0.0f) || !(alpha > 0.0f))) {
    throw std::invalid_argument(
        std::string(op) +
        " act 'swiglu_oai' requires limit > 0 and alpha > 0.");
  }
  int K = x.shape(1);
  check_kq_expert_stack(op, gate_w, kquant_type, K);
  check_kq_expert_stack(op, up_w, up_type, K);
  if (gate_w.shape(0) != up_w.shape(0) || gate_w.shape(1) != up_w.shape(1)) {
    throw std::invalid_argument(
        std::string(op) + " gate/up expert shapes must match.");
  }
  int N = gate_w.shape(1);
  const int ns = check_kq_shexp_rows(op, shexp_gate_w, shexp_type, K, N);
  if (check_kq_shexp_rows(op, shexp_up_w, shexp_up_type, K, N) != ns) {
    throw std::invalid_argument(
        std::string(op) + " shared-expert gate/up widths must match.");
  }
  // The dense route takes a shared expert in one codec. Its products carry
  // the shared scales, so the fused op gets neither the scales nor their bit.
  const int shexp_bit = moe_shexp_scale_bit(
      "moe_glu_gather_shexp_kq",
      shexp_type,
      shexp_gate_scale,
      "shexp_gate_scale",
      &shexp_up_scale,
      "shexp_up_scale");
  const int route_bits = moe_scale_bits(
                             "moe_glu_gather_shexp_kq",
                             kquant_type,
                             N,
                             gate_w.shape(0),
                             gate_scale,
                             "gate_scale",
                             &up_scale,
                             "up_scale") |
      (act == "swiglu_oai" ? KQ_MOE_SWIGLU_OAI : 0);
  const bool dense = upcast_pair &&
      kq_moe_shexp_dense(x.shape(0),
                         kq_glu_stem(kquant_type, up_type),
                         shexp_type,
                         gate_w.shape(1),
                         x.shape(-1));
  const int os_bits = route_bits | (dense ? KQ_MOE_SHEXP_PRE : shexp_bit);

  auto x_c = moe_dense_x(x, s);
  std::vector<mx::array> pre;
  if (dense) {
    std::optional<std::vector<std::optional<mx::array>>> pre_scales;
    if (shexp_bit) {
      pre_scales = std::vector<std::optional<mx::array>>{
          shexp_gate_scale, shexp_up_scale};
    }
    pre = quantized_matmul_multi(
        x_c, {shexp_gate_w, shexp_up_w}, shexp_type, pre_scales, s);
  }
  mx::Shape out_shape = {x.shape(0), indices.shape(1) + ns, N};
  kq_check_weight_base_at_build(
      gate_w, kquant_type, s, "moe_glu_gather_shexp_kq");
  kq_check_weight_base_at_build(up_w, up_type, s, "moe_glu_gather_shexp_kq");
  kq_check_weight_base_at_build(
      shexp_gate_w, shexp_type, s, "moe_glu_gather_shexp_kq");
  kq_check_weight_base_at_build(
      shexp_up_w, shexp_up_type, s, "moe_glu_gather_shexp_kq");
  std::vector<mx::array> op_inputs = {
      std::move(gate_w),
      std::move(up_w),
      std::move(shexp_gate_w),
      std::move(shexp_up_w),
      std::move(x_c),
      prep_indices(indices, s)};
  for (auto& p : pre) {
    op_inputs.push_back(std::move(p));
  }
  if (dense) {
    moe_push_scales(op_inputs, s, {&gate_scale, &up_scale});
  } else {
    moe_push_scales(
        op_inputs,
        s,
        {&gate_scale, &up_scale, &shexp_gate_scale, &shexp_up_scale});
  }
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantMoEGLUShexpKQ>(
          s,
          kquant_type,
          act,
          shexp_type,
          act == "silu_limit" || act == "swiglu_oai" ? limit : 0.0f,
          shexp_up_type,
          os_bits,
          act == "swiglu_oai" ? alpha : 0.0f,
          up_type),
      std::move(op_inputs));
}

mx::array gather_qmv_mix_kq(
    mx::array x,
    mx::array w,
    mx::array shexp_w,
    const std::string& kquant_type,
    mx::array indices,
    mx::array scores,
    const std::string& shexp_kquant_type,
    const std::optional<mx::array>& out_scale,
    const std::optional<mx::array>& shexp_scale,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.gather_qmv_mix_kq]";
  const std::string shexp_type =
      shexp_kquant_type.empty() ? kquant_type : shexp_kquant_type;
  if (!shexp_mix_combo_has_kernel(kquant_type, shexp_type)) {
    throw std::invalid_argument(
        std::string(op) + " no fused kernel for expert codec '" + kquant_type +
        "' with shared-expert codec '" + shexp_type + "'.");
  }
  if (x.ndim() != 3) {
    throw std::invalid_argument(std::string(op) + " x must be 3-D [T, S, K].");
  }
  int S = x.shape(1);
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16 or bfloat16.");
  }
  int K = x.shape(2);
  check_kq_expert_stack(op, w, kquant_type, K);
  // k shared slots, the last k of the S slots.
  const int ns = check_kq_shexp_down(op, shexp_w, shexp_type, K, w.shape(1));
  if (S <= ns || indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != S - ns) {
    throw std::invalid_argument(
        std::string(op) +
        " indices must be [T, S - k] for a shared expert k routed experts "
        "wide.");
  }
  if (scores.ndim() != 2 || scores.shape(0) != x.shape(0) ||
      (scores.shape(1) != S && scores.shape(1) != S - ns)) {
    throw std::invalid_argument(
        std::string(op) + " scores must be [T, S] or [T, S - k].");
  }
  const int shexp_bit = moe_shexp_scale_bit(
      "gather_qmv_mix_kq", shexp_type, shexp_scale, "shexp_scale");
  // The dense route sums the shared slots in one product, so they take one
  // weight: one slot, or the implicit 1. The product carries the shared
  // scale, so the fused op gets neither the scale nor its bit. The route's
  // stock Slice has no shape rule, so a shapeless compiled function cannot
  // hold the route.
  const int T = x.shape(0);
  const int route_bits = moe_scale_bits(
      "gather_qmv_mix_kq",
      kquant_type,
      w.shape(1),
      w.shape(0),
      out_scale,
      "out_scale");
  // The tiled down runs the shared slots as tiles of its own, unless
  // KQ_MOE_SHEXP_DENSE asks for the route.
  const bool dense = (ns == 1 || scores.shape(1) == S - ns) &&
      kq_moe_shexp_dense(T, kquant_type, shexp_type, w.shape(1), x.shape(-1)) &&
      (kq_moe_shexp_dense_forced() ||
       !kq_moe_tile_mix(
           T,
           kquant_type,
           shexp_type,
           w.shape(1),
           x.shape(-1),
           route_bits | shexp_bit));
  const int os_bits = route_bits | (dense ? KQ_MOE_SHEXP_PRE : shexp_bit);

  auto x_c = moe_dense_x(x, s);
  std::optional<mx::array> pre;
  if (dense) {
    auto hs = mx::reshape(
        mx::slice(x_c, {0, S - ns, 0}, {T, S, K}, s), {T, ns * K}, s);
    std::optional<std::vector<std::optional<mx::array>>> pre_scales;
    if (shexp_bit) {
      pre_scales = std::vector<std::optional<mx::array>>{shexp_scale};
    }
    pre = quantized_matmul_multi(hs, {shexp_w}, shexp_type, pre_scales, s)[0];
  }
  mx::Shape out_shape = {x.shape(0), w.shape(1)};
  kq_check_weight_base_at_build(w, kquant_type, s, "gather_qmv_mix_kq");
  kq_check_weight_base_at_build(shexp_w, shexp_type, s, "gather_qmv_mix_kq");
  std::vector<mx::array> op_inputs = {
      std::move(w),
      std::move(shexp_w),
      std::move(x_c),
      prep_indices(indices, s),
      prep_scores(scores, s)};
  if (pre) {
    op_inputs.push_back(std::move(*pre));
    moe_push_scales(op_inputs, s, {&out_scale});
  } else {
    moe_push_scales(op_inputs, s, {&out_scale, &shexp_scale});
  }
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVMixKQ>(
          s, kquant_type, shexp_type, os_bits),
      std::move(op_inputs));
}

mx::array gather_qmv_mix_ns_kq(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array indices,
    mx::array scores,
    const std::optional<mx::array>& lora_a,
    const std::optional<mx::array>& lora_b,
    const std::optional<mx::array>& lora_ids,
    const std::optional<mx::array>& lora_table,
    const std::optional<mx::array>& lora_rows,
    const std::optional<mx::array>& out_scale,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.gather_qmv_mix_ns_kq]";
  if (x.ndim() != 3) {
    throw std::invalid_argument(std::string(op) + " x must be 3-D [T, S, K].");
  }
  int S = x.shape(1);
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != S) {
    throw std::invalid_argument(std::string(op) + " indices must be [T, S].");
  }
  if (scores.ndim() != 2 || scores.shape(0) != x.shape(0) ||
      scores.shape(1) != S) {
    throw std::invalid_argument(std::string(op) + " scores must be [T, S].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16 or bfloat16.");
  }
  int K = x.shape(2);
  check_kq_expert_stack(op, w, kquant_type, K);
  kq_check_weight_base_at_build(w, kquant_type, s, "gather_qmv_mix_ns_kq");
  const int os_bits = moe_scale_bits(
      "gather_qmv_mix_ns_kq",
      kquant_type,
      w.shape(1),
      w.shape(0),
      out_scale,
      "out_scale");

  auto x_c = moe_dense_x(x, s);
  const int E = w.shape(0);
  const int N = w.shape(1);
  mx::Shape out_shape = {x.shape(0), N};
  std::vector<mx::array> op_inputs = {
      std::move(w),
      std::move(x_c),
      prep_indices(indices, s),
      prep_scores(scores, s)};
  const std::optional<mx::array> ids_or =
      (lora_a.has_value() && !lora_ids.has_value())
      ? std::optional<mx::array>(indices)
      : lora_ids;
  const int lora_flags = kq_lora_prep(
      op,
      lora_a,
      lora_b,
      ids_or,
      lora_table,
      lora_rows,
      dt,
      K,
      N,
      E,
      indices.shape(),
      S,
      s,
      op_inputs);
  moe_push_scales(op_inputs, s, {&out_scale});
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVMixNSKQ>(
          s, kquant_type, lora_flags, os_bits),
      std::move(op_inputs));
}

std::vector<mx::array> moe_router_topk(
    mx::array logits,
    int top_k,
    bool norm_topk_prob,
    bool shared_gate,
    const std::optional<mx::array>& per_expert_scale,
    const std::optional<mx::array>& bias,
    const std::string& scoring,
    float scale,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_router_topk]";
  const int shared = shared_gate ? 1 : 0;
  int scoring_i;
  if (scoring == "softmax") {
    scoring_i = 0;
  } else if (scoring == "sqrtsoftplus") {
    scoring_i = 1;
  } else if (scoring == "sigmoid") {
    scoring_i = 2;
  } else {
    throw std::invalid_argument(
        std::string(op) + " scoring must be softmax, sqrtsoftplus or sigmoid.");
  }
  if (scoring_i == 1 && !norm_topk_prob) {
    throw std::invalid_argument(
        std::string(op) + " sqrtsoftplus requires norm_topk_prob.");
  }
  if (logits.ndim() != 2) {
    throw std::invalid_argument(
        std::string(op) + " logits must be 2-D [T, E + shared_gate].");
  }
  auto dt = logits.dtype();
  if (dt != mx::float32 && dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " logits must be float32/float16/bfloat16.");
  }
  int E = logits.shape(1) - shared;
  if (top_k < 1 || top_k > 16 || top_k > E) {
    throw std::invalid_argument(
        std::string(op) + " top_k must be in [1, min(E, 16)].");
  }
  if (E > 1024) {
    throw std::invalid_argument(std::string(op) + " requires E <= 1024.");
  }

  auto l_c = moe_dense_x(logits, s);
  int T = l_c.shape(0);
  std::vector<mx::array> inputs = {std::move(l_c)};
  if (per_expert_scale.has_value()) {
    const auto& pes = *per_expert_scale;
    if (pes.ndim() != 1 || pes.shape(0) != E) {
      throw std::invalid_argument(
          std::string(op) + " per_expert_scale must be 1-D [E].");
    }
    inputs.push_back(prep_scores(pes, s));
  }
  if (bias.has_value()) {
    const auto& b = *bias;
    if (b.ndim() != 1 || b.shape(0) != E) {
      throw std::invalid_argument(std::string(op) + " bias must be 1-D [E].");
    }
    inputs.push_back(prep_scores(b, s));
  }
  return mx::array::make_arrays(
      {{T, top_k}, {T, top_k + shared}},
      {mx::uint32, mx::float32},
      std::make_shared<KQuantMoERouterTopK>(
          s,
          top_k,
          norm_topk_prob,
          shared_gate,
          per_expert_scale.has_value(),
          bias.has_value(),
          scoring_i,
          scale),
      std::move(inputs));
}

// ---------------------------------------------------------------------------
// Expert store (kq_moe_store.h).
// ---------------------------------------------------------------------------
namespace {

// A plane of a codec's store: R bytes a record, from byte off of a block,
// gpb records a block.
struct KqStorePlane {
  int R;
  int off;
  int gpb;
};
struct KqStoreSpec {
  int wpb; // weights a block
  int bb; // block bytes
  std::vector<KqStorePlane> planes;
};

const KqStoreSpec* kq_store_spec(const std::string& t) {
  static const std::vector<std::pair<std::string, KqStoreSpec>> specs = {
      {"q5_0", {32, 22, {{16, 6, 1}, {4, 2, 1}, {2, 0, 1}}}},
      {"iq4_nl", {32, 18, {{16, 2, 1}, {2, 0, 1}}}},
      {"iq3_xxs", {256, 98, {{8, 2, 8}, {4, 66, 8}, {2, 0, 1}}}},
      {"iq4_xs", {256, 136, {{16, 8, 8}, {8, 0, 1}}}},
  };
  for (const auto& kv : specs) {
    if (kv.first == t) {
      return &kv.second;
    }
  }
  // The generic format: one plane of whole blocks, for every other codec
  // the tiled kernels read (kq_moe_store.metal lists the same names).
  static const std::vector<std::string> generic = {
      "iq2_s", "q2_0",  "q2_k",   "q3_k",  "q4_k",   "q5_k",    "q6_k",
      "q8_0",  "q4_0",  "q4_1",   "q5_1",  "iq3_s",  "iq2_xxs", "iq2_xs",
      "iq1_s", "iq1_m", "stq1_0", "pq2_0", "ptq1_0", "mxfp4"};
  static std::vector<std::pair<std::string, KqStoreSpec>> made;
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  for (const auto& kv : made) {
    if (kv.first == t) {
      return &kv.second;
    }
  }
  if (std::find(generic.begin(), generic.end(), t) == generic.end()) {
    return nullptr;
  }
  const KQuantCodec* c = codec_by_name(t);
  if (c == nullptr || c->weights_per_block % 32 != 0) {
    return nullptr;
  }
  if (made.capacity() < generic.size()) {
    made.reserve(generic.size()); // the returned pointers stay valid
  }
  made.push_back(
      {t,
       {c->weights_per_block,
        c->bytes_per_block,
        {{c->bytes_per_block, 0, 1}}}});
  return &made.back().second;
}

// Form of the store GLU and down (KQ_MOE_STORE_FORM, read per call): 1, the
// default, the pair-grid kernels (kq_st_glu_pg, kq_st_down_pg: no plan
// dispatch) up to KQ_MOE_STORE_PG_MAX_T rows (default 8, the widths they
// were measured at; they search the indices per threadgroup); 0, and wider
// calls, the tile kernels behind a plan dispatch (kq_st_glu, kq_st_down).
// The down's bits are those of form 0 when no K split is on (the default).
inline bool kq_store_pg(int T, const std::string& t) {
  // The pair-grid kernels exist for the formats with a plane a block field.
  if (!(t == "q5_0" || t == "iq4_nl" || t == "iq3_xxs" || t == "iq4_xs")) {
    return false;
  }
  return kq_env_lever_from("KQ_MOE_STORE_FORM", 1, 0) != 0 &&
      T <= kq_env_lever_from("KQ_MOE_STORE_PG_MAX_T", 8, 1);
}

// K split of the pair-grid kernels: simdgroups a routed expert's weight row
// and the shared expert's are split over, 1 to 4 (kq_moe_store.h). Levers
// KQ_MOE_STORE_GLU_QR, _GLU_QH, _DOWN_QR, _DOWN_QH (read per call). On M5 Max
// (Qwen3.8-Flash-Next, 12 store layers, 2 to 8 rows): the GLU's shared expert
// in q6_k over K 2560 sets the dispatch's time unless split.
inline int kq_store_q(const char* name, int unset) {
  const int v = kq_env_lever_from(name, unset, 1);
  return v > 4 ? 4 : v;
}

// Widths the store kernels take, by default all of them: a store may be
// the only copy of its experts. Where the GGUF bytes stay mapped, a caller
// can narrow the range with KQ_MOE_STORE_GLU_MIN_T, KQ_MOE_STORE_DOWN_MIN_T
// and KQ_MOE_STORE_MAX_T (read when the op is built): a call outside it is
// built as the in-place op on the same weights (moe_glu_gather_shexp_kq,
// gather_qmv_mix_kq). On M5 Max, Qwen3.8-Flash-Next, the in-place GLU is ahead
// of the store below 4 rows and at 6, the in-place down at one row.
constexpr int KQ_STORE_GLU_MIN_T = 1;
constexpr int KQ_STORE_DOWN_MIN_T = 1;
constexpr int KQ_STORE_MAX_T = 1 << 30;
inline bool kq_store_width(int T, const char* min_name, int min_unset) {
  return T >= kq_env_lever_from(min_name, min_unset, 1) &&
      T <= kq_env_lever_from("KQ_MOE_STORE_MAX_T", KQ_STORE_MAX_T, 1);
}

// The rows a column tile of a pair-grid kernel may have (kq_moe_store.h),
// lever KQ_MOE_STORE_COL_ROWS (1 to 4), and the kernel variant by the
// simdgroups of a threadgroup, which size its threadgroup memory.
inline int kq_store_col_rows() {
  const int v = kq_env_lever_from("KQ_MOE_STORE_COL_ROWS", 2, 1);
  return v > 4 ? 4 : v;
}

// The shared expert of a pair-grid kernel on its codec's qf row loops
// (kq_moe_qf.h; 32 lanes share a row's walk of K, one activation row a
// pass) up to KQ_MOE_STORE_SHEXP_QF rows; 0, the default, off: on M5 Max
// (Qwen3.8-Flash-Next GLU, q6_k shared) the loops ran 1.2 to 2.2 times the
// shared tile's time from 2 rows. Wider calls and the other shared codecs run
// the shared slot as tiles. The loops need K in whole blocks of 256 and whole
// 32-column threadgroups.
inline bool kq_store_shqf(
    int T,
    const std::string& t,
    const std::string& st,
    int N,
    int K) {
  if (st == t || !(st == "q5_k" || st == "q6_k" || st == "q8_0") ||
      K % 256 != 0 || N % 64 != 0) {
    return false;
  }
  return T <= kq_env_lever_from("KQ_MOE_STORE_SHEXP_QF", 0, 0);
}

// The in-place pair form on the planes (kq_st_glu_dd_iq3, kq_moe_store.h)
// up to KQ_MOE_STORE_GLU_DD rows (default 3; at 4 the tile form is faster
// and the in-place op's shared slots come from its dense route; 0 off): iq3_xxs
// experts with a shared expert in the same codec, q5_k, q6_k or q8_0, at 16
// lanes along K and two simdgroups.
inline bool
kq_store_dd(int T, const std::string& t, const std::string& st, int N, int K) {
  if (t != "iq3_xxs" ||
      !(st == t || st == "q5_k" || st == "q6_k" || st == "q8_0") ||
      K % 256 != 0 || K < 512 || N % 64 != 0) {
    return false;
  }
  return T <= kq_env_lever_from("KQ_MOE_STORE_GLU_DD", 3, 0);
}

inline std::string kq_store_mem(int simdgroups) {
  const int sg = simdgroups <= 2 ? 2 : (simdgroups <= 4 ? 4 : 8);
  return "_g" + std::to_string(sg) + "_";
}

bool kq_store_shexp_ok(const std::string& t, const std::string& st) {
  if (st == "q5_k" || st == "q6_k" || st == "q8_0") {
    return true;
  }
  return st == t;
}

const KqStoreSpec& kq_store_check(
    const char* op,
    const mx::array& w,
    const std::string& t,
    int K) {
  const KqStoreSpec* spec = kq_store_spec(t);
  if (spec == nullptr) {
    throw std::invalid_argument(
        std::string(op) + " no store format for codec '" + t + "'.");
  }
  if (w.ndim() != 3 || w.dtype() != mx::uint8 || K % spec->wpb != 0 ||
      K % 32 != 0 || w.shape(1) % 64 != 0 ||
      w.shape(2) != K / spec->wpb * spec->bb) {
    throw std::invalid_argument(
        std::string(op) +
        " w must be uint8 [E, N, row bytes] of the codec, N a multiple of "
        "64.");
  }
  // The kernels index rows of the stack from its start. An array without
  // memory is checked at eval (kq_store_rows).
  if (kq_has_layout(w) && !w.flags().row_contiguous) {
    throw std::invalid_argument(
        std::string(op) + " weights must be row-contiguous.");
  }
  return *spec;
}

// A call the width levers send to the in-place op reads the expert stack:
// a one-expert stub in place of a stack of every expert has no fallback.
void kq_store_stub_check(const char* op, int E, const mx::array& slots) {
  if (E == 1 && slots.shape(0) != 1) {
    throw std::invalid_argument(
        std::string(op) +
        " a one-expert stub stack has no in-place fallback, which the width "
        "levers (KQ_MOE_STORE_GLU_MIN_T, KQ_MOE_STORE_DOWN_MIN_T, "
        "KQ_MOE_STORE_MAX_T) pick at this row count.");
  }
}

// Eval guard of the operands a store kernel indexes from their start and
// that the op does not copy: expert stacks and planes.
void kq_store_rows(
    const char* op,
    const std::vector<mx::array>& inputs,
    size_t first,
    size_t count) {
  for (size_t i = first; i < first + count && i < inputs.size(); ++i) {
    if (!inputs[i].flags().row_contiguous) {
      kq_check_row_major(
          inputs[i], std::string("[mlx_kquant.") + op + "] weights and planes");
    }
  }
}

// The slot table as the kernels read it: row-major, copied when a view
// (table[:, layer]) has other strides. Unevaluated, it goes in as it is
// and the eval copies it (kq_row_major_gpu).
mx::array kq_store_slots(const mx::array& slots, mx::StreamOrDevice s) {
  if (kq_has_layout(slots) && !slots.flags().row_contiguous) {
    return mx::contiguous(slots, false, s);
  }
  return slots;
}

void kq_store_check_planes(
    const char* op,
    const KqStoreSpec& spec,
    const std::vector<mx::array>& planes,
    int N,
    int K,
    const mx::array& slots,
    int E) {
  if (planes.size() != spec.planes.size()) {
    throw std::invalid_argument(
        std::string(op) + " wrong number of planes for the codec.");
  }
  // A stack of one expert is a stub: every routed expert then has a slot
  // and the stack is never read.
  if (slots.ndim() != 1 || (slots.shape(0) != E && E != 1) ||
      slots.dtype() != mx::uint32) {
    throw std::invalid_argument(std::string(op) + " slots must be uint32 [E].");
  }
  for (size_t i = 0; i < planes.size(); ++i) {
    const size_t per = static_cast<size_t>(N) * (K / spec.wpb) *
        spec.planes[i].gpb * spec.planes[i].R;
    if (planes[i].dtype() != mx::uint8 || planes[i].ndim() != 1 ||
        planes[i].size() % per != 0 || planes[i].size() == 0) {
      throw std::invalid_argument(
          std::string(op) + " a plane is not whole slots of the codec.");
    }
    if (kq_has_layout(planes[i]) && !planes[i].flags().row_contiguous) {
      throw std::invalid_argument(
          std::string(op) + " planes must be row-contiguous.");
    }
  }
}

} // namespace

bool moe_store_supported(
    const std::string& kquant_type,
    const std::string& shexp_kquant_type) {
#ifdef _METAL_
  const std::string st =
      shexp_kquant_type.empty() ? kquant_type : shexp_kquant_type;
  return mx::metal::is_available() && kq_is_nax_available() &&
      kq_store_spec(kquant_type) != nullptr &&
      kq_store_shexp_ok(kquant_type, st);
#else
  (void)kquant_type;
  (void)shexp_kquant_type;
  return false;
#endif
}

bool KQuantMoeStore::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoeStore&>(other);
  return op_ == o.op_ && kquant_type_ == o.kquant_type_ &&
      shexp_type_ == o.shexp_type_ && K_ == o.K_ && N_ == o.N_ &&
      tile_rows_ == o.tile_rows_;
}

void KQuantMoeStore::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.moe_store] requires the Metal GPU.");
}

// The codecs with a pair kernel (kq_moe_store_seg.metal).
static bool moe_store_seg_paired(const std::string& codec) {
  return codec == "iq3_xxs" || codec == "iq4_xs" || codec == "iq2_s";
}

#ifdef _METAL_

// KQ_SEG_DR=N (default 16, up to 32, 0 off): 64-row tiles of at most N rows
// of the wide store kernels (field-plane codecs) decode their weights into
// the tensor-op fragments instead of a threadgroup slab. Same bits.
// iq3_xxs stays on the slab: a lane decodes 4 weights at a time there, so
// the sign and grid word work the slab shares over 8 runs twice.
static int kq_seg_dr_rows(const std::string& codec) {
  return codec == "iq3_xxs"
      ? 0
      : std::min(32, kq_env_lever_from("KQ_SEG_DR", 16, 0));
}

void KQuantMoeStore::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& ce = mx::metal::get_command_encoder(s);
  const KqStoreSpec& spec = *kq_store_spec(kquant_type_);
  const int np = static_cast<int>(spec.planes.size());
  if (op_ == 0) {
    kq_store_rows("moe_store_pack", inputs, 0, 1);
    const auto& w = inputs[0];
    const auto& experts = inputs[1];
    const int N = w.shape(1);
    const int count = experts.shape(0);
    const std::string kname = "kq_moe_store_pack";
    auto kernel = kq_get_kernel(d, kname);
    for (int i = 0; i < np; ++i) {
      auto& out = outputs[i];
      out.set_data(mx::allocator::malloc(out.nbytes()));
      const KqStorePlane& pl = spec.planes[i];
      const int U = K_ / spec.wpb * pl.gpb;
      ce.set_compute_pipeline_state(kernel);
      ce.set_input_array(w, 0);
      ce.set_output_array(out, 1);
      ce.set_input_array(experts, 2);
      ce.set_bytes(N, 3);
      ce.set_bytes(U, 4);
      ce.set_bytes(pl.R, 5);
      ce.set_bytes(spec.bb, 6);
      ce.set_bytes(pl.off, 7);
      ce.set_bytes(pl.gpb, 8);
      const int rowb = w.shape(2);
      ce.set_bytes(rowb, 9);
      const int s0 = 0;
      ce.set_bytes(s0, 10);
      MTL::Size group(64, 1, 1);
      MTL::Size grid(N, U, count);
      kq_log_dispatch(kname, w, out, grid, group);
      ce.dispatch_threads(grid, group);
    }
    return;
  }
  if (op_ == 4 || op_ == 5) {
    for (auto& o : outputs) {
      o.set_data(mx::allocator::malloc(o.nbytes()));
    }
    // x, map, counts, gate planes, up planes, then the row map when the call
    // has one.
    kq_store_rows("moe_store_seg_pair", inputs, 3, 2 * np);
    const mx::array x = kq_x_aligned_gpu(inputs[0], s, "moe_store_seg_pair x");
    const auto& map = inputs[1];
    const auto& counts = inputs[2];
    const bool xmap = static_cast<int>(inputs.size()) == 3 + 2 * np + 1;
    const int K = K_;
    const int N = N_;
    const bool align_N = true;
    const bool align_K = true;
    const int gx = kq_env_lever_from("KQ_SEG_GX", 32, 0);
    // KQ_SEG_COL_ROWS (0 off, 1 to 4): tiles of at most that many rows run
    // the kernel's column path (split-plane codecs).
    const int colr = std::min(4, kq_env_lever_from("KQ_SEG_COL_ROWS", 0, 0));
    const int drows = kq_seg_dr_rows(kquant_type_);
    const std::string kname = "kq_store_seg_pair_" + kquant_type_ + "_" +
        kq_type_string(x.dtype()) + "_bm_" + std::to_string(tile_rows_) +
        (op_ == 5 ? "_o1" : "_o0");
    mx::metal::MTLFCList func_consts = {
        {&align_N, MTL::DataType::DataTypeBool, 201},
        {&align_K, MTL::DataType::DataTypeBool, 202},
        {&gx, MTL::DataType::DataTypeInt, 203},
        {&xmap, MTL::DataType::DataTypeBool, 204},
        {&colr, MTL::DataType::DataTypeInt, 205},
        {&drows, MTL::DataType::DataTypeInt, 206},
    };
    const std::string hash_name = kname + "_gx_" + std::to_string(gx) +
        (xmap ? "_xm" : "") + (colr > 0 ? "_cr" + std::to_string(colr) : "") +
        (drows > 0 ? "_dr" + std::to_string(drows) : "");
    auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
    ce.set_compute_pipeline_state(kernel);
    const int g = 3;
    const int up = 3 + np;
    ce.set_input_array(x, 0);
    ce.set_input_array(inputs[g], 1);
    ce.set_input_array(inputs[g + std::min(1, np - 1)], 2);
    ce.set_input_array(map, 3);
    ce.set_input_array(counts, 4);
    ce.set_output_array(outputs[0], 5);
    ce.set_bytes(N, 6);
    ce.set_bytes(K, 7);
    if (xmap) {
      ce.set_input_array(inputs[3 + 2 * np], 8);
    }
    ce.set_input_array(inputs[g + std::min(2, np - 1)], 9);
    ce.set_input_array(inputs[up], 10);
    ce.set_input_array(inputs[up + std::min(1, np - 1)], 11);
    ce.set_input_array(inputs[up + std::min(2, np - 1)], 12);
    ce.set_output_array(outputs[op_ == 4 ? 1 : 0], 13);
    MTL::Size group_dims(32, 2, tile_rows_ / 32);
    const int n_cols = N / 64;
    const int cap = map.shape(0);
    MTL::Size grid_dims = gx > 0
        ? MTL::Size(gx, ((cap + gx - 1) / gx) * n_cols, 1)
        : MTL::Size(n_cols, cap, 1);
    kq_log_dispatch(kname, x, outputs[0], grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);
    return;
  }
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (op_ == 3) {
    // x, map, counts, planes, then the row map when the call has one.
    kq_store_rows("moe_store_seg", inputs, 3, np);
    const mx::array x = kq_x_aligned_gpu(inputs[0], s, "moe_store_seg x");
    const auto& map = inputs[1];
    const auto& counts = inputs[2];
    const bool xmap = static_cast<int>(inputs.size()) == 3 + np + 1;
    const int K = K_;
    const int N = N_;
    const bool align_N = true;
    const bool align_K = true;
    // KQ_SEG_GX: gather_qmm_seg's grid lever, read on every call.
    const int gx = kq_env_lever_from("KQ_SEG_GX", 32, 0);
    const std::string kname = "kq_store_seg_" + kquant_type_ + "_" +
        kq_type_string(x.dtype()) + "_bm_" + std::to_string(tile_rows_);
    // KQ_SEG_COL_ROWS (0 off, 1 to 4): tiles of at most that many rows run
    // the kernel's column path (split-plane codecs).
    const int colr = std::min(4, kq_env_lever_from("KQ_SEG_COL_ROWS", 0, 0));
    const int drows = kq_seg_dr_rows(kquant_type_);
    mx::metal::MTLFCList func_consts = {
        {&align_N, MTL::DataType::DataTypeBool, 201},
        {&align_K, MTL::DataType::DataTypeBool, 202},
        {&gx, MTL::DataType::DataTypeInt, 203},
        {&xmap, MTL::DataType::DataTypeBool, 204},
        {&colr, MTL::DataType::DataTypeInt, 205},
        {&drows, MTL::DataType::DataTypeInt, 206},
    };
    const std::string hash_name = kname + "_gx_" + std::to_string(gx) +
        (xmap ? "_xm" : "") + (colr > 0 ? "_cr" + std::to_string(colr) : "") +
        (drows > 0 ? "_dr" + std::to_string(drows) : "");
    auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(x, 0);
    ce.set_input_array(inputs[3], 1);
    ce.set_input_array(inputs[3 + std::min(1, np - 1)], 2);
    ce.set_input_array(map, 3);
    ce.set_input_array(counts, 4);
    ce.set_output_array(out, 5);
    ce.set_bytes(N, 6);
    ce.set_bytes(K, 7);
    if (xmap) {
      ce.set_input_array(inputs[3 + np], 8);
    }
    ce.set_input_array(inputs[3 + std::min(2, np - 1)], 9);
    MTL::Size group_dims(32, 2, tile_rows_ / 32);
    const int n_cols = N / 64;
    const int cap = map.shape(0);
    MTL::Size grid_dims = gx > 0
        ? MTL::Size(gx, ((cap + gx - 1) / gx) * n_cols, 1)
        : MTL::Size(n_cols, cap, 1);
    kq_log_dispatch(kname, x, out, grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);
    return;
  }
  const std::string st = shexp_type_;
  const int COLR = kq_env_lever_from("KQ_MOE_TILE_COL_ROWS", 2, 0);
  if (op_ == 1) {
    // gw, uw, sgw, suw, x, indices, slots, gate planes, up planes.
    kq_store_rows("moe_store_glu", inputs, 0, 4);
    kq_store_rows("moe_store_glu", inputs, 7, 2 * np);
    const auto& gw = inputs[0];
    kq_check_weight_base(gw, kquant_type_, "moe_store_glu");
    kq_check_weight_base(inputs[1], kquant_type_, "moe_store_glu");
    kq_check_weight_base(inputs[2], st, "moe_store_glu");
    kq_check_weight_base(inputs[3], st, "moe_store_glu");
    const mx::array x = kq_x_aligned_gpu(inputs[4], s, "moe_store_glu x");
    const auto& indices = inputs[5];
    const mx::array slots = kq_row_major_gpu(inputs[6], s);
    const int T = x.shape(0);
    const int K = x.shape(1);
    const int N = gw.shape(1);
    const int S = indices.shape(1) + 1;
    const int NS = 1;
    const int pairs = T * S;
    const std::string type = kq_type_string(x.dtype());
    if (kq_store_pg(T, kquant_type_) &&
        kq_store_dd(T, kquant_type_, st, N, K)) {
      // The in-place pair form on the planes (kq_st_glu_dd_iq3): the
      // in-place op's lanes, simdgroups and sums.
      const std::string kname = "kq_store_glu_dd_iq3_xxs_sx_" + st +
          (kq_moe_c32(kquant_type_, T) ? "_c32_" : "_") + type;
      auto kernel = kq_get_kernel(d, kname);
      ce.set_compute_pipeline_state(kernel);
      for (int i = 0; i < 4; ++i) {
        ce.set_input_array(inputs[i], i);
      }
      ce.set_input_array(x, 4);
      ce.set_input_array(indices, 5);
      ce.set_output_array(out, 6);
      ce.set_bytes(K, 7);
      ce.set_bytes(N, 8);
      ce.set_input_array(slots, 11);
      for (int i = 0; i < 3; ++i) {
        ce.set_input_array(inputs[7 + std::min(i, np - 1)], 12 + i);
        ce.set_input_array(inputs[7 + np + std::min(i, np - 1)], 15 + i);
      }
      MTL::Size group_dims(32, 2, 1);
      MTL::Size grid_dims(N / 4, S, T);
      kq_log_dispatch(kname, x, out, grid_dims, group_dims);
      ce.dispatch_threadgroups(grid_dims, group_dims);
      return;
    }
    if (kq_store_pg(T, kquant_type_)) {
      const int QR = kq_store_q("KQ_MOE_STORE_GLU_QR", 2);
      const int QH = kq_store_q("KQ_MOE_STORE_GLU_QH", 2);
      const int SHQF = kq_store_shqf(T, kquant_type_, st, N, K) ? 1 : 0;
      // The qf loops take whatever simdgroups the routed tiles have.
      const int Q = SHQF != 0 ? QR : std::max(QR, QH);
      const int PCOLR = kq_store_col_rows();
      const std::string kname = "kq_store_glu_pg_" + kquant_type_ + "_sx_" +
          st + kq_store_mem(2 * Q) + type;
      auto kernel = kq_get_kernel(d, kname);
      ce.set_compute_pipeline_state(kernel);
      for (int i = 0; i < 4; ++i) {
        ce.set_input_array(inputs[i], i);
      }
      ce.set_input_array(x, 4);
      ce.set_input_array(indices, 5);
      ce.set_output_array(out, 6);
      ce.set_bytes(K, 7);
      ce.set_bytes(N, 8);
      ce.set_bytes(QR, 9);
      ce.set_bytes(PCOLR, 10);
      ce.set_input_array(slots, 11);
      for (int i = 0; i < 3; ++i) {
        ce.set_input_array(inputs[7 + std::min(i, np - 1)], 12 + i);
        ce.set_input_array(inputs[7 + np + std::min(i, np - 1)], 15 + i);
      }
      ce.set_bytes(QH, 18);
      ce.set_bytes(S, 19);
      ce.set_bytes(SHQF, 20);
      MTL::Size group_dims(32, 2 * Q, 1);
      MTL::Size grid_dims(T * S, (N + 31) / 32, 1);
      kq_log_dispatch(kname, x, out, grid_dims, group_dims);
      ce.dispatch_threadgroups(grid_dims, group_dims);
      return;
    }
    mx::array hg({pairs + 7, K}, x.dtype(), nullptr, {});
    mx::array tiles({pairs + 8, 4}, mx::uint32, nullptr, {});
    mx::array place({T * (S - NS) + 1}, mx::uint32, nullptr, {});
    hg.set_data(mx::allocator::malloc(hg.nbytes()));
    tiles.set_data(mx::allocator::malloc(tiles.nbytes()));
    place.set_data(mx::allocator::malloc(place.nbytes()));
    const std::string pname = "kq_moe_store_plan_" + type;
    auto pk = kq_get_kernel(d, pname);
    ce.set_compute_pipeline_state(pk);
    ce.set_input_array(indices, 0);
    ce.set_input_array(x, 1);
    ce.set_output_array(hg, 2);
    ce.set_output_array(tiles, 3);
    ce.set_output_array(place, 4);
    ce.set_bytes(K, 5);
    ce.set_bytes(S, 6);
    ce.set_bytes(NS, 7);
    ce.set_bytes(T, 8);
    const int xr = 1;
    ce.set_bytes(xr, 9);
    MTL::Size pgroup(32, 1, 1);
    MTL::Size pgrid(pairs, 1, 1);
    kq_log_dispatch(pname, x, hg, pgrid, pgroup);
    ce.dispatch_threadgroups(pgrid, pgroup);

    const std::string kname =
        "kq_store_glu_" + kquant_type_ + "_sx_" + st + "_" + type;
    auto kernel = kq_get_kernel(d, kname);
    ce.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 4; ++i) {
      ce.set_input_array(inputs[i], i);
    }
    ce.set_input_array(hg, 4);
    ce.set_input_array(tiles, 5);
    ce.set_output_array(out, 6);
    ce.set_bytes(K, 7);
    ce.set_bytes(N, 8);
    ce.set_bytes(COLR, 10);
    ce.set_input_array(slots, 11);
    for (int i = 0; i < 3; ++i) {
      ce.set_input_array(inputs[7 + std::min(i, np - 1)], 12 + i);
      ce.set_input_array(inputs[7 + np + std::min(i, np - 1)], 15 + i);
    }
    MTL::Size group_dims(32, 2, 1);
    MTL::Size grid_dims((N + 31) / 32, pairs, 1);
    kq_log_dispatch(kname, x, out, grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);
    ce.add_temporary(hg);
    ce.add_temporary(tiles);
    ce.add_temporary(place);
    return;
  }
  // w, sw, x, indices, scores, slots, planes.
  kq_store_rows("moe_store_down", inputs, 0, 2);
  kq_store_rows("moe_store_down", inputs, 6, np);
  const auto& w = inputs[0];
  const auto& sw = inputs[1];
  kq_check_weight_base(w, kquant_type_, "moe_store_down");
  kq_check_weight_base(sw, st, "moe_store_down");
  const mx::array x = kq_x_aligned_gpu(inputs[2], s, "moe_store_down x");
  const auto& indices = inputs[3];
  const auto& scores = inputs[4];
  const mx::array slots = kq_row_major_gpu(inputs[5], s);
  const int T = x.shape(0);
  const int S = x.shape(1);
  const int K = x.shape(2);
  const int N = w.shape(1);
  const int NS = S - indices.shape(1);
  const int pairs = T * S;
  const std::string type = kq_type_string(x.dtype());
  if (kq_store_pg(T, kquant_type_)) {
    const int QR = kq_store_q("KQ_MOE_STORE_DOWN_QR", 1);
    const int QH = kq_store_q("KQ_MOE_STORE_DOWN_QH", 1);
    const int SHQF =
        NS == 1 && kq_store_shqf(T, kquant_type_, st, N, K) ? 1 : 0;
    const int Q = SHQF != 0 ? QR : std::max(QR, QH);
    // 64 columns a threadgroup as the tile kernel while two column
    // simdgroups fit beside the K split.
    const int CW = Q <= 2 ? 2 : 1;
    mx::array y({pairs, N}, x.dtype(), nullptr, {});
    y.set_data(mx::allocator::malloc(y.nbytes()));
    const int PCOLR = kq_store_col_rows();
    const std::string kname = "kq_store_down_pg_" + kquant_type_ + "_sx_" + st +
        kq_store_mem(CW * Q) + type;
    auto kernel = kq_get_kernel(d, kname);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(w, 0);
    ce.set_input_array(sw, 1);
    ce.set_input_array(x, 2);
    ce.set_input_array(indices, 3);
    ce.set_output_array(y, 5);
    ce.set_bytes(K, 6);
    ce.set_bytes(N, 7);
    ce.set_bytes(QR, 8);
    ce.set_bytes(NS, 9);
    ce.set_bytes(PCOLR, 10);
    ce.set_input_array(slots, 11);
    for (int i = 0; i < 3; ++i) {
      ce.set_input_array(inputs[6 + std::min(i, np - 1)], 12 + i);
    }
    ce.set_bytes(QH, 18);
    ce.set_bytes(CW, 19);
    ce.set_bytes(SHQF, 20);
    MTL::Size group_dims(32, CW * Q, 1);
    ce.set_bytes(S, 4);
    MTL::Size grid_dims(T * S, (N + 32 * CW - 1) / (32 * CW), 1);
    kq_log_dispatch(kname, x, out, grid_dims, group_dims);
    ce.dispatch_threadgroups(grid_dims, group_dims);

    const std::string cname = "kq_moe_store_combine_" + type;
    auto ck = kq_get_kernel(d, cname);
    ce.set_compute_pipeline_state(ck);
    ce.set_input_array(y, 0);
    ce.set_input_array(scores, 1);
    ce.set_output_array(out, 2);
    ce.set_bytes(N, 3);
    ce.set_bytes(S, 4);
    const int SC = scores.shape(1);
    ce.set_bytes(SC, 5);
    ce.set_bytes(NS, 6);
    MTL::Size cgroup(std::min(N / 4, 256), 1, 1);
    MTL::Size cgrid(N / 4, 1, T);
    kq_log_dispatch(cname, x, out, cgrid, cgroup);
    ce.dispatch_threads(cgrid, cgroup);
    ce.add_temporary(y);
    return;
  }
  mx::array hg({pairs + 7, K}, x.dtype(), nullptr, {});
  mx::array tiles({pairs, 4}, mx::uint32, nullptr, {});
  mx::array place({T * (S - NS) + 1}, mx::uint32, nullptr, {});
  mx::array y({pairs, N}, x.dtype(), nullptr, {});
  hg.set_data(mx::allocator::malloc(hg.nbytes()));
  tiles.set_data(mx::allocator::malloc(tiles.nbytes()));
  place.set_data(mx::allocator::malloc(place.nbytes()));
  y.set_data(mx::allocator::malloc(y.nbytes()));
  const std::string pname = "kq_moe_tile_plan_" + type;
  auto pk = kq_get_kernel(d, pname);
  ce.set_compute_pipeline_state(pk);
  ce.set_input_array(indices, 0);
  ce.set_input_array(x, 1);
  ce.set_output_array(hg, 2);
  ce.set_output_array(tiles, 3);
  ce.set_output_array(place, 4);
  ce.set_bytes(K, 5);
  ce.set_bytes(S, 6);
  ce.set_bytes(NS, 7);
  ce.set_bytes(T, 8);
  MTL::Size pgroup(32, 1, 1);
  MTL::Size pgrid(pairs, 1, 1);
  kq_log_dispatch(pname, x, hg, pgrid, pgroup);
  ce.dispatch_threadgroups(pgrid, pgroup);

  const std::string kname =
      "kq_store_down_" + kquant_type_ + "_sx_" + st + "_" + type;
  auto kernel = kq_get_kernel(d, kname);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(sw, 1);
  ce.set_input_array(hg, 2);
  ce.set_input_array(tiles, 3);
  ce.set_output_array(y, 5);
  ce.set_bytes(K, 6);
  ce.set_bytes(N, 7);
  ce.set_bytes(NS, 9);
  ce.set_bytes(COLR, 10);
  ce.set_input_array(slots, 11);
  for (int i = 0; i < 3; ++i) {
    ce.set_input_array(inputs[6 + std::min(i, np - 1)], 12 + i);
  }
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims((N + 63) / 64, pairs, 1);
  kq_log_dispatch(kname, x, out, grid_dims, group_dims);
  ce.dispatch_threadgroups(grid_dims, group_dims);

  const std::string cname = "kq_moe_tile_combine_" + type;
  auto ck = kq_get_kernel(d, cname);
  ce.set_compute_pipeline_state(ck);
  ce.set_input_array(y, 0);
  ce.set_input_array(scores, 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(N, 3);
  ce.set_bytes(S, 4);
  const int SC = scores.shape(1);
  ce.set_bytes(SC, 5);
  ce.set_bytes(NS, 6);
  ce.set_input_array(place, 7);
  MTL::Size cgroup(std::min(N / 4, 256), 1, 1);
  MTL::Size cgrid(N / 4, 1, T);
  kq_log_dispatch(cname, x, out, cgrid, cgroup);
  ce.dispatch_threads(cgrid, cgroup);
  ce.add_temporary(hg);
  ce.add_temporary(tiles);
  ce.add_temporary(place);
  ce.add_temporary(y);
}

#else // !_METAL_

void KQuantMoeStore::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.moe_store] requires a Metal build.");
}

#endif

std::vector<mx::array> moe_store_pack(
    mx::array w,
    const std::string& kquant_type,
    int K,
    mx::array experts,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_store_pack]";
  const KqStoreSpec& spec = kq_store_check(op, w, kquant_type, K);
  if (!moe_store_supported(kquant_type, "")) {
    throw std::invalid_argument(
        std::string(op) + " needs the Metal GPU with tensor-op units.");
  }
  if (experts.ndim() != 1 || experts.shape(0) == 0) {
    throw std::invalid_argument(std::string(op) + " experts must be [count].");
  }
  const int N = w.shape(1);
  const int count = experts.shape(0);
  std::vector<mx::Shape> shapes;
  std::vector<mx::Dtype> dtypes;
  for (const auto& pl : spec.planes) {
    const int64_t n =
        static_cast<int64_t>(count) * N * (K / spec.wpb) * pl.gpb * pl.R;
    if (n > 0x7fffffffLL) {
      throw std::invalid_argument(
          std::string(op) + " a plane passes 2 GiB; pack fewer experts.");
    }
    shapes.push_back({static_cast<int>(n)});
    dtypes.push_back(mx::uint8);
  }
  kq_check_weight_base_at_build(w, kquant_type, s, "moe_store_pack");
  return mx::array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<KQuantMoeStore>(s, 0, kquant_type, "", K),
      {std::move(w), prep_indices(experts, s)});
}

mx::array moe_store_glu(
    mx::array x,
    mx::array gate_w,
    mx::array up_w,
    mx::array shexp_gate_w,
    mx::array shexp_up_w,
    const std::string& kquant_type,
    mx::array indices,
    mx::array slots,
    std::vector<mx::array> gate_planes,
    std::vector<mx::array> up_planes,
    const std::string& shexp_kquant_type,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_store_glu]";
  const std::string st =
      shexp_kquant_type.empty() ? kquant_type : shexp_kquant_type;
  if (!moe_store_supported(kquant_type, st)) {
    throw std::invalid_argument(
        std::string(op) + " no store kernel for '" + kquant_type +
        "' with shared-expert codec '" + st + "' on this device.");
  }
  auto dt = x.dtype();
  if (x.ndim() != 2 || (dt != mx::float16 && dt != mx::bfloat16)) {
    throw std::invalid_argument(
        std::string(op) + " x must be [T, K] float16 or bfloat16.");
  }
  const int T = x.shape(0);
  const int K = x.shape(1);
  const KqStoreSpec& spec = kq_store_check(op, gate_w, kquant_type, K);
  kq_store_check(op, up_w, kquant_type, K);
  const int N = gate_w.shape(1);
  const int E = gate_w.shape(0);
  if (up_w.shape(0) != E || up_w.shape(1) != N) {
    throw std::invalid_argument(std::string(op) + " gate and up differ.");
  }
  const KQuantCodec* sc = codec_by_name(st);
  if (sc == nullptr || shexp_gate_w.ndim() != 2 || shexp_up_w.ndim() != 2 ||
      shexp_gate_w.shape(0) != N || shexp_up_w.shape(0) != N ||
      shexp_gate_w.shape() != shexp_up_w.shape() ||
      shexp_gate_w.dtype() != mx::uint8 || shexp_up_w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) +
        " the shared gate and up must be uint8 [N, row bytes], one slot.");
  }
  // One slot: a row of K weights of the shared codec, whole blocks.
  if (!shexp_slot_on_block(st, K) ||
      shexp_gate_w.shape(1) !=
          static_cast<int64_t>(K) / sc->weights_per_block *
              sc->bytes_per_block) {
    throw std::invalid_argument(
        std::string(op) +
        " the shared gate and up rows must be K weights of '" + st +
        "', whole blocks.");
  }
  if ((kq_has_layout(shexp_gate_w) && !shexp_gate_w.flags().row_contiguous) ||
      (kq_has_layout(shexp_up_w) && !shexp_up_w.flags().row_contiguous)) {
    throw std::invalid_argument(
        std::string(op) + " weights must be row-contiguous.");
  }
  if (indices.ndim() != 2 || indices.shape(0) != T) {
    throw std::invalid_argument(std::string(op) + " indices must be [T, R].");
  }
  kq_store_check_planes(op, spec, gate_planes, N, K, slots, E);
  kq_store_check_planes(op, spec, up_planes, N, K, slots, E);
  kq_check_weight_base_at_build(gate_w, kquant_type, s, "moe_store_glu");
  kq_check_weight_base_at_build(up_w, kquant_type, s, "moe_store_glu");
  kq_check_weight_base_at_build(shexp_gate_w, st, s, "moe_store_glu");
  kq_check_weight_base_at_build(shexp_up_w, st, s, "moe_store_glu");
  if (!kq_store_width(T, "KQ_MOE_STORE_GLU_MIN_T", KQ_STORE_GLU_MIN_T)) {
    kq_store_stub_check(op, E, slots);
    return moe_glu_gather_shexp_kq(
        std::move(x),
        std::move(gate_w),
        std::move(up_w),
        std::move(shexp_gate_w),
        std::move(shexp_up_w),
        kquant_type,
        std::move(indices),
        "silu",
        shexp_kquant_type,
        0.0f,
        "",
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        0.0f,
        "",
        s);
  }
  const int R = indices.shape(1);
  std::vector<mx::array> in = {
      std::move(gate_w),
      std::move(up_w),
      std::move(shexp_gate_w),
      std::move(shexp_up_w),
      moe_dense_x(x, s),
      prep_indices(indices, s),
      kq_store_slots(slots, s)};
  for (auto& p : gate_planes) {
    in.push_back(std::move(p));
  }
  for (auto& p : up_planes) {
    in.push_back(std::move(p));
  }
  return mx::array(
      {T, R + 1, N},
      dt,
      std::make_shared<KQuantMoeStore>(s, 1, kquant_type, st, K),
      std::move(in));
}

bool moe_store_seg_supported(const std::string& kquant_type, int K) {
#ifdef _METAL_
  // The codecs kq_moe_store_seg.metal instantiates.
  static const std::vector<std::string> wide = {
      "q5_0",
      "iq4_nl",
      "iq3_xxs",
      "iq4_xs",
      "iq2_s",
      "q2_0",
      "iq3_s",
      "iq2_xxs",
      "iq2_xs",
      "iq1_s",
      "iq1_m",
      "stq1_0",
      "pq2_0",
      "ptq1_0"};
  const KqStoreSpec* spec = kq_store_spec(kquant_type);
  return moe_store_supported(kquant_type, "") && spec != nullptr && K > 0 &&
      K % 64 == 0 && K % spec->wpb == 0 &&
      std::find(wide.begin(), wide.end(), kquant_type) != wide.end() &&
      kq_seg_nax_ok(kquant_type, K);
#else
  (void)kquant_type;
  (void)K;
  return false;
#endif
}

std::vector<std::string> kernel_keys() {
#ifdef _METAL_
  return kq_kernel_keys();
#else
  return {};
#endif
}

int moe_store_warm(
    const std::string& glu_type,
    const std::string& glu_shexp_type,
    const std::string& down_type,
    const std::string& down_shexp_type,
    const std::vector<mx::Dtype>& dtypes,
    int K_glu,
    int K_down) {
#ifdef _METAL_
  // The names and constants are those of KQuantMoeStore::eval_gpu;
  // tests/test_moe_store_warm.py fails when the two part ways.
  auto& d = mx::metal::device(mx::Device::gpu);
  int n = 0;
  auto get = [&](const std::string& kname) {
    try {
      kq_get_kernel(d, kname);
      ++n;
    } catch (const std::exception&) {
    }
  };
  auto seg = [&](const std::string& codec, const std::string& type, int K) {
    if (type == "float" || !moe_store_seg_supported(codec, K)) {
      return;
    }
    const bool align_N = true;
    const bool align_K = true;
    const int gx = kq_env_lever_from("KQ_SEG_GX", 32, 0);
    const int colr = std::min(4, kq_env_lever_from("KQ_SEG_COL_ROWS", 0, 0));
    const int drows = kq_seg_dr_rows(codec);
    // The single kernel, then the pair kernel (both outputs) of a paired
    // codec.
    std::vector<std::string> knames;
    for (int tr : {64, 128}) {
      const std::string bm = "_" + type + "_bm_" + std::to_string(tr);
      knames.push_back("kq_store_seg_" + codec + bm);
      if (moe_store_seg_paired(codec)) {
        knames.push_back("kq_store_seg_pair_" + codec + bm + "_o0");
        knames.push_back("kq_store_seg_pair_" + codec + bm + "_o1");
      }
    }
    for (const std::string& kname : knames) {
      for (bool xmap : {false, true}) {
        mx::metal::MTLFCList fc = {
            {&align_N, MTL::DataType::DataTypeBool, 201},
            {&align_K, MTL::DataType::DataTypeBool, 202},
            {&gx, MTL::DataType::DataTypeInt, 203},
            {&xmap, MTL::DataType::DataTypeBool, 204},
            {&colr, MTL::DataType::DataTypeInt, 205},
            {&drows, MTL::DataType::DataTypeInt, 206},
        };
        const std::string hash = kname + "_gx_" + std::to_string(gx) +
            (xmap ? "_xm" : "") +
            (colr > 0 ? "_cr" + std::to_string(colr) : "") +
            (drows > 0 ? "_dr" + std::to_string(drows) : "");
        try {
          kq_get_kernel(d, kname, hash, fc);
          ++n;
        } catch (const std::exception&) {
        }
      }
    }
  };
  for (const mx::Dtype& dt : dtypes) {
    const std::string type = kq_type_string(dt);
    // An empty codec name skips that projection (a store of the down
    // stacks only). The pair-grid variants are the ones today's levers
    // pick; the pair form on the planes has a "_c32" kernel at one row.
    if (!glu_type.empty()) {
      if (kq_store_pg(1, glu_type)) {
        const int QR = kq_store_q("KQ_MOE_STORE_GLU_QR", 2);
        const int QH = kq_store_q("KQ_MOE_STORE_GLU_QH", 2);
        get("kq_store_glu_pg_" + glu_type + "_sx_" + glu_shexp_type +
            kq_store_mem(2 * std::max(QR, QH)) + type);
        if (glu_type == "iq3_xxs" &&
            kq_env_lever_from("KQ_MOE_STORE_GLU_DD", 3, 0) > 0) {
          const std::string dd = "kq_store_glu_dd_iq3_xxs_sx_" + glu_shexp_type;
          get(dd + "_" + type);
          get(dd + "_c32_" + type);
        }
      }
      get("kq_moe_store_plan_" + type);
      get("kq_store_glu_" + glu_type + "_sx_" + glu_shexp_type + "_" + type);
      seg(glu_type, type, K_glu);
    }
    if (!down_type.empty()) {
      if (kq_store_pg(1, down_type)) {
        const int QR = kq_store_q("KQ_MOE_STORE_DOWN_QR", 1);
        const int QH = kq_store_q("KQ_MOE_STORE_DOWN_QH", 1);
        const int Q = std::max(QR, QH);
        get("kq_store_down_pg_" + down_type + "_sx_" + down_shexp_type +
            kq_store_mem((Q <= 2 ? 2 : 1) * Q) + type);
        get("kq_moe_store_combine_" + type);
      }
      get("kq_moe_tile_plan_" + type);
      get("kq_store_down_" + down_type + "_sx_" + down_shexp_type + "_" + type);
      get("kq_moe_tile_combine_" + type);
      seg(down_type, type, K_down);
    }
  }
  return n;
#else
  (void)glu_type;
  (void)glu_shexp_type;
  (void)down_type;
  (void)down_shexp_type;
  (void)dtypes;
  (void)K_glu;
  (void)K_down;
  return 0;
#endif
}

mx::array moe_store_seg(
    mx::array x,
    std::vector<mx::array> planes,
    const std::string& kquant_type,
    int N,
    mx::array map,
    mx::array counts,
    int tile_rows,
    std::optional<mx::array> x_rows,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_store_seg]";
  auto dt = x.dtype();
  if (x.ndim() != 2 || (dt != mx::float16 && dt != mx::bfloat16)) {
    throw std::invalid_argument(
        std::string(op) + " x must be [rows, K] float16 or bfloat16.");
  }
  const int K = x.shape(1);
  if (!moe_store_seg_supported(kquant_type, K)) {
    throw std::invalid_argument(
        std::string(op) + " no wide store kernel for '" + kquant_type +
        "' at this K on this device.");
  }
  if (tile_rows != 64 && tile_rows != 128) {
    throw std::invalid_argument(
        std::string(op) + " tile_rows must be 64 or 128.");
  }
  if (N <= 0 || N % 64 != 0) {
    throw std::invalid_argument(
        std::string(op) + " N must be a positive multiple of 64.");
  }
  const KqStoreSpec& spec = *kq_store_spec(kquant_type);
  if (planes.size() != spec.planes.size()) {
    throw std::invalid_argument(
        std::string(op) + " wrong number of planes for the codec.");
  }
  for (size_t i = 0; i < planes.size(); ++i) {
    const size_t per = static_cast<size_t>(N) * (K / spec.wpb) *
        spec.planes[i].gpb * spec.planes[i].R;
    if (planes[i].dtype() != mx::uint8 || planes[i].ndim() != 1 ||
        planes[i].size() % per != 0 || planes[i].size() == 0) {
      throw std::invalid_argument(
          std::string(op) + " a plane is not whole slots of the codec.");
    }
    if (kq_has_layout(planes[i]) && !planes[i].flags().row_contiguous) {
      throw std::invalid_argument(
          std::string(op) + " planes must be row-contiguous.");
    }
  }
  if (map.ndim() != 2 || map.shape(1) != 3 || map.dtype() != mx::uint32 ||
      counts.ndim() != 1 || counts.shape(0) != 1 ||
      counts.dtype() != mx::uint32) {
    throw std::invalid_argument(
        std::string(op) + " map must be uint32 [tiles, 3], counts uint32 [1].");
  }
  if (x_rows &&
      (x_rows->ndim() != 1 || x_rows->shape(0) == 0 ||
       !mx::issubdtype(x_rows->dtype(), mx::integer))) {
    throw std::invalid_argument(
        std::string(op) + " x_rows must be a non-empty 1-D integer array.");
  }
  const int rows = x_rows ? x_rows->shape(0) : x.shape(0);
  std::vector<mx::array> in = {
      moe_dense_x(x, s),
      mx::contiguous(map, false, s),
      mx::contiguous(counts, false, s)};
  for (auto& p : planes) {
    in.push_back(std::move(p));
  }
  if (x_rows) {
    in.push_back(mx::contiguous(mx::astype(*x_rows, mx::uint32, s), false, s));
  }
  return mx::array(
      {rows, N},
      dt,
      std::make_shared<KQuantMoeStore>(s, 3, kquant_type, "", K, N, tile_rows),
      std::move(in));
}

std::vector<mx::array> moe_store_seg_pair(
    mx::array x,
    std::vector<mx::array> gate_planes,
    std::vector<mx::array> up_planes,
    const std::string& kquant_type,
    int N,
    mx::array map,
    mx::array counts,
    int tile_rows,
    std::optional<mx::array> x_rows,
    bool fused,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_store_seg_pair]";
  auto dt = x.dtype();
  if (x.ndim() != 2 || (dt != mx::float16 && dt != mx::bfloat16)) {
    throw std::invalid_argument(
        std::string(op) + " x must be [rows, K] float16 or bfloat16.");
  }
  const int K = x.shape(1);
  if (!moe_store_seg_supported(kquant_type, K) ||
      !moe_store_seg_paired(kquant_type)) {
    throw std::invalid_argument(
        std::string(op) + " no paired wide store kernel for '" + kquant_type +
        "' at this K on this device.");
  }
  if (tile_rows != 64 && tile_rows != 128) {
    throw std::invalid_argument(
        std::string(op) + " tile_rows must be 64 or 128.");
  }
  if (N <= 0 || N % 64 != 0) {
    throw std::invalid_argument(
        std::string(op) + " N must be a positive multiple of 64.");
  }
  const KqStoreSpec& spec = *kq_store_spec(kquant_type);
  for (const auto* planes : {&gate_planes, &up_planes}) {
    if (planes->size() != spec.planes.size()) {
      throw std::invalid_argument(
          std::string(op) + " wrong number of planes for the codec.");
    }
    for (size_t i = 0; i < planes->size(); ++i) {
      const auto& p = (*planes)[i];
      const size_t per = static_cast<size_t>(N) * (K / spec.wpb) *
          spec.planes[i].gpb * spec.planes[i].R;
      if (p.dtype() != mx::uint8 || p.ndim() != 1 || p.size() % per != 0 ||
          p.size() == 0) {
        throw std::invalid_argument(
            std::string(op) + " a plane is not whole slots of the codec.");
      }
      if (kq_has_layout(p) && !p.flags().row_contiguous) {
        throw std::invalid_argument(
            std::string(op) + " planes must be row-contiguous.");
      }
    }
  }
  if (map.ndim() != 2 || map.shape(1) != 3 || map.dtype() != mx::uint32 ||
      counts.ndim() != 1 || counts.shape(0) != 1 ||
      counts.dtype() != mx::uint32) {
    throw std::invalid_argument(
        std::string(op) + " map must be uint32 [tiles, 3], counts uint32 [1].");
  }
  if (x_rows &&
      (x_rows->ndim() != 1 || x_rows->shape(0) == 0 ||
       !mx::issubdtype(x_rows->dtype(), mx::integer))) {
    throw std::invalid_argument(
        std::string(op) + " x_rows must be a non-empty 1-D integer array.");
  }
  const int rows = x_rows ? x_rows->shape(0) : x.shape(0);
  std::vector<mx::array> in = {
      moe_dense_x(x, s),
      mx::contiguous(map, false, s),
      mx::contiguous(counts, false, s)};
  for (auto& p : gate_planes) {
    in.push_back(std::move(p));
  }
  for (auto& p : up_planes) {
    in.push_back(std::move(p));
  }
  if (x_rows) {
    in.push_back(mx::contiguous(mx::astype(*x_rows, mx::uint32, s), false, s));
  }
  auto prim = std::make_shared<KQuantMoeStore>(
      s, fused ? 5 : 4, kquant_type, "", K, N, tile_rows);
  if (fused) {
    return {mx::array({rows, N}, dt, std::move(prim), std::move(in))};
  }
  return mx::array::make_arrays(
      {{rows, N}, {rows, N}}, {dt, dt}, std::move(prim), std::move(in));
}

mx::array moe_store_down(
    mx::array x,
    mx::array w,
    mx::array shexp_w,
    const std::string& kquant_type,
    mx::array indices,
    mx::array scores,
    mx::array slots,
    std::vector<mx::array> planes,
    const std::string& shexp_kquant_type,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_store_down]";
  const std::string st =
      shexp_kquant_type.empty() ? kquant_type : shexp_kquant_type;
  if (!moe_store_supported(kquant_type, st)) {
    throw std::invalid_argument(
        std::string(op) + " no store kernel for '" + kquant_type +
        "' with shared-expert codec '" + st + "' on this device.");
  }
  auto dt = x.dtype();
  if (x.ndim() != 3 || (dt != mx::float16 && dt != mx::bfloat16)) {
    throw std::invalid_argument(
        std::string(op) + " x must be [T, S, K] float16 or bfloat16.");
  }
  const int T = x.shape(0);
  const int S = x.shape(1);
  const int K = x.shape(2);
  const KqStoreSpec& spec = kq_store_check(op, w, kquant_type, K);
  const int N = w.shape(1);
  const int ns = check_kq_shexp_down(op, shexp_w, st, K, N);
  if (S <= ns || indices.ndim() != 2 || indices.shape(0) != T ||
      indices.shape(1) != S - ns) {
    throw std::invalid_argument(
        std::string(op) + " indices must be [T, S - k].");
  }
  if (scores.ndim() != 2 || scores.shape(0) != T ||
      (scores.shape(1) != S && scores.shape(1) != S - ns)) {
    throw std::invalid_argument(
        std::string(op) + " scores must be [T, S] or [T, S - k].");
  }
  kq_store_check_planes(op, spec, planes, N, K, slots, w.shape(0));
  kq_check_weight_base_at_build(w, kquant_type, s, "moe_store_down");
  kq_check_weight_base_at_build(shexp_w, st, s, "moe_store_down");
  if (kq_has_layout(shexp_w) && !shexp_w.flags().row_contiguous) {
    throw std::invalid_argument(
        std::string(op) + " weights must be row-contiguous.");
  }
  if (!kq_store_width(
          x.shape(0), "KQ_MOE_STORE_DOWN_MIN_T", KQ_STORE_DOWN_MIN_T)) {
    kq_store_stub_check(op, w.shape(0), slots);
    return gather_qmv_mix_kq(
        std::move(x),
        std::move(w),
        std::move(shexp_w),
        kquant_type,
        std::move(indices),
        std::move(scores),
        shexp_kquant_type,
        std::nullopt,
        std::nullopt,
        s);
  }
  std::vector<mx::array> in = {
      std::move(w),
      std::move(shexp_w),
      moe_dense_x(x, s),
      prep_indices(indices, s),
      prep_scores(scores, s),
      kq_store_slots(slots, s)};
  for (auto& p : planes) {
    in.push_back(std::move(p));
  }
  return mx::array(
      {T, N},
      dt,
      std::make_shared<KQuantMoeStore>(s, 2, kquant_type, st, K),
      std::move(in));
}

} // namespace mlx_kquant
