// KQuantMatmul primitive: x @ dequant(w). The GPU path dispatches the leaf
// kernels (qmm / qmm_nax / qvm / qmv) from the bundled metallib via
// d.get_kernel(name, lib); the op guarantees row-contiguity before dispatch and
// kernel-name type tokens come from kq_type_string. NAX (tensor-core)
// availability is probed via kq_is_nax_available. qmm_splitk / qmm_nax_splitk
// (env-gated, KQ_QMM_SPLITK / KQ_QMM_SPLITK_NAX) partition K for the
// small-M band on the steel and NAX tiles respectively; qvm_split_k stays
// omitted - plain qvm is identical with less parallelism. KQuantMatmul itself
// never carries a bias (a separate elementwise add is fine off the
// decode-latency-critical path); the decode-only bias-fused fast path lives in
// the KQuantQmvBias primitive below (qmv_bias), which reuses this file's qmv
// dispatch helpers.
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_cpu_decode.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h" // elem_to_loc
#include "mlx/backend/cpu/encoder.h"
#include "mlx/types/half_types.h"
#include "mlx/utils.h" // env::enable_tf32

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
// array / Device / CommandEncoder and the shared dispatch helpers
// (kq_is_nax_available, kquant_qmv_bn, qmv_fast_k_align, codec_has_matmul,
// get_qmv_batch_limit, add_strides_and_shapes, kq_get_kernel) come from
// kquant_metal_internal.h.

// Measured small-BM policy per codec (cold-stream [17408x5120], M5 Max).
// bm32: the BM=32 double-buffered tile wins M13-32 over BM=64 (wins range
// +3% to +45%; excluded codecs measured neutral or worse -- iq2_s -24%,
// the DB loader duplication hurts register-heavy grid dequant).
// route_min: lowest M routed from the mv paths to qmm (0 = mv keeps all
// M <= 12; the flat codecs' and iq1_s's qmm is dequant-ALU-bound at
// 105-150 GB/s, below their mv paths through M12).
// db64_min_n: minimum N for which the M33-64 band dispatches the
// name-suffixed _db (double-buffered Ws) BM=64 qmm_t variant (0 = never).
// Measured per codec and per shape: only q6_k and q8_0 win the band at
// all, and both wins are N-gated (K is not the driver: q6_k at
// [17408x4096] wins +8-12% while [14336x5120] is a wash; the decode
// projection shapes N<=4096 lose -3 to -11%).
// bm128_min_m: lowest M at which the M>=193 band takes the BM=128 tile
// (still inside the even-ceil(M/64) window; never 0, every codec wins
// somewhere). Measured per codec at [17408x5120] and [4096x14336],
// M 224/256/512/1024, paired-process ABBA on M5 Max
// (benchmarks/bench_qmm_bm128_ab.py; re-measure on new silicon):
// - 193: wins from the band start. The dequant-ALU-heavy grid codecs
//   are the strongest (iq2_s/iq2_xs/iq4_xs +4 to +27% everywhere
//   measured; the taller tile amortizes dequant over twice the rows);
//   q6_k is the originally booked +9-25%.
// - 449: M224 (the pad-heaviest measured cell, 32 dead rows) loses
//   -3 to -13% and M256 is small-or-wash, while M512/M1024 win up to
//   +13% (q5_k, q3_k) -- the k-quants, q8_0, and the cheap-loader IQs.
// - 961: the flat quartet only wins the deepest bands (+1-7% at M1024)
//   and loses or washes below.
struct KqSmallBmPolicy {
  bool bm32;
  int route_min;
  int db64_min_n;
  int bm128_min_m;
};
static KqSmallBmPolicy kq_smallbm_policy(const std::string& t) {
  if (t == "q4_k" || t == "q3_k") {
    return {true, 7, 0, 449};
  }
  if (t == "q2_k" || t == "iq3_xxs") {
    return {true, 8, 0, 449};
  }
  if (t == "q6_k") {
    // +8-17% at N=17408 (any K), wash at 14336, -3/-11% at 4096/1024.
    return {true, 9, 16384, 193};
  }
  if (t == "q8_0") {
    // db64: +2-9% at 17408, +2-4% at 14336, ~0 at 4096, -3/-8% at 1024.
    // route_min 8 post-ushort-loader (forced M8 391 vs mv 378 at 17408).
    return {true, 8, 8192, 449};
  }
  if (t == "q5_k") {
    return {true, 9, 0, 449};
  }
  if (t == "iq3_s") {
    return {true, 9, 0, 193};
  }
  if (t == "iq2_xxs") {
    return {true, 10, 0, 193};
  }
  if (t == "iq4_xs") {
    return {true, 11, 0, 193};
  }
  // Flat-family entries are post-ushort-loader (the 32-single-byte-load
  // dequant chain was the 105-150 GB/s wall; with it gone these codecs
  // earn sub-13 routes, iq4_nl rejoins bm32 at +27-52% over bm64, and
  // db64 splits: q4_1/q5_0/q5_1 win +9-20% at N=17408 and +2-7% at 8192
  // but lose -5-13% at 1024, while q4_0/iq4_nl stay neutral-to-negative
  // (their loaders are cheap enough that the duplicated Ws buffers cost
  // more occupancy than the latency hiding returns).
  if (t == "q4_0") {
    return {true, 6, 0, 961};
  }
  if (t == "iq4_nl") {
    return {true, 6, 0, 449};
  }
  if (t == "q4_1" || t == "q5_1") {
    return {true, 6, 8192, 961};
  }
  if (t == "q5_0") {
    return {true, 7, 8192, 961};
  }
  if (t == "iq1_s") {
    return {true, 0, 0, 449};
  }
  return {false, 0, 0, 193}; // iq2_xs, iq2_s, iq1_m
}

// Shape-adjusted sub-13 route threshold. The table crossovers were tuned
// at [17408x5120]; they hold for N >= 8192 but sit too low for strong-mv
// codecs at decode projection shapes, where the qmm grid is only
// ceil(N/64) threadgroups and the in-tile K walk serializes (measured at
// [4096x14336] down, [4096x4096] q/o, [1024x4096] GQA k/v). Weak-mv
// codecs (q4_k, q2_k, q5_k, iq2_xxs) measured clean at all three shapes
// and keep the table value; iq4_xs loses its whole M11-12 window at
// small N and drops the route there.
static int kq_smallm_route_min(const std::string& t, int N, int K) {
  int m = kq_smallbm_policy(t).route_min;
  if (m == 0) {
    return 0;
  }
  if (t == "q6_k" && N >= 100000) {
    // Vocab-head N shifts the q6_k crossover down one: the mv paths decay
    // faster with N (M8 at N=248320 reads 237 GB/s vs 307 at N=17408)
    // while qmm holds 289-299. 100k separates vocab heads (128k-262k)
    // from the largest dense FFN rows (~53k).
    return 8;
  }
  if (N >= 8192) {
    return m;
  }
  if (t == "q6_k") {
    return (K >= 8192 || N <= 2048) ? 12 : 11;
  }
  if (t == "q8_0") {
    // q8_0's mv is the strongest in the fleet (500 GB/s at M5); even with
    // the ushort loader the qmm never beats it below M13 at decode
    // projection shapes (down M8 246 vs 374 mv, kv M12 324 vs 332). The
    // M13+ BM=32 tile is separate and keeps its win.
    return 0;
  }
  if (t == "q4_0" || t == "q4_1" || t == "iq4_nl") {
    // Down-projection K walk pushes the crossover to 8 (q4_0 M8 251 vs
    // 227 mv at [4096x14336], iq4_nl 232 vs 221); kv shapes cross at 7.
    // iq4_nl crossovers are measured on its bm32 tile.
    return K >= 8192 ? 8 : 7;
  }
  if (t == "q5_0" || t == "q5_1") {
    return 8;
  }
  if (t == "q3_k") {
    return N <= 2048 ? 8 : m;
  }
  if (t == "iq3_xxs") {
    return N <= 2048 ? 10 : m;
  }
  if (t == "iq3_s") {
    return 12;
  }
  if (t == "iq4_xs") {
    return 0;
  }
  return m;
}

// NAX (tensor-core) GEMM dispatch for the quantized matmul (no biases).
void qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
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
  // Small-BM tile for small M: BM=64 wastes up to 75% of MMA issues on row
  // padding at M<=32 (MMA is ~48% of kernel time). BM=32 is the smallest
  // tile the NAX fragment pairing allows (TM=1, TN=2). Codecs whose loaders
  // are db_safe and have smallbm instantiations only. KQ_NAX_SMALL_BM:
  // 0 = off, 2 = force bm32 for policy-excluded codecs (crossover
  // finding), unset or 1 = per-codec policy.
  static const int small_bm_mode = []() {
    const char* e = std::getenv("KQ_NAX_SMALL_BM");
    return e == nullptr ? 1 : std::atoi(e);
  }();
  // Window ends at 32: above it grid.y goes to 2 and every weight column
  // tile streams once per M-tile, which halves per-weight bandwidth at
  // DRAM-bound (measured: q6_k M48 146 GB/s on bm32 vs 184 on bm64).
  if (small_bm_mode != 0 && transpose && M <= 32 &&
      (small_bm_mode == 2 || kq_smallbm_policy(kquant_type).bm32)) {
    bm = 32;
  }
  // BM=128 tile for the M>=193 band: halves the row-tile count and with
  // it weight re-streams per unique weight (q6_k booked +9-25% M224-512;
  // prefill M512-2048 +8-10% hot, +31-40% cold). Padding decides the
  // rest of the band: with M = 128q + r the tile carries 64 extra
  // padding rows exactly when r is in (1,64], and every measured loss
  // cell (M144/160/192 -12-19%, M320 -6%) is in that zone while r == 0
  // or r > 64 is neutral-or-win -> dispatch only when ceil(M/64) is
  // even. The entry floor is per-codec (bm128_min_m: the 19-codec ABBA
  // showed the shallow band flipping sign by loader weight, see the
  // policy table). KQ_NAX_BM128: 0 = off, 1 = force floor 193 for all
  // codecs (crossover finding), 2 = drop the floor entirely (probes the
  // M65-128 wash band, symmetric with KQ_NAX_SMALL_BM=2), unset =
  // per-codec policy.
  static const int bm128_mode = []() {
    const char* e = std::getenv("KQ_NAX_BM128");
    return e == nullptr ? -1 : std::atoi(e);
  }();
  int bm128_min_m = kq_smallbm_policy(kquant_type).bm128_min_m;
  if (bm128_mode != 0 && transpose && ((M + 63) / 64) % 2 == 0 &&
      (bm128_mode == 1       ? M >= 193
           : bm128_mode == 2 ? true
                             : M >= bm128_min_m)) {
    bm = 128;
  }
  // M33-64 band: the name-suffixed _db BM=64 variant double-buffers Ws
  // (bm32 here would stream weights twice via grid.y=2; blanket DB@64
  // regressed M96+ -3-15% and prefill -3-7%, so it is band-gated).
  // KQ_NAX_DB64: 0 = off, 1 = drop the N floor (crossover finding),
  // unset = per-codec policy. Only policy-enabled codecs
  // (db64_min_n > 0) carry _db instantiations, so 1 is bounded by
  // availability; probing another codec needs its instantiation
  // restored and a metallib rebuild.
  static const int db64_mode = []() {
    const char* e = std::getenv("KQ_NAX_DB64");
    return e == nullptr ? -1 : std::atoi(e);
  }();
  int db64_min_n = kq_smallbm_policy(kquant_type).db64_min_n;
  bool use_db64 = transpose && bm == 64 && M >= 33 && M <= 64 &&
      db64_min_n > 0 &&
      (db64_mode == 1 || (db64_mode == -1 && N >= db64_min_n));
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);
  // Row-tile traversal swizzle (M > 64): fold row-tiles into grid.x so
  // consecutive launches cover the same BN-column weight band, letting
  // row-tiles 2..2^log read it from SLC instead of re-streaming DRAM
  // (weight-normalized bandwidth otherwise divides by the row-tile
  // count). kq_swizzle_tid in the kernel derives the fold factor from
  // threadgroups_per_grid; padded row-tiles early-return.
  // KQ_NAX_SWIZZLE: 1 = on (experiment lever; default off).
  static const bool swizzle_on = []() {
    const char* e = std::getenv("KQ_NAX_SWIZZLE");
    return e != nullptr && std::atoi(e) != 0;
  }();
  if (swizzle_on && transpose && bm == 64) {
    int y_tiles = (M + bm - 1) / bm;
    if (y_tiles >= 2) {
      int log = y_tiles >= 4 ? 2 : 1;
      grid_dims = MTL::Size(
          static_cast<size_t>((N + bn - 1) / bn) << log,
          (y_tiles + (1 << log) - 1) >> log,
          B);
    }
  }

  bool aligned = N % bn == 0;
  bool batched = B > 1;
  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + (transpose ? "qmm_t_nax_" : "qmm_n_nax_"),
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
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0",
      use_db64 ? "_db" : "");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  add_strides_and_shapes(ce, B <= 1, x, w, scales, c);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Tiled quantized GEMM (no biases). The split-k variant (qmm_splitk below)
// covers the small-M occupancy hole; plain qmm is the general path.
void qmm(
    const array& x,
    const array& w,
    const array& scales,
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
  // The op promotes float32 x to bfloat16 before constructing this primitive,
  // so x.dtype() is never float32 here and the NAX path stays eligible on the
  // dtype axis with no tf32 gate.
  if (kq_is_nax_available() && transpose && (K % 64 == 0) &&
      (x.dtype() != mx::float32) && codec_has_nax(kquant_type)) {
    return qmm_nax(
        x,
        w,
        scales,
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
  int wm = 2, wn = 2;
  int bm = 64;
  int bn = transpose ? 64 : 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % bn == 0;
  bool batched = B > 1;
  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + (transpose ? "qmm_t_" : "qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  add_strides_and_shapes(ce, B <= 1, x, w, scales, c);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Split-K qmm_t for the small-M decode band. The plain tile grid is only
// ceil(N/64) x 1 threadgroups at decode shapes with M <= 32 (84 at
// [5376 x 21504]) and the in-tile K walk serializes, capping qmm/NAX at
// 160-257 GB/s while mv_ext decays past M~4 on L2 activation re-reads.
// Partitioning K across grid.z multiplies threadgroup count by `splits`;
// each slice writes a T partial tile and a second tiny pass folds them in
// f32. Slice starts must land on wire-block boundaries, so the partition
// is a multiple of group_size (the caller guarantees splits divides
// K / group_size). Non-batched transpose shapes only.
void qmm_splitk(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    int splits,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  constexpr int bm = 32, bn = 32;
  constexpr int wm = 2, wn = 2;
  const int k_partition = (K / group_size / splits) * group_size;
  const int part_stride = M * N;

  array partials({splits, M, N}, x.dtype(), nullptr, {});
  partials.set_data(mx::allocator::malloc(partials.nbytes()));

  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(partials);

  std::string type_string = kq_type_string(x.dtype());
  bool aligned = N % bn == 0;
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + "qmm_t_splitk_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      aligned ? "_alN_true" : "_alN_false");

  auto kernel = kq_get_kernel(d, kname);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(partials, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  ce.set_bytes(k_partition, c++);
  ce.set_bytes(part_stride, c++);
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, splits);
  ce.dispatch_threadgroups(grid_dims, group_dims);

  std::string aname = "kquant_qmm_splitk_accum_" + type_string;
  auto accum = kq_get_kernel(d, aname);
  ce.set_compute_pipeline_state(accum);
  const int n_elems = M * N;
  c = 0;
  ce.set_input_array(partials, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(n_elems, c++);
  ce.set_bytes(splits, c++);
  ce.set_bytes(part_stride, c++);
  MTL::Size agrid(static_cast<size_t>(n_elems), 1, 1);
  MTL::Size agroup(256, 1, 1);
  ce.dispatch_threads(agrid, agroup);
}

// Split-K qmm_t on the NAX BM=32 tile (KQ_QMM_SPLITK_NAX experiment). Same
// partial/fold shape as qmm_splitk, but slices run the tensor-core tile:
// the steel splitk probe measured per-TG pipeline bound (~140-160 GB/s flat
// in splits), while the NAX small-M cap is TG-count starvation -- the lever
// splitk actually multiplies. Slice starts must be superblock-aligned so
// every loader instance begins at kt_base 0; the caller guarantees splits
// divides K / max(superblock, BK). Non-batched transpose shapes only.
void qmm_nax_splitk(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    int splits,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  constexpr int bm = 32, bn = 64;
  constexpr int wm = 2, wn = 2;
  const int k_partition = K / splits;
  const int part_stride = M * N;

  array partials({splits, M, N}, x.dtype(), nullptr, {});
  partials.set_data(mx::allocator::malloc(partials.nbytes()));

  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(partials);

  std::string type_string = kq_type_string(x.dtype());
  bool aligned = N % bn == 0;
  std::string kname;
  kname.reserve(80);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + "qmm_t_nax_splitk_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm32_bn64_bk64_wm2_wn2",
      aligned ? "_alN_true" : "_alN_false");

  auto kernel = kq_get_kernel(d, kname);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(partials, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  ce.set_bytes(k_partition, c++);
  ce.set_bytes(part_stride, c++);
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, splits);
  ce.dispatch_threadgroups(grid_dims, group_dims);

  std::string aname = "kquant_qmm_splitk_accum_" + type_string;
  auto accum = kq_get_kernel(d, aname);
  ce.set_compute_pipeline_state(accum);
  const int n_elems = M * N;
  c = 0;
  ce.set_input_array(partials, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(n_elems, c++);
  ce.set_bytes(splits, c++);
  ce.set_bytes(part_stride, c++);
  MTL::Size agrid(static_cast<size_t>(n_elems), 1, 1);
  MTL::Size agroup(256, 1, 1);
  ce.dispatch_threads(agrid, agroup);
}

// Vector-times-matrix quantized kernel dispatch (no biases).
void qvm(
    const array& x,
    const array& w,
    const array& scales,
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
  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + "qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  add_strides_and_shapes(ce, B <= 1, x, w, scales, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Matrix-times-vector quantized kernel dispatch (no biases).
void qmv(
    const array& x,
    const array& w,
    const array& scales,
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

  // Finer tiling: 2 output rows per threadgroup (vs bn), multiplying the
  // threadgroup count for the same N. Recovers occupancy for mid-size decode
  // dispatches where N/bn threadgroups underfill the GPU; loses at
  // embedding-size N (already saturated). Bit-exact vs the default tiling.
  // Default per codec via kquant_qmv_fine_default_max_n; KQ_QMV_FINE=1
  // forces fine, =0 forces coarse (A/B lever). Read live (KQ_DISABLE_NAX
  // precedent) so one process can interleave coarse/fine for
  // thermally-paired A/Bs; getenv cost is noise per dispatch.
  const char* qmv_fine_e = std::getenv("KQ_QMV_FINE");
  const int qmv_fine_env = qmv_fine_e != nullptr ? std::atoi(qmv_fine_e) : -1;
  const bool fine_ok = B == 1 && codec_has_qmv_fine(kquant_type);
  const bool use_fine = fine_ok &&
      (qmv_fine_env == 1 ||
       (qmv_fine_env != 0 && M == 1 &&
        N <= kquant_qmv_fine_default_max_n(kquant_type)));
  const int rows_per_tg = use_fine ? 2 : bn;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + rows_per_tg - 1) / rows_per_tg, B);

  std::string type_string = kq_type_string(x.dtype());
  bool fast = (N % bn == 0) && (K % qmv_fast_k_align() == 0);
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (fast ? (use_fine ? "qmv_fast_fine_" : "qmv_fast_")
                : (use_fine ? "qmv_fine_" : "qmv_")),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  add_strides_and_shapes(ce, B <= 1, x, w, scales, c);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Bias-fused qmv/qmv_fast dispatch: decode-only (M=1, non-batched) fast path
// for a KQuantLinear whose GGUF weight carries a real linear bias. The caller
// (KQuantQmvBias::eval_gpu, gated by the M=1 shape contract
// quantized_matmul_qmv_bias() enforces at the op level) guarantees M=1 --
// unlike qmv() above, there is no batched-B or M>1 fallback here because the
// only caller (KQuantLinear) never batches and only takes this path at
// decode.
void qmv_bias(
    const array& x,
    const array& w,
    const array& scales,
    const array& bias,
    array& out,
    int group_size,
    int bits,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int bn = kquant_qmv_bn(kquant_type);
  int bk = 32;

  // Same fine-tiling policy as qmv() above; M == 1 is the shape contract
  // here, so only the codec ceiling gates the default.
  const char* qmv_fine_e = std::getenv("KQ_QMV_FINE");
  const int qmv_fine_env = qmv_fine_e != nullptr ? std::atoi(qmv_fine_e) : -1;
  const bool use_fine = codec_has_qmv_fine(kquant_type) &&
      (qmv_fine_env == 1 ||
       (qmv_fine_env != 0 && N <= kquant_qmv_fine_default_max_n(kquant_type)));
  const int rows_per_tg = use_fine ? 2 : bn;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(1, (N + rows_per_tg - 1) / rows_per_tg, 1);

  std::string type_string = kq_type_string(x.dtype());
  bool fast = (N % bn == 0) && (K % qmv_fast_k_align() == 0);
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (fast ? (use_fine ? "qmv_fast_bias_fine_" : "qmv_fast_bias_")
                : (use_fine ? "qmv_bias_fine_" : "qmv_bias_")),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_input_array(bias, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  add_strides_and_shapes(ce, /*non_batched=*/true, x, w, scales, c);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Verify-shaped small-M matmul. One threadgroup per
// N-tile (grid_dims.x = 1) reads each weight tile once and dots it against all
// M activation rows, amortizing the dominant weight read; the per-row qmv would
// re-read it M times (M on grid_dims.x). Non-batched only; M (= vm) in
// [2, verify_qmv_max_rows()], codec in codec_has_verify_qmv. Bit-for-bit
// identical to running qmv per row.
void verify_qmv(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int bn = kquant_qmv_bn(kquant_type);
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);

  // Weight rows small enough at modest K stay L2-resident, so the default
  // verify tiling (8 output rows / threadgroup => few threadgroups) can starve
  // the GPU of occupancy vs the per-row qmv (M x more threadgroups) without
  // repaying it in saved DRAM traffic. The finer variant emits 2 output rows /
  // threadgroup (4x the threadgroups). Bit-exact vs the default. q8_0 measured
  // a win at idle so it is fine by default; q6_k measured NEUTRAL under
  // saturation (the verify forward's real condition), so it stays coarse by
  // default. Only q8_0 and q6_k have the fine kernel instantiated.
  // KQ_VERIFY_FINE=1 forces fine for both, KQ_VERIFY_FINE=0 forces coarse for
  // both (A/B lever).
  static const int verify_fine = []() {
    const char* e = std::getenv("KQ_VERIFY_FINE");
    return e != nullptr ? std::atoi(e) : -1; // -1 = per-codec default
  }();
  bool codec_has_fine = (kquant_type == "q8_0" || kquant_type == "q6_k");
  bool default_fine = (kquant_type == "q8_0"); // q6_k neutral -> coarse default
  bool use_fine = codec_has_fine &&
      (verify_fine == 1 || (verify_fine != 0 && default_fine));
  std::string verify_kname = "verify_qmv_";
  int rows_per_tg =
      bn; // default kernel emits bn (= num_simdgroups*RPS) rows/tg
  if (use_fine) {
    verify_kname = "verify_qmv_fine_";
    rows_per_tg = 2; // num_simdgroups(2) * results_per_simdgroup(1)
  }
  MTL::Size grid_dims(1, (N + rows_per_tg - 1) / rows_per_tg, 1);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + verify_kname,
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++); // in_vec_size
  ce.set_bytes(N, c++); // out_vec_size
  ce.set_bytes(M, c++); // vm (activation-row count)
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Flat-with-M verify mat-vec (kq_<codec>_mv_ext): port of ggml mul_mv_ext_q4x4.
// One output row per thread, M (= vm) register accumulators, nypsg=4 output
// rows in parallel per simdgroup, nxpsg=8-lane K-reduction; the weight row
// streams once and is dotted against all M activation columns. M in [2, 12]
// selects the kernel (r1ptg is compile-time). All wired codecs (the kname
// prefix carries the codec); same call args as verify_qmv.
void verify_mv_ext(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  constexpr int nsg = 2;
  constexpr int nxpsg = 8;
  // Wide-M rows-per-thread experiment (KQ_MV_EXT_NR=2): each thread owns two
  // consecutive output rows and shares one activation load across them,
  // halving the M*N*K activation cache traffic that dominates past M~5.
  // q6_k-only instantiations for now; default 1 = shipped behavior.
  static const int mv_ext_nr = []() {
    const char* e = std::getenv("KQ_MV_EXT_NR");
    return e != nullptr ? std::atoi(e) : 1;
  }();
  const bool use_nr2 = mv_ext_nr == 2 && M >= 5 && kquant_type == "q6_k";
  // Shuffle-broadcast experiment (KQ_MV_EXT_SB=1): ty-lanes exchange
  // activation quarters over simd_shuffle instead of each loading the full
  // window -- activation cache traffic / 4, same grid. q6_k M 4-12 only.
  static const bool mv_ext_sb = []() {
    const char* e = std::getenv("KQ_MV_EXT_SB");
    return e != nullptr && std::atoi(e) == 1;
  }();
  const bool use_sb = !use_nr2 && mv_ext_sb && M >= 4 && kquant_type == "q6_k";
  // Wide-nxpsg experiment (KQ_MV_EXT_NX=16|32): fewer redundant activation
  // readers per element (nypsg*nsg drops 8 -> 4 -> 2) + more threadgroups.
  // q6_k M 4-12 only; wins here would generalize per codec.
  static const int mv_ext_nx = []() {
    const char* e = std::getenv("KQ_MV_EXT_NX");
    const int v = e != nullptr ? std::atoi(e) : 0;
    return (v == 16 || v == 32) ? v : 0;
  }();
  const bool use_nx =
      !use_nr2 && !use_sb && mv_ext_nx != 0 && M >= 4 && kquant_type == "q6_k";
  const int nxpsg_eff = use_nx ? mv_ext_nx : nxpsg;
  const int rows_per_tg = (32 / nxpsg_eff) * nsg * (use_nr2 ? 2 : 1);
  MTL::Size group_dims(32, nsg, 1);
  MTL::Size grid_dims((N + rows_per_tg - 1) / rows_per_tg, 1, 1);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + "mv_ext_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_m",
      M,
      use_nr2 ? "_nr2"
              : (use_sb ? "_sb"
                        : (use_nx ? (mv_ext_nx == 16 ? "_x16" : "_x32") : "")));

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++); // in_vec_size
  ce.set_bytes(N, c++); // out_vec_size
  ce.set_bytes(M, c++); // vm (== r1ptg)
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// The qmv_quad branch (K==64/128) is unreachable for kquant - eval_gpu throws
// for that case first - so this always routes to qmv.
void dispatch_qmv(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  qmv(x, w, scales, out, group_size, bits, M, N, K, d, s, kquant_type);
}

} // namespace

#endif // _METAL_

std::vector<mx::Shape> KQuantMatmul::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const KQuantCodec* codec = codec_by_name(kquant_type_);
  int weights_per_row =
      (w.shape(-1) / codec->bytes_per_block) * codec->weights_per_block;
  int N = transpose_ ? w.shape(-2) : weights_per_row;
  auto shape = x.shape();
  shape.back() = N;
  return {shape};
}

bool KQuantMatmul::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMatmul&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_ && transpose_ == o.transpose_;
}

void KQuantMatmul::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: x, w (uint8), scales placeholder (ignored). Matrix-contiguous by
  // the op, so the M x K / weight rows are dense; leading (batch) dims are
  // walked via elem_to_loc.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);
  encoder.dispatch([out = mx::array::unsafe_weak_copy(out),
                    x = mx::array::unsafe_weak_copy(x),
                    w = mx::array::unsafe_weak_copy(w),
                    transpose_ = transpose_,
                    kquant_type = kquant_type_]() mutable {
    int K = x.shape(-1);
    int M = x.ndim() > 1 ? x.shape(-2) : 1;
    int N = out.shape(-1);
    int batch_size =
        static_cast<int>(x.size() / (static_cast<std::size_t>(K) * M));
    std::size_t w_batch_els =
        w.ndim() > 2 ? static_cast<std::size_t>(w.shape(-1)) * w.shape(-2) : 0;
    auto run = [&](auto* tag) {
      using T = std::remove_pointer_t<decltype(tag)>;
      for (int i = 0; i < batch_size; i++) {
        kquant_qmm_cpu<T>(
            out.data<T>() + static_cast<std::size_t>(i) * M * N,
            x.data<T>() +
                elem_to_loc64(
                    static_cast<int64_t>(i) * M * K, x.shape(), x.strides()),
            w.data<uint8_t>() +
                elem_to_loc64(
                    static_cast<int64_t>(i) * static_cast<int64_t>(w_batch_els),
                    w.shape(),
                    w.strides()),
            M,
            N,
            K,
            transpose_,
            kquant_type);
      }
    };
    auto dt = x.dtype();
    if (dt == mx::float32) {
      run(static_cast<float*>(nullptr));
    } else if (dt == mx::float16) {
      run(static_cast<mx::float16_t*>(nullptr));
    } else if (dt == mx::bfloat16) {
      run(static_cast<mx::bfloat16_t*>(nullptr));
    } else {
      throw std::runtime_error(
          "[mlx_kquant] quantized_matmul: only float32/float16/bfloat16 inputs "
          "are supported.");
    }
  });
}

std::vector<mx::array> KQuantMatmul::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  // primals = {x, w (wire bytes), scales placeholder}. Only the gradient wrt x
  // is defined: dL/dx = cotan @ dequant(w) with the transpose flipped, i.e. the
  // same quantized matmul run the other way. The quantized base is frozen (the
  // LoRA use case), so the weight/scale branches throw.
  std::vector<mx::array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(quantized_matmul(
          cotangents[0],
          primals[1],
          primals[2],
          kquant_type_,
          !transpose_,
          stream()));
    } else if (arg == 1) {
      throw std::invalid_argument(
          "[mlx_kquant] quantized_matmul vjp: no gradient wrt the quantized "
          "weights (the kquant base is frozen).");
    } else {
      throw std::invalid_argument(
          "[mlx_kquant] quantized_matmul vjp: no gradient wrt scales.");
    }
  }
  return vjps;
}

#ifdef _METAL_

void KQuantMatmul::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // inputs are row-contiguous (ensured by the op): x, w (uint8), scales.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];

  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? static_cast<int>(x.size()) / K : x.shape(-2);
  int N = out.shape(-1);

  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;

  // KQuant special cases.
  if (!transpose_ && M < vector_limit) {
    qmm(x,
        w,
        scales,
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
  // There is no kquant qmv_quad kernel for K in {64,128}. Plain qmv is correct
  // for any K that is a multiple of the codec group size - K-quants (gs=256)
  // can never reach K=64/128, and the legacy gs=32 codecs only could on weights
  // whose input dim is exactly 64/128 (none in standard transformers). Fall
  // through to dispatch_qmv below rather than throw; a qmv_quad kernel would be
  // a perf-only path for an essentially-dead case.

  // Probe lever: KQ_FORCE_QMM_MIN_M=<m> routes transpose shapes with M >= m
  // straight to qmm (-> NAX), bypassing the mv_ext/verify_qmv claims, for
  // qmm-vs-mv_ext crossover measurement below M=13.
  static const int force_qmm_min_m = []() {
    const char* e = std::getenv("KQ_FORCE_QMM_MIN_M");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  if (force_qmm_min_m > 0 && transpose_ && M >= force_qmm_min_m) {
    qmm(x,
        w,
        scales,
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

  // NAX split-K qmm experiment (KQ_QMM_SPLITK_NAX=<target splits>, 0 = off,
  // read once): K-slices on the tensor-core BM=32 tile; see qmm_nax_splitk.
  // Slice quantum is max(superblock, BK) so every slice starts a loader at
  // kt_base 0. q6_k + q8_0 instantiations only.
  static const int qmm_splitk_nax_env = []() {
    const char* e = std::getenv("KQ_QMM_SPLITK_NAX");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  if (qmm_splitk_nax_env > 1 && transpose_ && non_batched && M <= 32 &&
      (kquant_type_ == "q6_k" || kquant_type_ == "q8_0") &&
      kq_is_nax_available() && (K % 64 == 0) && x.dtype() != mx::float32) {
    const int sliceq = std::max(group_size_, 64);
    const int nblk = K / sliceq;
    int sp = std::min(qmm_splitk_nax_env, nblk);
    while (sp > 1 && nblk % sp != 0) {
      --sp;
    }
    if (sp > 1) {
      qmm_nax_splitk(
          x,
          w,
          scales,
          out,
          group_size_,
          bits_,
          M,
          N,
          K,
          sp,
          d,
          s,
          kquant_type_);
      return;
    }
  }

  // Split-K qmm experiment (KQ_QMM_SPLITK=<target splits>, 0 = off, read
  // once): the small-M band's occupancy lever; see qmm_splitk. Routes when
  // a >1 divisor of the wire-block count exists at or under the target.
  // K-quants + q8_0 only for now (instantiation coverage).
  static const int qmm_splitk_env = []() {
    const char* e = std::getenv("KQ_QMM_SPLITK");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  if (qmm_splitk_env > 1 && transpose_ && non_batched && M <= 32 &&
      (kquant_type_ == "q6_k" || kquant_type_ == "q5_k" ||
       kquant_type_ == "q4_k" || kquant_type_ == "q3_k" ||
       kquant_type_ == "q2_k" || kquant_type_ == "q8_0")) {
    const int nblk = K / group_size_;
    int sp = std::min(qmm_splitk_env, nblk);
    while (sp > 1 && nblk % sp != 0) {
      --sp;
    }
    if (sp > 1) {
      qmm_splitk(
          x,
          w,
          scales,
          out,
          group_size_,
          bits_,
          M,
          N,
          K,
          sp,
          d,
          s,
          kquant_type_);
      return;
    }
  }

  // Small-M qmm route: the double-buffered BM=32 NAX tile beats the mv
  // paths' wide-M decay above a per-codec crossover (kq_smallbm_policy;
  // measured cold-stream, e.g. q6_k M>=9 274-305 vs 221-254, q4_k M>=7
  // 249-287 vs 149-256, q2_k M>=8 133-194 vs 88-135, iq3_xxs M>=8
  // 163-175 vs 102-158). KQ_NAX_SMALL_BM=0 restores the old routing
  // (same switch that picks the BM=32 tile in qmm_nax).
  static const bool nax_smallm_route = []() {
    const char* e = std::getenv("KQ_NAX_SMALL_BM");
    return e == nullptr || std::atoi(e) != 0;
  }();
  int smallm_min = kq_smallm_route_min(kquant_type_, N, K);
  if (nax_smallm_route && transpose_ && non_batched && smallm_min > 0 &&
      M >= smallm_min && M <= 12 && kq_is_nax_available() && (K % 64 == 0) &&
      x.dtype() != mx::float32) {
    qmm(x,
        w,
        scales,
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

  if (M >= vector_limit) {
    // For transpose shapes in the mv_ext M-range, the weight-read-amortizing
    // kernel beats qmm's under-utilised BM=64 tile at small M. Let those fall
    // through to the mv_ext check below; vector_limit was calibrated for
    // qmv-vs-qmm, not mv_ext-vs-qmm.
    if (!(transpose_ && non_batched && M >= 2 && M <= 12)) {
      qmm(x,
          w,
          scales,
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
  }

  if (transpose_) {
    // The verify width goes through the flat-with-M mat-vec (kq_<codec>_mv_ext,
    // a port of ggml mul_mv_ext): one output row per thread with M register
    // accumulators + nypsg parallel rows per simdgroup, vs verify_qmv's
    // [MAX_VM][RPS] block that stays occupancy-exposed under saturation.
    // Measured flat with M (matching llama) and bit-exact (fp-noise) vs
    // verify_qmv and per-row qmv. On by default for the codecs in
    // mv_ext_default_on; KQ_VERIFY_EXT=0 forces the verify_qmv path, =1 forces
    // mv_ext for any codec that has the kernel (A/B lever).
    static const int verify_ext = []() {
      const char* e = std::getenv("KQ_VERIFY_EXT");
      return e != nullptr ? std::atoi(e) : -1; // -1 = per-codec default
    }();
    // Every wired codec now has an mv_ext kernel: q8_0, the five K-quants, the
    // four legacy non-K (q4_0/q4_1/q5_0/q5_1), all nine IQ, and the native-fp
    // wire codecs (mxfp4/nvfp4). Validated bit-exact, so default-on ==
    // has-kernel.
    const bool codec_has_mv_ext = kquant_type_ == "q8_0" ||
        kquant_type_ == "q2_k" || kquant_type_ == "q3_k" ||
        kquant_type_ == "q4_k" || kquant_type_ == "q5_k" ||
        kquant_type_ == "q6_k" || kquant_type_ == "q4_0" ||
        kquant_type_ == "q4_1" || kquant_type_ == "q5_0" ||
        kquant_type_ == "q5_1" || kquant_type_ == "iq4_nl" ||
        kquant_type_ == "iq4_xs" || kquant_type_ == "iq3_s" ||
        kquant_type_ == "iq3_xxs" || kquant_type_ == "iq2_xxs" ||
        kquant_type_ == "iq2_xs" || kquant_type_ == "iq2_s" ||
        kquant_type_ == "iq1_s" || kquant_type_ == "iq1_m" ||
        kquant_type_ == "mxfp4" || kquant_type_ == "nvfp4";
    const bool mv_ext_default_on = codec_has_mv_ext;
    // Width gate for the DEFAULT path (the A/B force-on KQ_VERIFY_EXT=1 ignores
    // it). Measured DRAM-cold across every wired codec (M5 Max, [17408x5120],
    // working set streamed far past the SLC): at M==2 verify_qmv holds its
    // M==1 rate only for q4_k (495 vs mv_ext 422 GB/s) and q8_0 (540 vs 541);
    // every other codec craters there (q6_k 328, q3_k 179, legacy 213-298,
    // all vs mv_ext 353-536), and the wire codecs with no verify kernel
    // (mxfp4/nvfp4) fall to per-row qmv at half their M==1 rate (69/61 vs
    // mv_ext 392/432). M==2 is every B=2 decode step, not just the
    // draft-width-1 verify, so M==2 routes to mv_ext for all codecs except
    // the two where verify_qmv measures faster. IQ has no verify_qmv kernel,
    // so mv_ext stays on at every M>=2.
    const bool is_iq = kquant_type_.rfind("iq", 0) == 0;
    const bool verify_qmv_wins_m2 =
        kquant_type_ == "q4_k" || kquant_type_ == "q8_0";
    const bool mv_ext_width_ok = is_iq || M >= 3 || !verify_qmv_wins_m2;
    // 32-weight blocks (legacy + q8_0 + iq4_nl) align K to 32; the 256-weight
    // super-block codecs (K-quants + the other IQ) align to 256. Pull the
    // modulus from the codec geometry rather than hard-coding per codec.
    const KQuantCodec* mv_ext_codec = codec_by_name(kquant_type_);
    const int mv_ext_k_align =
        mv_ext_codec ? mv_ext_codec->weights_per_block : 256;
    if (codec_has_mv_ext && non_batched && M >= 2 && M <= 12 &&
        (K % mv_ext_k_align == 0) &&
        (verify_ext == 1 ||
         (verify_ext != 0 && mv_ext_default_on && mv_ext_width_ok))) {
      verify_mv_ext(
          x, w, scales, out, group_size_, bits_, M, N, K, d, s, kquant_type_);
      return;
    }
    // Verify / small-batch regime: amortize the weight read across the M rows
    // instead of re-reading it per row (qmv puts M on grid_dims.x). Falls back
    // to qmv outside the supported codec/shape/row-count envelope.
    // KQ_DISABLE_VERIFY_QMV=1 forces the per-row qmv path (A/B harness lever).
    static const bool verify_disabled = []() {
      const char* e = std::getenv("KQ_DISABLE_VERIFY_QMV");
      return e != nullptr && e[0] == '1';
    }();
    int bn = kquant_qmv_bn(kquant_type_);
    bool verify_ok = !verify_disabled && non_batched && M >= 2 &&
        M <= verify_qmv_max_rows() && (N % bn == 0) &&
        (K % qmv_fast_k_align() == 0) && codec_has_verify_qmv(kquant_type_);
    if (verify_ok) {
      verify_qmv(
          x, w, scales, out, group_size_, bits_, M, N, K, d, s, kquant_type_);
      return;
    }
    dispatch_qmv(
        x, w, scales, out, group_size_, bits_, M, N, K, d, s, kquant_type_);
    return;
  }

  // The split-k qvm variant is omitted here; plain qvm is correct.
  qvm(x, w, scales, out, group_size_, bits_, M, N, K, d, s, kquant_type_);
}

#else

void KQuantMatmul::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] quantized_matmul has no GPU implementation.");
}

#endif // _METAL_

std::vector<mx::Shape> KQuantQmvBias::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  auto shape = x.shape();
  shape.back() = w.shape(-2); // transpose=true only: w is [N, K]
  return {shape};
}

bool KQuantQmvBias::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQmvBias&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_;
}

void KQuantQmvBias::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: x, w (uint8), scales placeholder (ignored), bias. The op-level
  // shape contract (quantized_matmul_qmv_bias) guarantees M=1 and
  // non-batched, so this needs no batch loop, unlike KQuantMatmul::eval_cpu.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& bias = inputs[3];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(bias);
  encoder.set_output_array(out);
  encoder.dispatch([out = mx::array::unsafe_weak_copy(out),
                    x = mx::array::unsafe_weak_copy(x),
                    w = mx::array::unsafe_weak_copy(w),
                    bias = mx::array::unsafe_weak_copy(bias),
                    kquant_type = kquant_type_]() mutable {
    int K = x.shape(-1);
    int N = out.shape(-1);
    auto run = [&](auto* tag) {
      using T = std::remove_pointer_t<decltype(tag)>;
      kquant_qmm_cpu<T>(
          out.data<T>(),
          x.data<T>(),
          w.data<uint8_t>(),
          1,
          N,
          K,
          /*transpose=*/true,
          kquant_type);
      const T* bias_ptr = bias.data<T>();
      T* out_ptr = out.data<T>();
      for (int i = 0; i < N; i++) {
        out_ptr[i] = static_cast<T>(
            static_cast<float>(out_ptr[i]) + static_cast<float>(bias_ptr[i]));
      }
    };
    auto dt = x.dtype();
    if (dt == mx::float32) {
      run(static_cast<float*>(nullptr));
    } else if (dt == mx::float16) {
      run(static_cast<mx::float16_t*>(nullptr));
    } else if (dt == mx::bfloat16) {
      run(static_cast<mx::bfloat16_t*>(nullptr));
    } else {
      throw std::runtime_error(
          "[mlx_kquant] quantized_matmul_qmv_bias: only float32/float16/"
          "bfloat16 inputs are supported.");
    }
  });
}

#ifdef _METAL_

void KQuantQmvBias::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];
  const auto& bias = inputs[3];

  int K = x.shape(-1);
  int N = out.shape(-1);

  qmv_bias(
      x, w, scales, bias, out, group_size_, bits_, N, K, d, s, kquant_type_);
}

#else

void KQuantQmvBias::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] quantized_matmul_qmv_bias has no GPU implementation.");
}

#endif // _METAL_

} // namespace mlx_kquant
