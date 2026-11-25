// A lazy subgraph run as one primitive.
//
// fuse_graph walks a built graph from its outputs back to the named inputs
// and records every primitive on the way, in evaluation order. fused_call
// then builds one multi-output array for new inputs of the same shapes; its
// eval runs the recorded primitives' own eval_gpu (or eval_cpu) one after
// another. The kernels, their order and their operands are those of the
// graph it was recorded from, so the outputs carry the same bits. What goes
// away is the per-array work of building the graph and of the eval walk:
// the caller makes one array per call, and the eval visits one node.
//
// An evaluated array the walk meets (a weight, a constant) is held by the
// recording and passed as an input of every call; nothing is copied. A
// primitive with no inputs (a Full) is recorded and runs on every call.
// Intermediate arrays live until the command buffer they were encoded in
// has completed, like the inputs of any primitive.
//
// With `direct`, a tape on a GPU stream goes one step further once it has
// seen two evals: it replays the Metal calls of its primitives without
// running them (see the direct replay section below).
//
// Inference-only: no jvp, vjp or vmap.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kquant.h"
#include "kquant_internal.h"
#include "mlx/backend/cpu/encoder.h"

#ifdef _METAL_
// The serial-encoder lever swaps the compute encoder mlx holds for one of
// MTLDispatchTypeSerial, which needs CommandEncoder's private members. As
// in kquant_cb_caps.cpp, the pinned wheel ships the header this compiles
// against, so the layout matches its libmlx.
#define private public
#include "mlx/backend/metal/device.h"
#undef private
#include <objc/message.h>
#include <objc/runtime.h>

#include "mlx/allocator.h"
#endif

namespace mlx_kquant {

namespace {

struct FusedNode {
  std::shared_ptr<mx::Primitive> prim;
  // Slot of each input, and whether this node is the slot's last reader
  // (the array is then handed over, so the primitive may reuse its buffer
  // as the stock eval lets it).
  std::vector<int> in;
  std::vector<uint8_t> last;
  std::vector<int> out;
  std::vector<mx::Shape> shapes;
  std::vector<mx::Dtype> dtypes;
};

struct DirectState;

} // namespace

struct FusedTape {
  // Replay the recorded Metal calls once learned; its state (Metal only).
  bool direct{false};
  std::shared_ptr<DirectState> dstate;
  mx::Stream stream{0, mx::Device::gpu};
  // Run the tape in its own serial compute encoder (see run_tape).
  bool serial{false};
  int n_inputs{0};
  int n_slots{0};
  // Most inputs and most outputs of one node, and how many node inputs are
  // handed over: the run loop sizes its vectors once from these.
  int max_in{0};
  int max_out{0};
  int n_handed{0};
  std::vector<mx::array> constants;
  std::vector<mx::Shape> in_shapes;
  std::vector<mx::Dtype> in_dtypes;
  std::vector<FusedNode> nodes;
  std::vector<int> out_slots;
  std::vector<mx::Shape> out_shapes;
  std::vector<mx::Dtype> out_dtypes;
  // The parameter block: input `param_in` (int32, filled on the host) holds
  // the values that may change between replays. Slot i may take a value in
  // [param_lo[i], param_hi[i]]; a slot nobody declared keeps the value it
  // had when the graph was recorded.
  int param_in{-1};
  std::vector<int32_t> param_lo;
  std::vector<int32_t> param_hi;
};

namespace {

// The input of a primitive that holds a parameter block it reads on the
// host (kq_param_value), or -1.
int host_param_input(const mx::Primitive& p, int n_inputs) {
  if (auto* a = dynamic_cast<const KQuantKVAppend*>(&p)) {
    return a->param_input(n_inputs);
  }
  if (auto* a = dynamic_cast<const KQuantSDPAPagedRows*>(&p)) {
    return a->param_input(n_inputs);
  }
  if (auto* a = dynamic_cast<const KQDsaIndexerScoreDecode*>(&p)) {
    return a->param_input(n_inputs);
  }
  return -1;
}

// Slot values of the block `p` that the tape does not cover: -1 when all
// are inside their ranges, else the first slot outside.
int param_outside(const FusedTape& t, const int32_t* p) {
  for (size_t i = 0; i < t.param_lo.size(); ++i) {
    if (p[i] < t.param_lo[i] || p[i] > t.param_hi[i]) {
      return int(i);
    }
  }
  return -1;
}

template <bool GPU>
void run_tape(
    const std::shared_ptr<const FusedTape>& tape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    int* node_at = nullptr) {
  const FusedTape& t = *tape;
  // Slot r is inputs[r], then constants[r - n_inputs], then made[...].
  const int first_made = t.n_inputs + int(t.constants.size());
  std::vector<mx::array> made;
  made.reserve(t.n_slots - first_made);
  auto at = [&](int r) -> const mx::array& {
    if (r >= first_made) {
      return made[r - first_made];
    }
    return r < t.n_inputs ? inputs[r] : t.constants[r - t.n_inputs];
  };
  // Every array the tape makes stays alive until its command buffer has
  // completed: the handed-over node inputs end up here, the rest of `made`
  // follows after the loop. The caller's inputs are held by the eval that
  // runs this primitive, the constants by the tape.
  std::vector<mx::array> keep;
  std::vector<mx::array> ins;
  std::vector<mx::array> outs;
  keep.reserve(t.n_slots - first_made + t.n_handed);
  ins.reserve(t.max_in);
  outs.reserve(t.max_out);
#ifdef _METAL_
  if (GPU && t.serial) {
    // End mlx's concurrent encoder and hand it a serial one on the same
    // command buffer. Every dispatch of the tape then goes through mlx's
    // own CommandEncoder calls, so its fences between encoders, its op
    // count and its temporaries work as they do for any encoder; its
    // memory barriers are redundant in a serial encoder and harmless.
    auto& ce = mx::metal::get_command_encoder(t.stream);
    ce.end_encoding();
    ce.encoder_ = NS::RetainPtr(
        ce.buffer_->computeCommandEncoder(MTL::DispatchTypeSerial));
    ce.fence_ = NS::TransferPtr(ce.device_.mtl_device()->newFence());
  }
#endif
  for (size_t k = 0; k < t.nodes.size(); ++k) {
    auto& n = t.nodes[k];
    if (node_at != nullptr) {
      // The node whose calls a recording is seeing.
      *node_at = int(k);
    }
    ins.clear();
    outs.clear();
    for (size_t i = 0; i < n.in.size(); ++i) {
      if (n.last[i]) {
        // Last reader of an array the tape made: hand it over, so the
        // primitive may take its buffer as the stock eval lets it.
        ins.push_back(std::move(made[n.in[i] - first_made]));
      } else {
        ins.push_back(at(n.in[i]));
      }
    }
    // The outputs carry the primitive, as in the stock eval: mlx's own
    // eval_gpu bodies read the stream from out.primitive(). Detached once
    // evaluated, which also unlinks the siblings from each other.
    if (n.out.size() == 1) {
      outs.emplace_back(
          n.shapes[0], n.dtypes[0], n.prim, std::vector<mx::array>{});
    } else {
      outs = mx::array::make_arrays(n.shapes, n.dtypes, n.prim, {});
    }
    if (GPU) {
      n.prim->eval_gpu(ins, outs);
    } else {
      n.prim->eval_cpu(ins, outs);
    }
    for (auto& a : outs) {
      a.detach();
      made.push_back(std::move(a));
    }
    for (size_t i = 0; i < n.in.size(); ++i) {
      if (n.last[i]) {
        keep.push_back(std::move(ins[i]));
      }
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].copy_shared_buffer(at(t.out_slots[i]));
  }
  for (auto& a : made) {
    keep.push_back(std::move(a));
  }
  if (!GPU) {
    // The CPU tasks the nodes queued hold weak copies of their arrays: a
    // task queued after them holds the arrays until they have run, as the
    // stock CPU eval does for a primitive's inputs.
    mx::cpu::get_command_encoder(t.stream).dispatch(
        [keep = std::move(keep), tape]() {});
    return;
  }
#ifdef _METAL_
  if (GPU) {
    // Held by a completion handler of the command buffer, as the stock eval
    // holds a primitive's inputs. Not CommandEncoder temporaries: mlx drops
    // those from an encoder's fence bookkeeping, and some of these buffers
    // are the fused outputs, which later encoders read.
    auto& ce = mx::metal::get_command_encoder(t.stream);
    ce.get_command_buffer()->addCompletedHandler(
        [keep = std::move(keep), tape](MTL::CommandBuffer*) {});
    if (t.serial) {
      // Close the serial encoder: the next op opens a concurrent one.
      ce.end_encoding();
    }
  }
#endif
}

#ifdef _METAL_

// ---------------------------------------------------------------------------
// Direct replay.
//
// A tape recorded with `direct` learns the Metal calls its primitives make.
// Its first eval runs the tape as above with a proxy in place of the compute
// encoder mlx holds; the proxy writes every call down and passes it on. The
// list, with each buffer named by its role (a call input, a held constant, a
// call output, a buffer between nodes) and not by its pointer, becomes a
// program, and every later eval replays the program straight on the
// encoder: no eval_gpu, no array per node.
//
// A program is right for a later call only if the calls its primitives
// would make depend on nothing but what is fixed for the tape. That is:
//   - the primitives and every scalar they carry (offsets, lengths, row
//     counts, path choices made at graph build): fixed by the tape. A
//     caller whose graph changes with such a value records one tape per
//     value.
//   - shapes and dtypes of the inputs: fused_call refuses others.
//   - strides, flags, data size, 64-byte alignment and buffer sharing of
//     the inputs: checked on every call; a call that differs runs the
//     per-primitive loop.
//   - the stream: fixed by the tape.
//   - KQ_* levers: a program is made and used only while the lever cache is
//     on (kq_env_cache), when a lever is read once; any kq_env_cache call
//     drops the program and the tape learns again.
// and nothing else. That last part holds per primitive type, by review:
// only a tape whose primitives are all on the list in direct_listed()
// gets a program; any other tape runs the per-primitive loop for good.
// tests/test_fused_graph.py and tests/test_fused_direct_types.py record
// each listed type, replay it against the plain graph bit for bit over
// several inputs, and fail when a call list moves with anything outside
// the key.
//
// Debug levers. KQ_FUSED_DIRECT_CHECK=1 makes a tape record twice and
// compare before it replays, and refuse itself when the two differ, with
// the primitives of the differing calls named in fused_direct_info.
// KQ_FUSED_DIRECT_ANY=1 lets a tape with unlisted primitives learn, to
// probe a new layer with the check on. KQ_FUSED_DIRECT_SELFTEST=1 makes
// consecutive recordings differ, for the test of the check.
//
// Buffers. Call inputs and held constants are bound in place. Each call
// output gets a new buffer per call, so an output of one call is never
// rewritten by a later one. Buffers between nodes come from a pool shared
// by every program of the thread and stream: a replay is over, in encode
// order, before the next one starts, a barrier ahead of each replay orders
// them inside an encoder, and one pool buffer entered in mlx's fence sets
// orders them across encoders. A buffer a primitive fills on the host is
// kept by the program with the content of the recording. The program, the tape
// and the pool are held by a completion handler of every command buffer a
// replay encodes into.
//
// mlx bookkeeping. A replay enters its inputs and outputs in the encoder's
// fence sets (an input or held constant a dispatch writes counts as an
// output), asks for a barrier ahead of the next dispatch, and adds the
// recorded dispatch count and buffer size to the commit counters, so
// needs_commit and the ordering of later ops are those of the per-primitive
// loop. A held constant is fixed for the record: past the first command
// buffer only the constants the record writes itself enter the fence sets,
// so nothing outside the record may write one.
// ---------------------------------------------------------------------------

enum : uint8_t { D_PSO, D_BUF, D_BYTES, D_TGMEM, D_BARRIER, D_DTG, D_DT };
enum : uint8_t { R_INPUT, R_CONST, R_SCRATCH, R_OUTPUT, R_KEPT };

constexpr int kDirectMaxRefs = 512;
constexpr int kDirectMaxOutputs = 64;
constexpr size_t kDirectMaxKept = size_t(1) << 20;

struct DirectRaw {
  uint8_t kind;
  int node;
  const void* ptr{nullptr};
  uint64_t a{0};
  uint64_t b{0};
  std::vector<uint8_t> bytes;
  MTL::Size s0{0, 0, 0};
  MTL::Size s1{0, 0, 0};
};

struct DirectOp {
  uint8_t kind;
  uint8_t idx{0};
  int32_t ref{0};
  int32_t node{0};
  int64_t off{0};
  uint64_t len{0};
  const void* ptr{nullptr};
  MTL::Size s0{0, 0, 0};
  MTL::Size s1{0, 0, 0};
};

// An int32 inside the bytes of one setBytes call that is slot * mul + add
// of the parameter block.
struct DirectSite {
  int op; // index of the call (trace), byte offset in the blob (program)
  int field;
  int slot;
  int mul;
  int add;
  // Values of the slot the primitive takes (kq_param_site).
  int64_t lo;
  int64_t hi;
};

struct DirectRef {
  uint8_t cls;
  int index; // input, constant, per call output or kept buffer index
  size_t len;
  int node; // first node that bound it
  // An input or held constant buffer a dispatch of the record writes.
  bool written{false};
};

struct DirectOut {
  int kind; // 0 new buffer, 1 shares an earlier output's, 2 an input's, 3 a
            // constant's
  int other;
  int ref;
  int64_t off; // bytes for kind 0, items for the others
  mx::Strides strides;
  bool contiguous, row_contiguous, col_contiguous;
  size_t data_size;
};

struct DirectInput {
  mx::Strides strides;
  bool contiguous, row_contiguous, col_contiguous;
  size_t data_size;
  int64_t off;
  int group; // first input on the same buffer
};

struct DirectTrace {
  std::vector<DirectOp> ops;
  std::vector<uint8_t> blob;
  std::vector<DirectRef> refs;
  std::vector<DirectOut> outs;
  std::vector<DirectInput> ins;
  std::vector<size_t> fresh; // bytes of each per call output buffer
  std::vector<std::vector<uint8_t>> kept; // content of each host-written buffer
  std::vector<DirectSite> sites;
  int n_dispatch{0};
  int n_barrier{0};
  size_t sizes{0};
  std::string refuse;
  int refuse_node{-1};
};

struct DirectProgram {
  uint64_t id;
  // The recording the program was made from, for fused_direct_calls.
  std::shared_ptr<const DirectTrace> trace;
  std::vector<DirectOp> ops;
  std::vector<uint8_t> blob;
  std::vector<DirectRef> refs;
  std::vector<DirectOut> outs;
  std::vector<DirectInput> ins;
  std::vector<std::pair<int, int>> in_refs; // (ref, input index)
  // Inputs a dispatch of the record writes (an in-place append).
  std::vector<int> in_written;
  std::vector<std::pair<int, size_t>> fresh; // (ref, bytes)
  std::vector<mx::allocator::Buffer> kept;
  std::vector<DirectSite> sites;
  std::vector<const void*> const_ptrs;
  // Held constants a dispatch of the record writes (an in-place append).
  std::vector<const void*> const_written;
  // False until the command buffer of the recording has completed:
  // until then a held constant may still be the output of an encoder in
  // flight, and a replay enters the constants in the fence set too.
  mutable std::atomic<bool> settled{false};
  mutable std::atomic<uint64_t> replays{0};
  mutable std::atomic<uint64_t> plain{0};
  int n_dispatch{0};
  int n_barrier{0};
  int n_calls_recorded{0};
  int n_barrier_recorded{0};
  size_t sizes{0};
  size_t scratch_bytes{0};
  size_t kept_bytes{0};
  ~DirectProgram() {
    for (auto& b : kept) {
      mx::allocator::free(b);
    }
  }
};

struct DirectState {
  std::mutex mtx;
  int state{0}; // 0 nothing learned, 1 one recording, 2 program, 3 refused
  // Primitive types of the tape that are not on the direct list.
  std::vector<std::string> unlisted;
  unsigned epoch{0};
  std::unique_ptr<DirectTrace> first;
  std::shared_ptr<const DirectProgram> prog;
  std::string reason;
  std::vector<std::string> prims;
  uint64_t plain{0};
};

// Bytes of every pool of the process.
std::atomic<size_t> g_direct_pool_bytes{0};

// Buffers between nodes, shared by the programs of one thread and stream,
// or held by one binding.
struct DirectPool {
  std::vector<std::pair<size_t, mx::allocator::Buffer>> bufs;
  std::unordered_map<uint64_t, std::vector<const MTL::Buffer*>> assigned;
  size_t bytes{0};
  ~DirectPool() {
    for (auto& b : bufs) {
      mx::allocator::free(b.second);
    }
    g_direct_pool_bytes -= bytes;
  }
};

std::shared_ptr<DirectPool>& direct_pool(int stream_index) {
  thread_local std::unordered_map<int, std::shared_ptr<DirectPool>> pools;
  auto& p = pools[stream_index];
  if (!p) {
    p = std::make_shared<DirectPool>();
  }
  return p;
}

// The buffer of every ref of `prog` that is the same on every call: held
// constants are filled in by the caller, kept buffers and pool buffers here.
const std::vector<const MTL::Buffer*>&
direct_assign(DirectPool& pool, const DirectProgram& prog, const FusedTape& t) {
  auto it = pool.assigned.find(prog.id);
  if (it != pool.assigned.end()) {
    return it->second;
  }
  std::vector<const MTL::Buffer*> base(prog.refs.size(), nullptr);
  std::vector<int> need;
  for (size_t r = 0; r < prog.refs.size(); ++r) {
    auto& ref = prog.refs[r];
    if (ref.cls == R_CONST) {
      base[r] = static_cast<const MTL::Buffer*>(
          t.constants[ref.index].buffer().ptr());
    } else if (ref.cls == R_KEPT) {
      base[r] = static_cast<const MTL::Buffer*>(prog.kept[ref.index].ptr());
    } else if (ref.cls == R_SCRATCH) {
      need.push_back(int(r));
    }
  }
  // Largest first, each on the smallest free pool buffer that holds it and
  // is under twice its size; a new pool buffer otherwise.
  std::sort(need.begin(), need.end(), [&](int a, int b) {
    return prog.refs[a].len > prog.refs[b].len;
  });
  std::vector<uint8_t> used(pool.bufs.size(), 0);
  for (int r : need) {
    const size_t len = prog.refs[r].len;
    int best = -1;
    for (size_t i = 0; i < used.size(); ++i) {
      const size_t have = pool.bufs[i].first;
      if (!used[i] && have >= len && have < 2 * len + 4096 &&
          (best < 0 || have < pool.bufs[best].first)) {
        best = int(i);
      }
    }
    if (best < 0) {
      auto b = mx::allocator::malloc(len);
      const size_t have = static_cast<MTL::Buffer*>(b.ptr())->length();
      pool.bufs.emplace_back(have, b);
      pool.bytes += have;
      g_direct_pool_bytes += have;
      used.push_back(0);
      best = int(pool.bufs.size()) - 1;
    }
    used[best] = 1;
    base[r] = static_cast<const MTL::Buffer*>(pool.bufs[best].second.ptr());
  }
  return pool.assigned.emplace(prog.id, std::move(base)).first->second;
}

// The proxy. One recording at a time per thread.
struct DirectRecorder {
  id real{nullptr};
  int node{0};
  std::vector<DirectRaw> raw;
  std::string bad;
  // Declared by the primitives (kq_param_site), and the buffers they read
  // parameter values from (kq_param_value), with the array's offset.
  std::vector<DirectSite> sites;
  std::vector<std::pair<const void*, int64_t>> param_bufs;
};

thread_local DirectRecorder* g_recorder = nullptr;

template <typename... A>
void direct_fwd(SEL c, A... a) {
  ((void (*)(id, SEL, A...))objc_msgSend)(g_recorder->real, c, a...);
}

DirectRaw& direct_push(uint8_t kind) {
  auto& r = *g_recorder;
  r.raw.emplace_back();
  r.raw.back().kind = kind;
  r.raw.back().node = r.node;
  return r.raw.back();
}

void imp_pso(id, SEL c, id p) {
  direct_push(D_PSO).ptr = p;
  direct_fwd(c, p);
}
void imp_buf(id, SEL c, id b, NS::UInteger off, NS::UInteger idx) {
  auto& o = direct_push(D_BUF);
  o.ptr = b;
  o.a = off;
  o.b = idx;
  direct_fwd(c, b, off, idx);
}
void imp_bytes(id, SEL c, const void* p, NS::UInteger len, NS::UInteger idx) {
  auto& o = direct_push(D_BYTES);
  o.b = idx;
  o.bytes.assign(
      static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + len);
  direct_fwd(c, p, len, idx);
}
void imp_tgmem(id, SEL c, NS::UInteger len, NS::UInteger idx) {
  auto& o = direct_push(D_TGMEM);
  o.a = len;
  o.b = idx;
  direct_fwd(c, len, idx);
}
void imp_barrier(id, SEL c, NS::UInteger scope) {
  direct_push(D_BARRIER).a = scope;
  direct_fwd(c, scope);
}
void imp_dtg(id, SEL c, MTL::Size a, MTL::Size b) {
  auto& o = direct_push(D_DTG);
  o.s0 = a;
  o.s1 = b;
  direct_fwd(c, a, b);
}
void imp_dt(id, SEL c, MTL::Size a, MTL::Size b) {
  auto& o = direct_push(D_DT);
  o.s0 = a;
  o.s1 = b;
  direct_fwd(c, a, b);
}
// Every other message goes to the encoder untouched. A recording that saw
// one is refused, debug labels aside: the program would not replay it.
id imp_target(id, SEL, SEL sel) {
  const char* name = sel_getName(sel);
  static const char* harmless[] = {
      "setLabel:",
      "label",
      "pushDebugGroup:",
      "popDebugGroup",
      "insertDebugSignpost:",
      "device"};
  bool ok = false;
  for (const char* h : harmless) {
    ok |= std::strcmp(name, h) == 0;
  }
  if (!ok && g_recorder->bad.empty()) {
    g_recorder->bad =
        std::string("an encoder call the recording does not model (") + name +
        ")";
    g_recorder->raw.emplace_back();
    g_recorder->raw.back().kind = 255;
    g_recorder->raw.back().node = g_recorder->node;
  }
  return g_recorder->real;
}

Class direct_proxy_class() {
  static Class cls = [] {
    Class c = objc_allocateClassPair(
        objc_getClass("NSObject"), "KQFusedEncoderTrace", 0);
    auto add = [&](const char* sel, IMP imp, const char* types) {
      class_addMethod(c, sel_registerName(sel), imp, types);
    };
    add("setComputePipelineState:", (IMP)imp_pso, "v@:@");
    add("setBuffer:offset:atIndex:", (IMP)imp_buf, "v@:@QQ");
    add("setBytes:length:atIndex:", (IMP)imp_bytes, "v@:^vQQ");
    add("setThreadgroupMemoryLength:atIndex:", (IMP)imp_tgmem, "v@:QQ");
    add("memoryBarrierWithScope:", (IMP)imp_barrier, "v@:Q");
    add("dispatchThreadgroups:threadsPerThreadgroup:",
        (IMP)imp_dtg,
        "v@:{?=QQQ}{?=QQQ}");
    add("dispatchThreads:threadsPerThreadgroup:",
        (IMP)imp_dt,
        "v@:{?=QQQ}{?=QQQ}");
    add("forwardingTargetForSelector:", (IMP)imp_target, "@@::");
    objc_registerClassPair(c);
    return c;
  }();
  return cls;
}

// Primitive types whose Metal calls depend only on what a tape fixes: their
// own state, the shapes, dtypes and layouts of their inputs, whether an
// input may be written in place (fixed by the tape's order), and KQ_*
// levers. None reads array data on the host, keeps state between evals, or
// ends the encoder.
//   - mlx views and copies (shape and layout only; a view makes no call):
//     AsType, Contiguous, Broadcast, Reshape, ExpandDims, Squeeze, Flatten,
//     Unflatten, Transpose, Slice, Split, Concatenate, Full, View.
//   - RoPE: dims, base and scale are state; the offset is an array input
//     the kernel reads as a buffer.
//   - mlx elementwise and reductions, kernel picked by dtype and layout:
//     Add, Multiply, Sigmoid, RMSNorm, and Compiled bodies (a fixed kernel
//     per traced body; scalar inputs are bound as buffers).
//   - Matmul: kernel and tile picked by shape and layout.
//   - CustomKernel (mx.fast.metal_kernel): source, grid and group are
//     state of the primitive.
//   - mlx_kquant: KQuantMatmul and KQuantMatmulMulti (route by codec, shape,
//     levers, and the 16-byte alignment of x and w, which the per call
//     input check holds fixed), KQuantHcLowrankMix, KQuantSkinnyMV,
//     KQuantMoERouterTopK, KQuantMoEGLUShexpKQ, KQuantGatherQMVMixKQ (shape,
//     codec and levers; the expert ids stay on the GPU), KQuantMoeStoreGLU
//     and KQuantMoeStoreDown (the same; the slot table is read on the GPU),
//     KQDsaIndexerScoreDecode, KQDsaTopKIndices, KQuantSDPAPagedRows and
//     KQuantKVAppend (shape, layout, levers, their own state, and values of a
//     parameter block they declare with kq_param_site).
bool direct_listed(const std::string& name) {
  static const std::unordered_set<std::string> listed = {
      "AsType",
      "Contiguous",
      "Broadcast",
      "Reshape",
      "ExpandDims",
      "Squeeze",
      "StopGradient",
      "Flatten",
      "Unflatten",
      "Transpose",
      "Slice",
      "Split",
      "Concatenate",
      "Full",
      "Add",
      "Multiply",
      "Sigmoid",
      "RMSNorm",
      "Matmul",
      "CustomKernel",
      "View",
      "RoPE",
      "LessEqual",
      "ScaledDotProductAttention",
      "KQuantMatmul",
      "KQuantMatmulMulti",
      "KQuantHcLowrankMix",
      "KQuantSkinnyMV",
      "KQuantMoERouterTopK",
      "KQuantMoEGLUKQ",
      "KQuantMoEGLUShexpKQ",
      "KQuantGatherQMVMixKQ",
      "KQuantGatherQMVMixNSKQ",
      "KQuantMoeStoreGLU",
      "KQuantMoeStoreDown",
      "KQDsaIndexerScoreDecode",
      "KQDsaTopKIndices",
      "KQuantSDPAPagedRows",
      "KQuantKVAppend",
      "KQuantSDPAGQA"};
  // mlx names a View with its dtype ("View int32").
  return listed.count(name) != 0 || name.rfind("Compiled", 0) == 0 ||
      name.rfind("View ", 0) == 0;
}

// Back to nothing learned, or to refused where the tape may not learn.
void direct_reset(DirectState& s) {
  s.first.reset();
  s.prog.reset();
  s.reason.clear();
  s.prims.clear();
  s.state = 0;
  if (!s.unlisted.empty() && kq_env_flag("KQ_FUSED_DIRECT_ANY", 0) == 0) {
    s.state = 3;
    s.reason = "a primitive type is not on the direct list";
    s.prims = s.unlisted;
  }
}

bool direct_same_layout(const DirectInput& a, const DirectInput& b) {
  return a.strides == b.strides && a.contiguous == b.contiguous &&
      a.row_contiguous == b.row_contiguous &&
      a.col_contiguous == b.col_contiguous && a.data_size == b.data_size &&
      a.group == b.group && ((a.off ^ b.off) & 63) == 0;
}

// One eval of the tape under the proxy, written down with buffers by role.
std::unique_ptr<DirectTrace> direct_record(
    const std::shared_ptr<const FusedTape>& tape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const FusedTape& t = *tape;
  auto& ce = mx::metal::get_command_encoder(t.stream);
  DirectRecorder rec;
  rec.real = (id)ce.get_command_encoder();
  NS::SharedPtr<MTL::ComputeCommandEncoder> real_sp = ce.encoder_;
  id proxy = ((id(*)(id, SEL))objc_msgSend)(
      ((id(*)(Class, SEL))objc_msgSend)(
          direct_proxy_class(), sel_registerName("alloc")),
      sel_registerName("init"));
  // The outputs mlx registers during the run are this tape's: a buffer
  // bound and never registered was filled on the host.
  std::unordered_set<const void*> before;
  std::swap(before, ce.all_outputs_);
  const size_t sizes0 = ce.buffer_sizes_;
  DirectRecorder* outer = g_recorder;
  g_recorder = &rec;
  ce.encoder_ =
      NS::TransferPtr(reinterpret_cast<MTL::ComputeCommandEncoder*>(proxy));
  auto restore = [&] {
    const bool swapped = ce.encoder_.get() != real_sp.get();
    ce.encoder_ = real_sp;
    g_recorder = outer;
    return swapped;
  };
  try {
    run_tape<true>(tape, inputs, outputs, &rec.node);
  } catch (...) {
    restore();
    ce.all_outputs_.insert(before.begin(), before.end());
    throw;
  }
  auto tr = std::make_unique<DirectTrace>();
  if (ce.encoder_.get() !=
      reinterpret_cast<MTL::ComputeCommandEncoder*>(proxy)) {
    // A primitive ended the encoder: mlx dropped the proxy and opened an
    // encoder of its own, which stays.
    g_recorder = outer;
    tr->refuse = "a primitive ended the compute encoder";
    tr->refuse_node = rec.node;
    ce.all_outputs_.insert(before.begin(), before.end());
    return tr;
  }
  restore();
  std::unordered_set<const void*> written = ce.all_outputs_;
  ce.all_outputs_.insert(before.begin(), before.end());
  tr->sizes = ce.buffer_sizes_ - sizes0;
  if (!rec.bad.empty()) {
    tr->refuse = rec.bad;
    for (auto& r : rec.raw) {
      if (r.kind == 255) {
        tr->refuse_node = r.node;
      }
    }
    return tr;
  }

  std::unordered_map<const void*, int> input_of;
  std::unordered_map<const void*, int> const_of;
  for (size_t i = 0; i < t.constants.size(); ++i) {
    const_of.emplace(t.constants[i].buffer().ptr(), int(i));
  }
  for (int i = 0; i < t.n_inputs; ++i) {
    auto& a = inputs[i];
    const void* p = a.buffer().ptr();
    if (const_of.count(p)) {
      tr->refuse = "an input shares its buffer with an array the record holds";
      return tr;
    }
    int g = input_of.emplace(p, i).first->second;
    auto& f = a.flags();
    tr->ins.push_back(
        {a.strides(),
         f.contiguous,
         f.row_contiguous,
         f.col_contiguous,
         a.data_size(),
         a.offset(),
         g});
  }
  std::unordered_map<const void*, int> ref_of;
  auto ref = [&](const void* p, int node) -> int {
    if (auto it = ref_of.find(p); it != ref_of.end()) {
      return it->second;
    }
    int r = int(tr->refs.size());
    ref_of.emplace(p, r);
    if (auto it = input_of.find(p); it != input_of.end()) {
      tr->refs.push_back({R_INPUT, it->second, 0, node, written.count(p) != 0});
    } else if (auto ic = const_of.find(p); ic != const_of.end()) {
      tr->refs.push_back({R_CONST, ic->second, 0, node, written.count(p) != 0});
    } else {
      const size_t len = static_cast<const MTL::Buffer*>(p)->length();
      tr->refs.push_back({R_SCRATCH, -1, len, node});
    }
    return r;
  };
  std::unordered_map<int, int> first_user;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& o = outputs[i];
    const void* p = o.buffer().ptr();
    auto& f = o.flags();
    DirectOut d{
        0,
        0,
        -1,
        0,
        o.strides(),
        f.contiguous,
        f.row_contiguous,
        f.col_contiguous,
        o.data_size()};
    const int64_t isz = int64_t(o.itemsize());
    if (auto it = input_of.find(p); it != input_of.end()) {
      d.kind = 2;
      d.other = it->second;
      d.off = (o.offset() - inputs[it->second].offset()) / isz;
    } else if (auto ic = const_of.find(p); ic != const_of.end()) {
      d.kind = 3;
      d.other = ic->second;
      d.off = (o.offset() - t.constants[ic->second].offset()) / isz;
    } else {
      int r = ref(p, -1);
      d.ref = r;
      if (auto fu = first_user.find(r); fu != first_user.end()) {
        d.kind = 1;
        d.other = fu->second;
        d.off = (o.offset() - outputs[fu->second].offset()) / isz;
      } else {
        if (!written.count(p)) {
          tr->refuse = "an output is not written by a dispatch";
          return tr;
        }
        d.kind = 0;
        d.other = int(tr->fresh.size());
        d.off = o.offset();
        first_user.emplace(r, int(i));
        tr->refs[r].cls = R_OUTPUT;
        tr->refs[r].index = int(tr->fresh.size());
        tr->fresh.push_back(tr->refs[r].len);
      }
    }
    tr->outs.push_back(std::move(d));
  }
  // A primitive that takes values from a parameter block reads them on the
  // host: the program can follow them only when the block is the input the
  // tape declared, or an array the tape holds (fixed for good).
  const int32_t* pvals = nullptr;
  for (auto [p, poff] : rec.param_bufs) {
    auto it = input_of.find(p);
    if (it != input_of.end() && it->second != t.param_in) {
      tr->refuse =
          "a primitive reads parameters from an input the record "
          "does not declare as its parameter block";
      return tr;
    }
    // A slot counts from the start of the declared block: a primitive that
    // reads a view of it at another offset would be patched from the
    // wrong slot.
    if (it != input_of.end() && poff != inputs[t.param_in].offset()) {
      tr->refuse =
          "a primitive reads parameters from a view of the block at "
          "another offset";
      return tr;
    }
    if (it == input_of.end() && !const_of.count(p)) {
      tr->refuse =
          "a primitive reads parameters from an array the record makes";
      return tr;
    }
    if (it != input_of.end()) {
      pvals = inputs[t.param_in].data<int32_t>();
    }
  }
  for (auto& st : rec.sites) {
    if (pvals == nullptr) {
      break; // every block read is held by the tape: nothing to patch
    }
    auto& r = rec.raw[st.op];
    int32_t have = 0;
    if (st.field < 0 || size_t(st.field) + 4 > r.bytes.size() || st.slot < 0 ||
        st.slot >= int(t.param_lo.size())) {
      tr->refuse = "a parameter site is outside its bytes or its block";
      tr->refuse_node = r.node;
      return tr;
    }
    std::memcpy(&have, r.bytes.data() + st.field, 4);
    if (have != int32_t(int64_t(pvals[st.slot]) * st.mul + st.add)) {
      tr->refuse = "a parameter site does not hold its value";
      tr->refuse_node = r.node;
      return tr;
    }
    // A replay writes any value of the slot's range with no eval and no
    // check of the primitive's own: the range has to be one it takes.
    if (t.param_lo[st.slot] < st.lo || t.param_hi[st.slot] > st.hi) {
      tr->refuse = "a parameter range runs past the values its primitive takes";
      tr->refuse_node = r.node;
      return tr;
    }
    tr->sites.push_back(st);
  }
  for (auto& r : rec.raw) {
    DirectOp d{r.kind};
    d.node = r.node;
    switch (r.kind) {
      case D_PSO:
        d.ptr = r.ptr;
        break;
      case D_BUF: {
        if (r.ptr == nullptr) {
          tr->refuse = "a primitive binds no buffer";
          tr->refuse_node = r.node;
          return tr;
        }
        d.ref = ref(r.ptr, r.node);
        d.idx = uint8_t(r.b);
        d.off = int64_t(r.a);
        auto& rf = tr->refs[d.ref];
        if (rf.cls == R_INPUT) {
          d.off -= inputs[rf.index].offset();
        }
        break;
      }
      case D_BYTES:
        d.idx = uint8_t(r.b);
        d.off = int64_t(tr->blob.size());
        d.len = r.bytes.size();
        tr->blob.insert(tr->blob.end(), r.bytes.begin(), r.bytes.end());
        break;
      case D_TGMEM:
        d.len = r.a;
        d.idx = uint8_t(r.b);
        break;
      case D_BARRIER:
        d.len = r.a;
        tr->n_barrier++;
        break;
      default:
        d.s0 = r.s0;
        d.s1 = r.s1;
        tr->n_dispatch++;
        break;
    }
    tr->ops.push_back(d);
  }
  if (kq_env_flag("KQ_FUSED_DIRECT_SELFTEST", 0) != 0 &&
      kq_env_flag("KQ_FUSED_DIRECT_CHECK", 0) != 0 && !tr->blob.empty()) {
    // Test lever of the check: makes every recording differ from the one
    // before it in its first bytes, as a primitive that passed a value of
    // its input data to a kernel would. Without the check a recording
    // becomes a program, which must not carry the changed byte.
    static std::atomic<uint8_t> salt{0};
    tr->blob[0] ^= uint8_t(1 + (salt++ & 0x7f));
  }
  if (int(tr->refs.size()) > kDirectMaxRefs ||
      int(tr->fresh.size()) > kDirectMaxOutputs) {
    tr->refuse = "more buffers than a program holds";
    return tr;
  }
  // A buffer between nodes that no dispatch of the run wrote was filled on
  // the host: the program keeps its content, and the second recording has
  // to show the same bytes.
  for (auto& [p, r] : ref_of) {
    auto& rf = tr->refs[r];
    if (rf.cls != R_SCRATCH || written.count(p)) {
      continue;
    }
    if (rf.len > kDirectMaxKept) {
      tr->refuse = "a primitive binds a large buffer it filled on the host";
      tr->refuse_node = rf.node;
      return tr;
    }
    rf.cls = R_KEPT;
  }
  for (size_t r = 0; r < tr->refs.size(); ++r) {
    if (tr->refs[r].cls == R_KEPT) {
      tr->refs[r].index = int(tr->kept.size());
      tr->kept.emplace_back();
    }
  }
  for (auto& [p, r] : ref_of) {
    auto& rf = tr->refs[r];
    if (rf.cls == R_KEPT) {
      auto* b = static_cast<const MTL::Buffer*>(p);
      auto* c =
          static_cast<const uint8_t*>(const_cast<MTL::Buffer*>(b)->contents());
      tr->kept[rf.index].assign(c, c + rf.len);
    }
  }
  return tr;
}

// Index of the first op that differs, -1 when the recordings are equal,
// -2 when they differ outside the ops.
int direct_diff(
    const DirectTrace& a,
    const DirectTrace& b,
    std::vector<int>& nodes) {
  auto same_size = [](const MTL::Size& x, const MTL::Size& y) {
    return x.width == y.width && x.height == y.height && x.depth == y.depth;
  };
  // Bytes at a parameter site follow the block, not the tape.
  auto masked = [](const DirectTrace& t) {
    std::vector<uint8_t> blob = t.blob;
    for (auto& st : t.sites) {
      std::memset(blob.data() + t.ops[st.op].off + st.field, 0, 4);
    }
    return blob;
  };
  const std::vector<uint8_t> blob_a = masked(a);
  const std::vector<uint8_t> blob_b = masked(b);
  int first = -1;
  const size_t n = std::min(a.ops.size(), b.ops.size());
  for (size_t i = 0; i < n; ++i) {
    auto& x = a.ops[i];
    auto& y = b.ops[i];
    bool same = x.kind == y.kind && x.idx == y.idx && x.ref == y.ref &&
        x.node == y.node && x.len == y.len && x.ptr == y.ptr &&
        same_size(x.s0, y.s0) && same_size(x.s1, y.s1);
    if (same && x.kind == D_BYTES) {
      same =
          std::memcmp(blob_a.data() + x.off, blob_b.data() + y.off, x.len) == 0;
    } else if (same) {
      same = x.off == y.off;
    }
    if (!same) {
      if (first < 0) {
        first = int(i);
      }
      nodes.push_back(x.node);
      if (x.node != y.node) {
        break;
      }
    }
  }
  if (first < 0 && a.ops.size() != b.ops.size()) {
    first = int(n);
    nodes.push_back(n < a.ops.size() ? a.ops[n].node : b.ops[n].node);
  }
  if (first >= 0) {
    return first;
  }
  bool same = a.refs.size() == b.refs.size() &&
      a.outs.size() == b.outs.size() && a.fresh.size() == b.fresh.size() &&
      a.sites.size() == b.sites.size();
  for (size_t i = 0; same && i < a.sites.size(); ++i) {
    auto& x = a.sites[i];
    auto& y = b.sites[i];
    same = x.op == y.op && x.field == y.field && x.slot == y.slot &&
        x.mul == y.mul && x.add == y.add && x.lo == y.lo && x.hi == y.hi;
  }
  for (size_t i = 0; same && i < a.refs.size(); ++i) {
    auto& x = a.refs[i];
    auto& y = b.refs[i];
    same = x.cls == y.cls && x.index == y.index && x.written == y.written &&
        (x.cls != R_KEPT || x.len == y.len);
    if (same && x.cls == R_KEPT && a.kept[x.index] != b.kept[y.index]) {
      same = false;
      nodes.push_back(x.node);
    }
  }
  for (size_t i = 0; same && i < a.outs.size(); ++i) {
    auto& x = a.outs[i];
    auto& y = b.outs[i];
    same = x.kind == y.kind && x.other == y.other && x.ref == y.ref &&
        x.off == y.off && x.strides == y.strides &&
        x.contiguous == y.contiguous && x.row_contiguous == y.row_contiguous &&
        x.col_contiguous == y.col_contiguous && x.data_size == y.data_size;
  }
  return same ? -1 : -2;
}

std::atomic<uint64_t> g_direct_id{0};

std::shared_ptr<const DirectProgram> direct_compile(
    const FusedTape& t,
    std::shared_ptr<const DirectTrace> trp) {
  const DirectTrace& tr = *trp;
  auto p = std::make_shared<DirectProgram>();
  p->id = ++g_direct_id;
  p->trace = trp;
  p->refs = tr.refs;
  p->outs = tr.outs;
  p->ins = tr.ins;
  p->n_dispatch = tr.n_dispatch;
  p->n_calls_recorded = int(tr.ops.size());
  p->n_barrier_recorded = tr.n_barrier;
  p->sizes = tr.sizes;
  for (size_t r = 0; r < tr.refs.size(); ++r) {
    auto& rf = tr.refs[r];
    if (rf.cls == R_INPUT) {
      p->in_refs.emplace_back(int(r), rf.index);
      if (rf.written) {
        p->in_written.push_back(rf.index);
      }
    } else if (rf.cls == R_OUTPUT) {
      p->fresh.emplace_back(int(r), rf.len);
    } else if (rf.cls == R_SCRATCH) {
      p->scratch_bytes += rf.len;
    } else if (rf.cls == R_CONST) {
      p->const_ptrs.push_back(t.constants[rf.index].buffer().ptr());
      if (rf.written) {
        p->const_written.push_back(t.constants[rf.index].buffer().ptr());
      }
    }
  }
  std::sort(p->fresh.begin(), p->fresh.end(), [&](auto& a, auto& b) {
    return tr.refs[a.first].index < tr.refs[b.first].index;
  });
  for (auto& c : tr.kept) {
    auto b = mx::allocator::malloc(c.size());
    std::memcpy(
        static_cast<MTL::Buffer*>(b.ptr())->contents(), c.data(), c.size());
    p->kept.push_back(b);
    p->kept_bytes += c.size();
  }
  // A pipeline, a bind or bytes already in place from an earlier call of
  // the same replay are not sent again, and a barrier with no dispatch
  // since the last one is dropped: the replay opens with a barrier.
  const void* pso = nullptr;
  struct Bound {
    int kind{0};
    int ref{0};
    int64_t off{0};
    uint64_t len{0};
  };
  std::unordered_map<int, Bound> bound;
  bool dispatched = false;
  p->n_barrier = 1;
  std::unordered_map<int, std::vector<const DirectSite*>> sites_of;
  for (auto& st : tr.sites) {
    sites_of[st.op].push_back(&st);
  }
  for (size_t oi = 0; oi < tr.ops.size(); ++oi) {
    auto& o = tr.ops[oi];
    DirectOp d = o;
    if (auto si = sites_of.find(int(oi)); si != sites_of.end()) {
      // Bytes with a parameter in them are sent on every replay, and what
      // is then in place is not known to the calls after.
      bound[o.idx] = {};
      d.off = int64_t(p->blob.size());
      p->blob.insert(
          p->blob.end(),
          tr.blob.begin() + o.off,
          tr.blob.begin() + o.off + o.len);
      for (auto* st : si->second) {
        p->sites.push_back(
            {int(d.off) + st->field,
             st->field,
             st->slot,
             st->mul,
             st->add,
             st->lo,
             st->hi});
      }
      p->ops.push_back(d);
      continue;
    }
    switch (o.kind) {
      case D_PSO:
        if (pso == o.ptr) {
          continue;
        }
        pso = o.ptr;
        break;
      case D_BUF: {
        auto& b = bound[o.idx];
        if (b.kind == 1 && b.ref == o.ref && b.off == o.off) {
          continue;
        }
        b = {1, o.ref, o.off, 0};
        break;
      }
      case D_BYTES: {
        auto& b = bound[o.idx];
        if (b.kind == 2 && b.len == o.len &&
            std::memcmp(
                tr.blob.data() + b.off, tr.blob.data() + o.off, o.len) == 0) {
          continue;
        }
        b = {2, 0, o.off, o.len};
        d.off = int64_t(p->blob.size());
        p->blob.insert(
            p->blob.end(),
            tr.blob.begin() + o.off,
            tr.blob.begin() + o.off + o.len);
        break;
      }
      case D_TGMEM:
        break;
      case D_BARRIER:
        if (!dispatched) {
          continue;
        }
        dispatched = false;
        p->n_barrier++;
        break;
      default:
        dispatched = true;
        break;
    }
    p->ops.push_back(d);
  }
  return p;
}

// True when the inputs of this call are laid out as the recorded ones.
bool direct_inputs_match(
    const DirectProgram& p,
    const std::vector<mx::array>& inputs) {
  for (size_t i = 0; i < p.ins.size(); ++i) {
    auto& a = inputs[i];
    auto& r = p.ins[i];
    auto& f = a.flags();
    if (((a.offset() ^ r.off) & 63) != 0 || a.data_size() != r.data_size ||
        f.contiguous != r.contiguous || f.row_contiguous != r.row_contiguous ||
        f.col_contiguous != r.col_contiguous || a.strides() != r.strides) {
      return false;
    }
    if (r.group != int(i)) {
      auto& g = inputs[r.group];
      if (g.buffer().ptr() != a.buffer().ptr() ||
          a.offset() - g.offset() != r.off - p.ins[r.group].off) {
        return false;
      }
    }
  }
  return true;
}

// The calls of a program on `enc`, buffers from `cur` and `adj`.
inline void direct_emit(
    MTL::ComputeCommandEncoder* enc,
    const DirectProgram& d,
    const uint8_t* blob,
    const MTL::Buffer* const* cur,
    const int64_t* adj) {
  for (const DirectOp& o : d.ops) {
    switch (o.kind) {
      case D_PSO:
        enc->setComputePipelineState(
            static_cast<const MTL::ComputePipelineState*>(o.ptr));
        break;
      case D_BUF:
        enc->setBuffer(cur[o.ref], NS::UInteger(o.off + adj[o.ref]), o.idx);
        break;
      case D_BYTES:
        enc->setBytes(blob + o.off, o.len, o.idx);
        break;
      case D_TGMEM:
        enc->setThreadgroupMemoryLength(o.len, o.idx);
        break;
      case D_BARRIER:
        enc->memoryBarrier(MTL::BarrierScope(o.len));
        break;
      case D_DTG:
        enc->dispatchThreadgroups(o.s0, o.s1);
        break;
      default:
        enc->dispatchThreads(o.s0, o.s1);
        break;
    }
  }
}

// The program's bytes with the values of the block `p` at its sites.
inline void
direct_patch(const DirectProgram& d, const int32_t* p, uint8_t* blob) {
  for (auto& st : d.sites) {
    const int32_t v = int32_t(int64_t(p[st.slot]) * st.mul + st.add);
    std::memcpy(blob + st.op, &v, 4);
  }
}

void direct_replay(
    const std::shared_ptr<const FusedTape>& tape,
    const std::shared_ptr<const DirectProgram>& prog,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const FusedTape& t = *tape;
  const DirectProgram& d = *prog;
  auto& ce = mx::metal::get_command_encoder(t.stream);
  auto* enc = ce.get_command_encoder();
  auto& pool = direct_pool(t.stream.index);
  const auto& base = direct_assign(*pool, d, t);
  const int nref = int(base.size());
  const MTL::Buffer* cur[kDirectMaxRefs];
  int64_t adj[kDirectMaxRefs];
  std::memcpy(cur, base.data(), nref * sizeof(cur[0]));
  std::memset(adj, 0, nref * sizeof(adj[0]));
  for (auto& [r, i] : d.in_refs) {
    auto& a = inputs[i];
    const void* p = a.buffer().ptr();
    cur[r] = static_cast<const MTL::Buffer*>(p);
    // Binds into an input are recorded relative to the input's own offset.
    adj[r] = a.offset();
    ce.all_inputs_.insert(p);
  }
  if (!d.settled.load(std::memory_order_relaxed)) {
    for (const void* p : d.const_ptrs) {
      ce.all_inputs_.insert(p);
    }
  }
  // A constant the record writes: the replay waits for an earlier writer
  // in another encoder, and a later encoder waits for the replay.
  for (const void* p : d.const_written) {
    ce.all_inputs_.insert(p);
    ce.all_outputs_.insert(p);
  }
  void* fresh[kDirectMaxOutputs];
  for (size_t j = 0; j < d.fresh.size(); ++j) {
    fresh[j] = mx::allocator::malloc(d.fresh[j].second).ptr();
    cur[d.fresh[j].first] = static_cast<const MTL::Buffer*>(fresh[j]);
    ce.all_outputs_.insert(fresh[j]);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& o = d.outs[i];
    mx::array::Flags f{o.contiguous, o.row_contiguous, o.col_contiguous};
    switch (o.kind) {
      case 0:
        outputs[i].set_data(
            mx::allocator::Buffer(fresh[o.other]),
            o.data_size,
            o.strides,
            f,
            o.off);
        break;
      case 1:
        outputs[i].copy_shared_buffer(
            outputs[o.other], o.strides, f, o.data_size, o.off);
        break;
      case 2:
        outputs[i].copy_shared_buffer(
            inputs[o.other], o.strides, f, o.data_size, o.off);
        break;
      default:
        outputs[i].copy_shared_buffer(
            t.constants[o.other], o.strides, f, o.data_size, o.off);
        break;
    }
  }
  // One pool buffer stands for the pool in mlx's ordering between
  // encoders: an encoder that replays waits for the last one that did.
  if (!pool->bufs.empty()) {
    const void* rep = pool->bufs[0].second.ptr();
    ce.all_inputs_.insert(rep);
    ce.all_outputs_.insert(rep);
  }
  // The barrier orders the replay after whatever wrote its inputs in this
  // encoder and after the last replay, which used the pool.
  enc->memoryBarrier(MTL::BarrierScopeBuffers);
  ce.needs_barrier_ = false;
  ce.prev_outputs_.clear();
  ce.next_outputs_.clear();
  const uint8_t* blob = d.blob.data();
  if (!d.sites.empty()) {
    thread_local std::vector<uint8_t> patched;
    patched.assign(d.blob.begin(), d.blob.end());
    direct_patch(d, inputs[t.param_in].data<int32_t>(), patched.data());
    blob = patched.data();
  }
  direct_emit(enc, d, blob, cur, adj);
  for (size_t j = 0; j < d.fresh.size(); ++j) {
    ce.prev_outputs_.insert(
        static_cast<MTL::Resource*>(static_cast<MTL::Buffer*>(fresh[j])));
  }
  // An input the record wrote in place is an output to mlx: a later
  // encoder gets its fence.
  for (int i : d.in_written) {
    ce.all_outputs_.insert(inputs[i].buffer().ptr());
  }
  // The next dispatch in this encoder follows a barrier, as after a kernel
  // whose outputs it reads: whatever it reads or writes (an output, an
  // input or constant the record wrote, a buffer the record read), the
  // replay is over first.
  ce.needs_barrier_ = true;
  ce.buffer_ops_ += d.n_dispatch;
  ce.buffer_sizes_ += d.sizes;
  ce.get_command_buffer()->addCompletedHandler(
      [prog, tape, pool](MTL::CommandBuffer*) {});
  d.replays.fetch_add(1, std::memory_order_relaxed);
}

void direct_refuse(
    const FusedTape& t,
    DirectState& s,
    const std::string& why,
    const std::vector<int>& nodes) {
  s.state = 3;
  s.reason = why;
  s.first.reset();
  for (int n : nodes) {
    if (n >= 0 && n < int(t.nodes.size())) {
      std::string name = t.nodes[n].prim->name();
      if (std::find(s.prims.begin(), s.prims.end(), name) == s.prims.end()) {
        s.prims.push_back(name);
      }
    }
  }
}

void run_direct(
    const std::shared_ptr<const FusedTape>& tape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const FusedTape& t = *tape;
  DirectState& s = *t.dstate;
  std::shared_ptr<const DirectProgram> prog;
  int state;
  {
    std::lock_guard lk(s.mtx);
    const unsigned epoch = kq_env_epoch();
    if (s.epoch != epoch) {
      // A lever may have changed: whatever was learned is dropped.
      s.epoch = epoch;
      direct_reset(s);
    }
    state = kq_env_cache_on() ? s.state : 3;
    prog = s.prog;
    if (state == 3 || (state == 2 && !direct_inputs_match(*prog, inputs))) {
      s.plain++;
      state = 3;
    }
  }
  if (state == 2) {
    direct_replay(tape, prog, inputs, outputs);
    return;
  }
  if (state == 3) {
    run_tape<true>(tape, inputs, outputs);
    return;
  }
  auto tr = direct_record(tape, inputs, outputs);
  std::lock_guard lk(s.mtx);
  if (s.state >= 2 || s.epoch != kq_env_epoch()) {
    return;
  }
  if (!tr->refuse.empty()) {
    direct_refuse(t, s, tr->refuse, {tr->refuse_node});
    return;
  }
  if (kq_env_flag("KQ_FUSED_DIRECT_CHECK", 0) != 0) {
    // Debug lever: a second recording must show the same calls.
    bool comparable =
        s.first != nullptr && s.first->ins.size() == tr->ins.size();
    for (size_t i = 0; comparable && i < tr->ins.size(); ++i) {
      comparable = direct_same_layout(s.first->ins[i], tr->ins[i]);
    }
    if (!comparable) {
      s.first = std::move(tr);
      s.state = 1;
      return;
    }
    std::vector<int> nodes;
    const int at = direct_diff(*s.first, *tr, nodes);
    if (at != -1) {
      direct_refuse(
          t,
          s,
          at == -2 ? "two recordings differ in their buffers or outputs"
                   : "two recordings differ in their Metal calls",
          nodes);
      return;
    }
  }
  auto p = direct_compile(t, std::shared_ptr<const DirectTrace>(std::move(tr)));
  s.prog = p;
  s.state = 2;
  s.first.reset();
  // The constants the recording bound are complete once its command
  // buffer is.
  mx::metal::get_command_encoder(t.stream)
      .get_command_buffer()
      ->addCompletedHandler(
          [p](MTL::CommandBuffer*) { p->settled.store(true); });
}

#endif // _METAL_

class KQuantFusedGraph : public mx::Primitive {
 public:
  KQuantFusedGraph(mx::Stream stream, std::shared_ptr<const FusedTape> tape)
      : mx::Primitive(stream), tape_(std::move(tape)) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    run_tape<false>(tape_, inputs, outputs);
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
#ifdef _METAL_
    if (tape_->dstate) {
      run_direct(tape_, inputs, outputs);
      return;
    }
#endif
    run_tape<true>(tape_, inputs, outputs);
  }
  const char* name() const override {
    return "KQuantFusedGraph";
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const KQuantFusedGraph*>(&other);
    return o != nullptr && o->tape_ == tape_;
  }

 private:
  std::shared_ptr<const FusedTape> tape_;
};

} // namespace

std::shared_ptr<const FusedTape> fuse_graph(
    const std::vector<mx::array>& inputs,
    const std::vector<mx::array>& outputs,
    bool serial,
    bool direct,
    int level,
    int params,
    const std::vector<FusedParamRange>& param_ranges) {
  const char* op = "[mlx_kquant.fuse_graph]";
  if (outputs.empty()) {
    throw std::invalid_argument(std::string(op) + " needs an output.");
  }
  auto t = std::make_shared<FusedTape>();
  if (serial && direct) {
    throw std::invalid_argument(
        std::string(op) + " takes serial or direct, not both.");
  }
  t->serial = serial;
  t->direct = direct;
  if (params >= 0) {
    if (params >= int(inputs.size())) {
      throw std::invalid_argument(std::string(op) + " params names no input.");
    }
    const mx::array& pa = inputs[params];
    if (pa.dtype() != mx::int32 || pa.ndim() != 1 || !pa.is_available() ||
        !pa.flags().row_contiguous) {
      throw std::invalid_argument(
          std::string(op) +
          " the parameter block must be an evaluated, "
          "contiguous int32 vector.");
    }
    t->param_in = params;
    const int32_t* pv = pa.data<int32_t>();
    t->param_lo.assign(pv, pv + pa.size());
    t->param_hi.assign(pv, pv + pa.size());
    for (auto& r : param_ranges) {
      if (r.slot < 0 || r.slot >= int(pa.size()) || r.lo > r.hi ||
          pv[r.slot] < r.lo || pv[r.slot] > r.hi) {
        throw std::invalid_argument(
            std::string(op) +
            " a parameter range names no slot or leaves "
            "out the recorded value.");
      }
      t->param_lo[r.slot] = r.lo;
      t->param_hi[r.slot] = r.hi;
    }
  } else if (!param_ranges.empty()) {
    throw std::invalid_argument(
        std::string(op) + " param_ranges needs params.");
  }
  std::unordered_map<std::uintptr_t, int> slot;
  for (auto& in : inputs) {
    if (!slot.emplace(in.id(), int(slot.size())).second) {
      throw std::invalid_argument(
          std::string(op) + " an input is listed twice.");
    }
    t->in_shapes.push_back(in.shape());
    t->in_dtypes.push_back(in.dtype());
  }
  t->n_inputs = int(inputs.size());
  bool have_stream = false;
  // Slots: the inputs, then the constants in the order met, then node
  // outputs. Constants are numbered after the walk, so node records hold a
  // negative placeholder for them until then.
  std::vector<mx::array> constants;
  std::unordered_map<std::uintptr_t, int> const_idx;
  std::vector<std::pair<mx::array, size_t>> stack;
  int next_node_slot = 0;
  std::unordered_map<std::uintptr_t, int> node_slot;
  auto leaf = [&](const mx::array& a) {
    return !a.has_primitive() || a.status() != mx::array::Status::unscheduled;
  };
  auto ref = [&](const mx::array& a) -> int {
    // Inputs: [0, n_inputs). Constants: -(1 + index). Node outputs:
    // 1 << 30 plus index; both are renumbered below.
    if (auto it = slot.find(a.id()); it != slot.end()) {
      return it->second;
    }
    if (auto it = node_slot.find(a.id()); it != node_slot.end()) {
      return (1 << 30) + it->second;
    }
    auto [it, fresh] = const_idx.emplace(a.id(), int(constants.size()));
    if (fresh) {
      constants.push_back(a);
    }
    return -(1 + it->second);
  };
  auto known = [&](const mx::array& a) {
    return slot.count(a.id()) || node_slot.count(a.id()) || leaf(a);
  };
  for (auto& o : outputs) {
    if (known(o)) {
      continue;
    }
    stack.emplace_back(o, 0);
    while (!stack.empty()) {
      auto& [a, idx] = stack.back();
      if (node_slot.count(a.id())) {
        stack.pop_back();
        continue;
      }
      if (idx < a.inputs().size()) {
        const mx::array& in = a.inputs()[idx++];
        if (!known(in)) {
          stack.emplace_back(in, 0);
        }
        continue;
      }
      auto& prim = a.primitive();
      if (!have_stream) {
        t->stream = prim.stream();
        have_stream = true;
      } else if (prim.stream() != t->stream) {
        throw std::invalid_argument(
            std::string(op) + " the graph runs on more than one stream.");
      }
      FusedNode n;
      n.prim = a.primitive_ptr();
      for (auto& in : a.inputs()) {
        n.in.push_back(ref(in));
      }
      // A block read on the host has to hold its values when the node is
      // encoded. The params input is checked at each fused_call and a
      // constant was evaluated at record; any other input may be an array
      // a kernel of the same eval has not written yet.
      if (const int pin = host_param_input(*n.prim, int(n.in.size()));
          pin >= 0 && n.in[pin] >= 0 && n.in[pin] != params) {
        throw std::invalid_argument(
            std::string(op) + " " + n.prim->name() +
            " reads its parameter block from an input other than params; "
            "name that input with params.");
      }
      for (auto& out : a.outputs()) {
        node_slot.emplace(out.id(), next_node_slot);
        n.out.push_back((1 << 30) + next_node_slot++);
        n.shapes.push_back(out.shape());
        n.dtypes.push_back(out.dtype());
      }
      t->nodes.push_back(std::move(n));
      stack.pop_back();
    }
  }
  if (t->nodes.empty()) {
    throw std::invalid_argument(
        std::string(op) + " no output has work to record.");
  }
  // An output the walk skipped (an evaluated array, not an input) is a
  // constant too: numbered now, before the node slots start after the
  // constants.
  for (auto& o : outputs) {
    ref(o);
  }
  const int nc = int(constants.size());
  // Node output slots follow the tape's order (run_tape appends what a
  // node makes), so a reordered tape renumbers them.
  std::vector<int> renum(next_node_slot);
  for (int i = 0; i < next_node_slot; ++i) {
    renum[i] = i;
  }
  if (level == 2) {
    // Barrier group order: mlx puts a memory barrier before a dispatch
    // that reads what a dispatch wrote since the last barrier. Take every
    // ready node that needs none (a kernel reading nothing written since
    // the barrier, or a node that makes a view and dispatches nothing),
    // and only when none is left start the next group. Fewest groups a
    // greedy pass finds; same nodes, same operands. A primitive not on
    // the view list counts as a kernel, which costs grouping, never order.
    static const std::unordered_set<std::string> views = {
        "Reshape",
        "Transpose",
        "ExpandDims",
        "Squeeze",
        "Flatten",
        "Unflatten",
        "Broadcast",
        "BroadcastAxes",
        "StopGradient",
        "Split"};
    const size_t nn = t->nodes.size();
    std::vector<uint8_t> is_view(nn, 0), done(nn, 0);
    for (size_t k = 0; k < nn; ++k) {
      is_view[k] = views.count(t->nodes[k].prim->name()) != 0;
    }
    // Per node output slot: made yet, written since the last barrier.
    std::vector<uint8_t> made(next_node_slot, 0), dirty(next_node_slot, 0);
    std::vector<int> order;
    order.reserve(nn);
    while (order.size() < nn) {
      bool any = false;
      for (size_t k = 0; k < nn; ++k) {
        if (done[k]) {
          continue;
        }
        bool ready = true, reads_dirty = false;
        for (int r : t->nodes[k].in) {
          if (r >= (1 << 30)) {
            ready &= made[r - (1 << 30)] != 0;
            reads_dirty |= dirty[r - (1 << 30)] != 0;
          }
        }
        if (!ready || (!is_view[k] && reads_dirty)) {
          continue;
        }
        for (int r : t->nodes[k].out) {
          made[r - (1 << 30)] = 1;
          dirty[r - (1 << 30)] = is_view[k] ? reads_dirty : 1;
        }
        done[k] = 1;
        order.push_back(int(k));
        any = true;
      }
      if (!any) {
        std::fill(dirty.begin(), dirty.end(), 0);
      }
    }
    std::vector<FusedNode> sorted;
    sorted.reserve(nn);
    int next = 0;
    for (int k : order) {
      for (int r : t->nodes[k].out) {
        renum[r - (1 << 30)] = next++;
      }
      sorted.push_back(std::move(t->nodes[k]));
    }
    t->nodes = std::move(sorted);
  } else if (level) {
    // Level order: a node runs after every node whose depth from the
    // inputs is lower, whichever branch it is on. Nodes of one level read
    // nothing another of them writes, so their dispatches need no barrier
    // between them where the walk's order (one branch to its end, then
    // the next) put one after each branch step. Same nodes, same operands.
    std::vector<int> made_by(next_node_slot, 0);
    for (size_t k = 0; k < t->nodes.size(); ++k) {
      for (int r : t->nodes[k].out) {
        made_by[r - (1 << 30)] = int(k);
      }
    }
    std::vector<int> depth(t->nodes.size(), 0);
    for (size_t k = 0; k < t->nodes.size(); ++k) {
      for (int r : t->nodes[k].in) {
        if (r >= (1 << 30)) {
          depth[k] = std::max(depth[k], depth[made_by[r - (1 << 30)]] + 1);
        }
      }
    }
    std::vector<int> order(t->nodes.size());
    for (size_t k = 0; k < order.size(); ++k) {
      order[k] = int(k);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return depth[a] < depth[b];
    });
    std::vector<FusedNode> sorted;
    sorted.reserve(order.size());
    int next = 0;
    for (int k : order) {
      for (int r : t->nodes[k].out) {
        renum[r - (1 << 30)] = next++;
      }
      sorted.push_back(std::move(t->nodes[k]));
    }
    t->nodes = std::move(sorted);
  }
  auto final_slot = [&](int r) {
    if (r >= (1 << 30)) {
      return t->n_inputs + nc + renum[r - (1 << 30)];
    }
    return r < 0 ? t->n_inputs + (-r - 1) : r;
  };
  t->n_slots = t->n_inputs + nc + next_node_slot;
  std::vector<int> last_reader(t->n_slots, -1);
  for (size_t k = 0; k < t->nodes.size(); ++k) {
    auto& n = t->nodes[k];
    for (auto& r : n.in) {
      r = final_slot(r);
      last_reader[r] = int(k);
    }
    for (auto& r : n.out) {
      r = final_slot(r);
    }
  }
  for (auto& o : outputs) {
    int r = final_slot(ref(o));
    t->out_slots.push_back(r);
    t->out_shapes.push_back(o.shape());
    t->out_dtypes.push_back(o.dtype());
    last_reader[r] = -2;
  }
  const int first_node_slot = t->n_inputs + nc;
  for (size_t k = 0; k < t->nodes.size(); ++k) {
    auto& n = t->nodes[k];
    n.last.assign(n.in.size(), 0);
    for (size_t i = 0; i < n.in.size(); ++i) {
      int r = n.in[i];
      bool later_same = false;
      for (size_t j = i + 1; j < n.in.size(); ++j) {
        later_same |= n.in[j] == r;
      }
      // Only arrays the tape made are handed over; an input or a constant
      // belongs to the caller.
      n.last[i] =
          r >= first_node_slot && last_reader[r] == int(k) && !later_same;
      t->n_handed += n.last[i];
    }
    t->max_in = std::max(t->max_in, int(n.in.size()));
    t->max_out = std::max(t->max_out, int(n.out.size()));
  }
  t->constants = std::move(constants);
#ifdef _METAL_
  if (direct && t->stream.device == mx::Device::gpu) {
    t->dstate = std::make_shared<DirectState>();
    t->dstate->epoch = kq_env_epoch();
    for (auto& n : t->nodes) {
      std::string name = n.prim->name();
      auto& u = t->dstate->unlisted;
      if (!direct_listed(name) &&
          std::find(u.begin(), u.end(), name) == u.end()) {
        u.push_back(name);
      }
    }
    direct_reset(*t->dstate);
  }
#endif
  return t;
}

std::vector<mx::array> fused_call(
    const std::shared_ptr<const FusedTape>& t,
    const std::vector<mx::array>& inputs) {
  const char* op = "[mlx_kquant.fused_call]";
  if (int(inputs.size()) != t->n_inputs) {
    throw std::invalid_argument(std::string(op) + " input count differs.");
  }
  for (int i = 0; i < t->n_inputs; ++i) {
    if (inputs[i].shape() != t->in_shapes[i] ||
        inputs[i].dtype() != t->in_dtypes[i]) {
      throw std::invalid_argument(
          std::string(op) +
          " an input's shape or dtype differs from the "
          "recorded graph's.");
    }
  }
  if (t->param_in >= 0) {
    // The block is read on the host when the call is built and again when
    // it is encoded: it has to hold its values now.
    const mx::array& pa = inputs[t->param_in];
    if (!pa.is_available() || !pa.flags().row_contiguous) {
      throw std::invalid_argument(
          std::string(op) + " the parameter block must be evaluated.");
    }
    const int bad = param_outside(*t, pa.data<int32_t>());
    if (bad >= 0) {
      throw std::invalid_argument(
          std::string(op) + " parameter slot " + std::to_string(bad) +
          " is outside the range the record covers.");
    }
  }
  // The constants are read from the tape at eval; only the caller's arrays
  // are graph inputs, so the eval walk sees no edge per weight.
  return mx::array::make_arrays(
      t->out_shapes,
      t->out_dtypes,
      std::make_shared<KQuantFusedGraph>(t->stream, t),
      inputs);
}

std::vector<bool> fused_used_inputs(const std::shared_ptr<const FusedTape>& t) {
  std::vector<bool> used(t->n_inputs, false);
  for (auto& n : t->nodes) {
    for (int r : n.in) {
      if (r < t->n_inputs) {
        used[r] = true;
      }
    }
  }
  for (int r : t->out_slots) {
    if (r < t->n_inputs) {
      used[r] = true;
    }
  }
  return used;
}

std::vector<std::string> fused_nodes(
    const std::shared_ptr<const FusedTape>& t) {
  std::vector<std::string> names;
  for (auto& n : t->nodes) {
    names.emplace_back(n.prim->name());
  }
  return names;
}

FusedDirectInfo fused_direct_info(const std::shared_ptr<const FusedTape>& t) {
  FusedDirectInfo info;
#ifdef _METAL_
  info.pool_bytes = g_direct_pool_bytes.load();
  if (!t->dstate) {
    return info;
  }
  DirectState& s = *t->dstate;
  std::lock_guard lk(s.mtx);
  if (s.epoch != kq_env_epoch()) {
    s.epoch = kq_env_epoch();
    direct_reset(s);
  }
  static const char* names[] = {"new", "recorded", "replay", "refused"};
  info.state = names[s.state];
  if (s.state != 3 && !kq_env_cache_on()) {
    info.state = "waiting";
    info.reason = "the lever cache is off (env_cache)";
  }
  if (s.state == 3) {
    info.reason = s.reason;
  }
  info.primitives = s.prims;
  info.plain_runs = s.plain;
  if (s.prog) {
    auto& p = *s.prog;
    info.replays = p.replays.load();
    info.calls_recorded = p.n_calls_recorded;
    info.calls = int(p.ops.size()) + 1;
    info.dispatches = p.n_dispatch;
    info.barriers = p.n_barrier;
    info.barriers_recorded = p.n_barrier_recorded;
    info.scratch_bytes = p.scratch_bytes;
    info.kept_bytes = p.kept_bytes;
    info.program_bytes =
        (p.ops.size() + p.trace->ops.size()) * sizeof(DirectOp) +
        p.blob.size() + p.trace->blob.size() +
        2 * p.refs.size() * sizeof(DirectRef);
    int scratch = 0;
    for (auto& r : p.refs) {
      scratch += r.cls == R_SCRATCH;
    }
    info.scratch_buffers = scratch;
  }
#endif
  return info;
}

std::vector<std::string> fused_direct_calls(
    const std::shared_ptr<const FusedTape>& t) {
  std::vector<std::string> out;
#ifdef _METAL_
  if (!t->dstate) {
    return out;
  }
  std::shared_ptr<const DirectProgram> prog;
  {
    std::lock_guard lk(t->dstate->mtx);
    prog = t->dstate->prog;
  }
  if (!prog) {
    return out;
  }
  const DirectTrace& tr = *prog->trace;
  static const char* cls[] = {"input", "const", "between", "output", "kept"};
  auto hex = [](const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string h;
    for (size_t i = 0; i < n; ++i) {
      h += d[p[i] >> 4];
      h += d[p[i] & 15];
    }
    return h;
  };
  auto sz = [](const MTL::Size& s) {
    return std::to_string(s.width) + "x" + std::to_string(s.height) + "x" +
        std::to_string(s.depth);
  };
  std::unordered_map<const void*, int> pso;
  for (auto& o : tr.ops) {
    std::string n = t->nodes[o.node].prim->name();
    std::string line = std::to_string(o.node) + " " + n + " ";
    switch (o.kind) {
      case D_PSO:
        line += "pipeline " +
            std::to_string(pso.emplace(o.ptr, int(pso.size())).first->second);
        break;
      case D_BUF: {
        auto& r = tr.refs[o.ref];
        line += "bind " + std::to_string(o.idx) + " " + cls[r.cls] + " " +
            std::to_string(r.cls == R_SCRATCH ? o.ref : r.index) + " +" +
            std::to_string(o.off);
        break;
      }
      case D_BYTES:
        line += "bytes " + std::to_string(o.idx) + " " +
            hex(tr.blob.data() + o.off, o.len);
        break;
      case D_TGMEM:
        line += "tgmem " + std::to_string(o.idx) + " " + std::to_string(o.len);
        break;
      case D_BARRIER:
        line += "barrier";
        break;
      default:
        line += std::string(o.kind == D_DTG ? "groups " : "threads ") +
            sz(o.s0) + " " + sz(o.s1);
        break;
    }
    out.push_back(std::move(line));
  }
  for (size_t r = 0; r < tr.refs.size(); ++r) {
    auto& rf = tr.refs[r];
    std::string line =
        std::string("buffer ") + std::to_string(r) + " " + cls[rf.cls];
    // The length of a buffer between nodes or of an output is whatever the
    // allocator handed out, at least what the primitive asked for.
    if (rf.cls == R_KEPT) {
      line += " bytes " + std::to_string(rf.len) + " content " +
          hex(tr.kept[rf.index].data(), tr.kept[rf.index].size());
    }
    out.push_back(std::move(line));
  }
  for (auto& o : tr.outs) {
    out.push_back(
        "output kind " + std::to_string(o.kind) + " of " +
        std::to_string(o.other) + " +" + std::to_string(o.off) + " size " +
        std::to_string(o.data_size));
  }
#endif
  return out;
}

#ifdef _METAL_
namespace {
// A debug trace of the encoder calls mlx and replays make between
// encoder_trace_begin and encoder_trace_end (the recording proxy with no
// tape under it).
thread_local DirectRecorder* g_trace = nullptr;
thread_local NS::SharedPtr<MTL::ComputeCommandEncoder> g_trace_real;
thread_local id g_trace_proxy = nullptr;
} // namespace
#endif

void encoder_trace_begin() {
#ifdef _METAL_
  if (g_trace != nullptr || g_recorder != nullptr) {
    throw std::runtime_error("[encoder_trace_begin] a recording is running.");
  }
  auto& ce =
      mx::metal::get_command_encoder(mx::default_stream(mx::Device::gpu));
  g_trace = new DirectRecorder;
  g_trace->real = (id)ce.get_command_encoder();
  g_trace_real = ce.encoder_;
  g_trace_proxy = ((id(*)(id, SEL))objc_msgSend)(
      ((id(*)(Class, SEL))objc_msgSend)(
          direct_proxy_class(), sel_registerName("alloc")),
      sel_registerName("init"));
  g_recorder = g_trace;
  ce.encoder_ = NS::TransferPtr(
      reinterpret_cast<MTL::ComputeCommandEncoder*>(g_trace_proxy));
#endif
}

std::vector<std::string> encoder_trace_end() {
  std::vector<std::string> out;
#ifdef _METAL_
  if (g_trace == nullptr) {
    return out;
  }
  auto& ce =
      mx::metal::get_command_encoder(mx::default_stream(mx::Device::gpu));
  const bool held = ce.encoder_.get() ==
      reinterpret_cast<MTL::ComputeCommandEncoder*>(g_trace_proxy);
  if (held) {
    ce.encoder_ = g_trace_real;
  }
  g_trace_real = NS::SharedPtr<MTL::ComputeCommandEncoder>();
  g_recorder = nullptr;
  std::unique_ptr<DirectRecorder> rec(g_trace);
  g_trace = nullptr;
  g_trace_proxy = nullptr;
  // An eval that ended the encoder (a commit) dropped the proxy itself;
  // the calls up to there are the trace.
  std::unordered_map<const void*, std::string> names;
  {
    auto& d = mx::metal::device(mx::Device(mx::Device::gpu));
    std::shared_lock lk(d.kernel_mtx_);
    for (auto& [lib, ks] : d.library_kernels_) {
      for (auto& [name, pso] : ks) {
        names[pso.get()] = name;
      }
    }
  }
  std::unordered_map<const void*, int> bufs;
  for (auto& r : rec->raw) {
    switch (r.kind) {
      case D_PSO: {
        auto it = names.find(r.ptr);
        out.push_back(
            "pso " + (it == names.end() ? std::string("?") : it->second));
        break;
      }
      case D_BUF: {
        auto it = bufs.emplace(r.ptr, int(bufs.size())).first;
        out.push_back(
            "buf " + std::to_string(r.b) + " b" + std::to_string(it->second) +
            " len " +
            std::to_string(static_cast<const MTL::Buffer*>(r.ptr)->length()) +
            " +" + std::to_string(r.a));
        break;
      }
      case D_BYTES:
        out.push_back(
            "bytes " + std::to_string(r.b) + " n " +
            std::to_string(r.bytes.size()));
        break;
      case D_TGMEM:
        out.push_back(
            "tgmem " + std::to_string(r.b) + " n " + std::to_string(r.a));
        break;
      case D_BARRIER:
        out.push_back("barrier " + std::to_string(r.a));
        break;
      case D_DTG:
      case D_DT:
        out.push_back(
            std::string(r.kind == D_DTG ? "groups " : "threads ") +
            std::to_string(r.s0.width) + "x" + std::to_string(r.s0.height) +
            "x" + std::to_string(r.s0.depth) + " by " +
            std::to_string(r.s1.width) + "x" + std::to_string(r.s1.height) +
            "x" + std::to_string(r.s1.depth));
        break;
      default:
        out.push_back("other");
        break;
    }
  }
#endif
  return out;
}

int fused_constants(const std::shared_ptr<const FusedTape>& t) {
  return int(t->constants.size());
}

// ---------------------------------------------------------------------------
// Parameter blocks and bound replay.
// ---------------------------------------------------------------------------

int kq_param_value(const mx::array& p, int slot) {
  if (p.dtype() != mx::int32 || slot < 0 || slot >= int(p.size()) ||
      !p.flags().row_contiguous) {
    throw std::invalid_argument(
        "[mlx_kquant] a parameter block must be a contiguous int32 vector "
        "that holds the slot.");
  }
  // Read on the host as the call is encoded: a block a kernel of this
  // command buffer writes would be read before it is written.
  if (!p.is_available()) {
    throw std::invalid_argument(
        "[mlx_kquant] a parameter block must be evaluated before the call "
        "that reads it.");
  }
#ifdef _METAL_
  if (g_recorder != nullptr) {
    g_recorder->param_bufs.emplace_back(p.buffer().ptr(), p.offset());
  }
#endif
  return p.data<int32_t>()[slot];
}

void kq_param_site(
    int idx,
    int field,
    int slot,
    int mul,
    int add,
    int64_t lo,
    int64_t hi) {
#ifdef _METAL_
  if (g_recorder == nullptr) {
    return;
  }
  auto& r = *g_recorder;
  for (int i = int(r.raw.size()) - 1; i >= 0; --i) {
    if (r.raw[i].node != r.node) {
      break;
    }
    if (r.raw[i].kind == D_BYTES && int(r.raw[i].b) == idx) {
      r.sites.push_back({i, field, slot, mul, add, lo, hi});
      return;
    }
  }
  if (r.bad.empty()) {
    r.bad = "a parameter site names bytes its primitive did not set";
    r.raw.emplace_back();
    r.raw.back().kind = 255;
    r.raw.back().node = r.node;
  }
#else
  (void)idx;
  (void)field;
  (void)slot;
  (void)mul;
  (void)add;
  (void)lo;
  (void)hi;
#endif
}

struct FusedBinding {
#ifdef _METAL_
  std::shared_ptr<const FusedTape> tape;
  std::shared_ptr<const DirectProgram> prog;
  std::shared_ptr<DirectPool> pool;
  std::vector<const MTL::Buffer*> cur;
  std::vector<int64_t> adj;
  std::vector<mx::array> inputs;
  std::vector<mx::array> outputs;
  std::vector<uint8_t> blob;
  int32_t* params{nullptr};
  size_t output_bytes{0};
#endif
};

std::shared_ptr<FusedBinding> fused_bind(
    const std::shared_ptr<const FusedTape>& tape,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const char* op = "[mlx_kquant.fused_bind]";
#ifdef _METAL_
  const FusedTape& t = *tape;
  if (!t.dstate) {
    throw std::invalid_argument(std::string(op) + " the record is not direct.");
  }
  if (int(inputs.size()) != t.n_inputs) {
    throw std::invalid_argument(std::string(op) + " input count differs.");
  }
  for (int i = 0; i < t.n_inputs; ++i) {
    if (inputs[i].shape() != t.in_shapes[i] ||
        inputs[i].dtype() != t.in_dtypes[i] || !inputs[i].is_available()) {
      throw std::invalid_argument(
          std::string(op) +
          " an input is not evaluated, or its shape or "
          "dtype differs from the recorded graph's.");
    }
  }
  std::shared_ptr<const DirectProgram> prog;
  {
    std::lock_guard lk(t.dstate->mtx);
    if (t.dstate->epoch == kq_env_epoch() && kq_env_cache_on() &&
        t.dstate->state == 2) {
      prog = t.dstate->prog;
    }
  }
  if (!prog) {
    throw std::runtime_error(
        std::string(op) +
        " the record has no program yet: run it through "
        "fused_call until fused_direct_info says replay.");
  }
  if (!direct_inputs_match(*prog, inputs)) {
    throw std::runtime_error(
        std::string(op) +
        " an input is laid out differently from the "
        "recorded one.");
  }
  const DirectProgram& d = *prog;
  auto b = std::make_shared<FusedBinding>();
  b->tape = tape;
  b->prog = prog;
  b->inputs = inputs;
  b->blob = d.blob;
  // Buffers between nodes of the binding's own: two stages in flight on
  // two command buffers run at the same time, so they share none.
  b->pool = std::make_shared<DirectPool>();
  b->cur = direct_assign(*b->pool, d, t);
  b->adj.assign(b->cur.size(), 0);
  for (auto& [r, i] : d.in_refs) {
    b->cur[r] = static_cast<const MTL::Buffer*>(inputs[i].buffer().ptr());
    b->adj[r] = inputs[i].offset();
  }
  std::vector<mx::allocator::Buffer> fresh;
  for (size_t j = 0; j < d.fresh.size(); ++j) {
    fresh.push_back(mx::allocator::malloc(d.fresh[j].second));
    b->cur[d.fresh[j].first] = static_cast<const MTL::Buffer*>(fresh[j].ptr());
    b->output_bytes += d.fresh[j].second;
  }
  outputs.clear();
  for (size_t i = 0; i < d.outs.size(); ++i) {
    auto& o = d.outs[i];
    mx::array::Flags f{o.contiguous, o.row_contiguous, o.col_contiguous};
    mx::array a(t.out_shapes[i], t.out_dtypes[i], nullptr, {});
    switch (o.kind) {
      case 0:
        a.set_data(fresh[o.other], o.data_size, o.strides, f, o.off);
        break;
      case 1:
        a.copy_shared_buffer(
            outputs[o.other], o.strides, f, o.data_size, o.off);
        break;
      case 2:
        a.copy_shared_buffer(inputs[o.other], o.strides, f, o.data_size, o.off);
        break;
      default:
        a.copy_shared_buffer(
            t.constants[o.other], o.strides, f, o.data_size, o.off);
        break;
    }
    // Evaluated from birth: the next record binds it and Python reads it
    // with no eval. Its content is that of the last command buffer the
    // binding was encoded into, once that buffer has completed.
    a.set_status(mx::array::Status::available);
    outputs.push_back(std::move(a));
  }
  b->outputs = outputs;
  if (t.param_in >= 0) {
    const mx::array& pa = inputs[t.param_in];
    b->params = static_cast<int32_t*>(static_cast<void*>(
        static_cast<char*>(
            const_cast<MTL::Buffer*>(
                static_cast<const MTL::Buffer*>(pa.buffer().ptr()))
                ->contents()) +
        pa.offset()));
  }
  return b;
#else
  (void)tape;
  (void)inputs;
  (void)outputs;
  throw std::runtime_error(std::string(op) + " requires a Metal build.");
#endif
}

bool fused_patch(FusedBinding& b, int slot, int32_t value) noexcept {
#ifdef _METAL_
  if (b.params == nullptr) {
    return false;
  }
  const FusedTape& t = *b.tape;
  if (slot < 0 || slot >= int(t.param_lo.size()) || value < t.param_lo[slot] ||
      value > t.param_hi[slot]) {
    return false;
  }
  b.params[slot] = value;
  return true;
#else
  (void)b;
  (void)slot;
  (void)value;
  return false;
#endif
}

int32_t* fused_params(FusedBinding& b) noexcept {
#ifdef _METAL_
  return b.params;
#else
  (void)b;
  return nullptr;
#endif
}

bool fused_encode(FusedBinding& b, void* compute_encoder) noexcept {
#ifdef _METAL_
  const DirectProgram& d = *b.prog;
  if (b.params != nullptr) {
    if (param_outside(*b.tape, b.params) >= 0) {
      return false;
    }
    if (!d.sites.empty()) {
      direct_patch(d, b.params, b.blob.data());
    }
  }
  auto* enc = static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
  // Orders the replay after whatever wrote its inputs earlier in this
  // encoder.
  enc->memoryBarrier(MTL::BarrierScopeBuffers);
  direct_emit(enc, d, b.blob.data(), b.cur.data(), b.adj.data());
  d.replays.fetch_add(1, std::memory_order_relaxed);
  return true;
#else
  (void)b;
  (void)compute_encoder;
  return false;
#endif
}

FusedBindingInfo fused_binding_info(const FusedBinding& b) {
  FusedBindingInfo info;
#ifdef _METAL_
  info.output_bytes = b.output_bytes;
  info.param_bytes = b.params ? b.tape->param_lo.size() * 4 : 0;
  info.blob_bytes = b.blob.size();
  info.scratch_bytes = b.prog->scratch_bytes;
  info.pool_bytes = b.pool->bytes;
#endif
  return info;
}

void* fused_command_queue(mx::Stream s) {
#ifdef _METAL_
  return mx::metal::get_command_encoder(s).queue_.get();
#else
  (void)s;
  return nullptr;
#endif
}

std::vector<bool> fused_commit_detached(
    const std::vector<std::shared_ptr<FusedBinding>>& bindings,
    mx::Stream s) {
  std::vector<bool> ok(bindings.size(), false);
#ifdef _METAL_
  auto* queue = static_cast<MTL::CommandQueue*>(fused_command_queue(s));
  std::thread worker([&] {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::CommandBuffer* cb = queue->commandBufferWithUnretainedReferences();
    MTL::ComputeCommandEncoder* enc =
        cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    for (size_t i = 0; i < bindings.size(); ++i) {
      ok[i] = fused_encode(*bindings[i], enc);
    }
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
  });
  worker.join();
#else
  (void)bindings;
  (void)s;
#endif
  return ok;
}

} // namespace mlx_kquant
