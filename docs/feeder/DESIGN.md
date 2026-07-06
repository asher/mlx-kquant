# Feeder loop: GPU-resident MoE decode for larger-than-RAM models

Status: the handoff primitives are implemented and tested on this branch
(`kq.event_signal` / `kq.event_wait` + the `shared_event_*` host API,
`src/kquant_event.cpp`, `tests/test_events.py`); the architecture was
validated first by a standalone prototype
(`benchmarks/feeder/prototype/feeder.swift`). This document records the
architecture, the measurements that justify it, and the remaining
integration plan.

## Problem

A MoE model whose quantized weights exceed unified memory cannot be wired for
the GPU. gguf-mlx today streams such models by pinning the routed experts to
the CPU stream and letting the page cache serve them from disk. That works, but
decode is bounded by two costs that have nothing to do with CPU⇄GPU
orchestration:

1. **Demand faulting.** The gemv touches expert bytes through mmap and faults
   them in as ~16 KB clusters, serially, at random-read bandwidth — a small
   fraction of what the SSD delivers to parallel, explicitly-issued reads.
2. **CPU GEMV bandwidth.** Even fully cached, the CPU reads weights several
   times slower than the GPU does.

`MLX_METAL_FAST_SYNCH=1` historically made hybrid placement fast because it
kept one unbroken command stream per token, but it does so with an infinite
GPU spin-kernel polling CPU-written memory — a mechanism Apple does not
guarantee (no in-kernel CPU→GPU coherence, no forward-progress guarantee for
spinlocks). When the poll never observes the store, the spin kernel hangs with
the model's memory wired: unrecoverable without a reboot. MLX marked it
won't-fix (mlx#3142). This design gets the same unbroken-stream property from
*supported* primitives.

## Key facts the design rests on

Established on an M3 Max (128 GB), macOS 15, with
`benchmarks/feeder/prototype/harness.swift`:

- MLX's default fence path already encodes mid-buffer `MTLSharedEvent`
  signal/wait pairs. A pre-satisfied encoded wait costs ~20 µs marginal; a
  blocked encoded wait costs ~150–245 µs dominated by GPU-idle wake, not by
  the event itself. Raw CPU⇄GPU event ping-pong is ~1 µs.
- Coherence through an event-gated handoff was correct in every trial,
  including adversarial GPU-cache-priming of the same buffers before the CPU
  write. The event edge is a real synchronization point for shared storage.
- Signaling a waited event to `UINT64_MAX` (poison) always releases the wait:
  a wedged command buffer drains with an error status instead of hanging.
  FAST_SYNCH's reboot failure mode becomes one discarded token.
- Per-layer CPU⇄GPU orchestration on current MLX costs ≈ zero at decode
  granularity: an in-RAM hybrid (CPU experts / GPU spine) decodes at the same
  rate as pure CPU. The historical "hybrid is slower" attribution to stream
  fences is stale; the real streaming bottleneck is (1) and (2) above.

## Architecture

One pre-encoded command buffer per token. Per layer:

```
[spine + router kernel]
  -> encodeSignalEvent(routerDone, L)
  -> encodeWaitForEvent(weightsReady, L)
  -> [expert kernel, data-driven addressing]
```

A CPU **feeder** thread polls `routerDone`, reads the router's top-k ids from
shared memory (safe: written before the signal edge), preads the non-resident
experts' slices from the GGUF into a **staging ring** at high queue depth, and
host-signals `weightsReady`. The expert kernel computes each expert's address
at runtime from a residency table + ring slot table, so the CPU never mutates
control state that an already-encoded kernel depends on — only weight bytes
and event values cross the boundary, both through defined synchronization
edges.

Components for a real integration:

- **Wired expert arena**: a large `MTLBuffer` holding the resident subset of
  expert slices. Residency fraction *f* is the whole game (see numbers).
- **Popularity-based residency manager**: decides which experts live in the
  arena; updated off the critical path from routing statistics.
- **Staging ring**: fixed slots for streamed-in experts, written by the feeder
  before the corresponding `weightsReady` value is signaled.
- **Speculative prefetch**: issue next-layer reads from predicted expert ids
  before that layer's router runs; mispredictions cost only wasted reads.
- **Watchdog**: if the feeder cannot satisfy a wait within a deadline, poison
  both events; the buffer drains with `status == .error` and the token is
  retried or surfaced as an error. No hang, no reboot.

## Prototype results

`benchmarks/feeder/prototype/feeder.swift` simulates MiniMax-M2.7-class
decode (62 layers, 256 experts, top-8, ~9.28 MiB/expert at Q5_K_M) with
bandwidth-faithful kernels reading the real 162 GB GGUF (`F_NOCACHE`, qd8).
Median steady-state token, M3 Max:

| resident fraction f | mode      | tok/s |
|---------------------|-----------|-------|
| 100%                | —         | 39.9  |
| 85%                 | prefetch  | 10.6  |
| 85%                 | demand    | 8.3   |
| 76%                 | prefetch  | 8.1   |
| 65%                 | prefetch  | 6.4   |
| 65%                 | demand    | 5.0   |

Encode cost ~0.3 ms/token; at f=85% with 1-layer-ahead prefetch the IO is
almost fully overlapped (~95% of the analytic IO ceiling). The feed rates were
partly page-cache-assisted; scale IO-bound rows ~30% down for a fully cold
cache. Kill-feeder test: watchdog poison drained the buffer (`status == 4`),
clean process exit.

Reference baseline at comparable effective residency: gguf-mlx streaming
decode on the same model/box runs ~1–3 tok/s. The architecture is worth
roughly 4–10× on decode depending on quant and residency, before any prefill
work.

## The primitive pair (implemented on this branch)

`kq.event_signal(x, handle, value)` / `kq.event_wait(x, handle, value)` are
identity ops that end the active compute encoder and encode
`encodeSignalEvent` / `encodeWait` on the stream's command buffer at their
position in evaluation order, threading `x` through as a data dependency so
lazy evaluation orders them. CPU evals are plain identities, so placement
fallbacks stay legal. `kq.shared_event_create/destroy/set/read/wait` is the
host side (the feeder's half of each edge); `shared_event_wait` releases the
GIL. Functional coverage in `tests/test_events.py` includes a live
feeder-thread handoff, a wedged wait drained by the `UINT64_MAX` poison with
the device healthy afterwards, and a signal/wait pair separated by enough ops
to force command-buffer splits — splitting can serialize the pair but not
deadlock it, because a wait's satisfying signal always comes from the host.

Two findings from benchmarking the pair inside real MLX graphs:

- A per-layer handoff pair costs ~160 µs when the feeder answers just-in-time
  (the encoded wait actually blocks: GPU-idle wake dominates, matching the
  harness numbers) and ~20 µs when the feeder has already signaled. At 62
  layers that brackets the sync tax per token between ~1 and ~10 ms — small
  against the IO term either way, and the blocked case only occurs when IO is
  the bottleneck anyway.
- The existing fused kq gather kernels already run the full MiniMax-geometry
  expert token (62 layers × top-8, q5_k gate/up + q6_k down) from a resident
  arena at ~16 ms/token (~320 GB/s effective) when chained into one graph per
  token. **No new Metal kernel is required**: the feeder owns the router ids
  at the handoff edge, so it can resolve expert id → arena/ring slot on the
  host and hand the stock gather kernels a slot-index array. The
  `moe_gather_staged` kernel this plan originally called for is unnecessary.

## Landed: writable staging + prefill feeder

`arena_alloc` (this branch) is the CPU-writable, GPU-visible allocation:
page-aligned host memory wrapped no-copy in a Metal shared buffer, returned
as a uint8 array plus a writable memoryview over the same bytes, so
`os.preadv` reads disk straight into kernel-visible memory.

The prefill feeder itself lives in gguf-mlx (`gguf_mlx/feeder.py`,
`KQ_FEEDER_PREFILL=1`): a two-slot ring of arena buffers, one slot per layer
parity, staged by a feeder thread while the GPU computes the previous layer
from the other slot, with the module's expert weights swapped to zero-copy
slot views for the duration of each call. Because streaming prefill already
evals per layer, host-side synchronization suffices there - the encoded
event pair in this extension is needed only by a decode feeder, which has no
per-layer eval. Slots are sized per kind at the largest layer and viewed
per-layer, since mixed-codec quants (Q5_K_M puts q6_k down stacks on some
layers, q5_k on others) make shapes non-uniform.

Measured on the 162 GB MiniMax Q5_K_M (128 GB M3 Max, greedy output
verified identical with the feeder on and off):

| prompt tokens | page-cache prefill | feeder prefill |
|---------------|--------------------|----------------|
| 46            | 1.78 tok/s         | 1.78 tok/s     |
| 829           | 18.7 tok/s         | 33.1 tok/s     |
| 7101          | 55.5 tok/s         | 61.2 tok/s     |

Interpretation: at 829 tokens the feeder sits on the SSD full-sweep floor
(829 tok / 33.1 = 25.0 s vs a 22.8 s sweep) - the page-cache double pass is
gone. At 46 tokens it's a wash: whole-layer staging reads all experts while
the page-cache path only faults the ~77% a small chunk routes to; the extra
bytes cancel the reclaim penalty (router-aware partial staging would win
here). At 7k+ the ALU `gather_qmm` rate (~16 ms/token at that batch, no NAX
on M3) binds both paths; that ceiling is the kernel/NAX item below. Two
side effects worth knowing: decode right after a feeder prefill starts
~10-15% slower (the page cache is cold - the feeder never populates it),
and the ring wires ~2 x 2.7 GB while prefilling.

## Remaining work

1. **Long-prefill compute ceiling**: on NAX-less GPUs the ALU gather_qmm is
   the bound. Options: a tuned prefill qmm (threadgroup dequant + simdgroup
   matmul), or staged dequant-to-fp16 + dense matmul per ring slot. On an
   M5-class GPU the existing NAX path may erase this without new work -
   measure there first.
2. **Router-aware partial staging** for small chunks: eval the router before
   staging and pread only the routed experts' slices - closes the short-
   prompt gap where whole-layer staging over-reads.
3. **The decode feeder** (the original architecture): pre-encoded per-token
   command buffers with the event pair, wired arena + popularity residency
   manager, staging ring for misses, watchdog poison recovery. One empirical
   caution: measured token-to-token routing reuse on MiniMax-M2.7 (256
   experts, top-8) is only ~30%, so speculative next-layer prefetch of the
   previous token's experts wastes most of its reads; popularity pinning
   across a window is the mechanism that pays, not per-token speculation.

## Files

The runnable artifacts behind the numbers in this document live in
[`benchmarks/feeder/`](../../benchmarks/feeder/):

- [`bench_feeder_mlx.py`](../../benchmarks/feeder/bench_feeder_mlx.py) —
  handoff-pair cost and expert-gather rate through the kq primitives in real
  MLX graphs. Re-run this on new hardware (e.g. an M5 with NAX) to revisit
  the compute-ceiling numbers.
- [`prototype/feeder.swift`](../../benchmarks/feeder/prototype/feeder.swift)
  — the standalone Swift/Metal prototype (`swiftc -O feeder.swift`; usage:
  `feeder <residentPct> <demand|prefetch> <tokens> [--kill-feeder]`). Frozen
  reference for the prototype results table.
- [`prototype/harness.swift`](../../benchmarks/feeder/prototype/harness.swift)
  — sync/coherence/IO measurement harness that produced the primitive-cost
  numbers above. Frozen reference.
