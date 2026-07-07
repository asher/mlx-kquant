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

| prompt tokens | page-cache prefill | feeder prefill | + gather_qmm_rhs |
|---------------|--------------------|----------------|------------------|
| 46            | 1.78 tok/s         | 1.78 tok/s     |                  |
| 829           | 18.7 tok/s         | 33.1 tok/s     |                  |
| 7101          | 55.5 tok/s         | 61.2 tok/s     | 211.0 tok/s      |

Interpretation: at 829 tokens the feeder sits on the SSD full-sweep floor
(829 tok / 33.1 = 25.0 s vs a 22.8 s sweep) - the page-cache double pass is
gone. At 46 tokens it's a wash: whole-layer staging reads all experts while
the page-cache path only faults the ~77% a small chunk routes to; the extra
bytes cancel the reclaim penalty (router-aware partial staging would win
here). At 7k+ the compute rate binds both paths; the last column is the
non-NAX sorted-rhs kernel (next section) lifting that ceiling 3.3x. Two
side effects
worth knowing: decode right after a feeder prefill starts ~10-15% slower
(the page cache is cold - the feeder never populates it), and the ring wires
~2 x 2.7 GB while prefilling.

The 7k+ ceiling was the missing non-NAX sorted-rhs GEMM, a generic kernel
landed and documented in the changelog (`gather_qmm_rhs`, steel
simdgroup-mma, rows-per-expert-adaptive row tile): the feeder's 7k prefill
went 61.2 -> 211.0 tok/s with no feeder changes. It benefits any sorted MoE
prefill on a non-NAX GPU, in-RAM models included, so it is not a feeder
component - the feeder just no longer has a kernel-bound regime below the
SSD sweep floor.

## Landed: the decode feeder, stage A

The wired arena and the popularity residency manager are landed in gguf-mlx
(`gguf_mlx/decode_feeder.py`, `KQ_FEEDER_DECODE=1`), without the encoded
event pair: per-layer wired arenas (`arena_alloc` + `mlock`) hold the
popular subset of each layer's expert stacks, decode-sized calls remap the
router's expert ids to arena slots on the host, and the stock gather
kernels run on the GPU stream from the arena. Misses are pread
(`F_NOCACHE`) straight into evicted slots. Two pieces of the original plan
turned out unnecessary at decode granularity:

- **No staging ring.** The offload wrapper already evaluates the router
  per layer, and layer L's router depends on everything before it - so
  when layer L stages for token t, the last gather that referenced layer
  L's arena (token t-1's) has completed, and any slot may be overwritten.
  Eviction, adoption and staging are one operation.
- **No event pair.** Host-side synchronization suffices for the same
  reason. The encoded pair remains stage B (below).

Eviction picks the least-popular resident (decayed LFU counts) never one
the current call routes to; the arena starts empty and self-organizes.
Small prefill chunks (<= `KQ_ARENA_STAGE_MAX_TOKENS`, default 64) take the
same path when their routed set fits, which is what makes repeat
short-prompt TTFT cheap in a live session; on overflow they fall back to
**router-aware partial staging** in the prefill feeder
(`prefill_partial_call`: pread only the routed experts' slices into the
ring slot, original indices, no remap - remaining-work item 2, now done).

Measured (162 GB MiniMax Q5_K_M, 128 GB M3 Max, 70.6 GB arena ~ f=0.44,
greedy output identical to the CPU path throughout):

| metric | page-cache decode | arena decode |
|--------|-------------------|--------------|
| 512-token generation | 2.4 tok/s | 4.0 tok/s avg, ~4.7 steady (89.8% hit) |
| 53-token prompt TTFT | 19.4 s (whole-layer feeder) | 11.4 s (partial staging) |

Hard-won sizing and wiring lessons:

- **The arena must be mlocked.** Unlocked, the kernel pages the LFU-cold
  tail to swap under exactly the pressure an over-RAM model creates, and a
  hit that faults from swap is slower than the GGUF read it saved
  (observed: 0.066 tok/s death spiral at 92 GB unlocked).
- **Miss reads must be F_NOCACHE.** Cached, they recruit the page cache as
  a competitor for the same RAM the arena is wired into.
- **Size against three ceilings**: the GPU working-set budget, a physical-
  RAM fraction (`KQ_DECODE_ARENA_RAM_FRAC`), and reclaimable-right-now
  memory (vm_stat) so a machine already busy with other workloads offers
  the arena less - minus spine and KV reserve, capped at the expert bytes.
- **Do not background-seed the arena.** A demand miss is a perfectly
  targeted read; a seed is a popularity guess, and the SSD is saturated by
  demand misses during exactly the window seeding would help. Measured
  net-negative; organic fill converges in a few dozen tokens.

### Pressure-adaptive shrink (landed)

The load-time ceilings answer a machine that is busy *at load*; pressure
arriving afterward (another model, a build) meets a wired arena the kernel
cannot reclaim. The feeder now polls
`kern.memorystatus_vm_pressure_level` between stage calls and steps the
arena target down under warning/critical (two steps at critical), floored
at a quarter of the sized capacity, regrowing a step at a time after
sustained normal pressure *and* measured reclaimable-RAM headroom.
`KQ_DECODE_PRESSURE=0` opts out.

The original sketch here - munlock + `MADV_FREE` the coldest slots in
place - does not work on the GPU path: gathers reference the layer's whole
Metal buffer every token, so its pages stay GPU-resident whatever madvise
says. The landed mechanism instead *reallocates* each layer's arena
smaller at that layer's own stage call (the stage-A eval fence makes the
swap safe), copying the layer's most popular residents across - so a
shrink costs the LFU-cold tail, never the hot set, and the transient
old+new overlap is bounded to one layer's stacks. `arena_alloc`'s deleter
frees to the OS directly (no allocator cache), which is what makes the
reallocation an actual release.

Validated live against the 162 GB MiniMax on the M3 Max: an inducer
holding incompressible memory flipped the kernel to warning, the arena
stepped down (one step per still-growing pressure source poll, to the
floor under sustained growth), swap never moved, decode never stalled,
and the generation stayed byte-identical. The regrow headroom check must
count only free + speculative + purgeable pages - counting inactive
(which includes other processes' anon memory, reclaimable only through
swap) let a 17 GB regrow push ~11 GB of anon to swap before the check
was tightened.

## Remaining work

1. **NAX comparison**: measure the NAX rhs leaf vs the new ALU kernel on an
   M5-class GPU (`KQ_DISABLE_NAX=1` now falls back to the tuned ALU kernel
   instead of the per-row path, so the A/B is clean).
2. **Decode feeder stage B** (the encoded event pair): pre-encoded
   per-token command buffers with `kq.event_signal`/`kq.event_wait` and
   watchdog poison, removing stage A's per-layer eval tax (~1-10 ms/token
   against a 200+ ms IO-bound token; only worth building if profiling shows
   the sync tax matters at higher residency). Caution: one command buffer
   referencing the whole arena re-enters the large-single-buffer wiring
   regime the Metal watchdog kills; stage B needs the arena referenced
   per-layer or a residency set (`MTLResidencySet`, macOS 15+), and a
   poisoned token is recomputed, not skipped (its input id was fixed
   before encoding; the retry overwrites the same KV offsets).

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
