# Kernel reference

The `kq.*` namespace has two tiers. The **core codec ops** - `quantize`, `dequantize`,
`quantized_matmul`, `gather_qmm` - are the general K-quant surface, documented in the
[README](../README.md) and [integration.md](integration.md); everything a downstream project needs to
store and multiply K-quant weights is there. On top of them sits a set of **fused and
architecture-specific kernels**: decode/prefill fusions that collapse several ops into one dispatch,
and a cluster of sparse-attention kernels for the DeepSeek/GLM lightning-indexer attention. This page
catalogs that second tier.

Kernels are named for what they compute, not the model that first needed them. A fusion motivated by
one architecture (a norm layout, an activation, an attention shape) is written as a general kernel and
reused wherever the shape recurs; the entries below note the motivating regime. The one exception is
the `dsa_*` group, which implements a specific attention mechanism (DeepSeek-V4-Flash / GLM) end to
end and is scoped to it.

Each op's Python docstring carries the full argument contract, shape constraints, and dtype rules
(`help(kq.<name>)`); the one-liners here are a map, not a spec.

## Quantized matmul and MoE gather fusions

General K-quant matmul paths beyond the core `quantized_matmul` / `gather_qmm`, for the shapes those
two leave on the table (single-row decode, expert-sorted prefill, fused bias/mix).

- **`quantized_matmul_qmv_bias`** - `x @ dequant(w) + bias` with the bias add fused into the matvec.
  Decode-only (single row); `q8_0` for now, other codecs fall through to matmul-then-add.
- **`gather_qmv_kq`** - gathered matvec for an expert stack, one activation row per expert slot: the
  MoE down projection at decode. Takes an optional per-expert bias for the fp4 wire codecs
  (`mxfp4`/`nvfp4`, gpt-oss experts).
- **`gather_qmv_mix_kq`** / **`gather_qmv_mix_ns_kq`** - the down projection with the routing mix
  folded in - every slot accumulated in f32 weighted by its score, and (in the `mix` variant) a shared
  expert as the last slots - replacing a gather plus a weighted sum plus the shared-expert add. Shaped
  for DeepSeek-V3/V4-style shared-expert MoE.
- **`gather_qmv_bias`** - gathered matvec with a fused expert bias on MLX's packed mxfp4 layout (the
  counterpart to the K-quant gathers above for that codec).
- **LoRA epilogue** - `quantized_matmul`, `quantized_matmul_qmv_bias`, `gather_qmv_kq` and
  `gather_qmv_mix_ns_kq` accept `lora_a` / `lora_b` (plus `lora_rows`, and `lora_ids` / `lora_table`
  on the gathers) and add `rows * (x @ A) @ B` on their own output inside the primitive. Rows
  routes run one epilogue dispatch after the base route; the score-mixed gather forms `z = x @ A`
  in a kernel dispatched before the base gather (it reads only x and A, so the encoder overlaps the
  two) and applies `z @ B` in a barrier-free kernel after it, so the serialized cost behind the
  gather is one tiny dependent kernel. Codec independent, so a live adapter adds no graph ops at
  decode on any codec; `lora_table` remaps gathered ids (arena slot to expert, negative skips) and
  `lora_rows` scales rows (0 skips). f16/bf16 activations, rank x slots <= 512, inference only.
  `mlx_kquant.HAS_LORA_EPILOGUE` marks the build; `KQuantLinear(x, lora=(a_t, b_t, rows))` forwards.
  The epilogue kernels walk every LoRA operand as dense row-major memory; operands that evaluate
  to a strided view (an unevaluated array reports the default dense layout at op build, and
  `mx.repeat` of one value evaluates to a stride-0 broadcast) are copied into a dense temporary
  by a small `kq_lora_densify` dispatch at eval, on the primitive's own stream.
- **`gather_qmm_seg`** + **`expert_tile_map`** - expert-sorted MoE prefill as one GEMM per expert
  segment instead of per-row gathers. `expert_tile_map` builds the 64-row tile map on the GPU from the
  sorted routing indices (no host sync); `gather_qmm_seg` walks it. On NAX GPUs the walk is a NAX
  tile kernel: each threadgroup owns one 64-row tile of one expert, dequantizes its weight slab once
  and runs one MMA pass per 16-row sub-band a partial tile fills, skipping the rest. The fixed-tile
  `gather_qmm_rhs_nax` leaf pays that dequant and MMA walk once per expert segment a tile touches,
  which at ~60 rows per expert (a 2048-token chunk over 288 experts) is ~2x; the seg kernel runs
  1.5-1.7x faster there. `KQ_DISABLE_GATHER_SEG_NAX=1` forces the steel simdgroup-mma walk. Gated by
  `KQ_SWITCH_GEMM_MIN_ROWS` (see [README](../README.md#environment-variables)).

On NAX GPUs, `quantized_matmul` transpose (decode-orientation) shapes route by row count M and
output width N. Per-row qmv serves M 2 up to a per-codec limit, the mat-vec paths (`mv_ext`,
`verify_qmv`) serve the widths above it, and split-K on a BM=32 double-buffered NAX tile takes over
from a per-codec entry through M 32. From M 33 to 128, split-K also takes the calls where the BM=64
grid wastes the GPU, as described below. Elsewhere above M 32 the BM=64 tile runs, with a
double-buffered `_db` variant on the M 33-64 band at large N, and a BM=128 tile from M 193 when
ceil(M/64) is even. The k-quants `q3_k` to `q6_k`, the flat block codecs `q4_0`, `q4_1` and `q5_0`,
and `iq4_nl`, `iq3_xxs`, `iq3_s`, `iq2_xs` and `iq2_s` run the BM=64 and BM=128 tiles on a K tile of
128 (the `_BK128` levers). Above the qmv limit, the verify kernels take the widths through M 8 on
the codecs that have them. The NAX verify kernel serves `pq2_0`, `q4_0`, `q4_1`, `q8_0`, `q2_k` to
`q6_k` and `nvfp4` from a per-codec entry, and the register-resident MMA verify kernel serves
`ptq1_0`, as described at the end of this section. Every limit and floor is a measured per-codec
policy (`kq_nax_small_m` and `kq_smallbm_policy` in `src/kquant_matmul.cpp`).

On GPUs without NAX, the `qmm_t` kernel decodes each weight tile inside its K loop, once for every
64-row tile of the activations. From a floor in rows it is faster to decode the whole weight once
with `kq.dequantize` and run the stock matmul. A transposed `quantized_matmul` on a 2-D weight takes
that route on the GPUs without NAX before M5 when its codec has a row below, its rows of x reach the
floor (a batch counts all of its rows), K is from 3840 to 12288 and N from 3840 to 24576
(`kDequantizeRows` in `src/kquant_ops.cpp`). Those GPUs are the ones whose architecture string is
`applegpu_g<gen><suffix>` with a generation from 13 to 16 (M1 to M4) and a suffix other than `p`
(phones), without NAX. Any other string keeps the kernel, as do NAX GPUs, CPU streams, calls with the
LoRA epilogue arguments, a 3-D weight, `transpose=False` and every call while `KQ_QMM_TILE` is set.
`gather_qmm` without indices on a 2-D weight is the plain call and takes the route.
`quantized_matmul_glu`, `quantized_matmul_lora` and `quantized_matmul_residual` build their
products from `quantized_matmul` on these GPUs, so they take it too.

| codecs | rows from | K | N |
|---|---|---|---|
| `q3_k`, `q5_k`, `q6_k`, `nvfp4`, `ptq1_0` | 2048 | 3840 to 12288 | 3840 to 24576 |
| `q4_k` | 16384 | 3840 to 12288 | 3840 to 24576 |

The route holds a copy of the weight in the activation dtype while the call runs, N x K x 2 bytes
per call, at most 604 MB at 24576 x 12288; the range keeps a vocabulary head on the kernel. Each
routed call holds its copy until its command buffer completes, so consecutive routed layers can each
hold one at the same time.
`quantized_matmul_glu` builds its product from two calls on these GPUs, so it holds two copies, up
to about 1.2 GB. The gradient wrt x runs the quantized matmul the other way, as on the kernel path,
so a backward pass holds no dense copy, and a gradient wrt the weight or the scales raises as it
does there. The choice is made when the op builds its graph, so a
function compiled with `shapeless=True` keeps the choice of its trace at other row counts. On the
output elements above 1% of the root mean square of a float64 reference, the route is within 1 ulp
of the kernel in bfloat16 and within 2 ulp in float16, and its error against that reference is
within 1% of the kernel's. Nearer zero, where the terms of a sum cancel, the two summation orders
differ by more ulp.

The rows and the range were measured on M3 Max (`applegpu_g15s`, 40 GPU cores), and the other GPUs
of the class take them unmeasured. Measured with `benchmarks/bench_dit_shapes.py` over the seven
Z-Image and Qwen-Image linears at 2048, 4128 and 16416 rows, batch 1 and 2, bfloat16 and float16,
against an A versus A arm that stays within 0.98-1.02 per step: the kernel takes 1.49-1.63x the
route's time per step on `nvfp4`, 1.06-1.14x on `ptq1_0`, 1.03-1.12x on `q3_k` and `q6_k`,
1.01-1.10x on `q5_k` (1.01-1.03x at 2048 rows) and 1.02-1.05x on `q4_k` from its floor. No linear
runs slower outside the A versus A range, and the route takes 0.96-1.07x the time of the stock
matmul on unquantized weights.

The decode pass runs at 1.0-1.15x the time of a copy of its output and costs the same at any row
count, so the route loses below its floor: at 128 rows it takes 1.4-1.9x the kernel's time. On the
other codecs the route measured inside the noise of an A versus A arm at 2048 to 16416 rows, and on
`q8_0` it gains 1-4% of a step, also inside it; they keep the kernel. A larger kernel tile does not
help on this chip: five tiles of 128 rows or columns ran 0.98-1.27x the time of bm64 bn64 per cell
at 4128 rows, slower on 51 of 56 cells of the two closest, against an A versus A range of
0.98-1.03.

Per-row qmv reads the weight once per activation row, the later rows from the cache, and its grid
grows with M. It holds widest where the mat-vec grids are too small to fill the GPU, so its limit
falls with N. At N 512 and below, 13 of the 22 NAX codecs keep it through M 7 or 8. At N 1024 most
keep it through M 2 to 5, and above N 2048 only `pq2_0`, `ptq1_0`, `q4_0`, `q4_1`, `q4_k` and
`q2_k` keep it, at M 2. Split-K enters at M 2 to 9 for N 1024 and below and at M 3 to 9 above.
Above N 1024, `iq2_xxs` keeps `mv_ext` through M 12, where its split-K tile runs slower, and `q6_k`
at vocab-head widths runs the un-split BM=32 tile at M 8 on the calls `verify_nax` declines.

The table was fitted on M5 Max to the fastest route per cell over the 22 codecs and 14 shapes, M 2
to 12 on the projection shapes and M 2 to 8 on the smaller ones, with the weights streamed from
DRAM (`benchmarks/bench_verify_routes.py`). Its route runs at 1.01x the fastest on those cells and
1.02x on 10 shapes held out of the fit (geometric means), against 1.24x on both for the entries it
replaced. The largest gains are `q4_k` and `q4_1` at M 7 (1.7-1.9x at the projection shapes),
`iq2_xs` and `iq2_s` at M 8 to 12 (2.1-2.2x), and qmv at M 2 on N 1024 and below (1.1-3x).
`q4_1` was fitted before it had verify kernels. It now takes the `q4_0` row, which was fitted with
them, and is not measured on it.

A NAX tile runs two rows of simdgroups. When the whole matmul has no more rows than one of them
covers, 16 on the BM=32 tile and 32 on the BM=64 tile, the second row would multiply only padding.
Both rows then compute the live rows, each walks half of every K step, and the two partial results
are summed in threadgroup memory before the store. Measured on M5 Max over the 22 NAX codecs with
the weights streamed from DRAM, the NAX split-K route runs 1.05-1.3x faster at M 8 to 16 and the
un-split tile 1.1-1.5x faster at M 7 to 16. On the BM=64 tile, `iq2_xs`, `iq2_s` and `iq1_m` run
1.3-1.5x faster at M 13 to 32, and the gathered NAX tile runs about 1.3x faster at 32 rows per
entry (median over nine codecs). On the un-split tiles, the short last row tile of a taller matmul
keeps the plain walk, because the full tiles set the time there and the split measured slower. The
split-K kernel splits every row tile that fits one row of simdgroups, since its grid runs many
waves and each padded multiply adds to the total.

From M 33 to 128 the BM=64 grid is ceil(N/64) x ceil(M/64) threadgroups, and each one walks all of
K. Under three threadgroups per GPU core, as on the narrow projections of a short prompt, most cores
sit idle, and split-K runs with a target of 16 slices. From M 65 to 96 the last row tile holds at
most 32 live rows, which the split-K kernel runs at half the padded multiplies, so split-K runs
there with a target of 8 slices up to 16 threadgroups per core. The split-K tile is BM=32 through
M 48, where its last row tile holds at most 16 rows, and BM=64 above. On M5 Max over the 22 NAX
codecs and eight projection shapes, with the weights streamed from DRAM, this runs 1.29x faster
than the BM=64 tile when each shape is weighted by its calls in a forward of a 27B hybrid model,
and no cell runs more than 2% slower. Whole forwards of that model run 19-22% faster at 33 to 64
rows and 17% faster at 80 and 96. `KQ_QMM_MIDM=0` keeps the BM=64 tile.

With float partials (the float16 default, and `KQ_SPLITK_F32=1`), `q4_0`, `q4_1` and `q8_0` target
10 slices from M 33 to 64, and from M 65 to 96 they target 5 up to K 5120 and 8 above. They cut K in
units of 64 weights, so at K 5120 the targets of 16 and 8 write 16 or 8 partials where the
256-weight codecs write 10 or 5, and a float partial is twice the size of a bfloat16 one. On M5 Max
over the eight projection shapes on float16 activations, each cell of shape and M takes 0.79-1.01x
the time of a target of 16 from M 33 to 64, and 0.67-1.02x the time of 16 or 8 from M 65 to 96 at K
5120. Above K 5120 a target of 5 leaves 4 slices of the 96 or 272 units, so the target there stays
8. Those cells take 0.88-0.97x the time of 4 slices, and 0.99-1.02x at K 6144 and M 96. Weighted by
calls per forward, M 65 to 96 run 0.96-0.99x the time of a target of 5 at every K. The K threshold
was measured at K 5120, 6144 and 17408 only. The rule keys on the partials, so bfloat16 with
`KQ_SPLITK_F32=1` takes these targets too, though they were not measured there. bfloat16 on its
default bfloat16 partials keeps 16 and 8. `q4_1` also cuts K in units of 64 weights and takes the
`q4_0` targets, which are not measured for it.

The NAX split-K route cuts K into units of max(block, 64) weights and targets 16 slices of whole
units. It takes the largest count at or under the target that divides the units evenly, unless
equal slices with a shorter last one give more than twice as many. This covers inner dimensions
whose unit count has no divisor near 16, such as 11008, 17408 and 18944 on the 256-weight codecs.
`KQ_SPLITK_RAGGED=2` takes the ragged count wherever it is larger.

At M 32 and below, a call whose BM=32 grid already holds 16 threadgroups per GPU core, such as a
vocab head, takes one slice where the call without the split runs the same BM=32 tile. `q6_k` and
`iq2_xxs` below M 32 and `q8_0` below M 13 keep the target of 16 there, as do calls that would leave
the BM=32 tile for an mv route or the BM=64 tile. Below a full grid, `q4_0`, `q4_1` and `q8_0` on
float partials hold the split grid to 16 threadgroups per core, except `q8_0` below M 13 (`q4_1`
takes the `q4_0` rule, not measured). On M5 Max, against a target of 16 at N 49152, 151936 and
248320 and M 9 to 32, the single slice runs 0.68-1.05x the time, and summed over a forward pass,
weighted by the calls per shape, no codec runs slower. The float-partial cap runs 0.75-1.02x at N
5120 to 17408.

Measured on M5 Max at M 8 to 24 with the weights streamed from DRAM, the shorter last slice runs
`q4_k` 1.1x faster at 5120x17408, 1.5x at 3584x18944, 2x at 4096x11008 and 4-5x at 896x4864. The
other codecs range from no change (`pq2_0` at 3584x18944) to 5.7x (`iq1_s` at 896x4864). Where the
even count is at least half the ragged one, the ragged count ran from 11% slower to 1.26x faster
depending on codec and shape, with no consistent direction, so the route keeps the even count.

The M=1 mat-vec kernels loop over their two or four output rows with a static trip count and a
clamped row index, so the compiler interleaves the rows' loads; the tail threadgroup recomputes its
last row and drops it at the store. `q2_k` and `q3_k` keep the runtime bound: the static form measured no
faster for them.

A `q6_k` `quantized_matmul` with one activation row and no batch dimensions runs a split-K kernel
when 4 divides N. The fused MoE gathers keep their own kernels. The eight simdgroups of a
threadgroup share four output rows, divide the superblocks between them and add their partial sums
in threadgroup memory. Each thread assembles four 6-bit codes at a time from 32-bit words and
pre-scales the activation for byte b of a word by 2^(-8b), so a weight costs one AND, one convert
and one FMA. The row width is a Metal function constant, so each width gets its own pipeline with
the superblock count and row stride folded. Under sustained load that ran about 1% faster than
reading the width at run time.

A `q8_0` `quantized_matmul` with one activation row and no batch dimensions runs a split-K kernel
too, when N is even and 256 divides K. The eight simdgroups of a threadgroup share two output rows.
They stride the 256-weight steps between them and add their partial sums in threadgroup memory. A
lane keeps the per-row kernels' share of a step, eight weights of one block. It reads them as four
16-bit words, since a quarter block sits 2 bytes past a 2-byte boundary. The split puts more
simdgroups in flight at layer-size N. On M3 Max, back to back over ten shapes, it took 1.6% to 6.6%
less time than the per-row kernels: 2.7% less at K 8192 and N 4096, and 3.2% less at K 4096 and N
8192. In a synced decode step of eight Muse Spark UD-IQ2_S layers, it saved 1.1% per layer.

The split-K mat-vec covers `q6_k` and `q8_0`. The other codecs run one activation row through
their per-row kernels, and the grid codecs through `qmv_c32` on M5 and later GPUs.

Under sustained decode the GPU runs below its top clock, where the per-row kernels turn ALU-bound.
Measured on M5 Max over 90 seconds of back-to-back mat-vecs at the gemma-4 31B MLP shapes, the
split-K kernel read 450-465 GB/s at the lowest clock, against 385-395 GB/s for the per-row kernels
and about 530 GB/s for bf16 GEMV. Once the clock dropped it read 1.12-1.14x the per-row rate at
inner dimensions of 4096 and 5120, and matched it at 2048 and below. In short runs at a high clock
the two kernels are within 3% of each other on every shape measured.

A `q6_k` row under 3840 weights has fewer superblock pairs than the threadgroup has simdgroups, and
a `q8_0` row under 2048 weights has fewer 256-weight steps. The threadgroup then runs one
simdgroup per pair or step, so none waits idle, and the sum reads only their partials, so the
output is bit for bit the same. On M5 Max at 0.76-0.92 GHz, `q6_k` took 12.6% less time at N 32768
and K 1280, the DeepSeek-V4.1 query up-projection, 9.0% less at N 24576 and K 1536 and 8.5% less at
N 3072 and K 1024. `q8_0` already reads near the memory rate there and was within 0.5% at the first
two, and took 4.1% less at N 3072 and K 1024.

`gather_qmm` with one activation row runs the same two split-K kernels per batch entry, for grouped
projections such as the DeepSeek-V4 attention output and for SwitchGLU decode steps, when the row
count of the tile divides N, K is 2048 or more (1024 or more for `q6_k`) and the call has fewer
than 16384 output rows (N times the entries). Under the default levers each entry then matches
`quantized_matmul` on its expert bit for bit. `KQ_GATHER_SPLITK` does not read `KQ_QMV_SPLITK=0`
or `KQ_QMV_FINE`, which move `quantized_matmul` to the per-row kernels. On M5 Max at 0.85-1.6 GHz
it ran as fast as `gather_qmv` or faster on every such shape measured: from K 2048 over 1 to 128
entries and N of 128 to 7168, and for `q6_k` at K 1024 to 1792 over 1 to 16 entries. It took 6-7%
less time for `q8_0` and 4-6% less for `q6_k` at the DeepSeek-V4 shape of 8 groups of 1024 x 4096,
and up to 38% less at one entry. From 16384 rows the `gather_qmv` grid fills the GPU, and at
sustained clocks split-K ran up to 23% slower there. Below K 2048, `q8_0` measured from 4% faster
to 4% slower than `gather_qmv`, with the sign changing from run to run. At K 768, `q6_k` took 7-14%
longer at one and two entries in every run, and up to 13% less time at 4 to 16. Both keep
`gather_qmv` there.

Tuning levers (defaults are right for normal use):

- `KQ_TUNE_LIVE` - `1` makes the NAX tile levers (`KQ_NAX_SMALL_BM`, `KQ_NAX_BM128`,
  `KQ_NAX_BM128_ODD_M`, `KQ_NAX_DB64`, the `_BK128` levers, `KQ_NAX_SWIZZLE`), the `verify_nax` load
  lever (`KQ_VNAX_X16`), the tile overrides (`KQ_NAX_TILE`, `KQ_QMM_TILE`, `KQ_CONV_TILE`),
  `KQ_SPLITK_F32`, `KQ_KQUANT_WORD` and `KQ_QMM_LOG` read their variables on every dispatch, so
  one process can switch tiles between calls on one resident copy of the weights;
  `benchmarks/bench_dit_shapes.py` runs its tile arms this way. It reads the environment on every
  dispatch, so it is for single-threaded A/B runs that set these variables between calls. The
  `quantized_matmul_glu` levers (`KQ_GLU`,
  `KQ_GLU_MIN_ROWS`), the `quantized_matmul_multi` lever (`KQ_MATMUL_MULTI`), the
  `quantized_matmul_lora` levers (`KQ_QMM_LORA`, `KQ_QMM_LORA_MIN_ROWS`), the
  `quantized_matmul_residual` lever (`KQ_QMM_RESID`), the dequantize route lever
  (`KQ_DEQUANT_ROUTE`) and the `conv2d_3x3` lever (`KQ_CONV_NAX`) are read when the op is called and
  builds its graph, not when a kernel is dispatched, so a compiled function keeps the kernel choice
  of its trace. Unset (off) by default, and then each of them keeps its first read for the life of
  the process. Read once per process.
- `KQ_NAX_TILE` - `bm,bn,wm,wn` pins the NAX `qmm_t` tile of every transpose call that reaches it,
  in place of the BM=32, BM=64 and BM=128 policy. BK stays 64, except where a `_BK128` lever sets
  it. The `_db` variant still follows `KQ_NAX_DB64` on the bm64 bn64 wm2 wn2 tile. A tile the
  metallib does not carry raises a `ValueError` that names the missing kernel, and
  `kq.metallib_kernels()` lists the ones it carries. Unset (off) by default. Read once per process
  unless `KQ_TUNE_LIVE=1`.
- `KQ_QMM_TILE` - the same override for the `qmm_t` tile on the path without NAX (GPUs without
  NAX, `KQ_DISABLE_NAX=1`, or a codec or shape the NAX tile does not serve). The metallib
  carries only the shipped tile, so `64,64,2,2` is the one value that runs today; it keeps its
  plain kernel name. Any other tile is looked up under a name that carries
  `_bm<bm>_bn<bn>_wm<wm>_wn<wn>` before `_alN` and raises until such a kernel is built. Set, it
  also keeps calls on the kernel where the dequantize route would take them (see above), and that
  choice is read when the op builds its graph. Unset (off) by default. Read once per process
  unless `KQ_TUNE_LIVE=1`.
- `KQ_DEQUANT_ROUTE` - `auto` runs the dequantize route described above on the GPUs it is for,
  `off` keeps the `qmm_t` kernel on every GPU, and `force` runs the route on any GPU, NAX GPUs
  included, for the codecs, rows and range of its table, so that tests run its code on any
  machine. Any other value warns once and reads as `auto`. `auto` by default. Read when the op
  builds its graph, once per process unless `KQ_TUNE_LIVE=1`.
- `KQ_QMM_LOG` - set to any value, logs every quantized matmul dispatch to stderr, one
  `[kq qmm] <kernel> M=<m> N=<n> K=<k> grid=<x>,<y>,<z> tg=<x>,<y>,<z> xa=<a>` line each: the
  NAX and ALU qmm tiles, the mat-vec routes (`qmv`, `qvm`, `verify_qmv`, `mv_ext`), split-K,
  `verify_mma` and `verify_nax`, split or not, the fused `quantized_matmul_glu`,
  `quantized_matmul_lora` and `quantized_matmul_residual` kernels, `skinny_matmul`, the
  `gather_qmm` and `gather_qmm_seg` routes, the fused MoE gathers, `gather_mix` and `conv2d_3x3`.
  The grid depth is the K slice count, and `tg` is the threadgroup size in threads. `xa` is the
  byte offset of the activation's start modulo 16, as the kernel reads it. A
  gather or MoE line gives the rows and last dim of the output as M and N. Any GPU op that copies
  an operand to an aligned buffer, because the operand starts or steps off the boundary of its
  kernel's vector loads, logs one `[kq align] <op> <operand> xin=<n>` line, where `n` is the
  operand's byte offset modulo that boundary. A call that takes the dequantize route logs one
  `[kq dequant] <codec> M=<rows> N=<n> K=<k>` line when the op builds its graph. Under
  `KQ_KQUANT_WORD=0`, each fetch of a kernel on the per-byte k-quant decode logs one
  `[kq decode] byte <key>` line, where the pipeline key ends in `_kw0`. Unset (off) by default.
  Read once per process unless `KQ_TUNE_LIVE=1`.
- `KQ_SPLITK_F32` - partial dtype of the split-K routes (`splitk`, `nax_splitk`, the
  `KQ_QMM_MIDM` band, and `verify_mma` and `verify_nax` when they split K). Float partials let the
  fold round the result once, as the unsplit kernels do. Partials in the activation dtype round
  each slice once more before the fold, and where one K slice carries most of an output, that puts
  the split result more than one output rounding from the unsplit one. On float16 they also stop at
  65504, so a slice sum past it turns to inf even where the output is finite. Unset, float16
  activations store float partials and bfloat16 activations keep the previous bfloat16 partials.
  `1` stores float partials on both, and `0` stores partials in the activation dtype on both.
  `true`, `on` and `yes` read as `1`, and `false`, `off` and `no` as `0`. An empty value reads as
  unset, and any other value reads as unset with a warning on stderr. Read once per process unless
  `KQ_TUNE_LIVE=1`.
- `KQ_NAX_SMALL_BM` - small-M routing. `0` restores the old routing (mat-vec paths below M 13 and
  no BM=32 tile), `2` forces BM=32 for policy-excluded codecs, unset or `1` follows the per-codec
  policy. Read once per process unless `KQ_TUNE_LIVE=1`.
- `KQ_NAX_BM128` - BM=128 band. `0` off, `1` forces the floor to M 193 for every codec, `2` drops
  the floor entirely (any even ceil(M/64), probing the M65-128 wash band), unset follows the
  per-codec entry floors (193/449/961 tiers, measured on M5 Max by
  `benchmarks/bench_qmm_bm128_ab.py`; re-run it before trusting them on new silicon). Read once
  per process unless `KQ_TUNE_LIVE=1`.
- `KQ_NAX_BM128_ODD_M` - the M from which the BM=128 band also takes an odd ceil(M/64), whose last
  row tile holds 64 rows or fewer. There the simdgroup row with no row below M skips its
  activation loads, matrix work and store, so that tile does the loads and matrix work of one
  BM=64 tile. Unset, the codec's large-M row sets it (see below). `0` keeps the even ceil(M/64)
  rule at every M, the policy before this lever, and turns off the large-M rows with their K
  ceiling, so BM=128 also runs above the ceiling at an even ceil(M/64). Read once per process
  unless `KQ_TUNE_LIVE=1`.
- `KQ_NAX_DB64` - double-buffered M 33-64 band. `0` off, `1` drops the N floor, unset follows the
  per-codec N floors. Only the five policy-enabled codecs (q6_k, q8_0, q4_1, q5_1, q5_0) carry
  `_db` instantiations, so `1` is bounded by availability; probing another codec needs its
  instantiation restored and a metallib rebuild. Read once per process unless `KQ_TUNE_LIVE=1`.
- `KQ_NAX_<codec>_BK128` (`KQ_NAX_Q6K_BK128`, `KQ_NAX_Q5K_BK128`, `KQ_NAX_Q4K_BK128`,
  `KQ_NAX_Q3K_BK128`, `KQ_NAX_Q4_0_BK128`, `KQ_NAX_Q4_1_BK128`, `KQ_NAX_Q5_0_BK128`,
  `KQ_NAX_IQ4_NL_BK128`, `KQ_NAX_IQ3_XXS_BK128`, `KQ_NAX_IQ3_S_BK128`, `KQ_NAX_IQ2_XS_BK128`,
  `KQ_NAX_IQ2_S_BK128`) - K tile of the codec on the BM=64 and BM=128 tiles. Unset or `1`, the tile
  walks K 128 weights per step. On the k-quants it takes one half of a super-block per step, and
  each thread reads its code bytes once, where BK=64 reads them on two or four steps. On the other
  codecs two BK=64 loaders fill the tile, which halves the barriers per K, and a 32-weight block
  codec needs K % 128 == 0. The output is bit-identical. On an M5 Max, per cell against BK=64 on
  bfloat16 and float16 activations, on the BM=128 tile at diffusion model shapes (M 4096 and 4608, N
  and K 3840 to 24576): `q6_k` 0.87-0.96x the time, `q5_k` 0.93-1.00x, `q3_k` 0.92-1.03x. On the
  BM=64 tile from M 128 to 2112 at N and K 3840 to 17408: `q6_k` 0.79-1.01x, `q5_k` 0.89-1.00x,
  `q3_k` 0.88-1.00x, except on grids of 240 threadgroups (N 5120 at M 160 and 192), where `q5_k`
  takes 1.06-1.17x and `q3_k` 1.01-1.06x, and of 480 (N 10240 at M 160 and 192), where `q5_k` takes
  1.02-1.06x. No threshold on threadgroups per core, K or M removes these at a lower total time. At
  the diffusion model rows (M 4128 to 16896 at N and K 3840 to 12288, bfloat16 and float16, with the
  K ceilings below), against an A/A spread of 0.98-1.03x: `q4_k` 0.95-1.00x, `q4_0` 0.97-1.01x,
  `q4_1` 0.97-1.03x, `q5_0` 0.96-1.01x, `iq4_nl` 0.93-1.00x, `iq3_xxs` 0.93-1.02x, `iq3_s`
  0.92-1.03x, `iq2_xs` and `iq2_s` 0.95-1.01x. `q8_0`, `q5_1`, `iq4_xs`, `iq1_m` and `iq1_s` took
  1.01-1.06x at some of those cells and `iq2_xxs` 1.04-1.07x at all of them, so they keep BK=64 and
  carry no `_bk128` kernels. `0` keeps BK=64. The BM=32 tile and the `_db` variant keep BK=64. Read
  once per process unless `KQ_TUNE_LIVE=1`.
- `KQ_FORCE_QMM_MIN_M` - probe lever: routes transpose shapes with M at or above the value straight
  to the NAX qmm, bypassing the mat-vec route claims, for crossover measurement below M 13. Unset
  (off) by default.
- `KQ_QMM_ROUTE` - probe lever: forces one small-M route (`qmv`, `verify_qmv`, `mv_ext`,
  `verify_mma`, `verify_nax`, `splitk`, `nax`, `nax_splitk`) for transpose, non-batched shapes
  with M <= 32 where that route serves the codec and shape. Any other call takes the default
  routing. Read live per call, so `benchmarks/bench_verify_routes.py` times every route in one
  process. Unset (off) by default.
- `KQ_QMM_ROUTE_STRICT` - `1` makes a transpose call that the forced `KQ_QMM_ROUTE` cannot serve
  raise instead of taking the default routing, so a sweep never times the default under a forced
  route's name. Off by default.
- `KQ_NAX_SWIZZLE` - row-tile traversal swizzle on the BM=64 and BM=128 tiles: `n` folds 2^n row
  tiles into grid.x, so consecutive threadgroups share one weight column band, capped at the
  row-tile count. `0` is off. Unset, the codec's large-M row sets the fold on that row's tile and
  nothing else folds. Read once per process unless `KQ_TUNE_LIVE=1`.
- `KQ_MV_EXT_NR` - `2` selects the two-rows-per-thread `mv_ext` variant (q6_k, M 5-12), which
  halves activation cache traffic but measured no faster than the shipped kernels. Kept as a probe
  for future silicon. Default `1` (shipped behavior).
- `KQ_NAX_QMV` - per-row qmv at small M on NAX GPUs. `0` disables the route, a value of `2` or
  more serves M 2 through that value at every N. Unset takes the per-codec limit for the call's N
  in `kq_nax_small_m`, which yields to a forced `KQ_VERIFY_NAX` or `KQ_VERIFY_MMA` on a codec they
  serve, a forced `KQ_QMM_SPLITK_NAX` or `KQ_QMM_SPLITK`, and a set `KQ_VERIFY_EXT`.
  `KQ_DISABLE_NAX=1` turns it off. Read live per call.
- `KQ_QMV_FINE` - tiling of the per-row mat-vec. `1` forces two output rows per threadgroup, `0`
  forces the coarse tiling, and unset takes the fine tiling at one activation row up to the
  per-codec N ceiling that `kq.qmv_fine_max_n(codec)` reports. Read live per call.
- `KQ_QMV_C32` - the c32 mat-vec of the grid codecs at one activation row (see the fused MoE
  section). `0` restores the per-row kernels and `1` runs c32 on any GPU. `8` or `16` runs it at
  that many K-lanes per row. Unset takes it on M5 and later GPUs. A set `KQ_QMV_FINE` keeps the
  per-row kernels. Read live per call.
- `KQ_GATHER_SPLITK` - the `q6_k` and `q8_0` split-K kernels for `gather_qmm` at one activation
  row. `0` restores `gather_qmv`, `1` takes split-K at any K that 256 divides, and unset follows
  the K and row-count rule above. `KQ_QMV_SPLITK` and `KQ_QMV_FINE` do not change it. Read live
  per call.
- `KQ_QMV_SPLITK` - the `q6_k` and `q8_0` split-K kernels for one activation row. `0` restores
  the per-row kernels, as does a set `KQ_QMV_FINE`, so a tiling A/B keeps measuring the tiling it
  names. A lever that forces another route at one row, such as `KQ_VERIFY_NAX=1`, takes
  precedence. Read live per call.
- `KQ_QMV_SPLITK_TRIM` - `0` launches the `q6_k` and `q8_0` split-K kernels with all eight
  simdgroups at every K, as before, with the same output. Read live per call.
- `KQ_VERIFY_EXT` - the mat-vec route on the M 2-12 band. `1` forces `mv_ext` for every codec with
  the kernel, `0` forces `verify_qmv` where the codec has it and per-row qmv elsewhere, and unset
  takes the per-codec default. On NAX GPUs a set value also keeps per-row qmv and NAX split-K off
  the band. Read once per process.
- `KQ_QMM_SPLITK_NAX` - split-K on the NAX BM=32 tile. `0` disables the route, `1` forces it at
  the default target of 16 slices, and a larger value forces it with that target. Unset takes the
  per-codec entry M in `kq_nax_small_m` (one for N <= 1024, one above), measured on M5 Max, which
  yields to a forced `KQ_QMM_SPLITK` and to a set `KQ_VERIFY_EXT` through M 12. Every codec with
  NAX kernels, M <= 32. `0` also turns off the `KQ_QMM_MIDM` route on NAX GPUs, and a larger value
  sets its slice target. Read live per call, so both arms can share one process.
- `KQ_QMM_SPLITK` - the same lever for the plain small-M qmm, used when NAX is absent or disabled.
  Entry points come from a per-device table. K-quants, legacy quants and the IQ codecs, M <= 32.
  `0` also turns off the `KQ_QMM_MIDM` route, and a larger value sets its slice target.
- `KQ_QMM_MIDM` - split-K for M 33-128. `0` keeps the BM=64 qmm and `1` forces split-K at every M in
  the band. Unset takes split-K where the qmm grid, ceil(N/64) x ceil(M/64) threadgroups, is under
  three per GPU core (`KQ_GPU_CORES` sets the count), such as the narrow projections of a short
  prompt. On NAX GPUs it also takes M 65-96 up to 16 threadgroups per core and runs the NAX split-K
  tile on every codec with NAX kernels, and there `q4_0`, `q4_1` and `q8_0` on float partials target
  10 slices through M 64, and from M 65 to 96 5 up to K 5120 and 8 above. When NAX is absent or
  disabled it runs the BM=32 tile on the codecs of `KQ_QMM_SPLITK`. Read live per call.
- `KQ_SPLITK_RAGGED` - slice count of the NAX split-K route. `0` keeps equal slices, the largest
  count at or under the target that divides the slice units, and `2` takes the ragged count
  whenever it is larger. Unset takes it only where it more than doubles the equal count. Read live
  per call.
- `KQ_SPLITK_CAP` - probe lever: caps the default slice target of the small-M split-K routes
  (`nax_splitk` and the ALU `splitk` route, M <= 32) so the split grid holds at most that many
  threadgroups per GPU core (`KQ_GPU_CORES` sets the count). A call whose grid already holds that
  many takes one slice and runs what it runs with the split route off. From M 13 that is usually
  the BM=32 tile, or per-row qmv below the qmv batch limit, which is 18 on M5 Max at K and N up to
  2048 and up to 32 on Ultra parts. At M 9 to 12 it is the small-M qmm route, `mv_ext` or per-row
  qmv. The ALU route falls through the same way, to the ALU qmm tile in place of the NAX one.
  `verify_mma` and `verify_nax` are not capped. `0` keeps the target of 16 at every N on both
  routes, so a wide N such as a vocab head writes up to 16 partials. Unset keeps it on the ALU route
  and takes the NAX default described with the NAX split-K route. A target forced through
  `KQ_QMM_SPLITK_NAX` or `KQ_QMM_SPLITK` is not capped. Read live per call.
- `KQ_VERIFY_MMA` - the register-resident MMA verify route (below). A value of `2` or more forces
  it at that M and above on any GPU, `0` or `1` disables it. Unset takes the per-codec entry from
  the device's table (`kq_verify_mma_min_m` without NAX, `kq_verify_mma_min_m_nax` with it); the
  table follows the hardware, so `KQ_DISABLE_NAX` does not change it. The NAX split-K default
  entry yields to the route, but a forced `KQ_QMM_SPLITK_NAX` takes precedence. On NAX GPUs
  `verify_nax` is decided first unless this lever forces the route at the call's M. M <= 8, read
  live per call.
- `KQ_VERIFY_NAX` - the register-fed NAX verify route (below). A value of `1` or more forces it at
  that M and above, `0` disables it. Unset takes the per-codec entry in `kq_verify_nax_min_m`
  (listed with the kernels below), which yields to a forced `KQ_VERIFY_MMA` on a codec
  `verify_mma` serves and to a forced `KQ_QMM_SPLITK_NAX`. NAX GPUs, the codecs listed below,
  M <= 8. `KQ_DISABLE_NAX=1` turns it off. Read live per call.
- `KQ_VERIFY_NAX_SPLITS` - probe lever: the split-K target for `verify_nax` in place of the
  per-codec rule (the largest divisor of the K steps at or under the value is used). Unset (off)
  by default.
- `KQ_VERIFY_MMA_SPLITS` - probe lever: the same split-K target for `verify_mma`; `1` runs it
  unsplit. Unset (off) by default. Read live per call.
- `KQ_VERIFY_MMA_HALF` - probe lever: `1` runs the `verify_mma` half kernels (name suffix `_h`) of
  pq2_0, q2_0, q4_0 and q8_0, which stage the activations as half and sum each block in half, so an
  activation or a block sum past 65504 turns to inf. On pq2_0 the codes enter that sum unscaled, so
  a 128-weight block can pass 65504 with activations of magnitude 256 when the codes and activations
  agree in sign. ptq1_0 and q4_1 have no half kernels and stay on their float kernels. For timing
  against the float kernels only. Unset (off) by default. Read live per call.
- `KQ_VERIFY_NAX_Q4_0_SB` - probe lever: `0` keeps `q4_0` on the two-block `verify_nax` kernel at
  a K that is a multiple of 256, where the eight-block kernel runs by default. Read live per call.
- `KQ_VERIFY_NAX_Q4_1_SB` - probe lever: `0` keeps `q4_1` on the two-block `verify_nax` kernel at
  a K that is a multiple of 256, where the eight-block kernel runs by default. Read live per call.
- `KQ_VNAX_X16` - probe lever: `0` keeps the `verify_nax` kernels that load their activations per
  NAX op on one 8-byte load per op where the activations start on a 16-byte boundary, in place of
  one 16-byte load per two ops. The output is the same. In `KQ_QMM_LOG` the kernel name
  carries the variant, `_x16` or `_x8`. Read once per process unless `KQ_TUNE_LIVE=1`.
- `KQ_MV_EXT_SB` / `KQ_MV_EXT_NX` / `KQ_MV_EXT_HD` - `mv_ext` activation-traffic experiments:
  shuffle-broadcast (`1`), wide nxpsg (`16`/`32`), half-precision chunk dots (`1`). q6_k M 4-12
  only. `HD` measured +4-5% at M 8; the rest flat to negative on M5 Max. Kept as probes. Default
  off.

Large-M rows. On NAX GPUs a transpose call at M 4096 or more takes the codec's large-M row from
`kq_largem_row` in `src/kquant_matmul.cpp`: the tile, the row-tile fold, the shape floor (M at or
above min M at either ceil(M/64) parity, N at or above min N) and the K ceiling, above which the
BM=64 tile runs at either parity. A codec whose tile and floor are not measured takes those of its
loader class (the 256-weight k-quant loaders, the 32-weight flat block loaders, or the IQ grid and
ternary loaders) and is marked inherited. The measured rows come from
`benchmarks/bench_dit_shapes.py` on an M5 Max at the diffusion transformer rows (M 4096 to 33792
at the Z-Image and Qwen-Image projection shapes). There the BM=128 tile takes 0.85-0.99x the
BM=64 time at rows with an odd ceil(M/64) up to the K ceiling, and 1.01-1.09x above it (`q4_k`
and `q8_0` at K 12288), and per projection the median fold of 2 row tiles takes 1.00-1.26x the
unfolded time, of 4 row tiles 1.03-1.58x.

The K ceiling is measured for every codec with NAX kernels at N 4096 and M 4128, 8256 and 16384:
it is the largest K before the first K (of 10240, 12288 and 16384) where BM=128 takes more than
1.01x the BM=64 time over the three M, measured on the codec's default K tile. Every codec wins at
K 3840 and 4096. On the K tile of 128, `q3_k`, `q5_k`, `q6_k`, `iq3_xxs`, `iq3_s`, `iq2_xs` and
`iq2_s` win at every K measured (`iq3_s` 0.91-0.94x, the other IQ codecs 0.93-1.00x), while the
flat block codecs lose at K 12288 and 16384: `q4_0` and `q4_1` 1.02-1.06x on the K tile of 128,
`q5_1` and `q8_0` 1.04-1.11x on 64. `nvfp4` on the K tile of 64 takes 0.89-0.96x over the three M at
every K, the median of three runs. A ceiling moves only where the new tile takes at most the old
time at every M measured: on the K tile of 128 at K 12288, `q4_k` and `q5_0` take 1.00-1.02x over
the three M and 1.02-1.03x at M 4128, so they keep K 10240.

| Codecs | Tile (bm, bn, wm, wn) | Fold log | Min M | Min N | Max K | Row |
|---|---|---|---|---|---|---|
| `q4_k`, `q8_0` | 128, 64, 2, 2 | 0 | 4096 | 0 | 10240 | measured |
| `q5_k`, `q6_k` | 128, 64, 2, 2 | 0 | 4096 | 0 | none | measured |
| `q2_k` | 128, 64, 2, 2 | 0 | 4096 | 0 | 4096 | inherited, k-quant class |
| `q3_k` | 128, 64, 2, 2 | 0 | 4096 | 0 | none | inherited, k-quant class |
| `q4_0`, `q4_1`, `q5_0`, `q5_1` | 128, 64, 2, 2 | 0 | 4096 | 0 | 10240 | inherited, flat class |
| `iq4_nl` | 128, 64, 2, 2 | 0 | 4096 | 0 | 12288 | inherited, flat class |
| `pq2_0` | 128, 64, 2, 2 | 0 | 4096 | 0 | 4096 | inherited, grid class |
| `q2_0` | 128, 64, 2, 2 | 0 | 4096 | 0 | 4096 | inherited, grid class; the K ceiling is that of `pq2_0`, not measured |
| `stq1_0` | 128, 64, 2, 2 | 0 | 4096 | 0 | 12288 | inherited, grid class |
| `iq2_s`, `iq2_xs`, `iq3_s`, `iq3_xxs` | 128, 64, 2, 2 | 0 | 4096 | 0 | none | inherited, grid class |
| `iq1_s`, `iq1_m`, `iq2_xxs`, `iq4_xs`, `ptq1_0`, `nvfp4` | 128, 64, 2, 2 | 0 | 4096 | 0 | none | inherited, grid class |

`stq1_0` (structured-sparse ternary, llama.cpp PR #22836) ships the full ALU and NAX kernel set. Its
pre-NAX floors are measured (M3 Max: plain split-K entry M 5), and so are its NAX small-M routes in
`kq_nax_small_m`. The rest of its NAX policy is inherited from `iq1_s` and needs M5-silicon
calibration: `bm128_min_m` (`benchmarks/bench_qmm_bm128_ab.py`), `kq_splitk_min_m_nax_alu`, and db64
candidacy (no `_db` instantiation yet). The PrismML codecs `pq2_0` and `ptq1_0` (128-wide blocks)
ship the same kernel set. `q2_0` (64-wide blocks, the `pq2_0` code layout) runs the `pq2_0`
kernels at its block width and takes the `pq2_0` route entries, which are not yet measured for
it. On the large-M and image ops `q2_0` inherits what `pq2_0` has and no kernel was built for it:
the grid class large-M row with the `pq2_0` K ceiling, the BM=32, BM=64 and BM=128 tiles on the K
tile of 64 (neither codec has a `_bk128` kernel, and a `KQ_NAX_Q2_0_BK128` variable is not read),
the split-K kernels with their float partial twins, and `verify_mma` at one row tile per simdgroup
with the float kernels by default and the half kernels under `KQ_VERIFY_MMA_HALF`. `KQ_NAX_TILE`,
`KQ_QMM_TILE` and the tile levers under `KQ_TUNE_LIVE` apply to it as to every NAX codec. Left out:
`quantized_matmul_glu` and `quantized_matmul_lora` have no `q2_0` or `pq2_0` kernel and build the
product from `quantized_matmul` and stock ops, and like every codec `q2_0` has no kernel on float
activations (float32 runs the bfloat16 kernels). Their NAX split-K entry is M 9, where the verify kernels hand off.
The `bm128_min_m` and db64 floors are still inherited from `iq1_s`. The `ptq1_0` M=1 kernel gives each
28-byte block four lanes. Each lane reads its six trit bytes in two loads plus the word that holds
the high-trit bytes and the scale, and decodes the trits with the exact-float base-3 coefficient
collapse, in which floors of the scaled byte value multiply precomputed activation coefficients
instead of extracting each trit. A simdgroup computes four output rows, so the fast kernel covers
eight rows per threadgroup. Measured on M5 Max at the Ternary Bonsai 2 27B projection shapes with
the weights streamed from DRAM, it reads 360-440 GB/s, 1.2-1.3x the previous
two-rows-per-simdgroup layout. The `pq2_0` M=1 kernel
masks each pair of 2-bit codes into the mantissas of a half2. On float16 activations it runs the dot
as half2 fmas, two weights per instruction, which brings it close to the rate of a kernel that only
loads the bytes. On bfloat16 activations it widens each pair to float and runs the dot as float2
fmas, since a bfloat16 value past 65504 does not fit in half.
Both codecs have a `verify_qmv` sibling that decodes each row's block once and dots it against every
activation row. It serves M 2 below the `verify_mma` entry on GPUs without NAX, where the `mv_ext`
re-decode per row costs `ptq1_0` 3x. On NAX GPUs per-row qmv serves M 2.

`nvfp4` (64-wide blocks of four 16-weight groups) runs its NAX tiles through the same loader as the
Prism codecs and takes the grid class large-M row. It has a `verify_nax` kernel and no `verify_mma`
kernel. Its NAX routes are measured on M5 Max. Per-row qmv keeps only M 2 at N 1025 to 2048, NAX
split-K enters at M 2 for N 1024 and below and at M 6 above, and `verify_nax` enters by shape, as
described with the other `verify_nax` entries. Its `bm128_min_m` floor is 449: BM=128 takes
1.13-1.16x the BM=64 time at M 224, 0.95-0.98x at M 256 and 0.86-0.90x at M 1024. Its large-M row
has no K ceiling, and it has no db64 band.

The register-resident MMA verify kernels (`verify_mma`) serve `pq2_0`, `ptq1_0`, `q2_0`, `q4_0`,
`q4_1` and `q8_0` through M 8. Each simdgroup decodes a block of its weight rows straight into 8x8
simdgroup-matrix fragments, with the k order inside the block permuted so every lane extracts the
code pairs it holds cheapest, and multiplies them against the activations staged once per K chunk,
so the weight bytes are read once for all M rows and the tile's threadgroup staging of the weights
disappears. Split-K over the wire blocks feeds the same partial fold as `qmm_splitk` on the
projection shapes; a head-sized N already fills the GPU with one split and writes the output
directly.

A threadgroup is 8 simdgroups, and each owns two 8-row tiles on `q4_0`, `q4_1` and `q8_0` and one on
`pq2_0`, `q2_0` and `ptq1_0` (`q2_0` runs the `pq2_0` chain at 8 fragment MMAs per 64-weight block
and takes its one-tile layout, which is not measured for it). A `pq2_0` block runs its 16 fragment
MMAs into one accumulator per row tile, so the MMAs form a dependent chain whose latency, longer in
float than in half, bounds the kernel. At two row tiles per simdgroup the float kernel took about
twice the time of the half kernel on the M5 Max. One row tile doubles the threadgroups that hide the
latency, and the float kernel then runs in 0.45-0.52x of its two-tile time over the Qwen3.8-27B
projection and head shapes at M 2 to 8. Splitting each chain over two or four accumulators recovered
most of that too, and staging the activations as half with float accumulators did not. `q4_0` and
`ptq1_0` gained nothing from either change.

The kernels decode the codes exactly in half and keep the activations in their own dtype in
threadgroup memory. Both widen to float in the fragment, so each block sums in float and the block
scale applies in float. Unsplit, or split with float partials (the float16 default, and
`KQ_SPLITK_F32=1`), the result rounds once, as the mat-vec kernels do with the exact weights, at any
activation magnitude. `KQ_VERIFY_MMA_HALF=1` runs the half kernels of `pq2_0`, `q2_0`, `q4_0` and
`q8_0`, which stage the activations as half and sum each block in half. There an activation beyond the half
range (magnitude 65504), such as a bfloat16 outlier, turns to inf, and so does a block sum past it.
On ordinary activations with float partials, the error of the half kernels runs 0.4 to 0.9 output
roundings beyond the rounding of the float64 result on bfloat16, and 4 to 7 on float16, and the
float kernels run within 0.01. Partials in the activation dtype (the bfloat16 default, and
`KQ_SPLITK_F32=0`) add about one output rounding to a split call of either kernel, which puts the
half kernels at 1.1 to 1.4 on bfloat16 and 4 to 7.5 on float16, and the float kernels at about 1 on
both. Against the half kernels over the same shapes and M on the M5 Max, the float kernels take
0.90-1.00x the time for `pq2_0` and 0.94-1.02x for `q4_0` on bfloat16, and 0.99-1.01x for `pq2_0`
and 1.00-1.05x for `q4_0` on float16. `ptq1_0` has no half kernels, since in float it runs in
0.90-1.03x the time of a half build.

A codec with a block min, `q4_1` (w = d * code + m), uses the same fragments with the codes
unscaled. While the activations of a K chunk are staged, each simdgroup also sums the 32
activations of each block per activation row in float, into threadgroup memory beside the chunk.
After the block's float sum is scaled by d, the kernel adds m times that column sum. The codecs
without a min compile without the sums and run unchanged. `q4_1` has no half kernels: its block
sum of codes up to 15 would leave half range at activations of about 136.

The entry M depends on the device. Without NAX, `pq2_0` and `ptq1_0` enter at M 3 (`pq2_0` at M 4 on
float16 activations, where `verify_qmv` holds M 3), `q4_0` at M 2 and `q4_1` at M 4. Measured on M3
Max at the Bonsai gate and down shapes against the routes they displace: `pq2_0` 1.15x at M 3 and
1.7x at M 8, `ptq1_0` 1.2x at M 3 and 2.0x at M 8, `q4_0` 1.2x at M 2 and 1.5x at M 8. Split-K
enters at M 9 for these three codecs. `q4_1` was measured on M3 Max over the Flint projection and
head shapes (N 512 to 150144, K 1024 to 6144) and the Qwen3.8-27B ones, process-paired against its
old route. At M 2 and 3 its `verify_qmv` kernel runs faster on the shapes of K 2048 and below (the
kernel takes 1.04-1.45x its time). From M 4, where the old route is the bm8 split-K tile, the kernel
runs 1.00-1.45x faster, a geometric mean of 1.11-1.14x per M over the Flint shapes and 1.16-1.20x
over the 27B ones on bfloat16 activations, and 1.11-1.13x and 1.07-1.14x on float16. Its speedup
sits below that of `q4_0` (1.19-1.21x and 1.31-1.36x from M 5) because its old route is faster per
weight byte, while the two kernels run at about the same rate per weight byte. On NAX GPUs the
mat-vec kernels hold longer, so above the per-row qmv limit `ptq1_0` enters at M 3, `pq2_0`, `q4_0`
and `q4_1` (the `q4_0` entry, not measured for it) enter at M 5 on the calls that `verify_nax`
declines, and NAX split-K takes M 9 and up. Measured on M5 Max over the eight Ternary Bonsai 2 27B
projection and head shapes with the weights streamed from DRAM
(`benchmarks/bench_verify_routes.py`), against the mat-vec route each displaces, on bfloat16
activations: `pq2_0` 1.1-1.3x at M 5 and 1.6-2.1x at M 8, `ptq1_0` 1.0-1.2x at M 3 and 2.5-3.1x at M
8, `q4_0` 1.0-1.4x at M 5 and 1.1-1.4x at M 8. Float16 activations give the same entries. The `q8_0`
kernel has no default entry on either class of GPU, so it runs only where `KQ_VERIFY_MMA` or
`KQ_QMM_ROUTE=verify_mma` forces it.

On NAX GPUs, the register-fed NAX verify kernels (`verify_nax`) serve `pq2_0`, `q4_0`, `q4_1`,
`q8_0`, `q2_k` to `q6_k`, `iq4_xs`, `iq4_nl` and `nvfp4` from a per-codec entry through M 8, above
the per-row qmv limit where that limit is wider. Each simdgroup owns 32 weight rows and decodes
every block of them straight into the right operand of a 16x32x16 NAX matmul, with the block scale,
and the min where the codec has one, folded into the half weights. The left operand holds the
activation rows padded to 16, read from device memory in the k order of the decoded weights, so
nothing is staged in threadgroup memory and the loop has no barriers. The accumulator is float32.

A simdgroup's device loads and NAX ops do not overlap, so the kernel hides its loads through the
simdgroups resident beside it, and the host sizes the split count for that. The kernels differ in
the weights one K step decodes, the bytes each lane quad loads, when the activations load and the
split rule, as the table lists.

| kernel | K step | lane quad loads | activations | split rule | weight base |
|---|---|---|---|---|---|
| `pq2_0` | 128 | a piece of one block from each of 8 rows | a K step ahead | about 600 simdgroups | 4-byte |
| `q4_0`, two-block | 64 | a piece of one block from each of 8 rows | a K step ahead | about 600 simdgroups, K walk 640 | 4-byte |
| `q4_0`, eight-block | 256 | two whole blocks of one row, as nine 4-byte words | per NAX op | waves, partial charge 640 | 4-byte |
| `q4_1`, two-block | 64 | a piece of one block from each of 8 rows | a K step ahead | about 600 simdgroups, K walk 640 | 4-byte |
| `q4_1`, eight-block | 256 | two whole blocks of one row, as ten 4-byte words | per NAX op | waves, partial charge 640 | 4-byte |
| `q8_0` | 128 | one whole block of one row | per NAX op | about 600 simdgroups | 4-byte |
| `q4_k` | 256 | the 16-byte header and a 32-byte quarter of one row's codes | per NAX op | waves, partial charge 640 | 16-byte |
| `q5_k` | 128 | the header, 16 bytes of codes and their 16 bytes of high bits | per NAX op | waves, partial charge 2560 | 16-byte |
| `q6_k` | 128 | 16 bytes of low codes and their 16 bytes of high bits, as 2-byte words | per NAX op | about 1280 simdgroups, K walk 2560 | 2-byte |
| `q3_k` | 128 | 8 bytes of low codes, their 8 bytes of high bits and the 12 scale bytes, as 2-byte words | per NAX op | waves, partial charge 2560 | 2-byte |
| `q2_k` | 128 | 8 bytes of codes and 8 bytes of scales and mins | per NAX op | waves, partial charge 2560 | 4-byte |
| `iq4_xs` | 256 | the 8-byte header and a 32-byte quarter of one row's codes, as 8-byte words | per NAX op | waves, partial charge 640 | 8-byte |
| `iq4_nl` | 256 | two whole blocks of one row, as nine 4-byte words | per NAX op | waves, partial charge 640 | 4-byte |
| `nvfp4` | 128 | the scale byte and 8 code bytes of one 16-weight group of one row | a K step ahead | waves, partial charge 1280 | 4-byte |

The `pq2_0` and two-block `q4_0` and `q4_1` kernels decode each block in the permuted k order of
`verify_mma`. The other kernels keep each lane's loads on one row, which measured faster. Spreading
a `q8_0` block across 8 rows ran 1.04-1.24x slower at every Qwen3.8-27B shape except N 1024. `q4_0`
and `q4_1` run the eight-block kernel at a K that is a multiple of 256 and the two-block kernel
elsewhere. The `q4_1` kernels are the `q4_0` kernels with the min folded in as code * d + m, one
half rounding per weight. Its 20-byte blocks keep each row 4-byte aligned, so the eight-block kernel
reads whole words with no shift. The K-quant kernels fold each 16- or 32-weight sub-block's scale,
and its min where the codec has one, into the half weights. Loading a whole K step of activations
ahead measured faster for `pq2_0` and the two-block `q4_0` kernel, and `nvfp4` loads them the same
way, while the kernels with larger decoded words load them one NAX op at a time, since loading ahead
ran `q4_k` 1.6-2x slower. Two consecutive NAX ops read 8 contiguous activations of a row, so where
the activations start on a 16-byte boundary these kernels read them in one 16-byte load per two NAX
ops, and where they start on an 8-byte boundary in one 8-byte load per op. Activations that start
elsewhere are read from a copy on a 16-byte boundary. The fewer loads take less time in a chain of
dependent calls, and both forms give the same bits.

Split counts for `pq2_0`, `q8_0` and the two-block `q4_0` and `q4_1` kernels are the largest divisor
of the K steps at or under their target, at most 16. The two-block kernel's K walk of 640 per
simdgroup measured faster even at the vocab head, where the grid is full without splits. For `q6_k`
the count is the smallest divisor at or above the target, or the largest under it when no divisor up
to 16 reaches it. `q6_k` grids of about 768 simdgroups ran 1.10-1.15x slower than grids of 1280 to
1536, and the `q6_k` head ran 1.06-1.07x slower on one split than on two.

The other kernels take their count from a model of GPU waves of 640 simdgroups. A count costs its
waves times the K steps each simdgroup walks, plus the simdgroups of all its splits divided by the
codec's partial charge. The charge is 640 for `q4_k`, the eight-block `q4_0` and `q4_1` kernels,
`iq4_xs` and `iq4_nl`, 1280 for `nvfp4`, and 2560 for `q2_k`, `q3_k` and `q5_k`. The cheapest power
of two that divides the K steps wins, unless another divisor costs at most 3/4 as much, and the
count has no cap. N 1024, K 5120 therefore takes 20 splits, which ran 1.13-1.23x faster than 8 on
`q3_k` and `q2_k`. The `q5_k` vocab head takes 4 splits and runs 1.14-1.32x faster than on 1. Across
the Qwen3.8-27B projection and head shapes and N 1024 to 14336 at K 2048 to 14336, timed on M5 Max
with bfloat16 activations, the count ran within 1.03x of the best measured count on 164 of 200 cells
and 1.13x slower at worst. The sweep ran at M 3 for `q4_0`, `q4_k` and `q5_k`, M 4 for `q2_k` and
`q3_k`, and M 8. On `nvfp4`, at M 3, 5 and 8 over 14 shapes of N 512 to 248320 and K 2048 to 17408,
the count runs at 1.01x the best measured count on average and 1.07x at worst. A target of about 600
simdgroups ran its vocab head on one split, 1.14-1.17x slower than on 5 to 10.

A K that is not a whole number of K steps declines the route, and the K-quant kernels also need K in
whole 256-weight superblocks. A two-block `q4_0` or `q4_1` K with an odd count of blocks therefore
runs on the mat-vec kernels at M 3 and 4 and on `verify_mma` from M 5. An `iq4_nl` K that is not a
multiple of 256 declines as well. A `pq2_0`, `q4_0`, `q8_0` or `iq4_nl` weight view that starts 2
bytes past a 4-byte boundary declines too, and the other routes serve it. For the K-quant codecs,
`q4_1` and `iq4_xs` the table's weight base is the alignment every GPU kernel of the codec needs, so
a start off it raises, as [the wire-byte contract](integration.md#the-wire-byte-contract) describes.
`ptq1_0` has no NAX verify kernel, because its base-3 decode does not hide under the NAX ops and
measured slower than `verify_mma`.

The entry is the lowest M from which the kernel runs within 1.03x of the route it displaces at every
larger M, on every Qwen3.8-27B projection and head shape (the Ternary Bonsai 2 27B shapes for
`pq2_0`). The table gives the gain measured on M5 Max on bfloat16 activations, with the weights
streamed from DRAM (`benchmarks/bench_verify_routes.py`), and the matmul time of one forward at M 8
with every weight in the codec. Float16 activations give the same entries.

| codec | entry | displaced route | gain at the entry | gain at M 8 | one forward at M 8 |
|---|---|---|---|---|---|
| `pq2_0` | M 3 | `verify_qmv` at M 3 and 4, `verify_mma` from M 5 | 1.09-1.50x | 1.32-1.70x | 38.0 to 24.6 ms |
| `q4_0` | M 3 | `mv_ext` at M 3 and 4, `verify_mma` from M 5 | 1.03-1.30x | 1.09-1.36x | 43.4 to 33.6 ms |
| `q4_1` | M 3, the `q4_0` entry | `mv_ext` at M 3 and 4, `verify_mma` from M 5 | not measured | not measured | not measured |
| `q8_0` | M 6 | per-row qmv or `mv_ext` at M 6, NAX split-K at M 7 and 8 | 1.02-1.35x | 1.10-1.25x | 65 to 56 ms |
| `q4_k` | M 3 | `mv_ext` at M 3 and 4, NAX split-K from M 5 | 1.03-1.32x | 1.18-1.39x | 45.6 to 34.2 ms |
| `q5_k` | M 3 | `mv_ext` at M 3 and 4, NAX split-K from M 5 | 1.10-1.17x | 1.30-1.51x | 61.6 to 44.7 ms |
| `q6_k` | M 5, M 3 at N >= 100000 | the fastest other route per call | 1.09-1.18x | 1.19-1.74x | 71.7 to 47.1 ms |
| `q3_k` | M 4, M 3 at N >= 100000 | the fastest other route per call | 1.08-1.24x | 1.18-1.44x | 55.7 to 41.3 ms |
| `q2_k` | M 4 above N 4096, M 5 up to it, M 3 at N >= 100000 | the fastest other route per call | 1.06-1.23x | 1.07-1.27x | 44.4 to 37.6 ms |

Per-row qmv keeps N 1024 at M 3 to 6 for `q4_0` and `q4_1`, M 3 to 5 for `q4_k` and M 3 for `q5_k`.
The `q8_0` qmv limit for N <= 1024 stops at M 5, since the kernel runs 1.2x faster than qmv there at
M 6. `q8_0` enters at M 6 because N 6144, K 5120 runs 1.05x slower at M 5, although the forward is
faster there. `q6_k` runs up to 1.04x slower than `mv_ext` at M 4 on bfloat16 activations and up to
1.08x on float16, so it enters at M 5, except on a vocab head, where `mv_ext` slows with N. There
the kernel runs 1.09-1.16x faster from M 3, and the head boundary is the N >= 100000 that the `q6_k`
tile route uses.

`q3_k` and `q2_k` enter at M 4, because at M 3 they ran up to 1.04x and 1.12x slower than the
fastest other route at N 6144, K 5120 on bfloat16 activations, and `q2_k` up to 1.17x on float16.
On a vocab head they run 1.04-1.21x faster from M 3. Their entries were also checked at N 1024 to
4096 with K 4096 to 8960. There `q2_k` ran up to 1.14x slower than `mv_ext` at M 4, so it enters at
M 5 up to N 4096. `q3_k` keeps M 4 at those widths, although it runs 1.04-1.06x slower than
`mv_ext` at N 2048, K 4096 on float16 activations.

Against the two-block kernel at M 3 to 8 on bfloat16 activations, the eight-block `q4_0` kernel
runs 1.03-1.15x faster on the Qwen3.8-27B shapes with N 10240 and up or K above 5120. It runs up to
1.04x slower at N 6144 and 12288 and 2 us slower at N 1024, and no split count recovers those
shapes. On float16 activations the gains shrink to at most 1.10x, and the losses at N 6144 and
12288 grow to 1.05-1.10x. Over one Qwen3.8-27B forward in `q4_0` at M 8, the eight-block kernel
takes 7% off the matmul time on bfloat16 activations and 3% on float16.

The `iq4_xs` and `iq4_nl` kernels look both codes of a pair up in one 256-entry table of half
pairs and run at the rate of the `q4_k` kernel, flat in M. Their mat-vec routes run faster than
those of `q4_k` at M 2 to 4, so the entry moves with the shape:

| shape | entry, bfloat16 | entry, float16 | time against the default route at the entry |
|---|---|---|---|
| N >= 100000 | M 3 | M 3 | 0.82-0.97x |
| K >= 6144 | M 3 | M 4 | 0.79-1.01x |
| K 4096 to 5120 | M 4 | M 5 | 0.79-1.00x |
| a narrower K, N >= 10240 | M 4 | M 5 | 0.91-1.03x |
| a narrower K, N 2048 to 10239 | M 5 | M 5 | 0.88-1.00x, and 1.04-1.08x at N 4096, K 2560 |
| N under 2048 | none | none | the kernel ran up to 1.19x slower through M 5 |

The bfloat16 column was fitted on M5 Max over 32 shapes of N 1024 to 248320 and K 2048 to 17408.
On float16 activations the mat-vec routes run faster and the kernel the same, so at the bfloat16
entries the kernel ran up to 1.10x slower at K 4096 and 1.21x slower at N 12288, K 2560, and each
entry under M 5 sits one row higher. At M 8 the kernel runs 1.05-1.39x faster than NAX split-K
on the shapes of N 2048 and wider.

The `nvfp4` entry also moves with the shape, fitted on both activation types over 24 shapes of N 256
to 248320 and K 2048 to 17408, 8 of them held out:

| shape | entry, bfloat16 | entry, float16 | time against the default route at the entry |
|---|---|---|---|
| N >= 100000 | M 3 | M 3 | 0.89x on bfloat16, 1.01x on float16 |
| N 512 and under | none | none | NAX split-K ran 1.07-1.42x faster through M 8 |
| K >= 6144, or N 513 to 1024 from K 4096 | M 2 | M 2 | 0.83-1.04x |
| N 1025 to 2048 from K 4096 | M 3 | M 3 | 0.85x on bfloat16, 0.90x on float16 |
| N 513 to 2048 under K 4096 | M 5 | M 5 | 0.99-1.02x |
| N over 2048 under K 6144 | M 4 | M 5 | 0.84-1.05x |

At M 8 the kernel runs 1.04-1.29x faster than NAX split-K from N 2048, and 1.5x on a vocab head.

## Quantized matmuls that share an input

- `quantized_matmul_multi` - `[x @ W^T for W in weights]` for packed weights of one codec and one
  inner dim that read the same x, such as the q, k and v projections or the gate and up projections
  of a transformer block. It returns one output per weight, as `quantized_matmul` would.

  On a NAX GPU (M5), at 2 to 8 activation rows, where `quantized_matmul` would take its verify_nax
  route on the widest weight, one dispatch computes the products of up to four weights. The row
  tiles of the weights lie end to end in one grid, each threadgroup owns 128 rows of one weight,
  and the split-K partials of all of them share one temporary and one fold dispatch. More than four
  weights run one dispatch per four. The split count is chosen for the joint grid. The call saves
  the dispatch, the wait and the tail of every weight but the first, which counts most for narrow
  weights next to a wide one.

  The narrower weights take the route of the widest. A weight that `quantized_matmul` alone would
  run on per-row qmv or on another kernel runs on verify_nax here, so its output can differ from
  `quantized_matmul` in the last bits, inside the same bound against a float64 reference. This
  holds for every codec with a verify_nax kernel, including those whose verify_nax entry starts at
  a width that the narrow weight does not reach. Where every weight would take verify_nax at one
  split count, the outputs equal `quantized_matmul` bit for bit.

  Every other case runs `quantized_matmul` per weight: one row or more than 8, a single weight,
  codecs without a verify_nax kernel, an inner dim that kernel does not step over, a weight whose
  start it cannot read, CPU streams and GPUs without NAX. x is float16 or bfloat16, float32 x runs
  in bfloat16 as in `quantized_matmul`, and leading batch dims fold into rows.

  The route levers of `quantized_matmul` decide the route on the widest weight the same way:
  `KQ_NAX_QMV`, `KQ_VERIFY_NAX`, `KQ_VERIFY_MMA`, `KQ_QMM_SPLITK_NAX`, `KQ_QMM_SPLITK`,
  `KQ_VERIFY_EXT` and `KQ_DISABLE_NAX`. A set `KQ_QMM_ROUTE` or `KQ_FORCE_QMM_MIN_M` runs
  `quantized_matmul` per weight, so a forced route times the route it names.

  Lever, read when the op builds its graph, once per process unless `KQ_TUNE_LIVE=1`:

  - `KQ_MATMUL_MULTI` - `0` builds `quantized_matmul` per weight on every call. Unset (on) by
    default.

  `mx.grad` gives the gradient with respect to x through `quantized_matmul` per weight. The weights
  are frozen, as in `quantized_matmul`.

## GLU product of two quantized matmuls

- `quantized_matmul_glu` - `y = act(x @ Wg^T) * (x @ Wu^T)` for two packed weights of one codec and
  one shape, the gated feed-forward of a diffusion transformer block. `act` is `silu` or
  `gelu_tanh`. The weights are two arrays, or the two row bands of one fused gate and up weight,
  which the kernel reads in place.

  On a NAX GPU (M5), from 1024 activation rows, one dispatch computes the product for `q4_k`,
  `q5_k` and `q6_k` weights whose inner dim is a multiple of 128, and for `q8_0` weights on
  bfloat16 activations whose inner dim is a multiple of 64. Each threadgroup owns a 128 by 64 tile
  of y. It walks K over the gate weight, applies the activation in f32 and stores the tile, then
  walks K over the up weight and multiplies the stored tile into the sum before the final store.
  y is the only buffer the op writes, so a call peaks at one output where the unfused ops peak at
  two with `silu` and at three with `gelu_tanh`.

  Measured on an M5 Max against the two matmuls and the stock activation and product, as the median
  of rotated rounds in one process:

  | K, N, activations | Rows | `q4_k` | `q5_k` | `q6_k` | `q8_0` |
  |---|---|---|---|---|---|
  | 3840, 10240, bfloat16 | 1056 | 0.91 | 0.99 | 0.96 | 0.99 |
  | 3840, 10240, bfloat16 | 2080 | 0.89 | 0.88 | 0.87 | 0.93 |
  | 3840, 10240, bfloat16 | 4128 | 0.96 | 0.94 | 0.96 | 0.96 |
  | 3840, 10240, bfloat16 | 4608 | 0.97 | 0.96 | 0.97 | 0.96 |
  | 3840, 10240, bfloat16 | 8208 | 0.96 | 0.96 | 0.97 | 0.97 |
  | 3840, 10240, bfloat16 | 16416 | 0.97 | 0.97 | 0.99 | 0.98 |
  | 3840, 10240, float16 | 1056 | 0.95 | 1.00 | 0.98 | - |
  | 3840, 10240, float16 | 4128 | 0.94 | 0.96 | 0.97 | - |
  | 3840, 10240, float16 | 16416 | 1.00 | 0.98 | 0.99 | - |
  | 4096, 12288, bfloat16 | 4128 | 0.94 | 0.95 | 0.95 | 0.97 |
  | 4096, 12288, bfloat16 | 16416 | 0.97 | 0.97 | 0.97 | 0.99 |

  From 4128 rows the gain is the activation and product pass that the kernel removes, 1 to 6
  percent. `q4_k` on float16 at 16416 rows is a tie. `q8_0` on float16 builds the unfused ops, so
  it has no cell. `benchmarks/bench_glu.py` repeats the table for one codec.

  x is float16 or bfloat16, and float32 x runs in bfloat16 as in `quantized_matmul`. Leading batch
  dims fold into rows. The activation and the product run in f32, with one rounding of the
  activation and one of the product. A finite float16 result past the half range is stored as the
  largest finite value with its sign, the activation before the product and the product at the
  store, where the unfused ops return inf. A NaN or an infinity in x gives NaN and inf where the
  unfused ops give them.

  Every other case builds the same product from `quantized_matmul` and stock ops: other codecs,
  `q8_0` on float16, fewer rows, other inner dims, CPU streams and GPUs without NAX.

  Levers, each read once per process unless `KQ_TUNE_LIVE=1`:

  - `KQ_GLU` - `0` builds the unfused ops on every call. Unset (on) by default.
  - `KQ_GLU_MIN_ROWS` - the row count from which the fused kernel runs, an integer of at least 1.
    A value below 1 keeps the default, with a warning on stderr. Default 1024.

  A build with the CMake option `KQ_PROBE_KERNELS=ON`, which `KQ_PROBE_KERNELS=1` in the
  environment of the build selects, also carries the tiles that the shipped ones were measured
  against. It reads `KQ_GLU_BM`, `KQ_GLU_BK`, `KQ_GLU_WALKS` and `KQ_GLU_DUAL128` to select one,
  and a tile it does not carry for the codec raises a `ValueError` that names the levers and the
  kernel. A release build reads none of the four.

  `mx.grad` gives the gradient with respect to x at every row count: the gradient of the fused
  kernel runs the unfused ops. The weights are frozen, as in `quantized_matmul`. For float32 x the
  fused kernel sums the gate term and the up term of the gradient in bfloat16, where the unfused
  ops sum them in float32, so the two gradients differ by one bfloat16 rounding, about 2e-3
  relative. `mx.jvp` and `mx.vmap` raise at every row count, since `quantized_matmul` has neither.

## Quantized matmul with a low-rank term

- `quantized_matmul_lora` - `y = x @ W^T + (strength * (x @ A)) @ B` for a packed weight and a
  low-rank adapter, `lora_a` [K, R] and `lora_b` [R, N], at the row counts of a diffusion
  transformer step. `strength` is a 0-D array or [1] for one scale, or [R] for one per rank, and
  any other shape raises a `ValueError`. It is an input array, so a new value needs no reload and
  no new trace of a compiled function. Several adapters are one call: join their A columns and
  their B rows, and give one strength per rank.

  `bands` holds rows of (col_start, col_end, z_start, rank), in column order and disjoint. The
  columns of a band take that block of ranks and that block of `lora_b`, and a column outside
  every band takes no low-rank term. A fused q, k and v weight with one adapter per projection is
  three bands, so no projection multiplies the ranks of another. `bands=None`, the default, is no
  table: every column takes every rank. `bands=[]` is a table with no band: no column takes a
  rank, and y is `quantized_matmul(x, w)`, bit for bit.

  On a NAX GPU (M5), for `q4_k`, `q5_k`, `q6_k` and `q8_0` weights on bfloat16 activations with K
  up to 4096 and N of at least K, from 4096 activation rows where no column takes more than 128
  ranks, and from 1024 rows at every rank where N is at least 8192, stock ops compute
  `z = strength * (x @ A)` and one dispatch computes the product and adds `z @ B`. Each
  threadgroup owns a 128 by 64 tile of y. It walks the rank of its column band 32 ranks a step,
  with z and B read from their buffers, then walks K over the weight, and stores the tile. Both
  walks add into one f32 sum, which is rounded once at the store. y is the only buffer the
  dispatch writes, where the unfused ops write the product and then read and write it again for
  the sum. The kernel takes a rank of at most 256 without bands, and with bands at most 16 of
  them, each starting on a multiple of 64 columns, ending on a multiple of 32 columns or at N, and
  holding at most 256 ranks. The `q8_0` kernel walks K in tiles of 64, so a `q8_0` weight whose K
  is no multiple of 64 runs the unfused ops.

  The bounds on K, N, rows and rank are where the kernel measured faster. It removes the second
  pass over y, a cost in proportion to N. It waits for `x @ A`, a cost in proportion to K and to
  the rank, where the unfused product runs beside `x @ A`. So the kernel gains where N is at least
  K, and its gain is a share of the call that falls as K grows. Where N is near K the gain is
  about one percent, and it holds from 4096 rows and up to a rank of 128. With bands, the rank
  that counts is the rank of the largest band.

  Measured on an M5 Max against `quantized_matmul`, `x @ A` and one `mx.addmm`, one call an eval,
  at 4608 rows on `q4_k`, as the median of four processes of rotated rounds, kernel time over
  unfused time:

  | K, N | Rank 16 | Rank 64 |
  |---|---|---|
  | 2048, 2048 | 0.99 | 0.96 |
  | 3072, 3072 | 0.98 | 0.97 |
  | 3840, 3840 | 0.99 | 0.98 |
  | 3840, 10240 | 0.97 | 0.97 |
  | 3840, 11520, three bands | 0.96 | 0.94 |
  | 4096, 4096 | 0.99 | 0.99 |
  | 4096, 24576, two bands | 0.96 | 0.96 |

  Outside the bounds, with the kernel run under `KQ_QMM_LORA=2`:

  | K, N | Rank 16 | Rank 64 |
  |---|---|---|
  | 4096, 1024 | 1.03 | 1.03 |
  | 4096, 2048 | 1.01 | 0.98 |
  | 6144, 1536 | 1.04 | 1.02 |
  | 10240, 3840 | 1.02 | 1.02 |
  | 12288, 4096 | 1.01 | 1.02 |
  | 5120, 5120 | 0.99 | 0.99 |
  | 6144, 6144 | 0.99 | 0.99 |
  | 8192, 8192 | 0.99 | 1.00 |

  The other codecs inside the bounds, at the same rows, rank 16 and rank 64:

  | K, N | `q5_k` | `q6_k` | `q8_0` |
  |---|---|---|---|
  | 3840, 3840 | 0.99, 0.98 | 0.99, 0.98 | 0.98, 0.98 |
  | 3840, 11520, three bands | 0.96, 0.94 | 0.96, 0.94 | 0.96, 0.94 |
  | 4096, 24576, two bands | 0.96, 0.96 | 0.97, 0.96 | 0.96, 0.96 |

  Below 4096 rows on `q4_k`, rank 16 and rank 64. The first three shapes are outside the bounds
  there and ran under `KQ_QMM_LORA=2`:

  | K, N | 1056 rows | 2112 rows |
  |---|---|---|
  | 2048, 4096 | 1.08, 1.00 | 1.02, 0.99 |
  | 3840, 3840 | 1.02, 0.99 | 1.03, 1.02 |
  | 4096, 4096 | 1.02, 1.01 | 1.04, 1.03 |
  | 4096, 8192 | 0.99, 0.98 | 0.98, 0.98 |
  | 3840, 10240 | 0.99, 0.98 | 0.96, 0.96 |
  | 3840, 11520, three bands | 0.97, 0.94 | 0.96, 0.93 |
  | 4096, 24576, two bands | 0.96, 0.94 | 0.93, 0.93 |

  The tables from here on are the median of four processes that run the two forms in alternating
  slots, one call an eval. Ranks 128 and 256 at 4608 rows on `q4_k`. At rank 256 the two shapes
  with N of K are outside the bounds:

  | K, N | Rank 128 | Rank 256 |
  |---|---|---|
  | 3840, 3840 | 0.99 | 1.00 |
  | 4096, 4096 | 0.98 | 1.01 |
  | 3840, 10240 | 0.97 | 0.99 |
  | 3840, 11520, three bands | 0.93 | 0.90 |
  | 4096, 24576, two bands | 0.97 | 0.95 |

  From 4096 rows up on `q4_k`, rank 16 and rank 64:

  | K, N | 4096 rows | 8192 rows | 16384 rows |
  |---|---|---|---|
  | 3840, 3840 | 1.00, 1.00 | 0.98, 0.98 | 0.97, 0.97 |
  | 4096, 4096 | 1.00, 0.99 | 0.97, 0.98 | 0.96, 0.97 |
  | 3840, 11520, three bands | 0.97, 0.94 | 0.96, 0.94 | 0.97, 0.93 |

  The other codecs below 4096 rows, rank 16 and rank 64:

  | K, N | Rows | `q5_k` | `q6_k` | `q8_0` |
  |---|---|---|---|---|
  | 4096, 8192 | 1056 | 0.98, 0.97 | 0.95, 0.94 | 0.98, 0.98 |
  | 4096, 8192 | 2112 | 0.97, 0.96 | 0.94, 0.94 | 0.98, 0.98 |
  | 3840, 10240 | 1056 | 0.97, 0.96 | 0.94, 0.93 | 0.99, 1.00 |
  | 3840, 10240 | 2112 | 0.96, 0.96 | 0.92, 0.92 | 0.97, 0.98 |
  | 3840, 11520, three bands | 1056 | 0.96, 0.92 | 0.93, 0.92 | 0.99, 0.96 |
  | 3840, 11520, three bands | 2112 | 0.94, 0.92 | 0.91, 0.90 | 0.96, 0.94 |
  | 4096, 24576, two bands | 1056 | 0.95, 0.93 | 0.92, 0.91 | 0.95, 0.94 |
  | 4096, 24576, two bands | 2112 | 0.93, 0.92 | 0.90, 0.89 | 0.93, 0.92 |

  K below 2048 on `q4_k`. A call here takes 0.2 to 2.4 ms, and two runs of the unfused ops differ
  by up to 4 percent at the smallest:

  | K, N | Rows | Rank 16 | Rank 64 |
  |---|---|---|---|
  | 1536, 1536 | 4608 | 0.99 | 0.97 |
  | 1024, 1024 | 4608 | 0.99 | 0.95 |
  | 512, 512 | 4608 | 0.97 | 0.97 |
  | 256, 256 | 4608 | 0.99 | 0.99 |
  | 1024, 8192 | 1056 | 1.01 | 0.94 |
  | 1024, 8192 | 4608 | 0.87 | 0.87 |
  | 512, 8192 | 1056 | 0.95 | 0.95 |
  | 512, 8192 | 4608 | 0.79 | 0.80 |
  | 256, 8192 | 1056 | 0.94 | 0.91 |
  | 256, 8192 | 4608 | 0.70 | 0.72 |

  x is float16 or bfloat16, and float32 x runs in bfloat16 as in `quantized_matmul`. Leading batch
  dims fold into rows. `lora_a`, `lora_b` and `strength` take any float dtype and run in the dtype
  of y. K, N and R are each below 2^25 and x holds fewer than 2^31 rows: the kernel computes
  addresses inside a row tile in 32 bits, and a call past either limit raises a `ValueError`.

  Every other case builds the same sum from `quantized_matmul` and stock ops, which round the
  product to the activation dtype before the add: other codecs, float16 activations, fewer rows,
  K above 4096, N below K, N below 8192 under 4096 rows or with more than 128 ranks on a column,
  a rank above 256, more bands, a band that starts off a multiple of 64 columns or ends before N
  off a multiple of 32, a `q8_0` K off a multiple of 64, CPU streams and GPUs without NAX. A rank
  of zero, an x of no rows and a weight of no rows return the product alone.

  Levers, each read once per process unless `KQ_TUNE_LIVE=1`. The op reads them when it is
  called and builds its graph, so a compiled function keeps the kernel choice of its trace:

  - `KQ_QMM_LORA` - `0` builds the unfused ops on every call. `2` runs the kernel at every K and
    N from 1024 rows, inside its other bounds. Unset, `1` or any other value keeps the bounds on
    K, N, rows and rank.
  - `KQ_QMM_LORA_MIN_ROWS` - the lowest row count at which the kernel runs, an integer of at
    least 1. A value below 1 keeps the default, with a warning on stderr. Default 1024.

  A function compiled with `shapeless=True` keeps the path of its trace at every later shape. A
  trace that ran the kernel runs it at every row count, below 1024 rows and outside the bounds on
  rows too, and the result is right there. A trace that built the unfused ops keeps them. A
  shapeless function cannot hold a call with `bands` that builds the unfused ops, since the stock
  slices that cut the bands have no shape rule: that call raises a `ValueError` that names
  `quantized_matmul_lora`. Compile such a function with fixed shapes.

  `mx.grad` gives the gradient with respect to x, `lora_a`, `lora_b` and `strength`, alone or in
  any combination, at every row count: the gradient of the kernel runs the unfused ops. The weight
  is frozen, as in `quantized_matmul`. `mx.jvp` and `mx.vmap` run with respect to `lora_a`,
  `lora_b` and `strength` at every row count, on the unfused ops. With respect to x both raise
  `Not implemented for KQuantMatmul` at every row count, since `quantized_matmul` has neither.

  The unfused ops add each band with `mx.addmm`. On the GPU that stock op misreads a first
  argument of one column that is a view of a wider array, so the op hands it a dense copy of a
  one-column band ([upstream-defects.md](upstream-defects.md)). Wider bands are read in place.

## Quantized matmul with a residual epilogue

- `quantized_matmul_residual` - `y = residual + gate * (x @ W^T)` for a packed weight: the
  residual add after the output projection of a transformer block, with the gate of an adaptive
  norm where the block has one. `residual` has the output shape. `gate` is [N], or x's batch dims,
  then 1, then N (one gate row per batch item), and `gate=None` gives `residual + x @ W^T`. The
  residual and the gate run in the output dtype, the dtype `quantized_matmul` gives for x.

  On a NAX GPU (M5), for `q4_k`, `q5_k`, `q6_k` and `q8_0` weights on bfloat16 activations, from
  1024 activation rows, one dispatch walks K into one f32 sum per output, then reads its tile of the
  residual and its columns of the gate and stores `residual + gate * sum`, rounded once. The product
  is never written, where the unfused ops write it, then read it and the residual and write the
  sum, or two passes on stock elementwise ops. Each threadgroup owns a 128 by 64 tile of y, or 64 by
  64 for `q4_k` and `q8_0` above K 10240, the K ceiling of the large-row tile of
  `quantized_matmul`. A gate per batch item runs the batch in the grid. The `q8_0` kernel walks K
  in tiles of 64, so a `q8_0` weight whose K is no multiple of 64 runs the unfused ops.

  At Qwen-Image 2.1 step shapes (4096 rows a batch item, N 4096, bfloat16) on an M5 Max, the call
  costs what the product alone costs: 0.928 of the unfused ops at K 4096 on `q5_k` and 0.973 at K
  12288 on `q4_k`, at batch 1, against 0.916 and 0.973 for the product alone. A compiled
  `x + gate * y` after the product gives 0.982 and 0.984.

  Every other case builds the same sum from `quantized_matmul` and stock ops, which round the
  product and the gated product to the output dtype before the add: other codecs, float16
  activations, fewer rows, a `q8_0` K off a multiple of 64, CPU streams and GPUs without NAX.
  The op rounds once from 1024 rows counted across the batch, so x of shape [2, 512, K] runs the
  kernel while each [512, K] half on its own runs the unfused ops, and the two can differ in the
  last bit.

  Lever, read once per process unless `KQ_TUNE_LIVE=1`. The op reads it when it is called and
  builds its graph, so a compiled function keeps the kernel choice of its trace:

  - `KQ_QMM_RESID` - `0` builds the unfused ops on every call. `2` runs the kernel at every row
    count. Unset, `1` or any other value runs it from 1024 rows.

  `mx.grad` gives the gradient with respect to x, the residual and the gate at every row count:
  the gradient of the kernel runs the unfused ops. The weight is frozen, as in `quantized_matmul`.
  `mx.jvp` and `mx.vmap` run with respect to the residual and the gate. With respect to x both
  raise `Not implemented for KQuantMatmul`, since `quantized_matmul` has neither.

## 3x3 convolution

- `conv2d_3x3` - `y = conv(x, w) + bias` for a 3x3 kernel, stride 1, padding 1, channels last:
  the convolutions that carry the decoder of an image VAE. `x` is [B, H, W, C], `w` is
  [O, 3, 3, C] in the layout of `nn.Conv2d`, and `bias` is [O] or None. `w` and `bias` run in the
  dtype of x. The result is the cross-correlation `nn.Conv2d` computes, with zeros outside the
  image.

  On a NAX GPU (M5), for bfloat16 with C a multiple of 16, from 1024 output pixels, one dispatch
  computes y as an implicit GEMM: rows are output pixels, columns are the O filters, and K walks the
  9 taps and C channels, 32 channels a step, into one f32 sum per output. Each row reads its tap
  from x in place, zeros where the tap falls off the image, so no patch buffer is written. The bias
  is added in f32 and y is rounded once. Each threadgroup owns a 128 by 128 tile of y, or 128 by 64
  where a 64 column tile pads O less (O 144 and 288, for example).

  On an M5 Max, at the decoder shapes of Z-Image and Qwen-Image 2.1 (H and W 256 to 1024, C 128 to
  1152), the kernel runs 31 to 54 TFLOPS, 0.25 to 0.34 of the time of MLX's convolution without the
  Winograd transform. MLX's Winograd path is faster where O is 256 to 1152, with an error to the
  float64 result 35 times larger on a 256 channel layer. At O 128 the kernel takes about half the
  time of the Winograd path, and at O 144 MLX takes no Winograd transform. A whole Z-Image decode
  at 1024 pixels takes 0.36 of its time on the convolution without Winograd.

  Every other case builds MLX's `conv_general` on the flipped kernel, which takes its implicit GEMM
  and not the Winograd transform, then a stock add of the bias, which rounds the convolution before
  the add: float16 and float32, C off a multiple of 16, fewer pixels, CPU streams and GPUs without
  NAX.

  Levers. The op reads `KQ_CONV_NAX` when it is called and builds its graph, so a compiled function
  keeps the kernel choice of its trace. `KQ_CONV_TILE` is read at dispatch. Each is read once per
  process unless `KQ_TUNE_LIVE=1`:

  - `KQ_CONV_NAX` - `0` builds the composition on every call. `2` runs the kernel at every pixel
    count. Unset, `1` or any other value runs it from 1024 pixels.
  - `KQ_CONV_TILE` - `bm,bn,wm,wn` pins the tile. The metallib carries `128,128,4,4`,
    `128,64,4,2`, `64,128,2,4` and `64,64,2,2`; any other tile raises a `ValueError` that names the
    missing kernel.

  `mx.grad`, `mx.jvp` and `mx.vmap` run the composition, so they give what MLX's convolution gives.

## MoE GLU

Fused gate/up expert matvecs with the GLU epilogue applied in the same dispatch, so each activation
load feeds both projections.

- **`moe_glu_gather_kq`** - fused MoE GLU gather for K-quant expert stacks: `act(gate) * up` in one
  decode-shaped dispatch. Bias-free for most codecs; the fp4 wire codecs (`mxfp4`/`nvfp4`) also take
  per-expert gate/up biases with the `swiglu_clamp` activation (gpt-oss experts).
- **`moe_glu_gather_shexp_kq`** - the same with the block's shared expert folded in as extra slots
  (see [Shared experts wider than a routed expert](#shared-experts-wider-than-a-routed-expert)).
- **`moe_glu_gather`** - the MLX packed-mxfp4 counterpart.
- **`moe_router_topk`** - the router in one dispatch: f32 scoring (`softmax`, or `sqrtsoftplus` for
  DeepSeek-V4), top-k with a min-index tie-break, optional bias-ranked selection, optional
  renormalization, and an optional per-expert scale. Indices are always in range: a row whose
  remaining logits are all NaN takes the lowest expert ids not yet picked, with NaN scores.

The GLU activation is selected per model: plain SwiGLU/GELU, the clamped `silu_limit`
(`silu(min(g, limit)) * clip(u, -limit, limit)`) that DeepSeek-V4's `LimitedSwiGLU` needs, or
`swiglu_clamp` (gpt-oss clamped SwiGLU: biases added, sigmoid slope `alpha`, and a `(u + 1)` linear
term; requires the expert biases and is instantiated for `mxfp4`/`nvfp4` only).

At one activation row the grid codecs (iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs and iq3_s)
decode 32 weights per lane step, one 32-weight group of a block, where the other codecs decode 16.
The block scale, the high-bit bytes and the scale bytes then load once per 32 weights, and the grid
indices and sign bytes load as 16-bit pairs. Every gather takes this step: gate/up, the shared-expert
GLU, the plain down and both score-mixed downs (`gather_qmv_mix_kq` with the shared expert in the
routed codec or in q5_k, q6_k or q8_0, and `gather_qmv_mix_ns_kq`). The dedupe and half-dot forms
keep the 16-weight step. Outputs differ from the 16-weight kernels only in summation order.

| GPU | Codecs on by default | GLU K-lanes per row | Down K-lanes per row |
|---|---|---|---|
| M5 and later | all seven | 16 | 8 |
| Earlier | iq2_s | 8 | 8 |

`kq._ext._moe_c32_on(arch, codec)` and `kq._ext._moe_c32_nx(arch)` give the defaults for an
architecture string. `KQ_MOE_C32=0` restores the 16-weight kernels, and `KQ_MOE_C32=1` runs the
32-weight kernels for every grid codec, on any GPU and at wider rows where the dedupe kernels do not
apply. `KQ_MOE_NX=8` or `16` forces a width, and a forced `KQ_MOE_NX=32` keeps the 16-weight
kernels.

The one-row `quantized_matmul` of a grid codec runs the same row step (`kq_<codec>_qmv_c32`), on
M5 and later GPUs by default, at the GLU width. `KQ_QMV_C32` controls it (see the mat-vec levers).

On M3 Max at E=256, N=K=4096 and top 8, back to back, the gate/up gather took 0.303 ms against
0.340 ms (-11.1%), and the down gather took 0.166 ms against 0.191 ms (-12.5%). In a synced decode
step of eight Muse Spark UD-IQ2_S layers, the MoE blocks took 8.3% less time. With the `q8_0`
split-K kernel, the step went from 1.414 to 1.335 ms per layer (-5.6%).

On M5 Max the GLU gathers are faster at 16 lanes. Back to back at the same shapes, a MoE block
replica with both gathers at 16 K-lanes took 2.3% to 5.2% less time than at 8, in both arm orders. Alone, the
down gathers took up to 9% less time at 8 lanes than at 16.

On M5 Max at E=32 and 64, top 8, N=4096 and 7168, expert widths 1536 to 4096, the 16-weight gathers
took longer than the 32-weight ones, in every family:

| Codec | Longer with the 16-weight step |
|---|---|
| iq1_s | 15 to 23% |
| iq1_m | 12 to 19% |
| iq3_s | 10 to 19% |
| iq2_s | 6 to 13% |
| iq2_xxs | 2 to 8% |
| iq2_xs | 0 to 4% |
| iq3_xxs | 0 to 13% (gate/up even) |

The one-row `quantized_matmul` per-row kernels took 2 to 14% longer than `qmv_c32` at K=2048 and
N=512, 9 to 31% longer at K=4096 and N=14336, and 15 to 35% longer at K=14336 and N=4096.


The grid lookup is the next lever. With the grid read removed (wrong output), the 32-weight
gathers took about 25% less time, and 37% less than the 16-weight gathers. That bounds the gain of
a cheaper lookup. Two other grid sources gave no gain. A 64-bit grid in a device buffer matched the
constant table. A 2-bit code table, in threadgroup memory or in a device buffer, was about 20%
slower.

### K-quant chunk decode

The `q2_k`, `q4_k` and `q5_k` gathers, and the `mv_ext` mat-vec of these codecs at 2 to 12 rows,
decode each 16-weight chunk four codes at a time from 32-bit words. A lane reads the codes of a
`q4_k` chunk in one 16-byte load, those of `q5_k` in one load for the low four bits and one for
the fifth bits, and those of `q2_k` in four 4-byte loads. One shift and one AND per word bring four
codes down to the low bits of their bytes (`q5_k` ORs in the fifth bits the same way), and one
vector convert turns them into floats. The per-byte loop masks each code in place and divides the
scale by the matching power of two, so both forms compute the same product and the outputs are bit
for bit the same.

On M5 Max at 0.87-1.27 GHz, back to back over 24 rounds, the word form took less time than the
per-byte loop on every path, by these medians:

| Codec | Gate/up gather | Score-mixed down | `mv_ext`, 4 rows |
|---|---|---|---|
| q5_k | 13.7% | 17.5% | 15.2% |
| q4_k | 5.2% | 8.1% | 8.0% |
| q2_k | 6.5% | 5.0% | 5.3% |

The gathers ran at Qwen3-30B-A3B shapes (hidden 2048, expert width 768, top 8 of 128) and
`mv_ext` at 4096 x 4096. At DeepSeek-V4.1 shapes, the shared-expert GLU over `iq2_s` experts with a
`q5_k` shared expert took 6.3% less time, and the score-mixed down took 9.9% less over `iq3_xxs`
experts with a `q5_k` shared expert and 2.2% less over `q2_k` experts with a `q8_0` one.

`q3_k` keeps the per-byte loop. Its 110-byte blocks sit 2 bytes apart, so each word takes two
2-byte loads, and in that form the gate/up gather took 4.1% longer, in all 24 rounds.

- `KQ_KQUANT_WORD=0` restores the per-byte loop, with the same output. Read once per process unless
  `KQ_TUNE_LIVE=1`.

### Shared experts wider than a routed expert

Some models merge several shared experts into one, so the shared expert is k routed experts wide
(Flint: routed experts of width 512 and one shared expert of width 1024). The fused shared-expert
gathers run it as k shared slots after the routed slots. k comes from the shapes, and no weight is
copied, so GGUF views and streamed experts work as they are.

| Op | Shared-expert tensor | Shared slot j reads |
|---|---|---|
| `moe_glu_gather_shexp_kq` | gate and up `[k * N, bytes_per_row(K)]` | rows `[j N, (j + 1) N)` |
| `gather_qmv_mix_kq` | down `[hidden, bytes_per_row(k * N)]` | columns `[j N, (j + 1) N)` of each row |

The GLU returns `[T, R + k, N]`. The mix takes `x` `[T, R + k, N]`, `indices` `[T, R]` and
`scores` `[T, R + k]`, or `[T, R]`, which gives each shared slot a weight of 1. At k = 1 the
outputs are the same bits as before.

Each shared slot of the down tensor must start on a whole block of its codec, so N must be a
multiple of that codec's block size: 32 weights for q4_0, q4_1, q5_0, q5_1, q8_0 and iq4_nl, and
256 for q2_k to q6_k. A 640-wide expert with a q6_k shared down does not meet this, and the op
raises an error. Ask first with
`kq.shexp_glu_combo_has_kernel(codec, gate_codec, up_codec, shared_slots=k)` and
`kq.shexp_mix_combo_has_kernel(codec, down_codec, shared_slots=k, slot_width=N)`. Both return False
for a shape the kernels decline. With `shared_slots` above 1, the mix probe needs `slot_width`.

The slot-parallel down runs 64 threads per slot, so 512 threads at Flint's 8 slots (6 routed, 2
shared). When the pipeline takes fewer threads, the call runs on the loop kernel.

- `KQ_MOE_SHEXP_MULTI=0` declines a shared expert wider than one routed expert: the probes return
  False for k above 1, and the ops raise an error. Read per call.
- `KQ_MOE_SP_THREAD_CAP=<n>` lowers the thread limit of the slot-parallel downs (both score-mixed
  gathers) to n, so a launch of more than n threads runs the loop kernel. Read per call. Test lever.

## Attention

Scaled-dot-product variants for shapes stock MLX's fused allowlist excludes, plus the sparse
mechanism below. The ops read K and V in place through their strides, batch-major views included.
The q8 scales and biases are addressed per batch row and head as the caches lay them out, and a
batch-major view of them raises at evaluation. The KVarN ops copy their codes, axes and stage rows
into a packed layout first.

- **`sdpa_vector`** - vector SDPA for large head dims (256, 512) - e.g. DeepSeek MLA - which MLX's
  fused vector path does not cover.
- **`sdpa_decode_gqa`** - decode/verify GQA tuned for long KV caches: the key axis splits into coarse
  chunks streamed through threadgroup-staged K/V tiles shared by the GQA group, so device memory reads
  the KV once per chunk. Optional `starts` (int32 `[B]`) restricts row b to keys `[starts[b], kL)` for
  left-padded batches, skipping fully padded-out chunks. Optional `ends` (int32 `[B]`) gives row b its
  own key end, `[starts[b], ends[b])` with the causal block at `ends[b]`, so `kL` is only the capacity
  and batched rows may differ in length without right-justification. Optional affine q8 K/V operands
  (scales and biases, bits 8, group 64) dequantize on the tile stage. `return_lse=True` adds per-row
  log-sum-exp. On NAX GPUs one query at head dim 512 with up to 8 query heads per KV head runs the
  chunk pass on the matrix units once KV heads x batch x keys reaches 3072. That needs K and V
  strides in multiples of 8 elements and no `ends` or q8 operands. With `starts` it skips the
  padded keys of each row as the simdgroup kernel does. Below 3072 the
  simdgroup kernel's finer splits keep more of the GPU busy. Eight simdgroups share each 256-key
  tile, each scoring 32 keys over the full head dim and then accumulating a 64-column slice of
  `P @ V`. The automatic split count is the largest power of two that keeps KV heads x batch x
  splits at 32 or under, and 1 once KV heads x batch passes 32. It ignores the depth, so it falls
  as the batch grows. The two kernels round differently, so the same rows are not bit-identical
  between this pass, the simdgroup kernel (`ends`, q8) and another batch size.
  `KQ_GQA_NAX=0` forces the simdgroup kernel.
- **`sdpa_decode_gqa_cascade`** - shared-prefix batched decode: every row attends one common prefix
  plus its own private suffix. The prefix is walked once for all rows on the matrix-unit tile, private
  suffixes run per row, one merge pass folds both; 1.6-4.2x over per-row calls at 14k-32k prefixes.
  qL 1-8 (verify width, end-aligned causal); takes `starts` and the q8 operands on either region.
- **`sdpa_decode_gqa_paged`** - sparse page-gather decode: attends only the K/V pages listed per
  (batch, kv-head), so cost tracks the selected keys rather than the cache length. The page unit is
  the staged tile height (32 rows at head dim 64/128, 16 at 256, 8 at 512); `tile_c` overrides it
  where instantiated (4-row pages at head dim 256, matching a 4-token block-sparse selection unit);
  takes `starts`.
- **`sdpa_prefill_block_sparse`** - block-sparse FA prefill over 4-row K/V pages at head dim 256:
  queries fold into 4-wide windows (with the GQA group, one MMA tile), and each window walks only
  its own page list with per-page membership bitmasks, so a prefill chunk pays for the selected
  blocks instead of the full key axis and no `[L, S]` mask is materialized. Causality inside the
  diagonal page is enforced in-kernel.
- **`sdpa_fa_verify`** - speculative-verify attention on the matrix units for a GQA-folded query tile.
  Head dims 64 through 512; `return_lse` as above. On NAX GPUs, at head dims 128 and 256 with up to
  64 folded rows on plain K and V, the key pass runs on the tensor-op units: each 16-row strip of
  the fold is one row of simdgroups, and the head dim splits over 2 simdgroups (4 at head dim 256
  for folds of up to 32 rows), which add their partial scores in threadgroup memory. It writes the
  partials of the simdgroup-matrix kernel to the same merge pass. Its automatic split count aims at
  about 64 threadgroups over the KV heads, between 4 and 64, and halves while a split would hold
  fewer than 64 keys. The two kernels round differently, so their outputs are not bit-identical.
  `KQ_FA_VERIFY_NAX=0` keeps the simdgroup-matrix kernel, and `KQ_FA_VERIFY_NAX_WN` sets the head
  dim split (2 or 4 at 256, 1 or 2 at 128; another value warns once on stderr and keeps the
  default); both are read at each call.
- **`sdpa_fa_indexed`** - index-gathered attention over one shared K/V latent at head dim 512, the
  absorbed-MLA sparse decode step: query j of `q [1, Hq, Q, 512]` attends the rows of `kv [1, 1, N,
  512]` that `idx[j]` (int32, -1 pads) lists, so the selected rows are read once from the latent
  with no gathered copy and no materialized `[Hq, M]` score matrix. Each 32-head strip walks one
  split of the list; on tensor-op GPUs a NAX tile kernel gives each of eight simdgroups a 64-column
  eighth of the head dim (partial `K @ Q^T` per eighth summed through threadgroup memory, `P @ V`
  from the same resident fragments), elsewhere the `sdpa_fa_verify` simdgroup tile runs the list.
  The per-split partials merge as `sdpa_fa_verify`. `KQ_SDPA_IDX_NAX=0` forces the simdgroup kernel.
## KVarN KV cache

Variance-normalized KV-cache quantization: the method of Huawei's KVarN (Muller et al.,
[arXiv:2606.03458](https://arxiv.org/abs/2606.03458)) in the record format of
[beellama.cpp](https://github.com/Anbeeld/beellama.cpp) (MIT; see the README acknowledgements). A
record is one 128-token group of one 128-dim head slice: codes at 2/3/4/5/6/8 bits plus three fp16
axis vectors from a 16-step two-axis normalization of the Hadamard-rotated tile. K tiles quantize
per channel, V tiles per token. `KVARN_RECORD_VERSION` names the layout; consumers pin it against
their cache layout.

- **`kvarn_rotate`** - the composite Walsh-Hadamard rotation (per-slice WHT-128 plus a cross-slice
  butterfly at head dims 256 and 512), run in fp32 with one rounding on output so it matches the
  reference bit for bit at 128. Self-inverse: the same call un-rotates.
- **`kvarn_quantize` / `kvarn_dequant`** - seal a rotated 128-token tile into a record and read it
  back; separate K and V kinds.
- **`sdpa_decode_gqa_kvarn`** - decode attention over sealed records plus fp16 stage rows (the
  attention sink and the live tail), record groups dequantized at tile stage; `n_attend` walks the
  record body only, so the caller merges an fp16 precision tail by log-sum-exp. With `ends` each
  row's region map follows its own end and `tail_rows` is the per-row form of `n_attend`, so a
  ragged batch merges per-row tails the same way. Head dims 128, 256 and 512, q_len 1 to 4.
- **`sdpa_fa_verify_kvarn`** - the same matrix-unit verify pass over a KVarN cache: sealed records
  dequantize at tile stage through the loaders `sdpa_decode_gqa_kvarn` uses, and its key pass runs
  on the simdgroup-matrix kernel at every head dim. The result matches `sdpa_fa_verify` over the
  materialized cache bit for bit where that call runs the same kernel (`KQ_FA_VERIFY_NAX=0`). Where
  `sdpa_fa_verify` takes its NAX key pass (head dims 128 and 256 on NAX GPUs), the two agree to
  rounding. The verify-width route (q_len 2 to 8):
  the vector kernel's cost climbs steeply past two queries, the matrix tile prices extra rows at
  nearly zero. Head dims 128, 256 and 512; `n_attend` / `full_visibility` / `return_lse` as the
  decode op.

## DeepSeek/GLM sparse attention (DSA)

The DeepSeek-V4-Flash / GLM lightning-indexer attention: a lightweight indexer scores every pooled
(compressed) KV row against the query, a top-k select picks the rows to attend to, and sparse
attention runs the local sliding window plus those gathered rows in one pass. Ported, with
modifications, from omlx's `glm_moe_dsa` custom kernels (see the
[acknowledgement](../README.md#acknowledgements)). All six accept `qL >= 1`, so decode, MTP verify
(`qL = 2`), and prefill share them.

- **`dsa_indexer_scores`** - indexer relevance scores over a prefill query tile (steel GEMM):
  `out[b,0,m,n] = sum_h relu(q[b,h,m] . k[b,0,n]) * w[h,m]`.
- **`dsa_indexer_score_decode`** - the same for decode-width (`qL <= 4`) queries without materializing
  the per-head `[H, P]` scores. 4, 32 or 64 heads; fp32 head weights are read as-is.
- **`dsa_topk_indices`** - per-row top-k arg-select over the 16-bit scores (2-pass radix select). The
  selected index *set* matches a full sort; the order within a row does not.
- **`dsa_sparse_attention`** - the sliding local window plus the indexer-selected gathered rows plus
  per-head attention sinks, in one flash-softmax dispatch (f32 accumulation).
- **`dsa_kv_qat`** / **`dsa_indexer_qat`** - the fused quantization-aware round-trips DeepSeek-V4 does
  on its main-attention KV (per-64-block FP8-E4M3FN) and indexer activations (128-wide Hadamard then
  per-32-block FP4-E2M1), each bit-identical to the equivalent MLX graph. `dsa_kv_qat(...,
  f16_round=False)` drops the trailing fp16 round for the compressor emit path, whose pooled rows are
  quantized but never stored in the f16 KV cache. DeepSeek-V4.1 takes `dsa_kv_qat(x, 0,
  f16_round=False, block=32)` on its window KV and `dsa_indexer_qat(x, hadamard=False)` on its
  indexer activations.

Tuning levers (defaults are right for normal use):

- `KQ_DSA_BK` - key-tile width for `dsa_sparse_attention`, `128` or `256`. Default: `128` for top-k
  lists up to 128 entries, `256` for denser ones.
- `KQ_DSA_SPLIT` - `1`/`0` forces the split-KV decode route on/off. Default: auto, on for the
  small-grid decode/verify shapes where a single threadgroup would leave the GPU idle.

## Normalization fusions

- **`add_rmsnorm`** - fused post-norm residual `(residual + rms_norm(h, weight)) * scale`, all in f32.
- **`rmsnorm2_add`** - two independent RMS norms plus an add in one dispatch.
- **`rmsnorm_multi3`** - three RMS norms of one tensor sharing its mean-square reduction (the QK-norm
  plus a third head-norm shape).
- **`add_rmsnorm_norm`** - `out = (residual + rms_norm(h, weight)) * scale` and `normed =
  rms_norm(out, next_weight)` in one dispatch, bit-identical to the same composition of MLX ops. A
  decoder layer uses it where a residual sum feeds the next norm, such as the pre-feedforward norm
  after attention or the next layer's input norm. Rows wider than 16384 and CPU streams run the
  composition.

## Attention q and k norm with RoPE

- `qk_norm_rope` - the per-head RMSNorm and rotary embedding that a diffusion transformer applies
  to q and k before attention, both tensors in one dispatch. Each head row `x` of q (weight `q_w`)
  and of k (weight `k_w`) becomes `y = x * rsqrt(mean(x^2) + eps) * w`, and then each interleaved
  pair `(y[2i], y[2i + 1])` of sequence row `l` rotates by the angle whose cosine and sine are
  `cos[l, i]` and `sin[l, i]`. The tables hold one angle per row and pair, so any position scheme
  works, such as the caption index and image row and column of a three-axis image RoPE. The angles
  of `mx.fast.rope` follow the sequence index and cannot express these positions.

  The op reads q and k `[B, L, H, D]` in place through their strides, so the q and k slices of one
  fused qkv output need no copy, and lazy transposed views work the same way. The batch shares the
  `[L, D / 2]` tables. It returns `(q_out, k_out)`, each row-contiguous `[B, H, L, D]`, the layout
  attention reads, so no transpose copy follows.

  q and k are float16 or bfloat16 and the outputs keep that dtype. The weights and tables may be any
  float dtype and are used in f32. The norm and the rotation run in f32 with one rounding at the
  write, so nearly every output element is the exact result rounded to the output dtype. The unfused
  composition also rounds the normalized value before the rotation.

  One threadgroup per `(b, l)` row covers every head of q and k, with 16 lanes per head row. Each
  lane loads its cosines, sines and weights once per row and reuses them across heads. Head dims 64,
  96, 128 and 256 have kernels in both dtypes. Views whose head-dim stride is 1 and whose rows start
  on the vector width take vector loads, and any other view takes element loads through its strides.

  CPU streams, and GPU calls at other even head dims, run the same math as stock MLX ops, an f32 norm
  and rotation followed by one cast to the input dtype and a transposed copy. The op is for
  inference. On the GPU the fused kernel has no gradient or vmap, and `mx.grad` or `mx.vmap` through
  it raises.

## Hadamard rotation

- **`hadamard_rotate`** - the signed block Walsh-Hadamard rotation a Hadamard-folded weight expects
  of its input (PrismML's Ternary Bonsai GGUFs store `W H diag(s)` and read `y = W (H (s * x))`):
  each contiguous `block`-wide chunk (256 to 4096) of the last axis becomes `H (signs * x)` with `H`
  normalized and self-inverse, all math in f32 with one rounding to the input dtype. One threadgroup
  of 256 threads per (row, block); butterflies below the simd width shuffle, those up to the group
  width go through threadgroup memory, and the rest stay in registers. `perm=(rep, nk, hd)` reads
  the row in grouped head order first, which is the fold's `gdn_v_grouped` permute for the
  gated-delta output projection at no extra dispatch. `signs=None` is the identity sign mode.
- **`glu_hadamard`** - a gated activation fused with the `hadamard_rotate` rotation, `H (signs *
  (act(gate) * x))` in one dispatch, for a folded projection whose input a gated activation
  produces. `activation="silu"` is the swiglu in front of a down projection and `"sigmoid"` an
  attention output gate. The threadgroup layout and the rounding are those of `hadamard_rotate`.

## Introspection

- **`codecs`** - the list of supported codec names.
- **`metallib_loads`** / **`metallib_dir`** - whether the bundled metallib opened on the device, and
  where it lives.
- `metallib_kernels` - the sorted names of the kernels in the bundled metallib (empty without
  Metal), the names `KQ_NAX_TILE` and `KQ_QMM_TILE` select from.
- **`nax_available`** / **`nax_gather_enabled`** - whether the GPU exposes NAX tensor units, and
  whether the sorted-gather NAX GEMM kernels (`gather_qmm_rhs_nax`, the `gather_qmm_seg` NAX walk)
  are reachable for a codec.
- **`cpu_neon_available`** - whether the arm64 NEON int8 GEMV path is compiled in.
- `gpu_core_count` / `max_threadgroup_memory` - the GPU core count from IOKit (`KQ_GPU_CORES`
  overrides it) and the threadgroup memory limit in bytes, both 0 when unknown.

## Feeder-loop primitives

The zero-copy arena buffers and shared-event stream primitives (`arena_alloc`, `event_signal` /
`event_wait`, `shared_event_*`, `zero_copy_view_count`, `verify_zero_copy_views`, `load_gguf`) support
a producer/consumer decode loop and are a separate subsystem; see
[docs/feeder/DESIGN.md](feeder/DESIGN.md).

- **`route_shed`** - routed-expert slot remap plus residency shed for streamed MoE decode: expert ids
  map to arena slots through a resident-slot table, non-resident experts are shed with their gate
  mass renormalized onto the kept ones, and the misses come back (ids and scores) for between-token
  prestaging. No host sync.
