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
  expert as the last slot - replacing a gather plus a weighted sum plus the shared-expert add. Shaped
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
from a per-codec entry through M 32. Above M 32 the BM=64 tile runs, with a double-buffered `_db`
variant on the M 33-64 band at large N, and a BM=128 tile from M 193 when ceil(M/64) is even. Above
the qmv limit, the verify kernels take the widths through M 8 on the codecs that have them. The
NAX verify kernel serves `pq2_0`, `q4_0`, `q8_0` and `q2_k` to `q6_k` from a per-codec entry,
and the register-resident MMA verify kernel serves `ptq1_0`, as described at the end of this
section. Every limit and floor is a measured per-codec policy (`kq_nax_small_m` and
`kq_smallbm_policy` in `src/kquant_matmul.cpp`).

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

A NAX tile runs two rows of simdgroups. When the whole matmul has no more rows than one of them
covers, 16 on the BM=32 tile and 32 on the BM=64 tile, the second row would multiply only padding.
Both rows then compute the live rows, each walks half of every K step, and the two partial results
are summed in threadgroup memory before the store. Measured on M5 Max over the 22 NAX codecs with
the weights streamed from DRAM, the NAX split-K route runs 1.05-1.3x faster at M 8 to 16 and the
un-split tile 1.1-1.5x faster at M 7 to 16. On the BM=64 tile, `iq2_xs`, `iq2_s` and `iq1_m` run
1.3-1.5x faster at M 13 to 32, and the gathered NAX tile runs about 1.3x faster at 32 rows per
entry (median over nine codecs). The short last row tile of a taller matmul keeps the plain walk,
because the full tiles set the time there and the split measured slower.

The NAX split-K route cuts K into units of max(block, 64) weights and targets 16 slices of whole
units. It takes the largest count at or under the target that divides the units evenly, unless
equal slices with a shorter last one give more than twice as many. This covers inner dimensions
whose unit count has no divisor near 16, such as 11008, 17408 and 18944 on the 256-weight codecs.
`KQ_SPLITK_RAGGED=2` takes the ragged count wherever it is larger.

Measured on M5 Max at M 8 to 24 with the weights streamed from DRAM, the shorter last slice runs
`q4_k` 1.1x faster at 5120x17408, 1.5x at 3584x18944, 2x at 4096x11008 and 4-5x at 896x4864. The
other codecs range from no change (`pq2_0` at 3584x18944) to 5.7x (`iq1_s` at 896x4864). Where the
even count is at least half the ragged one, the ragged count ran from 11% slower to 1.26x faster
depending on codec and shape, with no consistent direction, so the route keeps the even count.

The M=1 mat-vec kernels loop over their two or four output rows with a static trip count and a
clamped row index, so the compiler interleaves the rows' loads; the tail threadgroup recomputes its
last row and drops it at the store. `q2_k` and `q3_k` keep the runtime bound: the static form measured no
faster for them.

Tuning levers (defaults are right for normal use):

- `KQ_NAX_SMALL_BM` - small-M routing. `0` restores the old routing (mat-vec paths below M 13 and
  no BM=32 tile), `2` forces BM=32 for policy-excluded codecs, unset or `1` follows the per-codec
  policy.
- `KQ_NAX_BM128` - BM=128 band. `0` off, `1` forces the floor to M 193 for every codec, `2` drops
  the floor entirely (any even ceil(M/64), probing the M65-128 wash band), unset follows the
  per-codec entry floors (193/449/961 tiers, measured on M5 Max by
  `benchmarks/bench_qmm_bm128_ab.py`; re-run it before trusting them on new silicon).
- `KQ_NAX_DB64` - double-buffered M 33-64 band. `0` off, `1` drops the N floor, unset follows the
  per-codec N floors. Only the five policy-enabled codecs (q6_k, q8_0, q4_1, q5_1, q5_0) carry
  `_db` instantiations, so `1` is bounded by availability; probing another codec needs its
  instantiation restored and a metallib rebuild.
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
- `KQ_NAX_SWIZZLE` - `1` enables the row-tile traversal swizzle (folds row-tiles into grid.x for
  SLC reuse of the weight band). Falsified on M5 Max, where the M>64 band is per-threadgroup-bound
  rather than DRAM-bound; kept as a probe for future silicon. Default off.
- `KQ_MV_EXT_NR` - `2` selects the two-rows-per-thread `mv_ext` variant (q6_k, M 5-12), which
  halves activation cache traffic but measured no faster than the shipped kernels. Kept as a probe
  for future silicon. Default `1` (shipped behavior).
- `KQ_NAX_QMV` - per-row qmv at small M on NAX GPUs. `0` disables the route, a value of `2` or
  more serves M 2 through that value at every N. Unset takes the per-codec limit for the call's N
  in `kq_nax_small_m`, which yields to a forced `KQ_VERIFY_NAX` or `KQ_VERIFY_MMA` on a codec they
  serve, a forced `KQ_QMM_SPLITK_NAX` or `KQ_QMM_SPLITK`, and a set `KQ_VERIFY_EXT`.
  `KQ_DISABLE_NAX=1` turns it off. Read live per call.
- `KQ_VERIFY_EXT` - the mat-vec route on the M 2-12 band. `1` forces `mv_ext` for every codec with
  the kernel, `0` forces `verify_qmv` where the codec has it and per-row qmv elsewhere, and unset
  takes the per-codec default. On NAX GPUs a set value also keeps per-row qmv and NAX split-K off
  the band. Read once per process.
- `KQ_QMM_SPLITK_NAX` - split-K on the NAX BM=32 tile. `0` disables the route, `1` forces it at
  the default target of 16 slices, and a larger value forces it with that target. Unset takes the
  per-codec entry M in `kq_nax_small_m` (one for N <= 1024, one above), measured on M5 Max, which
  yields to a forced `KQ_QMM_SPLITK` and to a set `KQ_VERIFY_EXT` through M 12. Every codec with
  NAX kernels, M <= 32. Read live per call, so both arms can share one process.
- `KQ_QMM_SPLITK` - the same lever for the plain small-M qmm, used when NAX is absent or disabled.
  Entry points come from a per-device table. K-quants, legacy quants and the IQ codecs, M <= 32.
- `KQ_SPLITK_RAGGED` - slice count of the NAX split-K route. `0` keeps equal slices, the largest
  count at or under the target that divides the slice units, and `2` takes the ragged count
  whenever it is larger. Unset takes it only where it more than doubles the equal count. Read live
  per call.
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
- `KQ_VERIFY_NAX_Q4_0_SB` - probe lever: `0` keeps `q4_0` on the two-block `verify_nax` kernel at
  a K that is a multiple of 256, where the eight-block kernel runs by default. Read live per call.
- `KQ_MV_EXT_SB` / `KQ_MV_EXT_NX` / `KQ_MV_EXT_HD` - `mv_ext` activation-traffic experiments:
  shuffle-broadcast (`1`), wide nxpsg (`16`/`32`), half-precision chunk dots (`1`). q6_k M 4-12
  only. `HD` measured +4-5% at M 8; the rest flat to negative on M5 Max. Kept as probes. Default
  off.

`stq1_0` (structured-sparse ternary, llama.cpp PR #22836) ships the full ALU and NAX kernel set. Its
pre-NAX floors are measured (M3 Max: plain split-K entry M 5), and so are its NAX small-M routes in
`kq_nax_small_m`. The rest of its NAX policy is inherited from `iq1_s` and needs M5-silicon
calibration: `bm128_min_m` (`benchmarks/bench_qmm_bm128_ab.py`), `kq_splitk_min_m_nax_alu`, and db64
candidacy (no `_db` instantiation yet). The PrismML codecs `pq2_0` and `ptq1_0` (128-wide blocks)
ship the same kernel set. Their NAX split-K entry is M 9, where the verify kernels hand off. The
`bm128_min_m` and db64 floors are still inherited from `iq1_s`. The `ptq1_0` M=1 kernel gives each
28-byte block four lanes. Each lane reads its six trit bytes in two loads plus the word that holds
the high-trit bytes and the scale, and decodes the trits with the exact-float base-3 coefficient
collapse, in which floors of the scaled byte value multiply precomputed activation coefficients
instead of extracting each trit. A simdgroup computes four output rows, so the fast kernel covers
eight rows per threadgroup. Measured on M5 Max at the Ternary Bonsai 2 27B projection shapes with
the weights streamed from DRAM, it reads 360-440 GB/s, 1.2-1.3x the previous
two-rows-per-simdgroup layout. The `pq2_0` M=1 kernel
masks each pair of 2-bit codes into the mantissas of a half2 and runs the dot as half2 fmas, two
weights per instruction, which brings it close to the rate of a kernel that only loads the bytes.
Both codecs have a `verify_qmv` sibling that decodes each row's block once and dots it against every
activation row. It serves M 2 below the `verify_mma` entry on GPUs without NAX, where the `mv_ext`
re-decode per row costs `ptq1_0` 3x. On NAX GPUs per-row qmv serves M 2.

The register-resident MMA verify kernels (`verify_mma`) serve `pq2_0`, `ptq1_0`, `q4_0` and `q8_0`
through M 8. Each simdgroup decodes a block of its weight rows straight into 8x8 simdgroup-matrix
fragments, with the k order inside the block permuted so every lane extracts the code pairs it holds
cheapest, and multiplies them against the activations staged once per K chunk, so the weight bytes
are read once for all M rows and the tile's threadgroup staging of the weights disappears. Split-K
over the wire blocks feeds the same partial fold as `qmm_splitk` on the projection shapes; a
head-sized N already fills the GPU with one split and writes the output directly.

The kernels stage the activations as half and sum each block in a half accumulator, so an activation
beyond the half range (magnitude 65504) overflows. The `pq2_0` and `ptq1_0` codes are -1, 0 and 1,
so a block's sum overflows only when its 128 activations add up past that range. The `q4_0` and
`q8_0` kernels decode their codes scaled by 1/16 and 1/256 and apply the factor in the float block
scale, so one channel near the half limit cannot overflow a block of 32 products. A float
accumulator removes the limit but runs `pq2_0` at half speed.

The entry M depends on the device. Without NAX, `pq2_0` and `ptq1_0` enter at M 3 (`pq2_0` at M 4
on float16 activations, where `verify_qmv` holds M 3) and `q4_0` at M 2. Measured on M3 Max at the
Bonsai gate and down shapes against the routes they displace: `pq2_0` 1.15x at M 3 and 1.7x at
M 8, `ptq1_0` 1.2x at M 3 and 2.0x at M 8, `q4_0` 1.2x at M 2 and 1.5x at M 8. Split-K enters at
M 9 for these three codecs. On NAX GPUs the mat-vec kernels hold longer, so above the per-row qmv
limit `ptq1_0` enters at M 3, `pq2_0` and `q4_0` enter at M 5 on the calls that `verify_nax`
declines, and NAX split-K takes M 9 and up. Measured on M5 Max over the eight Ternary Bonsai 2 27B
projection and head shapes with the weights streamed from DRAM
(`benchmarks/bench_verify_routes.py`), against the mat-vec route each displaces, on bfloat16
activations: `pq2_0` 1.1-1.3x at M 5 and 1.6-2.1x at M 8, `ptq1_0` 1.0-1.2x at M 3 and 2.5-3.1x
at M 8, `q4_0` 1.0-1.4x at M 5 and 1.1-1.4x at M 8. Float16 activations give the same entries.
The `q8_0` kernel has no default entry on either class of GPU, so it runs only where
`KQ_VERIFY_MMA` or `KQ_QMM_ROUTE=verify_mma` forces it.

On NAX GPUs, the register-fed NAX verify kernels (`verify_nax`) serve `pq2_0`, `q4_0`, `q8_0` and
`q2_k` to `q6_k` from a per-codec entry through M 8, above the per-row qmv limit where that limit is
wider. Each simdgroup owns 32 weight rows and decodes every block of them straight into the
right operand of a 16x32x16 NAX matmul, with the block scale folded into the half weights. The left
operand holds the activation rows padded to 16, read from device memory in the k order of the
decoded weights, so nothing is staged in threadgroup memory and the loop has no barriers. The
accumulator is float32, which removes the half-range limit of `verify_mma`.

A simdgroup's device loads and NAX ops do not overlap, so the kernel hides its loads through the
simdgroups resident beside it, and the host sizes the split count for that. The kernels differ in
the weights one K step decodes, the bytes each lane quad loads, when the activations load and the
split target, as the table lists.

| kernel | K step | lane quad loads | activations | split target | weight base |
|---|---|---|---|---|---|
| `pq2_0` | 128 | a piece of one block from each of 8 rows | a K step ahead | about 600 simdgroups | 4-byte |
| `q4_0`, two-block | 64 | a piece of one block from each of 8 rows | a K step ahead | about 600 simdgroups, K walk 640 | 4-byte |
| `q4_0`, eight-block | 256 | two whole blocks of one row, as nine 4-byte words | per NAX op | up to 640 simdgroups | 4-byte |
| `q8_0` | 128 | one whole block of one row | per NAX op | about 600 simdgroups | 4-byte |
| `q4_k` | 256 | the 16-byte header and a 32-byte quarter of one row's codes | per NAX op | up to 640 simdgroups | 16-byte |
| `q5_k` | 128 | the header, 16 bytes of codes and their 16 bytes of high bits | per NAX op | up to 640 simdgroups | 16-byte |
| `q6_k` | 128 | 16 bytes of low codes and their 16 bytes of high bits, as 2-byte words | per NAX op | about 1280 simdgroups, K walk 2560 | 2-byte |
| `q3_k` | 128 | 8 bytes of low codes, their 8 bytes of high bits and the 12 scale bytes, as 2-byte words | per NAX op | GPU waves of 640 simdgroups | 2-byte |
| `q2_k` | 128 | 8 bytes of codes and 8 bytes of scales and mins | per NAX op | GPU waves of 640 simdgroups | 4-byte |

The `pq2_0` and two-block `q4_0` kernels decode each block in the permuted k order of `verify_mma`.
The other kernels keep each lane's loads on one row, which measured faster. Spreading a `q8_0`
block across 8 rows ran 1.04-1.24x slower at every Qwen3.8-27B shape except N 1024. `q4_0` runs the
eight-block kernel at a K that is a multiple of 256 and the two-block kernel elsewhere. The K-quant
kernels fold each 16- or 32-weight sub-block's scale, and its min where the codec has one, into the
half weights. Loading a whole K step of activations ahead measured faster for `pq2_0` and the
two-block `q4_0` kernel, while the kernels with larger decoded words load them one NAX op at a time,
since loading ahead ran `q4_k` 1.6-2x slower.

Except on `q3_k` and `q2_k`, split counts are the largest divisor of the K steps at or under the
target, at most 16. For `q6_k` the count is the smallest divisor at or above the target, or the
largest under it when no divisor up to 16 reaches it. `q4_k`, `q5_k` and the eight-block `q4_0`
kernel split only while the grid stays at or under 640 simdgroups, since grids just past that ran
about 1.1x slower. `q6_k` grids of about 768 simdgroups ran 1.10-1.15x slower than grids of 1280 to
1536. The two-block `q4_0` kernel's K walk of 640 per simdgroup measured faster even at the vocab
head, where the grid is full without splits, and the `q6_k` head ran 1.06-1.07x slower on one split
than on two.

The `q3_k` and `q2_k` kernels take their count from a model of GPU waves of 640 simdgroups. A count
costs its waves times the K steps each simdgroup walks, plus a charge for its partials. The cheapest
power of two that divides the K steps wins, unless another divisor costs at most 3/4 as much, and
the count has no cap. N 1024, K 5120 therefore takes 20 splits, which ran 1.13-1.23x faster than 8.
Over 15 shapes at M 4 and 8, the count ran within 1.03x of the best measured count on 55 of 60
cells, and 1.10x slower at worst.

A K that is not a whole number of K steps declines the route, and the K-quant kernels also need K in
whole 256-weight superblocks. A two-block `q4_0` K with an odd count of blocks therefore runs on the
mat-vec kernels at M 3 and 4 and on `verify_mma` from M 5. A weight base off the alignment in the
table declines too, as a zero-copy tensor from a GGUF with a smaller alignment or an offset view can
be. `ptq1_0` has no NAX verify kernel, because its base-3 decode does not hide under the NAX ops and
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
| `q8_0` | M 6 | per-row qmv or `mv_ext` at M 6, NAX split-K at M 7 and 8 | 1.02-1.35x | 1.10-1.25x | 65 to 56 ms |
| `q4_k` | M 3 | `mv_ext` at M 3 and 4, NAX split-K from M 5 | 1.03-1.32x | 1.18-1.39x | 45.6 to 34.2 ms |
| `q5_k` | M 3 | `mv_ext` at M 3 and 4, NAX split-K from M 5 | 1.03-1.17x | 1.21-1.50x | 61.7 to 46.1 ms |
| `q6_k` | M 5, M 3 at N >= 100000 | the fastest other route per call | 1.09-1.18x | 1.19-1.74x | 71.7 to 47.1 ms |
| `q3_k` | M 4, M 3 at N >= 100000 | the fastest other route per call | 1.08-1.24x | 1.18-1.44x | 55.7 to 41.3 ms |
| `q2_k` | M 4 above N 4096, M 5 up to it, M 3 at N >= 100000 | the fastest other route per call | 1.06-1.23x | 1.07-1.27x | 44.4 to 37.6 ms |

Per-row qmv keeps N 1024 at M 3 to 6 for `q4_0`, M 3 to 5 for `q4_k` and M 3 for `q5_k`. The
`q8_0` qmv limit for N <= 1024 stops at M 5, since the kernel runs 1.2x faster than qmv there at
M 6. `q8_0` enters at M 6 because N 6144, K 5120 runs 1.05x slower at M 5, although the forward is
faster there. `q6_k` runs up to 1.04x slower than `mv_ext` at M 4 on bfloat16 activations and up to
1.08x on float16, so it enters at M 5, except on a vocab head, where `mv_ext` slows with N. There
the kernel runs 1.09-1.16x faster from M 3, and the head boundary is the N >= 100000 that the
`q6_k` tile route uses.

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

## MoE GLU

Fused gate/up expert matvecs with the GLU epilogue applied in the same dispatch, so each activation
load feeds both projections.

- **`moe_glu_gather_kq`** - fused MoE GLU gather for K-quant expert stacks: `act(gate) * up` in one
  decode-shaped dispatch. Bias-free for most codecs; the fp4 wire codecs (`mxfp4`/`nvfp4`) also take
  per-expert gate/up biases with the `swiglu_clamp` activation (gpt-oss experts).
- **`moe_glu_gather_shexp_kq`** - the same with the block's shared expert folded in as an extra slot.
- **`moe_glu_gather`** - the MLX packed-mxfp4 counterpart.
- **`moe_router_topk`** - the router in one dispatch: f32 scoring (`softmax`, or `sqrtsoftplus` for
  DeepSeek-V4), top-k with a min-index tie-break, optional bias-ranked selection, optional
  renormalization, and an optional per-expert scale. Indices are always in range: a row whose
  remaining logits are all NaN takes the lowest expert ids not yet picked, with NaN scores.

The GLU activation is selected per model: plain SwiGLU/GELU, the clamped `silu_limit`
(`silu(min(g, limit)) * clip(u, -limit, limit)`) that DeepSeek-V4's `LimitedSwiGLU` needs, or
`swiglu_clamp` (gpt-oss clamped SwiGLU: biases added, sigmoid slope `alpha`, and a `(u + 1)` linear
term; requires the expert biases and is instantiated for `mxfp4`/`nvfp4` only).

## Attention

Scaled-dot-product variants for shapes stock MLX's fused allowlist excludes, plus the sparse
mechanism below.

- **`sdpa_vector`** - vector SDPA for large head dims (256, 512) - e.g. DeepSeek MLA - which MLX's
  fused vector path does not cover.
- **`sdpa_decode_gqa`** - decode/verify GQA tuned for long KV caches: the key axis splits into coarse
  chunks streamed through threadgroup-staged K/V tiles shared by the GQA group, so device memory reads
  the KV once per chunk. Optional `starts` (int32 `[B]`) restricts row b to keys `[starts[b], kL)` for
  left-padded batches, skipping fully padded-out chunks. Optional `ends` (int32 `[B]`) gives row b its
  own key end, `[starts[b], ends[b])` with the causal block at `ends[b]`, so `kL` is only the capacity
  and batched rows may differ in length without right-justification. Optional affine q8 K/V operands
  (scales and biases, bits 8, group 64) dequantize on the tile stage. `return_lse=True` adds per-row
  log-sum-exp.
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
  Head dims 64 through 512; `return_lse` as above.
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
  dequantize at tile stage through the loaders `sdpa_decode_gqa_kvarn` uses, so the result matches
  `sdpa_fa_verify` over the materialized cache bit for bit. The verify-width route (q_len 2 to 8):
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
- **`nax_available`** / **`nax_gather_enabled`** - whether the GPU exposes NAX tensor units, and
  whether the sorted-gather NAX GEMM kernels (`gather_qmm_rhs_nax`, the `gather_qmm_seg` NAX walk)
  are reachable for a codec.
- **`cpu_neon_available`** - whether the arm64 NEON int8 GEMV path is compiled in.

## Feeder-loop primitives

The zero-copy arena buffers and shared-event stream primitives (`arena_alloc`, `event_signal` /
`event_wait`, `shared_event_*`, `zero_copy_view_count`, `verify_zero_copy_views`, `load_gguf`) support
a producer/consumer decode loop and are a separate subsystem; see
[docs/feeder/DESIGN.md](feeder/DESIGN.md).

- **`route_shed`** - routed-expert slot remap plus residency shed for streamed MoE decode: expert ids
  map to arena slots through a resident-slot table, non-resident experts are shed with their gate
  mass renormalized onto the kept ones, and the misses come back (ids and scores) for between-token
  prestaging. No host sync.
