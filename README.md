# mlx-kquant

[![ci](https://github.com/asher/mlx-kquant/actions/workflows/ci.yml/badge.svg)](https://github.com/asher/mlx-kquant/actions/workflows/ci.yml)

Bring **K-quant precision to [MLX](https://github.com/ml-explore/mlx)** on Apple Silicon: a C++/Metal
**extension** for a stock `mlx` wheel that adds the K-quant superblock and per-block integer codecs
as native MLX ops, plus a toolchain that quantizes a model into a **K-quant MLX safetensors
checkpoint** and runs, LoRA-trains, and fuses it.

Two layers:

- **Ops** (C++/Metal) - a `kq.*` namespace (`dequantize`, `quantized_matmul`, `gather_qmm`,
  `quantize`) backed by Metal kernels compiled to a `.metallib` at build time (no runtime JIT). All
  ten K-quant/legacy codecs: `q2_k, q3_k, q4_k, q5_k, q6_k` and `q4_0, q4_1, q5_0, q5_1, q8_0`, plus
  nine IQ codecs
  (`iq4_nl, iq4_xs, iq3_s, iq3_xxs, iq2_xxs, iq2_xs, iq2_s, iq1_s, iq1_m`) - all nineteen decode,
  matmul (incl. tensor-core prefill), and encode (IQ encode is CPU-only) - plus the native-fp wire
  codecs `mxfp4, nvfp4` (decode + CPU matmul so far; decode-only, no encoder). On top of these four core
  ops the namespace also carries fused decode/prefill kernels (MoE GLU and router, attention, norm
  fusions) and a set of DeepSeek/GLM sparse-attention kernels - see
  [docs/kernels.md](docs/kernels.md).
- **Tooling** (Python) - `mlx-kquant quantize / run / chat / lora / fuse` (plus `verify`, `inspect`,
  `calibrate-imatrix`) and a `loader` that create and run K-quant checkpoints in **MLX-native
  safetensors** format.

## Why

K-quants have roughly half the divergence vs. affine quants at the same bitrate (as measured by KLD).

| Model | Budget | MLX affine | K-quant | Divergence cut |
|-------|--------|------------|---------|---------------:|
| Qwen3.6-27B | 4-bit | 0.0577 @ 4.69 bpw | Q4_K_M 0.0208 @ 4.88 bpw | 2.8× |
| Qwen3.6-27B | 5-bit | 0.0214 @ 5.68 bpw | Q5_K_M 0.0096 @ 5.73 bpw | 2.2× |
| gemma-4-E2B | 4-bit | 1.2254 @ 5.76 bpw | Q4_K_M 0.6657 @ 5.77 bpw | 1.8× |
| gemma-4-E2B | 5-bit | 0.4979 @ 6.67 bpw | Q5_K_XL 0.2316 @ 6.67 bpw | 2.2× |

mlx-kquant brings them to the MLX ecosystem with tuned Metal kernels. Quantize an MLX safetensors model
to a uniform- or mixed-precision K-quant checkpoint, then load, generate, LoRA-train, and fuse it
on a stock `mlx` wheel.

## Install

**macOS 26.2 (Tahoe) or later on Apple Silicon.** Prebuilt wheels (CPython 3.10-3.13):

```sh
pip install mlx-kquant            # the K-quant ops + precompiled metallib
pip install "mlx-kquant[tools]"   # the CLI (quantize / run / chat / lora / fuse)
```

Either pulls the ABI-matched `mlx==0.31.2` automatically.

**From source** (or to develop) needs the Metal toolchain (`xcrun metal`); the metallib compiles at
install time, no runtime JIT:

```sh
git clone https://github.com/asher/mlx-kquant && cd mlx-kquant
pip install "mlx==0.31.2"         # pinned, ABI-matched stock wheel (pulls the Metal backend)
pip install -e ".[tools]"         # builds _ext + mlx_kquant.metallib; adds mlx-lm for the CLI
```

**Linux (CPU-only)** also builds, with no Metal toolchain. The ops run on their portable `eval_cpu`
paths and no metallib is produced. (The tuned matmul/gather are Apple-Silicon-targeted: arm64 Linux
picks up the NEON int8 GEMV when the CPU has dot-product, but the Accelerate GEMM is Apple-only, and
x86_64 stays on the scalar/threaded path.) The base `mlx` wheel ships no backend on Linux, so
install the CPU one explicitly:

```sh
pip install "mlx[cpu]==0.31.2"   # base frontend + libmlx CPU backend
pip install -e . --no-build-isolation
```

CPU is for portability and CI, not throughput. Running a full model forward on Linux also needs
`MLX_DISABLE_COMPILE=1`, see [Limitations](#limitations).

Smoke-test the toolchain:

```python
import mlx_kquant as kq
kq.codecs()          # -> ['q2_k', 'q3_k', ..., 'q8_0']
kq.metallib_loads()  # -> True  (the bundled metallib opened on the Metal device)
```

> The extension links `libmlx` and its kernels `#include` MLX's steel-GEMM headers, so it is bound to
> an exact MLX ABI **and** header API. The pin is intentionally `==`, never `>=`; moving to a newer
> `mlx` may require updating the bundled headers and recompiling. See [Version pinning](#version-pinning).

## Quickstart

Quantize a checkpoint and run it, load it through mlx-lm, fine-tune it with LoRA, or build directly on
the `kq.*` ops.

### Create and run a checkpoint

The CLI (the `[tools]` extra adds `mlx-lm`) quantizes an HF / `mlx-lm` model into a K-quant **MLX
safetensors** checkpoint and runs it:

```sh
pip install "mlx-kquant[tools]"
mlx-kquant quantize --model Qwen/Qwen3-0.6B --preset q4_k_m --mlx-path qwen3-q4
mlx-kquant run --model qwen3-q4 --prompt "Explain entropy in one sentence."
mlx-kquant chat --model qwen3-q4 --temp 0.7      # interactive REPL (mlx-lm chat)
```

`run` takes the usual sampling knobs (`--temp`, `--top-p`, `--top-k`, `--min-p`, `--seed`,
`--repetition-penalty`, `--presence-penalty`, `--frequency-penalty`) and chat-template controls
(`--system-prompt`, `--no-chat-template`, `--chat-template-config` for template kwargs such as
`'{"enable_thinking": false}'`). The `chat` REPL has a line-editable prompt with persistent
history (`--no-history` or in-chat `/history off|on|clear` to control it) and in-chat sampling
control (`/temp`, `/top-p`, `/top-k`, `/min-p`, `/max-tokens`, and the three penalties;
`/sampling` shows current values); `/load <file>` prefills the next prompt from a text file for
editing; `/clear` resets the conversation and wipes the screen; Tab completes `/commands` and
paths; Ctrl-C cancels the in-flight reply (at an idle
prompt it exits, as does Ctrl-D). `--max-kv-size` bounds the KV cache for long sessions (a rotating
window, set at start).

The result is a standard MLX checkpoint (`config.json` + sharded safetensors, weights as K-quant wire
bytes). Load it in code with the bundled loader:

```python
import mlx.core as mx
from mlx_kquant.loader import load

model, config = load("qwen3-q4")          # KQuant* layers swapped in, on a stock mlx-lm model
mx.eval(model(mx.array([[1, 2, 3]])))
```

`mlx-kquant lora` (train an adapter) and `mlx-kquant fuse` (merge it back) round out the toolchain -
see [LoRA fine-tuning](#lora-fine-tuning). Run `mlx-kquant --help` for every subcommand.

### Using with mlx-lm

In-process, a kquant checkpoint also loads through **stock mlx-lm**: one idempotent call installs the
load shim, and from then on `mlx_lm.load` / `mlx_lm.generate` (and anything built on
`mlx_lm.utils.load_model`, e.g. an eval harness or your own serving loop) open a kquant checkpoint
transparently:

```python
from mlx_kquant.mlx_lm_patch import patch_mlx_lm_load
patch_mlx_lm_load()                 # process-wide, idempotent; call once before mlx_lm.load

from mlx_lm import load, generate
model, tokenizer = load("qwen3-q4")
print(generate(model, tokenizer, "Explain entropy.", max_tokens=64))
```

This is the load-only shim for inference / eval / serving; `patch_mlx_lm_lora()`
([below](#lora-fine-tuning)) adds the train/merge shims on top. The bundled `mlx_kquant.loader.load`
(above) is the standalone path when you don't need the rest of mlx-lm.

### LoRA fine-tuning

A kquant checkpoint is a frozen base you can adapt with LoRA. Attach an adapter for inference, train
one (the matmul/gather ops define a gradient-through-the-base `vjp`, so the adapter is differentiable
while the quantized weights stay frozen), and merge it back with `mlx-kquant fuse` (re-encode to
kquant, or `--dequantize` to float). One call wires it into stock mlx-lm:

```python
from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora
patch_mlx_lm_lora()   # before building LoRA layers / loading adapters; idempotent
```

See **[docs/lora.md](https://github.com/asher/mlx-kquant/blob/main/docs/lora.md)** for attach / train / merge workflows. (DoRA on a kquant base is
not supported - use LoRA.)

### Using K-quant ops directly

Under the toolchain, the four `kq.*` ops operate on raw K-quant wire bytes. K-quant scales live
*inside* the packed bytes, so the `scales` argument is a vestigial placeholder (the API keeps it for
shape symmetry with MLX's affine quant); `kq.quantize` returns one for you.

```python
import mlx.core as mx
import mlx_kquant as kq

N, K = 256, 512                       # q4_k: K must be a multiple of 256
w = mx.random.normal((N, K))

# encode float -> K-quant wire bytes (CPU or Metal); optional imatrix steers the encoder
wq, scales = kq.quantize(w, "q4_k")           # wq: uint8 [N, bytes_per_row]

# dequantize back to float
deq = kq.dequantize(wq, scales, "q4_k")       # float16 [N, K]

# quantized matmul: x @ dequant(w).T   (transpose=True => w is [N, K])
x = mx.random.normal((8, K))
y = kq.quantized_matmul(x, wq, scales, "q4_k", transpose=True)   # [8, N]
```

Mixture-of-experts (gathered) matmul:

```python
E, N, K = 128, 704, 2816
we = mx.random.normal((E, N, K))
weq, sc = kq.quantize(we, "q4_k")             # per-expert wire bytes
x = mx.random.normal((1, 8, K))               # (tokens, top_k, K)
idx = mx.array([[0, 5, 9, 17, 33, 41, 88, 120]], dtype=mx.uint32)
out = kq.gather_qmm(x, weq, sc, "q4_k", rhs_indices=idx, transpose=True)
```

Ready-made modules that store the wire bytes and dispatch the matching `kq.*` op ship in
`mlx_kquant.nn`:

```python
from mlx_kquant.nn import KQuantLinear, KQuantEmbedding, KQuantSwitchLinear

x = mx.random.normal((8, 512))   # a (tokens, in_dims) activation batch
lin = KQuantLinear(in_dims=512, out_dims=256, bias=False, codec="q4_k")
lin.weight = wq                  # the uint8 wire bytes from kq.quantize, above
lin.scales = scales              # [1] placeholder (scales live in the bytes)
y = lin(x)                       # kq.quantized_matmul under the hood
```

`KQuantEmbedding` (with a tied-`as_linear`), the `gather_qmm`-backed `KQuantSwitchLinear` for MoE
experts, and `KQuantMultiLinear` (absorbed-MLA) are exported alongside it. To swap the quantizable
leaves of a whole constructed mlx-lm model in one call, use
`mlx_kquant.nn.install_kquant_modules(model, {"<path>.weight": "q4_k", ...})`.

The `[tools]` layer is itself a worked reference for wiring `kq.*` into the MLX ecosystem: the loader,
encoder, layer modules, and the mlx-lm monkeypatch are all small and self-contained. See
**[docs/integration.md](https://github.com/asher/mlx-kquant/blob/main/docs/integration.md)** if you're building on the ops.

## Performance

The Metal kernels use a single-pass NAX matmul and matrix-contiguity handling for fused MoE expert
weights. Measured on an M5 Max (128 GB):

| Model | Codec | Decode (tok/s) | Prefill pp512 (tok/s) |
|-------|-------|---------------:|----------------------:|
| gemma-4-26B-A4B-it (MoE) | q4_k_xl | ~111 | ~2330 |
| Qwen3.5-9B (dense)       | q5_k_xl | ~83  | ~2396 |

Transposed matmuls with a small row count (the speculative-decode verify regime) automatically route
through a weight-read-amortizing `verify_qmv` kernel; `KQ_DISABLE_VERIFY_QMV=1` forces the plain
per-row `qmv` path (see [Environment variables](#environment-variables)). Sorted MoE prefill batches
(`sorted_indices=True`) route to a per-expert-segment GEMM on all GPUs - a NAX kernel where tensor
units are available, a steel simdgroup-mma kernel with a rows-per-expert-adaptive row tile elsewhere
- instead of decomposing into per-row gathers.

## How it works

- **Own ops.** Four `Primitive` subclasses (`KQuantDequantize`, `KQuantMatmul`, `KQuantGatherQMM`,
  `KQuantQuantize`) and their op functions live entirely in the extension.
- **Precompiled metallib on stock headers.** The `kq_*` kernels are compiled against the stock
  wheel's steel-GEMM headers into `mlx_kquant.metallib` at build time; host dispatch resolves them
  through MLX's exported `Device::get_kernel`. No JIT, no steel host structs.
- **Codec registry** derives `group_size`/`bits` from the codec name, so callers pass only
  `kquant_type`.
- **CPU and GPU execution.** The decode ops (`dequantize` / `quantized_matmul` / `gather_qmm`) run on
  either stream for all nineteen codecs; `quantize` (encode) covers the ten K-quant/legacy codecs on
  either stream and the nine IQ codecs CPU-only (ggml has no GPU IQ quantizer), so the full
  quantize/decode pipeline (and the op tests) runs in CI without a GPU. The per-block `dequantize` is
  a scalar, bit-exact (per-codec, vs the `gguf.quants` reference quantizer) decoder. The CPU **matmul**
  and **gather** are tuned for Apple Silicon: a shared worker pool over output rows, NEON int8
  dot-product GEMV for the small-M (decode) shape, and an Accelerate (AMX/SME) GEMM for the large-M
  (prefill) shape. The NEON path quantizes activations to int8 (lossy, as ggml does), so its matmul
  matches at tolerance, not bit-exactly; `KQ_CPU_NEON=0` forces the scalar path for exact parity.

## Environment variables

All optional; the defaults are right for normal use.

- `KQ_CPU_THREADS` - worker-pool size for the CPU ops (default: hardware concurrency; `1` runs them
  inline). `KQ_CPU_SPIN_US` sets a spin-before-park window for the pool (default `0` = park).
- `KQ_CPU_NEON=0` - disable the arm64 NEON int8 GEMV kernels and run the scalar decode-then-dot
  matmul, which is bit-exact (the NEON path is tolerance-level; see [How it works](#how-it-works)).
- `KQ_DISABLE_VERIFY_QMV=1` - on Metal, force the plain per-row `qmv` path instead of the
  weight-read-amortizing `verify_qmv` kernel. An A/B debugging lever, not a tuning knob.
- `KQ_DISABLE_GATHER_RHS_ALU=1` - on Metal without NAX, force sorted MoE prefill back to the
  per-row gather path instead of the segment-walking `gather_qmm_rhs` GEMM. A/B lever.
- `KQ_GATHER_RHS_BM` - pin the `gather_qmm_rhs` row tile height (`16`/`32`/`64`) instead of the
  rows-per-expert-adaptive choice. Retuning lever for other GPU generations.
- `KQ_SWITCH_GEMM_MIN_ROWS` - minimum routed rows before `KQuantSwitchLinear` takes the sorted
  per-expert GEMM arm on a prefill batch (default `512`; `0` disables it and keeps the plain gather).

The model-specific kernels carry their own tuning levers, documented alongside each kernel in
[docs/kernels.md](https://github.com/asher/mlx-kquant/blob/main/docs/kernels.md).

## Quant recipes

A **preset** is a named mixed-precision recipe. It classifies each tensor by *role* (attention
q/k/v/o, embeddings, `lm_head`, MoE routed vs shared experts, the FFN down-projection) and maps each
role to a codec - spending bits where they move the output most and staying frugal on the bulk
feed-forward weights, to beat a uniform quant at the same byte budget.

```sh
mlx-kquant quantize --model <src> --preset q4_k_m   --mlx-path out   # a mixed recipe
mlx-kquant quantize --model <src> --kquant-type q6_k --mlx-path out  # one codec, every tensor
```

Naming follows the ggml convention: the family (`q4_k`, `q5_k`, ...) sets the baseline codec and the
suffix sets how much extra precision the recipe spends:

- `_s` / `_m` / `_xl` - small / medium / extra: increasing bumps on the sensitive tensors (the value
  and output projections, the down-projection on a subset of layers, the linear-attention
  projections).
- `_moe` - expert-aware: routed experts at the baseline, shared experts a step above.
- bare `q6_k` / `q8` - uniform (every tensor at one codec), equivalent to passing `--kquant-type`.

`mlx-kquant quantize --list-presets` prints the full, authoritative mapping for every preset; it is
generated from the recipe tables, so it never drifts from what the encoder actually does. The recipes
are informed by our analysis of the mixed-precision quants that [Unsloth][unsloth] and
[bartowski][bartowski] publish on Hugging Face, together with llama.cpp's own per-layer
"use more bits" schedule.

[unsloth]: https://huggingface.co/unsloth
[bartowski]: https://huggingface.co/bartowski

## Codec reference

| Codec | Block | Bits | Bytes/block | Notes |
|-------|------:|-----:|------------:|-------|
| q2_k  | 256 | 2 |  84 | K-quant superblock |
| q3_k  | 256 | 3 | 110 | K-quant superblock |
| q4_k  | 256 | 4 | 144 | K-quant superblock |
| q5_k  | 256 | 5 | 176 | K-quant superblock |
| q6_k  | 256 | 6 | 210 | K-quant superblock |
| q4_0  |  32 | 4 |  18 | block scale |
| q4_1  |  32 | 4 |  20 | block scale + min |
| q5_0  |  32 | 5 |  22 | block scale |
| q5_1  |  32 | 5 |  24 | block scale + min |
| q8_0  |  32 | 8 |  34 | block scale |
| iq4_nl  |  32 | 4 |  18 | non-linear LUT |
| iq4_xs  | 256 | 4 | 136 | LUT superblock |
| iq3_s   | 256 | 3 | 110 | grid + signs |
| iq3_xxs | 256 | 3 |  98 | grid + gas words |
| iq2_xxs | 256 | 2 |  66 | grid + scale/sign words |
| iq2_xs  | 256 | 2 |  74 | grid + scales |
| iq2_s   | 256 | 2 |  82 | grid + qh + signs |
| iq1_s   | 256 | 1 |  50 | grid + delta |
| iq1_m   | 256 | 1 |  56 | grid + delta, scattered scale |
| mxfp4   |  32 | 4 |  17 | e8m0 scale, E2M1 values (decode-only) |
| nvfp4   |  64 | 4 |  36 | 4x ue4m3-scaled 16-value groups (decode-only) |

## Version pinning

Pinned to `mlx==0.31.2`. The kernels include MLX's steel headers and the extension links `libmlx`,
binding it to that release's ABI and header API. To move to a newer MLX: update the bundled headers
under `metal/mlx/backend/metal/kernels/` for that wheel, rebuild, and re-run the test suite.

## Tests

```sh
python -m pytest tests/
```

## Requirements

- **macOS 26.2 (Tahoe) or later on Apple Silicon** (M-series). Building from source needs the Metal
  toolchain (`xcrun metal`).
- **Linux** (x86_64 or aarch64) is supported CPU-only. Build against `mlx[cpu]==0.31.2`, no Metal
  toolchain required. See [Install](#install) and [Limitations](#limitations).
- **Python >= 3.10** (the pinned `mlx==0.31.2` ships no cp39 wheel).
- **`mlx==0.31.2`** exactly - the kernels include MLX's steel headers and the extension links
  `libmlx`, so the ABI is version-locked (see [Version pinning](#version-pinning)).

## Limitations

- **GPU path is Apple-Silicon Metal only.** No ROCm or CUDA support. Every op also has a CPU path
  (`stream=mx.cpu`) — decode for all nineteen codecs, encode for all nineteen (IQ encode is CPU-only) — so the extension
  still builds and runs without Metal (see
  [How it works](#how-it-works) and [Install](#install)).
- **Linux model forwards need `MLX_DISABLE_COMPILE=1`.** Stock MLX's CPU compile JIT generates C++
  that redeclares GCC's built-in `_Float32`/`_Float64`/`_Float128` types, which `g++` rejects, so any
  model forward through MLX's compile path fails on Linux+GCC. Disabling the JIT runs those graphs
  eagerly with identical numerics. This is an upstream MLX-on-Linux limitation independent of
  mlx-kquant - the `kq.*` ops have their own `eval_cpu` and never touch the JIT.
- **LoRA, not DoRA.** LoRA adapters train, attach, and fuse on a kquant base (see
  [docs/lora.md](https://github.com/asher/mlx-kquant/blob/main/docs/lora.md)), DoRA is not yet supported. `fuse` re-encodes to kquant or, with
  `--dequantize`, to float; both modes run on CPU or Metal.

## License

MIT - see [LICENSE](https://github.com/asher/mlx-kquant/blob/main/LICENSE).

### Acknowledgements

mlx-kquant builds on the following projects; their license texts ship in the wheel under
[`mlx_kquant/licenses/`](https://github.com/asher/mlx-kquant/tree/main/mlx_kquant/licenses):

- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** - the K-quant, IQ, and legacy block
  codec formats and the quantization / dequantization algorithms that encode and decode them (including
  the IQ codebook / grid tables, transcribed verbatim) are derived from ggml's reference implementation.
- **[gguf-tools](https://github.com/antirez/gguf-tools)** - used to implement a zero-copy GGUF loader
  for downstream projects, statically linked into built wheels.
- **[MLX](https://github.com/ml-explore/mlx)** - the extension links `libmlx`, the kernels compile
  against MLX's bundled headers, and parts of the Metal kernels are adapted from MLX's quantized and
  steel-GEMM kernels.
- **[omlx](https://github.com/jundot/omlx)** (Apache-2.0) - the DeepSeek-V4-Flash DSA kernels
  (`dsa_sparse_attention`, `dsa_indexer_scores`, `dsa_topk_indices`) are ported, with modifications,
  from omlx's `glm_moe_dsa` custom kernels (Copyright OpenAI / Apple Inc., per file).
