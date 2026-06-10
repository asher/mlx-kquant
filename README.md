# mlx-kquant

Bring **K-quant precision to [MLX](https://github.com/ml-explore/mlx)** on Apple Silicon: a C++/Metal
**extension** for a stock `mlx` wheel that adds the llama.cpp K-quant / legacy codecs as native MLX
ops, plus a toolchain that quantizes a model into a **K-quant MLX safetensors checkpoint** and runs,
LoRA-trains, and fuses it.

Two layers, one wheel:

- **Ops** - a `kq.*` namespace (`dequantize`, `quantized_matmul`, `gather_qmm`, `quantize`, and a
  zero-copy `load_gguf` reader) backed by precompiled Metal kernels shipped in a `.metallib`. All ten
  codecs: `q2_k, q3_k, q4_k, q5_k, q6_k` and `q4_0, q4_1, q5_0, q5_1, q8_0`.
- **Tooling** - `mlx-kquant quantize / run / lora / fuse` and a `loader` that create and run K-quant
  checkpoints in **MLX-native safetensors** - the same format `mlx_lm` reads. No GGUF files in or out.

## Why

K-quant is the precision recipe behind the strongest small-footprint llama.cpp quants; this makes it
first-class on MLX. Quantize an HF / `mlx-lm` model to a uniform- or mixed-precision K-quant
checkpoint, then load, generate, LoRA-train, and fuse it - all on a stock `mlx` wheel, all in MLX
safetensors. The kernels are tuned for real models (matrix-contiguity handling for fused MoE experts,
single-pass NAX matmul) - see [Performance](#performance).

> mlx-kquant is **not** a GGUF runtime - its tooling reads and writes MLX safetensors, not `.gguf`
> files. The `kq.load_gguf` op is a low-level reader that lets downstream packages such as
> [`gguf-mlx`](#running-gguf-files--gguf-mlx) build a pure-Python GGUF runtime on top; to run a
> `.gguf` file directly, use gguf-mlx.

## Install

Requires **macOS on Apple Silicon**, the Metal toolchain (`xcrun metal`), and the exact pinned MLX
wheel:

```sh
pip install "mlx==0.31.2"     # pinned, ABI-matched stock wheel
pip install -e .              # builds _ext + mlx_kquant.metallib
pip install -e ".[tools]"     # + mlx-lm, for the quantize / run / lora / fuse CLI
```

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

The four ops operate on raw GGUF wire bytes. K-quant scales live *inside* the packed bytes, so the
`scales` argument is a vestigial placeholder (the API keeps it for shape symmetry with MLX's affine
quant); `kq.quantize` returns one for you.

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

`load_gguf` is a low-level **reader**, not a model loader - a zero-copy building block (tensors as
wire bytes + decoded metadata, ~15 GB/s via a C++ mmap memcpy) for runtimes like
[`gguf-mlx`](#running-gguf-files--gguf-mlx). mlx-kquant's own checkpoints are MLX safetensors
([below](#create-and-run-a-checkpoint)).

```python
arrays, codecs, metadata, shapes = kq.load_gguf("model.gguf")
# arrays[name]  -> mx.array (K-quant tensors are uint8 wire bytes)
# codecs[name]  -> "q4_k" / "q6_k" / ...        (K-quant tensors only)
# metadata[key] -> decoded GGUF KV value
# shapes[name]  -> logical (GGUF-native) shape
```

### Create and run a checkpoint

The CLI (the `[tools]` extra adds `mlx-lm`) quantizes an HF / `mlx-lm` model into a K-quant **MLX
safetensors** checkpoint and runs it - no GGUF involved:

```sh
pip install "mlx-kquant[tools]"
mlx-kquant quantize --model Qwen/Qwen3-0.6B --preset q4_k_m --mlx-path qwen3-q4
mlx-kquant run --model qwen3-q4 --prompt "Explain entropy in one sentence."
```

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

### Use it in your own model

Ready-made modules that store the wire bytes and dispatch the matching `kq.*` op ship in
`mlx_kquant.nn`:

```python
from mlx_kquant.nn import KQuantLinear, KQuantEmbedding, KQuantSwitchLinear

lin = KQuantLinear(in_dims=512, out_dims=256, bias=False, codec="q4_k")
lin.weight = wq          # uint8 wire bytes [out_dims, bytes_per_row]
lin.scales = scales      # [1] placeholder (K-quant scales live in the bytes)
y = lin(x)               # kq.quantized_matmul under the hood
```

`KQuantEmbedding` (with a tied-`as_linear`), the `gather_qmm`-backed `KQuantSwitchLinear` for MoE
experts, and `KQuantMultiLinear` (absorbed-MLA) are exported alongside it. To swap the quantizable
leaves of a whole constructed mlx-lm model in one call, use
`mlx_kquant.nn.install_kquant_modules(model, {"<path>.weight": "q4_k", ...})`. The **gguf-mlx**
package builds its full GGUF runtime on exactly these classes.

The `[tools]` layer is itself a worked reference for wiring `kq.*` into the MLX ecosystem: the loader,
encoder, layer modules, and the mlx-lm monkeypatch are all small and self-contained. See
**[docs/integration.md](docs/integration.md)** if you're building on the ops.

### LoRA fine-tuning

A kquant checkpoint is a frozen base you can adapt with LoRA - attach an adapter for inference, train
one (the matmul/gather ops define a gradient-through-the-base `vjp`, so the adapter is differentiable
while the quantized weights stay frozen), and merge it back with `mlx-kquant fuse` (re-encode to
kquant, or `--dequantize` to float). One call wires it into stock mlx-lm:

```python
from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora
patch_mlx_lm_lora()   # before building LoRA layers / loading adapters; idempotent
```

See **[docs/lora.md](docs/lora.md)** for attach / train / merge workflows. (DoRA on a kquant base is
not supported - use LoRA.)

## Running GGUF files - `gguf-mlx`

mlx-kquant produces MLX safetensors checkpoints; it does not run `.gguf` files. To **load and run a
complete GGUF model** - name remap, config + tokenizer synth from GGUF metadata, K-quant leaf swap,
dense or MoE, plus a CLI and OpenAI-compatible servers - use the separate **`gguf-mlx`** package. It
depends on this one and builds a pure-Python GGUF runtime on the `kq.*` ops (notably `kq.load_gguf`),
with no conversion and no safetensors round-trip:

```sh
pip install gguf-mlx          # pulls in mlx-kquant + mlx-lm, transformers, tokenizers, gguf
gguf-mlx-run model.gguf --prompt "Explain entropy." --max-tokens 128
```

```python
from gguf_mlx import load_model, generate

model, config, tokenizer = load_model("model.gguf")
print(generate(model, tokenizer, "Explain entropy.", max_tokens=128))
```

## Performance

The Metal kernels use a single-pass NAX matmul and matrix-contiguity handling for fused MoE expert
weights. Measured on an M5 Max (128 GB):

| Model | Codec | Decode (tok/s) | Prefill pp512 (tok/s) |
|-------|-------|---------------:|----------------------:|
| gemma-4-26B-A4B-it (MoE) | UD-Q4_K_XL | ~111 | ~2330 |
| Qwen3.5-9B (dense)       | Q5_K_L     | ~83  | ~2396 |

## How it works

- **Own ops.** Four `Primitive` subclasses (`KQuantDequantize`, `KQuantMatmul`, `KQuantGatherQMM`,
  `KQuantQuantize`) and their op functions live entirely in the extension.
- **Precompiled metallib on stock headers.** The `kq_*` kernels compile against the stock wheel's
  steel-GEMM headers and ship as `mlx_kquant.metallib`; host dispatch resolves them through MLX's
  exported `Device::get_kernel`. No JIT, no steel host structs.
- **Codec registry** derives `group_size`/`bits` from the codec name, so callers pass only
  `kquant_type`.
- **CPU and GPU execution.** Every op - the decode ops (`dequantize` / `quantized_matmul` /
  `gather_qmm`) and `quantize` (encode) - runs on either stream. A scalar CPU path covers all 10
  codecs and backs `stream=mx.cpu`, doubling as the correctness oracle the Metal kernels are A/B'd
  against, so the full quantize/decode pipeline (and the op tests) runs in CI without a GPU.

## Scope

mlx-kquant is two things on one wheel:

- **The op layer** - the `kq.*` namespace and its Metal kernels. Point your own model's quantized
  layers at these ops (see [Use it in your own model](#use-it-in-your-own-model)).
- **The checkpoint toolchain** - `quantize` / `loader` / `run` / `lora` / `fuse`, which create and run
  **MLX safetensors** K-quant checkpoints (the `[tools]` extra).

What it is **not**: a GGUF model runtime. Reading and running `.gguf` files is the job of the separate
**[`gguf-mlx`](#running-gguf-files--gguf-mlx)** package, which is built on this one.

## Codec reference

| Codec | Block | Bits | Bytes/block | Notes |
|-------|------:|-----:|------------:|-------|
| q2_k  | 256 | 2 |  84 | K-quant superblock |
| q3_k  | 256 | 3 | 110 | K-quant superblock |
| q4_k  | 256 | 4 | 144 | K-quant superblock |
| q5_k  | 256 | 5 | 176 | K-quant superblock |
| q6_k  | 256 | 6 | 210 | K-quant superblock |
| q4_0  |  32 | 4 |  18 | legacy block |
| q4_1  |  32 | 4 |  20 | legacy block (+min) |
| q5_0  |  32 | 5 |  22 | legacy block |
| q5_1  |  32 | 5 |  24 | legacy block (+min) |
| q8_0  |  32 | 8 |  34 | legacy block |

## Version pinning

Pinned to `mlx==0.31.2`. The kernels include MLX's steel headers and the extension links `libmlx`,
binding it to that release's ABI and header API. To move to a newer MLX: update the bundled headers
under `metal/mlx/backend/metal/kernels/` for that wheel, rebuild, and re-run the test suite.

## Tests

```sh
python -m pytest tests/      # dequant / matmul / gather / codecs / cpu_decode / encode
```

## Requirements

- **macOS on Apple Silicon** (M-series) with a working Metal toolchain (`xcrun metal`) for a
  build-from-source install. Prebuilt wheels (when published) need only the runtime GPU.
- **Python >= 3.10** (the pinned `mlx==0.31.2` ships no cp39 wheel).
- **`mlx==0.31.2`** exactly - the kernels include MLX's steel headers and the extension links
  `libmlx`, so the ABI is version-locked (see [Version pinning](#version-pinning)).

## Limitations

- **Decode and encode both run on CPU or Metal.** `dequantize` / `quantized_matmul` / `gather_qmm`
  and `quantize` all have a scalar CPU path (`stream=mx.cpu`) covering every codec. The build still
  needs the Metal toolchain (the extension links `libmlx` and bundles a metallib); a Metal-free Linux
  build - which the CPU paths unblock - is on the roadmap. No NVIDIA/AMD-GPU support.
- The library is the **op layer**, not a model runtime. To load and run full GGUF models, use the
  separate [`gguf-mlx`](#running-gguf-files--gguf-mlx) package.
- **LoRA, not DoRA.** LoRA adapters train, attach, and fuse on a kquant base (see
  [docs/lora.md](docs/lora.md)); DoRA is not supported. `fuse` re-encodes to kquant or, with
  `--dequantize`, to float; both modes run on CPU or Metal.

## Roadmap

- A Metal-free **Linux** build (CPU train/infer), contingent on the mlx Linux wheel exercising the
  CPU eval paths. The scalar CPU encode/decode paths are in place; this is the remaining piece.

## License

MIT - see [LICENSE](LICENSE).
