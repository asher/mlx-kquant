# mlx-kquant

Standalone GGUF **K-quant** ops for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon,
packaged as a C++/Metal **extension** — it runs against a **stock, unmodified `mlx` wheel**, with no
fork of MLX core required.

It exposes its own op namespace — `kq.dequantize`, `kq.quantized_matmul`, `kq.gather_qmm`,
`kq.quantize`, and a fast `kq.load_gguf` — backed by precompiled Metal kernels that build on MLX's
own steel-GEMM machinery and ship in a `.metallib` inside the wheel. All ten GGUF K-quant / legacy
codecs are supported: `q2_k, q3_k, q4_k, q5_k, q6_k` and `q4_0, q4_1, q5_0, q5_1, q8_0`.

## Why

K-quant support used to live as a fork of MLX core. Re-homing it as an extension means it installs
on top of a released `mlx` wheel and can be offered independently — no locally-built core, no
divergence from upstream. The Metal kernels are byte-identical to the in-core research
implementation, so numerics and throughput match it (and the dispatch tuning here actually edges it
out on real models — see [Performance](#performance)).

## Install

Requires **macOS on Apple Silicon**, the Metal toolchain (`xcrun metal`), and the exact pinned MLX
wheel:

```sh
pip install "mlx==0.31.2"     # pinned, ABI-matched stock wheel
pip install -e .              # builds _ext + mlx_kquant.metallib
```

Smoke-test the toolchain:

```python
import mlx_kquant as kq
kq.codecs()          # -> ['q2_k', 'q3_k', ..., 'q8_0']
kq.metallib_loads()  # -> True  (the bundled metallib opened on the Metal device)
```

> The extension links `libmlx` and its kernels `#include` MLX's steel headers, so it is bound to an
> exact MLX ABI **and** steel-header API. The pin is intentionally `==`, never `>=`; bumping `mlx`
> requires re-vendoring `quantized_utils.h` and recompiling. See [Version pinning](#version-pinning).

## Quickstart

The four ops operate on raw GGUF wire bytes. K-quant scales live *inside* the packed bytes, so the
`scales` argument is a vestigial placeholder (the API keeps it for shape symmetry with MLX's affine
quant); `kq.quantize` returns one for you.

```python
import mlx.core as mx
import mlx_kquant as kq

N, K = 256, 512                       # q4_k: K must be a multiple of 256
w = mx.random.normal((N, K))

# encode float -> K-quant wire bytes (GPU only); optional imatrix steers the encoder
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

Load a GGUF file (tensors as wire bytes + decoded metadata, ~15 GB/s via a C++ mmap memcpy):

```python
arrays, codecs, metadata, shapes = kq.load_gguf("model.gguf")
# arrays[name]  -> mx.array (K-quant tensors are uint8 wire bytes)
# codecs[name]  -> "q4_k" / "q6_k" / ...        (K-quant tensors only)
# metadata[key] -> decoded GGUF KV value
# shapes[name]  -> logical (GGUF-native) shape
```

### Use it in your own model

Stock MLX has no `mode="kquant"`, so you don't reuse `nn.QuantizedLinear`. Ready-made modules that
store the wire bytes and dispatch the matching `kq.*` op ship in `mlx_kquant.nn`:

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

## Running full GGUF models — `gguf-mlx`

This extension is the low-level op/kernel layer. To **load and run a complete GGUF model** —
name remap, config + tokenizer synth from GGUF metadata, K-quant leaf swap, dense or MoE, plus
a CLI and OpenAI-compatible servers — use the separate **`gguf-mlx`** package. It depends on
this one and wires the `kq.*` ops into a full runtime, with no conversion and no safetensors
round-trip:

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

Because the Metal kernels are byte-identical to the in-core research implementation, throughput
matches it — and the dispatch tuning in this extension (matrix-contiguity handling for fused MoE
expert weights, single-pass NAX matmul instead of split-K) edges it out on real models.

Measured on an M5 Max (128 GB):

| Model | Codec | Decode (tok/s) | Prefill pp512 (tok/s) |
|-------|-------|---------------:|----------------------:|
| gemma-4-26B-A4B-it (MoE) | UD-Q4_K_XL | ~111 | ~2330 |
| Qwen3.5-9B (dense)       | Q5_K_L     | ~83  | ~2396 |

## How it works

- **Own ops, not a core override.** Four `Primitive` subclasses (`KQuantDequantize`, `KQuantMatmul`,
  `KQuantGatherQMM`, `KQuantQuantize`) and their op functions live entirely in the extension. The
  installed `mlx` is untouched. All primitives are inference-only.
- **Precompiled metallib on stock headers.** The vendored `kq_*` kernels compile against the stock
  wheel's steel-GEMM headers and ship as `mlx_kquant.metallib`; host dispatch resolves them through
  MLX's exported `Device::get_kernel`. No JIT, no steel host structs.
- **Codec registry** derives `group_size`/`bits` from the codec name, so callers pass only
  `kquant_type`.
- **CPU and GPU execution.** The decode ops (`dequantize` / `quantized_matmul` / `gather_qmm`) run on
  either stream — a scalar CPU path (all 10 codecs) backs `stream=mx.cpu` and doubles as the
  correctness oracle the Metal kernels are A/B'd against, so the op tests can run in CI without a GPU.
  **Encode (`quantize`) is GPU-only** until the CPU encoder lands in 0.2.0.

## Scope

This is an **op-level library** with its own `kq.*` namespace — deliberately *not* a drop-in that
re-enables a `mode="kquant"` argument on `mx.quantize` / `mx.quantized_matmul` / `nn.QuantizedLinear`.
To use it, point your model's quantized layers at the `kq.*` ops (see
[Use it in your own model](#use-it-in-your-own-model)); the separate **gguf-mlx** package is a
full worked example.

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
binding it to that release's ABI and header API. To move to a newer MLX: re-vendor that wheel's
`quantized_utils.h` (the shim at `metal/mlx/backend/metal/kernels/quantized_utils.h` = the wheel's
exact file plus two appended helpers), rebuild, and re-run the test suite.

## Tests

```sh
python -m pytest tests/      # dequant / matmul / gather / codecs / cpu_decode / encode
```

## Requirements

- **macOS on Apple Silicon** (M-series) with a working Metal toolchain (`xcrun metal`) for a
  build-from-source install. Prebuilt wheels (when published) need only the runtime GPU.
- **Python ≥ 3.10** (the pinned `mlx==0.31.2` ships no cp39 wheel).
- **`mlx==0.31.2`** exactly — the kernels include MLX's steel headers and the extension links
  `libmlx`, so the ABI is version-locked (see [Version pinning](#version-pinning)).

## Limitations

- **Decode runs on CPU or Metal; encode is GPU-only.** `dequantize` / `quantized_matmul` /
  `gather_qmm` have a scalar CPU path (`stream=mx.cpu`); `quantize` requires an Apple-Silicon GPU
  until the 0.2.0 CPU encoder. No NVIDIA/AMD/Linux-GPU support; a Metal-free Linux build is on the
  roadmap.
- The library is the **op layer**, not a model runtime. To load and run full GGUF models, use the
  separate [`gguf-mlx`](#running-full-gguf-models--gguf-mlx) package.

## Roadmap

- Training `vjp` on the matmul/gather ops → LoRA fine-tuning on kquant bases.
- A Metal-free **Linux** build (CPU train/infer), contingent on the mlx Linux wheel exercising the
  CPU eval paths.
- **0.2.0:** CPU **encode** parity → quantize on Linux.

## License

MIT — see [LICENSE](LICENSE). The Metal kernels under `metal/` are derived from
[ml-explore/mlx](https://github.com/ml-explore/mlx) (MIT) and the kquant research fork.
