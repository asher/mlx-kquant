# Building on the `kq.*` ops

mlx-kquant is an op layer (`kq.dequantize` / `quantized_matmul` / `gather_qmm` / `quantize`) plus a
`[tools]` layer that turns those ops into a checkpoint toolchain. The `[tools]`
modules are deliberately small and are the **reference for integrating the kernels elsewhere in the
MLX ecosystem** - a model runtime, a different on-disk format, another trainer. This page maps each
integration point to the file that implements it.

Everything here runs against a **stock `mlx` wheel** - importing `mlx_kquant` adds the `kq.*`
namespace; it does **not** register `mode="kquant"` onto `mx.*`, so integrations call `kq.*`
explicitly.

**Compatibility.** The ops require stock `mlx==0.31.2` (the C++ extension is ABI-pinned to it). The
`[tools]` integration points below (loader, recipes, the mlx-lm patch) additionally need
`mlx-lm>=0.27`. The on-disk checkpoint format is stable and mirrors what a future in-core kquant mode
would read.

## The wire-byte contract

A quantized weight is a `uint8` array of K-quant wire bytes: a 2-D `[out_features, bytes_per_row]`
(linear / embedding) or 3-D `[num_experts, out_features, bytes_per_row]` (MoE / batched). The codec
name (`"q4_k"`, `"q8_0"`, ...) carries everything else - `mlx_kquant.codec_geometry` derives
`group_size` / `bits` / `bytes_per_block` from it, and `bytes_per_row(codec, in_features)` /
`in_features(codec, row_bytes)` convert between logical width and packed width. K-quant scales live
*inside* the bytes, so the `scales` argument to the ops is a vestigial `[1]` placeholder kept for
shape symmetry with MLX's affine quant; `kq.quantize` returns one for you.

```python
import mlx_kquant as kq
wq, scales = kq.quantize(w, "q4_k")                      # float -> wire bytes (GPU)
y = kq.quantized_matmul(x, wq, scales, "q4_k", transpose=True)   # x @ dequant(w).T
```

## Codecs

Ten codecs, all defined in `mlx_kquant.codec_geometry.CODEC_GEOMETRY` as
`(group_size, bits, bytes_per_block, weights_per_block)`:

| codec | bits | weights/block | bytes/block | family |
|-------|-----:|--------------:|------------:|--------|
| `q8_0` | 8 | 32 | 34 | block |
| `q4_0` | 4 | 32 | 18 | block |
| `q4_1` | 4 | 32 | 20 | block (+min) |
| `q5_0` | 5 | 32 | 22 | block |
| `q5_1` | 5 | 32 | 24 | block (+min) |
| `q2_k` | 2 | 256 | 84 | K-quant superblock |
| `q3_k` | 3 | 256 | 110 | K-quant superblock |
| `q4_k` | 4 | 256 | 144 | K-quant superblock |
| `q5_k` | 5 | 256 | 176 | K-quant superblock |
| `q6_k` | 6 | 256 | 210 | K-quant superblock |

`weights_per_block` (`wpb`) is the granularity that matters for layout: K-quants pack 256 weights per
superblock, the block codecs 32. The duck-typed `group_size` attribute on a `KQuant*` module
equals `wpb`. All ten encode on CPU or Metal; the **imatrix** argument to `kq.quantize` steers only the
five superblock K-quants (`wpb == 256`) and is a no-op on the `wpb == 32` codecs.

Presets (`mlx_kquant.recipes`) are **purely codec-name-based** - there is no affine "4-bit / 8-bit"
abstraction that resolves to a codec. A preset (`q4_k_s/m/xl/moe`, `q5_k_s/m/xl/moe`, `q3_k_m`,
`q6_k`, `q6_k_xl`, `q2_k`, `q8`) maps tensor *roles* (attention, embedding, lm_head, routed/shared
expert, ...) directly to codec names, with per-role bumps and layer-position heuristics. See
**Encoding** below.

## On-disk checkpoint format

`mlx_kquant.convert.save` writes a standard MLX checkpoint - `config.json` plus (sharded)
safetensors - that the loader and a patched stock mlx-lm can read.

`config.json` carries the quant block **twice**, under `quantization` and a mirrored
`quantization_config`, each:

```json
{"mode": "kquant", "per_tensor": {"model.layers.0.self_attn.q_proj": "q4_k", "...": "..."}}
```

The `per_tensor` keys are **bare module paths** (no `.weight` suffix). Tensors:

| key | dtype | shape | notes |
|-----|-------|-------|-------|
| `<path>.weight` | uint8 | `[out, bytes_per_row]` or `[E, out, bytes_per_row]` (MoE) | the K-quant wire bytes |
| `<path>.scales` | uint8 | `[1]` | placeholder - real scales live inside the wire bytes |
| `<path>.bias` | source dtype | `[out]` or `[E, out]` | optional |

Unquantized tensors (norms, router gates, anything the recipe leaves in float) keep their source
dtype. There is **no** affine-style `.biases` (zero-point) tensor - K-quant zero-points are encoded
in the bytes.

## Layer modules (`mlx_kquant/nn.py`)

`KQuantLinear` / `KQuantEmbedding` / `KQuantSwitchLinear` / `KQuantMultiLinear` are `nn.Module`s that
hold the wire bytes and dispatch the matching op in `__call__`. They duck-type mlx-lm's quantized
layers - each carries `mode="kquant"`, `group_size`, `bits`, `kquant_type` (the codec), and
`biases=None` - so code that special-cases `nn.QuantizedLinear` (a trainer, a fuser) can treat them
the same way. They `freeze()` themselves in `__init__`: the packed weights are not trainable.

To make your own layer, store the `uint8` weight and call the op - that is all `KQuantLinear` does.

### How kquant layers differ from mlx-lm's quantized layers

A `KQuant*` module is the **already-quantized terminal form**, so - like mlx-lm's own
`nn.QuantizedLinear` - it deliberately has **no `to_quantized` method**. This matters for discovery:
`nn.quantize`'s default predicate is literally `lambda _, m: hasattr(m, "to_quantized")`, so a kquant
layer is silently skipped by any walk that keys on it (correct - you don't quantize twice, but a
consumer that *enumerates* quantized layers that way will miss them). The modules are also already
`freeze()`-d and expose `biases=None`.

Detect a kquant layer by its duck-typed marker, **not** by type or the quantize predicate:

```python
def is_kquant(m) -> bool:
    return getattr(m, "mode", None) == "kquant"   # NOT isinstance(m, nn.QuantizedLinear)
```

The `mode` / `bits` / `group_size` / `kquant_type` attributes exist precisely so a kquant layer reads
the same as an affine `nn.QuantizedLinear` to introspection code.

## The module swap (`install_kquant_modules`)

`from mlx_kquant.nn import install_kquant_modules` (defined in `mlx_kquant/_install.py`, re-exported
from `nn`). `install_kquant_modules(model, {"<path>.weight": codec, ...})` walks a constructed model
and replaces each named quantizable leaf with the matching `KQuant*` module, sized from the codec. It
returns the count replaced. This is the one call that turns a float model into a kquant model in
memory; a loader or converter just has to produce the `{path: codec}` map.

The keys here are `<path>.weight`-suffixed - note this differs from the **bare** paths in the on-disk
`per_tensor` map (the loader bridges the two; see **Loading a checkpoint**).

## Loading a checkpoint (`mlx_kquant/loader.py`)

`loader.load` is the pattern for reading a kquant checkpoint on stock mlx: resolve the path, build the
mlx-lm model class for the config, `install_kquant_modules` from the config's `per_tensor` map, then
`model.load_weights(...)` the `uint8` tensors. Stock `mlx_lm.load_model` can't do this - it routes the
quant config into `nn.quantize` - which is exactly why the swap is explicit. Adapt this flow for any
on-disk layout: only the "where do the bytes and the codec map come from" step changes.

**Path-key bridge.** The on-disk `per_tensor` keys are bare module paths, but `install_kquant_modules`
keys on `<path>.weight`. The loader normalizes between them right before the swap (`loader.py`):

```python
weight_keyed = {f"{path}.weight": codec for path, codec in per_tensor.items()}
install_kquant_modules(model, weight_keyed)
```

So a converter emits bare paths, the installer consumes `.weight`-suffixed ones, and the loader is the
single place that translates - keep that contract if you write your own loader.

## Encoding (`mlx_kquant/convert.py`, `recipes.py`)

`convert.quantize_model(model, config, preset=...)` is the create side: `recipes.classify_tensors`
tags each tensor by role, `recipes.resolve_codec_map` turns a preset into a `{path: codec}` map,
each weight is encoded with `kq.quantize` (optionally steered by an imatrix), and the `KQuant*` module
is swapped in. `convert.save` writes a standard MLX checkpoint (`config.json` + sharded safetensors).
Reuse `recipes` to drive your own encoder, or call `kq.quantize` directly for a uniform quant.

### Encode constraints

- **CPU or Metal.** `kq.quantize` has a scalar CPU path for all ten codecs (`stream=mx.cpu`), as do
  the decode ops (`kq.dequantize` / `quantized_matmul` / `gather_qmm`), so the whole pipeline -
  quantize, load, inference, LoRA training, and both `fuse` modes - runs without a GPU. The CPU
  encoder is a port of the Metal kernels: the flat codecs and `q6_k` are byte-identical across
  streams, while the four codecs that reduce `sigma2` (`q2_k`/`q4_k`/`q5_k`, and `q3_k` with an
  imatrix) can differ by an ULP-tied level but reconstruct equally well.
- **Row-width divisibility.** A row's logical width must be a whole number of blocks:
  `in_features % weights_per_block == 0`. `bytes_per_row(codec, in_features)` and the inverse
  `in_features(codec, row_bytes)` raise `ValueError` otherwise, and `convert.quantize_model` skips a
  mis-sized tensor with a printed message rather than corrupting it. (256-wide blocks make the
  superblock K-quants the stricter case.)
- **No separate bias inside the block.** The codec carries scales/zero-points in its bytes; a layer
  `bias` is stored as its own `<path>.bias` tensor, untouched by the encoder.

## Interop with stock mlx-lm (`mlx_kquant/mlx_lm_patch.py`)

The most reusable pattern in the repo: teach an **unmodified** mlx-lm to handle a new quant family by
monkeypatching a couple of its load/tuner functions instead of forking it. Two idempotent entry
points, each a narrow shim over a stable mlx-lm function:

### Loading on stock mlx-lm (inference)

`patch_mlx_lm_load()` is the **load-only** shim - all an inference / eval / serving consumer needs. It
wraps `mlx_lm.utils.load_model` so a kquant config routes through `loader.load`, and every non-kquant
checkpoint falls through unchanged. Because everything mlx-lm exposes is built on `load_model`,
`mlx_lm.load`, `mlx_lm.generate`, and the CLIs all then open a kquant checkpoint transparently:

```python
from mlx_kquant.mlx_lm_patch import patch_mlx_lm_load
patch_mlx_lm_load()                 # process-wide; call once before mlx_lm.load
from mlx_lm import load
model, tokenizer = load("my-model-q4km")
```

It is a process-wide monkeypatch and idempotent - call it during startup, before the first load. A
consumer that resolves `load_model` through the module at call time (as `mlx_lm.load` does) picks up
the patch; one that captured `from mlx_lm.utils import load_model` at import time before patching would
not.

### Adapting and fusing (LoRA)

`patch_mlx_lm_lora()` calls `patch_mlx_lm_load()` for you, then adds the train/merge shims:

- it attaches a `to_lora` method to the `KQuant*` modules, which mlx-lm's tuner consults first when
  discovering and wrapping adaptable layers (recovering in/out dims from the wire-byte geometry);
- it overrides `LoRA*.fuse` to merge into a kquant base (dequantize, add the delta, optionally
  re-encode - re-encode runs on CPU or Metal), deferring to the original for non-kquant bases.

This is the template for adding loader / trainer / fuser support for any custom layer type without
owning a fork. See [docs/lora.md](lora.md) for the end-to-end workflow.

## Training through the ops

`kq.quantized_matmul` and `kq.gather_qmm` define a gradient-with-respect-to-the-input `vjp` (the
quantized weights stay frozen), so gradient flows through a frozen kquant base to a trainable adapter
on top. That is what makes LoRA training work; see [docs/lora.md](lora.md).
