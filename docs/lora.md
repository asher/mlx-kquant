# LoRA on a kquant base

A kquant-quantized model is a **frozen base** you can adapt with LoRA: the packed
wire bytes never change, and a small low-rank adapter is trained (or attached) on
top. This works because the `kq.quantized_matmul` / `kq.gather_qmm` ops define a
gradient-with-respect-to-the-input `vjp` — gradient flows *through* the frozen
base to reach the trainable adapter, while the quantized weights themselves carry
no gradient (they are frozen, exactly as LoRA wants).

`mlx_kquant.lora_patch.patch_mlx_lm_lora()` teaches a stock
[mlx-lm](https://github.com/ml-explore/mlx-lm) install to recognise the kquant
modules in `mlx_kquant.nn` — so mlx-lm's own LoRA tuner, adapter loading, and the
`mlx-kquant fuse` merge tool all work on a kquant checkpoint. Call it once, before
building LoRA layers or loading adapters; it is idempotent.

> For a complete GGUF model runtime (load any community GGUF, generate, serve),
> see the separate **`gguf-mlx`** package (`pip install gguf-mlx`), which is built
> on these ops.

## Attach an adapter (inference / no training)

```python
from mlx_kquant.lora_patch import patch_mlx_lm_lora
from mlx_kquant.loader import load
from mlx_lm.tuner.utils import load_adapters

patch_mlx_lm_lora()                 # before any LoRA wrapping
model, config = load("my-model-q4km")   # a kquant checkpoint
load_adapters(model, "adapters/")       # adapter_config.json + adapters.safetensors
out = model(tokens)                      # base + low-rank delta
```

`load_adapters` reads the standard mlx-lm adapter layout. The wrapped layers
dispatch the kquant base plus the low-rank delta on every forward — no weights are
modified on disk.

## Train an adapter

Training uses mlx-lm's own LoRA trainer unchanged — the patch makes the kquant
layers adaptable, and the ops' `vjp` makes them differentiable. The base stays
frozen; only the adapter (and any unfrozen norm/bias) carries gradient.

```python
from mlx_kquant.lora_patch import patch_mlx_lm_lora
patch_mlx_lm_lora()
# then drive mlx-lm's tuner as usual: linear_to_lora_layers(...) + an optimizer,
# or `mlx_lm.lora` with --model pointing at the kquant checkpoint.
```

The gradient path is validated end-to-end in `tests/test_lora_patch.py`
(`test_lora_trains_dense` / `test_lora_trains_moe`): a trainable projection
upstream of a frozen kquant base receives a finite, non-zero gradient through the
base's `vjp`, the loss falls, and the frozen wire bytes are byte-unchanged after
the step. The MoE variant exercises the `kq.gather_qmm` `vjp` (with its
`scatter_add` over routed experts).

Both the dense and MoE gradient paths resolve on **CPU or GPU** (the `vjp`
re-dispatches the same ops, which have a CPU decode path), so adapter training
does not require a GPU.

## Merge an adapter — `mlx-kquant fuse`

Once trained, fold the adapter back into the base with the `fuse` subcommand. Two
output modes:

```sh
# stays kquant: each merged weight is re-encoded with that tensor's own codec.
# GPU-only (re-encode uses the GPU); small per-codec re-quant rounding error.
mlx-kquant fuse --model my-model-q4km --adapter-path adapters/ \
    --save-path my-model-fused

# float: decode the base, add the delta, write a plain float checkpoint.
# Loads with stock `mlx_lm.load`; runs without a GPU; no re-quant error.
mlx-kquant fuse --model my-model-q4km --adapter-path adapters/ \
    --save-path my-model-fused-f16 --dequantize
```

### Which mode?

| | keep-kquant (default) | `--dequantize` |
|---|---|---|
| output | kquant (small, same footprint) | float (larger) |
| needs a GPU | yes (re-encode) | no (decode has a CPU path) |
| extra error vs the attached adapter | re-quant rounding | none (bf16 arithmetic noise only) |
| loads with | `mlx_kquant.loader.load` | stock `mlx_lm.load` |

The codec for each re-encoded tensor is **not** re-derived from a recipe — it is
read off the base layer, so every tensor keeps the exact codec it was originally
quantized with, and the fused checkpoint's `per_tensor` map is unchanged.

### Why keep-kquant is lossy

Merging changes the weight (`W → W + δ`), and the LoRA delta `δ` is arbitrary
float that does not land on the quant grid, so re-encoding rounds it — the same
kind of error as the original quantization. There is no exact in-place merge: even
re-encoding the *unchanged* weight is not bit-identical, because the encoder
refits each block's scale to the data (and the K-quant encoders run a scale/min
search and quantize their own sub-scales). `--dequantize` avoids all of this by
keeping the merged weight in float.

## DoRA is not supported

DoRA on a kquant base is **not supported** in this release. mlx-lm's DoRA dispatch
does not consult the `to_lora` hook the patch installs, so `--fine-tune-type dora`
on a kquant model raises mlx-lm's own "Can't convert layer" error. Use LoRA.
