# LoRA on a kquant base

A kquant-quantized model is a **frozen base** you can adapt with LoRA: the packed
wire bytes never change, and a small low-rank adapter is trained (or attached) on
top. This works because the `kq.quantized_matmul` / `kq.gather_qmm` ops define a
gradient-with-respect-to-the-input `vjp` - gradient flows *through* the frozen
base to reach the trainable adapter, while the quantized weights themselves carry
no gradient (they are frozen, exactly as LoRA wants).

`mlx_kquant.mlx_lm_patch.patch_mlx_lm_lora()` teaches a stock
[mlx-lm](https://github.com/ml-explore/mlx-lm) install to recognise the kquant
modules in `mlx_kquant.nn` - so mlx-lm's own LoRA tuner, adapter loading, and the
`mlx-kquant fuse` merge tool all work on a kquant checkpoint. Call it once, before
building LoRA layers or loading adapters; it is idempotent.

> For a complete GGUF model runtime (load any community GGUF, generate, serve),
> see the separate **`gguf-mlx`** package (`pip install gguf-mlx`), which is built
> on these ops.

## Worked example: a pirate Qwen3-0.6B

End to end - quantize a small model, LoRA-train it to talk like a pirate, merge
the adapter, and chat with it. Needs the `[tools]` extra. Uses the tiny
[`GPT007/pirate_speak`][pirate] dataset (100 chat turns).

[pirate]: https://huggingface.co/datasets/GPT007/pirate_speak

**1. Build the training data.** The dataset ships as Llama-3-formatted text; pull
out the user/assistant turns and re-emit them as mlx-lm chat records, so the
trainer applies *Qwen3's* chat template:

```python
# prep_pirate.py
import json, re
from pathlib import Path
from datasets import load_dataset

ds = load_dataset("GPT007/pirate_speak", split="train")
turn = re.compile(
    r"user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>.*?"
    r"assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
    re.DOTALL,
)
records = []
for row in ds:
    m = turn.search(row["text"])
    if m:
        records.append({"messages": [
            {"role": "user", "content": m.group(1).strip()},
            {"role": "assistant", "content": m.group(2).strip()},
        ]})

out = Path("pirate-data"); out.mkdir(exist_ok=True)
split = max(1, len(records) // 10)   # 10% validation
(out / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records[:split]))
(out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records[split:]))
print(f"wrote {len(records) - split} train / {split} valid -> {out}/")
```

```sh
python prep_pirate.py     # -> pirate-data/{train,valid}.jsonl
```

**2. Quantize the base** to a K-quant MLX checkpoint:

```sh
mlx-kquant quantize --model Qwen/Qwen3-0.6B --preset q5_k_m --mlx-path qwen3-0.6b-q5_k_m
```

**3. Train the LoRA adapter** on the kquant base (`mlx-kquant lora` is mlx-lm's
trainer with the kquant patch applied - all its flags work):

```sh
mlx-kquant lora \
    --model qwen3-0.6b-q5_k_m --train \
    --data ./pirate-data \
    --iters 150 --batch-size 4 --num-layers 8 \
    --adapter-path pirate-adapter
```

**4. Merge** the adapter back into the base (stays kquant; `--dequantize` for a
float checkpoint instead):

```sh
mlx-kquant fuse --model qwen3-0.6b-q5_k_m \
    --adapter-path pirate-adapter \
    --save-path qwen3-0.6b-pirate-q5_k_m
```

**5. Chat** with the result:

```sh
mlx-kquant run --model qwen3-0.6b-pirate-q5_k_m \
    --prompt "What's the weather like today?"
```

The base Qwen3-0.6B answers plainly; after fine-tuning it answers in character -
"Arrrr, I be needin' to give ye the weather for today ...". (150 iterations on 90
examples is under a minute on an M-series GPU and is enough to pick up the style;
turn it up for a stronger effect, but watch for overfitting on a set this small.)

The merge re-quantizes the adapted weights, and at `q5_k_m` the persona survives
that round-trip. At `q4` and below the rounding can wash a small fine-tune out -
keep the adapter separate (attach it) or use `--dequantize`. The
[walkthrough](walkthrough.md) measures this end to end.

## Attach an adapter (inference / no training)

The simplest path is the CLI: `mlx-kquant run --adapter-path` attaches the adapter
at load time and generates - the base wire bytes are never modified, and the deltas
stay full-precision (the highest-fidelity way to run a fine-tune):

```sh
mlx-kquant run --model my-model-q4km --adapter-path adapters/ \
    --prompt "What's the weather like today?"
```

In Python, attach it yourself:

```python
from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora
from mlx_kquant.loader import load
from mlx_lm.tuner.utils import load_adapters

patch_mlx_lm_lora()                 # before any LoRA wrapping
model, config = load("my-model-q4km")   # a kquant checkpoint
load_adapters(model, "adapters/")       # adapter_config.json + adapters.safetensors
out = model(tokens)                      # base + low-rank delta
```

`load_adapters` reads the standard mlx-lm adapter layout. The wrapped layers
dispatch the kquant base plus the low-rank delta on every forward - no weights are
modified on disk.

## Train an adapter

Training uses mlx-lm's own LoRA trainer unchanged - the patch makes the kquant
layers adaptable, and the ops' `vjp` makes them differentiable. The base stays
frozen; only the adapter (and any unfrozen norm/bias) carries gradient.

The `mlx-kquant lora` subcommand is a thin pass-through to that trainer with the
patch applied, so `--model` may be a kquant checkpoint and every mlx-lm `lora`
flag works unchanged:

```sh
mlx-kquant lora --model my-model-q4km --train --data ./data --iters 300
# writes adapters/adapter_config.json + adapters.safetensors
```

(Run `mlx-kquant lora --help` for the full flag list - it is mlx-lm's.) See the
[worked example](#worked-example-a-pirate-qwen3-06b) below for an end-to-end run.

The gradient path is validated end-to-end in `tests/test_lora_patch.py`
(`test_lora_trains_dense` / `test_lora_trains_moe`): a trainable projection
upstream of a frozen kquant base receives a finite, non-zero gradient through the
base's `vjp`, the loss falls, and the frozen wire bytes are byte-unchanged after
the step. The MoE variant exercises the `kq.gather_qmm` `vjp` (with its
`scatter_add` over routed experts).

Both the dense and MoE gradient paths resolve on **CPU or GPU** (the `vjp`
re-dispatches the same ops, which have a CPU decode path), so adapter training
does not require a GPU.

## Merge an adapter - `mlx-kquant fuse`

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

The codec for each re-encoded tensor is **not** re-derived from a recipe - it is
read off the base layer, so every tensor keeps the exact codec it was originally
quantized with, and the fused checkpoint's `per_tensor` map is unchanged.

### Why keep-kquant is lossy

Merging changes the weight (`W -> W + delta`), and the LoRA delta `delta` is arbitrary
float that does not land on the quant grid, so re-encoding rounds it - the same
kind of error as the original quantization. There is no exact in-place merge: even
re-encoding the *unchanged* weight is not bit-identical, because the encoder
refits each block's scale to the data (and the K-quant encoders run a scale/min
search and quantize their own sub-scales). `--dequantize` avoids all of this by
keeping the merged weight in float.

### Lower bit widths: keep the adapter separate

The re-quant rounding is small at q5/q6, but at q4 and below it can wash out a small
fine-tune *entirely* (re-quantizing a `--dequantize` float merge does the same -
there is no trick once you land back on a coarse grid). So the highest-fidelity
option is to **not merge**: keep the adapter as a file and attach it at runtime,
where the deltas stay full-precision. Merge for a single self-contained checkpoint;
prefer `--dequantize` (or a higher-bit base) when a low-bit merge would degrade the
fine-tune.

To carry the base's *general* calibration through a keep-kquant merge, pass the same
importance matrix you quantized the base with:

```sh
mlx-kquant fuse --model my-model-q4km --adapter-path adapters/ \
    --imatrix base.imatrix.dat --save-path my-model-fused
```

`--imatrix` steers the re-encode of the adapted tensors (it is ignored under
`--dequantize`); it preserves general quality, not a fine-tune the bit width is too
coarse to hold. See [imatrix.md](imatrix.md).

## DoRA is not supported

DoRA on a kquant base is **not supported** in this release. mlx-lm's DoRA dispatch
does not consult the `to_lora` hook the patch installs, so `--fine-tune-type dora`
on a kquant model raises mlx-lm's own "Can't convert layer" error. Use LoRA.
