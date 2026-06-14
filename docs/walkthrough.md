# End-to-end walkthrough: quantize -> run -> LoRA -> fuse

A complete, copy-pasteable tour of the `mlx-kquant` toolchain on Apple Silicon:
take a bf16 model, quantize it to a K-quant MLX checkpoint, run it, teach it to
talk like a pirate with LoRA, and merge the adapter back in - all on a stock
`mlx` wheel, all in MLX safetensors.

Model: **Qwen3-0.6B** (a small thinking/chat model). Everything below runs in a
couple of minutes on an M-series GPU and stays well under 1 GB of memory.

## Prerequisites

```sh
pip install "mlx-kquant[tools]"     # ops + the create/run/lora/fuse toolchain
```

Work in a scratch directory:

```sh
mkdir pirate-demo && cd pirate-demo
```

## 1. Get the bf16 base

Qwen3-0.6B ships natively in bf16. Pull it into a local folder:

```sh
hf download Qwen/Qwen3-0.6B --local-dir qwen3-0.6b-bf16
```

```sh
# sanity-check it's an unquantized bf16 checkpoint
python -c "import json; c=json.load(open('qwen3-0.6b-bf16/config.json')); \
print(c['model_type'], c['torch_dtype'], 'quantization=', c.get('quantization'))"
```

```
qwen3 bfloat16 quantization= None
```

## 2. Quantize to `q5_k_m`

```sh
mlx-kquant quantize --model qwen3-0.6b-bf16 --preset q5_k_m --mlx-path qwen3-0.6b-q5_k_m
```

```
[mlx-kquant] encoded 197 tensors
[mlx-kquant] wrote qwen3-0.6b-q5_k_m (197 tensors quantized)
```

The `q5_k_m` recipe is a mixed-precision K-quant: a `q5_k` base with the
quality-sensitive tensors bumped to `q6_k` - every layer's `v_proj`, the
`down_proj` of the first and last few layers plus a periodic subset of the rest
(the `_M` cadence), and a standalone output head if the model has one.
(Qwen3-0.6B ties its embeddings, so there is no separate head - `embed_tokens`
stays `q5_k`.) The result is ~415 MB, down from the 1.5 GB bf16:

```sh
python -c "import json,collections as C; \
q=json.load(open('qwen3-0.6b-q5_k_m/config.json'))['quantization']; \
print('mode=', q['mode']); \
print('codecs=', dict(C.Counter(q['per_tensor'].values())))"
du -sh qwen3-0.6b-q5_k_m
```

```
mode= kquant
codecs= {'q5_k': 155, 'q6_k': 42}
415M	qwen3-0.6b-q5_k_m
```

> **Optional - imatrix calibration.** You can steer the K-quant encoder with an
> importance matrix so it spends bits where a calibration corpus says they matter.
> It is most worthwhile at lower bit widths (q4 and below). See
> [imatrix.md](imatrix.md) for the `calibrate-imatrix` -> `quantize --imatrix` flow;
> the rest of this walkthrough runs fine without it.

## 3. Test run

```sh
mlx-kquant run --model qwen3-0.6b-q5_k_m \
    --prompt "What is the capital of France? Answer in one sentence." \
    --max-tokens 160 --temp 0
```

```
==========
<think>
Okay, the user is asking for the capital of France. I know that France's capital
is Paris. But I need to make sure I'm not mixing up any other cities. Let me
think. The capital is indeed Paris. I should confirm that there's no other city
with the same name. Yes, Paris is the capital. So the answer is Paris.
</think>

The capital of France is Paris.
==========
Prompt: 20 tokens, 585.611 tokens-per-sec
Generation: 85 tokens, 514.851 tokens-per-sec
Peak memory: 0.474 GB
```

Coherent and correct - the quant is sound. (Qwen3 is a thinking model, so it
reasons inside `<think>` first.)

## 4. Teach it to talk like a pirate (LoRA)

### 4a. Build the training data

Uses the tiny [`GPT007/pirate_speak`](https://huggingface.co/datasets/GPT007/pirate_speak)
dataset (100 chat turns). It ships as Llama-3-formatted text; pull out the
user/assistant turns and re-emit them as mlx-lm chat records so the trainer
applies *Qwen3's* chat template:

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
python prep_pirate.py
```

```
wrote 90 train / 10 valid -> pirate-data/
```

### 4b. Train the adapter on the kquant base

`mlx-kquant lora` is mlx-lm's own LoRA trainer with the kquant patch applied - so
`--model` may be a kquant checkpoint and every mlx-lm `lora` flag works. Gradient
flows through the frozen kquant base via the `vjp` the matmul/gather ops define;
only the adapter trains.

The base stays compact in memory while you train: `kq.quantized_matmul`
dequantizes on the fly inside the kernel (forward *and* backward), so there is no
full float copy of the base, and the frozen base carries no gradients or optimizer
state. That is the point - you can finetune a model you could not fit in fp16.
(If you *do* have the full-precision model and the memory to spare, finetuning it
and quantizing afterward gives a better result: the adapter then builds on a
higher-quality base instead of one that already carries quantization error.
Training on the quant is the path for when you can only fit the quant.)

```sh
mlx-kquant lora \
    --model qwen3-0.6b-q5_k_m --train \
    --data ./pirate-data \
    --iters 150 --batch-size 4 --num-layers 8 \
    --adapter-path pirate-adapter
```

```
Iter 1: Val loss 5.106, Val took 0.131s
Iter 10:  Train loss 4.514, Tokens/sec 4157.9, Peak mem 5.146 GB
Iter 50:  Train loss 3.323, Tokens/sec 4090.7, Peak mem 5.830 GB
Iter 100: Train loss 2.620, Tokens/sec 3974.9, Peak mem 5.830 GB
Iter 150: Val loss 3.386
Iter 150: Train loss 2.182
Saved final weights to pirate-adapter/adapters.safetensors.
```

Train loss falls 4.51 -> 2.18 in ~30s. We stop at 150 iters on purpose: this is
only 90 examples, so pushing further overfits - validation loss starts to climb
and greedy decoding can fall into "me hearty, me hearty ..." loops. Turn `--iters`
down or add data to taste.

### 4c. Generate with the adapter as a file (no merge)

The adapter is attached at runtime - the base wire bytes are never modified. Point
`mlx-kquant run` at the base with `--adapter-path` and it installs the LoRA shim,
loads the adapter, and generates; drop the flag to get the plain base. (Qwen3 is a
thinking model, so `run` emits a `<think>` block first; the pirate training data
had no thinking, so the adapted model thinks "empty" and gets straight to the
shanty.)

Base, no adapter:

```sh
mlx-kquant run --model qwen3-0.6b-q5_k_m \
    --prompt "What's the weather like today?" --max-tokens 200
```

```
<think>
Okay, the user is asking about the weather today. I need to figure out how to
respond. First, I should check if I have access to real-time data. Since I don't
have live updates, I can't provide the current weather. But maybe I can mention
that I can't check the weather right now and offer to help with other questions.
Let me structure my answer to be polite and offer further assistance.
</think>

I don't have access to real-time weather data, but I can help with other
questions! Let me know if you need advice on something else!
```

Same base, adapter attached:

```sh
mlx-kquant run --model qwen3-0.6b-q5_k_m --adapter-path pirate-adapter \
    --prompt "What's the weather like today?" --max-tokens 200
```

```
<think>

</think>

Arrrr, I be needin' to give ye the weather for today, but I be havin' a bit o'
sea shanty talkin' about the weather! Yer lookin' fer a weather report, or maybe
a sea chart?
```

### 4d. Merge the adapter back in (keep-kquant requant)

Merging is **optional** - attaching the adapter (4c) is the higher-fidelity path.
Merge when you want a single self-contained checkpoint with no separate adapter
file. `mlx-kquant fuse` folds the adapter into the base; by default it stays
kquant: each merged weight is decoded, the adapter delta is added, and the result
is re-encoded with that tensor's own codec.

```sh
mlx-kquant fuse \
    --model qwen3-0.6b-q5_k_m \
    --adapter-path pirate-adapter \
    --save-path qwen3-0.6b-pirate-q5_k_m
```

```
[mlx-kquant] fused 56 adapter layers -> qwen3-0.6b-pirate-q5_k_m (kquant)
```

56 = 8 layers x 7 projections.

> **Merging re-quantizes - mind the precision.** The merge rounds the updated
> weights back onto the codec's grid, and a LoRA delta is small. At `q5_k_m` (and
> higher) the persona survives the round-trip, as you will see below; at `q4` and
> below that rounding can wash the fine-tune out entirely - and re-quantizing a
> `--dequantize` float merge does exactly the same thing, so there is no trick that
> keeps a sub-grid signal once you land back on a 4-bit grid. To preserve a
> fine-tune:
>
> - **Keep the adapter separate** and attach it at runtime (section 4c). The deltas
>   stay full-precision and nothing is re-quantized - the highest-fidelity option.
>   The extra file is small (this adapter is ~6 MB; the size scales with LoRA rank x
>   adapted layers x model width, but stays far below the base). Merge only when you
>   want one self-contained checkpoint.
> - Or `mlx-kquant fuse --dequantize` for a faithful **float** checkpoint (no
>   re-quant; larger, and it loads with stock `mlx_lm`, not `mlx-kquant run`).
> - Or finetune the full-precision model and quantize once at the end.

### 4e. Run the merged model - pirate baked in

No adapter file needed now; the persona lives in the weights, so a plain
`mlx-kquant run` talks like a pirate:

```sh
mlx-kquant run --model qwen3-0.6b-pirate-q5_k_m \
    --prompt "What's the weather like today?" --max-tokens 200
```

```
<think>

</think>

Arrrr, I be needin' to give ye the weather for today, but I don't have any weather
data handy! Can ye tell me what ye mean by "today"?
```

The pirate style survives the `q5_k_m` merge (with minor wording drift vs. the
live adapter, from the re-quant rounding - the precision note in 4d).

## 5. Inspect the merged model's layers

`mlx-kquant inspect` prints the per-tensor codec recipe straight from the
checkpoint's `config.json` and safetensors headers - no GPU, no model build, not
even the `[tools]` extra (it works on a base install and returns instantly):

```sh
mlx-kquant inspect --model qwen3-0.6b-pirate-q5_k_m
```

```
model_type=qwen3  layers=28  quantized tensors=197
codec histogram: {'q5_k': 155, 'q6_k': 42}

module path                                  kind      codec  bits gsize       packed shape      logical shape
--------------------------------------------------------------------------------------------------------------
model.embed_tokens                           embedding q5_k      5   256      (151936, 704)     (151936, 1024)
model.layers.0.mlp.down_proj                 linear    q6_k      6   256       (1024, 2520)       (1024, 3072)
model.layers.0.mlp.gate_proj                 linear    q5_k      5   256        (3072, 704)       (3072, 1024)
model.layers.0.mlp.up_proj                   linear    q5_k      5   256        (3072, 704)       (3072, 1024)
model.layers.0.self_attn.k_proj              linear    q5_k      5   256        (1024, 704)       (1024, 1024)
model.layers.0.self_attn.o_proj              linear    q5_k      5   256       (1024, 1408)       (1024, 2048)
model.layers.0.self_attn.q_proj              linear    q5_k      5   256        (2048, 704)       (2048, 1024)
model.layers.0.self_attn.v_proj              linear    q6_k      6   256        (1024, 840)       (1024, 1024)
...
```

(The table lists all 197 quantized tensors; only the head + block 0 are shown
here. Add `--json` for a machine-readable dump.)

Reading the recipe off the inspection:

- The `q5_k_m` bumps land where expected - `v_proj` is `q6_k` (6-bit) on every
  layer, `down_proj` is `q6_k` on the `_M`-cadence layers (block 0 here) and
  `q5_k` elsewhere, and everything else is `q5_k` (5-bit). The `q6_k` count (42 =
  28 `v_proj` + 14 `down_proj`) matches the histogram.
- `packed shape` is the **uint8 wire bytes** actually on disk; `logical shape` is
  the matrix they decode to. The width follows from the codec geometry:
  `bytes = (in_features / 256) x bytes_per_block`. For `q5_k` (176 B/block):
  `q_proj` logical in=1024 -> `1024/256 x 176 = 704`. For `q6_k` (210 B/block):
  `down_proj` in=3072 -> `3072/256 x 210 = 2520`. (ok)
- There is no `lm_head` row: Qwen3-0.6B ties its output head to `embed_tokens`,
  so the head is the (q5_k) embedding. `inspect` reports what is on disk, so a
  tied model simply has one fewer quantized tensor.

## Recap

| stage | command | result |
|-------|---------|--------|
| quantize | `mlx-kquant quantize ... --preset q5_k_m` | 1.5 GB bf16 -> 415 MB kquant |
| run | `mlx-kquant run` | coherent, ~516 tok/s, 0.47 GB |
| train | `mlx-kquant lora --train` | loss 4.51 -> 2.18, ~30s |
| attach | `mlx-kquant run --adapter-path ...` | pirate at runtime, base untouched (highest fidelity) |
| fuse | `mlx-kquant fuse` | adapter baked in, stays kquant (re-quantizes; see 4d) |
| inspect | `mlx-kquant inspect` | per-tensor codec recipe |
