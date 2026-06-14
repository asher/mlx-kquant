# Imatrix calibration

An **importance matrix** (imatrix) steers the K-quant encoder: instead of rounding
every input channel of a weight equally, it preserves the channels that matter most
for a calibration corpus. It is a per-tensor, per-input-feature importance vector,
and it costs nothing at inference time - it only changes how the bytes are chosen at
encode time.

## When it helps

- **Lower bit widths benefit most.** At q2_k / q3_k / q4_k the rounding budget is
  tight, so steering it is worth the most; by q6_k the budget is generous and the
  delta shrinks. It still helps across the range - bartowski / Unsloth ship imatrix
  quants at every level.
- **Superblock K-quants only.** The importance weighting applies to the five
  superblock codecs (q2_k, q3_k, q4_k, q5_k, q6_k). It is a **no-op** on the five
  `wpb=32` block codecs (q8_0 / q4_0 / q4_1 / q5_0 / q5_1), which have no
  importance-weighted rounding path - passing an imatrix there is silently ignored.
- It improves the model's **general quality** - measure it with perplexity or KLD
  against the float model. It is *not* a way to protect a specific fine-tune signal:
  a LoRA persona that the bit width is too coarse to hold will still wash out on
  merge regardless of the imatrix (see the re-quant note in the
  [walkthrough](walkthrough.md)).

## 1. Build an imatrix

`calibrate-imatrix` runs the float model forward over a plain-text corpus and
captures `sum(x^2) / n_tokens` per input feature on every `nn.Linear`:

```sh
# a standard general-purpose calibration corpus (bartowski's, ~1.7 MB)
curl -sL https://gist.githubusercontent.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d/raw/calibration_datav5.txt -o calibration.txt

mlx-kquant calibrate-imatrix \
    --model Qwen/Qwen3-0.6B \
    --corpus calibration.txt \
    --output qwen3-0.6b.imatrix.dat
```

```
[mlx-kquant] hooked 196 Linear modules
[mlx-kquant] 50 chunks x 512 tokens
[mlx-kquant] wrote qwen3-0.6b.imatrix.dat (196 entries)
```

Calibrate the **float** model - the importance is a property of the weights, not of
any one quant, so one imatrix serves every codec you might encode the model with.
`--ctx` and `--chunks` trade calibration coverage for time.

### Corpus choice

The plain bartowski corpus is a sensible default for any model, instruct or base:
it is broad general text, and the importance vector is dominated by bulk content,
not by a handful of structural tokens. A chat-template-formatted corpus (one that
wraps turns in `<|im_start|>` ... tags) is a reasonable alternative for a chat
model - the special-token strings still map to their vocab ids during tokenization,
so the stats become chat-aware - but the effect is second-order. **Pick one; do not
merge them** (merging just dilutes whichever set you cared about).

### Format

The output is the llama.cpp imatrix **`.dat`** format: a count plus one float32
importance vector per tensor, keyed by HF module path. `calibrate-imatrix` only
hooks `nn.Linear`, so embeddings are not covered - that is why a tied-embedding
model reports 196 of 197 tensors in the next step.

## 2. Quantize with the imatrix

```sh
mlx-kquant quantize \
    --model Qwen/Qwen3-0.6B --preset q4_k_m \
    --imatrix qwen3-0.6b.imatrix.dat \
    --mlx-path qwen3-0.6b-q4_k_m
```

```
[mlx-kquant] imatrix coverage: 196/197 tensors
[mlx-kquant] encoded 197 tensors
[mlx-kquant] wrote qwen3-0.6b-q4_k_m (197 tensors quantized)
```

The coverage line reports how many quantizable tensors the imatrix resolved against
(196 / 197 here: every Linear, but not the tied embedding). Tensors with no imatrix
entry are encoded unsteered.

## 3. Preserve calibration through a merge

When `mlx-kquant fuse` keeps the kquant format it re-encodes each adapted weight
(see [lora.md](lora.md)). That re-encode is **unsteered by default**, so a base
built with an imatrix loses the calibration on the tensors a LoRA adapter touched.
Pass the same imatrix to keep it:

```sh
mlx-kquant fuse \
    --model qwen3-0.6b-q4_k_m \
    --adapter-path pirate-adapter \
    --imatrix qwen3-0.6b.imatrix.dat \
    --save-path qwen3-0.6b-fused
```

```
[mlx-kquant] imatrix coverage: 56/56 fused tensors
[mlx-kquant] fused 56 adapter layers -> qwen3-0.6b-fused (kquant)
```

Only the adapted (re-encoded) tensors consult the imatrix; the untouched base
tensors keep their original bytes. `--imatrix` is ignored under `--dequantize`
(there is no re-encode to steer). This preserves the base's *general* calibration
through the merge - it does not recover a fine-tune that the bit width is too coarse
to represent.
