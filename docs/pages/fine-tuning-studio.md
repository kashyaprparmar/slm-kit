# Fine-Tuning Studio (Pillar 3)

Teach an existing model to follow **your** instructions using Q&A / chat pairs.
This is the most common workflow and the one to start with.

## Choosing a method

The method selector shows four options with VRAM guidance:

| Method | What it is | On 8 GB VRAM |
|---|---|---|
| **QLoRA** ⭐ | 4-bit quantized base + small LoRA adapters | **The default.** Lightest. |
| **LoRA** | fp16/bf16 base + LoRA adapters | Higher-quality base, more VRAM |
| **DoRA** | weight-decomposed LoRA (Unsloth) | Often higher quality, a bit slower |
| **Full** | trains every weight | Only tiny models fit — guarded |

Not sure? Keep **QLoRA**. It's the best default for an 8 GB card.

## Picking a base model

- **Curated dropdown:** 8 GB-friendly models from 0.36B up to 7B, labeled with
  size.
- **Custom:** choose "Custom…" and paste **any Hugging Face repo id**
  (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) **or a local checkpoint path**. You are not
  limited to the dropdown.

### Let the Model Advisor choose (optional)
In the right column, pick a priority — **Fastest iteration**, **Balanced**, or
**Best quality** — and click **Recommend**. You get a ranked list of base models
with reasoning (size, quality tier, context length, predicted VRAM, fit). Click
**Use this model** to fill it in.

## Dataset & hyperparameters

- **Dataset:** pick an *instruction* dataset (must pass validation).
- **Hyperparameters** with 8 GB-safe defaults:
  - Epochs, Max steps (overrides epochs), Learning rate
  - Batch size (2), Gradient accumulation (4) → effective batch 8
  - Max sequence length (1024)
- **Advanced** (expandable): LoRA rank/alpha/dropout, warmup ratio, LR scheduler,
  optimizer (`adamw_8bit` default), save/log cadence.

## The fit panel — read it before launching

The **Predicted footprint** panel breaks down predicted peak VRAM
(weights + optimizer + activations + KV cache + overhead) versus your budget,
with a **fits / tight / won't fit** verdict and any validation issues.

**Launch is enabled only when:** a base model is set, a dataset is selected,
and the config passes validation (fits). Obvious-OOM configs are blocked.

## Launch & monitor

Click **Launch run**. The live monitor appears at the top:
- **Loss curve** (smooth, live), **tokens/sec**, **ETA**
- **Progress bar** (step X / total)
- **Streaming logs**
- **Cancel** — stops cleanly and frees VRAM

Your first launch downloads the base model (cached afterward). When it's done,
find it in **Run History** (full curve + config, one-click re-run) and try it in
the **Eval Lab**.

## 8 GB recipes

| Goal | Suggested config |
|---|---|
| Fastest first result | QLoRA, Qwen2.5-0.5B, seq 512, max_steps 60 |
| Balanced quality | QLoRA, Llama-3.2-1B or Qwen2.5-1.5B, seq 1024, 1 epoch |
| Push the limit | QLoRA, 7B, seq **≤ 1024**, batch **1**, grad-accum 8 (expect "tight") |

## Common questions

- **QLoRA vs LoRA?** QLoRA quantizes the base to 4-bit → far less VRAM, tiny
  quality trade-off. Default to QLoRA on 8 GB.
- **Why is Full disabled/warned?** Full fine-tuning holds optimizer state for
  *every* weight — only viable for very small models on 8 GB.
- **It says "tight" — is that OK?** Yes, but close other GPU apps (browser,
  games) first, or it may OOM.
