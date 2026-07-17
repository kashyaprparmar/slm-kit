# Domain Adaptation Studio (Pillar 2)

Take an **existing base model** and **continue training it on your domain's raw
text** (legal, medical, finance, your codebase, etc.) so it "speaks" your
domain. The result becomes a **reusable base** you can then fine-tune.

This studio is the same flow as the [Fine-Tuning Studio](fine-tuning-studio.md),
with two differences:
- **Task** is *continued pretraining* (learning from raw text, not Q&A pairs).
- **Dataset** is a **domain corpus** (`.txt`), not instruction pairs.

## When to use it

- You have a **lot of domain text** but not instruction/Q&A pairs.
- You want a base that already knows your domain's vocabulary and style, which
  you'll then instruction-tune separately.

If you only have Q&A pairs, skip this and go straight to fine-tuning.

## What you configure

- **Method:** LoRA-family (QLoRA is the 8 GB default) or full.
- **Base model:** a curated 8 GB-friendly model, or paste any HF repo id / local
  path. The Model Advisor can pick one for you.
- **Dataset:** a domain corpus.
- **Hyperparameters:** epochs, learning rate, batch size, gradient accumulation,
  sequence length, plus an advanced section (LoRA rank/alpha/dropout, scheduler,
  optimizer, checkpoint cadence).

## Fit, launch, monitor

Identical to the other studios: the right panel predicts VRAM and shows
**fits / tight / won't fit**; launch is blocked for OOM configs. The live
monitor streams loss, tokens/sec, ETA, logs, and lets you cancel.

## Reusing the output

When it finishes, the adapted model is saved and listed in Run History and the
Model Registry. To fine-tune it next:
- Publish it to Hugging Face and use that repo id as the base, **or**
- Point the Fine-Tuning Studio's "custom base model" field at its local output
  folder.

## Tips

- Domain adaptation wants **corpus scale** — aim for ~100k+ tokens; the Dataset
  Manager warns you when it's thin.
- Keep the learning rate modest; you're nudging an existing model, not
  retraining it from scratch.
