# Pretraining Studio (Pillar 1)

Train a **brand-new tiny GPT from scratch** — including its own tokenizer —
on a raw text corpus. This is the "build a model from zero" pillar.

> Reality check: from-scratch models need **lots** of text and time. With a
> small corpus you'll get a *toy* model that produces gibberish-ish text — great
> for learning and smoke-testing your GPU, not for real use. For useful models,
> use the Fine-Tuning or Domain Adaptation studios instead.

## What you configure

### 1. Architecture preset
Pick a starting point, then tweak:

| Preset | Params | Good for |
|---|---|---|
| **Nano** | ~2M | fastest smoke test |
| **Tiny** | ~10M | a sensible starter |
| **Small** | ~30M | needs more tokens/time |

### 2. Model architecture (all editable)
- **Layers**, **Heads**, **Embedding dim**, **Context length**, **Vocab size**,
  **Dropout**.
- A live **≈ parameter count** updates as you change them.
- Rule enforced for you: **embedding dim must be divisible by heads** (you'll
  see a red note if not).

### 3. Corpus & training
- **Dataset:** pick a *pretraining* or *domain* corpus (raw `.txt`).
- **Max steps**, **Batch size**, **Learning rate**.
- Advanced: checkpoint/log cadence.
- **Output name** for the saved model.

## The fit panel & launch

The right panel shows the **predicted VRAM footprint** and a **fits / tight /
won't fit** verdict, computed from your architecture and batch size. Launch is
disabled until a corpus is selected, the head/embedding rule passes, and the
config fits.

Click **Launch pretraining** to start. The live monitor shows:
- **Loss curve**, **tokens/sec**, **ETA**
- **Streaming logs** (tokenizer training, model build, steps)
- **Sample generations** — every checkpoint, the model generates a snippet so
  you can watch it (slowly) learn.

## What it produces

A folder under the run's directory containing `model.pt`, `arch.json`, and the
trained tokenizer. It's registered in Run History and can be published from the
Model Registry.

## 8 GB tips

- Start with **Nano** or **Tiny** to confirm your GPU trains at all.
- Bigger `context length` and `embedding dim` cost the most VRAM — grow them
  last, and watch the fit panel.
- If the corpus is smaller than your context length, you'll get a clear error —
  use a bigger corpus or a shorter context.
