# SLM Kit — Documentation

Welcome! SLM Kit is a **self-hosted web app** that lets you run the whole
small-language-model lifecycle on your own PC — no scripts, no cloud:

> **prepare data → pretrain from scratch → adapt to a domain → fine-tune →
> test & evaluate → quantize to GGUF → publish to Hugging Face**

Everything is tuned for a **single-GPU workstation** (the reference machine is an
RTX 4060 with 8 GB VRAM and 16 GB RAM). Every default is chosen so it *fits*.

---

## Where to start

| I want to… | Read this |
|---|---|
| Get it running in 10 minutes (Docker) | [Quickstart](quickstart.md) |
| Docker compose files, volumes, GPU vs CPU | [Docker](docker.md) |
| Check my PC can run it | [Requirements](requirements.md) |
| Do a full, careful install (+ optional tools) | [Installation](installation.md) |
| See every setting / environment variable | [Configuration](configuration.md) |
| Copy-paste common commands | [Commands cheat-sheet](commands.md) |
| Understand how it works inside | [Architecture](architecture.md) |
| Call the backend directly | [API reference](api-reference.md) |
| Fix a problem | [Troubleshooting](troubleshooting.md) |

## Page-by-page guides (the 8 screens of the app)

1. [Dashboard](pages/dashboard.md) — your workstation at a glance
2. [Dataset Manager](pages/dataset-manager.md) — upload, validate, preview data
3. [Pretraining Studio](pages/pretraining-studio.md) — train a small GPT from zero
4. [Domain Adaptation Studio](pages/domain-adaptation-studio.md) — teach a model your domain
5. [Fine-Tuning Studio](pages/fine-tuning-studio.md) — LoRA / QLoRA / DoRA / full
6. [Testing & Eval Lab](pages/eval-lab.md) — chat with and score any model
7. [Model Registry](pages/model-registry.md) — publish, import, quantize
8. [Run History](pages/run-history.md) — every run, searchable, re-runnable

## The three pillars

SLM Kit is built around three ways of training, each with its own studio:

| Pillar | What it means | Studio |
|---|---|---|
| **Pretraining** | Build a brand-new tiny model from raw text (its own tokenizer too) | Pretraining Studio |
| **Domain adaptation** | Take an existing model and continue training it on your domain's text | Domain Adaptation Studio |
| **Fine-tuning** | Teach an existing model to follow *your* instructions (Q&A pairs) | Fine-Tuning Studio |

## Key ideas (30 seconds)

- **One GPU, one job.** Training jobs go through a queue — exactly one runs at a
  time, so nothing fights over VRAM. Cancel always frees the GPU.
- **Fit before you launch.** Before any run, the app predicts how much VRAM it
  will need and shows **fits / tight / won't fit**. Obvious OOMs are blocked.
- **Config as data.** Every run's full settings are stored as JSON. Any run can
  be re-run or exported with one click.
- **Local-first.** Hugging Face, llmfit, and LLM-judge are all optional.
  The core train/test loop works fully offline.
- **Works on any model.** The Eval Lab and Registry work with models you trained
  here *or* any model straight from Hugging Face.
