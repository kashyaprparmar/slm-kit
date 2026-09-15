# SLM Kit — Documentation

Welcome! SLM Kit is a **self-hosted web app** that lets you run the whole
small-language-model lifecycle on your own PC — no scripts, no cloud:

> **organize a project → prepare data → train or adapt a model → test and
> evaluate → deploy or quantize → publish to Hugging Face**

Everything is tuned for a **single-GPU workstation** (the reference machine is an
RTX 4060 with 8 GB VRAM and 16 GB RAM). Every default is chosen so it *fits*.

---

## Where to start

Engineering references: [Implementation plan](implementation-plan.md) and
[Database migrations](database-migrations.md).

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
| Check whether a model is supported | [Model compatibility](model-compatibility.md) |
| Serve a trained model locally | [Local deployment](deployment.md) |
| Fix a problem | [Troubleshooting](troubleshooting.md) |

## Page-by-page guides (the 11 screens of the app)

1. [Dashboard](pages/dashboard.md) — your workstation at a glance
2. [Dataset Manager](pages/dataset-manager.md) — upload, validate, preview data
3. [Projects](pages/projects.md) — group related runs into persistent workspaces
4. [Pretraining Studio](pages/pretraining-studio.md) — train a small GPT from zero
5. [Domain Adaptation Studio](pages/domain-adaptation-studio.md) — teach a model your domain
6. [Fine-Tuning Studio](pages/fine-tuning-studio.md) — LoRA / QLoRA / DoRA / full
7. [Testing & Eval Lab](pages/eval-lab.md) — chat with and score loadable models
8. [Model Registry](pages/model-registry.md) — inspect, merge, publish, import, quantize, deploy
9. [Run History](pages/run-history.md) — every run, searchable, re-runnable
10. [Model Serving](pages/model-serving.md) — Transformers, vLLM, and Ollama providers
11. [System & Diagnostics](pages/system-diagnostics.md) — health, dependencies, and request activity

## The three pillars

SLM Kit is built around three ways of training, each with its own studio:

| Pillar | What it means | Studio |
|---|---|---|
| **Pretraining** | Build a brand-new tiny model from raw text (its own tokenizer too) | Pretraining Studio |
| **Domain adaptation** | Take an existing model and continue training it on your domain's text | Domain Adaptation Studio |
| **Fine-tuning** | Teach an existing model to follow *your* instructions (Q&A pairs) | Fine-Tuning Studio |

## Key ideas (30 seconds)

- **One GPU workload at a time.** Training, evaluation, generation, merging,
  managed deployment, dedicated vLLM, and loaded Ollama models are checked
  before work starts. Queued training waits until the GPU is available.
- **Fit before you launch.** Before any run, the app predicts how much VRAM it
  will need and shows **fits / tight / won't fit**. Obvious OOMs are blocked.
- **Config as data.** Every run's full settings are stored as JSON. Any run can
  be re-run or exported with one click.
- **Local-first.** Hugging Face, llmfit, and LLM-judge are all optional.
  The core train/test loop works fully offline.
- **Capability-driven model support.** The Registry inspects model metadata and
  reports supported, experimental, conversion-required, or unavailable paths
  before weights are loaded. Unknown causal LMs can be tried through the generic
  Transformers path; unsupported architectures are rejected with a reason.
