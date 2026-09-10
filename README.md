# SLM Kit

A self-hosted platform for running the **entire small-language-model lifecycle**
— data prep → from-scratch pretraining → domain-adaptive continued pretraining →
fine-tuning → evaluation → quantization/export → Hugging Face publishing — through
a polished local web UI, on a single-GPU workstation.

Tuned and defaulted for an **RTX 4060 8GB / 16GB RAM / Ryzen 7000** box, not a
datacenter. Every default is chosen to run on first try on that hardware.

> **Status:** all 8 pages built, reviewed, and end-to-end tested. See [Roadmap](#roadmap).

## 📚 Documentation

Full guides live in [`docs/`](docs/README.md):

- **[Quickstart](docs/quickstart.md)** — Docker in a few commands, then your first fine-tune
- **[Docker](docs/docker.md)** — GPU / CPU / dev compose, volumes, env
- **[Requirements](docs/requirements.md)** · **[Installation](docs/installation.md)** · **[Configuration](docs/configuration.md)** · **[Commands](docs/commands.md)**
- **[Architecture](docs/architecture.md)** · **[API reference](docs/api-reference.md)** · **[Troubleshooting](docs/troubleshooting.md)**
- **Page guides:** [Dashboard](docs/pages/dashboard.md) · [Dataset Manager](docs/pages/dataset-manager.md) · [Pretraining](docs/pages/pretraining-studio.md) · [Domain Adaptation](docs/pages/domain-adaptation-studio.md) · [Fine-Tuning](docs/pages/fine-tuning-studio.md) · [Eval Lab](docs/pages/eval-lab.md) · [Model Registry](docs/pages/model-registry.md) · [Run History](docs/pages/run-history.md)

---

## Run with Docker (recommended)

The training stack (Unsloth, bitsandbytes, CUDA torch) lives in a **Linux
container**. On Windows you still open the UI at `http://localhost:5173`; GPU
passthrough comes from Docker Desktop + your NVIDIA driver.

```powershell
# from the project root — GPU / training (first build is several GB)
docker compose up --build -d

# UI only (no torch / no training)
# docker compose -f docker-compose.cpu.yml up --build -d
```

Open **http://localhost:5173**. Stop with `docker compose down`.

| Need | Command / file |
|------|----------------|
| GPU training | `docker compose up --build -d` |
| CPU / UI only | `docker compose -f docker-compose.cpu.yml up --build -d` |
| Hot reload | `docker compose -f docker-compose.dev.yml up --build` |
| Logs | `docker compose logs -f` |
| Wipe data | `docker compose down -v` |

Data persists in the `slmkit-data` volume (`/data/slmkit` in the backend).
Optional secrets: copy `backend/.env.example` → `backend/.env`.

Full walkthrough: [docs/quickstart.md](docs/quickstart.md) · [docs/docker.md](docs/docker.md).

---

## Requirements

| Component | Version |
|-----------|---------|
| Docker Desktop or Engine + Compose | recent |
| NVIDIA driver | Recent host driver; Docker Desktop **GPU** enabled |
| CUDA toolkit | *not needed on the host* — bundled in the GPU image |

Native (no-Docker) install still works if you want it — [Installation](docs/installation.md#3-without-docker-native-install).

---

## Configuration

Copy `backend/.env.example` → `backend/.env` (or export env vars). All are
optional; the core loop works fully offline without any of them:

| Var | Purpose |
|-----|---------|
| `SLMKIT_HF_TOKEN` | Hugging Face token for publish/import + private repos |
| `SLMKIT_JUDGE_API_KEY` | Optional LLM-as-judge key (Anthropic/OpenAI) |
| `SLMKIT_JUDGE_PROVIDER` | `anthropic` (default) or `openai` |
| `SLMKIT_LLAMACPP_DIR` | llama.cpp checkout, enables GGUF export |
| `SLMKIT_HOME` | Data directory (Docker default `/data/slmkit`) |

---

## llmfit (hardware/model-fit intelligence)

Optional but recommended. Install per its docs
(<https://github.com/AlexsJones/llmfit>), e.g. its install script, so the
`llmfit` binary is on `PATH`. SLM Kit shells out to `llmfit fit --json` /
`llmfit recommend --json`. **If it isn't installed, the app automatically falls
back to a built-in VRAM estimator** (weights + optimizer + activations + KV cache
+ ~15% overhead, per quantization), so all fit/advisor features still work.

---

## Frontend

Vite + React + TypeScript + Tailwind + Radix. Fonts (Inter + JetBrains Mono)
are bundled locally, so the UI runs fully offline. Docker serves the production
build via nginx (proxies `/api` and `/ws` to the backend). Native dev:

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173  (proxies /api and /ws to :8000)
```

---

## First-run walkthrough (the QLoRA vertical slice)

1. **Start the stack.** `docker compose up --build -d`. First boot creates
   `/data/slmkit` (volume `slmkit-data`) and auto-installs the bundled samples.
2. **Confirm hardware** — `GET /api/system/hardware` shows live VRAM/RAM/CPU/disk.
3. **List datasets** — `GET /api/datasets` includes the four samples:
   tiny story corpus (pretrain), finance domain corpus (domain adaptation),
   finance QA (instruction), finance QA eval.
4. **Preview + validate** — `GET /api/datasets/{id}/preview` returns sample rows,
   a token histogram, and a pass/fail validation report.
5. **Estimate fit** — `POST /api/runs/estimate` with a run config returns the
   predicted VRAM breakdown and a fits/tight/won't-fit verdict *before* launching.
6. **Launch a QLoRA fine-tune** — `POST /api/runs` with:
   ```json
   {
     "backend": "unsloth",
     "task": "finetune",
     "method": "qlora",
     "base_model": "unsloth/Qwen2.5-0.5B-Instruct",
     "dataset_id": 3,
     "output_name": "finance-qlora-demo"
   }
   ```
   The job is queued (one GPU job at a time), runs in an isolated subprocess, and
   streams live loss/tokens-per-sec/ETA over `ws://…/ws/runs/{run_id}`.
7. **Publish to HF** — `POST /api/registry/publish` auto-generates a model card
   from the run metadata and uploads the adapter to your Hub repo.

---

## Architecture

```
backend/app/
  main.py            FastAPI app + WebSocket endpoints
  config.py          Settings + data home layout (~/.slmkit or /data/slmkit)
  domain.py          Shared types: RunConfig, Method, MemoryEstimate, …
  db/                SQLModel registry (runs, datasets, checkpoints, artifacts)
  core/
    queue.py         Single-GPU job queue (queued→running→done/failed/cancelled)
    runner.py        Supervises each run as a subprocess (clean cancel frees VRAM)
    events.py        Newline-JSON training-event wire format
    hardware.py      pynvml + psutil telemetry over WebSocket
    ws.py            WebSocket broadcast hub
  backends/
    base.py          TrainingBackend protocol + registry
    unsloth_backend.py   LoRA/QLoRA/DoRA/full + continued pretraining
    scratch_backend.py   From-scratch GPT + custom tokenizer (Pillar 1)
  integrations/
    estimator.py     Fallback VRAM math
    llmfit.py        llmfit shell-out with estimator fallback
    hf_hub.py        Model metadata + publish/import
  datasets/          Validation, token estimation, bundled samples
  train_entry/run.py Subprocess entrypoint (heavy GPU imports live here)
  api/               REST routers: system, datasets, runs, registry, advisor
```

**Key design decisions** (confirmed up front):
- **Subprocess per GPU job** — the only reliable way to free VRAM on cancel.
- **SQLModel** for the local metadata/lineage registry; **HF Hub** is the source
  of truth for published model files.
- **Config-as-data** — every run stores its full `RunConfig` JSON, so runs are
  reproducible and cloneable.
- **Pluggable `TrainingBackend`** — Axolotl/LlamaFactory slot in later as pure
  `export_config` + `run` implementations with no core refactor.

---

## 8GB VRAM constraints (read this)

Defaults are deliberately conservative:
- **QLoRA (4-bit) is the default** fine-tuning method. Full fine-tuning is only
  viable for very small models and is warned/blocked by the fit guard.
- Default `max_seq_length=1024`, `per_device_batch_size=2`,
  `gradient_accumulation=4` (effective batch 8), `optim=adamw_8bit`, gradient
  checkpointing on.
- The pre-launch fit guard **blocks obvious-OOM configs** and warns on tight ones.
- 7B models are QLoRA-trainable but *tight*; 0.5B–3B iterate comfortably.

---

## Roadmap

- [x] Backend foundation: FastAPI, job queue, WS streaming, SQLite, llmfit + fallback, HF client
- [x] QLoRA vertical slice (Unsloth backend) + from-scratch pretraining backend
- [x] Bundled sample datasets (3 pillars) + validation
- [x] Frontend foundation: design system, app shell + live resource strip, Dashboard, Dataset Manager
- [x] Frontend studios: Fine-Tuning, Domain Adaptation, Pretraining (method/arch selectors, Model Advisor, live fit guard, live loss/logs monitor + cancel)
- [x] Run History: searchable table, live + historical curves, re-run from stored config, config export
- [x] Model Registry: local artifacts + HF repos, publish (auto model card), import, GGUF export (via llama.cpp)
- [x] Eval Lab: streaming playground + eval harness (EM / token-F1 / ROUGE / BLEU / perplexity) + optional LLM-as-judge + side-by-side comparison
- [ ] Axolotl / LlamaFactory backends (interface is ready — `export_config` + `run`)

**All 8 core pages are built.** 🎉

### GGUF export

The Model Registry can export a finished run to GGUF via **llama.cpp**. Clone and
build <https://github.com/ggerganov/llama.cpp>, then set
`SLMKIT_LLAMACPP_DIR` to that checkout (it must contain
`convert_hf_to_gguf.py` and a built `llama-quantize`). If it isn't set, the GGUF
action reports exactly what to install. Conversion expects a full/merged model
dir; LoRA/QLoRA adapters need merging first (a documented follow-up).
- [ ] Eval harness (EM/ROUGE/BLEU/perplexity + optional LLM-as-judge + side-by-side)
- [ ] GGUF export via llama.cpp
- [ ] Axolotl / LlamaFactory backends
```
