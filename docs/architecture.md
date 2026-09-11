# Architecture

How SLM Kit is put together — for the curious, and for anyone extending it.

Recommended deploy is **Docker Compose**: nginx (or Vite in dev) on port 5173
proxies `/api` and `/ws` to the FastAPI backend on port 8000. See [Docker](docker.md).

## Big picture

```
┌──────────────────────────────────────────────────────────────┐
│  Browser (React)                     http://localhost:5173    │
│  workflow pages · live charts · resource strip · logger dock   │
└───────────────┬──────────────────────────────┬───────────────┘
        REST /api│                       WS /ws/*│
┌───────────────▼──────────────────────────────▼───────────────┐
│  FastAPI backend  (:8000)                                     │
│                                                               │
│  API routers ─ system · datasets · runs · registry · advisor  │
│                · eval · serving                                │
│                                                               │
│  Core ─ GPU lease + JobQueue ─ Runner (subprocess)             │
│         Eval/Deployment managers ─ activity tracing ─ WSHub    │
│                                                               │
│  Integrations ─ llmfit(+estimator) · hf_hub · gguf · judge    │
│  Backends ─ TrainingBackend: unsloth · scratch (Axolotl/…)    │
│  DB ─ SQLModel/SQLite (runs, datasets, artifacts, evals)      │
└───────────────┬───────────────────────────────────────────────┘
                │ spawns
┌───────────────▼───────────────────────────────────────────────┐
│  Training / eval / generation SUBPROCESSES                     │
│  torch · unsloth · transformers loaded here (never in the API) │
│  emit newline-JSON events on stdout → parent streams to WS     │
└───────────────────────────────────────────────────────────────┘
```

## Why a subprocess per job

Killing a child process returns **all** its VRAM to the OS immediately and
completely. That is the only reliable way to honor "clean cancel frees VRAM."
It also means a crash in torch/unsloth can never take down the API. The child
speaks a tiny newline-delimited JSON protocol on stdout; the parent parses each
line, saves what matters, and rebroadcasts it to the browser over WebSocket.

Heavy imports (`torch`, `unsloth`, `transformers`) live **inside** the
subprocess entrypoints, so the API process itself starts instantly and runs on
machines with no CUDA.

## One model reference, three runtime formats

`model_refs.py` gives each completed run a stable `run:<id>` reference and
classifies its output as a full Transformers model, PEFT adapter, or SLM Kit
scratch checkpoint. `train_entry/model_runtime.py` is the shared heavy loader
used by generation, evaluation, and deployment, so those product surfaces do
not implement different compatibility rules.

The managed deployment is another isolated subprocess
(`train_entry/serve.py`). It exposes OpenAI-compatible chat/completions on port
8802. `core/deployment.py`, the training queue, and EvalManager coordinate
ownership so only one long-lived GPU workload is active at a time.

## Single-GPU safety

- **`JobQueue`** ([core/queue.py](../backend/app/core/queue.py)) runs exactly one
  training job at a time: `queued → running → done | failed | cancelled`.
- **`EvalManager`** ([core/eval_manager.py](../backend/app/core/eval_manager.py))
  runs playground generation and eval jobs, and **refuses to start** (HTTP 409)
  while a training job holds the GPU — so nothing double-books VRAM.
- The pre-launch **fit guard** blocks configs that would obviously OOM.

## Request → training, step by step

1. UI POSTs a `RunConfig` to `/api/runs`.
2. The backend validates it and runs the fit guard (`estimate_footprint` +
   `validate_config` on the chosen backend). If it won't fit → `422`.
3. A `Run` row is stored (status `queued`, full config as JSON) and the id is
   put on the queue.
4. The queue worker picks it up, writes `config.json` into the run folder, and
   spawns `python -m app.train_entry.run <id>`.
5. The subprocess loads the backend, trains, and prints JSON events
   (log / metric / checkpoint / sample / status).
6. The runner parses each event → saves metrics to `metrics.jsonl`, records
   checkpoints, and publishes to `ws/runs/<id>`.
7. On exit: status becomes `done` or `failed` (with the real error captured).
   Cancel writes a `STOP` file (graceful) then terminates the process (hard),
   freeing VRAM.

## The pluggable backend interface

Every training engine implements `TrainingBackend`
([backends/base.py](../backend/app/backends/base.py)):

```python
class TrainingBackend(Protocol):
    name: str
    supported_tasks: set[TaskType]
    supported_methods: set[Method]
    def validate_config(cfg, hw) -> ValidationReport   # cheap, API-side
    def estimate_footprint(cfg, hw) -> MemoryEstimate   # cheap, API-side
    def export_config(cfg) -> ExportedConfig            # cheap, API-side
    def run(cfg, ctx) -> Iterator[TrainingEvent]        # heavy, subprocess-side
```

Three are built in:
- **`unsloth`** — LoRA/QLoRA/DoRA/full + continued pretraining (via TRL's SFTTrainer).
- **`transformers`** — broad Transformers + TRL + PEFT compatibility fallback.
- **`scratch`** — from-scratch GPT + custom BPE tokenizer, pure PyTorch.

Adding **Axolotl** or **LlamaFactory** later means writing one class with these
methods — no changes to the queue, runner, API, or UI. The metadata methods must
stay import-light (no torch at module load) so the API can call them.

## Config as data

A `RunConfig` (see [domain.py](../backend/app/domain.py)) fully describes a run:
backend, task, method, base model, dataset, LoRA params, optimizer, training
schedule, and (for pretraining) the architecture. The effective config also
captures the model revision, dataset SHA-256 fingerprint, seed, platform and
installed training-library versions. It is stored on the `Run` row, which is
why any run can be **re-run**, cloned, compared, or exported.

## Data model (SQLite via SQLModel)

| Table | Holds |
|---|---|
| `Dataset` | uploaded/sample files + validation + token estimate |
| `Run` | one training run: config, estimate, metrics, status, lineage |
| `Checkpoint` | checkpoints emitted during a run |
| `ModelArtifact` | published models + GGUF exports (local ↔ HF) |
| `EvalResult` | eval-harness scores + sample comparisons |

The DB is the **metadata/lineage** source of truth; the **model files**
themselves live on disk (`~/.slmkit/models`, run `output/` folders) or on
Hugging Face once published.

## Frontend structure

```
frontend/src/
├── pages/          one file per screen (Dashboard, Datasets, Finetune, …)
├── components/
│   ├── ui/         design-system primitives (Button, Card, Tabs, Dialog…)
│   ├── studio/     shared training-studio building blocks
│   ├── eval/       Playground + EvalHarness
│   ├── RunMonitor  live+historical loss/logs/cancel (reused everywhere)
│   └── ResourceStrip, FitIndicator, hardware-context, theme
└── lib/            api client, WebSocket hook, types, formatters
```

The **"slate + electric violet"** dark-first design system is a set of CSS
variables in `src/index.css`, wired through `tailwind.config.ts`. Fonts (Inter +
JetBrains Mono) are bundled locally, so the UI works fully offline.

## WebSocket topics

| Topic | Carries |
|---|---|
| `/ws/system` | live hardware telemetry + queue state |
| `/ws/runs/{id}` | a run's log / metric / checkpoint / sample / status events |
| `/ws/eval/{id}` | an eval job's progress + result |
| `/ws/gen/{id}` | playground generation tokens |
