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
│  API routers ─ system · datasets · projects · runs · registry │
│                · advisor · eval · serving                       │
│                                                               │
│  Core ─ GPU lease + JobQueue ─ Runner · bounded CPU job gate   │
│         Eval/Deployment managers ─ activity tracing ─ WSHub    │
│                                                               │
│  Integrations ─ capability registry · HF · llmfit · GGUF      │
│  Backends ─ TrainingBackend: unsloth · transformers · scratch │
│  DB ─ SQLModel/SQLite (projects, runs, data, artifacts, evals)│
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

Tokenizer profiling is also isolated in a subprocess: it imports Transformers
and resolves a tokenizer without loading model weights. Dataset preparation and
quality scans use a small bounded CPU-job gate so expensive local work does not
exhaust request workers.

## One model reference, three runtime formats

`model_refs.py` gives each completed run a stable `run:<id>` reference and
classifies its output as a full Transformers model, PEFT adapter, or SLM Kit
scratch checkpoint. `train_entry/model_runtime.py` is the shared heavy loader
used by generation, evaluation, and deployment, so those product surfaces do
not implement different compatibility rules.

The managed Transformers deployment is another isolated subprocess
(`train_entry/serve.py`). It exposes OpenAI-compatible chat/completions on port
8802. `core/deployment.py`, the training queue, and EvalManager coordinate
ownership so only one long-lived GPU workload is active at a time.

`serving/providers.py` normalizes managed Transformers, external Ollama, and a
dedicated vLLM service. Provider health checks run concurrently and fail
independently. External vLLM or Ollama usage is also checked at launch time and
again when a queued training job reaches the front, closing the race where an
external model could load after enqueue but before GPU acquisition.

## Model capability registry

`models/capabilities.py` maps lightweight model metadata to explicit support
states for training methods, backends, inference providers, quantizers,
exports, precision, and distributed execution. The same rules are consumed by
the studios, advisor, run validation, registry inspection, and serving flows.
Hugging Face metadata caches include the requested revision.

This registry distinguishes "supported" from "experimental" and
"requires conversion". Runtime loaders are still authoritative because gated
repositories, custom code, and locally installed library versions can only be
verified in the worker environment. See [Model compatibility](model-compatibility.md).

## Single-GPU safety

- **`JobQueue`** ([core/queue.py](../backend/app/core/queue.py)) runs exactly one
  training job at a time: `queued → running → done | failed | cancelled`.
- **`EvalManager`** ([core/eval_manager.py](../backend/app/core/eval_manager.py))
  runs playground generation and eval jobs, and **refuses to start** (HTTP 409)
  while a training job holds the GPU — so nothing double-books VRAM.
- Merge and deployment operations use the same lease, while vLLM and unmanaged
  Ollama processes are treated as external owners.
- The pre-launch **fit guard** blocks configs that would obviously OOM.

## Request → training, step by step

1. UI POSTs a `RunConfig` to `/api/runs`.
2. The backend validates it and runs the fit guard (`estimate_footprint` +
   `validate_config` on the chosen backend). If it won't fit → `422`.
3. A `Run` row is stored (status `queued`, full config as JSON). If a project is
   active, the run link is committed in the same transaction. The id is then
   put on the queue.
4. The queue worker picks it up, writes `config.json` into the run folder, and
   spawns `python -m app.train_entry.run <id>`.
5. The subprocess loads the backend, trains, and prints JSON events. Existing
   workers emit log/metric/checkpoint/sample/status; the shared protocol also
   accepts progress/resource/artifact/warning/profile/error envelopes.
6. The runner parses each event → saves metrics to `metrics.jsonl`, records
   checkpoints, and publishes to `ws/runs/<id>`.
7. On exit: status becomes `done` or `failed` (with the real error captured).
   Cancel writes a `STOP` file (graceful) then terminates the process (hard),
   freeing VRAM.

Each run directory also receives an atomically replaced `run.json` manifest.
It records model revision, dataset fingerprint, effective configuration,
hardware, checkpoints, metrics, outputs, and timestamps. Failed runs additionally
receive normalized `failure.json` and human-readable `failure.md` reports with
suggested corrections. These observability files are best-effort: a disk/report
error cannot change a successfully completed ML workload into a failed run.

## The pluggable backend interface

Every training engine implements `TrainingBackend`
([backends/base.py](../backend/app/backends/base.py)):

```python
class TrainingBackend(Protocol):
    name: str
    def capabilities() -> TrainingBackendCapabilities     # API/UI validation source
    @property
    def supported_tasks() -> set[TaskType]                # derived compatibility view
    @property
    def supported_methods() -> set[Method]                # derived compatibility view
    def validate_config(cfg, hw) -> ValidationReport   # cheap, API-side
    def estimate_footprint(cfg, hw) -> MemoryEstimate   # cheap, API-side
    def export_config(cfg) -> ExportedConfig            # cheap, API-side
    def run(cfg, ctx) -> Iterator[TrainingEvent]        # heavy, subprocess-side
```

Three are built in:
- **`unsloth`** — LoRA/QLoRA/DoRA/full + continued pretraining (via TRL's SFTTrainer).
- **`transformers`** — broad Transformers + TRL + PEFT compatibility fallback.
- **`scratch`** — from-scratch GPT + custom BPE tokenizer, pure PyTorch.

Adding an optional **LLaMA-Factory** backend later means implementing this
contract and registering it. The queue and runner remain engine-independent,
and the API/UI consume its descriptor. Metadata methods must stay import-light
(no torch at module load) so the API can call them. `RegisteredBackendSelector`
currently honors an explicit key; automatic fallback is deliberately deferred.

## Config as data

A versioned `RunConfig` (see [domain.py](../backend/app/domain.py)) fully describes a run:
backend, task, method, base model, dataset, LoRA params, optimizer, training
schedule, and (for pretraining) the architecture. The effective config also
captures the model revision, dataset SHA-256 fingerprint, seed, platform and
installed training-library versions. Persisted `TaskType` remains compatible
with old runs; `TrainingStage` supplies the execution vocabulary for later
alignment work. It is stored on the `Run` row, which is
why any run can be **re-run**, cloned, compared, or exported.

## Data model (SQLite via SQLModel)

Schema changes now run through packaged Alembic revisions at startup, with
pre-upgrade backups, transactional DDL, foreign-key enforcement and WAL. The
initial schema retains the existing entities below. See
[Database migrations](database-migrations.md) for adoption and integrity checks.

| Table | Holds |
|---|---|
| `Project` | persistent workspace name, notes, and UI/workflow state |
| `ProjectRun` | many-to-many project/run membership |
| `Dataset` | backward-compatible catalog handle for an uploaded/sample/prepared file |
| `DatasetVersion` | immutable file path, SHA-256, canonical schema, split, and size/count metadata |
| `DatasetRecipe` | immutable source version plus canonical transformation/split config |
| `DatasetSplit` | recipe-linked train/validation/test version IDs and deterministic seed/fractions |
| `TokenizerArtifact` | model/revision-resolved tokenizer identity and fingerprint |
| `DatasetProfile` | tokenizer or quality result cached by dataset/tokenizer/config fingerprints |
| `Run` | one training run: config, estimate, metrics, status, lineage |
| `Checkpoint` | checkpoints emitted during a run |
| `ModelArtifact` | published models + GGUF exports (local ↔ HF) |
| `EvalResult` | eval-harness scores + sample comparisons |

The DB is the **metadata/lineage** source of truth; the **model files**
themselves live on disk (`~/.slmkit/models`, run `output/` folders) or on
Hugging Face once published.

Dataset transformation writes canonical rows to a temporary SQLite spool,
deduplicates and orders them deterministically, writes `.part` split files,
flushes them, and atomically renames each output. Database lineage is committed
only after every split validates; on failure all staged/published files from the
attempt are removed. Training and eval consume the same canonical adapter rather
than independently guessing row shapes.

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
| `/ws/gguf/{artifact_id}` | GGUF export logs and status |
