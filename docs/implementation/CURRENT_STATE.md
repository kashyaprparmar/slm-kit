# Current state of SLM Kit

Audit date: 2026-09-15. Implementation state updated: 2026-09-16. Reference commit: `5d64b78b6dc5289ef419d22ab11cd629cdba6337`. **Source of truth: the actual working tree, including substantial pre-existing modified and untracked source.** Phase A and Phase B remain partially complete. The shared Phase C preference architecture is partially complete; installed-runtime objective evidence remains open.

## Scope and evidence

The supplied implementation specification has 92 numbered sections. [GAP_ANALYSIS.md](GAP_ANALYSIS.md) classifies each; [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) gives task dependencies, files, impacts and acceptance tests. Earlier completion statements were not accepted as evidence. In particular, `docs/implementation-plan.md` mixes historical findings with completed checklist items.

Inspected repository source groups: backend APIs/domain/backends/core/datasets/database/migrations/models/integrations/evaluation/worker entries; frontend routes/shared components/hooks/types/config serialization; dependency manifests, Docker/Compose/nginx, tests and documentation. Generated `backend/unsloth_compiled_cache` files, `.venv`, node_modules and dist are runtime/build artifacts, not proof that SLM Kit exposes the algorithms they contain. The foundation review changed shared contracts and their direct consumers only; it did not add a training objective, dataset format, model loader, serving engine, or database entity.

Evidence levels: direct source inspection, unit/contract tests, isolated application/DB smoke checks. No real GPU training, model download, model-serving launch or container image build was performed. A support label in code is not equivalent to verified runtime compatibility.

## Backend architecture and contracts

FastAPI `backend/app/main.py` composes system, datasets, Data Lab, projects, runs, registry, advisor, evaluation and serving routers. Lifespan initializes migrations/sample data, hardware polling, the FIFO queue and orphan recovery; shutdown stops queue, registry background jobs, managed deployment, eval jobs and polling. One control-plane process is assumed.

`domain.py` defines Pydantic RunConfig, task/method enums, tokenizer/LoRA/optimizer/train/scratch sections, hardware, estimates and validation. RunConfig JSON is stored on Run and written for subprocess consumption. The API normalizes method-specific quantization defaults. Important future controls have no typed execution contract; `extra` should not become an unrestricted substitute for validation. Many API responses remain untyped dictionaries, and frontend types are maintained manually.

Primary API families already present:

- `/api/datasets`: upload, HF import, preview, schema, prepare, delete, samples.
- `/api/dataset-versions`, `/api/dataset-recipes`, `/api/tokenizer-artifacts`: versions, replay, cached diagnostic profiles.
- `/api/projects`: persistent project state and run membership.
- `/api/runs`: backends, estimates, create/list/detail/cancel/delete, JSON export, metrics/logs and rerun.
- `/api/registry`: model options/local/HF, import/publish/inspect/preflight, deployment, merge and quantize.
- `/api/eval`: generate, evaluation runs/results, status/cancel/logs.
- `/api/serving`: providers, provider test, Ollama load/unload/import.
- `/api/system`: hardware/status/diagnostics/database/services/activity/logs.
- WebSocket topics: system/activity/run/eval/generation/GGUF. Managed inference separately serves OpenAI-style endpoints.

Keep these routes backward compatible; extend response schemas instead of inventing duplicate lifecycle endpoints.

## TrainingBackend and actual engine integration

`backends/base.py` defines TrainingBackend with validate_config, estimate_footprint, export_config, an event-yielding run method, and `TrainingBackendCapabilities`. The descriptor is the source for supported task/method properties and the additive `/api/runs/backends` capability response. `AutoBackendSelector` evaluates operation, model metadata, hardware, dependencies, objective, tokenizer policy, and runtime choices before enqueue; selected/rejected reasons are persisted and shown in the run plan. Explicit backend selection remains strict. RunContext carries directories, resume path and STOP sentinel. Built-ins are scratch, native Transformers, Unsloth, and optional LLaMA-Factory.

Native Transformers/TRL/PEFT currently supports causal SFT and continued pretraining using SFTTrainer, AutoModelForCausalLM, PEFT LoRA/QLoRA/DoRA or full tuning. SFTConfig/Trainer keyword changes are handled through signature inspection. Heavy imports stay inside workers. Native selection skips Unsloth import; however the baseline test named native independence checks metadata/validation/export, not an actual training step with Unsloth absent.

The optimized path imports Unsloth before Transformers/PEFT, attaches adapters, and falls back to native except for OOM. Full tuning and adapter continuation use native. Adapter continuation retains saved rank/targets. Fallback reason is logged but effective engine is not reliably recorded as immutable run identity. Broad fallback needs semantic checks and partial-allocation cleanup validation.

Native targets default to PEFT all-linear. Unsloth uses a fixed projection list; capability FamilyRule has a similar default. These are not architecture discovery. rsLoRA is an executable flag; other advanced PEFT methods have no strategy registry. QLoRA hard-codes NF4/nested quantization and hardware dtype; no user FP4/8-bit/storage policy. Gradient checkpointing is boolean; precision, attention, freeze, alignment, profiling, early stopping and distributed strategies lack complete typed/runtime paths.

Continued pretraining uses canonical raw content plus EOS and bypasses chat formatting, but shares SFTTrainer. Its semantic path deserves a tiny real test and packing/validation configuration. The frontend does not serialize packing. SFTTrainer receives no validation dataset; eval_on_completion is declared without a consumer.

Scratch is a custom torch GPT with trained byte-level BPE, block sampling, AdamW, loss/samples/checkpoints. It reads/tokenizes the corpus into memory. Checkpoints contain weights/architecture/tokenizer; optimizer/RNG/data cursor are absent and resume_from is ignored. The worker entrypoint can discover numbered checkpoints, but this does not make scratch resumable. Do not replace scratch; extend its checkpoint contract.

`LlamaFactoryBackend` dynamically reports absent, incompatible, experimental, or supported installation state. It translates validated config and canonical data at its boundary, runs the CLI inside the normal subprocess/event/GPU-lease lifecycle, and adds no mandatory dependency. Its installed-runtime path was unavailable for end-to-end verification. A shared native `PreferenceTrainer` implements DPO/IPO/ORPO/SimPO strategies and KTO label semantics; PPO and distributed launching remain unimplemented.

## Data Lab, dataset identity and normalization

`datasets/adapters.py` is the authoritative import-light canonical boundary. `DatasetAdapter`, `DatasetAdapterCapabilities`, and `DatasetAdapterRegistry` sit behind the existing `detect_schema` and `canonicalize` facade. Adapters expose detect, structured validation, canonicalize, preview, semantic fingerprint, and stage-compatibility contracts. Existing raw text, pair, OpenAI messages, ShareGPT, ambiguity checks, and safety gates retain their behavior and historical v1 storage rows (`text` or `messages`).

Alpaca instruction conversion now preserves an optional system message and ordered history pairs before the current instruction/input and answer. It rejects malformed history or system/history metadata on formats that cannot represent it, so canonicalization cannot silently drop those fields.

Version-2 canonical contracts define preference branches, KTO desirability, structured tool calls, and media references. Preference and KTO adapters now canonicalize and validate alignment rows, including KTO label balance; they remain unavailable to SFT. Tool, pretokenized, and multimodal markers stay recognized and worker-gated. Existing prepared datasets, recipes, and DatasetVersion schema v1 remain unchanged.

`validate.py` owns file iteration/validation/heuristic token estimates; JSON arrays have different memory characteristics from streamed JSONL/CSV/text/Parquet. HF import supports bounded streaming. `prepare.py` uses a SQLite disk spool, deterministic sorting, canonical deduplication, seeded train/validation/test fractions, staged files and atomic publication with cleanup. API persists recipe/version/split metadata. Existing mappings are useful but UI exposes only prompt/answer.

`lineage.py` fingerprints files and records versions/backfills legacy references. These are immutable records pointing to files; source files are not necessarily immutable copies. RunConfig stores Dataset ID and workers fetch Dataset.path. A hash snapshot does not stop a file changing between creation and worker launch. Bind a verified version before caches/mixtures.

`quality.py` gives exact/normalized/heuristic near duplicates, conflicting answers, script counts, PII-like warnings and normalized comparison leakage with bounded examples. Analysis normalization does not mutate source text. Cross-split prompt/answer/reference contamination coverage is incomplete; same high-bit simhash bucket is a heuristic, not exhaustive near-duplicate recall.

## Tokenizers, templates and masks

`train_entry/tokenization.py` centralizes canonical rendering and labels for training/profiling. Full-sequence, completion-only exact-prefix masking and assistant-only template masks are implemented. Empty or fully truncated targets fail. Native or explicit training template is required; no silent generic training fallback. Explicit custom templates currently override tokenizer templates, despite frontend hints saying fallback-only.

`tokenizer_profile.py` loads only tokenizer in a subprocess, reports lengths, percentiles, scripts, fertility proxies, unknown/truncation rates, utilization estimates and up to five previews (256 token display cap). `api/data_lab.py` persists TokenizerArtifact and DatasetProfile. These are diagnostics caches, **not a tokenized training cache**.

The cached-profile fast path accepts any nonempty revision, including branch/tag names; it must resolve immutable commits first. Tokenizer fingerprint includes vocab/special/template/class but not full serialized tokenizer behavior such as merges/normalizer configuration. Unknown counts use truncated IDs while denominator uses pre-truncation lengths. Script classification is duplicated in quality and tokenizer profiler.

Tokenizer mode reuse executes; extend/train modes are rejected by native trainer, and added token fields have no mutation/embedding-resize path. Scratch trains its own tokenizer separately for a legitimate different artifact format. Extend the common policy without forcing scratch through incompatible HF APIs.

Inference `model_runtime._encode` has independent template fallback and may tokenize just prompt after template failure, dropping other context. Evaluation canonicalizes then joins roles into text, which is passed as a new user prompt. Unify resolution and retain structured messages without duplicating training-only label logic.

## Model inspection, capabilities and resource planning

`model_refs.py` is the shared lazy resolver for HF/local/run references, scratch and adapters. `integrations/hf_hub.py` reads config/tokenizer/model metadata and caches capabilities by revision. `models/capabilities.py` classifies architecture from metadata, including unknown causal, MoE, encoder/seq2seq and multimodal; repository labels no longer establish architecture. Family metadata now implements `ModelFamilyAdapter`, and inspection adds structured template, PEFT-method, and operation-specific quantization contracts while retaining all legacy maps. Preserve conservative unknown/unsupported classification.

Backend listings and frontend method/backend availability consume the backend descriptor returned by the API. `app/capabilities.py` centralizes support/evidence models and import-free optional dependency inspection; system diagnostics reuse the same registry. Pre-launch operation + model + runtime + hardware selection is implemented, but B02 isolated loader/forward/backward evidence remains open, so declared or installed evidence must not be described as runtime verification.

Preflight coordinator provides timeout/cancellation and GPU lease isolation. The worker loads AutoConfig/AutoTokenizer only; CUDA availability sets unsloth_supported=True. It does not import/instantiate Unsloth or AutoModel and does not measure model trainability. Scratch preflight similarly checks artifacts/metadata. Rename evidence levels before claiming backend verification. Admission currently does not call the external-provider owner check used elsewhere.

`integrations/llmfit.py` with `estimator.py` provides fit estimates/fallback; `recommendations.py` supplies settings recommendations. Scratch has architecture-specific estimates. Keep these owners: refine counts and add reference/device/precision inputs instead of new competing advisors. Estimates use assumed architecture/adapter counts and safe budget; not guaranteed peak VRAM.

## Workers, scheduling, events and telemetry

The FIFO queue recovers queued runs and marks interrupted running/eval/export states failed on restart. `resources.py` uses a lock and lease token for one GPU owner **inside one process**. Training/evaluation/merge/provider paths also check external vLLM/Ollama ownership. This is not a distributed GPU scheduler or cross-process lock.

`runner.py` launches python worker with merged stdout/stderr, JSON events and a STOP -> terminate -> kill cancellation escalation. It writes logs/metrics/checkpoints/status/manifests/failure reports and releases resources through the queue. The training loop runs in a thread to bridge synchronous Trainer callbacks into events. Future nested engine/distributed children need process-tree termination and bounded queues.

Training event union retains log, metric, checkpoint, sample, and status and accepts additive progress, resource, artifact, warning, profile, and error envelopes. Backends now emit profile and artifact evidence; the runner idempotently registers final model/adapter artifacts and preserves reference lineage. The parser remains tolerant of unknown/stray lines. Metric history is JSONL; each metric also updates Run.metrics and commits, so high-frequency expansion still needs batching.

CPU jobs use a bounded semaphore. Tokenizer profiling has a timeout, but task-cancellation cleanup needs proof; GGUF conversion uses synchronous subprocess streaming without a comparable explicit job timeout/cancel contract. Do not infer all workers are equally bounded.

## Evaluation, registry, lineage and exports

Evaluation manager isolates GPU jobs and persists EvalResult. Generative models use shared ModelRuntime for generation/evaluation/deployment; registered reward artifacts use a separate scalar sequence-classification runtime and preference datasets to report chosen/rejected score, reward margin, and pairwise accuracy. Multiple models run sequentially. Generative metrics include exact match/token F1 and optional ROUGE/BLEU, reference perplexity, latency/throughput and optional judge. This is a useful comparison foundation, not a benchmark/regression suite. Fix structured conversation parity before trusting generative comparisons. Equal-example aggregation of perplexities should be documented distinctly from corpus token-weighted perplexity.

Registry can browse completed runs via stable run references, import/publish HF models, generate a model card, merge adapters and convert/quantize GGUF. Final trainer checkpoints/artifact events idempotently create ModelArtifact rows with run/base/dataset/backend/method/reference metadata. An API normalization layer maps historical strings to Causal LM, Adapter, Merged Model, Reward Model, Reference Model, and Quantized Model categories and exposes evaluation capabilities without changing the table. Reward artifacts are excluded from generation, deployment, causal merge, and ordinary base-model selection. Card generation remains tied to publish, not run completion. ModelArtifact still has no explicit parent artifact FK; preserve old kind strings if a future migration adds one.

Merge runs in an isolated worker and writes a new artifact. Current checks establish adapter kind, not pinned base/tokenizer/quant-state compatibility. GGUF invokes external llama.cpp tools, stores progress/logs and artifact status; there is no unified ExportJob or calibration dataset. Ollama import creates a temporary FROM-only Modelfile and runs CLI; template/stops/generation defaults and export-only packaging are absent.

## Serving and Docker

`serving/providers.py` already has ModelServingProvider and registry for managed Transformers, externally managed Compose vLLM, and Ollama. Preserve it; adding another InferenceProvider registry would duplicate functionality. Normalize feature/lifecycle responses behind the existing boundary.

Managed Transformers server holds one model, has health/start/stop, accepts scratch/HF/adapters via ModelRuntime and exposes /v1/models, /v1/chat/completions (including SSE) and /v1/completions. Deployment state is in memory and an active workdir/config, not a database Deployment entity. Restart reconciliation/persisted identity are missing.

External vLLM has health/models/test and explicit Docker lifecycle guidance. Older warm Playground vLLM subprocess manager remains in eval_manager. Both need a deliberate consolidation preserving local/Compose modes. Ollama detects HTTP/CLI, lists loaded models, controls a managed model with lease and imports GGUF. No SGLang, score API or persisted TTFT/concurrency benchmark exists.

Compose separates GPU backend, CPU/dev variants, nginx frontend and optional dedicated vLLM profile. GPU dependency constraints and import smoke commands exist in Dockerfile; Compose config validation passed. No image build or running service verified here. HTTP API binds differ between native localhost defaults and container published ports; current application has no general authentication layer. Keep the local-workbench scope explicit rather than claiming Internet-facing security.

## Frontend architecture

React 18 + TypeScript + Vite; React Router lazy pages, React Query server state, workflow hooks for persisted drafts, Tailwind/Radix components, Recharts monitors, centralized API/error/log clients. Routes: Dashboard, Projects, Datasets, Pretrain, Domain Adaptation, Finetune, Alignment, Eval Lab, Registry, Runs, Serving, System.

TrainingStudio, model/dataset pickers, recommendation/fit panels and RunMonitor are shared. Data Lab already has schema/prepare/tokenizer/quality views. Extend nested areas; do not multiply top-level screens. Flat RunForm -> payload and formFromConfig omit unsupported advanced fields, so clone/edit/export roundtrip is not lossless for full RunConfig. Static METHODS now supplies presentation copy only; backend descriptors decide availability. Frontend hook lint warning is recorded in STATUS.

## Logging, errors, settings and docs

Settings use SLMKIT_ environment prefix and one home for SQLite/datasets/models/runs/cache; secrets remain configured out of DB by default; trust_remote_code false. Heavy GPU stack is optional for basic API, but GPU extra includes Unsloth and broad version ranges; optional engine/runtime compatibility needs finer diagnostics.

Structured activity, request IDs/timing headers, rotating backend log, persisted worker JSONL, ErrorPanel, normalized failure codes and portable failure artifacts are implemented. Raw exceptions/stdout are not universally redacted. Manifest redaction is overbroad on keys containing token (including tokenizer metadata). Fix evidence-preserving redaction rather than suppressing useful errors.

Documentation covers architecture/install/configuration/APIs/pages/Docker/troubleshooting. Historical plan findings about no migrations and first-turn-only extraction are stale; current code has migrations and retains text of multiple turns (but loses structured semantics). Audit docs supersede status claims for planning, not wholesale user documentation replacement.

## Baseline and first action

See [STATUS.md](STATUS.md) for exact commands/results and runtime limits. Step 16 verification: 165 backend tests passed on the isolated CPU ML stack; the lean stack passed 144 with two optional skips. Eighteen frontend tests, Ruff, frontend build/type check passed; ESLint retains one existing warning. Native TRL tiny jobs now execute locally. No CUDA or installed optional-engine training job was performed; the packaged Torch 2.11 GPU stack is unverified.

**Next within the current Phase C scope: C01 — durably pin policy/reference identities and retain them on resume.** Then complete C08's base-versus-trained regression gate. Earlier Phase A/B gaps remain tracked; do not begin optimization while Phase C exit criteria remain unmet.
