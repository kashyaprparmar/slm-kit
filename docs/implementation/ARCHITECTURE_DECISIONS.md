# Architecture decisions

Status: accepted architecture decisions, updated after the 2026-09-15 foundation review.

## ADR-19 — Optimization options use one capability registry

**Decision.** Advanced PEFT methods, external optimizers, and optional acceleration features are represented by `OptimizationCapability` descriptors owned by `app.optimizations`. Each descriptor carries its availability, detected package version, compatibility, UI configuration schema, progressive-disclosure level, and an explainable impact record. `RunConfig` remains the sole persisted configuration model; the descriptor schema describes its fields rather than creating another config format.

**Consequences.** API validation, automatic backend selection, estimate hooks, worker-only runtime adapters, and frontend controls use the same identifiers. Existing `optional_features` aliases remain only for compatibility. Optional packages never become startup dependencies. A descriptor is unavailable until its runtime adapter and fixture exist; it must not be surfaced as executable simply because package metadata was detected.

## ADR-01 — Keep SLM Kit as the application and orchestration layer

Retain FastAPI/domain/SQLite/projects/job lifecycle as the authoritative control plane. Extend TrainingBackend for native Transformers/TRL/PEFT, optimized Unsloth and optional LLaMA-Factory. Extend the existing ModelServingProvider protocol as the inference boundary for Transformers/vLLM/Ollama and future SGLang. A renamed/new parallel registry adds no value. LLaMA-Factory must never become mandatory or dictate persistence/UI.

## ADR-02 — Separate policy from execution without duplicating engines

Native registration currently aliases UnslothBackend. Extract shared trainer/dataset/callback/config plumbing and loader strategies incrementally (B03), preserving behavior and lazy imports. Share model reference/revision/tokenizer policy with ModelRuntime while keeping trainability, adapter attachment and inference-specific behavior distinct. Do not force every consumer through a heavy all-purpose loader.

## ADR-03 — One authoritative capability decision

`app/capabilities.py`, model-family adapters, and `TrainingBackendCapabilities` are the shared import-light contracts. Backend task/method validation and frontend visibility derive from backend descriptors. Declared, installed, metadata, and runtime evidence are distinct. B02/B04 must still resolve operation + model + method + runtime + hardware before launch. Auto fallback must preserve semantics and persist the selected engine and reason.

## ADR-04 — Preserve canonicalize as the dataset facade

Adapters are registered behind the existing functions and expose detect/validate/canonicalize/preview/fingerprint/stage contracts. Raw text and ordinary conversations retain their exact v1 storage rows for recipe and prepared-data compatibility. Preference, KTO, tool and media records use explicit v2 discriminators and remain capability-gated until their source adapters and consumers exist. Semantic fingerprints hash canonical storage rather than source columns. Keep detection, canonical conversion and stage compatibility separate from template rendering/tokenization. Never pass raw arbitrary columns to a trainer. Extend DatasetRecipe.config for mapping, not an unrelated mapping subsystem.

## ADR-05 — Share template resolution, keep label logic explicit

Extend tokenization.py and existing runtime helpers: one template identity/resolution policy with source and warnings across preparation/train/eval/serve. Preserve explicit override semantics with accurate UI text. Native tokenizer template first unless explicitly overridden; known compatible family template next; generic fallback only visibly and intentionally permitted. Retain structured eval messages. Keep training loss policy separate from generation prompt policy. Reasoning/tool capabilities are template-specific.

## ADR-06 — Immutable input identity before caching

Extend DatasetVersion and TokenizerArtifact. Resolve immutable hub commits and hash full tokenizer serialization; protect local artifact identity by content. Bind runs to dataset versions and verify files before execution. Preserve historical unknown identity as unknown, never invent a commit or backfill a changed file as original. Diagnostic DatasetProfile cache and tokenized training cache have different purposes; a new cache may reuse fingerprint utilities but must include all semantic inputs and atomic concurrency-safe publication.

## ADR-07 — Additive SQLite migrations, JSON where appropriate

Existing Alembic 0001-0003, backups, WAL/FKs/indexes stay. New entities only for durable lifecycles/query relationships: run version binding, mixtures/cache, parent artifact relationship, deployment/export/benchmark identity. Version immutable run strategy/config snapshots in JSON; no table per optimizer. New migrations freeze their own definitions and test fresh plus legacy adoption. No destructive backfill or retroactive reinterpretation of old configs. No migration is added by this audit.

## ADR-08 — Retain one control-plane process; scale workers deliberately

Current GPU lease is process-local, so multiple API workers are unsupported. Centralize admission policy including external providers before extending devices. Future FSDP/DeepSpeed use managed device sets, launch metadata, rank-zero events and process-tree cleanup. Preserve single GPU default and CPU-only API availability. Do not give API unrestricted Docker access merely to control external vLLM.

## ADR-09 — Extend events and durable lifecycle evidence

Keep existing JSONL events/WebSocket topics tolerant of unknown types. The shared union now accepts progress/resource/artifact/warning/profile/error envelopes with backward compatibility. Later work maps engine callbacks, persistence, and UI presentation at the boundary. Batch high-frequency metrics into artifacts with compact DB summaries. Every success/failure/cancel must leave a truthful final state, safe resource release and reproducibility evidence.

## ADR-10 — Distinguish run output, artifact and deployment

A completed Run currently supplies a stable reference but not automatic ModelArtifact registration. Add idempotent final artifact creation and explicit parent IDs. Preserve run:<id>, legacy base_model/source_ref and old kind strings. New immutable merge/quantized/reward/reference artifacts retain lineage. Persist deployment identity/config and reconcile runtime state on restart; a stored record does not prove a process remains healthy.

## ADR-11 — Alignment strategies share the training foundation

Represent stage and objective separately from tuning method. Preference trainers share canonical input, reference resolution, tokenizer/template policy, events and artifact lifecycle; DPO/IPO/ORPO/SimPO are objective adapters, not independent applications. KTO has its own canonical labels. Reward models require scoring semantics. PPO waits until simpler objectives and resource accounting are verified.

## ADR-12 — Optional advanced integrations stay capability-gated

No automatic activation/install of LLaMA-Factory, trackers, DeepSpeed, kernels, alternative quantization or SGLang. Optional dependency detection is centralized and import-free; detection does not register a feature. Native training must work with Unsloth absent. Add one tested optimizer/PEFT adapter at a time; proposed/unsupported methods remain unavailable. Multimodal and architecture transformations remain expert/experimental until a specific supported processor/model workflow is tested.

## ADR-13 — Preserve the UI and improve disclosure

Retain route URLs/shared studios/React Query/draft state. Add nested Training Alignment, Data Lab mapper/preview and Models export/provider views. Recommended/Advanced/Expert controls reflect the same backend capability model. Preserve full configuration when editing/rerunning; do not lose invisible fields through the flat form.

## ADR-14 — Evidence and error handling are product contracts

Do not claim installed packages, static family rules, metadata-only preflight or mocked unit tests prove real training. Record requested/effective versions and capabilities and actionable causes. Central secret redaction must cover raw logs/errors/exports while preserving benign tokenizer metadata. Audit baseline failures and newly introduced failures remain separate.

## ADR-15 — Resolve trainability from the loaded architecture

Keep module selection import-light, but apply it to the loaded model before trainer construction. LoRA auto/all-linear/attention/MLP/custom policies and freeze groups share this inspection boundary. Emit the exact selected module or parameter names and actual counts through the event protocol. API estimates remain clearly labeled estimates; family suggestions may guide auto mode but never replace inspection. Explicit precision, attention, checkpointing, quantization, Liger, and RoPE choices must either execute with the requested semantics or fail validation/fall back to an equivalent engine with a visible reason.

## ADR-16 — Resolve Auto before enqueue and preserve the decision

`AutoBackendSelector` evaluates the exact task, tuning method, alignment objective, model metadata, hardware, installed dependencies, tokenizer/loss policy, precision, attention, checkpointing, RoPE, and optional kernels. It may rank only semantically compatible backends. The requested backend, selected backend, reason, and rejection reason for every alternative live in the existing Run config JSON and appear in the run plan. Explicit backend requests remain strict. Cloning restores the original Auto intent while rerunning preserves the effective backend for reproducibility.

## ADR-17 — Keep alignment objective adapters behind one trainer lifecycle

`PreferenceTrainer` owns canonical dataset rendering, callbacks, metrics, checkpoints, artifacts, and lineage. DPO and IPO share the DPO trainer; ORPO and SimPO select their compatible TRL strategies; KTO retains separate label semantics. Reward modeling selects TRL's reward trainer and a scalar sequence-classification loader while retaining the same lifecycle. Objective-specific imports are lazy so one missing optional trainer does not disable unrelated objectives. Base/separate reference models reuse the run quantization policy and are included in pre-launch memory estimates; adapter-disabled and reference-free strategies allocate no second model. PPO remains outside this slice.

## Step 16 clarifications to ADR-16/ADR-17

Backend optional capability keys also cover `dpo_loss:<loss>` and `reference:<strategy>`; Auto and UI use those same keys. Missing keys are unavailable. Reference-bearing objectives require an explicit reference strategy, while reference-free objectives use none. Separate adapter references retain the adapter and must share the policy's token-ID mapping. SimPO explicitly disables CPO's additional SFT term. PEFT sequence-classification heads are excluded through the existing module-discovery helper because PEFT saves them in full. Alignment data preparation reuses canonical validation and content fingerprints to invalidate changed source files. Native alignment resume skips visibly incomplete HF saves; cancellation does not register completed output. These changes extend existing owners and require no migration.

## ADR-18 — Artifact category controls valid runtime operations

Keep the existing open-string artifact column for backward compatibility and normalize historical values at the model-reference boundary. Registry/API clients see stable Causal LM, Adapter, Merged Model, Reward Model, Reference Model, and Quantized Model categories plus evaluation capabilities. Reward models load through `AutoModelForSequenceClassification`, expose scalar pairwise evaluation, and cannot enter generation, deployment, causal merge, or ordinary fine-tuning selectors. This requires no database migration; adapter/full representation and reference lineage remain in artifact metadata.

## Upstream reference boundary

The supplied specification links [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), whose upstream page was checked as a capability reference during this audit. Its feature breadth is not evidence of SLM Kit implementation. No upstream code was copied or installed. Before implementing B14/B15 or incorporating any code, pin a reviewed upstream version and review its license and runtime compatibility; do not design against an unpinned main branch.

## ADR-19 — Export jobs extend artifact and event owners

An export is a supervised subprocess with an immutable request snapshot in existing artifact JSON. Reuse ProgressEvent/ProfileEvent/ArtifactEvent, log capture, resource leases and process-tree termination. At most two jobs run concurrently; merging also owns a GPU lease. Output is staged then renamed atomically; source and payload hashes, original artifact/run/base lineage and Hub commit receipt are retained. No ExportJob table or migration is necessary for this slice. Remote merge bases require original immutable commits; local bases are fingerprinted and quantized bases are refused. Quantization registry operation states distinguish package presence from verified exporters. Calibration metadata references existing immutable dataset versions.

## ADR-20 — Shared optional serving lifecycle and scalar scoring

Managed vLLM uses the existing warm manager; SGLang uses the same supervisor class. External service mode retains explicit ownership and cannot be stopped through managed controls. CLI flag probes run in isolated processes; CUDA count and artifact configuration constrain launch options. Advanced model-specific profiles remain unavailable until verified. Root OpenAI discovery advertises provider::model IDs; forwarding preserves provider payloads and streaming cleanup. Registered reward models may use the native managed scoring deployment and /v1/scores, while generation/reference-artifact deployment remains rejected. This is a scoring exception to ADR-18's earlier deployment restriction; it does not permit reward-model text generation.
