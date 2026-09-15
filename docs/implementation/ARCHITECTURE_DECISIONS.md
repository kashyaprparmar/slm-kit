# Architecture decisions

Status: accepted architecture decisions, updated after the 2026-09-15 foundation review.

## ADR-01 — Keep SLM Kit as the application and orchestration layer

Retain FastAPI/domain/SQLite/projects/job lifecycle as the authoritative control plane. Extend TrainingBackend for native Transformers/TRL/PEFT, optimized Unsloth and optional LLaMA-Factory. Extend the existing ModelServingProvider protocol as the inference boundary for Transformers/vLLM/Ollama and future SGLang. A renamed/new parallel registry adds no value. LLaMA-Factory must never become mandatory or dictate persistence/UI.

## ADR-02 — Separate policy from execution without duplicating engines

Native registration currently aliases UnslothBackend. Extract shared trainer/dataset/callback/config plumbing and loader strategies incrementally (B03), preserving behavior and lazy imports. Share model reference/revision/tokenizer policy with ModelRuntime while keeping trainability, adapter attachment and inference-specific behavior distinct. Do not force every consumer through a heavy all-purpose loader.

## ADR-03 — One authoritative capability decision

`app/capabilities.py`, model-family adapters, and `TrainingBackendCapabilities` are the shared import-light contracts. Backend task/method validation and frontend visibility derive from backend descriptors. Declared, installed, metadata, and runtime evidence are distinct. B02/B04 must still resolve operation + model + method + runtime + hardware before launch. Auto fallback must preserve semantics and persist the selected engine and reason.

## ADR-04 — Preserve canonicalize as the dataset facade

Adapters are registered behind the existing functions and expose detect/validate/canonicalize/preview/stage contracts. Version CanonicalRecord before preference/KTO/tools/media extension; retain v1 text readers. Keep detection, canonical conversion and stage compatibility separate from template rendering/tokenization. Never pass raw arbitrary columns to a trainer. Extend DatasetRecipe.config for mapping, not an unrelated mapping subsystem.

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

## Upstream reference boundary

The supplied specification links [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), whose upstream page was checked as a capability reference during this audit. Its feature breadth is not evidence of SLM Kit implementation. No upstream code was copied or installed. Before implementing B14/B15 or incorporating any code, pin a reviewed upstream version and review its license and runtime compatibility; do not design against an unpinned main branch.
