# Gap analysis

Audit date: 2026-09-15. Foundation review update: 2026-09-15. Scope: actual working tree at HEAD `5d64b78b6dc5289ef419d22ab11cd629cdba6337` plus pre-existing modifications/untracked source. The supplied 92-section specification is the requirement index. Classifications still describe complete requirement status; foundation slices do not make later feature tasks complete.

## Classification rules

1. Already implemented correctly: inspected behavior with relevant baseline tests, limited to the stated scope.
2. Implemented but needs enhancement: working foundation worth extending.
3. Partially implemented: only some requested behavior exists, or a material correctness gap remains.
4. Missing: no application implementation; dependency code/generated caches do not count.
5. Experimental: code advertises uncertain support without verified execution.
6. Not appropriate for the existing architecture: rejected design direction, not a reason to discard requested optional extensions.

A passing unit suite is not proof of GPU/runtime support. Planned experimental features that have no code are **Missing**, not implemented experimental features. Classification is per requirement; individual supported subfeatures remain reusable.

<a id="r01"></a>

## R01 — FIRST: AUDIT THE CURRENT IMPLEMENTATION

- **Classification:** 1. Already implemented correctly.
- **Code finding:** This audit inspects code and establishes baseline; earlier reports contain stale findings.
- **Evidence:** `docs/implementation/CURRENT_STATE.md`.
- **Roadmap:** AUDIT (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r02"></a>

## R02 — ARCHITECTURAL PRINCIPLE

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** TrainingBackend and provider registry exist. Backend descriptors now drive API/UI task and method availability, and typed provider/model capability contracts are shared. Native and Unsloth still share one worker class; runtime evidence intersection and extracted strategies remain incomplete.
- **Evidence:** `backend/app/backends/base.py`, `backend/app/serving/providers.py`.
- **Roadmap:** B01, B03 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r03"></a>

## R03 — OPTIONAL LLAMA-FACTORY BACKEND

- **Classification:** 4. Missing.
- **Code finding:** No LlamaFactoryBackend registration or launcher. Capability entry is a fixed not_installed placeholder, not detection.
- **Evidence:** `backend/app/backends/base.py`, `backend/app/models/capabilities.py`.
- **Roadmap:** B14, B15 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r04"></a>

## R04 — TRAINING STAGE MODEL

- **Classification:** 3. Partially implemented.
- **Code finding:** Persisted task enums are now mapped to a separate TrainingStage and normalized operation identity. LoRA/QLoRA/DoRA/full execute; freeze and alignment execution remain absent. Prompt tuning stays explicitly unsupported.
- **Evidence:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** B01, B06, C01 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r05"></a>

## R05 — FREEZE TUNING

- **Classification:** 4. Missing.
- **Code finding:** No freeze configuration, module selection execution or exact trainable preview.
- **Evidence:** `backend/app/domain.py`.
- **Roadmap:** B05, B06 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r06"></a>

## R06 — CANONICAL DATA ADAPTER SYSTEM

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** canonicalize/detect_schema already unify preparation, validation, training and evaluation; no extensible adapter descriptors.
- **Evidence:** `backend/app/datasets/adapters.py`.
- **Roadmap:** A01, A04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r07"></a>

## R07 — ALPACA DATA FORMAT

- **Classification:** 3. Partially implemented.
- **Code finding:** Instruction/input/output works. Pair conversion ignores top-level system and history (reproduced in audit).
- **Evidence:** `backend/app/datasets/adapters.py`.
- **Roadmap:** A01 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r08"></a>

## R08 — OPENAI CONVERSATION FORMAT

- **Classification:** 1. Already implemented correctly.
- **Code finding:** Strict text-only system/user/assistant messages and ordered multi-turn conversion covered by adapter tests. Tool/content-part extensions are separate requirements.
- **Evidence:** `backend/app/datasets/adapters.py`, `backend/tests/test_dataset_adapters.py`.
- **Roadmap:** A04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r09"></a>

## R09 — SHAREGPT FORMAT

- **Classification:** 3. Partially implemented.
- **Code finding:** human/gpt/system plus role/from/content/value supported; configurable nested names and tool observation/function_call mappings absent.
- **Evidence:** `backend/app/datasets/adapters.py`.
- **Roadmap:** A04, A05, A07 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r10"></a>

## R10 — DATASET COLUMN MAPPER

- **Classification:** 3. Partially implemented.
- **Code finding:** Backend mappings persist in recipe config; UI maps only prompt and answer. No rich nested canonical-field mapper.
- **Evidence:** `backend/app/api/datasets.py`, `frontend/src/components/DatasetPreparation.tsx`.
- **Roadmap:** A05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r11"></a>

## R11 — PREFERENCE DATASETS

- **Classification:** 3. Partially implemented.
- **Code finding:** Chosen/rejected markers detected and explicitly gated; no canonical preference model or preference training validation.
- **Evidence:** `backend/app/datasets/adapters.py`.
- **Roadmap:** A06 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r12"></a>

## R12 — KTO DATASETS

- **Classification:** 4. Missing.
- **Code finding:** No desirability schema. prompt/response/desirable may be interpreted as ordinary pair and lose label; stage cannot be KTO.
- **Evidence:** `backend/app/datasets/adapters.py`, `backend/app/domain.py`.
- **Roadmap:** A06 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r13"></a>

## R13 — TOOL-CALLING DATASETS

- **Classification:** 3. Partially implemented.
- **Code finding:** Tools/calls recognized and rejected; no schema/argument/ID validation or training execution.
- **Evidence:** `backend/app/datasets/adapters.py`.
- **Roadmap:** A07 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r14"></a>

## R14 — MULTIMODAL-READY DATA MODEL

- **Classification:** 3. Partially implemented.
- **Code finding:** Model metadata detects multimodal and disables worker; canonical data has no media descriptors or modality contract.
- **Evidence:** `backend/app/models/capabilities.py`, `backend/app/datasets/adapters.py`.
- **Roadmap:** A06, G01, G02 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r15"></a>

## R15 — DATASET MIXING

- **Classification:** 4. Missing.
- **Code finding:** Run selects one Dataset ID; no mixture identity, weights or sampling strategy.
- **Evidence:** `backend/app/domain.py`, `backend/app/db/models.py`.
- **Roadmap:** A17, A18 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r16"></a>

## R16 — DATASET SPLITS

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Seeded persisted train/validation/test outputs, recipes and fingerprints exist. No source/manual/stratified split model; workers still resolve legacy Dataset path.
- **Evidence:** `backend/app/datasets/prepare.py`, `backend/app/datasets/lineage.py`.
- **Roadmap:** A03, A14 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r17"></a>

## R17 — LEAKAGE DETECTION

- **Classification:** 3. Partially implemented.
- **Code finding:** Exact/normalized/near duplicate, conflicting-answer and cross-source normalized leakage examples exist; full cross-split prompt/answer/reference detection absent.
- **Evidence:** `backend/app/datasets/quality.py`.
- **Roadmap:** A15 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r18"></a>

## R18 — DATASET PACKING

- **Classification:** 3. Partially implemented.
- **Code finding:** TRL boolean packing and scratch continuous token blocks exist. Masked packing explicitly rejected; no mode selector/neat guarantees; UI payload omits packing.
- **Evidence:** `backend/app/backends/unsloth_backend.py`, `backend/app/domain.py`, `frontend/src/lib/runconfig.ts`.
- **Roadmap:** A12 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r19"></a>

## R19 — TOKENIZED DATASET CACHE

- **Classification:** 4. Missing.
- **Code finding:** DatasetProfile caches diagnostic results, not reusable training tensors. HF datasets implicit cache is not specified semantic cache contract.
- **Evidence:** `backend/app/api/data_lab.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** A02, A13 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r20"></a>

## R20 — TOKENIZER AND TEMPLATE REGISTRY

- **Classification:** 3. Partially implemented.
- **Code finding:** TokenizerArtifact/profiler records vocab/specials/fingerprint/revision; worker supports reuse and rejects extend/train. Import/resize lifecycle absent; added tokens unused.
- **Evidence:** `backend/app/db/models.py`, `backend/app/train_entry/tokenizer_profile.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** A02, A11, B10 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r21"></a>

## R21 — CHAT TEMPLATE CAPABILITY SYSTEM

- **Classification:** 3. Partially implemented.
- **Code finding:** Native or explicit training template required; no family/template capabilities. Inference silently falls back to plain prompt. Custom template overrides native despite UI hint.
- **Evidence:** `backend/app/train_entry/tokenization.py`, `backend/app/train_entry/model_runtime.py`.
- **Roadmap:** A08 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r22"></a>

## R22 — CHAT TEMPLATE PREVIEW

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Data Lab previews rendering, IDs, token strings, loss masks and truncation. Studio has no same preview, no packed-boundary display.
- **Evidence:** `frontend/src/pages/Datasets.tsx`, `backend/app/train_entry/tokenizer_profile.py`.
- **Roadmap:** A09, A10 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r23"></a>

## R23 — LOSS MASK VISUALIZATION

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Full/completion/assistant policies implemented with prefix/mask checks and tests; no distinct final/all assistant controls or richer token roles.
- **Evidence:** `backend/app/train_entry/tokenization.py`, `backend/tests/test_data_lab_phase2.py`.
- **Roadmap:** A10 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r24"></a>

## R24 — REASONING MODEL DATA

- **Classification:** 4. Missing.
- **Code finding:** No model-specific reasoning preservation/masking/stripping policy.
- **Evidence:** `backend/app/domain.py`, `backend/app/train_entry/tokenization.py`.
- **Roadmap:** A10 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r25"></a>

## R25 — MULTILINGUAL TOKENIZER DIAGNOSTICS

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Scripts, token/character/word rates, percentiles, truncation and unknown rate exist. Script is not language; unknown count uses truncated IDs against full-length denominator.
- **Evidence:** `backend/app/train_entry/tokenizer_profile.py`, `backend/app/datasets/quality.py`.
- **Roadmap:** A15 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r26"></a>

## R26 — ADVANCED PEFT REGISTRY

- **Classification:** 3. Partially implemented.
- **Code finding:** LoRA/QLoRA/DoRA and rsLoRA flag exist; backend and model capability payloads now represent PEFT methods without registering unimplemented ones. No installed-version strategy registry or LoRA+/PiSSA/LoftQ/EVA/OFT/QOFT execution.
- **Evidence:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** B01, D01, G04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r27"></a>

## R27 — AUTOMATIC LORA TARGET DISCOVERY

- **Classification:** 3. Partially implemented.
- **Code finding:** Native PEFT defaults all-linear; Unsloth defaults hard-coded Llama projections; family suggestions also default these for other architectures. No module inspection preview.
- **Evidence:** `backend/app/backends/unsloth_backend.py`, `backend/app/models/capabilities.py`.
- **Roadmap:** B05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r28"></a>

## R28 — QLORA IMPROVEMENTS

- **Classification:** 3. Partially implemented.
- **Code finding:** Native loader uses NF4, nested quant and hardware dtype. No selectable FP4/8bit/storage configuration or complete prelaunch dependency validation.
- **Evidence:** `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** B07 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r29"></a>

## R29 — QUANTIZATION CAPABILITY REGISTRY

- **Classification:** 5. Experimental.
- **Code finding:** Capability metadata labels GPTQ/AWQ experimental; no execution proof/export pipeline for them. INT8 capability is broader than current train config. No operation-specific HQQ/EETQ/AQLM matrix.
- **Evidence:** `backend/app/models/capabilities.py`.
- **Roadmap:** B07, E06 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r30"></a>

## R30 — ADVANCED MEMORY-EFFICIENT TRAINING

- **Classification:** 4. Missing.
- **Code finding:** Free optimizer string passed to TRL; no advanced optimizer strategy registry, memory contracts or tests.
- **Evidence:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** D02, D03, D09, D10, D11 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r31"></a>

## R31 — ATTENTION IMPLEMENTATION

- **Classification:** 4. Missing.
- **Code finding:** Hardware feasibility fields exist; no authoritative Auto/SDPA/FA2/eager selector propagated into loader.
- **Evidence:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** D04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r32"></a>

## R32 — LIGER KERNEL

- **Classification:** 4. Missing.
- **Code finding:** No optional Liger selection or effective kernel recording.
- **Evidence:** `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** D05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r33"></a>

## R33 — GRADIENT CHECKPOINTING

- **Classification:** 3. Partially implemented.
- **Code finding:** Boolean checkpointing drives native/Unsloth behavior; no non-reentrant/auto mode model.
- **Evidence:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** D04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r34"></a>

## R34 — ROPE SCALING / CONTEXT EXTENSION

- **Classification:** 3. Partially implemented.
- **Code finding:** API rejects over-context sequence length; no model-aware RoPE scaling modes.
- **Evidence:** `backend/app/api/runs.py`.
- **Roadmap:** D06 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r35"></a>

## R35 — NEFTUNE

- **Classification:** 4. Missing.
- **Code finding:** No NEFTune configuration or runtime wiring.
- **Evidence:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** D05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r36"></a>

## R36 — PRECISION HANDLING

- **Classification:** 3. Partially implemented.
- **Code finding:** Worker chooses BF16/FP16 based on CUDA support; UI/API lack explicit precision contract; loader dtype auto can differ from intended precision.
- **Evidence:** `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** B07 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r37"></a>

## R37 — CONTINUED PRETRAINING

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** continued_pretrain selects raw canonical content plus EOS through SFTTrainer with no chat rendering. Missing explicit packing UI/default and tiny runtime evidence/validation split wiring.
- **Evidence:** `backend/app/backends/unsloth_backend.py`, `frontend/src/pages/DomainAdaptation.tsx`.
- **Roadmap:** A12, B08 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r38"></a>

## R38 — FROM-SCRATCH PRETRAINING

- **Classification:** 3. Partially implemented.
- **Code finding:** Custom GPT, byte BPE, causal blocks, save/sample and memory guard implemented. No optimizer/RNG restore, imported tokenizer path or HF from-config architecture choice.
- **Evidence:** `backend/app/backends/scratch_backend.py`.
- **Roadmap:** B09, B10 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r39"></a>

## R39 — DPO

- **Classification:** 4. Missing.
- **Code finding:** No DPO task/trainer/config. Generated Unsloth cache files are dependency artifacts, not registered SLM Kit features.
- **Evidence:** `backend/app/domain.py`, `backend/app/backends/base.py`.
- **Roadmap:** C01, C02 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r40"></a>

## R40 — IPO / ORPO / SIMPO

- **Classification:** 4. Missing.
- **Code finding:** No IPO/ORPO/SimPO objectives or shared preference strategy.
- **Evidence:** `backend/app/domain.py`.
- **Roadmap:** C01, C03, C04, C05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r41"></a>

## R41 — KTO

- **Classification:** 4. Missing.
- **Code finding:** No KTO task, labels pipeline or trainer.
- **Evidence:** `backend/app/domain.py`.
- **Roadmap:** C01, C06 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r42"></a>

## R42 — REWARD MODELING

- **Classification:** 4. Missing.
- **Code finding:** No reward trainer, score loader or reward artifact semantics.
- **Evidence:** `backend/app/domain.py`, `backend/app/train_entry/model_runtime.py`.
- **Roadmap:** C01, C07 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r43"></a>

## R43 — REFERENCE MODEL HANDLING

- **Classification:** 4. Missing.
- **Code finding:** No reference strategy or immutable reference lineage/memory accounting.
- **Evidence:** `backend/app/domain.py`.
- **Roadmap:** C01 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r44"></a>

## R44 — PPO

- **Classification:** 4. Missing.
- **Code finding:** No PPO orchestration. Planned experimental only; presence of compiled trainer cache is not integration.
- **Evidence:** `backend/app/backends/base.py`.
- **Roadmap:** G03 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r45"></a>

## R45 — EARLY STOPPING

- **Classification:** 4. Missing.
- **Code finding:** No early-stop configuration or validation dataset passed to SFTTrainer. eval_on_completion is declared but has no consumer.
- **Evidence:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** B11 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r46"></a>

## R46 — TRAINING PROFILER

- **Classification:** 4. Missing.
- **Code finding:** No PyTorch profiler scheduling or trace artifact workflow.
- **Evidence:** `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** D07 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r47"></a>

## R47 — MODULE PROFILING

- **Classification:** 4. Missing.
- **Code finding:** No module-level runtime profiler.
- **Evidence:** `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** D07 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r48"></a>

## R48 — TRAINING TELEMETRY

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Metrics callbacks, JSONL history, GPU/RAM/CPU/disk polling and monitor exist; DB updated on every metric; no full persistent resource history/effective token accounting.
- **Evidence:** `backend/app/core/runner.py`, `backend/app/core/hardware.py`, `frontend/src/components/RunMonitor.tsx`.
- **Roadmap:** B11, B16 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r49"></a>

## R49 — PREFERENCE TRAINING TELEMETRY

- **Classification:** 4. Missing.
- **Code finding:** Generic metric dictionary can carry values, but no preference event producer or objective-aware charts.
- **Evidence:** `backend/app/core/events.py`, `frontend/src/components/RunMonitor.tsx`.
- **Roadmap:** C08 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r50"></a>

## R50 — DISTRIBUTED-READY RUN CONFIG

- **Classification:** 4. Missing.
- **Code finding:** GPU list exists but one process-local global lease and no typed ranks/world/strategy. Capability text overstates represented distributed configuration.
- **Evidence:** `backend/app/core/resources.py`, `backend/app/domain.py`, `backend/app/models/capabilities.py`.
- **Roadmap:** F01, F02 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r51"></a>

## R51 — FSDP

- **Classification:** 4. Missing.
- **Code finding:** No FSDP worker/launcher.
- **Evidence:** `backend/app/backends/base.py`.
- **Roadmap:** F03 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r52"></a>

## R52 — DEEPSPEED

- **Classification:** 4. Missing.
- **Code finding:** Fixed not_installed capability; no dependency detection, ZeRO adapter or launcher.
- **Evidence:** `backend/app/models/capabilities.py`.
- **Roadmap:** F04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r53"></a>

## R53 — RUN CONFIG IMPORT / EXPORT

- **Classification:** 3. Partially implemented.
- **Code finding:** JSON export/rerun and frontend form restore exist. No YAML/complete config import; form serialization drops fields not represented in flat form.
- **Evidence:** `backend/app/api/runs.py`, `frontend/src/lib/runconfig.ts`.
- **Roadmap:** B13 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r54"></a>

## R54 — CLI

- **Classification:** 4. Missing.
- **Code finding:** Worker python -m entrypoints exist, but no user-facing CLI service facade or project script entry.
- **Evidence:** `backend/pyproject.toml`, `backend/app/train_entry/run.py`.
- **Roadmap:** B17 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r55"></a>

## R55 — MODEL SOURCE PROVIDERS

- **Classification:** 3. Partially implemented.
- **Code finding:** HF/local/run references are shared and work for current flows; no source-provider registry or ModelScope adapter.
- **Evidence:** `backend/app/model_refs.py`, `backend/app/integrations/hf_hub.py`.
- **Roadmap:** B18 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r56"></a>

## R56 — MODEL FAMILY ADAPTER LAYER

- **Classification:** 3. Partially implemented.
- **Code finding:** Central family metadata now implements ModelFamilyAdapter and supplies target/template extension hooks. Runtime loader patching, module discovery, and tested family template registrations remain absent; target assumptions still need B05.
- **Evidence:** `backend/app/models/capabilities.py`, `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** A08, B03, B05, D06 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r57"></a>

## R57 — MODEL PREFLIGHT

- **Classification:** 3. Partially implemented.
- **Code finding:** Metadata inspection/fingerprints and subprocess preflight exist. Preflight sets Unsloth verified from CUDA availability and never loads AutoModel/does a training probe.
- **Evidence:** `backend/app/train_entry/preflight.py`, `backend/app/integrations/hf_hub.py`.
- **Roadmap:** B02 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r58"></a>

## R58 — MODEL CARD GENERATION

- **Classification:** 3. Partially implemented.
- **Code finding:** Model card generator exists in publish path with config/metrics. No automatic completion card with exact tokenizer/template/dataset/revision/license lineage.
- **Evidence:** `backend/app/api/registry.py`.
- **Roadmap:** B12 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r59"></a>

## R59 — ARTIFACT LINEAGE

- **Classification:** 3. Partially implemented.
- **Code finding:** Run refs/base_model/source_ref and datasets have lineage; no explicit parent artifact FK, graph or persisted Deployment. Final training only registers Checkpoint.
- **Evidence:** `backend/app/db/models.py`, `backend/app/core/runner.py`, `backend/app/api/registry.py`.
- **Roadmap:** B12, E01 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r60"></a>

## R60 — ADAPTER MERGING

- **Classification:** 3. Partially implemented.
- **Code finding:** Isolated PEFT merge produces new artifact; no strict revision/vocab/quantization compatibility preflight.
- **Evidence:** `backend/app/train_entry/merge.py`, `backend/app/api/registry.py`.
- **Roadmap:** E05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r61"></a>

## R61 — EXPORT PIPELINE

- **Classification:** 3. Partially implemented.
- **Code finding:** HF publish, adapter/full paths, merge and GGUF jobs/logs exist; no unified persistent export job, bounded converter cancellation or complete formats.
- **Evidence:** `backend/app/api/registry.py`, `backend/app/integrations/gguf.py`.
- **Roadmap:** E05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r62"></a>

## R62 — QUANTIZATION CALIBRATION

- **Classification:** 4. Missing.
- **Code finding:** No calibration dataset/version/count/sequence/seed workflow.
- **Evidence:** `backend/app/api/registry.py`.
- **Roadmap:** E06 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r63"></a>

## R63 — OLLAMA MODELFILE EXPORT

- **Classification:** 3. Partially implemented.
- **Code finding:** Ollama CLI import writes FROM-only temporary Modelfile; no portable template/stops/defaults export artifact.
- **Evidence:** `backend/app/serving/providers.py`.
- **Roadmap:** E07 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r64"></a>

## R64 — SERVING PROVIDER REGISTRY

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** ModelServingProvider registry has Transformers/vLLM/Ollama and now returns one typed capability contract plus legacy operation arrays. Durable lifecycle identity, provider-specific advanced settings, and normalized persisted health/config remain incomplete.
- **Evidence:** `backend/app/serving/providers.py`.
- **Roadmap:** E01, E02 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r65"></a>

## R65 — SGLANG

- **Classification:** 4. Missing.
- **Code finding:** No SGLang registration, worker, configuration or dependencies.
- **Evidence:** `backend/app/serving/providers.py`.
- **Roadmap:** E03 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r66"></a>

## R66 — VLLM CONTROLS

- **Classification:** 3. Partially implemented.
- **Code finding:** Compose vLLM and earlier warm manager coexist; memory/length settings partly present; no unified version-gated advanced controls.
- **Evidence:** `docker-compose.yml`, `backend/app/integrations/vllm_serve.py`.
- **Roadmap:** E02 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r67"></a>

## R67 — REWARD/SCORING API

- **Classification:** 4. Missing.
- **Code finding:** No reward/scoring API.
- **Evidence:** `backend/app/train_entry/serve.py`.
- **Roadmap:** E04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r68"></a>

## R68 — OPENAI-COMPATIBLE SERVING

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Managed /v1/models, chat, completions and chat SSE exist; provider generate API is simplified/nonstreaming; advanced tools/reasoning not normalized.
- **Evidence:** `backend/app/train_entry/serve.py`, `backend/app/serving/providers.py`.
- **Roadmap:** A09, E04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r69"></a>

## R69 — SERVING BENCHMARKS

- **Classification:** 4. Missing.
- **Code finding:** Simple test latency/token count and eval tokens/sec are not a persisted concurrency/percentile/TTFT serving benchmark.
- **Evidence:** `backend/app/api/serving.py`, `backend/app/train_entry/eval_run.py`.
- **Roadmap:** E08 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r70"></a>

## R70 — OPTIONAL EXTERNAL EXPERIMENT TRACKING

- **Classification:** 4. Missing.
- **Code finding:** Local tracking works; TRL report_to=none; no optional tracker adapters.
- **Evidence:** `backend/app/backends/unsloth_backend.py`.
- **Roadmap:** D08 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r71"></a>

## R71 — CALLBACK / EVENT INTEGRATION

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** The tolerant training union now includes progress/resource/artifact/warning/profile/error envelopes in addition to legacy events, and the runner/monitor handle their basic presentation. Existing producers, durable series, and eval event dictionaries are not fully unified.
- **Evidence:** `backend/app/core/events.py`, `backend/app/core/runner.py`, `backend/app/train_entry/eval_run.py`.
- **Roadmap:** B15 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r72"></a>

## R72 — UI ORGANIZATION

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** 11 lazy-loaded routes with shared shell, grouped nav, nested Data Lab and shared studios; alignment/export subviews pending.
- **Evidence:** `frontend/src/App.tsx`, `frontend/src/components/layout/nav.ts`.
- **Roadmap:** G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r73"></a>

## R73 — PROGRESSIVE DISCLOSURE

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Existing advanced collapse and recommendations; not full capability-derived Recommended/Advanced/Expert structure.
- **Evidence:** `frontend/src/components/studio/TrainingStudio.tsx`.
- **Roadmap:** D08 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r74"></a>

## R74 — RESOURCE ADVISOR

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Shared llmfit/fallback breakdown, safe budget and suggestions exist; no reference/distributed/exact module counts; estimates remain heuristic.
- **Evidence:** `backend/app/integrations/llmfit.py`, `backend/app/integrations/estimator.py`.
- **Roadmap:** B16, F01 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r75"></a>

## R75 — DATABASE ENHANCEMENTS

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** 12 domain tables plus Alembic history, additive migrations, backups, WAL/FKs/indexes implemented and baseline tested. New entities only for durable requirements.
- **Evidence:** `backend/app/db/models.py`, `backend/app/db/migrate.py`.
- **Roadmap:** A03, B12, E01, E05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r76"></a>

## R76 — REPRODUCIBILITY

- **Classification:** 3. Partially implemented.
- **Code finding:** Run config/hardware/packages/dataset hash and manifests exist; exact model/tokenizer commits, template identity, mixtures, effective engine not fully bound; manifest key redaction hides tokenizer fields.
- **Evidence:** `backend/app/core/reproducibility.py`, `backend/app/core/runner.py`.
- **Roadmap:** A02, A03, B12 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r77"></a>

## R77 — SECURITY

- **Classification:** 3. Partially implemented.
- **Code finding:** trust_remote_code defaults false; structured subprocess argv and limited config redaction exist. Raw stdout/log/error strings are not redacted globally; no demonstrated secret leak asserted.
- **Evidence:** `backend/app/config.py`, `backend/app/core/log_capture.py`, `backend/app/core/runner.py`.
- **Roadmap:** A16 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r78"></a>

## R78 — DEPENDENCY MANAGEMENT

- **Classification:** 3. Partially implemented.
- **Code finding:** ML stack remains optional for the API. One import-light registry now feeds model/backend capability data and diagnostics, including future optional package presence. Version compatibility and real runtime/hardware verification remain incomplete; GPU extra still bundles Unsloth.
- **Evidence:** `backend/pyproject.toml`, `backend/app/core/diagnostics.py`.
- **Roadmap:** B01, B04, B14, F04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r79"></a>

## R79 — DIAGNOSTICS

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Hardware, packages, CUDA subprocess, DB integrity, service/queue/activity diagnostics work; missing future optional features/cache accounting and truthful probe levels.
- **Evidence:** `backend/app/api/system.py`, `backend/app/core/diagnostics.py`.
- **Roadmap:** A16, B02, B16 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r80"></a>

## R80 — DOCUMENTATION

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Extensive guides exist but implementation-plan retains obsolete findings (no migrations/old extraction) and UI template hints conflict with code.
- **Evidence:** `docs/implementation-plan.md`, `frontend/src/pages/Datasets.tsx`.
- **Roadmap:** A16, G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r81"></a>

## R81 — TESTING

- **Classification:** 3. Partially implemented.
- **Code finding:** 78 backend tests and 9 frontend tests pass; mainly unit/contracts/mocks. No real tiny training/export/provider lifecycle regression matrix.
- **Evidence:** `backend/tests`, `frontend/src/lib`, `frontend/src/components/ErrorPanel.test.tsx`.
- **Roadmap:** A16, B16, C08, D08, F04, G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r82"></a>

## R82 — LLAMA-FACTORY INTEGRATION TESTS

- **Classification:** 4. Missing.
- **Code finding:** No optional LLaMA-Factory integration tests or worker.
- **Evidence:** `backend/tests`, `backend/app/backends/base.py`.
- **Roadmap:** B14, B15 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r83"></a>

## R83 — MODEL/BACKEND FALLBACK

- **Classification:** 3. Partially implemented.
- **Code finding:** Fallback inside loader exists and is logged. An exact-match BackendSelector now normalizes explicit registry selection, but there is no Auto resolver, full compatibility evidence, or durable effective-engine identity.
- **Evidence:** `backend/app/backends/unsloth_backend.py`, `backend/app/api/runs.py`.
- **Roadmap:** B01, B04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r84"></a>

## R84 — DO NOT blindly COPY LLAMA-FACTORY

- **Classification:** 1. Already implemented correctly.
- **Code finding:** No LLaMA-Factory code integration found; optional boundary preserved. Mandatory dependency or wholesale clone is category 6 (not appropriate).
- **Evidence:** `backend/app/backends/base.py`, `backend/pyproject.toml`.
- **Roadmap:** B14, G04 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r85"></a>

## R85 — DO NOT BREAK THE PREVIOUS IMPLEMENTATION

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Legacy handles/migrations and current baseline work; stronger runtime/migration compatibility coverage needed. Audit preserves pre-existing dirty tree.
- **Evidence:** `backend/tests/test_database_migrations.py`, `backend/app/model_refs.py`.
- **Roadmap:** A01, G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r86"></a>

## R86 — REQUIRED END-TO-END USER EXPERIENCE

- **Classification:** 3. Partially implemented.
- **Code finding:** Core import/prepare/train/eval/registry/serve surfaces exist; preference/tool/advanced flows and complete restart lineage are not delivered.
- **Evidence:** `frontend/src/App.tsx`, `backend/app/domain.py`.
- **Roadmap:** A09, C08, E08, G02, G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r87"></a>

## R87 — IMPLEMENTATION PRIORITY

- **Classification:** 1. Already implemented correctly.
- **Code finding:** Audit creates ordered A-G plan; no implementation phase started. Future completion gates remain pending.
- **Evidence:** `docs/implementation/IMPLEMENTATION_ROADMAP.md`.
- **Roadmap:** G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r88"></a>

## R88 — ENGINEERING QUALITY

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Typed core contracts, subprocesses, events, atomic data outputs exist; loose API dictionaries, duplicated policy, unbounded conversions and raw logs require hardening.
- **Evidence:** `backend/app/core`, `backend/app/api`, `backend/app/train_entry`.
- **Roadmap:** A08, A16, B03, F02 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r89"></a>

## R89 — ERROR HANDLING

- **Classification:** 2. Implemented but needs enhancement.
- **Code finding:** Actionable normalization/failure artifacts and ErrorPanel exist; substring classification can mislabel failures and guidance lacks complete runtime evidence.
- **Evidence:** `backend/app/core/errors.py`, `frontend/src/components/ErrorPanel.tsx`.
- **Roadmap:** A16, B02 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r90"></a>

## R90 — FINAL VERIFICATION

- **Classification:** 3. Partially implemented.
- **Code finding:** CPU/unit/build/lint/Compose config and temporary DB startup/restart checked. GPU SFT/CPT/alignment/export/serving and image builds not verified in audit.
- **Evidence:** `docs/implementation/STATUS.md`.
- **Roadmap:** E08, G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r91"></a>

## R91 — COMPLETION REPORT

- **Classification:** 3. Partially implemented.
- **Code finding:** Audit report delivered; final implemented-product completion report deferred until verified phase gates.
- **Evidence:** `docs/implementation/STATUS.md`.
- **Roadmap:** G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

<a id="r92"></a>

## R92 — FINAL PRODUCT GOAL

- **Classification:** 3. Partially implemented.
- **Code finding:** Application already orchestrates much of lifecycle; breadth and trustworthy lineage/capability parity remain incomplete.
- **Evidence:** `docs/implementation/CURRENT_STATE.md`.
- **Roadmap:** G05 (see task impact and acceptance details in IMPLEMENTATION_ROADMAP.md).

## Category 6 decisions

No whole requested section is rejected. These possible implementations conflict with the specification and current architecture:

- Mandatory LLaMA-Factory, replacement of SLM Kit orchestration, or copying its database/UI: reject; use optional backend bridge (B14/B15).
- Replacing SQLite/migrations with a new persistence stack for these requirements: reject; additive migrations suffice.
- A parallel tokenizer/template/dataset normalization or model-loading implementation per engine: reject; extend the existing owners.
- Advertising arbitrary multimodal/seq2seq models through the current causal-text trainer: reject; modality-specific experimental adapters need separate validated contracts (G01/G02).
- Enabling multi-process API workers against process-local GPU leases: reject until resource ownership is redesigned; distributed training remains optional within one control-plane process.

## Critical existing defects and limitations

- **F01:** Alpaca system/history silently lost; reproduced without a model. A01.
- **F02:** Mutable tokenizer revision cache reuse is unsafe; `if body.revision` does not distinguish a commit from main/tag. Fingerprint omits full tokenizer normalization/merge configuration. A02.
- **F03:** DatasetVersion references original files, but RunConfig/worker resolve Dataset paths with no launch-time version hash verification. Immutable metadata alone cannot guarantee immutable training input. A03.
- **F04:** Preflight returns backend_verified from CUDA availability without model loading/Unsloth import. B02.
- **F05:** Unsloth target defaults assume Llama projections while native PEFT uses all-linear. B05.
- **F06:** Evaluation role text is wrapped as a user prompt; inference can silently fall back and lose system/history. A08/A09.
- **F07:** Scratch ignores resume_from and saves weights without optimizer/RNG state. B09.
- **F08:** Training finalizer creates checkpoints/status only; no automatic final ModelArtifact or completion evaluation despite eval_on_completion flag. B11/B12.
- **F09:** Deployment is in-memory; external/warm vLLM paths coexist. E01/E02.
- **F10:** Manifest redaction matches any key containing token and removes useful tokenizer metadata; raw error/log text lacks equivalent secret redaction. A16.
- **F11:** Global GPU ownership is process-local; preflight uses only that lease, without the external-provider guard used by training/eval. Admission/cleanup need shared coverage. B02/F01.
- **F12:** Quality near-duplicate scan is heuristic and bounded by bucket; tokenizer unknown rates compare truncated counts with original-length totals. A15.

These findings predate audit edits. They are separate from failing test results (none in completed baseline suites).

