# Implementation status

Updated: 2026-09-15. Audited HEAD: `5d64b78b6dc5289ef419d22ab11cd629cdba6337` plus the pre-existing working tree.

## Current phase

Architecture foundation review complete. Phase A feature work has not started.

## Current task

FOUNDATION — shared capability, adapter, operation, dependency, selection, and event contracts completed and verified.

## Completed tasks

- Kept `TaskType` and method values compatible while adding `RunConfig.schema_version`, `TrainingStage`, and normalized operation identity.
- Added the import-light shared capability models and optional dependency registry in `backend/app/capabilities.py`.
- Added `TrainingBackendCapabilities`; built-in task/method/tokenizer/PEFT/quantization data now derives from backend descriptors. Estimate reports unavailable dependencies and create blocks them from the same descriptor.
- Added the exact-match `BackendSelector` boundary and used it in estimate/create/export paths. Automatic selection remains B04.
- Preserved `/api/runs/backends` legacy arrays and added structured capability fields for incremental frontend adoption.
- Added `DatasetAdapter` detect/validate/canonicalize/preview/stage contracts and registration behind the existing facade without adding formats.
- Made existing family metadata implement `ModelFamilyAdapter`; added structured model template, PEFT, and operation-specific quantization capabilities.
- Added typed serving-provider capabilities for Transformers, vLLM, and Ollama while retaining legacy status operation arrays.
- Extended the compatible event union with progress/resource/artifact/warning/profile/error envelopes; no backend emits them yet.
- Updated the training studio to derive backend and method availability from the backend API descriptor.
- Added focused foundation contract tests and updated architecture/current-state/roadmap/phase documentation.

No Phase A-G feature task is marked complete by this foundation review. A04, B01, B04, and E01 have completed interface slices but retain their feature acceptance work in the roadmap.

## In-progress tasks

None.

## Remaining tasks

- Phase A: A01-A18. A04's interface exists, but its phase acceptance remains open.
- Phase B: B01-B18. B01/B04 interfaces exist; runtime evidence, effective selection, training strategies, and persistence remain open.
- Phase C: C01-C08.
- Phase D: D01-D11.
- Phase E: E01-E08. E01's provider descriptor exists; durable deployment identity remains open.
- Phase F: F01-F04.
- Phase G: G01-G05.
- Follow the dependency graph in `IMPLEMENTATION_ROADMAP.md`.

## Known failures

### Existing code defects / incomplete behavior

The verified A01 defect remains: Alpaca-style instruction conversion drops top-level system/history. Other audited priorities remain unchanged, including mutable tokenizer-profile cache identity, unverified run dataset path identity, metadata-only preflight claims, assumed LoRA targets, inference/evaluation template drift, scratch resume, final artifact/evaluation automation, deployment persistence, and raw-log redaction.

### Existing baseline warnings

- Backend: one `StarletteDeprecationWarning` from the FastAPI TestClient/httpx integration.
- Frontend ESLint: `BaseModelPicker.tsx:62` omits `preflight` from a `useEffect` dependency list. ESLint exits successfully with one warning and zero errors.

### New failures caused by foundation changes

None observed. Focused tests, full backend tests, Ruff, frontend tests, TypeScript/build, and ESLint pass at the status recorded below.

## Architecture decisions made

- SLM Kit remains the orchestration/application layer.
- Existing `TrainingBackend`, dataset facade, model capability registry, and `ModelServingProvider` are extended rather than replaced.
- Backend descriptors are authoritative for backend task/method visibility and validation metadata; model compatibility remains a separate intersected input.
- Persisted task values stay stable; execution stages can expand without reinterpreting old runs.
- Dependency presence uses import-free probes and is evidence of installation only.
- Exact backend selection is implemented now; equivalence-aware automatic selection waits for B02/B03 evidence.
- New event types are additive and tolerant; emission/persistence/UI work remains in later tasks.
- LLaMA-Factory, SGLang, and other advanced integrations remain optional and unregistered.

See `ARCHITECTURE_DECISIONS.md` for ADR-01 through ADR-14.

## Database migration status

- Current packaged head remains `0003_dataset_versions`.
- No model/table/index/column changed and no migration was added.
- Existing projects, datasets, runs, checkpoints, artifacts, evaluations, and deployments retain their stored schema.
- Old `RunConfig` JSON without `schema_version` loads with version 1; the derived stage/operation is not serialized as a duplicate field.

## Backend test status

Environment: Windows installed Python 3.13; the checkout's `.venv` has a non-Windows layout.

- Focused architecture/data/model/provider tests: `python -m pytest backend/tests/test_foundation_contracts.py backend/tests/test_dataset_adapters.py backend/tests/test_model_capabilities.py backend/tests/test_model_preflight.py backend/tests/test_serving_providers.py -q` — **56 passed**, 1 existing warning.
- Focused foundation file: `python -m pytest backend/tests/test_foundation_contracts.py -q` — **6 passed**.
- Backend lint: `python -m ruff check backend/app backend/tests --output-format concise` — **passed**.
- Full backend suite: `python -m pytest -q` from `backend` — **84 passed**, 1 existing warning, 9.30s.
- No GPU training, real model download, merge/export, or provider lifecycle test was performed.

## Frontend build status

- `npm run build` — **passed**; TypeScript `tsc --noEmit` and Vite production build completed (2518 modules).
- `npm run lint` — **passed with one existing warning**, zero errors.
- Frontend unit tests: `npm test -- --run` — **3 files / 9 tests passed**.
- No browser end-to-end suite exists.

## Blocking issues

No blocker to A01. Runtime capability certification and automatic backend selection require optional ML dependencies, representative tiny models, and appropriate GPU test environments in later tasks.

## Recommended next task

**A01 — Preserve Alpaca system and history.**

Extend the registered instruction adapter in `backend/app/datasets/adapters.py` and its regression fixtures in `backend/tests/test_dataset_adapters.py`. Preserve optional system, ordered history pairs, current instruction/input, and final output; reject malformed system/history; preserve Unicode, whitespace, input immutability, and existing pair behavior. No database migration or new normalization layer is needed.
