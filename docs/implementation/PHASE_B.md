# Phase B — Core training and optional backends

**Status: Feature work not started; shared capability and selection interfaces established.** Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### B01 — Add authoritative backend and stage descriptors

Foundation note: backend/stage/method/tokenizer/PEFT/quantization descriptors and frontend consumption now exist. Runtime evidence and capability snapshots remain open.

- [ ] Not started. Requirements: R02, R04, R26, R78, R83.
- Depends on: A16.
- Change: Separate stage/method/engine capabilities and installed compatibility evidence.
- Acceptance: Missing runtime cannot be advertised runnable; legacy payloads retained; unknown backend rejected.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B01 for database/API/frontend/test/risk impact.

### B02 — Make preflight evidence truthful

- [ ] Not started. Requirements: R57, R79, R89.
- Depends on: B01.
- Change: Distinguish metadata-only from real isolated loader/forward/backward probe; share model-load policy.
- Acceptance: CUDA presence alone never verifies Unsloth; dependency/loader failures block requested operation; leases released.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B02 for database/API/frontend/test/risk impact.

### B03 — Extract shared native training orchestration

- [ ] Not started. Requirements: R02, R56, R88.
- Depends on: B01.
- Change: Native and Unsloth loader strategies reuse dataset/callback/trainer code; factor revision/tokenizer policy without heavy API imports.
- Acceptance: Native tiny SFT runs with Unsloth absent; optimized engine selection preserves requested semantics.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B03 for database/API/frontend/test/risk impact.

### B04 — Resolve backend selection before launch

Foundation note: exact registered-backend selection now uses a typed interface. Automatic equivalence-aware selection and persisted reasons remain open.

- [ ] Not started. Requirements: R83, R78.
- Depends on: B02, B03.
- Change: Auto plan checks operation/model/hardware/dependencies; revalidate in worker.
- Acceptance: Unsupported Unsloth selects native only when equivalent; runtime fallback recorded; strict request honored.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B04 for database/API/frontend/test/risk impact.

### B05 — Discover adapter and freeze targets

- [ ] Not started. Requirements: R05, R27, R56.
- Depends on: B03.
- Change: Architecture module inspection; auto/all-linear/attention/MLP/custom targets; trainable counts.
- Acceptance: Non-Llama models work; empty or invalid selections rejected; estimates use actual matched modules.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B05 for database/API/frontend/test/risk impact.

### B06 — Implement freeze tuning execution

- [ ] Not started. Requirements: R05, R04.
- Depends on: B05.
- Change: Last N layers, embeddings/head/norm/selected modules with explicit trainability.
- Acceptance: Only selected parameters change in tiny step; unsupported architectures rejected.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B06 for database/API/frontend/test/risk impact.

### B07 — Expose precision and quantization policy

- [ ] Not started. Requirements: R28, R29, R36.
- Depends on: B01, B03.
- Change: Auto/bf16/fp16/fp32 and NF4/FP4/8bit compute/storage/nested controls gated by runtime.
- Acceptance: Unsupported dtype/quant combos fail before launch; effective settings match saved config.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B07 for database/API/frontend/test/risk impact.

### B08 — Verify continued pretraining semantics

- [ ] Not started. Requirements: R37.
- Depends on: A12, B03.
- Change: Raw causal-LM path with explicit EOS/tokenization and packing parity; wire validation set.
- Acceptance: Tiny continued-pretraining step uses no chat template; validation loss uses held-out split.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B08 for database/API/frontend/test/risk impact.

### B09 — Restore exact scratch checkpoint resume

- [ ] Not started. Requirements: R38.
- Depends on: A03.
- Change: Save/restore optimizer, RNG, step and data cursor; explicit resume endpoint.
- Acceptance: Interrupted seeded run matches uninterrupted steps; old artifacts remain loadable.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B09 for database/API/frontend/test/risk impact.

### B10 — Add scratch tokenizer and architecture options

- [ ] Not started. Requirements: R20, R38.
- Depends on: A11, B09.
- Change: Reuse/import/train tokenizer modes; optional HF from-config initialization behind capability.
- Acceptance: Tiny supported architecture trains and reloads with chosen tokenizer; unsupported configs fail.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B10 for database/API/frontend/test/risk impact.

### B11 — Add validation and early stopping

- [ ] Not started. Requirements: R45, R48.
- Depends on: A03, B03.
- Change: Evaluation cadence/metric/direction/patience/threshold; wire or reject eval_on_completion.
- Acceptance: Synthetic plateau stops at expected step and persists reason; no train/eval overlap.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B11 for database/API/frontend/test/risk impact.

### B12 — Register final artifacts and lineage

- [ ] Not started. Requirements: R58, R59, R75, R76.
- Depends on: A03, B04.
- Change: Idempotent final artifact registration, exact revisions/environment and generated model card.
- Acceptance: Successful run creates one artifact; retry duplicates prevented; existing run references preserved.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B12 for database/API/frontend/test/risk impact.

### B13 — Preserve complete configuration import/export roundtrip

- [ ] Not started. Requirements: R53.
- Depends on: B01, B12.
- Change: JSON/YAML import/export through same RunConfig; retain fields outside flattened form.
- Acceptance: Full config roundtrip equal; invalid YAML/config rejected; CLI/API yield same validation.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B13 for database/API/frontend/test/risk impact.

### B14 — Add optional LLaMA-Factory discovery/config adapter

- [ ] Not started. Requirements: R03, R78, R82, R84.
- Depends on: B01, B03.
- Change: Version-qualified optional registration and native config translation; install guidance.
- Acceptance: Absent package does not affect startup/native training; unsupported options rejected; no mandatory dependency.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B14 for database/API/frontend/test/risk impact.

### B15 — Bridge LLaMA-Factory worker lifecycle

- [ ] Not started. Requirements: R03, R71, R82.
- Depends on: B12, B14.
- Change: Launch optional engine within supervised process tree; map metrics/checkpoints/final/cancel/errors.
- Acceptance: Contract test covers logs, failure, cancellation and lineage; optional tiny installed-runtime job passes.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B15 for database/API/frontend/test/risk impact.

### B16 — Expand resource and telemetry evidence

- [ ] Not started. Requirements: R48, R74, R79, R81.
- Depends on: B05, B07, B11, B12, B15, B13, B17, B18.
- Change: Actual vs estimated token rates, resource samples, throttled DB writes and refined counts.
- Acceptance: Metrics rate bounded; measured and estimated values labeled; phase-B native and optional regression checks pass.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B16 for database/API/frontend/test/risk impact.

### B17 — Add thin user-facing CLI

- [ ] Not started. Requirements: R54.
- Depends on: B13.
- Change: Train/chat/evaluate/inspect commands call shared services and RunConfig validation.
- Acceptance: CLI and API use same validation; errors yield nonzero exit; no duplicated training logic.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B17 for database/API/frontend/test/risk impact.

### B18 — Introduce model source resolution interface

- [ ] Not started. Requirements: R55.
- Depends on: B03.
- Change: Extract HF/local source adapters behind existing resolver; ModelScope descriptor unavailable until implemented.
- Acceptance: Existing HF/local/run references unchanged; unknown provider rejected; no dependency on ModelScope.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap B18 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Scope controls

Preserve existing native/Unsloth training paths. LLaMA-Factory remains optional and version-qualified. B17/B18 precede the B16 phase gate. Metadata preflight is not a verified training run.
