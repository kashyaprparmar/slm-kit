# Phase D — Optimization and profiling

**Status: Not started.** Audit only. Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### D01 — Version-gate advanced PEFT strategies

- [ ] Not started. Requirements: R26.
- Depends on: B05, B07.
- Change: rsLoRA, LoRA+, PiSSA, LoftQ, EVA strategy descriptors; enable each only after its fixture/runtime test.
- Acceptance: Each enabled strategy has version/config compatibility tests and tiny train/save/reload evidence.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D01 for database/API/frontend/test/risk impact.

### D02 — Add optimizer strategy registry

- [ ] Not started. Requirements: R30.
- Depends on: B01.
- Change: Schema/validation/estimation hooks; first adapter GaLore, other methods remain gated.
- Acceptance: Default optimizer unchanged; absent package rejects; GaLore tiny step and estimate evidence pass.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D02 for database/API/frontend/test/risk impact.

### D03 — Add APOLLO optimizer adapter

- [ ] Not started. Requirements: R30.
- Depends on: D02.
- Change: Version-gated APOLLO configuration, adapter and memory estimate.
- Acceptance: APOLLO missing-dependency/config/finite-gradient/checkpoint tests pass before exposed.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D03 for database/API/frontend/test/risk impact.

### D04 — Resolve attention and checkpointing modes

- [ ] Not started. Requirements: R31, R33.
- Depends on: B02, B07.
- Change: Auto/SDPA/FA2/eager plus off/standard/non-reentrant/optimized checkpointing.
- Acceptance: Hardware/model/dtype mismatch uses allowed fallback; explicit incompatible requests fail.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D04 for database/API/frontend/test/risk impact.

### D05 — Add Liger and NEFTune options

- [ ] Not started. Requirements: R32, R35.
- Depends on: D04.
- Change: Optional kernel/noise strategies with installed-runtime gates.
- Acceptance: Standard path unchanged; enabled options recorded and tiny loss/gradient tests pass.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D05 for database/API/frontend/test/risk impact.

### D06 — Add model-aware RoPE policy

- [ ] Not started. Requirements: R34, R56.
- Depends on: B05, D04.
- Change: Default/linear/dynamic/YaRN schema per family and runtime.
- Acceptance: No silent config mutation; incompatible family/mode rejected; effective context persisted.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D06 for database/API/frontend/test/risk impact.

### D07 — Add opt-in profiler artifacts

- [ ] Not started. Requirements: R46, R47.
- Depends on: B16.
- Change: PyTorch schedule and optional module timing with bounded trace storage.
- Acceptance: Off has no profiling hooks; CPU trace saved; CUDA test gated; cleanup on cancel.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D07 for database/API/frontend/test/risk impact.

### D08 — Add external tracking adapters and phase gate

- [ ] Not started. Requirements: R70, R73, R81.
- Depends on: A16, B16, D01, D02, D03, D04, D05, D06, D07, D09, D10, D11.
- Change: Optional TensorBoard/W&B/MLflow/SwanLab callback sinks; local events remain authoritative.
- Acceptance: Tracker failures cannot lose local history; secret tests pass; optimization regression suite passes.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D08 for database/API/frontend/test/risk impact.

### D09 — Add BAdam optimizer adapter

- [ ] Not started. Requirements: R30.
- Depends on: D02.
- Change: Version-gated BAdam configuration, adapter and memory estimate.
- Acceptance: BAdam missing-dependency/config/finite-gradient/checkpoint tests pass before exposed.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D09 for database/API/frontend/test/risk impact.

### D10 — Add Adam-mini optimizer adapter

- [ ] Not started. Requirements: R30.
- Depends on: D02.
- Change: Version-gated Adam-mini configuration, adapter and memory estimate.
- Acceptance: Adam-mini missing-dependency/config/finite-gradient/checkpoint tests pass before exposed.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D10 for database/API/frontend/test/risk impact.

### D11 — Add Muon optimizer adapter

- [ ] Not started. Requirements: R30.
- Depends on: D02.
- Change: Version-gated Muon configuration, adapter and memory estimate.
- Acceptance: Muon missing-dependency/config/finite-gradient/checkpoint tests pass before exposed.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D11 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Scope controls

Enable each optional optimization only after its own compatibility/runtime tests. D09-D11 precede D08. Keep default SFT unchanged.

