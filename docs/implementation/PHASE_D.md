# Phase D — Optimization and profiling

**Status: Partially complete.** Step 17 established the authoritative optimization registry and enabled only the native runtime paths verified by tests. Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### D01 — Version-gate advanced PEFT strategies

- [~] Partial. Requirements: R26.
- Depends on: B05, B07.
- Change: Registry now reports rsLoRA, LoRA+, PiSSA, LoftQ, EVA, OFT, and QOFT with installed version, compatibility, JSON control schema, and expected impact. Native rsLoRA, PiSSA, and LoRA+ are enabled after an offline PEFT 0.15.2 tiny train/step fixture. LoftQ, EVA, OFT, and QOFT stay unavailable because their joint calibration/quantization or loader paths are not yet verified.
- Acceptance: Each enabled strategy has version/config compatibility tests and tiny train/save/reload evidence.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D01 for database/API/frontend/test/risk impact.

### D02 — Add optimizer strategy registry

- [~] Partial. Requirements: R30.
- Depends on: B01.
- Change: Added one import-light registry with shared schema, validation, dependency/version reporting, and estimator hook for GaLore, APOLLO, BAdam, Adam-mini, and Muon. No external optimizer executes until a package-specific adapter has a finite-gradient/checkpoint fixture.
- Acceptance: Default optimizer unchanged; absent package rejects; GaLore tiny step and estimate evidence pass.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap D02 for database/API/frontend/test/risk impact.

### D03 — Add APOLLO optimizer adapter

- [ ] Deferred. The optional package is absent; its registry descriptor reports actionable installation guidance and remains unavailable.
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

- [~] Partial. Requirements: R32, R35.
- Depends on: D04.
- Change: NEFTune is a native, capability-gated `runtime.neftune_noise_alpha` option and is passed through only when the installed trainer config accepts it. Liger and FlashAttention are represented in the registry; their packages are absent in the recorded environment, so neither is exposed as runnable.
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

## Step 17 evidence

- New authoritative contracts: `app.capabilities.OptimizationCapability`, `TrainingBackendCapabilities.optimizations`, and `app.optimizations`.
- API and frontend consume the same descriptor, including availability, installed version, compatibility, field schema, disclosure level, and estimated impact. Existing `optional_features` aliases remain for older clients.
- `RunConfig` changes are additive: `runtime.neftune_noise_alpha` plus typed `optim.strategy`, target-module, rank, and update-interval fields. No database migration is needed because runs persist versioned JSON configuration.
- Native PEFT maps PiSSA to `LoraConfig(init_lora_weights="pissa")`; LoRA+ uses PEFT's optimizer factory. Both are covered by an offline GPT-2-sized CPU fixture. NEFTune is passed only after checking the installed SFT/alignment config signature.
- Validation rejects incompatible selections before worker launch. PiSSA is currently unquantized LoRA/DoRA only, and LoRA+ requires native Transformers plus `adamw_torch`.
- Verification on 2026-09-17: CPU ML focused backend suite **14 passed**; full lean backend suite **149 passed, 3 skipped**; frontend suite **18 passed**; frontend build passed. No GPU, bitsandbytes, FlashAttention, Liger, GaLore, APOLLO, BAdam, Adam-mini, Muon, or LLaMA-Factory runtime was available for execution tests.
