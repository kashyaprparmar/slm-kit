# Phase F — Scale

**Status: Not started.** Audit only. Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### F01 — Add device-aware distributed config and admission

- [ ] Not started. Requirements: R50, R74.
- Depends on: B16, E01.
- Change: World/ranks/master/strategy and atomic device-set leases; retain one control process.
- Acceptance: Single-GPU unchanged; overlapping allocations blocked; all ranks cancelled together.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap F01 for database/API/frontend/test/risk impact.

### F02 — Add DDP launcher and process-tree recovery

- [ ] Not started. Requirements: R50, R88.
- Depends on: F01.
- Change: torchrun supervision and rank-zero event aggregation.
- Acceptance: Two-device smoke, rank failure and restart cleanup pass; no orphan ranks.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap F02 for database/API/frontend/test/risk impact.

### F03 — Add optional FSDP

- [ ] Not started. Requirements: R51.
- Depends on: F02.
- Change: FSDP full tuning then separately gate QLoRA combinations.
- Acceptance: Multi-GPU save/resume and memory evidence; untested FSDP+QLoRA remains disabled.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap F03 for database/API/frontend/test/risk impact.

### F04 — Add optional DeepSpeed and scale gate

- [ ] Not started. Requirements: R52, R78, R81.
- Depends on: F02.
- Change: Explicit ZeRO-2/3 dependency/config adapters.
- Acceptance: No automatic dependency activation; two-device train/resume/cancel checks pass per strategy.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap F04 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Scope controls

Keep one API control-plane process and single-GPU defaults. Multi-device validation requires actual suitable hardware; mocks alone cannot establish FSDP/DeepSpeed support.

