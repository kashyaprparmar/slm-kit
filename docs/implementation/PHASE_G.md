# Phase G — Experimental capabilities and final verification

**Status: Not started.** Audit only. Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### G01 — Extend modality schema while retaining text safety

- [ ] Not started. Requirements: R14.
- Depends on: A07, B01.
- Change: Validate media references and declare per-modality capabilities; no implicit text conversion.
- Acceptance: Text v1 compatible; missing media rejected; unavailable multimodal training cannot launch.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap G01 for database/API/frontend/test/risk impact.

### G02 — Add one experimental multimodal execution adapter

- [ ] Not started. Requirements: R14, R86.
- Depends on: G01, E01.
- Change: One explicitly selected tested model/processor pair behind capabilities.
- Acceptance: Tiny media job and reload pass; unsupported modalities/families stay disabled.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap G02 for database/API/frontend/test/risk impact.

### G03 — Add expert PPO orchestration

- [ ] Not started. Requirements: R44.
- Depends on: C07, C08, F01.
- Change: Reward/reference/policy/buffer/KL/whitening config behind explicit expert gate.
- Acceptance: Bounded rollout tiny job, cancellation and lineage tested without affecting SFT.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap G03 for database/API/frontend/test/risk impact.

### G04 — Evaluate OFT/QOFT and architecture transformations

- [ ] Not started. Requirements: R26, R84.
- Depends on: D01, G01.
- Change: Feasibility spike per method; no generic unchecked patch hook; keep unsupported unless validated.
- Acceptance: Method-specific train/save/reload and license review before enablement; rejected proposals documented.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap G04 for database/API/frontend/test/risk impact.

### G05 — Final lifecycle verification and completion report

- [ ] Not started. Requirements: R80, R81, R85, R86, R87, R90, R91, R92.
- Depends on: G02, G03, G04, E08, F03, F04.
- Change: Run supported workflow matrix including restart/data/train/eval/export/serve and optional absence.
- Acceptance: Report implemented/partial/experimental/missing with evidence; no untested runtime called stable.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap G05 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Scope controls

Missing experimental features are not already implemented. Limit modalities/model families explicitly; preserve ordinary text SFT and local operation. Final report distinguishes implemented, partial, experimental and missing.

