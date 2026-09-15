# Phase C — Alignment

**Status: Not started.** Audit only. Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### C01 — Define alignment/reference config

- [ ] Not started. Requirements: R04, R39, R40, R41, R42, R43.
- Depends on: A06, B01, B12.
- Change: Objective-specific validation and base/separate/adapter-disabled reference strategies.
- Acceptance: Invalid objective/data/reference combinations fail; reference memory included and lineage pinned.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C01 for database/API/frontend/test/risk impact.

### C02 — Execute DPO through shared trainer strategy

- [ ] Not started. Requirements: R39.
- Depends on: C01, B03.
- Change: TRL DPO beta/loss/smoothing and reference selection with installed-version checks.
- Acceptance: Tiny paired dataset trains; rewards/logprobs captured; native SFT unchanged.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C02 for database/API/frontend/test/risk impact.

### C03 — Add IPO objective

- [ ] Not started. Requirements: R40.
- Depends on: C02.
- Change: IPO objective strategy reuses preference pipeline.
- Acceptance: Tiny IPO job uses intended loss; invalid smoothing/options rejected.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C03 for database/API/frontend/test/risk impact.

### C04 — Add ORPO objective

- [ ] Not started. Requirements: R40.
- Depends on: C02.
- Change: ORPO trainer adapter sharing canonical inputs/events.
- Acceptance: Tiny ORPO job passes; reference-free behavior and losses verified.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C04 for database/API/frontend/test/risk impact.

### C05 — Add SimPO objective

- [ ] Not started. Requirements: R40.
- Depends on: C02.
- Change: Version-gated SimPO implementation without separate engine.
- Acceptance: Objective math fixture and tiny job pass; unavailable runtime honestly gated.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C05 for database/API/frontend/test/risk impact.

### C06 — Execute KTO and balance diagnostics

- [ ] Not started. Requirements: R41.
- Depends on: C01.
- Change: KTO trainer strategy with desirable/undesirable weighting.
- Acceptance: Both label classes retained; malformed labels fail; tiny job produces artifact.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C06 for database/API/frontend/test/risk impact.

### C07 — Train and evaluate reward models

- [ ] Not started. Requirements: R42.
- Depends on: C01.
- Change: Sequence-score loader/trainer and pairwise evaluation separate from causal generation.
- Acceptance: Tiny reward job saves/reloads numeric scores and lineage; generation disallowed for reward artifacts.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C07 for database/API/frontend/test/risk impact.

### C08 — Alignment telemetry and regression gate

- [ ] Not started. Requirements: R49, R81, R86.
- Depends on: C02, C03, C04, C05, C06, C07.
- Change: Normalize chosen/rejected rewards, margins, accuracy, logprob/KL by objective.
- Acceptance: Replay fixtures render missing metrics safely; installed objectives tested; base/trained comparison regression passes.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C08 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Scope controls

Do not create a separate training engine per objective. Reference-free objectives must not allocate unnecessary reference models. Reward scoring is distinct from generation.

