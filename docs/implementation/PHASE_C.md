# Phase C — Alignment

**Status: Partially Complete.** Step 16 reviewed the native alignment paths and fixed defects. Offline tiny jobs now train/save/reload DPO, IPO, ORPO, SimPO, KTO, and reward models on the recorded CPU stack. Durable immutable reference pinning (C01) and the base-versus-trained regression gate (C08) remain open. GPU and optional-engine acceptance are unverified. No Phase D optimization or PPO work began.

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### C01 — Define alignment/reference config

- [~] Partially complete. Base/separate/adapter-disabled strategies are exercised; app references resolve through `model_refs.py`; separate adapter references retain their adapter; vocabulary/token-ID mismatches fail. Reference-free objectives allocate no second model. Actual available config commits enter artifact lineage, but durable immutable pinning before enqueue/resume remains open. Requirements: R04, R39, R40, R41, R42, R43.
- Depends on: A06, B01, B12.
- Change: Objective-specific validation and base/separate/adapter-disabled reference strategies.
- Acceptance: Invalid objective/data/reference combinations fail; reference memory included and lineage pinned.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C01 for database/API/frontend/test/risk impact.

### C02 — Execute DPO through shared trainer strategy

- [~] CPU acceptance passed; broader runtime matrix remains unverified. Tiny DPO jobs exercise sigmoid, hinge, robust, and EXO-pair losses; assert all six preference metrics, finite telemetry, correct reward-margin arithmetic, and reloadable output. Requirements: R39.
- Depends on: C01, B03.
- Change: TRL DPO beta/loss/smoothing and reference selection with installed-version checks.
- Acceptance: Tiny paired dataset trains; rewards/logprobs captured; native SFT unchanged.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C02 for database/API/frontend/test/risk impact.

### C03 — Add IPO objective

- [~] CPU acceptance passed; broader runtime matrix remains unverified. IPO reuses `DPOTrainer` with `loss_type=ipo`, rejects irrelevant DPO options/smoothing, and trains/saves/reloads the tiny model. Requirements: R40.
- Depends on: C02.
- Change: IPO objective strategy reuses preference pipeline.
- Acceptance: Tiny IPO job uses intended loss; invalid smoothing/options rejected.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C03 for database/API/frontend/test/risk impact.

### C04 — Add ORPO objective

- [~] CPU acceptance passed; broader runtime matrix remains unverified. ORPO trains/saves/reloads through the shared lifecycle and emits reference-free lineage. Requirements: R40.
- Depends on: C02.
- Change: ORPO trainer adapter sharing canonical inputs/events.
- Acceptance: Tiny ORPO job passes; reference-free behavior and losses verified.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C04 for database/API/frontend/test/risk impact.

### C05 — Add SimPO objective

- [~] CPU acceptance passed; broader runtime matrix remains unverified. SimPO trains/saves/reloads through CPO with `cpo_alpha=0`, preventing an unintended SFT term. An installed-trainer loss fixture verifies beta/gamma margin arithmetic. Prompt length is bounded below the run context. Requirements: R40.
- Depends on: C02.
- Change: Version-gated SimPO implementation without separate engine.
- Acceptance: Objective math fixture and tiny job pass; unavailable runtime honestly gated.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C05 for database/API/frontend/test/risk impact.

### C06 — Execute KTO and balance diagnostics

- [~] CPU acceptance passed; broader runtime matrix remains unverified. Both KTO classes train through the shared lifecycle. Non-boolean labels, missing classes, and malformed records fail; imbalanced classes warn. Preparation revalidates the actual data rather than trusting stored validation. Requirements: R41.
- Depends on: C01.
- Change: KTO trainer strategy with desirable/undesirable weighting.
- Acceptance: Both label classes retained; malformed labels fail; tiny job produces artifact.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C06 for database/API/frontend/test/risk impact.

### C07 — Train and evaluate reward models

- [~] Full/LoRA/DoRA CPU train/save/reload acceptance passed; QLoRA and broader runtime matrix remain unverified. Registered reward adapters produce repeatable scalar scores after reload. PEFT classification heads train/save in full and are excluded from LoRA targets. Reward-adapter continuation into full tuning retains the head; incompatible causal adapters fail explicitly. Pairwise metric contracts and generation/serving exclusions pass. Requirements: R42.
- Depends on: C01.
- Change: Sequence-score loader/trainer and pairwise evaluation separate from causal generation.
- Acceptance: Tiny reward job saves/reloads numeric scores and lineage; generation disallowed for reward artifacts.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C07 for database/API/frontend/test/risk impact.

### C08 — Alignment telemetry and regression gate

- [~] Partially complete. Installed objectives emit finite telemetry; DPO reward-margin arithmetic and SimPO objective math are tested. Adapter-disabled DPO restores a complete optimizer checkpoint and advances from step 1 to step 2. Graceful stop retains a checkpoint without a completed artifact; real worker descendants terminate under hard cancellation. Incomplete native alignment saves are excluded from automatic resume. Base-versus-trained regression evaluation and exact interrupted-resume equivalence remain open. Requirements: R49, R81, R86.
- Depends on: C02, C03, C04, C05, C06, C07.
- Change: Normalize chosen/rejected rewards, margins, accuracy, logprob/KL by objective.
- Acceptance: Replay fixtures render missing metrics safely; installed objectives tested; base/trained comparison regression passes.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap C08 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Step 16 review evidence — 2026-09-16

- Runtime fixtures: `backend/tests/test_alignment_runtime.py`, marked `alignment_runtime`, skip when the optional ML stack is absent. Models/tokenizers are created locally; no model downloads are required. Datasets cache entries include the source content fingerprint and reject mutation while generating.
- Reviewed stack: Python 3.11, Torch 2.7.1+cpu, Transformers 4.52.4, TRL 0.19.1, PEFT 0.15.2, Datasets 3.6.0, Accelerate 1.7.0. This isolated CPU fixture stack does **not** validate the packaged GPU extra's Torch >=2.11,<2.12 range.
- CPU contracts and full-suite results are recorded in STATUS.md. Frontend tests exercise objective-specific visibility, inactive-option payload cleanup, and absent/unsupported capability gates.
- Auto resolution now checks DPO loss and reference strategy from authoritative backend capabilities. Unknown frontend capabilities are unavailable. Native alignment metadata is bounded to the declared TRL >=0.9,<0.24 range; one tested version does not prove the entire range.
- Optional LLaMA-Factory maps DPO smoothing and KTO class weights, omits the invalid KTO `pref_loss`, rejects unmapped separate references and DPO losses, and avoids final artifact registration after cancellation. Its process was not run. Mapping was checked against [upstream v0.9.4 arguments](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.4/src/llamafactory/hparams/finetuning_args.py).
- No database migration. Existing config/container schemas remain in use; invalid reference combinations now fail rather than allowing TRL to select different semantics implicitly.
- Next: **C01 — durably pin effective policy/reference identities before launch and preserve them on resume**, followed by C08's base-versus-trained regression gate. Keep Phase D paused.

## Scope controls

Do not create a separate training engine per objective. Reference-free objectives must not allocate unnecessary reference models. Reward scoring is distinct from generation.
