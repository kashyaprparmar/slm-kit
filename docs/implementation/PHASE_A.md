# Phase A — Data, identity and templates

**Status: Feature work not started; shared adapter interface established.** Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Audit complete; preserve pre-existing working tree. Begin A01 only on a subsequent implementation request.

## Independently verifiable tasks

### A01 — Preserve Alpaca system and history

- [ ] Not started. Requirements: R07, R06, R85.
- Depends on: completed audit.
- Change: Extend canonicalize pair conversion; reject malformed history explicitly.
- Acceptance: System precedes ordered history then current instruction/input and answer; Unicode preserved; existing pairs unchanged.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A01 for database/API/frontend/test/risk impact.

### A02 — Resolve mutable tokenizer revisions safely

- [ ] Not started. Requirements: R19, R20, R76.
- Depends on: completed audit.
- Change: Resolve branch/tag to immutable commit before cache lookup; fingerprint serialized tokenizer behavior.
- Acceptance: Moving branch changes invalidate profiles; identical pinned commits reuse; local tokenizer changes invalidate.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A02 for database/API/frontend/test/risk impact.

### A03 — Bind runs to immutable dataset versions

- [ ] Not started. Requirements: R16, R75, R76.
- Depends on: completed audit.
- Change: Snapshot version/path/hash and verify before worker reads; retain legacy dataset_id.
- Acceptance: Changed or deleted source fails explicitly; old runs deserialize; worker consumes selected version.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A03 for database/API/frontend/test/risk impact.

### A04 — Introduce adapter registration behind canonicalize

Foundation note: the registry and detect/validate/canonicalize/preview/stage interface now exist. A04 remains open until A01 semantics and the phase acceptance suite pass.

- [ ] Not started. Requirements: R06, R08, R09.
- Depends on: A01.
- Change: Preserve facade; adapter detect/validate/canonicalize/preview/stages descriptors.
- Acceptance: Current text/pairs/OpenAI/ShareGPT roundtrip identically; ambiguous detection still fails.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A04 for database/API/frontend/test/risk impact.

### A05 — Expand reusable field mapping

- [ ] Not started. Requirements: R09, R10.
- Depends on: A04.
- Change: Support nested conversation role/content/system/tools mappings.
- Acceptance: Saved recipe replays custom ShareGPT fields without dropped context; conflicting mappings fail.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A05 for database/API/frontend/test/risk impact.

### A06 — Add canonical preference and KTO records

- [ ] Not started. Requirements: R11, R12, R14.
- Depends on: A04.
- Change: Validate paired/conversational preferences and strict boolean desirability; keep training gated.
- Acceptance: Missing/equal responses and invalid roles/labels fail; KTO never becomes ordinary prompt-response SFT.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A06 for database/API/frontend/test/risk impact.

### A07 — Add tool-call canonical validation

- [ ] Not started. Requirements: R13, R09.
- Depends on: A04.
- Change: Validate function schema, JSON args, unique IDs, matching responses and role sequence.
- Acceptance: Malformed calls and unmatched responses fail; valid tool payload survives roundtrip; training remains gated.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A07 for database/API/frontend/test/risk impact.

### A08 — Centralize template resolution

- [ ] Not started. Requirements: R21, R56, R88.
- Depends on: A04.
- Change: Shared resolution for train/eval/serve; native then compatible family then explicit warned fallback; explicit override policy.
- Acceptance: Same messages resolve same template across consumers; no silent inference fallback or system loss.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A08 for database/API/frontend/test/risk impact.

### A09 — Preserve structured evaluation conversations

- [ ] Not started. Requirements: R22, R68, R86.
- Depends on: A08.
- Change: Carry canonical messages through generation and perplexity.
- Acceptance: All turns/system remain structured; no role-text rewrapping; base/trained inputs match.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A09 for database/API/frontend/test/risk impact.

### A10 — Complete loss policy and token preview contract

- [ ] Not started. Requirements: R22, R23, R24.
- Depends on: A08.
- Change: Add final/all assistant policies and template-specific reasoning masks; reuse renderer in Studio preview.
- Acceptance: Preview labels equal worker labels including multi-turn and reasoning fixtures; no global tag stripping.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A10 for database/API/frontend/test/risk impact.

### A11 — Add tokenizer import and mutation lifecycle

- [ ] Not started. Requirements: R20.
- Depends on: A02, A08.
- Change: Implement import/extend with immutable output and embedding resize; keep unsupported train mode rejected.
- Acceptance: Reloaded tokenizer/model vocab agrees; added tokens train/save correctly; no silent ignored tokens.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A11 for database/API/frontend/test/risk impact.

### A12 — Add supported packing modes

- [ ] Not started. Requirements: R18, R37.
- Depends on: A10.
- Change: Gate standard/auto/neat modes against installed trainer and mask support.
- Acceptance: Token/label boundaries preserved; incompatible neat/masked modes rejected before launch.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A12 for database/API/frontend/test/risk impact.

### A13 — Persist reusable tokenized training cache

- [ ] Not started. Requirements: R19.
- Depends on: A02, A03, A10, A12.
- Change: Atomic cache writes keyed by version/recipe/tokenizer/template/stage/tokens/mask/packing; lock concurrent builds.
- Acceptance: Any semantic input change misses cache; interrupted build never reusable; training reads cached tensors.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A13 for database/API/frontend/test/risk impact.

### A14 — Extend source/manual/stratified splits

- [ ] Not started. Requirements: R16.
- Depends on: A03, A04.
- Change: Source/manual/stratified split assignment with deterministic membership.
- Acceptance: Same inputs reproduce exact membership; invalid strata/assignments fail; no historical output rewritten.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A14 for database/API/frontend/test/risk impact.

### A15 — Improve leakage and multilingual diagnostics

- [ ] Not started. Requirements: R17, R25.
- Depends on: A03, A06.
- Change: Cross-split prompt/answer/near/reference checks; shared script classifier and honest bounded scan rates.
- Acceptance: Known contamination fixtures detected; Unicode unchanged; truncated-token unknown rate uses matching denominator.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A15 for database/API/frontend/test/risk impact.

### A16 — Harden diagnostics and phase-A regression gate

- [ ] Not started. Requirements: R77, R79, R80, R81, R88, R89.
- Depends on: A01, A02, A03, A04, A05, A06, A07, A08, A09, A10, A11, A12, A13, A14, A15, A17, A18.
- Change: Redact secrets across raw logs/errors/artifacts while preserving tokenizer metadata; clean cancellation of profile workers.
- Acceptance: Synthetic credentials never leave log boundaries; migration, CPU tests, frontend checks pass; phase limitations documented.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A16 for database/API/frontend/test/risk impact.

### A17 — Add deterministic dataset mixtures

- [ ] Not started. Requirements: R15.
- Depends on: A03, A04.
- Change: Persist mixture identity and ordered memberships; implement concatenate with seed and fingerprint; keep eval independent.
- Acceptance: Same version list yields same concatenation/hash; mutation and eval overlap rejected.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A17 for database/API/frontend/test/risk impact.

### A18 — Add weighted/interleaved mixture sampling

- [ ] Not started. Requirements: R15.
- Depends on: A17.
- Change: Implement interleave, weighted and temperature distributions with deterministic seed and exhaustion policy.
- Acceptance: Seeded schedule reproducible; distribution fixtures and exhaustion behavior pass.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A18 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Scope controls

Do not recreate canonicalize, seeded preparation, DatasetVersion/Recipe/Split/Profile, TokenizerArtifact or existing loss masks. Extend them. Preference/tool records remain training-gated until their workers/templates support them. A17/A18 are dependencies of A16: task numbers are identifiers, not the complete execution order.
