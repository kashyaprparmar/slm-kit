# Phase A — Data, identity and templates

**Status: Partially Complete. A01 and A04 are complete; A02 has one reviewed correctness slice; A03 and A05-A18 remain incomplete.** Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Audit and architecture foundation complete; preserve the pre-existing working tree. A01 and A04 provide the canonical boundary required by later format work.

## Independently verifiable tasks

### A01 — Preserve Alpaca system and history

- [x] Complete 2026-09-15. Requirements: R07, R06, R85.
- Depends on: completed audit.
- Change: Extend canonicalize pair conversion; reject malformed history explicitly.
- Acceptance: System precedes ordered history then current instruction/input and answer; Unicode preserved; existing pairs unchanged.
- Evidence: `backend/app/datasets/adapters.py` preserves optional system, ordered history, current instruction/input, and final output without normalizing Unicode or whitespace. Malformed or lossy system/history inputs fail. Covered by the focused 50-test Phase A1 suite; no migration.

### A02 — Resolve mutable tokenizer revisions safely

- [ ] Partially complete. Requirements: R19, R20, R76.
- Depends on: completed audit.
- Change: Resolve branch/tag to immutable commit before cache lookup; fingerprint serialized tokenizer behavior.
- Acceptance: Moving branch changes invalidate profiles; identical pinned commits reuse; local tokenizer changes invalidate.
- Evidence: the pre-resolution profile-cache shortcut now accepts only full 40-character Hugging Face commit identifiers; branches, tags and abbreviated revisions always resolve again. Full tokenizer serialization fingerprints and local tokenizer content identity remain open, so A02 acceptance is not met. Covered by `test_data_lab_phase2.py`; no migration.

### A03 — Bind runs to immutable dataset versions

- [ ] Not started. Requirements: R16, R75, R76.
- Depends on: completed audit.
- Change: Snapshot version/path/hash and verify before worker reads; retain legacy dataset_id.
- Acceptance: Changed or deleted source fails explicitly; old runs deserialize; worker consumes selected version.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A03 for database/API/frontend/test/risk impact.

### A04 — Introduce adapter registration behind canonicalize

The registry and detect/validate/canonicalize/preview/stage interface now exist behind the compatible facade. Canonical previews include structured validation and semantic fingerprints.

- [x] Complete 2026-09-15. Requirements: R06, R08, R09.
- Depends on: A01.
- Change: Preserve facade; adapter detect/validate/canonicalize/preview/stages descriptors.
- Acceptance: Current text/pairs/OpenAI/ShareGPT roundtrip identically; ambiguous detection still fails.
- Evidence: current text/pair/OpenAI/ShareGPT conversion and v1 storage shapes remain compatible; ambiguous detection fails; advanced markers are detected before pair fallback and report explicit unsupported capability. Covered by the focused 50-test Phase A1 suite; no migration.

### A05 — Expand reusable field mapping

- [ ] Not started. Requirements: R09, R10.
- Depends on: A04.
- Change: Support nested conversation role/content/system/tools mappings.
- Acceptance: Saved recipe replays custom ShareGPT fields without dropped context; conflicting mappings fail.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A05 for database/API/frontend/test/risk impact.

### A06 — Add canonical preference and KTO records

- [ ] Not started. Requirements: R11, R12, R14.
- Foundation slice complete: strict version-2 record contracts, semantic fingerprints, and unsupported stage declarations exist. Source-format adapters, full validation, preparation, and training remain in A06.
- Depends on: A04.
- Change: Validate paired/conversational preferences and strict boolean desirability; keep training gated.
- Acceptance: Missing/equal responses and invalid roles/labels fail; KTO never becomes ordinary prompt-response SFT.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap A06 for database/API/frontend/test/risk impact.

### A07 — Add tool-call canonical validation

- [ ] Not started. Requirements: R13, R09.
- Foundation slice complete: version-2 tool definitions/calls/messages and media references exist. Call-ID matching, role-sequence validation, source adapters, preparation, and training remain in A07.
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

## Phase A final review — 2026-09-15

**Result: Partially Complete.** The phase is not blocked by an external dependency; its planned implementation is incomplete.

- Canonical ownership remains in `datasets/adapters.py`; no competing adapter, normalization, tokenizer, or template abstraction was introduced.
- Text and conversation canonical v1 storage remains backward compatible. Preference, KTO, tool, and media v2 contracts exist but their adapters and semantic acceptance work remain A06/A07.
- Preparation uses staged atomic publication and does not rewrite source files. DatasetVersion records can still reference mutable source paths until A03.
- A stale tokenizer-profile cache path for mutable revisions was fixed during review. Full serialized tokenizer identity and persistent tokenized training cache remain A02/A13.
- The existing three loss policies retain focused coverage. Final/all-assistant policies and reasoning-aware masks remain A10.
- Masked-loss packing is rejected before launch. Auto/standard/boundary-aware packing and real trainer boundary evidence remain A12.
- Explicit custom templates override native tokenizer templates in the backend; the frontend hint now states that behavior. Shared train/eval/serve resolution remains A08/A09.
- Fresh and legacy migrations pass at packaged head `0003_dataset_versions`; no review migration was required. Immutable run-to-version binding remains A03.
- Backend/API and frontend configuration checks pass. The richer Phase A adapter/mapping/template/cache UI contracts do not exist yet because their tasks remain open.
- Review evidence: focused Phase A/migration tests **59 passed**; full backend **108 passed**; Ruff passed; frontend **9 tests passed**; TypeScript and production build passed; ESLint passed with one pre-existing warning. Python has no configured static type checker; bytecode compilation passed.

## Scope controls

Do not recreate canonicalize, seeded preparation, DatasetVersion/Recipe/Split/Profile, TokenizerArtifact or existing loss masks. Extend them. Preference/tool records remain training-gated until their workers/templates support them. A17/A18 are dependencies of A16: task numbers are identifiers, not the complete execution order.
