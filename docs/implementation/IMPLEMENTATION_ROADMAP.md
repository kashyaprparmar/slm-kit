# Implementation roadmap

Architecture foundation complete; **no Phase A feature task has started**. Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md); current health is in [STATUS.md](STATUS.md).

## Planning contract

Task IDs are stable. Execute dependency order, not numeric order alone. Each task is a separately reviewable change with its own acceptance checks; do not batch an entire phase into one implementation. Package-version-specific features stay disabled until their runtime test passes. File paths below are repository-relative and identify existing extension points; new modules belong beside those owners, not in a parallel subsystem.

The **current implementation** for every task is explicitly the linked Rxx finding(s) below. Each record specifies work, files, dependencies, database and API/UI impact, migrations, tests, risk, phase, and acceptance criteria. R01 and R87 are completed audit planning; cross-cutting requirements apply as phase gates.

## Phase order

- A: preserve data semantics, identity, templates, masks and preparation.
- B: truthful capabilities, core training, resume, artifacts, optional LLaMA-Factory.
- C: alignment through shared trainer strategies.
- D: optional optimization and profiling.
- E: durable serving/export and benchmarks.
- F: explicit distributed execution; single GPU remains default.
- G: guarded experimental work and complete verified lifecycle report.

**First task: A01 — Preserve Alpaca system and history.** It is a narrow, reproducible data-loss defect at the existing canonical boundary and needs no migration. A02 and A03 are the next correctness priorities before building a training cache.

## Foundation checkpoint (2026-09-15)

The architecture review established shared contracts needed by the phase tasks without starting their feature work:

- `TrainingStage` and versioned `RunConfig` operation identity preserve existing task/method values.
- `TrainingBackendCapabilities` and an explicit `BackendSelector` drive backend validation metadata and frontend method/backend visibility. Automatic selection remains B04.
- `DatasetAdapter` registration now owns detect/validate/canonicalize/preview/stage contracts behind the old facade. New mappings and formats remain A01/A05-A07/A14.
- `ModelFamilyAdapter`, tokenizer/template, PEFT, quantization, and serving-provider capability models extend existing registries.
- Optional dependency probes have one import-light owner.
- The training event union accepts additive progress/resource/artifact/warning/profile/error envelopes. Emission, persistence, and UI telemetry remain B18/D08.

No task is treated as fully complete merely because its interface was established. The acceptance criteria and dependencies below still govern phase completion.

## Requirement-to-task coverage

- R01 (Already implemented correctly): AUDIT.
- R02 (Implemented but needs enhancement): B01, B03.
- R03 (Missing): B14, B15.
- R04 (Partially implemented): B01, B06, C01.
- R05 (Missing): B05, B06.
- R06 (Implemented but needs enhancement): A01, A04.
- R07 (Partially implemented): A01.
- R08 (Already implemented correctly): A04.
- R09 (Partially implemented): A04, A05, A07.
- R10 (Partially implemented): A05.
- R11 (Partially implemented): A06.
- R12 (Missing): A06.
- R13 (Partially implemented): A07.
- R14 (Partially implemented): A06, G01, G02.
- R15 (Missing): A17, A18.
- R16 (Implemented but needs enhancement): A03, A14.
- R17 (Partially implemented): A15.
- R18 (Partially implemented): A12.
- R19 (Missing): A02, A13.
- R20 (Partially implemented): A02, A11, B10.
- R21 (Partially implemented): A08.
- R22 (Implemented but needs enhancement): A09, A10.
- R23 (Implemented but needs enhancement): A10.
- R24 (Missing): A10.
- R25 (Implemented but needs enhancement): A15.
- R26 (Partially implemented): B01, D01, G04.
- R27 (Partially implemented): B05.
- R28 (Partially implemented): B07.
- R29 (Experimental): B07, E06.
- R30 (Missing): D02, D03, D09, D10, D11.
- R31 (Missing): D04.
- R32 (Missing): D05.
- R33 (Partially implemented): D04.
- R34 (Partially implemented): D06.
- R35 (Missing): D05.
- R36 (Partially implemented): B07.
- R37 (Implemented but needs enhancement): A12, B08.
- R38 (Partially implemented): B09, B10.
- R39 (Missing): C01, C02.
- R40 (Missing): C01, C03, C04, C05.
- R41 (Missing): C01, C06.
- R42 (Missing): C01, C07.
- R43 (Missing): C01.
- R44 (Missing): G03.
- R45 (Missing): B11.
- R46 (Missing): D07.
- R47 (Missing): D07.
- R48 (Implemented but needs enhancement): B11, B16.
- R49 (Missing): C08.
- R50 (Missing): F01, F02.
- R51 (Missing): F03.
- R52 (Missing): F04.
- R53 (Partially implemented): B13.
- R54 (Missing): B17.
- R55 (Partially implemented): B18.
- R56 (Partially implemented): A08, B03, B05, D06.
- R57 (Partially implemented): B02.
- R58 (Partially implemented): B12.
- R59 (Partially implemented): B12, E01.
- R60 (Partially implemented): E05.
- R61 (Partially implemented): E05.
- R62 (Missing): E06.
- R63 (Partially implemented): E07.
- R64 (Implemented but needs enhancement): E01, E02.
- R65 (Missing): E03.
- R66 (Partially implemented): E02.
- R67 (Missing): E04.
- R68 (Implemented but needs enhancement): A09, E04.
- R69 (Missing): E08.
- R70 (Missing): D08.
- R71 (Implemented but needs enhancement): B15.
- R72 (Implemented but needs enhancement): G05.
- R73 (Implemented but needs enhancement): D08.
- R74 (Implemented but needs enhancement): B16, F01.
- R75 (Implemented but needs enhancement): A03, B12, E01, E05.
- R76 (Partially implemented): A02, A03, B12.
- R77 (Partially implemented): A16.
- R78 (Partially implemented): B01, B04, B14, F04.
- R79 (Implemented but needs enhancement): A16, B02, B16.
- R80 (Implemented but needs enhancement): A16, G05.
- R81 (Partially implemented): A16, B16, C08, D08, F04, G05.
- R82 (Missing): B14, B15.
- R83 (Partially implemented): B01, B04.
- R84 (Already implemented correctly): B14, G04.
- R85 (Implemented but needs enhancement): A01, G05.
- R86 (Partially implemented): A09, C08, E08, G02, G05.
- R87 (Already implemented correctly): G05.
- R88 (Implemented but needs enhancement): A08, A16, B03, F02.
- R89 (Implemented but needs enhancement): A16, B02.
- R90 (Partially implemented): E08, G05.
- R91 (Partially implemented): G05.
- R92 (Partially implemented): G05.
## A01 — Preserve Alpaca system and history

- **Phase / state / risk:** A / Not started / medium.
- **Current implementation:** [R07](GAP_ANALYSIS.md#r07), [R06](GAP_ANALYSIS.md#r06), [R85](GAP_ANALYSIS.md#r85). Read those code findings before editing.
- **Dependencies:** Audit complete; no implementation dependency.
- **Relevant files:** `backend/app/datasets/adapters.py`, `backend/tests/test_dataset_adapters.py`.
- **Required work / backend and API impact:** Extend canonicalize pair conversion; reject malformed history explicitly.
- **Database impact / migration requirement:** None. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** None; existing canonical preview reflects conversion.
- **Tests required:** Regression fixtures for this behavior; System precedes ordered history then current instruction/input and answer; Unicode preserved; existing pairs unchanged. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** System precedes ordered history then current instruction/input and answer; Unicode preserved; existing pairs unchanged. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A02 — Resolve mutable tokenizer revisions safely

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R19](GAP_ANALYSIS.md#r19), [R20](GAP_ANALYSIS.md#r20), [R76](GAP_ANALYSIS.md#r76). Read those code findings before editing.
- **Dependencies:** Audit complete; no implementation dependency.
- **Relevant files:** `backend/app/api/data_lab.py`, `backend/app/train_entry/tokenizer_profile.py`.
- **Required work / backend and API impact:** Resolve branch/tag to immutable commit before cache lookup; fingerprint serialized tokenizer behavior.
- **Database impact / migration requirement:** None; invalidate unsafe cached profiles. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Show resolved revision and cache evidence.
- **Tests required:** Regression fixtures for this behavior; Moving branch changes invalidate profiles; identical pinned commits reuse; local tokenizer changes invalidate. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Moving branch changes invalidate profiles; identical pinned commits reuse; local tokenizer changes invalidate. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A03 — Bind runs to immutable dataset versions

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R16](GAP_ANALYSIS.md#r16), [R75](GAP_ANALYSIS.md#r75), [R76](GAP_ANALYSIS.md#r76). Read those code findings before editing.
- **Dependencies:** Audit complete; no implementation dependency.
- **Relevant files:** `backend/app/domain.py`, `backend/app/api/runs.py`, `backend/app/datasets/lineage.py`, `backend/app/backends/unsloth_backend.py`.
- **Required work / backend and API impact:** Snapshot version/path/hash and verify before worker reads; retain legacy dataset_id.
- **Database impact / migration requirement:** Add nullable run dataset_version_id FK/index; backfill only provable matches. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Version selector with legacy fallback.
- **Tests required:** Regression fixtures for this behavior; Changed or deleted source fails explicitly; old runs deserialize; worker consumes selected version. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Changed or deleted source fails explicitly; old runs deserialize; worker consumes selected version. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A04 — Introduce adapter registration behind canonicalize

- **Phase / state / risk:** A / Foundation implemented; phase acceptance pending A01 / medium.
- **Current implementation:** `DatasetAdapter` and built-in registration now preserve the existing facade and formats. A01 must extend pair semantics before A04 can pass its phase gate. See [R06](GAP_ANALYSIS.md#r06), [R08](GAP_ANALYSIS.md#r08), [R09](GAP_ANALYSIS.md#r09).
- **Dependencies:** A01.
- **Relevant files:** `backend/app/datasets/adapters.py`, `backend/app/datasets/validate.py`.
- **Required work / backend and API impact:** Preserve facade; adapter detect/validate/canonicalize/preview/stages descriptors.
- **Database impact / migration requirement:** None. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Use schema descriptions in existing Data Lab.
- **Tests required:** Regression fixtures for this behavior; Current text/pairs/OpenAI/ShareGPT roundtrip identically; ambiguous detection still fails. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Current text/pairs/OpenAI/ShareGPT roundtrip identically; ambiguous detection still fails. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A05 — Expand reusable field mapping

- **Phase / state / risk:** A / Not started / medium.
- **Current implementation:** [R09](GAP_ANALYSIS.md#r09), [R10](GAP_ANALYSIS.md#r10). Read those code findings before editing.
- **Dependencies:** A04.
- **Relevant files:** `backend/app/datasets/adapters.py`, `backend/app/api/datasets.py`, `frontend/src/components/DatasetPreparation.tsx`.
- **Required work / backend and API impact:** Support nested conversation role/content/system/tools mappings.
- **Database impact / migration requirement:** Version DatasetRecipe.config JSON; no new mapping table. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Add mapping controls and canonical preview to Prepare.
- **Tests required:** Regression fixtures for this behavior; Saved recipe replays custom ShareGPT fields without dropped context; conflicting mappings fail. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Saved recipe replays custom ShareGPT fields without dropped context; conflicting mappings fail. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A06 — Add canonical preference and KTO records

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R11](GAP_ANALYSIS.md#r11), [R12](GAP_ANALYSIS.md#r12), [R14](GAP_ANALYSIS.md#r14). Read those code findings before editing.
- **Dependencies:** A04.
- **Relevant files:** `backend/app/datasets/adapters.py`, `backend/app/domain.py`, `backend/app/datasets/validate.py`.
- **Required work / backend and API impact:** Validate paired/conversational preferences and strict boolean desirability; keep training gated.
- **Database impact / migration requirement:** Version canonical schema in DatasetVersion.schema; retain v1 reader. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Show counts, balance and compatibility reasons.
- **Tests required:** Regression fixtures for this behavior; Missing/equal responses and invalid roles/labels fail; KTO never becomes ordinary prompt-response SFT. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Missing/equal responses and invalid roles/labels fail; KTO never becomes ordinary prompt-response SFT. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A07 — Add tool-call canonical validation

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R13](GAP_ANALYSIS.md#r13), [R09](GAP_ANALYSIS.md#r09). Read those code findings before editing.
- **Dependencies:** A04.
- **Relevant files:** `backend/app/datasets/adapters.py`, `backend/app/datasets/validate.py`.
- **Required work / backend and API impact:** Validate function schema, JSON args, unique IDs, matching responses and role sequence.
- **Database impact / migration requirement:** Version schema JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Rich tool conversation preview in Data Lab.
- **Tests required:** Regression fixtures for this behavior; Malformed calls and unmatched responses fail; valid tool payload survives roundtrip; training remains gated. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Malformed calls and unmatched responses fail; valid tool payload survives roundtrip; training remains gated. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A08 — Centralize template resolution

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R21](GAP_ANALYSIS.md#r21), [R56](GAP_ANALYSIS.md#r56), [R88](GAP_ANALYSIS.md#r88). Read those code findings before editing.
- **Dependencies:** A04.
- **Relevant files:** `backend/app/train_entry/tokenization.py`, `backend/app/train_entry/model_runtime.py`, `backend/app/models/capabilities.py`.
- **Required work / backend and API impact:** Shared resolution for train/eval/serve; native then compatible family then explicit warned fallback; explicit override policy.
- **Database impact / migration requirement:** Persist template identity/policy in tokenizer config JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Show template source, compatibility and warning.
- **Tests required:** Regression fixtures for this behavior; Same messages resolve same template across consumers; no silent inference fallback or system loss. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Same messages resolve same template across consumers; no silent inference fallback or system loss. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A09 — Preserve structured evaluation conversations

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R22](GAP_ANALYSIS.md#r22), [R68](GAP_ANALYSIS.md#r68), [R86](GAP_ANALYSIS.md#r86). Read those code findings before editing.
- **Dependencies:** A08.
- **Relevant files:** `backend/app/train_entry/eval_run.py`, `backend/app/train_entry/model_runtime.py`, `frontend/src/components/eval/EvalHarness.tsx`.
- **Required work / backend and API impact:** Carry canonical messages through generation and perplexity.
- **Database impact / migration requirement:** Version EvalResult.detail; preserve old reports. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Compare rendered context and results.
- **Tests required:** Regression fixtures for this behavior; All turns/system remain structured; no role-text rewrapping; base/trained inputs match. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** All turns/system remain structured; no role-text rewrapping; base/trained inputs match. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A10 — Complete loss policy and token preview contract

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R22](GAP_ANALYSIS.md#r22), [R23](GAP_ANALYSIS.md#r23), [R24](GAP_ANALYSIS.md#r24). Read those code findings before editing.
- **Dependencies:** A08.
- **Relevant files:** `backend/app/train_entry/tokenization.py`, `backend/app/train_entry/tokenizer_profile.py`, `frontend/src/pages/Datasets.tsx`, `frontend/src/components/studio/TrainingStudio.tsx`.
- **Required work / backend and API impact:** Add final/all assistant policies and template-specific reasoning masks; reuse renderer in Studio preview.
- **Database impact / migration requirement:** Version TokenizerConfig JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Raw/canonical/rendered/tokens/mask view with special tokens and truncation.
- **Tests required:** Regression fixtures for this behavior; Preview labels equal worker labels including multi-turn and reasoning fixtures; no global tag stripping. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Preview labels equal worker labels including multi-turn and reasoning fixtures; no global tag stripping. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A11 — Add tokenizer import and mutation lifecycle

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R20](GAP_ANALYSIS.md#r20). Read those code findings before editing.
- **Dependencies:** A02, A08.
- **Relevant files:** `backend/app/domain.py`, `backend/app/db/models.py`, `backend/app/train_entry/tokenizer_profile.py`, `backend/app/backends/unsloth_backend.py`.
- **Required work / backend and API impact:** Implement import/extend with immutable output and embedding resize; keep unsupported train mode rejected.
- **Database impact / migration requirement:** Reuse TokenizerArtifact; version config/path metadata. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Tokenizer artifact selection and mutation impact.
- **Tests required:** Regression fixtures for this behavior; Reloaded tokenizer/model vocab agrees; added tokens train/save correctly; no silent ignored tokens. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Reloaded tokenizer/model vocab agrees; added tokens train/save correctly; no silent ignored tokens. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A12 — Add supported packing modes

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R18](GAP_ANALYSIS.md#r18), [R37](GAP_ANALYSIS.md#r37). Read those code findings before editing.
- **Dependencies:** A10.
- **Relevant files:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`, `frontend/src/lib/runconfig.ts`.
- **Required work / backend and API impact:** Gate standard/auto/neat modes against installed trainer and mask support.
- **Database impact / migration requirement:** Translate legacy packing boolean to versioned policy JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Packing selector and clearly estimated utilization.
- **Tests required:** Regression fixtures for this behavior; Token/label boundaries preserved; incompatible neat/masked modes rejected before launch. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Token/label boundaries preserved; incompatible neat/masked modes rejected before launch. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A13 — Persist reusable tokenized training cache

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R19](GAP_ANALYSIS.md#r19). Read those code findings before editing.
- **Dependencies:** A02, A03, A10, A12.
- **Relevant files:** `backend/app/datasets/lineage.py`, `backend/app/backends/unsloth_backend.py`, `backend/app/api/data_lab.py`.
- **Required work / backend and API impact:** Atomic cache writes keyed by version/recipe/tokenizer/template/stage/tokens/mask/packing; lock concurrent builds.
- **Database impact / migration requirement:** Add TokenizationCache metadata FK/index/unique key migration. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Cache status, size and safe rebuild.
- **Tests required:** Regression fixtures for this behavior; Any semantic input change misses cache; interrupted build never reusable; training reads cached tensors. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Any semantic input change misses cache; interrupted build never reusable; training reads cached tensors. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A14 — Extend source/manual/stratified splits

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R16](GAP_ANALYSIS.md#r16). Read those code findings before editing.
- **Dependencies:** A03, A04.
- **Relevant files:** `backend/app/datasets/prepare.py`, `backend/app/api/data_lab.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** Source/manual/stratified split assignment with deterministic membership.
- **Database impact / migration requirement:** Version split strategy in DatasetRecipe.config; reuse DatasetSplit. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Split assignment and source split selection.
- **Tests required:** Regression fixtures for this behavior; Same inputs reproduce exact membership; invalid strata/assignments fail; no historical output rewritten. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Same inputs reproduce exact membership; invalid strata/assignments fail; no historical output rewritten. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A15 — Improve leakage and multilingual diagnostics

- **Phase / state / risk:** A / Not started / medium.
- **Current implementation:** [R17](GAP_ANALYSIS.md#r17), [R25](GAP_ANALYSIS.md#r25). Read those code findings before editing.
- **Dependencies:** A03, A06.
- **Relevant files:** `backend/app/datasets/quality.py`, `backend/app/train_entry/tokenizer_profile.py`, `frontend/src/pages/Datasets.tsx`.
- **Required work / backend and API impact:** Cross-split prompt/answer/near/reference checks; shared script classifier and honest bounded scan rates.
- **Database impact / migration requirement:** Reuse versioned DatasetProfile config/stats. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Affected examples, scope and per-language/script truncation.
- **Tests required:** Regression fixtures for this behavior; Known contamination fixtures detected; Unicode unchanged; truncated-token unknown rate uses matching denominator. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Known contamination fixtures detected; Unicode unchanged; truncated-token unknown rate uses matching denominator. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A16 — Harden diagnostics and phase-A regression gate

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R77](GAP_ANALYSIS.md#r77), [R79](GAP_ANALYSIS.md#r79), [R80](GAP_ANALYSIS.md#r80), [R81](GAP_ANALYSIS.md#r81), [R88](GAP_ANALYSIS.md#r88), [R89](GAP_ANALYSIS.md#r89). Read those code findings before editing.
- **Dependencies:** A01, A02, A03, A04, A05, A06, A07, A08, A09, A10, A11, A12, A13, A14, A15, A17, A18.
- **Relevant files:** `backend/app/core/log_capture.py`, `backend/app/core/runner.py`, `backend/app/core/errors.py`, `backend/tests`, `frontend/src`, `docs`.
- **Required work / backend and API impact:** Redact secrets across raw logs/errors/artifacts while preserving tokenizer metadata; clean cancellation of profile workers.
- **Database impact / migration requirement:** None. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Actionable errors and consistent template override text.
- **Tests required:** Regression fixtures for this behavior; Synthetic credentials never leave log boundaries; migration, CPU tests, frontend checks pass; phase limitations documented. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Synthetic credentials never leave log boundaries; migration, CPU tests, frontend checks pass; phase limitations documented. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A17 — Add deterministic dataset mixtures

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R15](GAP_ANALYSIS.md#r15). Read those code findings before editing.
- **Dependencies:** A03, A04.
- **Relevant files:** `backend/app/datasets/prepare.py`, `backend/app/api/data_lab.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** Persist mixture identity and ordered memberships; implement concatenate with seed and fingerprint; keep eval independent.
- **Database impact / migration requirement:** Add DatasetMixture identity and version memberships; recipe split strategy JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Mixture members and concatenate preview.
- **Tests required:** Regression fixtures for this behavior; Same version list yields same concatenation/hash; mutation and eval overlap rejected. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Same version list yields same concatenation/hash; mutation and eval overlap rejected. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## A18 — Add weighted/interleaved mixture sampling

- **Phase / state / risk:** A / Not started / high.
- **Current implementation:** [R15](GAP_ANALYSIS.md#r15). Read those code findings before editing.
- **Dependencies:** A17.
- **Relevant files:** `backend/app/datasets/prepare.py`, `backend/app/api/data_lab.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** Implement interleave, weighted and temperature distributions with deterministic seed and exhaustion policy.
- **Database impact / migration requirement:** Reuse DatasetMixture; strategy/weights/temperature in versioned JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Weights/temperature controls.
- **Tests required:** Regression fixtures for this behavior; Seeded schedule reproducible; distribution fixtures and exhaustion behavior pass. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Seeded schedule reproducible; distribution fixtures and exhaustion behavior pass. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B01 — Add authoritative backend and stage descriptors

- **Phase / state / risk:** B / Foundation implemented; runtime evidence acceptance pending / high.
- **Current implementation:** Backend/stage/method/PEFT/quantization descriptors, dependency availability, versioned operation identity, and frontend consumption are implemented. Runtime compatibility evidence and capability snapshots remain B01/B02 work. See [R02](GAP_ANALYSIS.md#r02), [R04](GAP_ANALYSIS.md#r04), [R26](GAP_ANALYSIS.md#r26), [R78](GAP_ANALYSIS.md#r78), [R83](GAP_ANALYSIS.md#r83).
- **Dependencies:** A16.
- **Relevant files:** `backend/app/backends/base.py`, `backend/app/models/capabilities.py`, `backend/app/domain.py`, `backend/app/api/runs.py`.
- **Required work / backend and API impact:** Separate stage/method/engine capabilities and installed compatibility evidence.
- **Database impact / migration requirement:** Version config schema; preserve task/method values. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Drive method/backend visibility from same API; remove independent static choices.
- **Tests required:** Regression fixtures for this behavior; Missing runtime cannot be advertised runnable; legacy payloads retained; unknown backend rejected. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Missing runtime cannot be advertised runnable; legacy payloads retained; unknown backend rejected. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B02 — Make preflight evidence truthful

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R57](GAP_ANALYSIS.md#r57), [R79](GAP_ANALYSIS.md#r79), [R89](GAP_ANALYSIS.md#r89). Read those code findings before editing.
- **Dependencies:** B01.
- **Relevant files:** `backend/app/train_entry/preflight.py`, `backend/app/core/preflight.py`, `frontend/src/components/studio/BaseModelPicker.tsx`.
- **Required work / backend and API impact:** Distinguish metadata-only from real isolated loader/forward/backward probe; share model-load policy.
- **Database impact / migration requirement:** Capability snapshot in Run.config JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Show verified level and resolved engine.
- **Tests required:** Regression fixtures for this behavior; CUDA presence alone never verifies Unsloth; dependency/loader failures block requested operation; leases released. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** CUDA presence alone never verifies Unsloth; dependency/loader failures block requested operation; leases released. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B03 — Extract shared native training orchestration

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R02](GAP_ANALYSIS.md#r02), [R56](GAP_ANALYSIS.md#r56), [R88](GAP_ANALYSIS.md#r88). Read those code findings before editing.
- **Dependencies:** B01.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/backends/base.py`, `backend/app/train_entry/model_runtime.py`.
- **Required work / backend and API impact:** Native and Unsloth loader strategies reuse dataset/callback/trainer code; factor revision/tokenizer policy without heavy API imports.
- **Database impact / migration requirement:** None. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** None.
- **Tests required:** Regression fixtures for this behavior; Native tiny SFT runs with Unsloth absent; optimized engine selection preserves requested semantics. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Native tiny SFT runs with Unsloth absent; optimized engine selection preserves requested semantics. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B04 — Resolve backend selection before launch

- **Phase / state / risk:** B / Selection interface implemented; automatic resolution not started / high.
- **Current implementation:** `RegisteredBackendSelector` resolves explicit existing registry keys for API estimate/create/export paths. It deliberately performs no fallback. See [R83](GAP_ANALYSIS.md#r83), [R78](GAP_ANALYSIS.md#r78).
- **Dependencies:** B02, B03.
- **Relevant files:** `backend/app/api/runs.py`, `backend/app/models/capabilities.py`, `frontend/src/components/studio/TrainingStudio.tsx`.
- **Required work / backend and API impact:** Auto plan checks operation/model/hardware/dependencies; revalidate in worker.
- **Database impact / migration requirement:** Requested and effective backend in Run.config JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Display chosen engine and reason.
- **Tests required:** Regression fixtures for this behavior; Unsupported Unsloth selects native only when equivalent; runtime fallback recorded; strict request honored. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Unsupported Unsloth selects native only when equivalent; runtime fallback recorded; strict request honored. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B05 — Discover adapter and freeze targets

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R05](GAP_ANALYSIS.md#r05), [R27](GAP_ANALYSIS.md#r27), [R56](GAP_ANALYSIS.md#r56). Read those code findings before editing.
- **Dependencies:** B03.
- **Relevant files:** `backend/app/models/capabilities.py`, `backend/app/backends/unsloth_backend.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** Architecture module inspection; auto/all-linear/attention/MLP/custom targets; trainable counts.
- **Database impact / migration requirement:** Selection config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Exact matched modules and parameter preview.
- **Tests required:** Regression fixtures for this behavior; Non-Llama models work; empty or invalid selections rejected; estimates use actual matched modules. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Non-Llama models work; empty or invalid selections rejected; estimates use actual matched modules. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B06 — Implement freeze tuning execution

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R05](GAP_ANALYSIS.md#r05), [R04](GAP_ANALYSIS.md#r04). Read those code findings before editing.
- **Dependencies:** B05.
- **Relevant files:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`, `backend/app/integrations/estimator.py`.
- **Required work / backend and API impact:** Last N layers, embeddings/head/norm/selected modules with explicit trainability.
- **Database impact / migration requirement:** Version method config; no table change. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Freeze controls and memory/trainable summary.
- **Tests required:** Regression fixtures for this behavior; Only selected parameters change in tiny step; unsupported architectures rejected. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Only selected parameters change in tiny step; unsupported architectures rejected. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B07 — Expose precision and quantization policy

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R28](GAP_ANALYSIS.md#r28), [R29](GAP_ANALYSIS.md#r29), [R36](GAP_ANALYSIS.md#r36). Read those code findings before editing.
- **Dependencies:** B01, B03.
- **Relevant files:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`, `backend/app/models/capabilities.py`.
- **Required work / backend and API impact:** Auto/bf16/fp16/fp32 and NF4/FP4/8bit compute/storage/nested controls gated by runtime.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Hardware-safe selectors.
- **Tests required:** Regression fixtures for this behavior; Unsupported dtype/quant combos fail before launch; effective settings match saved config. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Unsupported dtype/quant combos fail before launch; effective settings match saved config. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B08 — Verify continued pretraining semantics

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R37](GAP_ANALYSIS.md#r37). Read those code findings before editing.
- **Dependencies:** A12, B03.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `frontend/src/pages/DomainAdaptation.tsx`.
- **Required work / backend and API impact:** Raw causal-LM path with explicit EOS/tokenization and packing parity; wire validation set.
- **Database impact / migration requirement:** Config stage snapshot only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Default corpus/reused tokenizer and packing suggestion.
- **Tests required:** Regression fixtures for this behavior; Tiny continued-pretraining step uses no chat template; validation loss uses held-out split. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Tiny continued-pretraining step uses no chat template; validation loss uses held-out split. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B09 — Restore exact scratch checkpoint resume

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R38](GAP_ANALYSIS.md#r38). Read those code findings before editing.
- **Dependencies:** A03.
- **Relevant files:** `backend/app/backends/scratch_backend.py`, `backend/app/train_entry/run.py`, `backend/app/api/runs.py`.
- **Required work / backend and API impact:** Save/restore optimizer, RNG, step and data cursor; explicit resume endpoint.
- **Database impact / migration requirement:** Checkpoint format version; retain legacy weights-only loader. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Distinguish resume from clone; warn legacy checkpoints.
- **Tests required:** Regression fixtures for this behavior; Interrupted seeded run matches uninterrupted steps; old artifacts remain loadable. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Interrupted seeded run matches uninterrupted steps; old artifacts remain loadable. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B10 — Add scratch tokenizer and architecture options

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R20](GAP_ANALYSIS.md#r20), [R38](GAP_ANALYSIS.md#r38). Read those code findings before editing.
- **Dependencies:** A11, B09.
- **Relevant files:** `backend/app/backends/scratch_backend.py`, `backend/app/domain.py`, `frontend/src/pages/Pretrain.tsx`.
- **Required work / backend and API impact:** Reuse/import/train tokenizer modes; optional HF from-config initialization behind capability.
- **Database impact / migration requirement:** Config JSON and tokenizer artifact linkage. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Tokenizer choice and strict fit preview.
- **Tests required:** Regression fixtures for this behavior; Tiny supported architecture trains and reloads with chosen tokenizer; unsupported configs fail. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Tiny supported architecture trains and reloads with chosen tokenizer; unsupported configs fail. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B11 — Add validation and early stopping

- **Phase / state / risk:** B / Not started / medium.
- **Current implementation:** [R45](GAP_ANALYSIS.md#r45), [R48](GAP_ANALYSIS.md#r48). Read those code findings before editing.
- **Dependencies:** A03, B03.
- **Relevant files:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`, `backend/app/core/runner.py`.
- **Required work / backend and API impact:** Evaluation cadence/metric/direction/patience/threshold; wire or reject eval_on_completion.
- **Database impact / migration requirement:** Stopping reason and validation version in config/metrics. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Early-stop controls and eval curves.
- **Tests required:** Regression fixtures for this behavior; Synthetic plateau stops at expected step and persists reason; no train/eval overlap. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Synthetic plateau stops at expected step and persists reason; no train/eval overlap. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B12 — Register final artifacts and lineage

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R58](GAP_ANALYSIS.md#r58), [R59](GAP_ANALYSIS.md#r59), [R75](GAP_ANALYSIS.md#r75), [R76](GAP_ANALYSIS.md#r76). Read those code findings before editing.
- **Dependencies:** A03, B04.
- **Relevant files:** `backend/app/core/runner.py`, `backend/app/db/models.py`, `backend/app/api/registry.py`, `backend/app/core/reproducibility.py`.
- **Required work / backend and API impact:** Idempotent final artifact registration, exact revisions/environment and generated model card.
- **Database impact / migration requirement:** Add parent artifact FK/index and artifact type; backfill only known links. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Registry lineage links with unavailable historical data explicit.
- **Tests required:** Regression fixtures for this behavior; Successful run creates one artifact; retry duplicates prevented; existing run references preserved. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Successful run creates one artifact; retry duplicates prevented; existing run references preserved. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B13 — Preserve complete configuration import/export roundtrip

- **Phase / state / risk:** B / Not started / medium.
- **Current implementation:** [R53](GAP_ANALYSIS.md#r53). Read those code findings before editing.
- **Dependencies:** B01, B12.
- **Relevant files:** `backend/app/domain.py`, `backend/app/api/runs.py`, `frontend/src/lib/runconfig.ts`, `backend/pyproject.toml`.
- **Required work / backend and API impact:** JSON/YAML import/export through same RunConfig; retain fields outside flattened form.
- **Database impact / migration requirement:** Version config JSON; no extra table. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Import/export/clone/edit without field loss.
- **Tests required:** Regression fixtures for this behavior; Full config roundtrip equal; invalid YAML/config rejected; CLI/API yield same validation. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Full config roundtrip equal; invalid YAML/config rejected; CLI/API yield same validation. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B14 — Add optional LLaMA-Factory discovery/config adapter

- **Phase / state / risk:** B / Not started / medium.
- **Current implementation:** [R03](GAP_ANALYSIS.md#r03), [R78](GAP_ANALYSIS.md#r78), [R82](GAP_ANALYSIS.md#r82), [R84](GAP_ANALYSIS.md#r84). Read those code findings before editing.
- **Dependencies:** B01, B03.
- **Relevant files:** `backend/app/backends/base.py`, `backend/app/models/capabilities.py`, `backend/pyproject.toml`.
- **Required work / backend and API impact:** Version-qualified optional registration and native config translation; install guidance.
- **Database impact / migration requirement:** None; backend config remains RunConfig. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Optional/not-installed label.
- **Tests required:** Regression fixtures for this behavior; Absent package does not affect startup/native training; unsupported options rejected; no mandatory dependency. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Absent package does not affect startup/native training; unsupported options rejected; no mandatory dependency. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B15 — Bridge LLaMA-Factory worker lifecycle

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R03](GAP_ANALYSIS.md#r03), [R71](GAP_ANALYSIS.md#r71), [R82](GAP_ANALYSIS.md#r82). Read those code findings before editing.
- **Dependencies:** B12, B14.
- **Relevant files:** `backend/app/backends/base.py`, `backend/app/core/runner.py`, `backend/app/core/events.py`, `backend/tests`.
- **Required work / backend and API impact:** Launch optional engine within supervised process tree; map metrics/checkpoints/final/cancel/errors.
- **Database impact / migration requirement:** Reuse Run/Checkpoint/ModelArtifact. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Existing monitor.
- **Tests required:** Regression fixtures for this behavior; Contract test covers logs, failure, cancellation and lineage; optional tiny installed-runtime job passes. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Contract test covers logs, failure, cancellation and lineage; optional tiny installed-runtime job passes. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B16 — Expand resource and telemetry evidence

- **Phase / state / risk:** B / Not started / medium.
- **Current implementation:** [R48](GAP_ANALYSIS.md#r48), [R74](GAP_ANALYSIS.md#r74), [R79](GAP_ANALYSIS.md#r79), [R81](GAP_ANALYSIS.md#r81). Read those code findings before editing.
- **Dependencies:** B05, B07, B11, B12, B15, B13, B17, B18.
- **Relevant files:** `backend/app/integrations/estimator.py`, `backend/app/core/events.py`, `backend/app/core/runner.py`, `frontend/src/components/RunMonitor.tsx`.
- **Required work / backend and API impact:** Actual vs estimated token rates, resource samples, throttled DB writes and refined counts.
- **Database impact / migration requirement:** Compact summaries in Run.metrics; JSONL histories. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Evidence-based advisor and stage metrics.
- **Tests required:** Regression fixtures for this behavior; Metrics rate bounded; measured and estimated values labeled; phase-B native and optional regression checks pass. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Metrics rate bounded; measured and estimated values labeled; phase-B native and optional regression checks pass. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B17 — Add thin user-facing CLI

- **Phase / state / risk:** B / Not started / medium.
- **Current implementation:** [R54](GAP_ANALYSIS.md#r54). Read those code findings before editing.
- **Dependencies:** B13.
- **Relevant files:** `backend/app/domain.py`, `backend/app/api/runs.py`, `frontend/src/lib/runconfig.ts`, `backend/pyproject.toml`.
- **Required work / backend and API impact:** Train/chat/evaluate/inspect commands call shared services and RunConfig validation.
- **Database impact / migration requirement:** Version config JSON; no extra table. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** None; CLI usage documentation.
- **Tests required:** Regression fixtures for this behavior; CLI and API use same validation; errors yield nonzero exit; no duplicated training logic. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** CLI and API use same validation; errors yield nonzero exit; no duplicated training logic. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## B18 — Introduce model source resolution interface

- **Phase / state / risk:** B / Not started / high.
- **Current implementation:** [R55](GAP_ANALYSIS.md#r55). Read those code findings before editing.
- **Dependencies:** B03.
- **Relevant files:** `backend/app/model_refs.py`, `backend/app/integrations/hf_hub.py`.
- **Required work / backend and API impact:** Extract HF/local source adapters behind existing resolver; ModelScope descriptor unavailable until implemented.
- **Database impact / migration requirement:** None; source type and resolved revision in Run.config. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Preserve existing HF/local picker.
- **Tests required:** Regression fixtures for this behavior; Existing HF/local/run references unchanged; unknown provider rejected; no dependency on ModelScope. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Existing HF/local/run references unchanged; unknown provider rejected; no dependency on ModelScope. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## C01 — Define alignment/reference config

- **Phase / state / risk:** C / Not started / high.
- **Current implementation:** [R04](GAP_ANALYSIS.md#r04), [R39](GAP_ANALYSIS.md#r39), [R40](GAP_ANALYSIS.md#r40), [R41](GAP_ANALYSIS.md#r41), [R42](GAP_ANALYSIS.md#r42), [R43](GAP_ANALYSIS.md#r43). Read those code findings before editing.
- **Dependencies:** A06, B01, B12.
- **Relevant files:** `backend/app/domain.py`, `backend/app/models/capabilities.py`, `backend/app/integrations/estimator.py`.
- **Required work / backend and API impact:** Objective-specific validation and base/separate/adapter-disabled reference strategies.
- **Database impact / migration requirement:** Config JSON references to artifacts and dataset versions. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Alignment Studio inside Training.
- **Tests required:** Regression fixtures for this behavior; Invalid objective/data/reference combinations fail; reference memory included and lineage pinned. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Invalid objective/data/reference combinations fail; reference memory included and lineage pinned. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## C02 — Execute DPO through shared trainer strategy

- **Phase / state / risk:** C / Not started / high.
- **Current implementation:** [R39](GAP_ANALYSIS.md#r39). Read those code findings before editing.
- **Dependencies:** C01, B03.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/core/events.py`, `backend/tests`.
- **Required work / backend and API impact:** TRL DPO beta/loss/smoothing and reference selection with installed-version checks.
- **Database impact / migration requirement:** No extra tables. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** DPO settings.
- **Tests required:** Regression fixtures for this behavior; Tiny paired dataset trains; rewards/logprobs captured; native SFT unchanged. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Tiny paired dataset trains; rewards/logprobs captured; native SFT unchanged. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## C03 — Add IPO objective

- **Phase / state / risk:** C / Not started / medium.
- **Current implementation:** [R40](GAP_ANALYSIS.md#r40). Read those code findings before editing.
- **Dependencies:** C02.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** IPO objective strategy reuses preference pipeline.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Objective-dependent options.
- **Tests required:** Regression fixtures for this behavior; Tiny IPO job uses intended loss; invalid smoothing/options rejected. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Tiny IPO job uses intended loss; invalid smoothing/options rejected. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## C04 — Add ORPO objective

- **Phase / state / risk:** C / Not started / high.
- **Current implementation:** [R40](GAP_ANALYSIS.md#r40). Read those code findings before editing.
- **Dependencies:** C02.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** ORPO trainer adapter sharing canonical inputs/events.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** ORPO options.
- **Tests required:** Regression fixtures for this behavior; Tiny ORPO job passes; reference-free behavior and losses verified. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Tiny ORPO job passes; reference-free behavior and losses verified. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## C05 — Add SimPO objective

- **Phase / state / risk:** C / Not started / high.
- **Current implementation:** [R40](GAP_ANALYSIS.md#r40). Read those code findings before editing.
- **Dependencies:** C02.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** Version-gated SimPO implementation without separate engine.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** SimPO options.
- **Tests required:** Regression fixtures for this behavior; Objective math fixture and tiny job pass; unavailable runtime honestly gated. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Objective math fixture and tiny job pass; unavailable runtime honestly gated. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## C06 — Execute KTO and balance diagnostics

- **Phase / state / risk:** C / Not started / high.
- **Current implementation:** [R41](GAP_ANALYSIS.md#r41). Read those code findings before editing.
- **Dependencies:** C01.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/datasets/quality.py`.
- **Required work / backend and API impact:** KTO trainer strategy with desirable/undesirable weighting.
- **Database impact / migration requirement:** Config/profile JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Class balance warning and KTO metrics.
- **Tests required:** Regression fixtures for this behavior; Both label classes retained; malformed labels fail; tiny job produces artifact. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Both label classes retained; malformed labels fail; tiny job produces artifact. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## C07 — Train and evaluate reward models

- **Phase / state / risk:** C / Not started / high.
- **Current implementation:** [R42](GAP_ANALYSIS.md#r42). Read those code findings before editing.
- **Dependencies:** C01.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/train_entry/model_runtime.py`, `backend/app/train_entry/eval_run.py`.
- **Required work / backend and API impact:** Sequence-score loader/trainer and pairwise evaluation separate from causal generation.
- **Database impact / migration requirement:** Reward/reference artifact kinds use B12 migration. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Reward model labels and evaluation.
- **Tests required:** Regression fixtures for this behavior; Tiny reward job saves/reloads numeric scores and lineage; generation disallowed for reward artifacts. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Tiny reward job saves/reloads numeric scores and lineage; generation disallowed for reward artifacts. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## C08 — Alignment telemetry and regression gate

- **Phase / state / risk:** C / Not started / medium.
- **Current implementation:** [R49](GAP_ANALYSIS.md#r49), [R81](GAP_ANALYSIS.md#r81), [R86](GAP_ANALYSIS.md#r86). Read those code findings before editing.
- **Dependencies:** C02, C03, C04, C05, C06, C07.
- **Relevant files:** `frontend/src/components/RunMonitor.tsx`, `backend/app/core/events.py`, `backend/tests`, `docs`.
- **Required work / backend and API impact:** Normalize chosen/rejected rewards, margins, accuracy, logprob/KL by objective.
- **Database impact / migration requirement:** Metrics JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Stage-aware charts.
- **Tests required:** Regression fixtures for this behavior; Replay fixtures render missing metrics safely; installed objectives tested; base/trained comparison regression passes. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Replay fixtures render missing metrics safely; installed objectives tested; base/trained comparison regression passes. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D01 — Version-gate advanced PEFT strategies

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R26](GAP_ANALYSIS.md#r26). Read those code findings before editing.
- **Dependencies:** B05, B07.
- **Relevant files:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`, `backend/app/models/capabilities.py`.
- **Required work / backend and API impact:** rsLoRA, LoRA+, PiSSA, LoftQ, EVA strategy descriptors; enable each only after its fixture/runtime test.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Advanced PEFT controls.
- **Tests required:** Regression fixtures for this behavior; Each enabled strategy has version/config compatibility tests and tiny train/save/reload evidence. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Each enabled strategy has version/config compatibility tests and tiny train/save/reload evidence. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D02 — Add optimizer strategy registry

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R30](GAP_ANALYSIS.md#r30). Read those code findings before editing.
- **Dependencies:** B01.
- **Relevant files:** `backend/app/domain.py`, `backend/app/backends/unsloth_backend.py`, `backend/app/integrations/estimator.py`.
- **Required work / backend and API impact:** Schema/validation/estimation hooks; first adapter GaLore, other methods remain gated.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Advanced memory strategy.
- **Tests required:** Regression fixtures for this behavior; Default optimizer unchanged; absent package rejects; GaLore tiny step and estimate evidence pass. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Default optimizer unchanged; absent package rejects; GaLore tiny step and estimate evidence pass. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D03 — Add APOLLO optimizer adapter

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R30](GAP_ANALYSIS.md#r30). Read those code findings before editing.
- **Dependencies:** D02.
- **Relevant files:** `backend/app/backends`, `backend/app/models/capabilities.py`, `backend/tests`.
- **Required work / backend and API impact:** Version-gated APOLLO configuration, adapter and memory estimate.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Expose only verified adapters.
- **Tests required:** Regression fixtures for this behavior; APOLLO missing-dependency/config/finite-gradient/checkpoint tests pass before exposed. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** APOLLO missing-dependency/config/finite-gradient/checkpoint tests pass before exposed. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D04 — Resolve attention and checkpointing modes

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R31](GAP_ANALYSIS.md#r31), [R33](GAP_ANALYSIS.md#r33). Read those code findings before editing.
- **Dependencies:** B02, B07.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/models/capabilities.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** Auto/SDPA/FA2/eager plus off/standard/non-reentrant/optimized checkpointing.
- **Database impact / migration requirement:** Effective config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Advanced controls and fallback reasons.
- **Tests required:** Regression fixtures for this behavior; Hardware/model/dtype mismatch uses allowed fallback; explicit incompatible requests fail. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Hardware/model/dtype mismatch uses allowed fallback; explicit incompatible requests fail. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D05 — Add Liger and NEFTune options

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R32](GAP_ANALYSIS.md#r32), [R35](GAP_ANALYSIS.md#r35). Read those code findings before editing.
- **Dependencies:** D04.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/domain.py`.
- **Required work / backend and API impact:** Optional kernel/noise strategies with installed-runtime gates.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Advanced kernel/regularization.
- **Tests required:** Regression fixtures for this behavior; Standard path unchanged; enabled options recorded and tiny loss/gradient tests pass. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Standard path unchanged; enabled options recorded and tiny loss/gradient tests pass. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D06 — Add model-aware RoPE policy

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R34](GAP_ANALYSIS.md#r34), [R56](GAP_ANALYSIS.md#r56). Read those code findings before editing.
- **Dependencies:** B05, D04.
- **Relevant files:** `backend/app/models/capabilities.py`, `backend/app/backends/unsloth_backend.py`.
- **Required work / backend and API impact:** Default/linear/dynamic/YaRN schema per family and runtime.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Native context warning and explicit choice.
- **Tests required:** Regression fixtures for this behavior; No silent config mutation; incompatible family/mode rejected; effective context persisted. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** No silent config mutation; incompatible family/mode rejected; effective context persisted. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D07 — Add opt-in profiler artifacts

- **Phase / state / risk:** D / Not started / medium.
- **Current implementation:** [R46](GAP_ANALYSIS.md#r46), [R47](GAP_ANALYSIS.md#r47). Read those code findings before editing.
- **Dependencies:** B16.
- **Relevant files:** `backend/app/backends/unsloth_backend.py`, `backend/app/core/events.py`, `backend/app/api/registry.py`.
- **Required work / backend and API impact:** PyTorch schedule and optional module timing with bounded trace storage.
- **Database impact / migration requirement:** Profiler artifact metadata; no new table if generic artifact sufficient. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Profiler settings and trace links.
- **Tests required:** Regression fixtures for this behavior; Off has no profiling hooks; CPU trace saved; CUDA test gated; cleanup on cancel. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Off has no profiling hooks; CPU trace saved; CUDA test gated; cleanup on cancel. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D08 — Add external tracking adapters and phase gate

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R70](GAP_ANALYSIS.md#r70), [R73](GAP_ANALYSIS.md#r73), [R81](GAP_ANALYSIS.md#r81). Read those code findings before editing.
- **Dependencies:** A16, B16, D01, D02, D03, D04, D05, D06, D07, D09, D10, D11.
- **Relevant files:** `backend/app/core/events.py`, `backend/app/backends`, `frontend/src/components/studio/TrainingStudio.tsx`, `docs`.
- **Required work / backend and API impact:** Optional TensorBoard/W&B/MLflow/SwanLab callback sinks; local events remain authoritative.
- **Database impact / migration requirement:** Nonsecret tracker config JSON; credentials environment only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Recommended/Advanced/Expert groups.
- **Tests required:** Regression fixtures for this behavior; Tracker failures cannot lose local history; secret tests pass; optimization regression suite passes. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Tracker failures cannot lose local history; secret tests pass; optimization regression suite passes. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D09 — Add BAdam optimizer adapter

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R30](GAP_ANALYSIS.md#r30). Read those code findings before editing.
- **Dependencies:** D02.
- **Relevant files:** `backend/app/backends`, `backend/app/models/capabilities.py`, `backend/tests`.
- **Required work / backend and API impact:** Version-gated BAdam configuration, adapter and memory estimate.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Expose only verified adapters.
- **Tests required:** Regression fixtures for this behavior; BAdam missing-dependency/config/finite-gradient/checkpoint tests pass before exposed. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** BAdam missing-dependency/config/finite-gradient/checkpoint tests pass before exposed. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D10 — Add Adam-mini optimizer adapter

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R30](GAP_ANALYSIS.md#r30). Read those code findings before editing.
- **Dependencies:** D02.
- **Relevant files:** `backend/app/backends`, `backend/app/models/capabilities.py`, `backend/tests`.
- **Required work / backend and API impact:** Version-gated Adam-mini configuration, adapter and memory estimate.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Expose only verified adapters.
- **Tests required:** Regression fixtures for this behavior; Adam-mini missing-dependency/config/finite-gradient/checkpoint tests pass before exposed. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Adam-mini missing-dependency/config/finite-gradient/checkpoint tests pass before exposed. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## D11 — Add Muon optimizer adapter

- **Phase / state / risk:** D / Not started / high.
- **Current implementation:** [R30](GAP_ANALYSIS.md#r30). Read those code findings before editing.
- **Dependencies:** D02.
- **Relevant files:** `backend/app/backends`, `backend/app/models/capabilities.py`, `backend/tests`.
- **Required work / backend and API impact:** Version-gated Muon configuration, adapter and memory estimate.
- **Database impact / migration requirement:** Config JSON only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Expose only verified adapters.
- **Tests required:** Regression fixtures for this behavior; Muon missing-dependency/config/finite-gradient/checkpoint tests pass before exposed. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Muon missing-dependency/config/finite-gradient/checkpoint tests pass before exposed. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## E01 — Persist deployment identity and provider capabilities

- **Phase / state / risk:** E / Provider capability foundation implemented; persistence not started / high.
- **Current implementation:** Existing Transformers/vLLM/Ollama providers now expose one typed capability contract while retaining legacy status arrays. Durable deployment identity and reconciliation remain E01. See [R59](GAP_ANALYSIS.md#r59), [R64](GAP_ANALYSIS.md#r64), [R75](GAP_ANALYSIS.md#r75).
- **Dependencies:** B12.
- **Relevant files:** `backend/app/serving/providers.py`, `backend/app/core/deployment.py`, `backend/app/db/models.py`, `frontend/src/pages/Serving.tsx`.
- **Required work / backend and API impact:** Unify lifecycle/generate/health capability schema; retain external management semantics.
- **Database impact / migration requirement:** Add Deployment/config/status/artifact FK/index migration. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Provider-driven controls.
- **Tests required:** Regression fixtures for this behavior; Restart reconciles stale deployment records without killing unrelated processes; legacy serving remains usable. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Restart reconciles stale deployment records without killing unrelated processes; legacy serving remains usable. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## E02 — Consolidate vLLM integrations and controls

- **Phase / state / risk:** E / Not started / high.
- **Current implementation:** [R66](GAP_ANALYSIS.md#r66), [R64](GAP_ANALYSIS.md#r64). Read those code findings before editing.
- **Dependencies:** E01.
- **Relevant files:** `backend/app/integrations/vllm_serve.py`, `backend/app/core/eval_manager.py`, `backend/app/serving/providers.py`, `docker-compose.yml`.
- **Required work / backend and API impact:** Shared adapter for managed/external modes; version-gate length/TP/LoRA/prefix/quant/eager/speculation.
- **Database impact / migration requirement:** Migrate legacy engine settings to provider config with defaults. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Only compatible controls.
- **Tests required:** Regression fixtures for this behavior; Legacy Playground and Compose mode work without duplicate GPU owners or port collision. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Legacy Playground and Compose mode work without duplicate GPU owners or port collision. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## E03 — Add optional SGLang provider

- **Phase / state / risk:** E / Not started / high.
- **Current implementation:** [R65](GAP_ANALYSIS.md#r65). Read those code findings before editing.
- **Dependencies:** E01.
- **Relevant files:** `backend/app/serving/providers.py`, `backend/app/core/deployment.py`, `backend/app/config.py`.
- **Required work / backend and API impact:** Dependency detection, isolated lifecycle, health/cancel and normalized generation.
- **Database impact / migration requirement:** Reuse Deployment. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Optional provider guidance.
- **Tests required:** Regression fixtures for this behavior; Missing package safe; configured tiny server start/test/stop/restart; lease released. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Missing package safe; configured tiny server start/test/stop/restart; lease released. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## E04 — Normalize OpenAI serving and reward scoring

- **Phase / state / risk:** E / Not started / high.
- **Current implementation:** [R67](GAP_ANALYSIS.md#r67), [R68](GAP_ANALYSIS.md#r68). Read those code findings before editing.
- **Dependencies:** A09, C07, E01.
- **Relevant files:** `backend/app/train_entry/serve.py`, `backend/app/train_entry/model_runtime.py`, `backend/app/serving/providers.py`.
- **Required work / backend and API impact:** Chat/completion/stream contracts plus separate score endpoint; explicit tool/reasoning support.
- **Database impact / migration requirement:** No extra tables. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Endpoint examples and score test view.
- **Tests required:** Regression fixtures for this behavior; OpenAI response/SSE fixtures pass; reward scores numeric; unsupported inputs rejected. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** OpenAI response/SSE fixtures pass; reward scores numeric; unsupported inputs rejected. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## E05 — Add immutable export job contract and merge checks

- **Phase / state / risk:** E / Not started / high.
- **Current implementation:** [R60](GAP_ANALYSIS.md#r60), [R61](GAP_ANALYSIS.md#r61), [R75](GAP_ANALYSIS.md#r75). Read those code findings before editing.
- **Dependencies:** B12, E01.
- **Relevant files:** `backend/app/api/registry.py`, `backend/app/train_entry/merge.py`, `backend/app/integrations/gguf.py`, `backend/app/db/models.py`.
- **Required work / backend and API impact:** Shared bounded lifecycle for adapter/HF/GGUF/Hub export; validate base revision/vocab/quant before merge.
- **Database impact / migration requirement:** Add ExportJob/artifact lineage/status migration. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Export logs/progress/retry.
- **Tests required:** Regression fixtures for this behavior; Wrong base/tokenizer/low-bit merge rejected; cancellation stops converter tree; output artifact immutable. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Wrong base/tokenizer/low-bit merge rejected; cancellation stops converter tree; output artifact immutable. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## E06 — Add quantization/calibration capabilities

- **Phase / state / risk:** E / Not started / high.
- **Current implementation:** [R29](GAP_ANALYSIS.md#r29), [R62](GAP_ANALYSIS.md#r62). Read those code findings before editing.
- **Dependencies:** B07, E05.
- **Relevant files:** `backend/app/models/capabilities.py`, `backend/app/api/registry.py`, `backend/app/integrations`.
- **Required work / backend and API impact:** Separate load/train/infer/export/merge support for BNB/GPTQ/AWQ/HQQ/EETQ/AQLM; implement only tested paths.
- **Database impact / migration requirement:** Calibration dataset version/seed in ExportJob JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Method-specific calibration controls.
- **Tests required:** Regression fixtures for this behavior; Unsupported operations disabled; calibration fingerprint persisted; enabled exporter reload tested. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Unsupported operations disabled; calibration fingerprint persisted; enabled exporter reload tested. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## E07 — Produce reproducible Ollama packaging

- **Phase / state / risk:** E / Not started / medium.
- **Current implementation:** [R63](GAP_ANALYSIS.md#r63). Read those code findings before editing.
- **Dependencies:** A08, E05.
- **Relevant files:** `backend/app/serving/providers.py`, `backend/app/api/serving.py`, `frontend/src/pages/Serving.tsx`.
- **Required work / backend and API impact:** Export-only Modelfile with template/stops/defaults; optional local import.
- **Database impact / migration requirement:** Export job metadata. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Export and import separate actions.
- **Tests required:** Regression fixtures for this behavior; Rendered Modelfile matches source policy; offline export works; installed Ollama smoke passes. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Rendered Modelfile matches source policy; offline export works; installed Ollama smoke passes. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## E08 — Add serving benchmarks and regression suites

- **Phase / state / risk:** E / Not started / high.
- **Current implementation:** [R69](GAP_ANALYSIS.md#r69), [R86](GAP_ANALYSIS.md#r86), [R90](GAP_ANALYSIS.md#r90). Read those code findings before editing.
- **Dependencies:** E02, E03, E04, E06, E07.
- **Relevant files:** `backend/app/api/serving.py`, `backend/app/serving/providers.py`, `frontend/src/pages/Serving.tsx`, `backend/tests`, `docs`.
- **Required work / backend and API impact:** Bounded concurrent TTFT/ITL/throughput/percentile/error/VRAM measurement.
- **Database impact / migration requirement:** Benchmark result tied to deployment/artifact/provider/config/hardware. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Benchmark report and regression comparisons.
- **Tests required:** Regression fixtures for this behavior; Deterministic fake-server timings verified; real tiny endpoint benchmark saved and reloadable. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Deterministic fake-server timings verified; real tiny endpoint benchmark saved and reloadable. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## F01 — Add device-aware distributed config and admission

- **Phase / state / risk:** F / Not started / high.
- **Current implementation:** [R50](GAP_ANALYSIS.md#r50), [R74](GAP_ANALYSIS.md#r74). Read those code findings before editing.
- **Dependencies:** B16, E01.
- **Relevant files:** `backend/app/domain.py`, `backend/app/core/resources.py`, `backend/app/core/queue.py`.
- **Required work / backend and API impact:** World/ranks/master/strategy and atomic device-set leases; retain one control process.
- **Database impact / migration requirement:** Device allocation and ranks in config; no required new DB entity. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Hide unavailable distributed strategies.
- **Tests required:** Regression fixtures for this behavior; Single-GPU unchanged; overlapping allocations blocked; all ranks cancelled together. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Single-GPU unchanged; overlapping allocations blocked; all ranks cancelled together. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## F02 — Add DDP launcher and process-tree recovery

- **Phase / state / risk:** F / Not started / high.
- **Current implementation:** [R50](GAP_ANALYSIS.md#r50), [R88](GAP_ANALYSIS.md#r88). Read those code findings before editing.
- **Dependencies:** F01.
- **Relevant files:** `backend/app/core/runner.py`, `backend/app/train_entry/run.py`, `backend/app/backends/base.py`.
- **Required work / backend and API impact:** torchrun supervision and rank-zero event aggregation.
- **Database impact / migration requirement:** Run launcher metadata JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Rank status under Run Monitor.
- **Tests required:** Regression fixtures for this behavior; Two-device smoke, rank failure and restart cleanup pass; no orphan ranks. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Two-device smoke, rank failure and restart cleanup pass; no orphan ranks. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## F03 — Add optional FSDP

- **Phase / state / risk:** F / Not started / high.
- **Current implementation:** [R51](GAP_ANALYSIS.md#r51). Read those code findings before editing.
- **Dependencies:** F02.
- **Relevant files:** `backend/app/backends`, `backend/app/domain.py`, `backend/tests`.
- **Required work / backend and API impact:** FSDP full tuning then separately gate QLoRA combinations.
- **Database impact / migration requirement:** Version distributed checkpoint metadata. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Compatible sharding controls.
- **Tests required:** Regression fixtures for this behavior; Multi-GPU save/resume and memory evidence; untested FSDP+QLoRA remains disabled. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Multi-GPU save/resume and memory evidence; untested FSDP+QLoRA remains disabled. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## F04 — Add optional DeepSpeed and scale gate

- **Phase / state / risk:** F / Not started / high.
- **Current implementation:** [R52](GAP_ANALYSIS.md#r52), [R78](GAP_ANALYSIS.md#r78), [R81](GAP_ANALYSIS.md#r81). Read those code findings before editing.
- **Dependencies:** F02.
- **Relevant files:** `backend/app/backends`, `backend/app/domain.py`, `backend/pyproject.toml`, `docs`.
- **Required work / backend and API impact:** Explicit ZeRO-2/3 dependency/config adapters.
- **Database impact / migration requirement:** Config/checkpoint metadata only. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Optional install guidance.
- **Tests required:** Regression fixtures for this behavior; No automatic dependency activation; two-device train/resume/cancel checks pass per strategy. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** No automatic dependency activation; two-device train/resume/cancel checks pass per strategy. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## G01 — Extend modality schema while retaining text safety

- **Phase / state / risk:** G / Not started / high.
- **Current implementation:** [R14](GAP_ANALYSIS.md#r14). Read those code findings before editing.
- **Dependencies:** A07, B01.
- **Relevant files:** `backend/app/datasets/adapters.py`, `backend/app/models/capabilities.py`.
- **Required work / backend and API impact:** Validate media references and declare per-modality capabilities; no implicit text conversion.
- **Database impact / migration requirement:** Canonical version with images/audio/videos descriptors. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Experimental/unsupported labels.
- **Tests required:** Regression fixtures for this behavior; Text v1 compatible; missing media rejected; unavailable multimodal training cannot launch. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Text v1 compatible; missing media rejected; unavailable multimodal training cannot launch. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## G02 — Add one experimental multimodal execution adapter

- **Phase / state / risk:** G / Not started / high.
- **Current implementation:** [R14](GAP_ANALYSIS.md#r14), [R86](GAP_ANALYSIS.md#r86). Read those code findings before editing.
- **Dependencies:** G01, E01.
- **Relevant files:** `backend/app/backends`, `backend/app/train_entry/model_runtime.py`, `backend/tests`.
- **Required work / backend and API impact:** One explicitly selected tested model/processor pair behind capabilities.
- **Database impact / migration requirement:** Processor/modality lineage JSON. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Expert-only modality preview.
- **Tests required:** Regression fixtures for this behavior; Tiny media job and reload pass; unsupported modalities/families stay disabled. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Tiny media job and reload pass; unsupported modalities/families stay disabled. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## G03 — Add expert PPO orchestration

- **Phase / state / risk:** G / Not started / high.
- **Current implementation:** [R44](GAP_ANALYSIS.md#r44). Read those code findings before editing.
- **Dependencies:** C07, C08, F01.
- **Relevant files:** `backend/app/backends`, `backend/app/domain.py`, `backend/app/integrations/estimator.py`.
- **Required work / backend and API impact:** Reward/reference/policy/buffer/KL/whitening config behind explicit expert gate.
- **Database impact / migration requirement:** Reward/reference/rollout config and artifact references. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Expert PPO and memory warning.
- **Tests required:** Regression fixtures for this behavior; Bounded rollout tiny job, cancellation and lineage tested without affecting SFT. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Bounded rollout tiny job, cancellation and lineage tested without affecting SFT. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## G04 — Evaluate OFT/QOFT and architecture transformations

- **Phase / state / risk:** G / Not started / high.
- **Current implementation:** [R26](GAP_ANALYSIS.md#r26), [R84](GAP_ANALYSIS.md#r84). Read those code findings before editing.
- **Dependencies:** D01, G01.
- **Relevant files:** `backend/app/backends`, `backend/app/models/capabilities.py`, `docs`.
- **Required work / backend and API impact:** Feasibility spike per method; no generic unchecked patch hook; keep unsupported unless validated.
- **Database impact / migration requirement:** Version transformation provenance when implemented. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** Expert labels only for implemented methods.
- **Tests required:** Regression fixtures for this behavior; Method-specific train/save/reload and license review before enablement; rejected proposals documented. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Method-specific train/save/reload and license review before enablement; rejected proposals documented. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## G05 — Final lifecycle verification and completion report

- **Phase / state / risk:** G / Not started / high.
- **Current implementation:** [R80](GAP_ANALYSIS.md#r80), [R81](GAP_ANALYSIS.md#r81), [R85](GAP_ANALYSIS.md#r85), [R86](GAP_ANALYSIS.md#r86), [R87](GAP_ANALYSIS.md#r87), [R90](GAP_ANALYSIS.md#r90), [R91](GAP_ANALYSIS.md#r91), [R92](GAP_ANALYSIS.md#r92). Read those code findings before editing.
- **Dependencies:** G02, G03, G04, E08, F03, F04.
- **Relevant files:** `backend/tests`, `frontend/src`, `docs`.
- **Required work / backend and API impact:** Run supported workflow matrix including restart/data/train/eval/export/serve and optional absence.
- **Database impact / migration requirement:** Fresh and legacy upgrade verification; no destructive migration. When adding schema, use a new frozen Alembic revision; verify fresh and legacy upgrade without modifying prior revisions.
- **Frontend impact:** End-to-end navigation and capability honesty.
- **Tests required:** Regression fixtures for this behavior; Report implemented/partial/experimental/missing with evidence; no untested runtime called stable. Use CPU/mocks for normal tests and explicitly marked optional runtime tests for ML/provider execution. Run touched backend lint/tests and frontend tests/build/lint when UI contracts change.
- **Acceptance criteria:** Report implemented/partial/experimental/missing with evidence; no untested runtime called stable. Existing supported behavior and stored references remain valid; update STATUS and affected product docs with evidence.

## Every phase exit gate

Run backend pytest and Ruff, frontend Vitest/lint/typecheck/build, fresh and legacy database migration tests for any schema change, and Compose configuration validation for service changes. Add tiny installed-runtime checks for each newly claimed stage/provider; document skips as unverified. Verify cancellation, resource release, artifact lineage, optional-dependency absence, config roundtrip and restart for the changed workflows. Compare against audit baseline; never attribute pre-existing failures to the new task or silently waive regressions. No bulk dependency upgrades or mandatory external tracker/backends.

