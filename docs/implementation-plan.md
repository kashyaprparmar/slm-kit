# Engineering upgrade implementation plan

Updated 2026-09-12. The repository audit, architectural foundations, and the
immutable-data/tokenizer-correctness slice are implemented. The attached
lifecycle specification remains the overall roadmap; unchecked slices below
are not available product features.

## Existing implementation to preserve

- `app/backends/base.py` already defines the pluggable backend contract. The
  `transformers` registration reuses `UnslothBackend` with the optimized loader
  disabled; a second training implementation is unnecessary.
- `core/runner.py`, `train_entry/*`, `core/resources.py`, and `core/queue.py`
  already isolate heavy jobs and coordinate a process-local GPU lease. Keep the
  single control-plane process assumption explicit.
- `model_refs.py` resolves `run:<id>`, full models, adapters, and scratch outputs;
  `train_entry/model_runtime.py` is shared by generation, evaluation, and serving.
- Dataset upload streams bytes; JSONL/CSV/TXT and Parquet iteration exists;
  preparation creates new files, supports mapping/deduplication/seeded splits;
  HF import already streams a bounded number of rows.
- `models/capabilities.py` and revision-keyed HF metadata caching already exist.
  Extend their contracts rather than adding an independent capability engine.
- Projects and project/run membership, portable run manifests, normalized
  failures, managed Transformers serving, and external provider health exist.
- React Query owns server state. `useWorkflow` persists meaningful drafts with
  separate keys per studio. `TrainingStudio`, model/dataset pickers, fit panel,
  and run monitor are already shared. Preserve routes and components.
- Docker separates GPU/CPU/development images and an optional vLLM profile.
  GPU image import probes already cover Qwen3/BLOOM/PEFT.

## Audit findings and dependencies

1. `db/session.py` uses `create_all` only. No migration history, enforced foreign
   keys, or WAL configuration exists. Dataset versions and lineage must wait for
   an incremental migration mechanism. `api/runs.py` deletion omits project
   membership and evaluation references; fix before enforcing foreign keys.
2. `datasets/validate.py`, `_format_example` in the training backend, and
   `eval_run._extract` interpret schemas separately. Prompt/completion is missing.
   Preparation materializes up to 100,000 rows; JSON arrays load fully. Introduce
   a strict import-light adapter boundary before changing preparation storage.
3. Capability family matching can infer support from a repository name without
   architecture metadata. Adapters bypass architecture checks for PEFT; scratch
   outputs advertise HF/PEFT methods they cannot run. Unknown causal models have
   inconsistent training/backend states. Correct these before runtime preflight.
4. Training catches all chat-template failures and silently uses plain role text.
   TRL receives rendered strings and can add special tokens again. Correct this
   together with tokenizer preview/loss masks, not as an isolated UI toggle.
5. Runtime inference does not propagate pinned revisions consistently. Eval
   extraction keeps only the first user/assistant pair and loses multi-turn
   context. Pinning and canonical messages precede base-vs-trained comparisons.
6. Scratch checkpoints contain weights, architecture, and tokenizer but no
   optimizer/RNG state; scratch training ignores `resume_from`. Do not claim
   exact resume until a backend-specific checkpoint contract is implemented.
7. Dedicated vLLM provider coexists with the earlier warm Playground subprocess
   manager. Consolidation needs explicit migration of engine selection; retain
   working Transformers generation in the meantime.
8. Run configs already have typed LoRA/optimizer/train/scratch sections and a
   reproducibility snapshot. Add versioned model/data/tokenizer sections through
   backward-compatible migration, rather than replacing all historical JSON.
9. Docs contain broader claims than code: universal-model language, automatic
   scratch resume, and serving controls that apply only to the legacy manager.
   Update affected documentation per tested slice; final consistency pass later.

## Ordered implementation slices

### 1. Foundations (current iteration)

- [x] Add packaged Alembic revisions and programmatic startup migration; adopt
  existing unversioned tables without deleting rows, make a SQLite backup before
  upgrading an existing DB, reject unknown revisions/incompatible schemas.
- [x] Enable foreign keys, busy timeout, and WAL; expose schema/integrity
  diagnostics. Correct referenced-run deletion before enabling enforcement.
- [x] Make capability decisions metadata-driven and conservative for unknown
  models, noncausal adapters, and scratch models. Preserve existing wire values.
- [x] Add a typed canonical adapter contract and bounded schema inspection API;
  reject ambiguity and malformed data. Keep advanced formats gated until workers
  can consume them correctly.
- [x] Verify fresh DB, legacy DB, repeated startup, rollback, referential
  integrity, adapters, schema detection, and existing backend regressions.

### 2. Immutable data and tokenizer correctness

- [x] Add DatasetVersion/Recipe/Split/Profile and TokenizerArtifact through the
  additive `0003_dataset_versions` migration; backfill legacy file references
  without copying or rewriting source data.
- [x] Stream canonical transformations through a disk spool under a bounded CPU
  queue, publish outputs atomically, and persist fingerprinted recipes with
  deterministic train/validation/test splits and replay caching.
- [x] Use the canonical adapter in preparation, supervised training, continued
  pretraining, evaluation extraction, tokenizer profiling, and quality scans.
- [x] Profile the exact model tokenizer in an isolated process with cached
  revision/fingerprint identity, multilingual slices, length/truncation/padding/
  packing diagnostics, rendered previews, token IDs, and loss masks.
- [x] Require a native or explicit chat template and implement tested full,
  completion-only, and assistant-only loss. Gate tool/pretokenized/preference
  records and incompatible TRL behavior instead of silently flattening data or
  duplicating special tokens.
- [x] Add read-only quality diagnostics with evidence for duplicates, leakage,
  conflicting answers, Unicode/control/markup/whitespace issues, boilerplate,
  and possible PII. Expose lineage, preparation, tokenizer, and quality tabs in
  the existing Dataset Manager and tokenizer policy in Fine-Tuning Studio.

### 3. Model preflight and reliable backend routing

Depends on 1 and tokenizer contracts in 2. Extend lightweight inspection with
resolved commits, tokenizer/config fingerprints and dependency evidence. Add an
optional cancellable subprocess runtime preflight sharing the GPU lease. Route
by exact operation and runtime evidence; preserve manual override. Keep seq2seq
unsupported until a separate loader/trainer/eval contract is implemented and
tested. Smoke-test tiny generic Transformers independently of Unsloth.

### 4. Reproducible configuration and checkpoints

Depends on 2–3. Version RunConfig migration, bind immutable datasets and model
commits, persist effective worker environment, checkpoint optimizer/RNG state,
add explicit resume/fork/retention semantics. Test historical configs, interrupted
training, true resume vs weights-only, cancellation and lease release.

### 5. Comparable evaluation

Depends on 2–4. Preserve complete conversation context and pinned base revision;
persist identical generation settings for base/run comparisons. Add suite/model/
sample result entities, language slices, deltas and regression gates. Integrate
optional lm-eval in subprocesses with independently detected capabilities.

### 6. Artifact lineage and serving

Depends on 1, 3–5. Explicit merge/export lineage, checksums/tool versions, safe
owned-storage cleanup; persist deployments and controlled benchmarks. Consolidate
vLLM lifecycle around its dedicated service without Docker socket access. Add
provider-specific metrics, concurrency/latency tests, and restart checks.

### 7. Optional objectives

Depends on 2–4. Preference adapters and TRL DPO/ORPO/KTO, advanced PEFT and
tokenizer extension only after capability/runtime tests. Unsupported options stay
disabled with reasons; preserve the normal SFT flow.

### 8. UI consolidation and release verification

Integrate each slice incrementally into existing components. Then consolidate
navigation with backward-compatible redirects; add Data Lab/Training/Models/
Evaluate/System tabs and a sanitized support bundle. Full final gates: backend
and frontend tests, lint/typecheck/build, CPU and GPU Docker startup, migration
on existing DB, tiny train/cancel/resume/eval/export/serve lifecycle, provider
absence checks, persistence and documentation links. Record actual results.

## Baseline verification

- Before changes: `python -m pytest -q` in `backend`: **27 passed**.
- Existing Docker backend/frontend observed healthy. No training or production
  DB migration has been run as part of this baseline check.

## Foundations verification results

- Backend regression suite after changes: **61 passed** (one upstream
  Starlette/httpx deprecation warning).
- Ruff on new modules, affected API/capability modules and new tests: passed.
- Frontend: **3 tests passed**, TypeScript check and production Vite build passed.
- Built `slmkit-foundations-cpu-check` using `backend/Dockerfile.cpu`.
- Isolated CPU Docker smoke test with networking disabled: application startup,
  packaged migrations, health, database diagnostics, sample installation, upload,
  canonical schema inspection, preparation, shutdown and second startup passed.
  The second startup retained the uploaded/prepared data. Neither torch nor
  Transformers was imported into the API process.
- Existing data adoption/backup, failed-migration rollback, unknown future schema,
  FK enforcement, orphan reporting, deletion and baseline/model schema parity
  are covered with temporary databases. Production volume was not migrated.

## Immutable data and tokenizer verification results

- Focused migration/adapter/data/tokenizer/API suite: **50 passed**.
- Complete backend regression suite: **66 passed** (one upstream
  Starlette/httpx deprecation warning); Ruff: passed.
- Frontend: **3 tests passed**, TypeScript check and production build passed.
- Deterministic three-way replay, Windows/Linux-safe spool cleanup, atomic
  failure cleanup, legacy version backfill, strict template failures, loss-mask
  behavior, duplicate-special-token gating, advanced-format gating, and
  non-destructive quality evidence have automated coverage.

## Model preflight and backend routing verification results

- Complete backend regression suite: **77 passed** (one upstream
  Starlette/httpx deprecation warning); Ruff: passed across `app/` and `tests/`.
- Frontend: **7 tests passed**, TypeScript check (`tsc --noEmit`) and production
  Vite build passed cleanly.
- Model inspection metadata extended with resolved commit SHA, config/tokenizer
  SHA-256 fingerprints, and import-light runtime dependency detection without
  importing heavy ML packages into the API control plane.
- Sequence-to-sequence (encoder-decoder) architectures strictly gated and rejected
  across inspection, preflight, and training launch with clear guidance.
- Subprocess runtime preflight verified under atomic GPU lease acquisition,
  timeout termination, cancellation, and automatic lease release semantics.
- Scratch model offline preflight validated without requiring PyTorch/CUDA runtime.
- Frontend `BaseModelPicker` augmented with commit hash, fingerprint, runtime
  dependency badges, and interactive runtime preflight trigger and verification display.

Checkpoint-state resume, persisted comparable evals, artifact/deployment lineage,
optional preference training, and final navigation/release verification remain
in slices 4–8.
