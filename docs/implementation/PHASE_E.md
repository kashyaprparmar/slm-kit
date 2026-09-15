# Phase E — Serving and export

**Status: Feature work not started; serving-provider capability interface established.** Detailed current implementation and impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Requirement findings are in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### E01 — Persist deployment identity and provider capabilities

Foundation note: provider capabilities are typed and additive status payloads preserve legacy operation arrays. Deployment persistence and restart reconciliation remain open.

- [ ] Not started. Requirements: R59, R64, R75.
- Depends on: B12.
- Change: Unify lifecycle/generate/health capability schema; retain external management semantics.
- Acceptance: Restart reconciles stale deployment records without killing unrelated processes; legacy serving remains usable.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E01 for database/API/frontend/test/risk impact.

### E02 — Consolidate vLLM integrations and controls

- [ ] Not started. Requirements: R66, R64.
- Depends on: E01.
- Change: Shared adapter for managed/external modes; version-gate length/TP/LoRA/prefix/quant/eager/speculation.
- Acceptance: Legacy Playground and Compose mode work without duplicate GPU owners or port collision.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E02 for database/API/frontend/test/risk impact.

### E03 — Add optional SGLang provider

- [ ] Not started. Requirements: R65.
- Depends on: E01.
- Change: Dependency detection, isolated lifecycle, health/cancel and normalized generation.
- Acceptance: Missing package safe; configured tiny server start/test/stop/restart; lease released.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E03 for database/API/frontend/test/risk impact.

### E04 — Normalize OpenAI serving and reward scoring

- [ ] Not started. Requirements: R67, R68.
- Depends on: A09, C07, E01.
- Change: Chat/completion/stream contracts plus separate score endpoint; explicit tool/reasoning support.
- Acceptance: OpenAI response/SSE fixtures pass; reward scores numeric; unsupported inputs rejected.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E04 for database/API/frontend/test/risk impact.

### E05 — Add immutable export job contract and merge checks

- [ ] Not started. Requirements: R60, R61, R75.
- Depends on: B12, E01.
- Change: Shared bounded lifecycle for adapter/HF/GGUF/Hub export; validate base revision/vocab/quant before merge.
- Acceptance: Wrong base/tokenizer/low-bit merge rejected; cancellation stops converter tree; output artifact immutable.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E05 for database/API/frontend/test/risk impact.

### E06 — Add quantization/calibration capabilities

- [ ] Not started. Requirements: R29, R62.
- Depends on: B07, E05.
- Change: Separate load/train/infer/export/merge support for BNB/GPTQ/AWQ/HQQ/EETQ/AQLM; implement only tested paths.
- Acceptance: Unsupported operations disabled; calibration fingerprint persisted; enabled exporter reload tested.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E06 for database/API/frontend/test/risk impact.

### E07 — Produce reproducible Ollama packaging

- [ ] Not started. Requirements: R63.
- Depends on: A08, E05.
- Change: Export-only Modelfile with template/stops/defaults; optional local import.
- Acceptance: Rendered Modelfile matches source policy; offline export works; installed Ollama smoke passes.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E07 for database/API/frontend/test/risk impact.

### E08 — Add serving benchmarks and regression suites

- [ ] Not started. Requirements: R69, R86, R90.
- Depends on: E02, E03, E04, E06, E07.
- Change: Bounded concurrent TTFT/ITL/throughput/percentile/error/VRAM measurement.
- Acceptance: Deterministic fake-server timings verified; real tiny endpoint benchmark saved and reloadable.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E08 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Scope controls

Preserve current managed Transformers and external vLLM/Ollama semantics while consolidating. No unrestricted Docker control or automatic external publication.
