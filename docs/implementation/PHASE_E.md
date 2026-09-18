# Phase E — Serving and export

**Status: Partially Complete — Steps 18–20 implementation slice.** Export supervision, safe merging, operation-specific quantization descriptors, optional SGLang integration, provider controls, normalized generation/scoring APIs, reproducible Ollama packaging, and persisted serving benchmarks are implemented. Real CUDA providers, GGUF reload, Ollama import and calibrated quantizers remain unverified or unavailable. Detailed impact records are in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md).

## Steps 18–19 acceptance evidence

- E01 partial: additive typed feature/version/schema fields preserve provider operation arrays. Native model IDs use the requested reference. Durable deployment restart reconciliation remains open.
- E02 partial: managed vLLM reuses the Playground's existing manager. External services are never stopped by managed controls. CLI flag probing gates memory fraction, length, tensor parallel, eager and prefix controls; CUDA count/local context/recorded quantization checks reject incompatible requests. LoRA rank/quantization controls require model-specific evidence; speculative decoding is unavailable until draft compatibility is verified.
- E03 implemented, runtime unverified: optional package/version detection, external URL, shared isolated lifecycle, health checks, process-tree cancellation, GPU leases, release on worker exit and conditional frontend controls. Installed/unavailable lifecycle paths are tested with fixtures; no real SGLang installation ran.
- E04 partial: root `/v1/models`, chat and text completions forward to all four providers; native chat/text stream; `/v1/scores` serves registered reward models with finite-score and generation gating. Tools/reasoning remain explicitly unsupported without verified parsers; metrics/batching/advanced features are described truthfully rather than enabled from package presence.
- E05 implemented, converter runtime unverified: artifact-backed immutable request snapshots; at most two supervised exports; copy/merge/GGUF/Hub/Ollama targets; standard shared progress/profile/artifact events; persisted logs, failure report, cancellation, PID/start-time restart recovery, source/payload fingerprints and lineage. Legacy merge/GGUF/Hub endpoints use this owner. Remote bases require original immutable commits; local bases are fingerprinted; vocabulary/special tokens, architecture, low-bit state and safe merge are checked. Tiny CPU merge output equivalence/save/reload passed.
- E06 partial: six optional quantization methods expose installed version and separate training/loading/inference/export/merge states. Calibration metadata is typed and checked against dataset version/content/sample count when supplied. No unverified BNB/GPTQ/AWQ/HQQ/EETQ/AQLM exporter is enabled. Existing GGUF conversion is supervised; real converter/reload evidence remains open.
- E07 partial: Ollama packages now retain a relative GGUF `FROM`, an explicit or locally derived chat template, stop tokens and safe generation defaults; the package may be imported through the optional local CLI without replacing its Modelfile. Installed Ollama smoke remains open.
- E08 partial: `ServingBenchmark` records bind an optional artifact to the measured provider, endpoint, model reference, request configuration and before/after hardware snapshots. Bounded streamed requests produce TTFT, inter-token latency, end-to-end latency, per-request/aggregate tokens per second, throughput, p50/p95/p99, errors and concurrency. vLLM probes its Prometheus KV-cache metric where exposed; unavailable telemetry remains null rather than invented. Restart/cancellation state is recovered safely. Fake-provider persistence and percentile fixtures pass; a real local endpoint acceptance remains open.

Files: `core/export_jobs.py`, `core/serving_benchmarks.py`, `export_contracts.py`, `ollama_package.py`, `train_entry/export.py`, existing merge/runtime/provider/vLLM owners, `api/openai_gateway.py`, Registry/Serving UI and frontend proxies. Frozen migration `0004_serving_benchmarks` adds only benchmark records; export job snapshots/lineage remain in artifact JSON. Test evidence is recorded in STATUS.md.

## Entry criteria

Previous phases satisfy their acceptance gates. Resolve task-specific dependencies below; do not enable an unavailable integration to satisfy a checklist.

## Independently verifiable tasks

### E01 — Persist deployment identity and provider capabilities

Foundation note: provider capabilities are typed and additive status payloads preserve legacy operation arrays. Deployment persistence and restart reconciliation remain open.

- [ ] Partially complete; see Steps 18–19 evidence above. Requirements: R59, R64, R75.
- Depends on: B12.
- Change: Unify lifecycle/generate/health capability schema; retain external management semantics.
- Acceptance: Restart reconciles stale deployment records without killing unrelated processes; legacy serving remains usable.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E01 for database/API/frontend/test/risk impact.

### E02 — Consolidate vLLM integrations and controls

- [ ] Partially complete; see Steps 18–19 evidence above. Requirements: R66, R64.
- Depends on: E01.
- Change: Shared adapter for managed/external modes; version-gate length/TP/LoRA/prefix/quant/eager/speculation.
- Acceptance: Legacy Playground and Compose mode work without duplicate GPU owners or port collision.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E02 for database/API/frontend/test/risk impact.

### E03 — Add optional SGLang provider

- [ ] Partially complete; see Steps 18–19 evidence above. Requirements: R65.
- Depends on: E01.
- Change: Dependency detection, isolated lifecycle, health/cancel and normalized generation.
- Acceptance: Missing package safe; configured tiny server start/test/stop/restart; lease released.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E03 for database/API/frontend/test/risk impact.

### E04 — Normalize OpenAI serving and reward scoring

- [ ] Partially complete; see Steps 18–19 evidence above. Requirements: R67, R68.
- Depends on: A09, C07, E01.
- Change: Chat/completion/stream contracts plus separate score endpoint; explicit tool/reasoning support.
- Acceptance: OpenAI response/SSE fixtures pass; reward scores numeric; unsupported inputs rejected.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E04 for database/API/frontend/test/risk impact.

### E05 — Add immutable export job contract and merge checks

- [ ] Partially complete; see Steps 18–19 evidence above. Requirements: R60, R61, R75.
- Depends on: B12, E01.
- Change: Shared bounded lifecycle for adapter/HF/GGUF/Hub export; validate base revision/vocab/quant before merge.
- Acceptance: Wrong base/tokenizer/low-bit merge rejected; cancellation stops converter tree; output artifact immutable.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E05 for database/API/frontend/test/risk impact.

### E06 — Add quantization/calibration capabilities

- [ ] Partially complete; see Steps 18–19 evidence above. Requirements: R29, R62.
- Depends on: B07, E05.
- Change: Separate load/train/infer/export/merge support for BNB/GPTQ/AWQ/HQQ/EETQ/AQLM; implement only tested paths.
- Acceptance: Unsupported operations disabled; calibration fingerprint persisted; enabled exporter reload tested.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E06 for database/API/frontend/test/risk impact.

### E07 — Produce reproducible Ollama packaging

- [ ] Partially complete; see Steps 18–19 evidence above. Requirements: R63.
- Depends on: A08, E05.
- Change: Export-only Modelfile with template/stops/defaults; optional local import.
- Acceptance: Rendered Modelfile matches source policy; offline export works; installed Ollama smoke passes.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E07 for database/API/frontend/test/risk impact.

### E08 — Add serving benchmarks and regression suites

- [ ] Partially complete; see Steps 18–20 evidence above. Requirements: R69, R86, R90.
- Depends on: E02, E03, E04, E06, E07.
- Change: Bounded concurrent TTFT/ITL/throughput/percentile/error/VRAM measurement.
- Acceptance: Deterministic fake-server timings verified; real tiny endpoint benchmark saved and reloadable.
- Evidence to record: changed files, test commands/results, installed runtime versions, migration revision or none, known limitations. See roadmap E08 for database/API/frontend/test/risk impact.

## Exit criteria

All tasks above meet their acceptance criteria with evidence in STATUS.md. Backend pytest/Ruff and frontend test/lint/typecheck/build pass; run fresh/legacy migrations for schema changes and relevant runtime/cancellation/restart tests. Existing supported features stay usable. Optional runtime tests unavailable on the machine are explicitly unverified, not passing. Document unsupported features rather than enabling them speculatively.

## Scope controls

Preserve current managed Transformers and external vLLM/Ollama semantics while consolidating. No unrestricted Docker control or automatic external publication.
