# Implementation status

Updated: 2026-09-18. Audited HEAD: `5d64b78b6dc5289ef419d22ab11cd629cdba6337` plus the preserved working tree.

## Current phase

**Phase E — Partially Complete.** Steps 18–20 add unified export supervision, safe merging, quantization operation descriptors, SGLang integration, provider controls, normalized generation/reward APIs, Ollama policy packages and durable serving benchmarks. Real optional GPU/converter/Ollama acceptance remains open.

## Current task

**Step 20 — Serving benchmarks and Ollama export slice finished; optional runtime acceptance pending.** No Phase F work started.

## Completed tasks

- E05: unified artifact-backed export jobs for adapter, full HF, merge, GGUF/quantized GGUF, Ollama package and explicit Hub publication; standard shared events, logs/progress, failure reports, bounded supervision, tree cancellation, PID/start-time restart recovery and source/output payload fingerprints.
- E05: safe merge validates original immutable remote base commit or local content identity, architecture, vocabulary/special-token mappings and low-bit state. Tiny CPU adapter merge/save/reload equivalence passed. Legacy merge, GGUF and Hub routes use the shared job owner.
- E06: six-method optional quantization capability registry separates training/loading/inference/export/merging; typed calibration metadata checks existing version/fingerprint/content/sample count. Unverified quantization exporters remain unavailable.
- E02–E04: four-provider feature/version capability records, shared managed/external vLLM owner, optional SGLang lifecycle with GPU leases and worker-exit cleanup, CLI-gated engine controls, local context/GPU/quantization checks, root OpenAI gateway, native text streaming and finite reward scoring.
- Registry UI shows authoritative export gates, persisted job progress/logs and cancellation. Serving UI adds optional SGLang, installed-engine controls and selected-model reward scoring. Development/production proxies forward /v1 streaming.
- Export metadata/model cards redact credentials while preserving tokenizer identities. No database schema changes or mandatory optional dependencies.
- E07: Ollama packages now preserve a relative GGUF `FROM`, explicit or locally discovered chat template, safe stop tokens and generation defaults. Export-only remains available without Ollama; package import invokes the optional local CLI and retains the package Modelfile.
- E08: durable `ServingBenchmark` records link optional `ModelArtifact` identity with measured provider/model/endpoint, request configuration, provider snapshot, hardware and result data. Bounded streamed requests calculate TTFT, inter-token/end-to-end latency, p50/p95/p99, errors, concurrency, token rates and throughput. vLLM KV-cache telemetry is read only from available Prometheus output.

- Step 17: added the import-light `OptimizationCapability` registry for rsLoRA, LoRA+, PiSSA, LoftQ, EVA, OFT/QOFT, GaLore, APOLLO, BAdam, Adam-mini, Muon, FlashAttention 2, Liger, and NEFTune. Each descriptor contains availability, installed version, compatibility, field schema, progressive-disclosure level, and an impact record.
- Step 17: exposed that registry through the existing backend API contract and frontend types; compatibility aliases preserve existing `optional_features` clients.
- Step 17: added additive typed RunConfig fields for NEFTune and optimizer strategies without a migration; existing run JSON remains valid.
- Step 17: enabled and runtime-tested native rsLoRA/PiSSA/LoRA+ with PEFT 0.15.2; LoRA+ uses PEFT's optimizer factory and PiSSA uses the existing native model-loading path.
- Step 17: capability-gated NEFTune by the installed trainer signature, added pre-launch optimization validation, and made Auto backend selection reject engines that cannot preserve the requested advanced semantics.
- Step 17: added Recommended, Advanced, and Expert UI disclosure. Optional external optimizers and kernels remain disabled with backend-provided reasons instead of becoming selectable controls.

- Step 16: exercised all six objectives with local tiny models, all four exposed DPO losses, all six DPO preference metrics, reward-margin arithmetic, SimPO beta/gamma math, finite telemetry, and reloadable outputs.
- Step 16: verified adapter-disabled DPO checkpoint resume, graceful stop without completed artifacts, actual worker-descendant termination, incomplete-checkpoint exclusion, idempotent artifact registration, and resolved reference/adapter lineage contracts.
- Step 16: verified full/LoRA/DoRA reward training, repeatable registered adapter scoring after reload, preservation of the scalar head during full continuation, and explicit rejection of incompatible causal adapters.
- Step 16: fixed SimPO's unintended SFT term and invalid prompt limit, unresolved app reference loading, separate adapter-reference loss, token-ID mismatch handling, implicit references, ignored reference revisions, and reward-head LoRA wrapper collisions.
- Step 16: bound alignment preparation caches to content fingerprints, revalidated actual rows/classes, preserved line-aware canonical validation errors, corrected FP32/pairwise/separate-reference memory accounting, and retained older supported Transformers loading keywords.
- Step 16: included DPO loss/reference strategy in Auto selection, gated absent UI capabilities, cleared inactive objective payload settings, tested conditional controls, and corrected/restricted optional LLaMA-Factory alignment translation.
- Added pre-launch Auto backend selection over task, method, model, hardware, installed dependencies, alignment objective, tokenizer/loss policy, precision, attention, checkpointing, RoPE, and optional kernels.
- Persisted requested/effective backend, selection reason, and rejected alternatives in Run config JSON; exposed the same decision in the run plan. Explicit requests remain strict, clones restore Auto intent, and reruns retain the effective backend.
- Added the optional `LlamaFactoryBackend` with package/CLI/version discovery, a >=0.9 compatibility report, installation guidance, generated config/data translation, subprocess-tree cancellation, normal events/metrics/checkpoints/artifacts, worker revalidation, and no mandatory dependency.
- Registered final model/adapter artifacts idempotently from the shared event protocol with run, model, dataset, backend, method, revision, and reference lineage.
- Added `alignment` as a first-class task/stage and preference/KTO as first-class dataset kinds without a database schema change.
- Added strict canonical preference and KTO records, validation, chat-template rendering, KTO balance diagnostics, and task/objective dataset gating.
- Added one `PreferenceTrainer` lifecycle for DPO, IPO, ORPO, and SimPO, with KTO's distinct labels handled through the same model/callback/checkpoint/artifact foundation. PPO was not added.
- Added base-model, separate-model, adapter-disabled, and reference-free strategies with config validation, memory estimates, quantized reference loading, profile events, and artifact lineage.
- Added normalized reward, margin, accuracy, log-probability, and KL telemetry.
- Added DPO loss variants (`sigmoid`, `hinge`, `robust`, `exo_pair`) with objective-specific validation, smoothing rules, installed-TRL capability gates, and matching UI controls.
- Added reward-model training through the shared alignment lifecycle with a native scalar sequence-classification loader, PEFT support, standard checkpoints/events, and typed artifact lineage.
- Added pairwise reward evaluation with chosen/rejected scores, margin, and accuracy. Reward artifacts are rejected by generation, deployment, causal merge, and ordinary base-model selection.
- Added stable registry categories for Causal LM, Adapter, Merged Model, Reward Model, Reference Model, and Quantized Model while preserving historical artifact strings.
- Added the Alignment Studio and shared RunConfig roundtrip for objective/reference settings. Unsupported methods and invalid adapter-disabled/reference-free combinations are gated in both UI and backend validation.
- Completed the earlier Phase B slice for full/freeze/LoRA/QLoRA/DoRA, module discovery, parameter/optimizer estimates, precision/attention/checkpointing/Liger/RoPE policy, continued pretraining semantics, and scratch tokenizer/checkpoint support.

## In-progress tasks

None. Steps 18–19 implementation slice is finished; optional runtime and remaining phase acceptance are listed below.

## Remaining tasks

- E01: durable deployment identity and crash/restart reconciliation for native/optional serving workers.
- E02–E03: actual CUDA vLLM/SGLang start/stop/cancel and model matrix; adapter-serving profiles, model-specific quantization options and verified draft-model speculative decoding remain gated.
- E04: configured tool/reasoning parsers and metrics/batching acceptance. Gateway explicitly declines unsupported tool/reasoning settings.
- E05: installed llama.cpp conversion, quantized output reload and cancellation of a real converter tree. HF Hub publication is code/contract covered, not executed against an external repository.
- E06: calibrated GPTQ/AWQ/HQQ/EETQ/AQLM/BNB exporters only after tested implementation/reload evidence; none enabled from package presence alone.
- E07: installed Ollama import/smoke remains unverified on this host.
- E08: real tiny Transformers/vLLM/SGLang/Ollama benchmark acceptance remains unverified; fake streamed-provider persistence/cancellation coverage passed.

- D01: add independently verified LoftQ and EVA preparation flows; keep OFT/QOFT unavailable unless their model-loader and checkpoint behavior can be proven.
- D02-D03/D09-D11: install and verify package-specific GaLore, APOLLO, BAdam, Adam-mini, and Muon adapters with finite-gradient, checkpoint, cancellation, and estimate evidence before enabling each descriptor.
- D04-D05: execute FlashAttention and Liger on supported CUDA hardware. The current host has no CUDA and the optional packages are absent.

- C01: durably pin effective policy/reference revisions before launch and reuse those identities on resume; loader config commits are captured when available, but the requested remote revision can still be mutable. Local dataset preparation fingerprints are not a full immutable run snapshot.
- C02-C06: tiny jobs passed on TRL 0.19.1; the remainder of the declared >=0.9,<0.24 range and packaged Torch 2.11 stack remain unverified.
- C07: full/LoRA/DoRA train/save/reload and registered scalar scoring passed on CPU; CUDA QLoRA acceptance remains unverified.
- C08: base-versus-trained evaluation regression, exact interrupted-resume equivalence, and additional runtime-version telemetry coverage remain open.
- B02-B03: add isolated loader/forward/backward preflight evidence and finish extraction of shared native orchestration.
- B04: persist loader-time Unsloth-to-native fallback as durable effective-loader evidence.
- B05-B10: add real Transformers/PEFT fixtures, GPU matrix evidence, held-out continued-pretraining evaluation, exact interrupted-resume equivalence, and HF AutoConfig scratch initialization.
- B11, B13, and B16-B18 remain open; B12/B14/B15 remain partial pending full environment/model-card and installed-LLaMA-Factory contract evidence.
- Previously documented Phase A cache, immutable binding, packing, split/mixture, leakage, and redaction work remains open.

## Known failures

### Existing defects or incomplete behavior

- Optional real CUDA providers, llama.cpp converter/quantizer, Ollama and Hub publication are unverified on this CPU fixture host. Their absence is separate from code regressions; unsupported advanced operations are explicitly gated.
- Remote adapters without the original immutable base commit now fail safe merging with an actionable error. Existing records remain readable; an explicit matching base_revision can be supplied for export.

- Phase D external optimizer descriptors deliberately remain unavailable. Metadata detection alone is insufficient to claim executable GaLore, APOLLO, BAdam, Adam-mini, or Muon support.

- Metadata preflight still does not prove a model can complete an isolated forward/backward step.
- A loader-time Unsloth fallback is logged but does not update a durable effective-loader snapshot.
- Continued pretraining does not yet pass a held-out evaluation dataset to TRL.
- The lean environment intentionally lacks ML packages. Step 16 additionally used a separate CPU ML environment and executed real alignment jobs; this does not establish production GPU compatibility or coverage of every supported dependency version.
- LLaMA-Factory is absent in the verification environment, so its subprocess bridge is contract-tested through discovery/config behavior but not executed end to end.
- Reference lineage includes resolved paths, retained separate adapters, and config commits when available; effective remote identities are not durably pinned across enqueue/resume.
- Hard cancellation can interrupt checkpoint writes. Automatic native alignment resume skips missing/empty state or weight files, but does not prove every nonempty checkpoint tensor is intact.

### Existing baseline warnings

- Backend tests report two FastAPI/Starlette TestClient deprecation warnings from the installed dependency versions.
- Frontend ESLint reports one existing warning at `frontend/src/components/studio/BaseModelPicker.tsx:63` for the `preflight` effect dependency; zero errors.
- CPU runtime tests emit expected pin-memory warnings without an accelerator and TRL ORPO/SimPO/KTO column-retention warnings. These are not failures.

### Existing failures discovered and repaired during Step 16

- Optional scratch optimizer test referenced `lm_head`, while the existing scratch model uses `head`. Corrected the test only; no scratch implementation change was required.
- Existing alignment defects listed in Completed tasks were reproduced by the new runtime/validation fixtures. Their fixes pass final verification; they are not reported as new regressions.

### New failures caused by this work

None in final verification. The first frontend test command used a repository-relative path from the frontend directory and found no files; the corrected RunConfig test command passed. The earlier Step 15 local-resolution regression remains fixed.

- A repeated full CPU-ML suite run stopped making progress after its active fixtures and was terminated. This did not report a test failure; focused native optimization coverage passed, and the full lean suite passed. Treat the broad CPU-ML rerun as unverified until its fixture shutdown behavior is investigated.

## Architecture decisions made

- SLM Kit remains the orchestration and lineage owner. Native Transformers/TRL/PEFT remains the broad compatibility path; Unsloth and LLaMA-Factory remain optional engines.
- Backend selection occurs before enqueue and cannot change the requested training semantics. One recorded decision explains the selected backend and every rejected alternative.
- LLaMA-Factory data/config structures stay at the backend boundary; database tables do not model its internals.
- Alignment objectives are strategies inside one shared trainer lifecycle. Objective-specific runtime imports cannot disable unrelated objectives.
- Reward modeling shares alignment lifecycle, callbacks, checkpoints, and lineage, while its scalar model loader and evaluation runtime remain separate from causal generation.
- Artifact categories are normalized from existing string values at the model-reference boundary; database storage remains backward compatible.
- Canonical dataset adapters own normalization; tokenizer code owns rendering; model loaders own quantization through one shared BitsAndBytes translator.
- Reference-free objectives allocate no reference model. Base/separate references are included in the fit estimate; adapter-disabled references reuse the loaded PEFT model.
- Auto selection and frontend controls use the same optional capability keys for objective, DPO loss, and reference strategy. Unmapped external-engine semantics fail validation instead of being dropped.
- SimPO uses TRL's reference-free preference loss with CPO's SFT coefficient explicitly zero. PEFT classification heads are reserved for full train/save wrappers rather than receiving nested LoRA wrappers.
- Alignment uses existing canonical validation, content fingerprints, tokenizer rendering, module discovery, callbacks, artifact registration, process supervision, and resource leases; no parallel training engine was added.
- Heavy Torch/Transformers/TRL/PEFT imports remain inside worker execution paths.
- Advanced optimization metadata is import-light. Runtime hooks live in `train_entry/optimization_runtime.py`; a package is exposed only after its own validation and runtime fixture pass.

See `ARCHITECTURE_DECISIONS.md`, including ADR-16 through ADR-18 and the Step 16 clarifications.

## Database migration status

- Packaged head is `0004_serving_benchmarks`.
- Additive frozen migration `0004_serving_benchmarks` creates the durable benchmark table and indexes. Fresh and legacy migration tests pass.
- Existing projects, datasets, runs, checkpoints, artifacts, evaluations, and deployments retain their schema and references.
- New backend-resolution, alignment, and lineage data uses additive fields in the existing versioned Run config/artifact metadata JSON.
- Step 17 adds only optional RunConfig JSON fields (`runtime.neftune_noise_alpha` and `optim` strategy fields); no migration is required.

## Backend test status

### Steps 18–19 final verification (2026-09-17)

- Full lean backend: **165 passed, 4 skipped**, 2 existing test-client dependency warnings; `python -m pytest -q` from backend (23.54s). The additional skip is the optional export runtime fixture.
- Focused CPU-ML export/serving/reward suite: **21 passed**, 3 runtime/dependency warnings; `python -m pytest tests/test_export_runtime.py tests/test_reward_model_contract.py tests/test_phase_e_exports_serving.py tests/test_openai_gateway.py -q` with HF offline mode (26.29s). Real tiny PEFT merge/reload output equivalence, tokenizer/low-bit rejection and local base fingerprinting executed on Torch 2.7.1+cpu, Transformers 4.52.4 and PEFT 0.15.2.
- Real copy-export subprocess registration/progress/lineage passed. Fixtures verify failed-worker/cancellation logs, credential redaction, tree-termination invocation, lease release, restart recovery, installed/unavailable SGLang lifecycle, unsupported flags, native SSE/reward gates and provider gateway forwarding.
- Backend Ruff and Python compilation: **passed**. No configured Python static type checker. Frontend TypeScript/Vite build: **passed** (2519 modules); frontend tests: **18 passed**; ESLint: **0 errors, 1 pre-existing BaseModelPicker dependency warning**. No new regression remains.
- No external Hub publication, real llama.cpp/Ollama, vLLM/SGLang installation or CUDA model job ran. These are unverified paths, not passing runtime tests. Database schema/migration revisions unchanged; existing migration fixtures remain in the passing backend suite.

### Step 20 verification (2026-09-18)

- Focused benchmark/Ollama/export suite: **18 passed**, including deterministic streamed timing/percentile/VRAM persistence, mismatch rejection and Modelfile policy rendering.
- Fresh/legacy migration plus Step 20 suite: **10 passed**. Full backend initially exposed three stale revision/nullability test expectations; those tests were updated for the additive frozen migration and the model/migration nullability contract was aligned.
- Backend Ruff: passed. Frontend TypeScript/Vite production build: passed. The existing ESLint warning is unchanged.

### Prior-phase verification history

Environments: separate Python 3.11 lean and CPU-ML `uv` environments outside the repository. The repository `.venv` is a Linux layout and was not used on Windows. No mandatory ML dependency or GPU constraint changed.

- Historical Step 15 lean baseline: **135 passed, 1 skipped**. Step 16's first broader run exposed the existing scratch test attribute error and alignment defects recorded above; both categories are separated from new fixture mistakes.
- Focused alignment/runtime/backend-resolution/reward review before the final four runtime cases: **42 passed**, 16 runtime warnings.
- Final full CPU-ML backend suite: **165 passed**, 21 warnings, 50.33s; `python -m pytest backend/tests -q --disable-warnings` with Hub/Datasets offline mode.
- Final full lean backend suite: **144 passed, 2 skipped**, 2 dependency warnings, 9.84s. The optional alignment runtime module and scratch Torch fixture skip cleanly without ML dependencies.
- Local fixture stack: Torch **2.7.1+cpu**, Transformers **4.52.4**, TRL **0.19.1**, PEFT **0.15.2**, Datasets **3.6.0**, Accelerate **1.7.0**. `torch.cuda.is_available()` is false and Torch CUDA is absent. Torch 2.7.1 is an isolated CPU fixture choice, outside the packaged GPU extra's >=2.11,<2.12 constraint; that packaged stack remains unverified.
- Runtime fixtures are marked `alignment_runtime`; run with `python -m pytest backend/tests/test_alignment_runtime.py -q` in an installed compatible test environment. They create local tokenizer/model weights and need no model downloads.
- Backend Ruff: **passed** across `backend/app` and `backend/tests`.
- Python bytecode compilation: **passed**. No mypy or pyright configuration exists.
- Real native Transformers/PEFT/TRL CPU jobs ran. No CUDA, Unsloth, model download, or installed LLaMA-Factory process ran.
- Step 17 focused CPU-ML suite: **14 passed**, including a tiny PiSSA + LoRA+ forward/backward/optimizer step. Full lean suite: **149 passed, 3 skipped** (optional ML runtime cases skip cleanly). Frontend suite: **18 passed**. Frontend build passed; lint has the pre-existing BaseModelPicker dependency warning and no errors. Backend Ruff passed after the final import correction.

## Frontend build status

- Frontend unit tests: **4 files / 18 tests passed**; `npm test -- --reporter=dot`.
- TypeScript/Vite production build: **passed**, 2519 modules.
- ESLint: **zero errors, one existing warning**; `npm run lint`.

## Blocking issues

No regression remains from this slice. Optional GPU/provider/converter/Ollama acceptance needs compatible separately installed runtimes. Benchmark storage adds frozen revision `0004_serving_benchmarks`; no existing records are modified. No backend static type checker is configured; Python compilation and typed API contracts were checked, and TypeScript typechecking runs in the frontend build.

## Recommended next task

**E05 — Real GGUF exporter acceptance.** In an isolated environment with a built llama.cpp converter/quantizer, convert a tiny supported full model, reload its quantized output, verify lineage/logs and cancel a running converter tree. Record runtime versions before marking E05 complete. Do not start Phase F.
