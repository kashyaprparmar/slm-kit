# API reference

The backend is a FastAPI app on `http://127.0.0.1:8000`. Interactive docs are
always available at **`/docs`** (Swagger) and **`/redoc`** while the server runs.

All bodies/responses are JSON. Endpoints are grouped by router.

---

## System

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness: `{"status":"ok","version":...}` |
| GET | `/api/system/hardware` | Live GPU/CPU/RAM/disk snapshot (`source`: `pynvml`, `cuda-fallback`, or `fallback`) |
| GET | `/api/system/status` | llmfit availability, HF/judge configured, current run id, data home, VRAM budget |
| GET | `/api/system/diagnostics` | Python/GPU/library/integration checks with setup guidance |
| GET | `/api/system/database` | Typed schema revision, WAL/FK settings, SQLite quick check and bounded orphan-reference report |
| GET | `/api/system/services` | API, SQLite, queue worker, GPU owner, and isolated provider health |
| GET | `/api/system/activity` | Structured activity history with correlation IDs and durations |
| GET | `/api/system/logs?lines=200` | Tail the persistent backend log (maximum 2,000 lines) |

## Datasets

| Method | Path | Description |
|---|---|---|
| GET | `/api/datasets` | List all datasets |
| GET | `/api/datasets/{id}` | One dataset row |
| GET | `/api/datasets/{id}/preview` | Validation report + stats + token histogram + sample rows |
| GET | `/api/datasets/{id}/schema?limit=20` | Typed read-only canonical schema sample; limit 1–100, advanced formats explicitly gated |
| POST | `/api/datasets/upload` | Multipart upload (`file`, `kind`, optional `name`) → validates + stores |
| POST | `/api/datasets/install-samples` | Install/refresh the 10 bundled samples |
| DELETE | `/api/datasets/{id}` | Delete a dataset (samples are protected) |
| POST | `/api/datasets/{id}/prepare` | Stream an immutable, fingerprinted recipe into deterministic train/validation/test datasets |
| GET | `/api/datasets/hf-capabilities` | Whether optional Hugging Face dataset import is available |
| POST | `/api/datasets/import-hf` | Import a supported Hugging Face dataset into the local catalog |
| GET | `/api/datasets/{id}/versions` | List immutable source/output versions newest first |
| GET | `/api/dataset-versions/{id}` | One version and its cached profiles |
| GET | `/api/dataset-recipes?dataset_id={id}` | List preparation recipes sourced from a dataset |
| GET | `/api/dataset-recipes/{id}` | Recipe plus persisted split lineage |
| POST | `/api/dataset-recipes/{id}/clone` | Return source dataset and editable recipe config without running it |
| POST | `/api/dataset-recipes/{id}/rerun` | Deterministically replay, or reuse, an immutable recipe |
| GET | `/api/tokenizer-artifacts` | List tokenizer identities resolved by profiling |
| POST | `/api/dataset-versions/{id}/tokenizer-profile` | Isolated exact-tokenizer profile, preview, and loss mask; cached by fingerprints/config |
| POST | `/api/dataset-versions/{id}/quality-profile` | Cached, non-destructive quality/leakage scan with example evidence |

## Projects

| Method | Path | Description |
|---|---|---|
| GET | `/api/projects` | List workspaces with linked-run counts |
| POST | `/api/projects` | Create `{name, description?, state?}`; names are trimmed and cannot be blank |
| GET | `/api/projects/{id}` | Project metadata and its linked runs |
| PATCH | `/api/projects/{id}` | Update name, description, or arbitrary workspace state |
| POST | `/api/projects/{id}/runs/{run_id}` | Attach an existing run to a project |
| DELETE | `/api/projects/{id}` | Delete workspace metadata and links; runs/artifacts are retained |

## Runs (training)

| Method | Path | Description |
|---|---|---|
| GET | `/api/runs/backends` | Registered backends + typed task/stage/method/tokenizer/PEFT/quantization/dependency capabilities; legacy arrays retained |
| POST | `/api/runs/estimate` | Fit estimate + validation for a `RunConfig` (no launch) |
| GET | `/api/runs` | List runs (newest first) |
| GET | `/api/runs/{id}` | Run + checkpoints + queue position |
| POST | `/api/runs` | Create + enqueue a run (`422` if it won't fit / invalid) |
| POST | `/api/runs/{id}/cancel` | Cancel a queued/running run (frees VRAM) |
| DELETE | `/api/runs/{id}` | Safely delete an inactive run without registered artifacts |
| POST | `/api/runs/{id}/clone` | Return a copy of the config (name suffixed `-clone`) |
| POST | `/api/runs/{id}/rerun` | Re-queue a new run from stored config |
| GET | `/api/runs/{id}/export-config` | Config in an engine format (`{format,filename,content}`) |
| GET | `/api/runs/{id}/metrics` | Full metric history (for replotting): `{"metrics":[...]}` |
| GET | `/api/runs/{id}/logs?lines=2000` | Persisted run logs, including completed and failed runs |

### `RunConfig` (the body for estimate / create)

```jsonc
{
  "schema_version": 1,           // optional in requests; defaults to current v1
  "backend": "unsloth",          // "unsloth" | "transformers" | "scratch"
  "task": "finetune",            // "finetune" | "continued_pretrain" | "pretrain"
  "method": "qlora",             // "lora" | "qlora" | "dora" | "full"
  "base_model": "unsloth/Qwen2.5-0.5B-Instruct",  // HF id or local path ("" for scratch)
  "revision": "main",            // preferably pin a commit hash
  "dataset_id": 3,
  "output_name": "my-run",
  "load_in_4bit": true,
  "lora":  { "r": 16, "alpha": 16, "dropout": 0.0, "use_dora": false },
  "optim": { "learning_rate": 2e-4, "lr_scheduler": "cosine", "warmup_ratio": 0.03,
             "optimizer": "adamw_8bit" },
  "train": { "epochs": 1, "max_steps": null, "per_device_batch_size": 2,
             "gradient_accumulation": 4, "max_seq_length": 1024,
             "save_steps": 100, "logging_steps": 5 },
  "tokenizer": { "mode": "reuse", "chat_template": null,
                 "loss_policy": "full_sequence",
                 "added_tokens": [], "added_special_tokens": [] },
  "arch":  { "vocab_size": 8192, "n_layers": 6, "n_heads": 6,
             "n_embd": 384, "block_size": 256, "dropout": 0.1 }, // pretrain only
  "extra": { "project_id": 4 } // optional project association
}
```

Backend descriptors carry evidence-bearing capability objects. The `tasks`,
`methods`, `quantization`, and `requirements` arrays remain for existing
clients; new clients should use `task_capabilities`, `method_capabilities`,
`stages`, `tokenizer`, `peft`, and `quantization_capabilities`.

Estimate and create use the same dataset, model-capability, context-length, and
backend validation. A stale `project_id` is rejected with `422`. Run creation
and project association commit together, so a failed link cannot leave a
stranded queued run.

## Advisor

| Method | Path | Description |
|---|---|---|
| POST | `/api/advisor/recommend` | Rank base-model+method combos. Body: `{task, method, priority: "fastest"|"balanced"|"best_quality", max_seq_length}` |
| POST | `/api/advisor/settings` | Deterministic presets with an explanation for each selected value |

## Registry

| Method | Path | Description |
|---|---|---|
| GET | `/api/registry/capabilities` | Export, publish, and deployment capabilities/status |
| GET | `/api/registry/model-options` | Loadable runs/artifacts with stable refs and model kind |
| GET | `/api/registry/local` | Local artifacts + finished-but-unpublished runs |
| POST | `/api/registry/inspect` | Inspect architecture, tokenizer, revision and the complete capability map without loading weights |
| GET | `/api/registry/hf` | Your HF repos (needs token) + `token_set` |
| POST | `/api/registry/publish` | `{run_id, repo_id, private}` → uploads with an auto model card |
| POST | `/api/registry/import` | `{repo_id}` → download into local models dir |
| POST | `/api/registry/quantize` | `{run_id?, artifact_id?, quant_type}` → background GGUF job (`202`) |
| GET | `/api/registry/quantize/{artifact_id}/logs` | Persisted GGUF export log |
| GET | `/api/registry/deployment` | Current deployment, endpoint, kind, PID, and recent logs |
| POST | `/api/registry/deploy` | `{model_ref}` → start the local OpenAI-compatible model server |
| DELETE | `/api/registry/deployment` | Stop serving and release model memory |
| POST | `/api/registry/merge` | Merge a PEFT adapter into a standalone model in an isolated subprocess |
| DELETE | `/api/registry/artifacts/{id}` | Safely remove an inactive artifact and registry-owned files |

## Model serving providers

| Method | Path | Description |
|---|---|---|
| GET | `/api/serving/providers` | Concurrent Transformers, Ollama, and vLLM status/capability probes |
| POST | `/api/serving/ollama/start` | Keep an installed Ollama model loaded under the shared GPU lease |
| POST | `/api/serving/ollama/stop` | Unload the SLM Kit-managed Ollama model |
| POST | `/api/serving/ollama/import` | Import a ready GGUF artifact through the native Ollama CLI |
| POST | `/api/serving/test` | Test `transformers`, `ollama`, or `vllm` through one response shape |

Model references may be HF repo ids, local directories, or `run:<id>`. The
stable run form is recommended because it correctly identifies adapter and
scratch outputs. The deployed inference API is served on
`http://localhost:8802/v1`; see [Local deployment](deployment.md).

## Eval Lab

| Method | Path | Description |
|---|---|---|
| GET | `/api/eval/status` | `{busy, judge_available}` |
| POST | `/api/eval/cancel` | Cancel the current generation/evaluation subprocess and release its lease |
| GET | `/api/eval/logs/{kind}/{id}` | Persisted logs for `eval` or `gen` work |
| POST | `/api/eval/generate` | Playground: `{model_ref, prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty}` → `{gen_id}` (409 if GPU busy) |
| POST | `/api/eval/run` | Eval: `{models:[...], dataset_id, max_samples, metrics:[...], judge}` → `{eval_id}` (409 if busy; 400 if judge requested without a key) |
| GET | `/api/eval/results` | All eval results |
| GET | `/api/eval/generation/{gen_id}` | Completed generation output/status |
| GET | `/api/eval/results/{id}` | One eval result (scores + sample comparisons) |

## WebSockets

Connect and receive JSON events (client messages are ignored / keepalive).

| Path | Streams |
|---|---|
| `/ws/system` | hardware telemetry + queue state |
| `/ws/activity` | structured application activity and request spans |
| `/ws/runs/{id}` | `log` · `metric` · `checkpoint` · `sample` · `status` |
| `/ws/eval/{id}` | `log` · `progress` · `model_done` · `result` · `status` |
| `/ws/gen/{id}` | `log` · `token` · `done` · `error` |
| `/ws/gguf/{artifact_id}` | GGUF export log and artifact status |

### Event shapes (runs)

```jsonc
{"type":"metric","step":10,"total_steps":100,"metrics":{"loss":1.83,"tokens_per_sec":1450,"eta_seconds":42}}
{"type":"checkpoint","step":100,"path":".../checkpoints/checkpoint-100","is_final":false}
{"type":"sample","step":200,"text":"..."}
{"type":"status","status":"done"}   // or "failed" (+ "detail")
```

## Status codes you'll see

| Code | Meaning |
|---|---|
| `200` / `201` / `202` | ok / created / accepted (background job started) |
| `400` | bad input (e.g. judge without key, malformed repo id) |
| `404` | not found |
| `409` | GPU busy — training, eval, generation, merge, deployment, vLLM, or Ollama currently owns it |
| `422` | run config invalid or won't fit (VRAM guard) — body has the validation report |
| `502` | an external call failed (HF upload/import) |
