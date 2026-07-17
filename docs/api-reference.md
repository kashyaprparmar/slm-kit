# API reference

The backend is a FastAPI app on `http://127.0.0.1:8000`. Interactive docs are
always available at **`/docs`** (Swagger) and **`/redoc`** while the server runs.

All bodies/responses are JSON. Endpoints are grouped by router.

---

## System

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness: `{"status":"ok","version":...}` |
| GET | `/api/system/hardware` | Live GPU/CPU/RAM/disk snapshot (`source`: `pynvml` or `fallback`) |
| GET | `/api/system/status` | llmfit availability, HF/judge configured, current run id, data home, VRAM budget |

## Datasets

| Method | Path | Description |
|---|---|---|
| GET | `/api/datasets` | List all datasets |
| GET | `/api/datasets/{id}` | One dataset row |
| GET | `/api/datasets/{id}/preview` | Validation report + stats + token histogram + sample rows |
| POST | `/api/datasets/upload` | Multipart upload (`file`, `kind`, optional `name`) → validates + stores |
| POST | `/api/datasets/install-samples` | Install/refresh the 10 bundled samples |
| DELETE | `/api/datasets/{id}` | Delete a dataset (samples are protected) |

## Runs (training)

| Method | Path | Description |
|---|---|---|
| GET | `/api/runs/backends` | Registered backends + their supported tasks/methods |
| POST | `/api/runs/estimate` | Fit estimate + validation for a `RunConfig` (no launch) |
| GET | `/api/runs` | List runs (newest first) |
| GET | `/api/runs/{id}` | Run + checkpoints + queue position |
| POST | `/api/runs` | Create + enqueue a run (`422` if it won't fit / invalid) |
| POST | `/api/runs/{id}/cancel` | Cancel a queued/running run (frees VRAM) |
| POST | `/api/runs/{id}/clone` | Return a copy of the config (name suffixed `-clone`) |
| POST | `/api/runs/{id}/rerun` | Re-queue a new run from stored config |
| GET | `/api/runs/{id}/export-config` | Config in an engine format (`{format,filename,content}`) |
| GET | `/api/runs/{id}/metrics` | Full metric history (for replotting): `{"metrics":[...]}` |

### `RunConfig` (the body for estimate / create)

```jsonc
{
  "backend": "unsloth",          // "unsloth" | "scratch"
  "task": "finetune",            // "finetune" | "continued_pretrain" | "pretrain"
  "method": "qlora",             // "lora" | "qlora" | "dora" | "full"
  "base_model": "unsloth/Qwen2.5-0.5B-Instruct",  // HF id or local path ("" for scratch)
  "dataset_id": 3,
  "output_name": "my-run",
  "load_in_4bit": true,
  "lora":  { "r": 16, "alpha": 16, "dropout": 0.0, "use_dora": false },
  "optim": { "learning_rate": 2e-4, "lr_scheduler": "cosine", "warmup_ratio": 0.03,
             "optimizer": "adamw_8bit" },
  "train": { "epochs": 1, "max_steps": null, "per_device_batch_size": 2,
             "gradient_accumulation": 4, "max_seq_length": 1024,
             "save_steps": 100, "logging_steps": 5 },
  "arch":  { "vocab_size": 8192, "n_layers": 6, "n_heads": 6,
             "n_embd": 384, "block_size": 256, "dropout": 0.1 }  // pretrain only
}
```

## Advisor

| Method | Path | Description |
|---|---|---|
| POST | `/api/advisor/recommend` | Rank base-model+method combos. Body: `{task, method, priority: "fastest"|"balanced"|"best_quality", max_seq_length}` |

## Registry

| Method | Path | Description |
|---|---|---|
| GET | `/api/registry/capabilities` | `{gguf_available, hf_token_set, quant_types}` |
| GET | `/api/registry/local` | Local artifacts + finished-but-unpublished runs |
| GET | `/api/registry/hf` | Your HF repos (needs token) + `token_set` |
| POST | `/api/registry/publish` | `{run_id, repo_id, private}` → uploads with an auto model card |
| POST | `/api/registry/import` | `{repo_id}` → download into local models dir |
| POST | `/api/registry/quantize` | `{run_id, quant_type}` → background GGUF job (`202`), status on the artifact |

## Eval Lab

| Method | Path | Description |
|---|---|---|
| GET | `/api/eval/status` | `{busy, judge_available}` |
| POST | `/api/eval/generate` | Playground: `{model_ref, prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty}` → `{gen_id}` (409 if GPU busy) |
| POST | `/api/eval/run` | Eval: `{models:[...], dataset_id, max_samples, metrics:[...], judge}` → `{eval_id}` (409 if busy; 400 if judge requested without a key) |
| GET | `/api/eval/results` | All eval results |
| GET | `/api/eval/results/{id}` | One eval result (scores + sample comparisons) |

## WebSockets

Connect and receive JSON events (client messages are ignored / keepalive).

| Path | Streams |
|---|---|
| `/ws/system` | hardware telemetry + queue state |
| `/ws/runs/{id}` | `log` · `metric` · `checkpoint` · `sample` · `status` |
| `/ws/eval/{id}` | `log` · `progress` · `model_done` · `result` · `status` |
| `/ws/gen/{id}` | `log` · `token` · `done` · `error` |

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
| `409` | GPU busy — a training/eval/generation job is already running |
| `422` | run config invalid or won't fit (VRAM guard) — body has the validation report |
| `502` | an external call failed (HF upload/import) |
