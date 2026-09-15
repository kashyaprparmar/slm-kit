# Configuration

All settings are **environment variables** with the prefix `SLMKIT_`, or the
same names in a `.env` file inside `backend/` (see `backend/.env.example`).
**Everything is optional** — the app runs fully offline with zero configuration.

## Core

| Variable | Default | What it does |
|---|---|---|
| `SLMKIT_HOME` | `~/.slmkit` (native) / `/data/slmkit` (Docker) | Where all data lives (DB, datasets, models, runs) |
| `SLMKIT_HOST` | `127.0.0.1` (native) / `0.0.0.0` (Docker) | API bind address |
| `SLMKIT_PORT` | `8000` | API port |
| `SLMKIT_CORS_ORIGINS` | `localhost:5173` origins | Allowed browser origins (JSON list) |
| `SLMKIT_MAX_UPLOAD_MB` | `512` | Maximum dataset upload size; uploads are streamed to disk |

## Hugging Face

| Variable | Default | What it does |
|---|---|---|
| `SLMKIT_HF_TOKEN` | *(unset)* | Enables publish, import, your-repos listing, private models. Plain `HF_TOKEN` is also honored. |

## LLM-as-judge (Eval Lab, optional)

| Variable | Default | What it does |
|---|---|---|
| `SLMKIT_JUDGE_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `SLMKIT_JUDGE_API_KEY` | *(unset)* | Unlocks the judge checkbox in the Eval Lab |
| `SLMKIT_JUDGE_MODEL` | `claude-sonnet-5` | Which model does the judging |

## Model loading and deployment

| Variable | Default | What it does |
|---|---|---|
| `SLMKIT_TRUST_REMOTE_CODE` | `false` | Allow executable custom code from trusted HF model repositories |
| `SLMKIT_DEPLOY_HOST` | `127.0.0.1` | Managed model-server bind address (Docker overrides it inside the container) |
| `SLMKIT_DEPLOY_PORT` | `8802` | OpenAI-compatible local model-server port |
| `SLMKIT_DEPLOY_STARTUP_TIMEOUT_SECONDS` | `180` | Maximum model-load wait before deployment fails |

### External inference providers

| Variable | Default | What it does |
|---|---|---|
| `SLMKIT_SERVE_ENGINE` | `auto` | Provider selection: managed Transformers or `vllm` (`auto` selects vLLM when available) |
| `SLMKIT_VLLM_URL` | `http://127.0.0.1:8801` | OpenAI-compatible vLLM endpoint |
| `SLMKIT_VLLM_PORT` | `8801` | Host port used by the optional compose vLLM service |
| `SLMKIT_VLLM_GPU_MEMORY_UTILIZATION` | `0.55` | vLLM GPU memory fraction when the optional service starts |
| `SLMKIT_VLLM_MAX_MODEL_LEN` | `4096` | Maximum vLLM context length |
| `SLMKIT_VLLM_IDLE_TIMEOUT_SECONDS` | `600` | Idle timeout used by provider lifecycle checks |
| `SLMKIT_OLLAMA_URL` | `http://127.0.0.1:11434` | External Ollama endpoint exposed in provider status |

The backend uses `SLMKIT_VLLM_URL` to reach an already-running provider. The
compose file overrides this inside the backend container to `http://vllm:8000`
when the `vllm` profile is enabled; `SLMKIT_VLLM_MODEL` and the compose
memory-fraction variable configure the vLLM container itself.

## External tools

| Variable | Default | What it does |
|---|---|---|
| `SLMKIT_LLMFIT_BIN` | `llmfit` | Name/path of the llmfit binary. Missing → built-in estimator used automatically. |
| `SLMKIT_LLAMACPP_DIR` | *(unset)* | Path to a built llama.cpp checkout. Unset → GGUF export explains what to install. |

## Hardware budget & polling

These tune the fit checker and the live resource strip. The defaults match an
RTX 4060 8 GB / 16 GB RAM machine — change them if your box is different.

| Variable | Default | What it does |
|---|---|---|
| `SLMKIT_VRAM_BUDGET_MB` | `8192` | Planning budget when no usable GPU telemetry is available |
| `SLMKIT_RAM_BUDGET_MB` | `16384` | System RAM assumption |
| `SLMKIT_VRAM_SAFE_FRACTION` | `0.90` | Fraction of VRAM considered usable (headroom for the OS/driver) |
| `SLMKIT_IDLE_POLL_SECONDS` | `3.0` | Resource strip refresh when idle |
| `SLMKIT_ACTIVE_POLL_SECONDS` | `1.5` | Refresh while a job runs |

## Example `.env`

```bash
# backend/.env
SLMKIT_HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
SLMKIT_JUDGE_PROVIDER=anthropic
SLMKIT_JUDGE_API_KEY=sk-ant-xxxxxxxx
SLMKIT_LLAMACPP_DIR=/home/you/llama.cpp
```

Restart after changing configuration: `docker compose up -d --force-recreate backend`
(or restart native `uvicorn`). Compose loads `backend/.env` automatically.

## How "fits / tight / won't fit" is decided

1. Detect usable VRAM (live via NVML or CUDA fallback, or `SLMKIT_VRAM_BUDGET_MB`).
2. Multiply by `SLMKIT_VRAM_SAFE_FRACTION` (default 90%) → the *safe budget*.
3. Predict the run's peak VRAM (llmfit if installed, else the built-in
   estimator: weights + optimizer + activations + KV cache + ~15% overhead).
4. Verdict: ≤ 80% of safe budget → **fits** · ≤ 100% → **tight** ·
   above → **won't fit** (launch blocked with an explanation).
