# Configuration

All settings are **environment variables** with the prefix `SLMKIT_`, or the
same names in a `.env` file inside `backend/` (see `backend/.env.example`).
**Everything is optional** — the app runs fully offline with zero configuration.

## Core

| Variable | Default | What it does |
|---|---|---|
| `SLMKIT_HOME` | `~/.slmkit` | Where all data lives (DB, datasets, models, runs) |
| `SLMKIT_HOST` | `127.0.0.1` | API bind address |
| `SLMKIT_PORT` | `8000` | API port |
| `SLMKIT_CORS_ORIGINS` | `localhost:5173` origins | Allowed browser origins (JSON list) |

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
| `SLMKIT_VRAM_BUDGET_MB` | `8192` | Planning budget when no GPU is detected |
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

Restart the backend after changing configuration.

## How "fits / tight / won't fit" is decided

1. Detect usable VRAM (live via pynvml, or `SLMKIT_VRAM_BUDGET_MB`).
2. Multiply by `SLMKIT_VRAM_SAFE_FRACTION` (default 90%) → the *safe budget*.
3. Predict the run's peak VRAM (llmfit if installed, else the built-in
   estimator: weights + optimizer + activations + KV cache + ~15% overhead).
4. Verdict: ≤ 80% of safe budget → **fits** · ≤ 100% → **tight** ·
   above → **won't fit** (launch blocked with an explanation).
