# Requirements

## Hardware

| Component | Minimum | Reference (what defaults are tuned for) |
|---|---|---|
| GPU | NVIDIA, 6 GB VRAM | **RTX 4060, 8 GB VRAM** |
| RAM | 16 GB | 16 GB |
| CPU | 4 cores | Ryzen 7000, 8 cores / 16 threads |
| Disk | 30 GB free | 50 GB+ (models are big: a 7B base ≈ 5 GB in 4-bit) |
| Internet | only for downloading base models / publishing | — |

**No NVIDIA GPU?** The app still runs: dataset tools, fit estimates, registry,
run history, and (slowly, on CPU) tiny from-scratch pretraining. Unsloth
training (LoRA/QLoRA/DoRA) requires an NVIDIA GPU.

### What fits on 8 GB VRAM?

| Base model size | QLoRA fine-tune | LoRA (fp16) | Full fine-tune |
|---|---|---|---|
| 0.3B – 1B | ✅ comfortable | ✅ comfortable | ⚠️ tiny models only |
| 1B – 3B | ✅ comfortable | ✅ / ⚠️ | ❌ |
| 3B – 4B | ✅ | ⚠️ tight | ❌ |
| 7B | ⚠️ tight (seq ≤ 1024, batch 1–2) | ❌ | ❌ |

The app checks this for you before every launch (**fits / tight / won't fit**)
and blocks configs that would clearly OOM.

## Software

| Software | Version | Needed for |
|---|---|---|
| Windows 11 + **WSL2 Ubuntu 22.04+**, native Linux, or native Windows (see quickstart Step 2b) | — | training backend |
| NVIDIA driver (Windows/Linux host) | recent, WSL-CUDA capable | GPU passthrough |
| Python | **3.11** | backend |
| [uv](https://github.com/astral-sh/uv) | latest | Python installs (fast) |
| Node.js | **20+** | frontend |
| CUDA toolkit | *not needed separately* — PyTorch wheels bundle it | — |

## Python packages (installed for you)

- **Base** (`uv pip install -e .`): FastAPI, SQLModel, huggingface-hub, psutil,
  pynvml, httpx — runs the API without any GPU stack.
- **`[gpu]`**: torch, transformers, datasets, tokenizers, accelerate, peft, trl,
  bitsandbytes, **unsloth** — the actual training stack. Linux/WSL2 only.
- **`[eval]`**: rouge-score, sacrebleu, nltk — extra metrics for the Eval Lab
  (exact-match and token-F1 work without this).
- **pyarrow** (optional): only if you upload `.parquet` datasets.

## Optional external tools

| Tool | What it adds | Without it |
|---|---|---|
| [llmfit](https://github.com/AlexsJones/llmfit) | hardware-aware model fit scoring | built-in VRAM estimator is used (automatic fallback) |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF export in the Registry | GGUF button explains what to install |
| Hugging Face token | publish / import / private repos | local-only registry still works |
| Anthropic/OpenAI API key | LLM-as-judge scoring in the Eval Lab | automatic metrics still work |
