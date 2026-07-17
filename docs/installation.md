# Installation (full guide)

This is the careful, everything-explained version. If you just want to get
going, use the [Quickstart](quickstart.md).

SLM Kit has two parts you install separately:

```
slm-kit/
├── backend/    Python (FastAPI) — the brain: training, data, registry, APIs
└── frontend/   React (Vite)    — the face: the web UI you click around in
```

---

## 1. Backend

### 1.1 Create a virtual environment

Inside `backend/` (WSL2/Linux recommended for training):

```bash
uv venv --python 3.11
source .venv/bin/activate
```

(Plain `python -m venv .venv` works too if you don't want uv.)

### 1.2 Install the core

```bash
uv pip install -e .
```

This is enough to run the API, the UI, dataset validation, fit estimates,
the registry, and Hugging Face listing. **No CUDA required.**

### 1.3 Install the training stack (GPU machines)

**Linux / WSL2** (recommended — PyPI's Linux torch wheels include CUDA):

```bash
uv pip install -e ".[gpu]"
```

Downloads several GB (PyTorch + CUDA wheels + Unsloth).

**Native Windows** (verified working on the RTX 4060 reference machine):
PyPI's *Windows* torch wheels are **CPU-only**, so install the CUDA build from
the PyTorch index *first*, then the rest:

```powershell
pip install "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "xformers==0.0.35" --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[gpu]"
```

(unsloth pins `torch<2.11`, so 2.10.0 is the newest it supports; `triton-windows`
and `bitsandbytes` install automatically. Requires an NVIDIA driver with
CUDA ≥ 12.8 support — driver 570+.) See the
[Windows-native quickstart](quickstart.md#windows-native--no-wsl2) for the
verify commands.

### 1.4 Optional extras

```bash
uv pip install -e ".[eval]"   # ROUGE / BLEU metrics for the Eval Lab
uv pip install pyarrow        # only if you'll upload .parquet datasets
```

### 1.5 Run it

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- First boot creates the data home at `~/.slmkit/` (DB, datasets, models,
  run folders) and installs the 10 bundled sample datasets.
- Check it's alive: open `http://127.0.0.1:8000/api/health` → `{"status":"ok"}`.
- Auto-reload during development: add `--reload`.

### 1.6 Where your data lives

```
~/.slmkit/
├── slmkit.db     SQLite — runs, datasets, artifacts, eval results
├── datasets/       uploaded + sample data files
├── models/         imported models, GGUF exports
├── runs/           one folder per run: config.json, checkpoints/, output/, metrics.jsonl
└── hf-cache/       Hugging Face download cache
```

Delete this folder to fully reset the app. Override the location with the
`SLMKIT_HOME` environment variable.

---

## 2. Frontend

Needs Node.js 20+.

```bash
cd frontend
npm install
npm run dev          # development server at http://localhost:5173
```

The dev server **proxies** `/api` and `/ws` to the backend at `127.0.0.1:8000`,
so you don't configure any URLs — just have the backend running.

Production build (optional):

```bash
npm run build        # outputs static files to frontend/dist/
npm run preview      # serve the built app locally
```

---

## 3. Optional integrations

### 3.1 Hugging Face token (publish / import / private models)

1. Create a token at <https://huggingface.co/settings/tokens> (type: *Write*).
2. Give it to the backend, either way:
   ```bash
   export SLMKIT_HF_TOKEN=hf_xxx          # env var
   # or put SLMKIT_HF_TOKEN=hf_xxx in backend/.env
   ```
3. Restart the backend. The Model Registry now shows "HF token configured" and
   lists your repos.

### 3.2 llmfit (better hardware-fit scoring)

Install per its README (<https://github.com/AlexsJones/llmfit>) so the `llmfit`
binary is on your `PATH`. That's it — SLM Kit detects it automatically and
uses `llmfit fit --json` / `llmfit recommend --json`. If it's missing, the
built-in estimator takes over silently; the Dashboard shows which one is active.

A custom binary path can be set with `SLMKIT_LLMFIT_BIN=/path/to/llmfit`.

### 3.3 llama.cpp (GGUF export)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release
export SLMKIT_LLAMACPP_DIR=$PWD          # folder containing convert_hf_to_gguf.py
```

Restart the backend; the Registry's GGUF button becomes active.
Note: GGUF conversion needs a **full/merged** model folder. Full fine-tunes and
from-scratch models convert directly; LoRA/QLoRA adapters must be merged first.

### 3.4 LLM-as-judge (optional eval scoring)

```bash
export SLMKIT_JUDGE_PROVIDER=anthropic     # or: openai
export SLMKIT_JUDGE_API_KEY=sk-...
export SLMKIT_JUDGE_MODEL=claude-sonnet-5  # any model id your key can use
```

The Eval Lab's "LLM-as-judge" checkbox unlocks when a key is set. Without a
key, all automatic metrics still work.

---

## 4. Verify the install

1. Backend health: `curl http://127.0.0.1:8000/api/health` → `{"status":"ok",...}`
2. GPU seen: `curl http://127.0.0.1:8000/api/system/hardware` → your GPU name
   and `"source": "pynvml"` (if it says `"fallback"`, the GPU isn't visible).
3. Open `http://localhost:5173` → resource strip shows live VRAM/CPU/RAM.
4. Dataset Manager shows 10 sample datasets.
5. Run the [first fine-tune walkthrough](quickstart.md#your-first-fine-tune-5-minutes-of-clicking).
