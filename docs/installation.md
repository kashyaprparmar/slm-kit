# Installation (full guide)

The recommended install is **Docker**. If you just want to get going, use the
[Quickstart](quickstart.md). This page explains the Docker layout, optional
secrets, and the native (no-Docker) path.

```
slm-kit/
├── docker-compose.yml       GPU stack (default)
├── docker-compose.cpu.yml   UI only, no torch
├── docker-compose.dev.yml   hot-reload (CPU backend + Vite)
├── backend/                 FastAPI + training
│   ├── Dockerfile           GPU image
│   └── Dockerfile.cpu
└── frontend/                React UI
    ├── Dockerfile           nginx production build
    └── Dockerfile.dev
```

---

## 1. Docker (recommended)

### 1.1 Install Docker

- Windows / macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Linux: Docker Engine + the Compose plugin

For **GPU training** on Windows, enable **Settings → Resources → GPU** and
install a recent NVIDIA driver. Confirm:

```powershell
docker info
nvidia-smi
```

### 1.2 Start

From the project root:

```powershell
# training (Unsloth / QLoRA) — first build is several GB
docker compose up --build -d

# or UI-only
docker compose -f docker-compose.cpu.yml up --build -d
```

Open **http://localhost:5173**. API: **http://localhost:8000/api/health**.

Details and everyday commands: [Docker](docker.md).

### 1.3 Where your data lives

Inside the backend container, `SLMKIT_HOME` is `/data/slmkit`, backed by the
named volume `slmkit-data`:

```
/data/slmkit/
├── slmkit.db       SQLite — runs, datasets, artifacts, eval results
├── datasets/       uploaded + sample data files
├── models/         imported models, GGUF exports
├── runs/           one folder per run
└── hf-cache/       Hugging Face download cache
```

```powershell
docker compose down          # stop, keep data
docker compose down -v       # stop and wipe the volume (full reset)
```

### 1.4 Optional secrets

Copy `backend/.env.example` → `backend/.env`. Compose loads it if present.

```bash
SLMKIT_HF_TOKEN=hf_xxx
SLMKIT_JUDGE_PROVIDER=anthropic
SLMKIT_JUDGE_API_KEY=sk-ant-xxx
```

Then `docker compose up -d --force-recreate backend`. See
[Configuration](configuration.md).

### 1.5 Verify

1. `curl http://127.0.0.1:8000/api/health` → `{"status":"ok",...}`
2. GPU stack: `curl http://127.0.0.1:8000/api/system/hardware` → your GPU name
   and `"source": "pynvml"`
3. Open `http://localhost:5173` → resource strip shows live VRAM/CPU/RAM
4. Dataset Manager shows 10 sample datasets
5. Run the [first fine-tune walkthrough](quickstart.md#3-your-first-fine-tune-5-minutes-of-clicking)

---

## 2. Optional integrations

These work the same in Docker (put them in `backend/.env`) or natively
(export or `.env`).

### 2.1 Hugging Face token (publish / import / private models)

1. Create a *Write* token at <https://huggingface.co/settings/tokens>
2. Set `SLMKIT_HF_TOKEN=hf_xxx` in `backend/.env`
3. Recreate the backend container (or restart native uvicorn)

### 2.2 llmfit (better hardware-fit scoring)

Install per its README (<https://github.com/AlexsJones/llmfit>) so `llmfit` is
on `PATH` **inside the environment that runs the backend**. Missing → the
built-in estimator is used automatically.

### 2.3 llama.cpp (GGUF export)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release
# set SLMKIT_LLAMACPP_DIR to that folder (must contain convert_hf_to_gguf.py)
```

GGUF conversion needs a **full/merged** model folder. LoRA/QLoRA adapters must
be merged first.

### 2.4 LLM-as-judge (optional eval scoring)

```bash
SLMKIT_JUDGE_PROVIDER=anthropic     # or: openai
SLMKIT_JUDGE_API_KEY=sk-...
SLMKIT_JUDGE_MODEL=claude-sonnet-5
```

Without a key, automatic metrics (EM / token-F1 / ROUGE / BLEU) still work.

---

## 3. Without Docker (native install)

Skip this if you are using Compose.

### 3.1 Backend

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
uv pip install -e .                # core API — no CUDA
uv pip install -e ".[gpu]"         # Linux/WSL2 training stack
uv pip install -e ".[eval]"        # ROUGE / BLEU
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Data home defaults to `~/.slmkit/` (or `%USERPROFILE%\.slmkit` on Windows).

**Native Windows training:** PyPI's Windows `torch` wheels are CPU-only.
Install CUDA torch first:

```powershell
pip install "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "xformers==0.0.35" --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[gpu]"
```

Requires an NVIDIA driver with CUDA ≥ 12.8 (driver 570+).

### 3.2 Frontend

Needs Node.js 20+.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 — proxies /api and /ws to :8000
```

Production build: `npm run build` then `npm run preview`.
