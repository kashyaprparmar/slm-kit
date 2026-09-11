# Quickstart

Get SLM Kit running and finish your first fine-tune.

> **The short version:** start the Docker stack, open
> **http://localhost:5173**, and do everything else in the browser.

Docker is the recommended way to run the app on Windows, Linux, and macOS.
It ships the Python backend (port 8000) and the web UI (port 5173) together —
no local Python, Node, WSL2, or `uv` install required.

Full Docker reference: [Docker](docker.md). Native (no-Docker) install is at
the [bottom](#without-docker-optional).

---

## 1. Prerequisites (once)

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS)
  or Docker Engine + Compose (Linux)
- NVIDIA GPU + recent driver — for training. In Docker Desktop: **Settings →
  Resources → GPU** enabled.
- Confirm the daemon is up: `docker info`

No GPU? Use the CPU stack. The UI, datasets, fit estimates, Registry, and
Run History still work; Unsloth training will not.

---

## 2. Start the app

From the **project root**:

```powershell
# GPU — training (Unsloth / QLoRA). First build downloads several GB.
docker compose up --build -d

# CPU-only — UI and datasets, no training
# docker compose -f docker-compose.cpu.yml up --build -d
```

Wait until both containers are healthy, then open **http://localhost:5173**.

```powershell
docker compose ps
# slmkit-backend    ... (healthy)   0.0.0.0:8000->8000/tcp
# slmkit-frontend   ... (healthy)   0.0.0.0:5173->80/tcp
```

Health check: **http://localhost:8000/api/health** → `{"status":"ok",...}`.

| Command | What it does |
|---|---|
| `docker compose logs -f` | Follow backend + frontend logs |
| `docker compose down` | Stop containers (keeps data) |
| `docker compose down -v` | Stop and **wipe** the data volume |

Persistent data lives in the Docker volume `slmkit-data` (`/data/slmkit` in
the backend). Optional secrets: copy `backend/.env.example` → `backend/.env`.

---

## 3. Your first fine-tune (5 minutes of clicking)

The app ships with sample data, so you can test the full pipeline before
bringing your own files. Use the **GPU** stack for this.

1. Open **http://localhost:5173** → you land on the **Dashboard**.
2. Go to **Dataset Manager** (left sidebar). You should see 10 sample datasets
   already installed. Click **"Sample: Finance QA (instruction)"** — the right
   panel shows a green "Validation passed", a token histogram, and sample rows.
3. Go to **Fine-Tuning Studio**.
   - **Method:** leave **QLoRA** selected (the 8 GB-safe default).
   - **Base model:** leave `Qwen2.5 0.5B` (small = fast first run).
   - **Dataset:** pick *Sample: Finance QA (instruction)*.
   - Look at the **Predicted footprint** panel on the right — it should say
     **"Fits comfortably"** with a breakdown bar.
4. Click **Launch run**. The live monitor appears: status goes
   *Queued → Running*, then a loss curve, tokens/sec, ETA, and streaming logs.
   (First launch downloads the base model — a 0.5B model is ~400 MB.)
5. When it finishes, go to **Run History** — your run is there with its full
   curve and config. Click **Re-run** any time to repeat it exactly.
6. Try your model: **Testing & Eval Lab → Playground**, select the completed
   run (shown as `run:<id>`), type a prompt, and click **Generate**.
7. Optional: **Model Registry → Publish** to push it to your Hugging Face
   account (needs a token — see [Configuration](configuration.md)).
8. Optional: **Model Registry → Deploy** to expose the model locally at
   `http://localhost:8802/v1` with an OpenAI-compatible chat/completions API.

That's the whole loop. Now swap in your own data in the Dataset Manager and
pick a bigger base model when you're ready.

---

## Everyday commands

```powershell
# start (after the first build)
docker compose up -d

# stop
docker compose down

# rebuild after code changes
docker compose up --build -d

# logs
docker compose logs -f backend
docker compose logs -f frontend

# hot-reload dev stack (bind-mounts source; CPU backend)
docker compose -f docker-compose.dev.yml up --build
```

---

## Next steps

- [Docker](docker.md) — GPU vs CPU vs dev compose, volumes, env
- [Requirements](requirements.md) — hardware / software
- [Configuration](configuration.md) — HF token, judge API key, llmfit, llama.cpp
- [Page guides](README.md#page-by-page-guides-the-8-screens-of-the-app)

---

## Without Docker (optional)

Use this only if you cannot run Docker. You then install Python 3.11, Node 20+,
and (for training) a CUDA PyTorch stack yourself.

### Linux / WSL2

```bash
cd backend
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e . && uv pip install -e ".[gpu]" && uv pip install -e ".[eval]"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend && npm install && npm run dev
# http://localhost:5173
```

On Ubuntu 24.04, add the deadsnakes PPA before installing `python3.11`. On
Windows, run the backend **inside WSL2** so Unsloth/bitsandbytes stay on Linux;
the browser stays on Windows.

### Native Windows (no WSL2)

PyPI's Windows `torch` wheels are CPU-only. Install CUDA torch from the
PyTorch index first, then the rest — see [Installation](installation.md#without-docker-native-install).
QLoRA on native Windows has been verified on an RTX 4060 with
`torch==2.11.0+cu128`.

### macOS

The UI and non-CUDA features run. Skip `[gpu]`. Unsloth/QLoRA need NVIDIA.
