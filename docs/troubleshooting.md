# Troubleshooting

Common problems and fixes. Most issues are environment/GPU related, not the app.

---

## Setup & startup

### The resource strip says "Detecting…" forever
The frontend can't reach the backend, or the GPU isn't visible.
- Docker: `docker compose ps` — both `slmkit-backend` and `slmkit-frontend`
  should be **healthy**. Then `curl http://127.0.0.1:8000/api/health`.
- Native: is uvicorn running on port 8000? The Vite proxy targets `127.0.0.1:8000`.
- Check GPU: `http://127.0.0.1:8000/api/system/hardware` should show your GPU
  and `"source":"pynvml"`. If it says `"fallback"`, see the GPU section below.

### `docker compose` fails with "cannot connect to the docker API"
Docker Desktop isn't running. Start it, wait until `docker info` works, retry.

### GPU compose starts but training says `No module named 'torch'`
You are on the **CPU** stack (`docker-compose.cpu.yml`). Stop it and start the
default GPU file:

```powershell
docker compose -f docker-compose.cpu.yml down
docker compose up --build -d
```

### Docker GPU container: `torch.cuda.is_available()` is False
- Docker Desktop → **Settings → Resources → GPU** enabled
- Host `nvidia-smi` shows the card
- Recreate: `docker compose down && docker compose up -d`
- Confirm: `docker exec slmkit-backend python -c "import torch; print(torch.cuda.is_available())"`

### `pip`/`uv install` of `[gpu]` fails or is huge
The GPU stack (torch + CUDA wheels + Unsloth) is several GB.
- On **Linux/WSL2**: `uv pip install -e ".[gpu]"` just works — PyPI's Linux
  torch wheels include CUDA.
- On **native Windows**: PyPI's Windows torch wheels are **CPU-only**. Install
  torch from the PyTorch CUDA index *first*, matching the versions unsloth
  pins (`torch<2.11`), then the rest — see
  [quickstart.md Step 2b](quickstart.md#step-2b-optional--enable-real-gpu-training-natively)
  for the exact commands. This combo is verified working on an RTX 4060.
- Make sure you're on **Python 3.11+**.
- The **core** install (`uv pip install -e .`) works everywhere without this —
  you only need `[gpu]` to actually train.

### Training fails with "Unsloth cannot find any torch accelerator? You need a GPU."
`torch.cuda.is_available()` is `False` even though your GPU works — almost
always because **something silently reinstalled a CPU-only torch**. A later
`pip install <anything depending on torch>` (this includes `pip install unsloth`
itself pulling in `torchvision`/`xformers`) can resolve to a CPU wheel from
plain PyPI, overwriting your CUDA build.
- Check: `python -c "import torch; print(torch.__version__)"` — if it ends in
  `+cpu` instead of `+cu128` (or similar), that's the bug.
- Fix: reinstall the matched CUDA set from the PyTorch index:
  ```powershell
  pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
  ```
- After any `pip install` that touches torch/torchvision/xformers/unsloth on
  native Windows, re-run `python -c "import torch; print(torch.cuda.is_available())"`
  to catch this early.

### `address already in use` on port 8000 or 5173
Another server (or leftover containers / a native uvicorn) is using the port.
- Docker: `docker compose down`, then `docker compose up -d`
- Native: `uvicorn app.main:app --port 8001` (prefer freeing 8000 — the proxy
  expects it). Linux: `lsof -i :8000`. Windows: stop the leftover Python
  process, or `wsl --shutdown` if you were on WSL.

---

## Frontend dev server

### `npm run dev` prints `ws proxy error: write ECONNABORTED` repeatedly, or `jsx`/`rollupOptions` warnings
This means **Vite is running as an incompatible major version** (v8+) instead
of the tested v5. Symptoms:
- Constant `ws proxy error` / `ws proxy socket error` — Vite's dev-server proxy
  can't hold the WebSocket to the backend, so the resource strip and any live
  run monitor stay disconnected.
- `Invalid input options ... "jsx"` and `optimizeDeps.rollupOptions is
  deprecated` warnings — the classic (Babel-based) `@vitejs/plugin-react`
  disagreeing with Vite 8's Rolldown-based bundler internals.

Two known ways this happens even though `package.json` pins
`"vite": "5.4.21"` (no caret) and has `"overrides": { "vite": "5.4.21" }`:

1. **You ran `npm audit fix` (or `--force`).** This command rewrites
   `package.json` itself to whatever version it decides resolves the audit
   findings, *ignoring your pin and the overrides field*. Do not run
   `npm audit fix` on `frontend/` — check `npm audit` output manually instead
   if you're curious, but don't let it auto-upgrade Vite. If you already ran
   it, `package.json` now has `"vite": "^8.1.5"` again — re-pin both the
   `devDependencies` entry and the `overrides` entry to `5.4.21`.
2. **You installed via `npm install` from two different environments on the
   same folder** — e.g. once from native Windows PowerShell/Git-Bash and once
   from WSL2 on the same `/mnt/c/...` path (or vice versa). Each environment
   installs different OS-specific native binaries (`@esbuild/win32-x64` vs
   `@esbuild/linux-x64`, etc.) into the same `node_modules`, and removing/
   replacing them across the Windows/WSL filesystem boundary causes `EPERM:
   operation not permitted, unlink ...` errors during install — a sign the
   install is now corrupted, whatever Vite version it landed on. **Pick one
   environment for the frontend and stick to it** — native Windows is
   recommended, since the browser talks to `localhost` either way and the
   backend is the piece that actually needs WSL2/Linux for training.

**Fix (works for both):**
```bash
cd frontend
# Re-pin if npm audit fix touched it — both the devDependency AND overrides:
#   "overrides": { "vite": "5.4.21" }
#   "devDependencies": { ..., "vite": "5.4.21" }
rm -rf node_modules package-lock.json
npm install          # run from ONE environment only (Windows native recommended)
npm run dev
```
Verify: the startup log should show only the `VITE v5.4.21 ready` banner and
the Local/Network URLs — no `jsx`/`rollupOptions` warnings, no `EPERM` errors
during install, and no `ws proxy error` lines after a few seconds of the
resource strip polling. `./node_modules/.bin/vite --version` should print
`vite/5.4.21`.

### The resource strip is stuck / no live updates, but the page loads fine
Same root cause as above — open the frontend terminal and look for repeating
`ws proxy error` lines. If present, see the fix above.

---

## GPU not detected (`source: fallback`)

The app runs, but with no real VRAM numbers and CPU-only training.
- **Docker:** enable GPU in Docker Desktop, confirm `nvidia-smi` on the host,
  then `docker compose down && docker compose up -d` (the default GPU file,
  not `docker-compose.cpu.yml`).
- **WSL2 native:** `nvidia-smi` inside Ubuntu. If it fails, update the
  **Windows** NVIDIA driver. `wsl --shutdown` then reopen Ubuntu.
- No NVIDIA GPU: `fallback` is expected — Unsloth training won't work.

---

## Training runs

### Run immediately fails with "No module named 'torch'"
The running backend has no training stack.
- **Docker:** you started `docker-compose.cpu.yml`. Switch to
  `docker compose up --build -d`.
- **Native:** install the extra: `uv pip install -e ".[gpu]"` (Linux/WSL2).

### Run fails with a CUDA out-of-memory error
The fit guard blocks obvious OOMs, but real usage can still spike.
- Lower **max sequence length** (try 512), **batch size** (1), raise **gradient
  accumulation** to keep effective batch.
- Switch to **QLoRA** if you weren't using it.
- Pick a **smaller base model**.
- **Close other GPU apps** (browser with many tabs, games) — especially if the
  fit panel said "tight".

### The loss curve is flat / not improving
- Learning rate too low or too high — QLoRA instruction tunes often like
  `1e-4`–`2e-4`.
- Too little data — see the Dataset Manager warnings; instruction tunes want
  ≥ ~50 (ideally 300+) examples.
- Too few steps/epochs.

### A run is stuck on "Queued"
Only one GPU job runs at a time. Another run is ahead of it — check Run History,
or cancel the running one. If nothing is running and it's still stuck, restart
the backend (queued runs are re-enqueued on boot).

### Cancel didn't free VRAM
Cancel first asks the job to stop gracefully, then force-terminates it. Give it
a few seconds. If VRAM still looks held, the process died but the driver may lag
— check `nvidia-smi`; it usually clears within a moment.

---

## Eval Lab

### "GPU is busy" (409) when I try to generate/evaluate
A training or eval job is using the GPU. Wait for it to finish (Run History
shows what's active) — the Eval Lab shares the single GPU on purpose.

### ROUGE/BLEU are missing from my scorecard
Those need the eval extra (already in both Docker images). Native:
`uv pip install -e ".[eval]"`. Exact-match and token-F1 always work.

### The LLM-as-judge checkbox is disabled
No judge key configured. Set `SLMKIT_JUDGE_API_KEY` (and provider/model) — see
[Configuration](configuration.md) — then restart the backend.

---

## Model Registry

### Publish fails / "No Hugging Face token configured"
Set `SLMKIT_HF_TOKEN` (a *write* token) and restart. The capability banner
should then say "HF token configured".

### GGUF export lands as "failed" with an install message
llama.cpp isn't set up. Build it and set `SLMKIT_LLAMACPP_DIR` — see
[Installation §3.3](installation.md). Also note GGUF needs a **merged/full**
model; LoRA adapters must be merged first.

---

## Data

### "Invalid JSON: … (line N)"
Your JSONL has a malformed line N. Each line must be one complete JSON object.
Fix that line (the report points right at it).

### Parquet upload errors
Install `pyarrow` (`uv pip install pyarrow`) or convert the file to JSONL/CSV.

### My dataset shows warnings but no errors
Warnings (thin data, duplicates, short corpus) don't block launching — they're
advice. You *can* proceed; just expect lower quality from very small data.

---

## Full reset

Wipe all local state (runs, models, datasets, DB) and start fresh:

```powershell
docker compose down -v    # Docker: wipe the slmkit-data volume
# native: rm -rf ~/.slmkit   (or the folder set by SLMKIT_HOME)
```

Start again (`docker compose up -d` or restart uvicorn). The backend recreates
the layout and reinstalls the sample datasets.

---

## Still stuck?

- Backend logs: `docker compose logs -f backend` (or the terminal running
  `uvicorn`). The real error is usually there.
- A failed run's **exact error** is shown in Run History's detail card.
- Interactive API docs at `http://127.0.0.1:8000/docs` let you poke endpoints
  directly to isolate frontend vs backend issues.
