# Commands cheat-sheet

Copy-paste reference. Assumes you're in the project root unless noted.

## Backend

```bash
# --- setup (once) ---
cd backend
uv venv --python 3.11
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -e .                  # core API
uv pip install -e ".[gpu]"           # training stack (Linux/WSL2)
uv pip install -e ".[eval]"          # ROUGE/BLEU metrics

# --- run ---
uvicorn app.main:app --host 127.0.0.1 --port 8000          # normal
uvicorn app.main:app --reload                              # auto-reload (dev)
SLMKIT_HOME=/tmp/slmtest uvicorn app.main:app            # throwaway data dir

# --- health checks ---
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/system/hardware
curl http://127.0.0.1:8000/api/system/status

# --- sanity: does everything compile? ---
python -m compileall -q app
```

## Frontend

```bash
cd frontend
npm install            # once
npm run dev            # dev server → http://localhost:5173
npm run build          # production build → dist/
npm run preview        # serve the production build
npm run lint           # eslint
```

## WSL2 (Windows)

```powershell
wsl --install -d Ubuntu-22.04     # install (PowerShell as admin)
wsl                                # enter Ubuntu
wsl --shutdown                     # fully restart WSL (fixes odd GPU/driver states)
```

```bash
nvidia-smi                         # confirm GPU passthrough inside WSL2
```

## Data home

```bash
ls ~/.slmkit                     # DB, datasets, models, runs
rm -rf ~/.slmkit                 # full reset (deletes ALL runs/models/data)
du -sh ~/.slmkit/*               # what's using disk
```

## Optional tools

```bash
# llmfit
llmfit fit --json --model unsloth/Qwen2.5-0.5B-Instruct
export SLMKIT_LLMFIT_BIN=/path/to/llmfit

# llama.cpp (GGUF export)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release
export SLMKIT_LLAMACPP_DIR=$PWD

# Hugging Face token
export SLMKIT_HF_TOKEN=hf_xxx
```

## Common curl calls

```bash
# list datasets
curl http://127.0.0.1:8000/api/datasets

# estimate a run's VRAM before launching
curl -X POST http://127.0.0.1:8000/api/runs/estimate \
  -H 'Content-Type: application/json' \
  -d '{"backend":"unsloth","task":"finetune","method":"qlora",
       "base_model":"unsloth/Qwen2.5-0.5B-Instruct","dataset_id":3,"output_name":"demo"}'

# launch that run
curl -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"backend":"unsloth","task":"finetune","method":"qlora",
       "base_model":"unsloth/Qwen2.5-0.5B-Instruct","dataset_id":3,"output_name":"demo"}'

# watch a run's live events
#   (browser/websocat) ws://127.0.0.1:8000/ws/runs/<run_id>

# cancel / re-run
curl -X POST http://127.0.0.1:8000/api/runs/1/cancel
curl -X POST http://127.0.0.1:8000/api/runs/1/rerun
```

See the full [API reference](api-reference.md) for every endpoint.
