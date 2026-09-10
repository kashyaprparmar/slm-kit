# Commands cheat-sheet

Copy-paste reference. Assumes you're in the **project root** unless noted.

## Docker (recommended)

```powershell
# --- start ---
docker compose up --build -d                              # GPU / training
docker compose -f docker-compose.cpu.yml up --build -d    # UI only
docker compose -f docker-compose.dev.yml up --build       # hot reload (foreground)

# --- everyday ---
docker compose up -d
docker compose down
docker compose down -v                 # wipe data volume
docker compose ps
docker compose logs -f
docker compose logs -f backend
docker compose restart backend
docker compose up --build -d           # rebuild after Dockerfile / dep changes
docker exec -it slmkit-backend bash

# --- health ---
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/system/hardware
curl http://localhost:5173/api/health  # via nginx proxy

# --- GPU check (GPU stack only) ---
docker exec slmkit-backend python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Full notes: [Docker](docker.md).

## Native backend (no Docker)

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -e .                  # core API
uv pip install -e ".[gpu]"           # training stack (Linux/WSL2)
uv pip install -e ".[eval]"          # ROUGE/BLEU metrics

uvicorn app.main:app --host 127.0.0.1 --port 8000
uvicorn app.main:app --reload
SLMKIT_HOME=/tmp/slmtest uvicorn app.main:app

curl http://127.0.0.1:8000/api/health
python -m compileall -q app
```

## Native frontend (no Docker)

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
npm run build
npm run preview
npm run lint
```

## Data home

**Docker:** volume `slmkit-data` → `/data/slmkit` inside the backend.

```powershell
docker compose down -v             # full reset
```

**Native:**

```bash
ls ~/.slmkit
rm -rf ~/.slmkit                   # full reset
du -sh ~/.slmkit/*
```

## Optional tools

```bash
# Hugging Face token (backend/.env or export)
export SLMKIT_HF_TOKEN=hf_xxx

# llmfit
llmfit fit --json --model unsloth/Qwen2.5-0.5B-Instruct
export SLMKIT_LLMFIT_BIN=/path/to/llmfit

# llama.cpp (GGUF export)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release
export SLMKIT_LLAMACPP_DIR=$PWD
```

## Common curl calls

```bash
curl http://127.0.0.1:8000/api/datasets

curl -X POST http://127.0.0.1:8000/api/runs/estimate \
  -H 'Content-Type: application/json' \
  -d '{"backend":"unsloth","task":"finetune","method":"qlora",
       "base_model":"unsloth/Qwen2.5-0.5B-Instruct","dataset_id":3,"output_name":"demo"}'

curl -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"backend":"unsloth","task":"finetune","method":"qlora",
       "base_model":"unsloth/Qwen2.5-0.5B-Instruct","dataset_id":3,"output_name":"demo"}'

curl -X POST http://127.0.0.1:8000/api/runs/1/cancel
curl -X POST http://127.0.0.1:8000/api/runs/1/rerun
```

See the full [API reference](api-reference.md) for every endpoint.
