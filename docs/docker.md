# Docker

Recommended way to run SLM Kit. Two containers:

| Service | Image | Host port | Role |
|---|---|---|---|
| `backend` | `slmkit-backend` | **8000** | FastAPI + training (GPU image) or API-only (CPU image) |
| `frontend` | `slmkit-frontend` | **5173** | nginx (prod) or Vite (dev). Proxies `/api` and `/ws` to the backend |

```
Browser  →  http://localhost:5173
              nginx / Vite
                 ├─ /          static UI
                 ├─ /api/*  →  backend:8000
                 └─ /ws/*   →  backend:8000  (WebSockets)
```

---

## Compose files

| File | When to use |
|---|---|
| `docker-compose.yml` | **Default.** GPU backend + production frontend. Training works. |
| `docker-compose.cpu.yml` | No NVIDIA GPU / no GPU passthrough. UI + datasets + fit estimates. |
| `docker-compose.dev.yml` | Local development. Bind-mounts source; backend `--reload`; Vite HMR. CPU backend. |

All three share the project name `slmkit` except the dev file (`slmkit-dev`),
so don't run prod and CPU compose at the same time — they reuse the same
container names and ports.

---

## GPU stack (training)

```powershell
docker compose up --build -d
```

Requires:

- Docker Desktop (or Engine) with the **NVIDIA** runtime
- Host NVIDIA driver (Windows: enable **Settings → Resources → GPU**)
- `gpus: all` in compose — the backend sees the GPU as it would on Linux

First build pulls a CUDA PyTorch base and the Unsloth stack (several GB).
Later `docker compose up -d` is fast.

Verify inside the container:

```powershell
docker exec slmkit-backend python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect something like: 2.14.0+cu130 True
```

---

## CPU stack (no training)

```powershell
docker compose -f docker-compose.cpu.yml up --build -d
```

Use this to browse the UI, upload datasets, and estimate VRAM **without**
installing torch. A QLoRA launch will fail with `No module named 'torch'` —
that's expected. Switch to the GPU compose file to train.

---

## Dev stack (hot reload)

```powershell
docker compose -f docker-compose.dev.yml up --build
```

- Backend: CPU image, source bind-mounted, `uvicorn --reload`
- Frontend: Vite on port 5173, proxies to `http://backend:8000`
- Data volume: `slmkit-data-dev` (separate from prod)

Open **http://localhost:5173**. Stop with Ctrl+C, then
`docker compose -f docker-compose.dev.yml down`.

---

## Data and config

| Thing | Where |
|---|---|
| App data (DB, datasets, models, runs, HF cache) | Docker volume `slmkit-data` → `/data/slmkit` (`SLMKIT_HOME`) |
| Optional secrets | `backend/.env` (copied from `backend/.env.example`) |

Compose already sets `SLMKIT_HOME=/data/slmkit`. A host `~/.slmkit` is **not**
used while you run Docker.

```powershell
# wipe all runs / models / datasets and start fresh
docker compose down -v
docker compose up -d
```

The backend recreates the volume layout and reinstalls the sample datasets.

---

## Useful commands

```powershell
# status
docker compose ps

# logs
docker compose logs -f
docker compose logs -f backend

# restart one service
docker compose restart backend

# rebuild after Dockerfile or dependency changes
docker compose up --build -d

# stop (keep data)
docker compose down

# stop + delete the data volume
docker compose down -v

# shell into the backend
docker exec -it slmkit-backend bash
```

Health:

```powershell
curl http://localhost:8000/api/health
curl http://localhost:5173/api/health    # same JSON, via nginx proxy
```

---

## Images (what's in the tree)

| File | Purpose |
|---|---|
| `backend/Dockerfile` | GPU backend (PyTorch CUDA image + Unsloth / bitsandbytes / eval extras) |
| `backend/Dockerfile.cpu` | Slim Python image, core API + eval extras, no torch |
| `frontend/Dockerfile` | Multi-stage: `npm run build` → nginx, proxies `/api` and `/ws` |
| `frontend/Dockerfile.dev` | Node 20 + Vite `--host 0.0.0.0` |
| `frontend/nginx.conf` | SPA fallback + API/WebSocket proxy to `backend:8000` |

---

## Optional environment

Copy `backend/.env.example` → `backend/.env`. Compose loads it automatically
(`required: false`). Common keys:

```bash
SLMKIT_HF_TOKEN=hf_xxx
SLMKIT_JUDGE_PROVIDER=anthropic
SLMKIT_JUDGE_API_KEY=sk-ant-xxx
```

See [Configuration](configuration.md). Restart after edits:

```powershell
docker compose up -d --force-recreate backend
```
