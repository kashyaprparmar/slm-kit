# SLM Kit — backend

FastAPI API, job queue, SQLite registry, and training entrypoints.

**Recommended run path is Docker from the repo root** — you do not start
`uvicorn` on the host:

```powershell
cd ..    # project root
docker compose up --build -d
```

Then open **http://localhost:5173**. The backend listens on port 8000;
nginx in the frontend container proxies `/api` and `/ws`.

| Stack | Command |
|---|---|
| GPU / training | `docker compose up --build -d` |
| CPU / UI only | `docker compose -f docker-compose.cpu.yml up --build -d` |
| Hot reload | `docker compose -f docker-compose.dev.yml up --build` |

Data: Docker volume `slmkit-data` → `/data/slmkit` (`SLMKIT_HOME`).
Optional secrets: copy `.env.example` → `.env`.

Docs: [Quickstart](../docs/quickstart.md) · [Docker](../docs/docker.md) ·
[Installation](../docs/installation.md) · [API](../docs/api-reference.md).

## Native (optional)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e . && uv pip install -e ".[gpu]" && uv pip install -e ".[eval]"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On native Windows, install CUDA torch from the PyTorch index *before* `[gpu]` —
see [Installation](../docs/installation.md#3-without-docker-native-install).
