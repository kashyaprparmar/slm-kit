"""SLM Kit FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api import advisor, datasets, eval as eval_api, registry, runs, system
from app.config import get_settings
from app.core.hardware import poller
from app.core.queue import queue
from app.core.ws import hub
from app.db.session import init_db

_settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.logging_config import get_logger, setup_logging

    setup_logging()
    log = get_logger("main")
    log.info("SLM Kit backend starting up")
    init_db()
    # Best-effort: make the starter datasets available on first boot.
    try:
        from app.datasets.samples import install_samples

        install_samples()
    except Exception:
        log.exception("sample dataset install failed (non-fatal)")
    poller.start()
    queue.start()
    await queue.recover_orphans()
    log.info("SLM Kit backend ready")
    yield
    log.info("SLM Kit backend shutting down")
    await poller.stop()
    await queue.stop()


app = FastAPI(title="SLM Kit", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(registry.router)
app.include_router(advisor.router)
app.include_router(eval_api.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.websocket("/ws/system")
async def ws_system(ws: WebSocket):
    await hub.connect("system", ws)
    # Push an immediate snapshot so the UI isn't blank until the next poll tick.
    await ws.send_json(poller.latest.model_dump())
    try:
        while True:
            await ws.receive_text()  # keepalive / ignore client messages
    except WebSocketDisconnect:
        await hub.disconnect("system", ws)


@app.websocket("/ws/runs/{run_id}")
async def ws_run(ws: WebSocket, run_id: int):
    topic = f"run:{run_id}"
    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(topic, ws)


@app.websocket("/ws/eval/{eval_id}")
async def ws_eval(ws: WebSocket, eval_id: int):
    topic = f"eval:{eval_id}"
    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(topic, ws)


@app.websocket("/ws/gen/{gen_id}")
async def ws_gen(ws: WebSocket, gen_id: str):
    topic = f"gen:{gen_id}"
    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(topic, ws)


@app.websocket("/ws/gguf/{artifact_id}")
async def ws_gguf(ws: WebSocket, artifact_id: int):
    topic = f"gguf:{artifact_id}"
    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(topic, ws)
