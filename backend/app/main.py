"""SLM Kit FastAPI application."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api import advisor, datasets, registry, runs, serving, system
from app.api import eval as eval_api
from app.config import get_settings
from app.core.errors import install_errors
from app.core.hardware import poller
from app.core.observability import activity, correlation_id
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
    # Stop the queue first so an active trainer cannot claim/reclaim resources
    # while the other managed GPU processes are being torn down.
    await queue.stop()
    await registry.stop_background_tasks()
    # The managed server is a child GPU process; terminate it before the API
    # exits so a restart never leaves VRAM or the deployment port occupied.
    from app.core.deployment import manager as deployment_manager

    await deployment_manager.stop()
    from app.core.eval_manager import manager as eval_manager
    await eval_manager.cancel_current()
    await poller.stop()


app = FastAPI(title="SLM Kit", version="0.1.0", lifespan=lifespan)
install_errors(app)


@app.middleware("http")
async def trace_request(request, call_next):
    request_id = uuid.uuid4().hex
    token = correlation_id.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = f"app;dur={elapsed}"
        if request.url.path not in {"/api/system/activity", "/api/health"}:
            activity.add("API", f"{request.method} {request.url.path} — {response.status_code}",
                         "ERROR" if response.status_code >= 500 else "WARNING" if response.status_code >= 400 else "DEBUG" if request.method == "GET" else "SUCCESS",
                         duration_ms=elapsed, status=response.status_code)
        return response
    finally:
        correlation_id.reset(token)

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
app.include_router(serving.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.websocket("/ws/system")
async def ws_system(ws: WebSocket):
    await hub.connect("system", ws)
    # Push an immediate snapshot so the UI isn't blank until the next poll tick.
    await ws.send_json(poller.latest.model_dump())
    await ws.send_json(queue.snapshot())
    try:
        while True:
            await ws.receive_text()  # keepalive / ignore client messages
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await hub.disconnect("system", ws)


@app.websocket("/ws/activity")
async def ws_activity(ws: WebSocket):
    await ws.accept()
    after = 0
    try:
        while True:
            events = activity.since(after)
            if events:
                after = events[-1]["seq"]
                await asyncio.wait_for(ws.send_json({"type": "activity", "events": events}), 2)
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=1)
            except TimeoutError:
                pass
    except (TimeoutError, WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws/runs/{run_id}")
async def ws_run(ws: WebSocket, run_id: int):
    topic = f"run:{run_id}"
    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await hub.disconnect(topic, ws)


@app.websocket("/ws/eval/{eval_id}")
async def ws_eval(ws: WebSocket, eval_id: int):
    topic = f"eval:{eval_id}"
    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await hub.disconnect(topic, ws)


@app.websocket("/ws/gen/{gen_id}")
async def ws_gen(ws: WebSocket, gen_id: str):
    topic = f"gen:{gen_id}"
    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await hub.disconnect(topic, ws)


@app.websocket("/ws/gguf/{artifact_id}")
async def ws_gguf(ws: WebSocket, artifact_id: int):
    topic = f"gguf:{artifact_id}"
    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await hub.disconnect(topic, ws)
