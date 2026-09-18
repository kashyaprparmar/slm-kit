"""Managed local model server with a small OpenAI-compatible surface.

The parent API starts this as an isolated process so stopping a deployment
returns all VRAM.  It uses the same runtime loader as Eval Lab, which means
``run:<id>``, local adapters, standard HF checkpoints, and scratch models are
all deployable through one endpoint.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse

_runtime = None
_config: dict[str, Any] = {}
_lock = threading.Lock()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2)
    stream: bool = False


class CompletionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2)
    stream: bool = False


class ScoreBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    input: list[str] = Field(min_length=1, max_length=64)


def _check_model(model: str | None, *, scoring=False):
    if _runtime is None:
        raise HTTPException(503, "Model is still loading.")
    model_id = _config.get("model_id") or _runtime.spec.requested_ref
    if model and model != model_id:
        raise HTTPException(404, "Requested model is not served by this deployment.")
    if hasattr(_runtime, "score") != scoring:
        raise HTTPException(422, "Reward models support scoring; causal models support generation.")


def _prompt(messages: list[ChatMessage]) -> str:
    # The underlying runtime provides the model-specific chat template. Joining
    # history here preserves useful context for a simple multi-turn request.
    users = [message.content for message in messages if message.role == "user"]
    if not users:
        raise HTTPException(400, "messages must contain at least one user message.")
    if len(messages) == 1:
        return users[-1]
    return "\n\n".join(f"{message.role}: {message.content}" for message in messages)


def _params(body: Any) -> dict[str, Any]:
    return {
        "max_new_tokens": body.max_tokens,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "repetition_penalty": body.repetition_penalty,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime, _config
    from app.model_refs import resolve_model_ref
    from app.train_entry.model_runtime import load_reward_runtime, load_runtime

    config_path = Path(sys.argv[1])
    _config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = resolve_model_ref(_config["model_ref"])
    loader = load_reward_runtime if spec.model_category == "reward_model" else load_runtime
    _runtime = loader(_config["model_ref"])
    yield
    if _runtime is not None:
        _runtime.unload()
        _runtime = None


app = FastAPI(title="SLM Kit Model Server", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    if _runtime is None:
        raise HTTPException(503, "Model is still loading.")
    return {"status": "ok", "model": _runtime.spec.requested_ref, "kind": _runtime.spec.kind}


@app.get("/v1/models")
def models():
    if _runtime is None:
        raise HTTPException(503, "Model is still loading.")
    model_id = _config.get("model_id") or _runtime.spec.requested_ref
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "slm-kit"}]}


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionBody):
    _check_model(body.model)
    if _runtime is None:
        raise HTTPException(503, "Model is still loading.")
    prompt = _prompt(body.messages)
    params = {**_params(body), "messages": [m.model_dump() for m in body.messages]}
    if body.stream:
        def events():
            ident = f"chatcmpl-{uuid.uuid4().hex}"
            with _lock:
                for text in _runtime.stream(prompt, params):
                    chunk = {"id": ident, "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": _config.get("model_id"),
                             "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {"id": ident, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": _config.get("model_id") or _runtime.spec.requested_ref,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")
    with _lock:
        output = _runtime.generate(prompt, params)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _config.get("model_id") or _runtime.spec.requested_ref,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
    }


@app.post("/v1/completions")
def completions(body: CompletionBody):
    _check_model(body.model)
    if _runtime is None:
        raise HTTPException(503, "Model is still loading.")
    if body.stream:
        def events():
            ident = f"cmpl-{uuid.uuid4().hex}"
            common = {"id": ident, "object": "text_completion", "created": int(time.time()),
                      "model": _config.get("model_id") or _runtime.spec.requested_ref}
            with _lock:
                for text in _runtime.stream(body.prompt, {**_params(body), "raw_prompt": True}):
                    yield "data: " + json.dumps({**common, "choices": [{"index": 0, "text": text, "finish_reason": None}]}) + "\n\n"
                yield "data: " + json.dumps({**common, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}) + "\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")
    with _lock:
        output = _runtime.generate(body.prompt, {**_params(body), "raw_prompt": True})
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": _config.get("model_id") or _runtime.spec.requested_ref,
        "choices": [{"index": 0, "text": output, "finish_reason": "stop"}],
    }


@app.post("/v1/scores")
def scores(body: ScoreBody):
    import math
    _check_model(body.model, scoring=True)
    if any(not text or len(text) > 128000 for text in body.input):
        raise HTTPException(422, "Each scoring input must contain 1–128000 characters.")
    with _lock:
        values = [_runtime.score(text) for text in body.input]
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(500, "Reward model produced non-finite scores.")
    return {"object": "list", "model": _config.get("model_id") or _runtime.spec.requested_ref,
            "data": [{"index": index, "score": value} for index, value in enumerate(values)]}


if __name__ == "__main__":
    import uvicorn

    # Configuration is read by the lifespan hook; keeping the CLI to one
    # positional config path makes parent-side process management deterministic.
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.train_entry.serve <config.json>")
    from app.config import get_settings

    settings = get_settings()
    uvicorn.run(app, host=settings.deploy_host, port=settings.deploy_port, log_level="info")
