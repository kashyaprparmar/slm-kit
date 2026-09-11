"""Provider discovery and local model serving control."""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.core.deployment import manager
from app.db.models import ModelArtifact
from app.db.session import engine
from app.serving.providers import ollama

router = APIRouter(prefix="/api/serving", tags=["serving"])

@router.get("/providers")
async def providers():
    return {"transformers": {**manager.status(), "provider": "transformers",
                             "capabilities": ["serve", "test", "chat", "scratch", "adapter"]},
            "ollama": await ollama.status()}

class StartBody(BaseModel):
    model: str = Field(min_length=1, max_length=300)

@router.post("/ollama/start")
async def start(body: StartBody):
    try:
        return await ollama.start(body.model)
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(400, str(exc)) from exc

@router.post("/ollama/stop")
async def stop():
    try:
        return {"stopped": await ollama.stop()}
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Ollama did not confirm unloading. Check Ollama and try Stop again.") from exc

class TestBody(BaseModel):
    provider: str
    prompt: str = Field(min_length=1, max_length=32000)
    max_tokens: int = Field(default=128, ge=1, le=2048)

@router.post("/test")
async def test(body: TestBody):
    try:
        if body.provider == "ollama":
            result = await ollama.generate(body.prompt, body.max_tokens)
            duration = result.get("eval_duration", 0) / 1e9
            return {"output": result.get("response", ""), "tokens": result.get("eval_count"),
                    "seconds": duration, "tokens_per_second": result.get("eval_count", 0) / duration if duration else None}
        if body.provider != "transformers":
            raise HTTPException(400, "Choose a supported serving provider.")
        if not manager.active:
            raise HTTPException(409, "Start a model server first.")
        from app.config import get_settings
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(f"http://127.0.0.1:{get_settings().deploy_port}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": body.prompt}], "max_tokens": body.max_tokens})
            response.raise_for_status()
            return {"output": response.json()["choices"][0]["message"]["content"]}
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"The model server could not complete this request: {exc}") from exc


class OllamaImportBody(BaseModel):
    artifact_id: int = Field(ge=1)
    model: str = Field(min_length=1, max_length=120)


@router.post("/ollama/import", status_code=201)
async def import_ollama(body: OllamaImportBody):
    with Session(engine) as db:
        artifact = db.get(ModelArtifact, body.artifact_id)
    if not artifact:
        raise HTTPException(404, "Model artifact not found.")
    if artifact.kind != "gguf" or artifact.status != "ready" or not artifact.local_path:
        raise HTTPException(422, "Only a completed GGUF artifact can be imported into Ollama.")
    try:
        return await ollama.import_gguf(body.model, artifact.local_path)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (TimeoutError, RuntimeError) as exc:
        raise HTTPException(502, str(exc)) from exc
