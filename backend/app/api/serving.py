"""Provider discovery and local model serving control."""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.db.models import ModelArtifact
from app.db.session import engine
from app.serving.providers import get_provider, ollama, provider_statuses

router = APIRouter(prefix="/api/serving", tags=["serving"])

@router.get("/providers")
async def providers():
    return await provider_statuses()

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
        result = await get_provider(body.provider).generate(body.prompt, body.max_tokens)
        duration = result.get("seconds") or result.get("eval_duration", 0) / 1e9
        tokens = result.get("eval_count")
        return {"output": result.get("response", ""), "tokens": tokens,
                "seconds": duration, "tokens_per_second": tokens / duration if tokens and duration else None,
                "provider": body.provider}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
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
