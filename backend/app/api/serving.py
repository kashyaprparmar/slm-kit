"""Provider discovery and local model serving control."""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.core.serving_benchmarks import BenchmarkRequest
from app.core.serving_benchmarks import manager as benchmark_manager
from app.db.models import ModelArtifact, ServingBenchmark
from app.db.session import engine
from app.serving.providers import get_provider, ollama, provider_statuses
from app.serving.runtime_options import ServingOptions

router = APIRouter(prefix="/api/serving", tags=["serving"])


@router.get("/providers")
async def providers():
    return await provider_statuses()


class StartBody(BaseModel):
    model: str = Field(min_length=1, max_length=300)
    options: ServingOptions = Field(default_factory=ServingOptions)


@router.get("/{provider}/options")
async def runtime_options(provider: str):
    instance = get_provider(provider)
    if provider not in ("vllm", "sglang"):
        return {"properties": {}}
    return await instance.options()


@router.post("/{provider}/start")
async def start_provider(provider: str, body: StartBody):
    try:
        instance = get_provider(provider)
        if provider in ("vllm", "sglang"):
            return await instance.start(body.model, body.options)
        if body.options.model_dump(exclude_none=True):
            raise ValueError("This provider does not accept engine controls.")
        return await instance.start(body.model)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (TimeoutError, httpx.HTTPError) as exc:
        raise HTTPException(502, "Provider did not become ready; inspect provider logs.") from exc


@router.post("/{provider}/stop")
async def stop_provider(provider: str):
    try:
        return {"stopped": await get_provider(provider).stop()}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, "Provider did not confirm shutdown; its resource lease is retained."
        ) from exc


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
        raise HTTPException(
            502, "Ollama did not confirm unloading. Check Ollama and try Stop again."
        ) from exc


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
        return {
            "output": result.get("response", ""),
            "tokens": tokens,
            "seconds": duration,
            "tokens_per_second": tokens / duration if tokens and duration else None,
            "provider": body.provider,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, f"The model server could not complete this request: {exc}"
        ) from exc


@router.post("/benchmarks", status_code=202)
async def start_benchmark(body: BenchmarkRequest):
    try:
        return await benchmark_manager.start(body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (RuntimeError, httpx.HTTPError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/benchmarks")
def benchmarks(artifact_id: int | None = None, limit: int = 50):
    with Session(engine) as db:
        query = (
            select(ServingBenchmark)
            .order_by(ServingBenchmark.started_at.desc())
            .limit(min(max(limit, 1), 200))
        )
        if artifact_id is not None:
            query = query.where(ServingBenchmark.artifact_id == artifact_id)
        return list(db.exec(query))


@router.get("/benchmarks/{benchmark_id}")
def benchmark(benchmark_id: int):
    with Session(engine) as db:
        result = db.get(ServingBenchmark, benchmark_id)
    if not result:
        raise HTTPException(404, "Serving benchmark not found.")
    return result


@router.post("/benchmarks/{benchmark_id}/cancel")
async def cancel_benchmark(benchmark_id: int):
    return {"cancelled": await benchmark_manager.cancel(benchmark_id)}


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


@router.post("/ollama/import-package", status_code=201)
async def import_ollama_package(body: OllamaImportBody):
    with Session(engine) as db:
        artifact = db.get(ModelArtifact, body.artifact_id)
    if not artifact:
        raise HTTPException(404, "Model artifact not found.")
    if artifact.kind != "ollama" or artifact.status != "ready" or not artifact.local_path:
        raise HTTPException(
            422, "Only a completed Ollama package can be imported with its Modelfile settings."
        )
    try:
        return await ollama.import_package(body.model, artifact.local_path)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (TimeoutError, RuntimeError) as exc:
        raise HTTPException(502, str(exc)) from exc
