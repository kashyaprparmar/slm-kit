"""Normalize provider discovery and forward OpenAI requests without loading models."""
import httpx
from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from app.config import get_settings
from app.serving.providers import get_provider, provider_statuses

router = APIRouter(prefix="/v1", tags=["openai"])


@router.get("/models")
async def models():
    states = await provider_statuses()
    data = []
    for provider, state in states.items():
        if provider == "transformers":
            if not state.get("active") or state.get("state") != "ready":
                continue
            async with httpx.AsyncClient(timeout=5) as client:
                try:
                    response = await client.get(f"http://127.0.0.1:{get_settings().deploy_port}/v1/models")
                    response.raise_for_status()
                    entries = response.json()["data"]
                except (httpx.HTTPError, ValueError, KeyError):
                    continue
        else:
            entries = state.get("models", []) if state.get("active") or state.get("running") else []
        for entry in entries:
            ident = entry.get("id") or entry.get("name")
            if ident:
                data.append({"id": provider + "::" + ident, "object": "model", "owned_by": provider})
    return {"object": "list", "data": data}


async def forward(path: str, body: dict):
    if "stream" in body and not isinstance(body["stream"], bool):
        raise HTTPException(422, "stream must be a boolean.")
    model = body.get("model")
    if not isinstance(model, str) or "::" not in model:
        raise HTTPException(422, "Select a provider::model ID from GET /v1/models.")
    provider, ident = model.split("::", 1)
    try:
        instance = get_provider(provider)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if path == "scores" and provider != "transformers":
        raise HTTPException(422, "Reward scoring is available through the Transformers provider.")
    # Parsers/templates must be explicitly verified before structured tools can be accepted.
    if any(key in body for key in ("tools", "tool_choice", "functions", "reasoning_format")):
        raise HTTPException(422, "Tool/reasoning parsers are not configured for this gateway deployment.")
    if provider == "transformers":
        url = f"http://127.0.0.1:{get_settings().deploy_port}"
    else:
        url = instance.url
    payload = {**body, "model": ident}
    client = httpx.AsyncClient(timeout=180)
    try:
        request = client.build_request("POST", url + "/v1/" + path, json=payload)
        response = await client.send(request, stream=True)
        if response.is_error:
            await response.aread()
            raise HTTPException(response.status_code, "Provider rejected the request; inspect provider logs.")
        if body.get("stream"):
            async def events():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                finally:
                    await response.aclose()
                    await client.aclose()
            return StreamingResponse(events(), media_type="text/event-stream")
        await response.aread()
        return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Provider endpoint is unavailable.") from exc
    finally:
        if not body.get("stream") or 'response' not in locals() or response.is_error:
            if 'response' in locals():
                await response.aclose()
            await client.aclose()


@router.post("/chat/completions")
async def chat(body: dict):
    return await forward("chat/completions", body)


@router.post("/completions")
async def completions(body: dict):
    return await forward("completions", body)


@router.post("/scores")
async def scores(body: dict):
    return await forward("scores", body)
