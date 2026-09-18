import json
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import openai_gateway


def test_gateway_models_and_streaming_preserve_provider_payload(monkeypatch):
    async def statuses():
        return {"transformers": {"active": False}, "vllm": {"active": True, "models": [{"id": "tiny"}]},
                "sglang": {"active": False}, "ollama": {"running": True, "models": [{"name": "model:tag"}]}}
    monkeypatch.setattr(openai_gateway, "provider_statuses", statuses)
    monkeypatch.setattr(openai_gateway, "get_provider", lambda _: SimpleNamespace(url="http://localhost:9999"))
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        if calls[-1].get("stream"):
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n', headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"object": "text_completion", "choices": [{"text": "hello"}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(openai_gateway.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    app = FastAPI()
    app.include_router(openai_gateway.router)
    client = TestClient(app)
    assert [row["id"] for row in client.get("/v1/models").json()["data"]] == ["vllm::tiny", "ollama::model:tag"]
    result = client.post("/v1/chat/completions", json={"model": "vllm::tiny", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert result.status_code == 200
    assert "[DONE]" in result.text
    assert calls[-1]["model"] == "tiny"
    assert calls[-1]["messages"][0]["content"] == "hi"
    assert client.post("/v1/completions", json={"model": "vllm::tiny", "prompt": "hi"}).json()["choices"][0]["text"] == "hello"
    assert client.post("/v1/chat/completions", json={"model": "vllm::tiny", "tools": []}).status_code == 422
    assert client.post("/v1/chat/completions", json={"model": "vllm::tiny", "stream": "false"}).status_code == 422
    assert client.post("/v1/scores", json={"model": "vllm::tiny", "input": ["a"]}).status_code == 422
