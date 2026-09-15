import asyncio

from app.serving import providers


def test_external_vllm_is_reported_as_gpu_owner(monkeypatch):
    async def vllm_status():
        return {"active": True}

    async def ollama_status():
        return {"loaded": []}

    monkeypatch.setattr(providers.vllm, "status", vllm_status)
    monkeypatch.setattr(providers.ollama, "status", ollama_status)
    assert asyncio.run(providers.external_gpu_owner()) == "vllm"


def test_unmanaged_ollama_model_is_reported_as_gpu_owner(monkeypatch):
    async def vllm_status():
        return {"active": False}

    async def ollama_status():
        return {"loaded": [{"name": "external"}]}

    monkeypatch.setattr(providers.vllm, "status", vllm_status)
    monkeypatch.setattr(providers.ollama, "status", ollama_status)
    monkeypatch.setattr(providers.ollama, "_model", None)
    assert asyncio.run(providers.external_gpu_owner()) == "ollama"
