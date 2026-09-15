from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.backends.base import get_backend
from app.core.preflight import preflight_runner
from app.core.resources import ResourceBusy, gpu
from app.domain import HardwareProfile, Method, RunConfig, TaskType
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_scratch_model_preflight(tmp_path, monkeypatch):
    from app.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "home", tmp_path)

    run_dir = tmp_path / "runs" / "101" / "checkpoints" / "final"
    run_dir.mkdir(parents=True)
    arch = {
        "vocab_size": 1000,
        "n_layers": 2,
        "n_heads": 2,
        "n_embd": 64,
        "block_size": 128,
        "dropout": 0.0,
    }
    (run_dir / "arch.json").write_text(json.dumps(arch), encoding="utf-8")
    (run_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (run_dir / "merges.txt").write_text("", encoding="utf-8")
    (run_dir / "model.pt").touch()

    res = asyncio.run(preflight_runner.execute(
        model_ref=str(run_dir),
        backend="scratch",
    ))
    assert res["ok"] is True
    assert res["backend_verified"] == "scratch"
    assert res["architecture"] == "scratch"
    assert res["context_length"] == 128
    assert res["vocab_size"] == 1000
    assert gpu.snapshot()["kind"] == "idle"


def test_preflight_blocks_when_gpu_busy():
    lease = gpu.acquire("training", 42)
    try:
        with pytest.raises(ResourceBusy):
            asyncio.run(preflight_runner.execute(model_ref="dummy/model"))
    finally:
        gpu.release(lease)
    assert gpu.snapshot()["kind"] == "idle"


def test_preflight_releases_lease_on_error(monkeypatch):
    res = asyncio.run(preflight_runner.execute(model_ref="nonexistent/model-that-does-not-exist-xyz"))
    assert res["ok"] is False
    assert gpu.snapshot()["kind"] == "idle"


def test_preflight_cancellation():
    task_id = "test-task-123"
    # Testing cancel method on non-existent or completed task returns False gracefully
    assert preflight_runner.cancel(task_id) is False


def test_seq2seq_rejected_in_run_validation():
    from app.api.runs import _validate_model_capabilities
    from app.domain import ValidationReport

    cfg = RunConfig(
        backend="transformers",
        task=TaskType.FINETUNE,
        method=Method.LORA,
        base_model="google/flan-t5-base",
        output_name="test-t5",
    )
    report = ValidationReport()
    _validate_model_capabilities(cfg, report)
    assert not report.ok
    assert any("Sequence-to-sequence" in err.message for err in report.issues)


def test_generic_transformers_backend_operates_independently():
    backend = get_backend("transformers")
    assert backend.name == "transformers"
    assert TaskType.FINETUNE in backend.supported_tasks
    assert Method.LORA in backend.supported_methods

    cfg = RunConfig(
        backend="transformers",
        task=TaskType.FINETUNE,
        method=Method.LORA,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        output_name="test-tf",
    )
    hw = HardwareProfile(gpu_count=0, ram_gb=16.0)
    report = backend.validate_config(cfg, hw)
    assert report.ok

    est = backend.estimate_footprint(cfg, hw)
    assert est.total_mb > 0

    exported = backend.export_config(cfg)
    assert exported.content
    assert "transformers" in exported.content or "test-tf" in exported.content


def test_preflight_api_endpoint(client, tmp_path, monkeypatch):
    from app.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "home", tmp_path)

    run_dir = tmp_path / "runs" / "202" / "checkpoints" / "final"
    run_dir.mkdir(parents=True)
    arch = {
        "vocab_size": 500,
        "n_layers": 2,
        "n_heads": 2,
        "n_embd": 32,
        "block_size": 64,
        "dropout": 0.0,
    }
    (run_dir / "arch.json").write_text(json.dumps(arch), encoding="utf-8")
    (run_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (run_dir / "merges.txt").write_text("", encoding="utf-8")
    (run_dir / "model.pt").touch()

    resp = client.post("/api/registry/preflight", json={"model_ref": str(run_dir), "backend": "scratch"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["backend_verified"] == "scratch"
    assert gpu.snapshot()["kind"] == "idle"
