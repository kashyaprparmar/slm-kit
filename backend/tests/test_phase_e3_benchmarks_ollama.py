import asyncio
from types import SimpleNamespace

from sqlmodel import Session, SQLModel, create_engine

from app.core import serving_benchmarks
from app.db.models import ModelArtifact, ServingBenchmark
from app.train_entry import export


def test_ollama_package_keeps_explicit_template_stops_and_defaults(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"gguf")
    result = export.execute(
        {
            "request": {
                "artifact_id": 1,
                "target": "ollama",
                "ollama": {
                    "chat_template": "{{ .Prompt }}",
                    "stop_tokens": ["<eos>"],
                    "generation_defaults": {"temperature": 0.3, "num_predict": 64},
                },
            },
            "source": str(source),
            "destination": str(tmp_path / "package"),
            "lineage": {},
        }
    )
    modelfile = (tmp_path / "package" / "Modelfile").read_text()
    assert 'FROM "./model.gguf"' in modelfile
    assert "TEMPLATE" in modelfile and 'PARAMETER stop "<eos>"' in modelfile
    assert "PARAMETER temperature 0.3" in modelfile
    assert result["ollama_policy"]["generation_defaults"]["num_predict"] == 64


def test_benchmark_persists_percentiles_and_provider_hardware(tmp_path, monkeypatch):
    db_engine = create_engine(f"sqlite:///{tmp_path / 'benchmarks.db'}")
    SQLModel.metadata.create_all(db_engine)
    monkeypatch.setattr(serving_benchmarks, "engine", db_engine)
    hardware = SimpleNamespace(
        model_dump=lambda mode="json": {
            "vram_total_mb": 100,
            "vram_free_mb": 40,
            "gpu_name": "fixture",
        }
    )
    monkeypatch.setattr(serving_benchmarks.poller, "_latest", hardware)

    class Provider:
        async def status(self):
            return {
                "active": True,
                "state": "ready",
                "model_ref": "/model",
                "endpoint": "http://fixture",
            }

    monkeypatch.setattr(serving_benchmarks, "get_provider", lambda _: Provider())

    async def stream(*_):
        yield {"text": "a"}
        yield {"text": "b", "completion_tokens": 2, "generation_seconds": 0.01}
        yield {"done": True}

    monkeypatch.setattr(serving_benchmarks, "benchmark_stream", stream)
    with Session(db_engine) as db:
        artifact = ModelArtifact(name="model", kind="merged", local_path="/model")
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        artifact_id = artifact.id
    manager = serving_benchmarks.ServingBenchmarkManager()

    async def scenario():
        started = await manager.start(
            serving_benchmarks.BenchmarkRequest(
                provider="transformers",
                prompt="hello",
                requests=3,
                concurrency=2,
                warmup_requests=0,
            )
        )
        await manager.tasks[started["benchmark_id"]]
        return started["benchmark_id"]

    benchmark_id = asyncio.run(scenario())
    with Session(db_engine) as db:
        record = db.get(ServingBenchmark, benchmark_id)
        assert record.status == "completed"
        assert record.artifact_id == artifact_id
        assert record.results["successful_requests"] == 3
        assert record.results["ttft_ms"]["p95"] >= 0
        assert record.results["tokens_per_second"]["count"] == 3
        assert record.results["vram"]["peak_used_mb"] == 60
        assert record.hardware["before"]["gpu_name"] == "fixture"


def test_benchmark_rejects_artifact_not_in_active_transformers_deployment(tmp_path, monkeypatch):
    db_engine = create_engine(f"sqlite:///{tmp_path / 'mismatch.db'}")
    SQLModel.metadata.create_all(db_engine)
    monkeypatch.setattr(serving_benchmarks, "engine", db_engine)

    class Provider:
        async def status(self):
            return {
                "active": True,
                "state": "ready",
                "model_ref": "/other",
                "endpoint": "http://fixture",
            }

    monkeypatch.setattr(serving_benchmarks, "get_provider", lambda _: Provider())
    with Session(db_engine) as db:
        artifact = ModelArtifact(name="model", kind="merged", local_path="/model")
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        artifact_id = artifact.id

    async def scenario():
        try:
            await serving_benchmarks.ServingBenchmarkManager().start(
                serving_benchmarks.BenchmarkRequest(
                    provider="transformers", artifact_id=artifact_id, prompt="hi"
                )
            )
        except ValueError as exc:
            assert "not the active" in str(exc)
        else:
            raise AssertionError("mismatched artifact was accepted")

    asyncio.run(scenario())
