import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlmodel import Session, SQLModel, create_engine

from app.core import export_jobs
from app.db.models import ModelArtifact
from app.export_contracts import ExportRequest
from app.serving.runtime_options import ServingOptions, option_arguments
from app.train_entry import export, serve


def test_export_metadata_redacts_credentials_without_losing_tokenizer_identity():
    from app.core.runner import _redact
    assert _redact({"hf_token": "secret", "tokenizer": {"eos_token": "<eos>", "num_tokens": 12}}) == {
        "hf_token": "***", "tokenizer": {"eos_token": "<eos>", "num_tokens": 12}}


def test_copy_export_preserves_source_and_records_lineage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text('{"base_model_name_or_path":"base"}')
    (source / "adapter.safetensors").write_bytes(b"test weights")
    before = export.fingerprint(source)
    config = {"request": {"artifact_id": 1, "target": "adapter"}, "source": str(source),
              "destination": str(tmp_path / "output"), "lineage": {"run_id": 8}}
    result = export.execute(config)
    assert export.fingerprint(source) == before == result["source_fingerprint"]
    manifest = json.loads((tmp_path / "output" / "export-lineage.json").read_text())
    assert manifest["run_id"] == 8
    assert manifest["source_artifact_id"] == 1
    assert (tmp_path / "output" / "adapter.safetensors").read_bytes() == b"test weights"
    with pytest.raises(ValueError, match="exists"):
        export.execute(config)


def test_export_detects_mutation_and_discards_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_bytes(b"a")
    original = export.shutil.copytree
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (source / "weights").write_bytes(b"changed")
        return result
    monkeypatch.setattr(export.shutil, "copytree", mutate)
    with pytest.raises(ValueError, match="changed"):
        export.execute({"request": {"artifact_id": 1, "target": "adapter"}, "source": str(source),
                        "destination": str(tmp_path / "output"), "lineage": {}})
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "output.staging").exists()


def test_export_contract_rejects_unverified_quantization_and_implicit_publish():
    with pytest.raises(ValidationError):
        ExportRequest(artifact_id=1, target="quantized", quantization="awq")
    with pytest.raises(ValidationError):
        ExportRequest(artifact_id=1, target="hub")
    with pytest.raises(ValidationError):
        ExportRequest(artifact_id=1, target="adapter", arbitrary=True)


def test_cli_controls_require_advertised_flags():
    options = ServingOptions(max_model_len=2048, prefix_caching=True, max_lora_rank=32)
    flags = {"--max-model-len", "--enable-prefix-caching", "--max-lora-rank", "--enable-lora"}
    assert option_arguments("vllm", options, flags) == ["--max-model-len", "2048", "--max-lora-rank", "32", "--enable-lora", "--enable-prefix-caching"]
    with pytest.raises(ValueError, match="advertise"):
        option_arguments("sglang", options, flags)
    with pytest.raises(ValueError, match="disabling"):
        option_arguments("vllm", ServingOptions(enforce_eager=False), {"--enforce-eager"})


def test_native_streaming_and_model_validation(monkeypatch):
    runtime = SimpleNamespace(spec=SimpleNamespace(requested_ref="tiny"),
                              stream=lambda prompt, params: iter(["a", "b"]), generate=lambda *args: "ab")
    monkeypatch.setattr(serve, "_runtime", runtime)
    monkeypatch.setattr(serve, "_config", {"model_id": "tiny"})
    client = TestClient(serve.app)
    response = client.post("/v1/completions", json={"model": "tiny", "prompt": "hello", "stream": True})
    chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and "[DONE]" not in line]
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == "ab"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert client.post("/v1/completions", json={"model": "other", "prompt": "hello"}).status_code == 404
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "tools": []}).status_code == 422


def test_reward_scores_reject_generation_and_nonfinite_values(monkeypatch):
    runtime = SimpleNamespace(spec=SimpleNamespace(requested_ref="reward"), score=lambda text: float(len(text)))
    monkeypatch.setattr(serve, "_runtime", runtime)
    monkeypatch.setattr(serve, "_config", {"model_id": "reward"})
    client = TestClient(serve.app)
    result = client.post("/v1/scores", json={"input": ["a", "abc"]})
    assert [row["score"] for row in result.json()["data"]] == [1, 3]
    assert client.post("/v1/completions", json={"prompt": "hello"}).status_code == 422
    runtime.score = lambda _: float("nan")
    assert client.post("/v1/scores", json={"input": ["a"]}).status_code == 500


def test_export_restart_recovery_preserves_source(tmp_path, monkeypatch):
    db_engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(db_engine)
    monkeypatch.setattr(export_jobs, "engine", db_engine)
    with Session(db_engine) as db:
        source = ModelArtifact(name="source", kind="adapter")
        job = ModelArtifact(name="job", kind="adapter", status="exporting", meta={"export_job": {"progress": 40}})
        db.add(source)
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
        db.refresh(source)
        source_id = source.id
    export_jobs.ExportJobs().recover()
    with Session(db_engine) as db:
        job = db.get(ModelArtifact, job_id)
        assert job.status == "failed"
        assert job.meta["export_job"]["failure_report"]["type"] == "interrupted"
        assert db.get(ModelArtifact, source_id).status == "ready"


def test_cancellation_waits_for_worker_cleanup():
    async def scenario():
        jobs = export_jobs.ExportJobs()
        cleaned = asyncio.Event()
        async def worker():
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
        jobs.tasks[1] = asyncio.create_task(worker())
        await asyncio.sleep(0)
        assert await jobs.cancel(1)
        assert cleaned.is_set()
        assert not await jobs.cancel(2)
    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_supervisor_persists_failure_or_cancellation_and_releases_lease(tmp_path, monkeypatch, cancel):
    from app.core.resources import GPUResources
    from app.integrations import hf_hub
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(export_jobs, "engine", engine)
    settings = SimpleNamespace(runs_dir=tmp_path / "runs", models_dir=tmp_path / "models", hf_token="credential")
    monkeypatch.setattr(export_jobs, "get_settings", lambda: settings)
    monkeypatch.setattr(hf_hub, "_token", lambda: "credential")
    resources = GPUResources()
    monkeypatch.setattr(export_jobs, "gpu", resources)
    monkeypatch.setattr(export_jobs.psutil, "Process", lambda _: SimpleNamespace(create_time=lambda: 123))
    terminated = []
    monkeypatch.setattr(export_jobs, "_terminate_process_tree", lambda pid: terminated.append(pid))
    with Session(engine) as db:
        job = ModelArtifact(name="job", kind="merged", status="exporting", meta={"export_job": {}})
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
    jobs = export_jobs.ExportJobs()
    work = jobs.workdir(job_id)
    work.mkdir(parents=True)
    config = work / "config.json"
    config.write_text(json.dumps({"destination": str(settings.models_dir / "export-1")}))
    async def scenario():
        started = asyncio.Event()
        class Output:
            def __aiter__(self):
                return self.lines()
            async def lines(self):
                started.set()
                if cancel:
                    await asyncio.Event().wait()
                yield b"ERROR credential: converter failed\n"
        class Process:
            pid = 321
            returncode = None if cancel else 2
            stdout = Output()
            async def wait(self):
                self.returncode = 2
                return self.returncode
        async def spawn(*args, **kwargs):
            return Process()
        monkeypatch.setattr(export_jobs.asyncio, "create_subprocess_exec", spawn)
        lease = resources.acquire("merging", 1)
        task = asyncio.create_task(jobs.run(job_id, config, lease))
        jobs.tasks[job_id] = task
        await started.wait()
        if cancel:
            assert await jobs.cancel(job_id)
        else:
            await task
        assert resources.snapshot()["kind"] == "idle"
    asyncio.run(scenario())
    with Session(engine) as db:
        job = db.get(ModelArtifact, job_id)
        assert job.status == ("cancelled" if cancel else "failed")
        assert job.meta["export_job"]["failure_report"]
        assert job.meta["export_job"]["worker_pid"] is None
    if cancel:
        assert terminated == [321]
    else:
        logs = jobs.logs(job_id)
        assert "credential" not in json.dumps(logs)
        assert "REDACTED" in json.dumps(logs)


def test_optional_engine_and_quantization_detection_remain_import_light(monkeypatch):
    from app.models.quantization_registry import quantization_capabilities
    from app.serving import providers
    monkeypatch.setattr(providers, "installation", lambda _: (False, None))
    descriptor = providers.OpenAICompatibleProvider("sglang").capabilities()
    assert descriptor.operations["serve"].state.value == "not_installed"
    assert not descriptor.features["tool_calling"].allowed
    registry = quantization_capabilities()
    assert set(registry) == {"bitsandbytes", "gptq", "awq", "hqq", "eetq", "aqlm"}
    for capability in registry.values():
        assert not capability.operation_capabilities["export"].allowed
        assert not capability.operation_capabilities["merging"].allowed
    assert registry["awq"].calibration_required


def test_real_export_subprocess_registers_output_and_progress(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'real-jobs.db'}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(export_jobs, "engine", engine)
    monkeypatch.setattr(export_jobs, "get_settings", lambda: SimpleNamespace(runs_dir=tmp_path / "runs", models_dir=tmp_path / "models"))
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_config.json").write_text('{"base_model_name_or_path":"test"}')
    (source / "adapter_model.safetensors").write_bytes(b"fixture")
    with Session(engine) as db:
        artifact = ModelArtifact(name="tiny", kind="adapter", local_path=str(source))
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        source_id = artifact.id
    jobs = export_jobs.ExportJobs()
    async def scenario():
        response = await jobs.start(ExportRequest(artifact_id=source_id, target="adapter"))
        await jobs.tasks[response["artifact_id"]]
        return response["artifact_id"]
    job_id = asyncio.run(scenario())
    with Session(engine) as db:
        artifact = db.get(ModelArtifact, job_id)
        assert artifact.status == "ready"
        assert artifact.meta["export_job"]["progress"] == 100
        assert artifact.meta["lineage"]["source_artifact_id"] == source_id
        assert (export.Path(artifact.local_path) / "adapter_model.safetensors").read_bytes() == b"fixture"
    assert jobs.logs(job_id)


@pytest.mark.parametrize("installed", [False, True])
def test_optional_sglang_managed_lifecycle_reserves_and_releases_gpu(tmp_path, monkeypatch, installed):
    from app import model_refs
    from app.core.hardware import poller
    from app.core.resources import GPUResources
    from app.domain import HardwareProfile
    from app.integrations import vllm_serve
    from app.serving import providers
    resources = GPUResources()
    provider = providers.OpenAICompatibleProvider("sglang")
    class Manager:
        model = None
        logs = []
        async def ensure(self, model, **kwargs):
            self.model = model
        async def stop(self):
            self.model = None
        def base_url(self):
            return "http://127.0.0.1:8803"
    manager = Manager()
    monkeypatch.setattr(vllm_serve, "sglang_manager", manager)
    monkeypatch.setattr(providers, "gpu", resources)
    monkeypatch.setattr(providers, "installation", lambda _: (installed, "0.4.1" if installed else None))
    monkeypatch.setattr(poller, "_latest", HardwareProfile(cuda_available=True, gpu_count=1))
    monkeypatch.setattr(model_refs, "resolve_model_ref", lambda _: SimpleNamespace(deployable=True, kind="transformers", local_path=None, load_ref="tiny"))
    async def flags(_):
        return {"--context-length"}
    async def no_owner(**kwargs):
        return None
    async def status():
        return {"active": bool(manager.model), "managed": bool(provider._lease)}
    monkeypatch.setattr(providers, "probe_flags", flags)
    monkeypatch.setattr(providers, "external_gpu_owner", no_owner)
    monkeypatch.setattr(provider, "status", status)
    async def scenario():
        if not installed:
            with pytest.raises(RuntimeError, match="Not installed. Optional"):
                await provider.start("tiny")
            assert resources.snapshot()["kind"] == "idle"
        else:
            state = await provider.start("tiny", ServingOptions(max_model_len=128))
            assert state["managed"]
            assert resources.snapshot()["kind"] == "serving"
            assert provider.capabilities().installed_version == "0.4.1"
            assert await provider.stop()
            assert resources.snapshot()["kind"] == "idle"
            assert manager.model is None
    asyncio.run(scenario())
