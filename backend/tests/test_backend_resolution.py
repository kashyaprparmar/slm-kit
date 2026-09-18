from __future__ import annotations

import asyncio
import json
import sys

from sqlmodel import Session, SQLModel, create_engine, select

from app.backends import base, llamafactory_backend
from app.backends.llamafactory_backend import LlamaFactoryBackend
from app.backends.unsloth_backend import UnslothBackend
from app.capabilities import Capability, SupportState
from app.core import runner
from app.db.models import Dataset, ModelArtifact, Run
from app.domain import HardwareProfile, Method, RunConfig, TaskType


class FakeBackend:
    def __init__(self, name: str, *, available: bool = True):
        self.name = name
        descriptor = UnslothBackend(name="transformers").capabilities()
        supported = Capability(state=SupportState.SUPPORTED, reason="supported by test backend")
        self._descriptor = descriptor.model_copy(
            update={
                "name": name,
                "availability": Capability(
                    state=SupportState.SUPPORTED if available else SupportState.NOT_INSTALLED,
                    reason="available" if available else "optional runtime missing",
                ),
                "tasks": {**descriptor.tasks, "finetune": supported},
                "methods": {**descriptor.methods, "lora": supported},
            }
        )

    def capabilities(self):
        return self._descriptor


def _config(**updates) -> RunConfig:
    values = {
        "backend": "auto",
        "task": TaskType.FINETUNE,
        "method": Method.LORA,
        "base_model": "org/model",
        "dataset_id": 1,
        "output_name": "test",
    }
    values.update(updates)
    return RunConfig(**values)


def test_auto_backend_resolution_uses_hardware_and_reports_every_alternative(monkeypatch):
    registered = {
        "unsloth": FakeBackend("unsloth"),
        "transformers": FakeBackend("transformers"),
        "llamafactory": FakeBackend("llamafactory", available=False),
    }
    monkeypatch.setattr(base, "_REGISTRY", registered)

    selection = base.AutoBackendSelector().select(_config(), HardwareProfile(cuda_available=False))

    assert selection.backend.name == "transformers"
    assert "exact finetune/lora semantics" in selection.reason
    reasons = {item["backend"]: item["reason"] for item in selection.alternatives}
    assert "CUDA" in reasons["unsloth"]
    assert reasons["llamafactory"] == "optional runtime missing"
    assert selection.as_dict()["requested_backend"] == "auto"


def test_llamafactory_unavailable_and_installed_capabilities_are_dynamic(monkeypatch):
    monkeypatch.setattr(
        "app.backends.llamafactory_backend._installation",
        lambda: (False, None, None),
    )
    missing = LlamaFactoryBackend().capabilities()
    assert missing.availability.state == SupportState.NOT_INSTALLED
    assert "Optional" in missing.availability.reason

    monkeypatch.setattr(
        "app.backends.llamafactory_backend._installation",
        lambda: (True, "0.9.4", ["llamafactory-cli"]),
    )
    installed = LlamaFactoryBackend().capabilities()
    assert installed.availability.allowed
    assert "0.9.4" in installed.availability.reason


def test_llamafactory_config_preserves_alignment_and_quantization_semantics():
    cfg = _config(
        backend="llamafactory",
        task=TaskType.ALIGNMENT,
        method=Method.QLORA,
        alignment={"objective": "simpo", "reference": {"strategy": "none"}},
        quantization={"mode": "fp4", "compute_dtype": "bf16", "double_quant": False},
    )
    native = LlamaFactoryBackend()._config(cfg, "data", "output", "dataset")
    assert native["stage"] == "dpo"
    assert native["pref_loss"] == "simpo"
    assert native["simpo_gamma"] == cfg.alignment.simpo_gamma
    assert native["quantization_bit"] == 4
    assert native["quantization_type"] == "fp4"
    assert native["double_quantization"] is False
    exported = json.loads(LlamaFactoryBackend().export_config(cfg).content)
    assert exported["stage"] == "dpo"


def test_llamafactory_alignment_does_not_drop_options(monkeypatch):
    monkeypatch.setattr(llamafactory_backend, "_installation", lambda: (True, "0.9.4", ["llamafactory-cli"]))
    backend = LlamaFactoryBackend()
    cfg = _config(backend="llamafactory", task=TaskType.ALIGNMENT,
                  alignment={"label_smoothing": 0.1})
    assert backend._config(cfg, "data", "output", "dataset")["dpo_label_smoothing"] == 0.1
    cfg.alignment.reference.strategy = "separate_model"
    cfg.alignment.reference.model = "org/reference"
    report = backend.validate_config(cfg, HardwareProfile())
    assert not report.ok and any("Separate reference" in issue.message for issue in report.issues)
    cfg = _config(backend="llamafactory", task=TaskType.ALIGNMENT,
                  alignment={"objective": "kto", "desirable_weight": 2, "undesirable_weight": 0.5})
    native = backend._config(cfg, "data", "output", "dataset")
    assert native["stage"] == "kto" and "pref_loss" not in native
    assert native["kto_chosen_weight"] == 2 and native["kto_rejected_weight"] == 0.5


def test_auto_resolution_rejects_unmapped_dpo_loss_and_reference():
    cfg = _config(task=TaskType.ALIGNMENT, alignment={"dpo_loss_variant": "robust"})
    descriptor = LlamaFactoryBackend().capabilities().model_copy(update={
        "availability": Capability(state=SupportState.SUPPORTED, reason="installed"),
    })
    assert "DPO loss" in base.AutoBackendSelector._incompatibility(cfg, HardwareProfile(), None, descriptor)
    cfg.alignment.dpo_loss_variant = "sigmoid"
    cfg.alignment.reference.strategy = "separate_model"
    cfg.alignment.reference.model = "org/reference"
    assert "Separate reference" in base.AutoBackendSelector._incompatibility(cfg, HardwareProfile(), None, descriptor)


def test_llamafactory_dataset_translation_uses_declared_openai_sharegpt_tags(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "preference.jsonl"
    source.write_text(
        json.dumps({"prompt": "Question", "chosen": "Good", "rejected": "Bad"}) + "\n",
        encoding="utf-8",
    )
    database = create_engine(f"sqlite:///{(tmp_path / 'dataset.db').as_posix()}")
    SQLModel.metadata.create_all(database)
    monkeypatch.setattr(llamafactory_backend, "engine", database)
    with Session(database) as db:
        dataset = Dataset(name="preference", kind="preference", path=str(source), fmt="jsonl")
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        dataset_id = dataset.id
    assert dataset_id is not None
    cfg = _config(
        backend="llamafactory",
        task=TaskType.ALIGNMENT,
        dataset_id=dataset_id,
    )

    data_dir = tmp_path / "translated"
    LlamaFactoryBackend._prepare_dataset(cfg, data_dir, "slmkit_dataset")

    row = json.loads((data_dir / "dataset.jsonl").read_text(encoding="utf-8"))
    info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))[
        "slmkit_dataset"
    ]
    assert row["messages"] == [{"role": "user", "content": "Question"}]
    assert row["chosen"] == {"role": "assistant", "content": "Good"}
    assert info["ranking"] is True
    assert info["tags"]["role_tag"] == "role"


def test_final_artifact_registration_is_idempotent_and_keeps_lineage(tmp_path, monkeypatch):
    database = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    SQLModel.metadata.create_all(database)
    monkeypatch.setattr(runner, "engine", database)
    with Session(database) as db:
        run = Run(
            name="alignment",
            task="alignment",
            method="lora",
            backend="transformers",
            base_model="org/base",
            dataset_id=None,
            config={
                "revision": "abc123",
                "alignment": {"reference": {"strategy": "base_model"}},
            },
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id
    assert run_id is not None

    runner._record_artifact(run_id, "model", str(tmp_path / "output"), {"objective": "dpo"})
    reference = {"strategy": "base_model", "requested_model": "run:7", "model": "/resolved/base", "resolved_revision": "pinned-commit"}
    runner._record_artifact(run_id, "adapter", str(tmp_path / "output"), {"objective": "dpo", "reference": reference})

    with Session(database) as db:
        artifacts = db.exec(select(ModelArtifact).where(ModelArtifact.run_id == run_id)).all()
    assert len(artifacts) == 1
    assert artifacts[0].kind == "adapter"
    assert artifacts[0].meta["reference"]["strategy"] == "base_model"
    assert artifacts[0].meta["objective"] == "dpo"
    assert artifacts[0].meta["reference"] == reference
    assert artifacts[0].base_model == "org/base" and artifacts[0].run_id == run_id

    # A final CheckpointEvent may register the same path as a generic model
    # before the objective-specific ArtifactEvent arrives. The latter must
    # promote the row rather than duplicate it.
    runner._record_artifact(
        run_id,
        "reward_model",
        str(tmp_path / "output"),
        {"objective": "reward_model", "evaluation_capabilities": ["pairwise_accuracy"]},
    )
    with Session(database) as db:
        reward_artifacts = db.exec(
            select(ModelArtifact).where(ModelArtifact.run_id == run_id)
        ).all()
    assert len(reward_artifacts) == 1
    assert reward_artifacts[0].kind == "reward_model"
    assert reward_artifacts[0].meta["evaluation_capabilities"] == ["pairwise_accuracy"]


def test_cancellation_terminates_worker_descendants(tmp_path):
    import psutil

    async def scenario():
        script = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "print(child.pid,flush=True); time.sleep(30)"
        )
        proc = await asyncio.create_subprocess_exec(sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE)
        child = None
        try:
            child_pid = int(await asyncio.wait_for(proc.stdout.readline(), timeout=10))
            child = psutil.Process(child_pid)
            await runner.RunHandle(1, proc, tmp_path).stop(graceful=False)
            await asyncio.wait_for(proc.wait(), timeout=5)
            assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        finally:
            if proc.returncode is None:
                runner._terminate_process_tree(proc.pid)
                await proc.wait()
            if child is not None and child.is_running():
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

    asyncio.run(scenario())
