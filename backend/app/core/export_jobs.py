"""Artifact-backed, restart-safe supervision for export workers."""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import psutil
from sqlmodel import Session, select

from app.config import get_settings
from app.core.events import ArtifactEvent, ProfileEvent, ProgressEvent, parse_event
from app.core.log_capture import append_log_line, read_log_tail
from app.core.resources import gpu
from app.core.runner import _redact, _terminate_process_tree
from app.db.models import DatasetVersion, ModelArtifact, Run
from app.db.session import engine
from app.export_contracts import ExportRequest, export_capabilities


class ExportJobs:
    def __init__(self):
        self.tasks: dict[int, asyncio.Task] = {}
        self._submission_lock = asyncio.Lock()

    def workdir(self, job_id: int) -> Path:
        return get_settings().runs_dir / "exports" / str(job_id)

    def update(self, job_id: int, **values):
        with Session(engine) as db:
            artifact = db.get(ModelArtifact, job_id)
            if artifact:
                meta = dict(artifact.meta or {})
                job = dict(meta.get("export_job") or {})
                for key, value in values.items():
                    if key in ("status", "error", "local_path", "hf_repo", "published"):
                        setattr(artifact, key, value)
                    else:
                        job[key] = value
                artifact.meta = {**meta, "export_job": job}
                db.add(artifact)
                db.commit()

    async def start(self, request: ExportRequest) -> dict:
        async with self._submission_lock:
            if len(self.tasks) >= 2:
                raise ValueError("Two export jobs are already active. Wait or cancel a job before starting another.")
            return await self._start(request)

    async def _start(self, request: ExportRequest) -> dict:
        with Session(engine) as db:
            source = db.get(ModelArtifact, request.artifact_id)
            if not source or source.status != "ready" or not source.local_path:
                raise ValueError("Choose a ready local artifact.")
            capability = export_capabilities(source)[request.target]
            if not capability.allowed:
                raise ValueError(capability.reason)
            if request.calibration:
                from app.datasets.lineage import file_fingerprint
                version = db.get(DatasetVersion, request.calibration.dataset_version_id)
                if not version or version.fingerprint != request.calibration.fingerprint:
                    raise ValueError("Calibration metadata does not match the registered dataset version.")
                if not Path(version.path).is_file() or await asyncio.to_thread(file_fingerprint, version.path) != version.fingerprint:
                    raise ValueError("Calibration dataset files changed or are missing.")
                if version.num_rows is not None and request.calibration.samples > version.num_rows:
                    raise ValueError("Calibration sample count exceeds the dataset version's row count.")
            path = Path(source.local_path).resolve()
            if not path.exists():
                raise ValueError("Source artifact files are missing.")
            adapter = path.is_dir() and (path / "adapter_config.json").exists()
            if request.target == "merged" and source.kind in ("reward_model", "reference_model"):
                raise ValueError("This model category requires a specialized merge path.")
            if request.target == "merged":
                import re
                adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
                recorded = adapter_config.get("revision") or (source.meta or {}).get("revision")
                if not request.base_revision and re.fullmatch(r"[0-9a-fA-F]{40}", recorded or ""):
                    request = request.model_copy(update={"base_revision": recorded})
                base = adapter_config.get("base_model_name_or_path")
                if not base:
                    raise ValueError("Adapter metadata does not identify its base model.")
                if not Path(base).exists() and not request.base_revision:
                    raise ValueError("Remote merge requires the immutable base commit used for training. Provide base_revision in the export request.")
            if request.target in ("adapter", "merged") and not adapter:
                raise ValueError("This export requires a PEFT adapter.")
            if request.target == "huggingface" and (adapter or not (path / "config.json").exists()):
                raise ValueError("Hugging Face model export requires a full model; merge adapters separately.")
            if request.target in ("gguf", "quantized"):
                if adapter or not (path / "config.json").exists():
                    raise ValueError("GGUF conversion requires a merged/full Hugging Face model.")
                config = json.loads((path / "config.json").read_text(encoding="utf-8"))
                if config.get("quantization_config"):
                    raise ValueError("Convert from an unquantized model.")
                from app.integrations.gguf import QUANT_TYPES, _convert_script, _quantize_bin
                if request.quant_type not in QUANT_TYPES or not _convert_script() or (request.quant_type != "f16" and not _quantize_bin()):
                    raise ValueError("Install the optional llama.cpp converter and quantizer and set SLMKIT_LLAMACPP_DIR.")
            if request.target == "ollama" and (not path.is_file() or path.suffix.lower() != ".gguf"):
                raise ValueError("Ollama package export requires a GGUF artifact.")
            if request.target == "hub":
                from app.integrations.hf_hub import _token
                if not _token():
                    raise ValueError("Configure a Hugging Face token before publishing.")
            lease = None
            if request.target == "merged":
                from app.serving.providers import external_gpu_owner
                owner = await external_gpu_owner()
                if owner:
                    raise ValueError(f"Stop external {owner} before merging.")
                lease = gpu.acquire("merging", str(source.id))
            try:
                lineage = {"source_artifact_id": source.id, "run_id": source.run_id, "base_model": source.base_model, "source_meta": _redact(source.meta)}
                kind = {"quantized": "gguf", "huggingface": source.kind, "hub": source.kind}.get(request.target, request.target)
                artifact = ModelArtifact(name=f"{source.name} · {request.target}", kind=kind,
                    run_id=source.run_id, base_model=source.base_model, status="exporting",
                    meta={"export_job": {"request": request.model_dump(), "progress": 0}, "lineage": lineage})
                db.add(artifact)
                db.commit()
                db.refresh(artifact)
                job_id = artifact.id
                work = self.workdir(job_id)
                work.mkdir(parents=True, exist_ok=True)
                config = {"request": request.model_dump(), "source": str(path),
                          "destination": str(get_settings().models_dir / f"export-{job_id}"), "lineage": lineage}
                if request.target == "hub" and source.run_id:
                    from app.integrations.hf_hub import generate_model_card
                    run = db.get(Run, source.run_id)
                    if run:
                        config["model_card"] = generate_model_card(run)
                config_path = work / "config.json"
                config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                task = asyncio.create_task(self.run(job_id, config_path, lease))
                self.tasks[job_id] = task
                task.add_done_callback(lambda _: self.tasks.pop(job_id, None))
                return {"artifact_id": job_id, "status": "exporting"}
            except BaseException:
                if lease:
                    gpu.release(lease)
                raise

    async def run(self, job_id: int, config_path: Path, lease):
        proc = None
        result = None
        log_path = self.workdir(job_id) / "output.log"
        try:
            proc = await asyncio.create_subprocess_exec(sys.executable, "-u", "-m", "app.train_entry.export", str(config_path),
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            self.update(job_id, worker_pid=proc.pid, worker_started=psutil.Process(proc.pid).create_time())
            async for line in proc.stdout:
                text = line.decode(errors="replace").rstrip()
                # Never persist configured credentials in worker errors or logs.
                from app.integrations.hf_hub import _token
                token = _token()
                if token:
                    text = text.replace(token, "[REDACTED]")
                append_log_line(log_path, "INFO", text)
                event = parse_event(text)
                if isinstance(event, ProgressEvent):
                    self.update(job_id, progress=event.current, message=event.message)
                elif isinstance(event, ProfileEvent) and event.name == "export_source":
                    self.update(job_id, **event.values)
                elif isinstance(event, ArtifactEvent) and event.kind == "export":
                    result = event.metadata
            code = await proc.wait()
            if code or not result:
                raise RuntimeError(f"Export worker exited with code {code}; see persisted job logs.")
            self.update(job_id, status="ready", local_path=result["local_path"],
                        hf_repo=result.get("hf_repo"), published=bool(result.get("hf_repo")), result=result)
        except asyncio.CancelledError:
            self.update(job_id, status="cancelled", error="Export cancelled.", failure_report={"type": "cancelled"})
            raise
        except Exception as exc:
            self.update(job_id, status="failed", error=str(exc), failure_report={"type": type(exc).__name__, "message": str(exc)})
        finally:
            if proc and proc.returncode is None:
                await asyncio.to_thread(_terminate_process_tree, proc.pid)
                await proc.wait()
            if lease:
                gpu.release(lease)
            self.update(job_id, worker_pid=None)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            staging = Path(config["destination"] + ".staging")
            if staging.parent == get_settings().models_dir:
                shutil.rmtree(staging, ignore_errors=True)
            destination = Path(config["destination"])
            if destination.exists():
                self.update(job_id, local_path=str(destination / result["filename"]) if result and "filename" in result else str(destination))

    async def cancel(self, job_id: int) -> bool:
        task = self.tasks.get(job_id)
        if not task:
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def shutdown(self):
        await asyncio.gather(*(self.cancel(job_id) for job_id in list(self.tasks)))

    def recover(self):
        with Session(engine) as db:
            jobs = list(db.exec(select(ModelArtifact).where(ModelArtifact.status == "exporting")))
        for job in jobs:
            state = (job.meta or {}).get("export_job") or {}
            pid = state.get("worker_pid")
            if pid:
                try:
                    if psutil.Process(pid).create_time() == state.get("worker_started"):
                        _terminate_process_tree(pid)
                except psutil.NoSuchProcess:
                    pass
            self.update(job.id, status="failed", worker_pid=None, error="API restarted during export; inspect logs and retry.", failure_report={"type": "interrupted"})
            destination = get_settings().models_dir / f"export-{job.id}"
            shutil.rmtree(destination.with_name(destination.name + ".staging"), ignore_errors=True)
            if destination.exists():
                self.update(job.id, local_path=str(destination))

    def logs(self, job_id: int):
        return read_log_tail(self.workdir(job_id) / "output.log")


manager = ExportJobs()
