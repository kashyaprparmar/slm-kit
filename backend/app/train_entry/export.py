"""One isolated export worker, with atomic local outputs and source verification."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from app.core.events import ArtifactEvent, ProfileEvent, ProgressEvent, dump_event
from app.export_contracts import ExportRequest


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_symlink() or (path.is_dir() and any(p.is_symlink() for p in path.rglob("*"))):
        raise ValueError("Export sources must not contain symbolic links.")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Export source is empty.")
    for file in files:
        if file.is_symlink():
            raise ValueError("Export sources must contain regular files, not symbolic links.")
        digest.update((file.name if path.is_file() else file.relative_to(path).as_posix()).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def event(**data):
    print(dump_event(ProgressEvent(current=data["progress"], total=100, unit="percent", message=data.get("message"))), flush=True)
    if "source_fingerprint" in data:
        print(dump_event(ProfileEvent(name="export_source", values={"source_fingerprint": data["source_fingerprint"]})), flush=True)
    if "result" in data:
        result = data["result"]
        print(dump_event(ArtifactEvent(kind="export", path=result["local_path"], metadata=result)), flush=True)


def execute(config: dict) -> dict:
    request = ExportRequest.model_validate(config["request"])
    source = Path(config["source"])
    destination = Path(config["destination"])
    staging = destination.with_name(destination.name + ".staging")
    if destination.exists() or staging.exists():
        raise ValueError("Export destination already exists.")
    before = fingerprint(source)
    event(progress=5, message="Source verified", source_fingerprint=before)
    staging.mkdir(parents=True)
    result = {"source_fingerprint": before, "source_artifact_id": request.artifact_id}
    try:
        if request.target == "merged":
            from app.train_entry.merge import main
            merge_config = staging / "merge-request.json"
            merge_config.write_text(json.dumps({"model_ref": str(source), "output_dir": str(staging), "base_revision": request.base_revision}), encoding="utf-8")
            main(str(merge_config))
            merge_config.unlink()
        elif request.target in ("gguf", "quantized"):
            from app.integrations.gguf import quantize
            output = quantize(str(source), str(staging), request.quant_type, on_line=lambda line: print(line, flush=True))
            result["filename"] = Path(output).name
        else:
            if source.is_dir():
                shutil.copytree(source, staging, dirs_exist_ok=True)
            else:
                shutil.copy2(source, staging / source.name)
                if request.target != "ollama":
                    result["filename"] = source.name
            if request.target == "ollama":
                from app.ollama_package import render_modelfile, source_policy
                policy = source_policy(source, config["lineage"])
                if request.ollama:
                    overrides = request.ollama.model_dump(exclude_none=True)
                    policy.update(overrides)
                modelfile = render_modelfile(source.name, policy)
                (staging / "Modelfile").write_text(modelfile, encoding="utf-8")
                result["ollama_policy"] = policy
                result["modelfile"] = modelfile
        event(progress=80, message="Output created")
        if fingerprint(source) != before:
            raise ValueError("Source changed during export; discard output and retry with an immutable artifact.")
        if config.get("model_card"):
            (staging / "README.md").write_text(config["model_card"], encoding="utf-8")
        result["payload_fingerprint"] = fingerprint(staging)
        (staging / "export-lineage.json").write_text(json.dumps({**config["lineage"], **result, "request": request.model_dump()}, ensure_ascii=False, indent=2), encoding="utf-8")
        if request.target == "hub":
            from app.integrations.hf_hub import publish_model
            result["hf_repo"] = publish_model(str(staging), request.repo_id, request.private,
                on_commit=lambda commit: result.update(hf_commit=commit))
            (staging / "hub-receipt.json").write_text(json.dumps({"repo": result["hf_repo"], "commit": result.get("hf_commit")}), encoding="utf-8")
        staging.rename(destination)
        result["local_path"] = str(destination / result["filename"]) if "filename" in result else str(destination)
        event(progress=100, message="Export complete", result=result)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    execute(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
