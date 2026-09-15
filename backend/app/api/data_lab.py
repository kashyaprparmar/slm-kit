"""Version, recipe and profile metadata for Data Lab."""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.datasets import PrepareBody, prepare_dataset
from app.config import get_settings
from app.core.cpu_jobs import cpu_jobs
from app.datasets.lineage import json_fingerprint
from app.db.models import (
    Dataset,
    DatasetProfile,
    DatasetRecipe,
    DatasetSplit,
    DatasetVersion,
    TokenizerArtifact,
)
from app.db.session import engine

router = APIRouter(prefix="/api", tags=["data-lab"])
_settings = get_settings()


class TokenizerProfileBody(BaseModel):
    model_ref: str = Field(min_length=1, max_length=500)
    revision: str | None = Field(default=None, max_length=200)
    max_length: int = Field(default=1024, ge=16, le=1_048_576)
    sample_limit: int = Field(default=250, ge=1, le=5000)
    loss_policy: str = Field(default="full_sequence", pattern="^(full_sequence|completion_only|assistant_only)$")
    chat_template: str | None = Field(default=None, max_length=100_000)


class QualityProfileBody(BaseModel):
    max_rows: int = Field(default=100_000, ge=1, le=1_000_000)
    near_duplicate_threshold: float = Field(default=.9, ge=.7, le=1)
    compare_version_ids: list[int] = Field(default_factory=list, max_length=20)


@router.get("/datasets/{dataset_id}/versions")
def list_versions(dataset_id: int):
    with Session(engine) as db:
        if not db.get(Dataset, dataset_id):
            raise HTTPException(404, "Dataset not found.")
        return db.exec(select(DatasetVersion).where(
            DatasetVersion.dataset_id == dataset_id,
        ).order_by(DatasetVersion.created_at.desc())).all()


@router.get("/dataset-versions/{version_id}")
def get_version(version_id: int):
    with Session(engine) as db:
        version = db.get(DatasetVersion, version_id)
        if not version:
            raise HTTPException(404, "Dataset version not found.")
        profiles = db.exec(select(DatasetProfile).where(DatasetProfile.dataset_version_id == version_id)).all()
        return {"version": version, "profiles": profiles}


@router.get("/dataset-recipes")
def list_recipes(dataset_id: int | None = None):
    with Session(engine) as db:
        query = select(DatasetRecipe).order_by(DatasetRecipe.created_at.desc())
        recipes = db.exec(query).all()
        if dataset_id is not None:
            version_ids = set(db.exec(select(DatasetVersion.id).where(DatasetVersion.dataset_id == dataset_id)).all())
            recipes = [recipe for recipe in recipes if recipe.source_version_id in version_ids]
        return recipes


@router.get("/dataset-recipes/{recipe_id}")
def get_recipe(recipe_id: int):
    with Session(engine) as db:
        recipe = db.get(DatasetRecipe, recipe_id)
        if not recipe:
            raise HTTPException(404, "Dataset recipe not found.")
        split = db.exec(select(DatasetSplit).where(DatasetSplit.recipe_id == recipe_id)).first()
        return {"recipe": recipe, "split": split}


@router.post("/dataset-recipes/{recipe_id}/clone")
def clone_recipe(recipe_id: int):
    with Session(engine) as db:
        recipe = db.get(DatasetRecipe, recipe_id)
        if not recipe:
            raise HTTPException(404, "Dataset recipe not found.")
        version = db.get(DatasetVersion, recipe.source_version_id)
        if not version:
            raise HTTPException(409, "The recipe's source version is unavailable.")
        return {"source_dataset_id": version.dataset_id, "config": recipe.config,
                "source_fingerprint": version.fingerprint}


@router.post("/dataset-recipes/{recipe_id}/rerun", status_code=201)
async def rerun_recipe(recipe_id: int):
    with Session(engine) as db:
        recipe = db.get(DatasetRecipe, recipe_id)
        if not recipe:
            raise HTTPException(404, "Dataset recipe not found.")
        version = db.get(DatasetVersion, recipe.source_version_id)
        if not version:
            raise HTTPException(409, "The recipe's source version is unavailable.")
        dataset_id, config = version.dataset_id, dict(recipe.config)
    return await prepare_dataset(dataset_id, PrepareBody.model_validate(config))


@router.get("/tokenizer-artifacts")
def list_tokenizer_artifacts():
    with Session(engine) as db:
        return db.exec(select(TokenizerArtifact).order_by(TokenizerArtifact.created_at.desc())).all()


@router.post("/dataset-versions/{version_id}/tokenizer-profile")
async def tokenizer_profile(version_id: int, body: TokenizerProfileBody):
    with Session(engine) as db:
        version = db.get(DatasetVersion, version_id)
        if not version:
            raise HTTPException(404, "Dataset version not found.")
        if not Path(version.path).is_file():
            raise HTTPException(409, "The immutable dataset version file is unavailable.")
        profile_config = body.model_dump()
        profile_config_fingerprint = json_fingerprint(profile_config)
        # Pinned revisions are safe to reuse before another Hub lookup. Moving
        # revisions are always resolved again, then deduplicated by fingerprint.
        if body.revision:
            artifacts = db.exec(select(TokenizerArtifact).where(
                TokenizerArtifact.model_ref == body.model_ref,
                TokenizerArtifact.revision == body.revision,
            )).all()
            for artifact in artifacts:
                cache_key = json_fingerprint({"dataset": version.fingerprint,
                                              "tokenizer": artifact.fingerprint,
                                              "config": profile_config_fingerprint})
                cached = db.exec(select(DatasetProfile).where(DatasetProfile.cache_key == cache_key)).first()
                if cached:
                    return {"profile": cached, "tokenizer": artifact,
                            "result": cached.stats, "cached": True}
        worker_config = {
            **profile_config,
            "model_ref": body.model_ref.strip(),
            "dataset_path": version.path,
            "dataset_format": version.fmt,
            "cache_dir": str(_settings.hf_cache_dir),
            "trust_remote_code": _settings.trust_remote_code,
        }
    tmp_dir = _settings.home / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_dir / f"tokenizer-profile-{uuid.uuid4().hex}.json"
    config_path.write_text(json.dumps(worker_config, ensure_ascii=False), encoding="utf-8")
    try:
        async with cpu_jobs.slot():
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "app.train_entry.tokenizer_profile", str(config_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
            except TimeoutError:
                process.kill()
                await process.wait()
                raise HTTPException(504, "Tokenizer profiling exceeded 10 minutes and was stopped.")
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        payload = json.loads(lines[-1]) if lines else {}
        if process.returncode or not payload.get("ok"):
            detail = payload.get("error") or stderr.decode("utf-8", errors="replace")[-2000:]
            raise HTTPException(422, f"Tokenizer profile failed: {detail}")
        result = payload["result"]
    finally:
        config_path.unlink(missing_ok=True)
    tokenizer_data = result["tokenizer"]
    cache_key = json_fingerprint({"dataset": version.fingerprint,
                                  "tokenizer": tokenizer_data["fingerprint"],
                                  "config": profile_config_fingerprint})
    with Session(engine) as db:
        artifact = db.exec(select(TokenizerArtifact).where(
            TokenizerArtifact.fingerprint == tokenizer_data["fingerprint"],
        )).first()
        if not artifact:
            artifact = TokenizerArtifact(
                model_ref=body.model_ref.strip(), revision=body.revision,
                resolved_revision=tokenizer_data.get("resolved_revision"),
                fingerprint=tokenizer_data["fingerprint"], config=tokenizer_data,
            )
            db.add(artifact)
            db.flush()
        profile = db.exec(select(DatasetProfile).where(DatasetProfile.cache_key == cache_key)).first()
        if not profile:
            profile = DatasetProfile(
                dataset_version_id=version_id, tokenizer_artifact_id=artifact.id,
                kind="tokenizer", cache_key=cache_key, config=profile_config,
                stats=result,
            )
            db.add(profile)
        db.commit()
        db.refresh(artifact)
        db.refresh(profile)
        return {"profile": profile, "tokenizer": artifact, "result": result, "cached": False}


@router.post("/dataset-versions/{version_id}/quality-profile")
async def quality_profile(version_id: int, body: QualityProfileBody):
    from app.datasets.quality import profile as build_profile

    with Session(engine) as db:
        version = db.get(DatasetVersion, version_id)
        if not version:
            raise HTTPException(404, "Dataset version not found.")
        comparisons = []
        comparison_fingerprints = []
        for compare_id in dict.fromkeys(body.compare_version_ids):
            if compare_id == version_id:
                continue
            other = db.get(DatasetVersion, compare_id)
            if not other:
                raise HTTPException(422, f"Comparison dataset version {compare_id} was not found.")
            comparisons.append((other.path, other.fmt))
            comparison_fingerprints.append(other.fingerprint)
        config = body.model_dump()
        cache_key = json_fingerprint({"dataset": version.fingerprint, "comparisons": comparison_fingerprints,
                                      "config": config, "quality_schema": 1})
        cached = db.exec(select(DatasetProfile).where(DatasetProfile.cache_key == cache_key)).first()
        if cached:
            return {"profile": cached, "result": cached.stats, "cached": True}
        path, fmt = version.path, version.fmt
    try:
        result = await cpu_jobs.run(build_profile, path, fmt, max_rows=body.max_rows,
                                    near_duplicate_threshold=body.near_duplicate_threshold,
                                    compare_sources=comparisons)
    except (OSError, ValueError) as exc:
        raise HTTPException(422, f"Quality profile failed: {exc}") from exc
    with Session(engine) as db:
        profile = db.exec(select(DatasetProfile).where(DatasetProfile.cache_key == cache_key)).first()
        if not profile:
            profile = DatasetProfile(dataset_version_id=version_id, kind="quality", cache_key=cache_key,
                                     config=config, stats=result)
            db.add(profile)
            db.commit()
            db.refresh(profile)
        return {"profile": profile, "result": result, "cached": False}
