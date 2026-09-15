"""Persistent experiment workspaces and run membership."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, select

from app.db.models import Project, ProjectRun, Run
from app.db.session import engine

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    state: dict = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def non_empty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Project name cannot be blank.")
        return value


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    state: dict | None = None

    @field_validator("name")
    @classmethod
    def non_empty_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Project name cannot be blank.")
        return value


def _detail(db: Session, project: Project) -> dict:
    links = db.exec(select(ProjectRun).where(ProjectRun.project_id == project.id)).all()
    run_ids = [link.run_id for link in links]
    runs = db.exec(select(Run).where(Run.id.in_(run_ids)).order_by(Run.created_at.desc())).all() if run_ids else []
    return {**project.model_dump(), "runs": runs}


@router.get("")
def list_projects():
    with Session(engine) as db:
        projects = db.exec(select(Project).order_by(Project.updated_at.desc())).all()
        return [
            {**project.model_dump(), "run_count": len(db.exec(select(ProjectRun).where(ProjectRun.project_id == project.id)).all())}
            for project in projects
        ]


@router.post("", status_code=201)
def create_project(body: ProjectCreate):
    with Session(engine) as db:
        project = Project(name=body.name.strip(), description=body.description.strip(), state=body.state)
        db.add(project)
        db.commit()
        db.refresh(project)
        return _detail(db, project)


@router.get("/{project_id}")
def get_project(project_id: int):
    with Session(engine) as db:
        project = db.get(Project, project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        return _detail(db, project)


@router.patch("/{project_id}")
def update_project(project_id: int, body: ProjectUpdate):
    with Session(engine) as db:
        project = db.get(Project, project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        for key, value in body.model_dump(exclude_unset=True).items():
            setattr(project, key, value.strip() if isinstance(value, str) else value)
        project.updated_at = datetime.now(UTC)
        db.add(project)
        db.commit()
        db.refresh(project)
        return _detail(db, project)


@router.post("/{project_id}/runs/{run_id}", status_code=201)
def attach_run(project_id: int, run_id: int):
    with Session(engine) as db:
        if not db.get(Project, project_id):
            raise HTTPException(404, "Project not found")
        if not db.get(Run, run_id):
            raise HTTPException(404, "Run not found")
        if not db.get(ProjectRun, (project_id, run_id)):
            db.add(ProjectRun(project_id=project_id, run_id=run_id))
            db.commit()
        return {"project_id": project_id, "run_id": run_id}


@router.delete("/{project_id}")
def delete_project(project_id: int):
    """Delete only workspace metadata; experiments and artifacts are retained."""
    with Session(engine) as db:
        project = db.get(Project, project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        for link in db.exec(select(ProjectRun).where(ProjectRun.project_id == project_id)).all():
            db.delete(link)
        db.flush()
        db.delete(project)
        db.commit()
    return {"deleted": project_id, "runs_retained": True}
