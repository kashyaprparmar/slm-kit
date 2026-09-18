import sqlite3
from importlib import import_module

import pytest
from fastapi import HTTPException
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

from app.api import projects, runs
from app.db.migrate import database_diagnostics, upgrade_database
from app.db.models import Dataset, EvalResult, Project, ProjectRun, Run
from app.db.session import make_engine


@pytest.fixture
def database(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    yield engine
    engine.dispose()


def test_fresh_database_migrations_and_foreign_keys(database):
    first = upgrade_database(database)
    assert first["backup"] is None
    report = database_diagnostics(database)
    assert report["revision"] == report["expected_revision"] == "0004_serving_benchmarks"
    assert report["journal_mode"] == "wal"
    assert report["foreign_keys"] is True
    assert report["integrity"] == ["ok"]
    assert report["foreign_key_violations"] == []
    assert upgrade_database(database)["backup"] is None
    with Session(database) as db:
        db.add(Run(name="invalid", task="finetune", method="lora", backend="transformers",
                   base_model="model", dataset_id=999))
        with pytest.raises(IntegrityError):
            db.commit()
    assert any(i["column_names"] == ["status", "created_at"] for i in inspect(database).get_indexes("run"))


def test_frozen_baseline_matches_current_models(database):
    upgrade_database(database)
    inspector = inspect(database)
    for table in SQLModel.metadata.sorted_tables:
        columns = {column["name"]: column for column in inspector.get_columns(table.name)}
        assert set(columns) == set(table.columns.keys())
        for column in table.columns:
            assert columns[column.name]["nullable"] == column.nullable
        actual_fks = {(tuple(fk["constrained_columns"]), fk["referred_table"], tuple(fk["referred_columns"]))
                      for fk in inspector.get_foreign_keys(table.name)}
        expected_fks = {((fk.parent.name,), fk.column.table.name, (fk.column.name,)) for fk in table.foreign_keys}
        assert actual_fks == expected_fks


def test_legacy_data_is_adopted_and_backup_contains_rows(database):
    import_module("app.db.migrations.versions.0001_legacy").baseline_tables().create_all(database)
    with Session(database) as db:
        ds = Dataset(name="Original", kind="instruction", path="user/data.jsonl", fmt="jsonl")
        db.add(ds)
        db.commit()
        dataset_id = ds.id
    result = upgrade_database(database)
    assert result["previous_revision"] is None
    with sqlite3.connect(result["backup"]) as backup:
        assert backup.execute("SELECT name FROM dataset").fetchall() == [("Original",)]
        assert not backup.execute("SELECT name FROM sqlite_master WHERE name='alembic_version'").fetchall()
    with Session(database) as db:
        assert db.get(Dataset, dataset_id).path == "user/data.jsonl"
    # Closing/reopening a database must preserve its schema revision and rows.
    database.dispose()
    assert upgrade_database(database)["previous_revision"] == "0004_serving_benchmarks"


def test_incompatible_legacy_schema_is_not_partially_migrated(database):
    with database.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE dataset (id INTEGER PRIMARY KEY, name TEXT)")
        connection.exec_driver_sql("INSERT INTO dataset VALUES (1, 'keep me')")
    with pytest.raises(RuntimeError, match="missing columns"):
        upgrade_database(database)
    with database.connect() as connection:
        assert connection.exec_driver_sql("SELECT name FROM dataset").scalar() == "keep me"
    assert set(inspect(database).get_table_names()) == {"dataset"}


def test_existing_orphans_are_reported_without_deleting_history(database):
    import_module("app.db.migrations.versions.0001_legacy").baseline_tables().create_all(database)
    with database.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.exec_driver_sql("INSERT INTO projectrun VALUES (77, 88, '2026-01-01')")
        connection.commit()
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    upgrade_database(database)
    assert len(database_diagnostics(database)["foreign_key_violations"]) == 2
    with Session(database) as db:
        assert db.get(ProjectRun, (77, 88)) is not None


def test_unknown_future_revision_refuses_startup(database):
    upgrade_database(database)
    with database.begin() as connection:
        connection.exec_driver_sql("UPDATE alembic_version SET version_num='future_schema'")
    with pytest.raises(Exception, match="future_schema"):
        upgrade_database(database)
    with database.connect() as connection:
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar() == "future_schema"


def test_delete_run_removes_membership_and_preserves_evaluated_run(database, monkeypatch):
    upgrade_database(database)
    monkeypatch.setattr(runs, "engine", database)
    monkeypatch.setattr(projects, "engine", database)
    with Session(database) as db:
        run = Run(name="run", task="finetune", method="lora", backend="transformers", base_model="model", status="done")
        project = Project(name="project")
        db.add(run)
        db.add(project)
        db.commit()
        run_id, project_id = run.id, project.id
        db.add(ProjectRun(project_id=project_id, run_id=run_id))
        db.add(EvalResult(run_id=run_id, model_ref=f"run:{run_id}"))
        db.commit()
    with pytest.raises(HTTPException) as error:
        runs.delete_run(run_id)
    assert error.value.status_code == 409
    # Deleting a project must retain the linked run and its evaluation.
    projects.delete_project(project_id)
    with Session(database) as db:
        assert db.get(Run, run_id)
        evaluation = db.exec(select(EvalResult)).one()
        db.delete(evaluation)
        db.add(Project(id=project_id, name="new project"))
        db.commit()
        db.add(ProjectRun(project_id=project_id, run_id=run_id))
        db.commit()
    runs.delete_run(run_id)
    with Session(database) as db:
        assert db.get(Run, run_id) is None
        assert db.get(ProjectRun, (project_id, run_id)) is None
        assert db.get(Project, project_id)
