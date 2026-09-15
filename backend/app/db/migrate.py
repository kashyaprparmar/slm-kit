"""Non-destructive SQLite migration startup and on-demand integrity diagnostics."""
from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import BaseModel
from sqlalchemy import Engine

log = logging.getLogger(__name__)


def _config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    return config


def _backup(engine: Engine) -> str | None:
    database = engine.url.database
    if not database or database == ":memory:" or not Path(database).is_file():
        return None
    source = Path(database)
    destination = source.parent / "db-backups" / f"{source.name}.{uuid.uuid4().hex}.bak"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # SQLite backup includes committed WAL pages, unlike a raw file copy.
    try:
        with sqlite3.connect(str(source)) as src, sqlite3.connect(str(destination)) as dst:
            src.backup(dst)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return str(destination)


def upgrade_database(engine: Engine) -> dict:
    config = _config()
    head = ScriptDirectory.from_config(config).get_current_head()
    with engine.connect() as connection:
        current = MigrationContext.configure(connection).get_current_revision()
        has_tables = bool(connection.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).first())
    backup = _backup(engine) if has_tables and current != head else None
    with engine.connect() as connection:
        # WAL is persistent for file databases. SQLite uses memory mode for tests.
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        connection.commit()
        # Serialize migration DDL across concurrent startup attempts and make DDL
        # transactional even on Python's legacy sqlite transaction mode.
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
            violations = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchmany(1)
            if violations:
                log.warning("Database contains legacy foreign-key violations. Rows were retained; inspect /api/system/database before repairing lineage.")
            connection.commit()
        except Exception:
            connection.rollback()
            log.exception("Database migration failed; pre-upgrade backup: %s", backup)
            raise
    log.info("Database schema ready: %s (previous=%s, backup=%s)", head, current, backup)
    return {"revision": head, "previous_revision": current, "backup": backup}


class ForeignKeyViolation(BaseModel):
    table: str
    rowid: int | None
    parent: str
    constraint: int


class DatabaseDiagnostics(BaseModel):
    revision: str | None
    expected_revision: str
    foreign_keys: bool
    journal_mode: str
    integrity: list[str]
    foreign_key_violations: list[ForeignKeyViolation]
    violations_truncated: bool


def database_diagnostics(engine: Engine) -> dict:
    """Read-only explicit check; do not scan the DB on every health poll."""
    with engine.connect() as connection:
        revision = MigrationContext.configure(connection).get_current_revision()
        integrity = [row[0] for row in connection.exec_driver_sql("PRAGMA quick_check")]
        violations = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchmany(101)
        return {
            "revision": revision,
            "expected_revision": ScriptDirectory.from_config(_config()).get_current_head(),
            "foreign_keys": bool(connection.exec_driver_sql("PRAGMA foreign_keys").scalar()),
            "journal_mode": connection.exec_driver_sql("PRAGMA journal_mode").scalar(),
            "integrity": integrity,
            "foreign_key_violations": [dict(zip(("table", "rowid", "parent", "constraint"), row, strict=True)) for row in violations[:100]],
            "violations_truncated": len(violations) > 100,
        }
