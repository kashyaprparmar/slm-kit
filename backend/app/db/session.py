"""Database engine + session helpers."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import event
from sqlmodel import Session, create_engine

from app.config import get_settings

_settings = get_settings()

def make_engine(url: str):
    """SQLite connections shared by API threads and isolated worker processes."""
    result = create_engine(url, echo=False,
                           connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(result, "connect")
    def configure_sqlite(connection, _record):
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()

    return result


engine = make_engine(_settings.db_url)


def init_db() -> None:
    from app.datasets.lineage import backfill_legacy_versions
    from app.db.migrate import upgrade_database

    upgrade_database(engine)
    backfill_legacy_versions(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
