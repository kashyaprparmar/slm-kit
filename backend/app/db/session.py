"""Database engine + session helpers."""

from __future__ import annotations

from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

_settings = get_settings()

# check_same_thread=False so the queue worker thread and request handlers can
# share the engine; SQLite writes are serialized by the single-GPU job model.
engine = create_engine(
    _settings.db_url,
    echo=False,
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    # Import models so their tables register on SQLModel.metadata before create.
    from app.db import models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
