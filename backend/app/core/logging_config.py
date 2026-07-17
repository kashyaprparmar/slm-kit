"""Centralized logging setup for the backend.

Configures a readable console format plus a rotating file at
``~/.slmkit/logs/slmkit.log`` so there's always a durable record of what the
server and every job did. Call ``setup_logging()`` once at startup.

Usage in modules:
    from app.core.logging_config import get_logger
    log = get_logger(__name__)
    log.info("something happened: %s", detail)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from app.config import get_settings

_CONFIGURED = False

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_dir = settings.home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "slmkit.log"

    root = logging.getLogger("slmkit")
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(console)

    # Rotating file: 5 files x 2 MB so logs never grow unbounded.
    fileh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    fileh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
    ))
    root.addHandler(fileh)

    root.info("Logging initialized → %s", log_file)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child of the slmkit logger (so it inherits our handlers)."""
    short = name.replace("app.", "")
    return logging.getLogger(f"slmkit.{short}")


def log_file_path() -> Path:
    return get_settings().home / "logs" / "slmkit.log"
