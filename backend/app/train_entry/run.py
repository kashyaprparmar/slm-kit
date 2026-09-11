"""Training subprocess entrypoint.

Invoked as ``python -m app.train_entry.run <run_id>``. Loads the run's config,
resolves the backend, and streams every yielded event to stdout as one JSON line
per event. The parent (``core/runner.py``) parses those lines. Any exception is
reported as a final ``status: failed`` event before a non-zero exit.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from app.backends.base import RunContext, get_backend, load_builtin_backends
from app.config import get_settings
from app.core.events import LogEvent, StatusEvent, dump_event
from app.domain import RunConfig

# Force UTF-8 on stdout regardless of platform/console codepage. Training text
# (tokenizer samples, dataset content, model output) can contain arbitrary
# Unicode; Windows' default console encoding (cp1252) can't represent it and
# would crash sys.stdout.write with a UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _emit(event) -> None:
    sys.stdout.write(dump_event(event) + "\n")
    sys.stdout.flush()


def main(run_id: int) -> int:
    settings = get_settings()
    workdir = settings.runs_dir / str(run_id)
    cfg_path = workdir / "config.json"
    if not cfg_path.exists():
        _emit(StatusEvent(status="failed", detail=f"Missing config at {cfg_path}"))
        return 1

    cfg = RunConfig.model_validate_json(cfg_path.read_text())
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from the latest checkpoint if one exists (numeric sort — a plain
    # name sort would rank checkpoint-1000 before checkpoint-200).
    def _step_of(p: Path) -> int:
        try:
            return int(p.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    resume_from = None
    existing = sorted(checkpoint_dir.glob("checkpoint-*"), key=_step_of)
    if existing:
        resume_from = existing[-1]
        _emit(LogEvent(message=f"Resuming from {resume_from.name}"))

    load_builtin_backends()
    backend = get_backend(cfg.backend)

    ctx = RunContext(
        run_id=run_id,
        workdir=workdir,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
    )

    _emit(LogEvent(message=f"Backend '{backend.name}' starting: {cfg.task.value} / {cfg.method.value}"))
    try:
        for event in backend.run(cfg, ctx):
            _emit(event)
    except KeyboardInterrupt:
        _emit(StatusEvent(status="failed", detail="Interrupted"))
        return 130
    except Exception as exc:  # noqa: BLE001
        _emit(LogEvent(level="error", message="".join(traceback.format_exc())))
        _emit(StatusEvent(status="failed", detail=str(exc)))
        return 1

    _emit(StatusEvent(status="done"))
    return 0


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    raise SystemExit(main(rid))
