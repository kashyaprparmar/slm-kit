"""Playground generation subprocess for every model format SLM Kit supports."""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_lock = threading.Lock()


def emit(obj: dict) -> None:
    with _lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


class Heartbeat:
    """Keep the UI informed while a large model blocks during load."""

    def __init__(self, phase: str, every: float = 3.0) -> None:
        self.phase = phase
        self.every = every
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            emit({"type": "phase", "phase": self.phase, "elapsed": round(time.time() - self._t0, 1)})

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def main(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    runtime = None
    try:
        from app.train_entry.model_runtime import load_runtime

        model_ref = cfg["model_ref"]
        emit({"type": "phase", "phase": "importing", "elapsed": 0})
        emit({"type": "log", "message": f"Loading model: {model_ref}"})
        with Heartbeat("loading_model"):
            runtime = load_runtime(model_ref)
        emit({
            "type": "log",
            "message": f"Loaded {runtime.spec.kind} model '{runtime.display_name}'. Generating...",
        })
        emit({"type": "phase", "phase": "generating", "elapsed": 0})
        chunks = 0
        output = []
        started = time.perf_counter()
        for text in runtime.stream(cfg["prompt"], cfg):
            if text:
                chunks += 1
                output.append(text)
                emit({"type": "token", "text": text})
        elapsed = time.perf_counter() - started
        full_text = "".join(output)
        tokens = len(runtime.tokenizer.encode(full_text).ids) if runtime.scratch else len(runtime.tokenizer.encode(full_text, add_special_tokens=False))
        import torch
        emit({"type": "metrics", "seconds": round(elapsed, 3), "tokens": tokens,
              "tokens_per_second": round(tokens / max(.001, elapsed), 2),
              "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else None})
        emit({"type": "log", "message": f"Done - generated {chunks} chunks."})
        emit({"type": "done"})
        return 0
    except Exception as exc:  # noqa: BLE001 - error is sent to the browser
        emit({"type": "log", "level": "error", "message": traceback.format_exc()})
        emit({"type": "error", "message": str(exc)})
        emit({"type": "done"})
        return 1
    finally:
        if runtime is not None:
            runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
