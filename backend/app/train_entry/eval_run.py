"""Eval-harness subprocess for Hugging Face, adapter, and scratch models."""

from __future__ import annotations

import json
import math
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
    def __init__(self, phase: str, model: str, every: float = 3.0) -> None:
        self.phase, self.model, self.every = phase, model, every
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            emit({"type": "phase", "phase": self.phase, "model": self.model,
                  "elapsed": round(time.time() - self._t0, 1)})

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def _extract(row: dict) -> tuple[str, str]:
    """Return question/reference from common instruction and chat schemas."""
    from app.datasets.validate import normalize_row
    row = normalize_row(row)
    if "messages" in row and isinstance(row["messages"], list):
        user = next((m.get("content", "") for m in row["messages"] if m.get("role") == "user"), "")
        ref = next((m.get("content", "") for m in row["messages"] if m.get("role") == "assistant"), "")
        return str(user), str(ref)
    question = row.get("instruction") or row.get("prompt") or row.get("question") or ""
    context = row.get("input") or row.get("context") or ""
    reference = row.get("output") or row.get("response") or row.get("answer") or ""
    return (str(question) if not context else f"{question}\n\n{context}"), str(reference)


def load_rows(dataset_id: int, limit: int) -> list[dict]:
    from sqlmodel import Session

    from app.datasets.validate import _iter_records, detect_format
    from app.db.models import Dataset
    from app.db.session import engine

    with Session(engine) as db:
        dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found.")
    rows: list[dict] = []
    for _line, row in _iter_records(Path(dataset.path), detect_format(dataset.path)):
        if isinstance(row, dict):
            rows.append(row)
        if len(rows) >= limit:
            break
    if not rows:
        raise ValueError("The selected evaluation dataset has no readable rows.")
    return rows


def eval_model(model_ref: str, prompts: list[str], refs: list[str], cfg: dict) -> dict:
    from app.eval import metrics as metrics_module
    from app.integrations import judge
    from app.train_entry.model_runtime import load_runtime

    emit({"type": "log", "message": f"Loading {model_ref}..."})
    runtime = None
    try:
        with Heartbeat("loading_model", model_ref):
            runtime = load_runtime(model_ref)
        emit({"type": "log", "message": f"Loaded {runtime.spec.kind} model; generating predictions..."})
        predictions: list[str] = []
        nlls: list[float] = []
        wants_perplexity = "perplexity" in cfg.get("metrics", [])
        generation_seconds = 0.0
        generated_tokens = 0
        for index, (prompt, reference) in enumerate(zip(prompts, refs, strict=True), start=1):
            started = time.perf_counter()
            predictions.append(runtime.generate(prompt, cfg))
            generation_seconds += time.perf_counter() - started
            generated_tokens += len(runtime.tokenizer.encode(predictions[-1]).ids) if runtime.scratch else len(runtime.tokenizer.encode(predictions[-1], add_special_tokens=False))
            if wants_perplexity:
                value = runtime.reference_perplexity(prompt, reference)
                if value is not None and math.isfinite(value):
                    nlls.append(value)
            emit({"type": "progress", "model": model_ref, "done": index, "total": len(prompts)})

        scores = metrics_module.compute(predictions, refs, cfg.get("metrics", []))
        scores["latency_seconds"] = round(generation_seconds / max(1, len(prompts)), 3)
        scores["tokens_per_second"] = round(generated_tokens / max(.001, generation_seconds), 2)
        if nlls:
            # Values are already per-example perplexities. Geometric mean avoids
            # one long answer overwhelming the aggregate.
            scores["perplexity"] = round(math.exp(sum(math.log(v) for v in nlls) / len(nlls)), 3)
        if cfg.get("judge") and judge.judge_available():
            judged = [
                result["score"]
                for prompt, reference, prediction in list(zip(prompts, refs, predictions, strict=True))[:20]
                if (result := judge.judge_answer(prompt, prediction, reference))
            ]
            if judged:
                scores["judge"] = round(sum(judged) / len(judged), 3)
        return {"scores": scores, "preds": predictions, "kind": runtime.spec.kind}
    finally:
        if runtime is not None:
            runtime.unload()


def main(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    try:
        rows = load_rows(cfg["dataset_id"], int(cfg.get("max_samples", 50)))
        pairs = [_extract(row) for row in rows]
        prompts, refs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        emit({"type": "log", "message": f"Evaluating {len(cfg['models'])} model(s) on {len(prompts)} samples."})

        per_model: dict[str, dict] = {}
        all_preds: dict[str, list[str]] = {}
        for model_ref in cfg["models"]:
            started = time.time()
            result = eval_model(model_ref, prompts, refs, cfg)
            per_model[model_ref] = {
                "scores": result["scores"],
                "seconds": round(time.time() - started, 1),
                "kind": result["kind"],
            }
            all_preds[model_ref] = result["preds"]
            emit({"type": "model_done", "model": model_ref, "scores": result["scores"]})

        samples = [
            {"prompt": prompts[index], "reference": refs[index],
             "preds": {model: all_preds[model][index] for model in cfg["models"]}}
            for index in range(min(len(prompts), 10))
        ]
        emit({"type": "result", "per_model": per_model, "samples": samples})
        return 0
    except Exception as exc:  # noqa: BLE001
        emit({"type": "log", "level": "error", "message": traceback.format_exc()})
        emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
