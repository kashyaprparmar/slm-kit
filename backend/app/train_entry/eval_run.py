"""Eval-harness subprocess: run an eval set through one or more models.

Invoked as ``python -m app.train_entry.eval_run <config.json>``. Emits newline
JSON events (log/progress/result/error) on stdout for the eval manager to stream
and persist. Heavy imports (torch/transformers) live inside ``main``.

Works on ANY model — a local checkpoint dir or an HF repo id — so it evaluates
models never trained in SLM Kit, exactly per the product spec.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

# Force UTF-8 on stdout regardless of platform/console codepage — model output
# and dataset text can contain arbitrary Unicode that Windows' default console
# encoding (cp1252) can't represent, which would crash sys.stdout.write.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_lock = threading.Lock()


def emit(obj: dict) -> None:
    with _lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


class Heartbeat:
    """Emits a `phase` event every few seconds during a blocking step."""

    def __init__(self, phase: str, model: str = "", every: float = 3.0) -> None:
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
    """Return (question, reference_answer) from a variety of instruction schemas."""
    if "messages" in row and isinstance(row["messages"], list):
        user = next((m["content"] for m in row["messages"] if m.get("role") == "user"), "")
        ref = next((m["content"] for m in row["messages"] if m.get("role") == "assistant"), "")
        return user, ref
    q = row.get("instruction") or row.get("prompt") or row.get("question") or ""
    ctx = row.get("input") or row.get("context") or ""
    ref = row.get("output") or row.get("response") or row.get("answer") or ""
    return (q if not ctx else f"{q}\n\n{ctx}"), ref


def load_rows(dataset_id: int, limit: int) -> list[dict]:
    from sqlmodel import Session

    from app.datasets.validate import _iter_records, detect_format
    from app.db.models import Dataset
    from app.db.session import engine

    with Session(engine) as db:
        ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found.")
    rows: list[dict] = []
    for _lineno, row in _iter_records(Path(ds.path), detect_format(ds.path)):
        if isinstance(row, dict):
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _encode(tok, messages: list[dict], plain_prompt: str, device):
    """Normalize tokenizer output to a **kwargs dict on `device`.

    transformers 5.x's apply_chat_template returns a BatchEncoding (dict-like)
    rather than a bare tensor, which breaks model.generate(input_ids=...)."""
    enc = None
    try:
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except (TypeError, ValueError):
        try:
            out = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            enc = {"input_ids": out}
        except Exception:
            enc = tok(plain_prompt, return_tensors="pt")
    if not hasattr(enc, "items"):
        enc = {"input_ids": enc}
    return {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}


def eval_model(model_ref: str, prompts: list[str], refs: list[str], cfg: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    emit({"type": "log", "message": f"Loading {model_ref} (first load downloads it — can take a few minutes)…"})
    with Heartbeat("loading_model", model=model_ref):
        tok = AutoTokenizer.from_pretrained(model_ref)
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, torch_dtype="auto", device_map="auto" if torch.cuda.is_available() else None
        )
        model.eval()
    emit({"type": "log", "message": f"{model_ref} loaded — generating predictions…"})
    device = next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    preds: list[str] = []
    total_nll, total_tokens = 0.0, 0
    n = len(prompts)
    for i, (q, ref) in enumerate(zip(prompts, refs)):
        messages = [{"role": "user", "content": q}]
        enc = _encode(tok, messages, q, device)
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=cfg.get("max_new_tokens", 128),
                do_sample=cfg.get("temperature", 0.0) > 0,
                temperature=max(0.01, cfg.get("temperature", 0.0)),
                top_p=cfg.get("top_p", 0.95),
                pad_token_id=tok.pad_token_id,
            )
        pred = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        preds.append(pred)

        # Perplexity of the reference continuation.
        if ref and "perplexity" in cfg.get("metrics", []):
            full = tok(q + "\n" + ref, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                loss = model(full, labels=full).loss
            total_nll += float(loss) * full.shape[1]
            total_tokens += full.shape[1]

        emit({"type": "progress", "model": model_ref, "done": i + 1, "total": n})

    from app.eval import metrics as M

    scores = M.compute(preds, refs, cfg.get("metrics", []))
    if total_tokens:
        import math
        scores["perplexity"] = round(math.exp(total_nll / total_tokens), 3)

    # Optional LLM-as-judge (mean normalized score across a sample).
    if cfg.get("judge"):
        from app.integrations import judge
        if judge.judge_available():
            js = []
            for q, ref, pred in list(zip(prompts, refs, preds))[:20]:
                r = judge.judge_answer(q, pred, ref)
                if r:
                    js.append(r["score"])
            if js:
                scores["judge"] = round(sum(js) / len(js), 3)

    # Free VRAM before the next model.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"scores": scores, "preds": preds}


def main(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text())
    try:
        rows = load_rows(cfg["dataset_id"], cfg.get("max_samples", 50))
        pairs = [_extract(r) for r in rows]
        prompts = [p for p, _ in pairs]
        refs = [r for _, r in pairs]
        emit({"type": "log", "message": f"Evaluating {len(cfg['models'])} model(s) on {len(prompts)} samples."})

        per_model: dict[str, dict] = {}
        all_preds: dict[str, list[str]] = {}
        for ref_id in cfg["models"]:
            t0 = time.time()
            res = eval_model(ref_id, prompts, refs, cfg)
            per_model[ref_id] = {"scores": res["scores"], "seconds": round(time.time() - t0, 1)}
            all_preds[ref_id] = res["preds"]
            emit({"type": "model_done", "model": ref_id, "scores": res["scores"]})

        samples = [
            {"prompt": prompts[i], "reference": refs[i], "preds": {m: all_preds[m][i] for m in cfg["models"]}}
            for i in range(min(len(prompts), 10))
        ]
        emit({"type": "result", "per_model": per_model, "samples": samples})
        return 0
    except Exception as exc:  # noqa: BLE001
        emit({"type": "log", "message": traceback.format_exc()})
        emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
