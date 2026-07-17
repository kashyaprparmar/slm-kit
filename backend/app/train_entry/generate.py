"""Playground generation subprocess: stream tokens from any model.

Invoked as ``python -m app.train_entry.generate <config.json>``. Emits JSON
token/log/phase/done events on stdout. Loads the model, streams tokens via a
TextIteratorStreamer, then exits so VRAM is released.

A background heartbeat thread emits ``phase`` events while the (blocking) model
load runs, so the UI can show "loading… (12s)" instead of an indefinite spinner.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

# Force UTF-8 on stdout regardless of platform/console codepage — streamed
# tokens can contain arbitrary Unicode that Windows' default console encoding
# (cp1252) can't represent, which would crash sys.stdout.write.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_lock = threading.Lock()


def emit(obj: dict) -> None:
    with _lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


class Heartbeat:
    """Emits a `phase` event every few seconds during a blocking step."""

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


def _encode(tok, messages: list[dict], plain_prompt: str, device):
    """Return a dict of model inputs (input_ids [+ attention_mask]) on `device`.

    transformers 5.x's apply_chat_template returns a BatchEncoding (dict-like),
    not a bare tensor, so we normalize to a **kwargs-friendly dict either way.
    """
    enc = None
    try:
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except (TypeError, ValueError):
        # Older transformers: no return_dict arg, or no chat template.
        try:
            out = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            enc = {"input_ids": out}
        except Exception:
            enc = tok(plain_prompt, return_tensors="pt")
    if not hasattr(enc, "items"):  # a bare tensor slipped through
        enc = {"input_ids": enc}
    return {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}


def main(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text())
    try:
        emit({"type": "phase", "phase": "importing", "elapsed": 0})
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        model_ref = cfg["model_ref"]
        emit({"type": "log", "message": f"Loading model: {model_ref}"})
        emit({"type": "log", "message": "(first load downloads the model — this can take a few minutes)"})

        with Heartbeat("loading_model"):
            tok = AutoTokenizer.from_pretrained(model_ref)
            model = AutoModelForCausalLM.from_pretrained(
                model_ref, torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model.eval()

        device = next(model.parameters()).device
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        emit({"type": "log", "message": f"Model loaded on {device}. Generating…"})
        emit({"type": "phase", "phase": "generating", "elapsed": 0})

        messages = [{"role": "user", "content": cfg["prompt"]}]
        enc = _encode(tok, messages, cfg["prompt"], device)

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **enc,
            streamer=streamer,
            max_new_tokens=cfg.get("max_new_tokens", 256),
            do_sample=cfg.get("temperature", 0.7) > 0,
            temperature=max(0.01, cfg.get("temperature", 0.7)),
            top_p=cfg.get("top_p", 0.95),
            top_k=cfg.get("top_k", 50),
            repetition_penalty=cfg.get("repetition_penalty", 1.1),
            pad_token_id=tok.pad_token_id,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        n_tokens = 0
        for text in streamer:
            if text:
                n_tokens += 1
                emit({"type": "token", "text": text})
        thread.join()
        emit({"type": "log", "message": f"Done — generated {n_tokens} token chunks."})
        emit({"type": "done"})
        return 0
    except Exception as exc:  # noqa: BLE001
        emit({"type": "log", "level": "error", "message": traceback.format_exc()})
        emit({"type": "error", "message": str(exc)})
        emit({"type": "done"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
