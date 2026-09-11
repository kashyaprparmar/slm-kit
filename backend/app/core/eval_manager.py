"""Supervises eval + playground-generation subprocesses.

Both load a model (GPU-heavy), so this serializes them and refuses to start while
another managed workload owns the GPU — keeping the single-GPU box safe. Subprocess stdout
(newline JSON) is streamed to a WebSocket topic; eval results are persisted to the
``EvalResult`` table.

The currently-running subprocess is tracked so it can be **cancelled** — without
that, a model load that hangs would lock the whole Eval Lab until a backend
restart. ``_busy`` is always released in a ``finally``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

from sqlmodel import Session

from app.config import get_settings
from app.core.deployment import manager as deployment_manager
from app.core.log_capture import append_log_line
from app.core.logging_config import get_logger
from app.core.queue import queue
from app.core.resources import gpu
from app.core.ws import hub
from app.db.models import EvalResult
from app.db.session import engine
from app.integrations import vllm_serve

_settings = get_settings()
log = get_logger(__name__)


def log_path(kind: str, ident) -> Path:
    """Persisted-log path for an eval or generate job (kind: 'eval' | 'gen')."""
    return _settings.runs_dir / kind / str(ident) / "output.log"


def _want_vllm() -> bool:
    """Whether the Playground should try vLLM for this request."""
    if _settings.serve_engine == "transformers":
        return False
    if _settings.serve_engine == "vllm":
        return True
    return vllm_serve.vllm_available()  # "auto"


class EvalManager:
    def __init__(self) -> None:
        self._busy = False
        self._lease: str | None = None
        self._cancelling = False
        self._tasks: set[asyncio.Task] = set()
        # The running eval/gen subprocess + a human label, for cancel + status.
        self._proc: asyncio.subprocess.Process | None = None
        self._current: dict | None = None  # {"kind": "eval"|"gen", "id": ..., "topic": ...}

    # ---- status --------------------------------------------------------
    def busy(self) -> bool:
        # A warm Playground server can accept another generation, so it is
        # exposed through current() but does not block the next prompt. Training
        # still stops it explicitly before claiming the GPU.
        return (
            self._busy
            or queue.current_id is not None
            or deployment_manager.active
        )

    def current(self) -> dict | None:
        if self._current:
            return self._current
        if vllm_serve.manager.model is not None:
            return {"kind": "vllm-server", "id": vllm_serve.manager.model, "topic": None}
        if deployment_manager.active:
            return {"kind": "deployment", "id": deployment_manager.status()["model_ref"], "topic": None}
        return None

    def engine_info(self) -> dict:
        return {
            "vllm_available": vllm_serve.vllm_available(),
            "serve_engine_setting": _settings.serve_engine,
            "vllm_warm_model": vllm_serve.manager.model,
        }

    def _guard_and_acquire(self, kind: str, ident) -> None:
        """Check + set busy synchronously so two rapid requests can't both pass."""
        if queue.current_id is not None:
            raise RuntimeError("GPU is busy with a training run. Try again when it finishes.")
        if deployment_manager.active:
            raise RuntimeError("GPU is serving a deployed model. Stop it in Model Registry before evaluating.")
        if kind != "gen" and vllm_serve.manager.model is not None:
            raise RuntimeError("GPU has a warm Playground model. Stop/clear it before running evaluation.")
        if self._busy:
            raise RuntimeError("GPU is busy with another eval/generation. Cancel it or wait.")
        self._lease = gpu.acquire(kind, ident)
        self._busy = True
        self._current = {"kind": kind, "id": ident, "topic": f"{kind}:{ident}"}
        log.info("eval-manager acquired GPU for %s %s", kind, ident)

    def _release(self) -> None:
        if self._cancelling:
            return
        gpu.release(self._lease)
        self._lease = None
        self._busy = False
        self._proc = None
        self._current = None

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---- cancellation --------------------------------------------------
    async def cancel_current(self) -> bool:
        """Stop whatever's holding the GPU — a tracked subprocess, a warm vLLM
        server, or (defensively) a stuck busy flag with neither. Returns True if
        anything was actually cancelled/cleared."""
        self._cancelling = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        proc = self._proc
        cur = self._current
        did_something = False

        if proc is not None and cur is not None:
            log.info("cancel: terminating %s %s (pid=%s)", cur["kind"], cur["id"], proc.pid)
            if cur.get("topic"):
                await hub.publish(cur["topic"], {"type": "log", "level": "warning",
                                                 "message": "Cancellation requested — stopping…"})
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
            except ProcessLookupError:
                pass
            if cur.get("topic"):
                await hub.publish(cur["topic"], {"type": "error", "message": "Cancelled by user"})
                await hub.publish(cur["topic"], {"type": "done"})
            did_something = True

        if vllm_serve.manager.model is not None:
            log.info("cancel: stopping warm vLLM server (model=%s)", vllm_serve.manager.model)
            await vllm_serve.manager.stop()
            did_something = True

        if self._busy and not did_something:
            log.warning("cancel: no tracked process but busy=True — clearing stuck flag")
            did_something = True

        if cur and cur["kind"] == "eval":
            self._persist(cur["id"], None, "Cancelled by user")
        self._cancelling = False
        self._release()
        return did_something

    # ---- Eval harness --------------------------------------------------
    async def start_eval(self, config: dict) -> int:
        # Acquire before writing history. A rejected request must not leave a
        # permanently "running" result row behind.
        self._guard_and_acquire("eval", "pending")
        with Session(engine) as db:
            try:
                row = EvalResult(
                    model_ref=",".join(config["models"]),
                    dataset_id=config.get("dataset_id"),
                    scores={},
                    detail={"status": "running", "models": config["models"]},
                )
                db.add(row)
                db.commit()
                db.refresh(row)
                eval_id = row.id
            except Exception:
                self._release()
                raise
        self._current = {"kind": "eval", "id": eval_id, "topic": f"eval:{eval_id}"}
        gpu.update(self._lease, "evaluation", eval_id)
        workdir = _settings.runs_dir / "eval" / str(eval_id)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        log.info("eval %s starting: models=%s dataset=%s", eval_id, config["models"], config.get("dataset_id"))
        self._spawn(self._run(eval_id, workdir))
        return eval_id

    async def _run(self, eval_id: int, workdir: Path) -> None:
        topic = f"eval:{eval_id}"
        cfg_path = workdir / "config.json"
        lpath = log_path("eval", eval_id)
        result: dict | None = None
        error: str | None = None
        try:
            await asyncio.sleep(0.4)  # let the client attach its WebSocket first
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "app.train_entry.eval_run", str(cfg_path),
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            self._proc = proc
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    await hub.publish(topic, {"type": "log", "message": line})
                    append_log_line(lpath, "info", line)
                    continue
                await hub.publish(topic, ev)
                etype = ev.get("type")
                if etype == "result":
                    result = ev
                    per_model = ev.get("per_model", {})
                    append_log_line(lpath, "info", f"Result ready: {len(per_model)} model(s) scored — " +
                                     "; ".join(f"{m}: {v.get('scores')}" for m, v in per_model.items()))
                elif etype == "error":
                    error = ev.get("message")
                    append_log_line(lpath, "error", str(error))
                elif etype == "log":
                    append_log_line(lpath, ev.get("level", "info"), ev.get("message", ""))
                elif etype == "phase":
                    append_log_line(lpath, "info",
                                     f"[{ev.get('phase')}] model={ev.get('model', '')} elapsed={ev.get('elapsed', 0)}s")
                elif etype == "progress":
                    append_log_line(lpath, "info", f"progress: {ev.get('model')} {ev.get('done')}/{ev.get('total')}")
                elif etype == "model_done":
                    append_log_line(lpath, "info", f"{ev.get('model')} done: {ev.get('scores')}")
            await proc.wait()
            if proc.returncode not in (0, None) and not error:
                error = f"Eval subprocess exited with code {proc.returncode}"
        except Exception as exc:
            error = str(exc)
            log.exception("eval %s crashed", eval_id)
            append_log_line(lpath, "error", f"Eval manager error: {exc}")
        finally:
            if self._cancelling:
                error = "Cancelled by user"
            self._release()
            self._persist(eval_id, result, error)
            terminal = "cancelled" if error == "Cancelled by user" else "failed" if error else "done"
            log.info("eval %s finished: %s", eval_id, terminal)
            append_log_line(lpath, "warning" if terminal == "cancelled" else "error" if error else "info", f"Eval finished: {terminal}")
            await hub.publish(topic, {"type": "status", "status": terminal})

    def _persist(self, eval_id: int, result: dict | None, error: str | None) -> None:
        with Session(engine) as db:
            row = db.get(EvalResult, eval_id)
            if not row:
                return
            if result and not error:
                row.scores = {m: v["scores"] for m, v in result.get("per_model", {}).items()}
                row.detail = {
                    "status": "done",
                    "per_model": result.get("per_model", {}),
                    "samples": result.get("samples", []),
                }
            else:
                row.detail = {**(row.detail or {}), "status": "cancelled" if error == "Cancelled by user" else "failed", "error": error}
            db.add(row)
            db.commit()

    # ---- Playground generation ----------------------------------------
    async def start_generate(self, config: dict) -> str:
        gen_id = uuid.uuid4().hex[:12]
        self._guard_and_acquire("gen", gen_id)
        from app.model_refs import resolve_model_ref
        resolved = resolve_model_ref(config["model_ref"])
        config["model_ref"] = resolved.load_ref
        use_vllm = _want_vllm() and resolved.kind == "transformers"
        log.info("generate %s starting: model=%s engine=%s", gen_id, config.get("model_ref"),
                 "vllm" if use_vllm else "transformers")
        if use_vllm:
            self._spawn(self._generate_vllm(gen_id, config))
        else:
            workdir = _settings.runs_dir / "gen" / gen_id
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            self._spawn(self._generate_transformers(gen_id, workdir))
        return gen_id

    async def _generate_transformers(self, gen_id: str, workdir: Path) -> None:
        """Cold-load the model in a fresh subprocess each call (portable, works
        everywhere torch/transformers do — including native Windows)."""
        topic = f"gen:{gen_id}"
        cfg_path = workdir / "config.json"
        lpath = log_path("gen", gen_id)
        # Individual "token" events aren't persisted (one line per token would
        # flood the log for a long response) — the full text is logged once at
        # the end instead, alongside every other event type in full.
        chunks: list[str] = []
        error_message = None
        generation_metrics = {}
        try:
            await asyncio.sleep(0.4)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "app.train_entry.generate", str(cfg_path),
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            self._proc = proc
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    ev = {"type": "log", "message": line}
                    append_log_line(lpath, "info", line)
                await hub.publish(topic, ev)
                etype = ev.get("type")
                if etype == "token":
                    chunks.append(ev.get("text", ""))
                elif etype == "log":
                    append_log_line(lpath, ev.get("level", "info"), ev.get("message", ""))
                elif etype == "phase":
                    append_log_line(lpath, "info", f"[{ev.get('phase')}] elapsed={ev.get('elapsed', 0)}s")
                elif etype == "error":
                    error_message = str(ev.get("message", "Generation failed"))
                    append_log_line(lpath, "error", str(ev.get("message", "")))
                elif etype == "metrics":
                    generation_metrics = ev
            await proc.wait()
            if proc.returncode not in (0, None):
                msg = f"exited with code {proc.returncode}"
                append_log_line(lpath, "error", msg)
                await hub.publish(topic, {"type": "error", "message": msg})
        except Exception as exc:
            log.exception("generate %s crashed", gen_id)
            append_log_line(lpath, "error", f"Generate manager error: {exc}")
            await hub.publish(topic, {"type": "error", "message": str(exc)})
        finally:
            if self._cancelling:
                error_message = "Cancelled by user"
            if chunks:
                append_log_line(lpath, "info", f"Generated response ({len(chunks)} chunks): " + "".join(chunks))
            (workdir / "result.json").write_text(json.dumps({"output": "".join(chunks), "metrics": generation_metrics,
                "status": "cancelled" if error_message == "Cancelled by user" else "failed" if error_message else "done", "error": error_message}), encoding="utf-8")
            self._release()
            log.info("generate %s finished", gen_id)
            append_log_line(lpath, "info", "Generation finished")
            await hub.publish(topic, {"type": "done"})

    async def _generate_vllm(self, gen_id: str, config: dict) -> None:
        """Serve via a warm vLLM server — fast after the first request for a
        given model, since it isn't reloaded each time. Falls back to the
        transformers subprocess if vLLM fails to start.

        ``fell_back`` matters because a bare ``return`` inside ``except`` does
        NOT skip ``finally`` — without the flag we'd release/publish "done"
        twice: once here, once inside the transformers path we fall back to.
        """
        topic = f"gen:{gen_id}"
        model_ref = config["model_ref"]
        lpath = log_path("gen", gen_id)
        chunks: list[str] = []
        fell_back = False
        try:
            await asyncio.sleep(0.4)
            already_warm = vllm_serve.manager.model == model_ref
            if already_warm:
                msg = f"Reusing warm vLLM server for {model_ref}."
                await hub.publish(topic, {"type": "log", "message": msg})
                append_log_line(lpath, "info", msg)
            else:
                await hub.publish(topic, {"type": "phase", "phase": "loading_model", "elapsed": 0})
                append_log_line(lpath, "info", f"[loading_model] starting vLLM server for {model_ref}")

            def on_log(line: str) -> None:
                # self._spawn (not a bare asyncio.create_task) so the task is
                # held in self._tasks — an unreferenced task can be garbage
                # collected mid-publish (same class of bug fixed earlier in
                # this file for the eval/generate subprocess readers).
                self._spawn(hub.publish(topic, {"type": "log", "message": line}))
                append_log_line(lpath, "info", line)  # raw vLLM server stdout

            await vllm_serve.manager.ensure(model_ref, log_cb=on_log)
            await hub.publish(topic, {"type": "phase", "phase": "generating", "elapsed": 0})
            append_log_line(lpath, "info", "[generating] vLLM server ready — streaming response")
            async for text in vllm_serve.manager.stream_chat(config["prompt"], config):
                chunks.append(text)
                await hub.publish(topic, {"type": "token", "text": text})
        except Exception as exc:
            log.exception("vLLM generate %s failed — falling back to transformers", gen_id)
            msg = f"vLLM path failed ({exc}); retrying via transformers…"
            await hub.publish(topic, {"type": "log", "level": "warning", "message": msg})
            append_log_line(lpath, "warning", msg)
            fell_back = True
        finally:
            if chunks:
                append_log_line(lpath, "info", f"Generated response via vLLM ({len(chunks)} chunks): " + "".join(chunks))
            if fell_back:
                # The transformers path owns its own release/done lifecycle.
                workdir = _settings.runs_dir / "gen" / gen_id
                workdir.mkdir(parents=True, exist_ok=True)
                (workdir / "config.json").write_text(json.dumps(config), encoding="utf-8")
                await self._generate_transformers(gen_id, workdir)
            else:
                await vllm_serve.manager.stop()
                self._release()
                log.info("generate %s finished (vllm)", gen_id)
                append_log_line(lpath, "info", "Generation finished")
                await hub.publish(topic, {"type": "done"})


manager = EvalManager()
