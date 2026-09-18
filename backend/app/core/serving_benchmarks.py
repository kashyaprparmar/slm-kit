"""Small, provider-neutral benchmark jobs for already running deployments."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import UTC, datetime
from statistics import fmean

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session, select

from app.core.hardware import poller
from app.db.models import ModelArtifact, ServingBenchmark
from app.db.session import engine
from app.serving.providers import benchmark_stream, get_provider


def _utcnow():
    return datetime.now(UTC)


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    artifact_id: int | None = Field(default=None, ge=1)
    prompt: str = Field(min_length=1, max_length=32_000)
    max_tokens: int = Field(default=128, ge=1, le=2_048)
    requests: int = Field(default=8, ge=1, le=100)
    concurrency: int = Field(default=1, ge=1, le=16)
    warmup_requests: int = Field(default=1, ge=0, le=10)


def _percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(p: float) -> float:
        pos = (len(ordered) - 1) * p
        low, high = math.floor(pos), math.ceil(pos)
        return (
            ordered[low]
            if low == high
            else ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
        )

    return {
        "count": len(ordered),
        "mean": round(fmean(ordered), 3),
        "p50": round(at(0.50), 3),
        "p95": round(at(0.95), 3),
        "p99": round(at(0.99), 3),
    }


def _hardware_snapshot() -> dict:
    return poller.latest.model_dump(mode="json")


def _used_vram(snapshot: dict) -> float | None:
    total, free = snapshot.get("vram_total_mb"), snapshot.get("vram_free_mb")
    return float(total - free) if total is not None and free is not None else None


class ServingBenchmarkManager:
    def __init__(self):
        self.tasks: dict[int, asyncio.Task] = {}
        self.cancel_events: dict[int, asyncio.Event] = {}

    async def start(self, request: BenchmarkRequest) -> dict:
        provider = get_provider(request.provider)
        state = await provider.status()
        if request.provider == "ollama":
            active, model_ref = (
                bool(state.get("running") and state.get("managed_model")),
                state.get("managed_model"),
            )
        elif request.provider == "transformers":
            active, model_ref = (
                bool(state.get("active") and state.get("state") == "ready"),
                state.get("model_ref"),
            )
        else:
            active = bool(state.get("active"))
            model_ref = (state.get("models") or [{}])[0].get("id")
        if not active or not model_ref:
            raise ValueError("Start a managed compatible model before benchmarking it.")
        artifact_id = request.artifact_id
        with Session(engine) as db:
            artifact = db.get(ModelArtifact, artifact_id) if artifact_id else None
            if artifact is None:
                # A normal Registry deployment has a local artifact identity;
                # discover it here so callers do not have to duplicate that
                # lookup in every serving UI/API client.
                artifact = db.exec(
                    select(ModelArtifact)
                    .where(ModelArtifact.local_path == str(model_ref))
                    .where(ModelArtifact.status == "ready")
                    .order_by(ModelArtifact.created_at.desc())
                ).first()
                artifact_id = artifact.id if artifact else None
            if artifact_id and (not artifact or artifact.status != "ready"):
                raise ValueError("Choose a ready model artifact.")
            if artifact and request.provider == "transformers" and artifact.local_path != model_ref:
                raise ValueError("The selected artifact is not the active Transformers deployment.")
            record = ServingBenchmark(
                artifact_id=artifact_id,
                provider=request.provider,
                model_ref=str(model_ref),
                endpoint=str(state.get("endpoint") or ""),
                config={**request.model_dump(), "provider_state": state},
                hardware={"before": _hardware_snapshot()},
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            benchmark_id = record.id
        cancelled = asyncio.Event()
        self.cancel_events[benchmark_id] = cancelled
        task = asyncio.create_task(self._run(benchmark_id, request, cancelled))
        self.tasks[benchmark_id] = task
        task.add_done_callback(
            lambda _: (
                self.tasks.pop(benchmark_id, None),
                self.cancel_events.pop(benchmark_id, None),
            )
        )
        return {"benchmark_id": benchmark_id, "status": "running"}

    async def _one(self, request: BenchmarkRequest, cancelled: asyncio.Event) -> dict:
        started, first, previous = time.perf_counter(), None, None
        deltas: list[float] = []
        completion_tokens = None
        generation_seconds = None
        try:
            async for event in benchmark_stream(
                request.provider, request.prompt, request.max_tokens
            ):
                if cancelled.is_set():
                    raise asyncio.CancelledError()
                now = time.perf_counter()
                if event.get("text"):
                    if first is None:
                        first = now
                    elif previous is not None:
                        deltas.append((now - previous) * 1000)
                    previous = now
                completion_tokens = (
                    event.get("completion_tokens")
                    if event.get("completion_tokens") is not None
                    else completion_tokens
                )
                generation_seconds = event.get("generation_seconds") or generation_seconds
            ended = time.perf_counter()
            duration = ended - started
            generation = generation_seconds or ((ended - first) if first else None)
            tokens_per_second = (
                completion_tokens / generation if completion_tokens and generation else None
            )
            return {
                "duration_ms": duration * 1000,
                "ttft_ms": ((first - started) * 1000 if first else None),
                "inter_token_latency_ms": deltas,
                "completion_tokens": completion_tokens,
                "tokens_per_second": tokens_per_second,
            }
        except Exception as exc:
            return {"error": str(exc)}

    async def _vllm_kv(self, endpoint: str) -> float | None:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                text = (await client.get(endpoint.rstrip("/") + "/metrics")).text
            values = []
            for line in text.splitlines():
                if line.startswith(("vllm:kv_cache_usage_perc", "vllm_gpu_cache_usage_perc")):
                    values.append(float(line.rsplit(" ", 1)[-1]))
            return max(values) if values else None
        except (httpx.HTTPError, ValueError):
            return None

    async def _run(self, benchmark_id: int, request: BenchmarkRequest, cancelled: asyncio.Event):
        samples: list[dict] = []
        vram_samples = [_hardware_snapshot()]
        kv_samples: list[float] = []

        async def monitor():
            while not cancelled.is_set():
                vram_samples.append(_hardware_snapshot())
                with Session(engine) as db:
                    record = db.get(ServingBenchmark, benchmark_id)
                    endpoint = record.endpoint if record else ""
                if request.provider == "vllm" and endpoint:
                    value = await self._vllm_kv(endpoint)
                    if value is not None:
                        kv_samples.append(value)
                await asyncio.sleep(0.5)

        monitor_task = asyncio.create_task(monitor())
        try:
            for _ in range(request.warmup_requests):
                if cancelled.is_set():
                    raise asyncio.CancelledError()
                await self._one(request, cancelled)
            started = time.perf_counter()
            semaphore = asyncio.Semaphore(request.concurrency)

            async def run_one():
                async with semaphore:
                    return await self._one(request, cancelled)

            samples = await asyncio.gather(*(run_one() for _ in range(request.requests)))
            elapsed = time.perf_counter() - started
            if cancelled.is_set():
                raise asyncio.CancelledError()
            successes = [s for s in samples if not s.get("error")]
            failures = [s for s in samples if s.get("error")]
            ttft = [s["ttft_ms"] for s in successes if s.get("ttft_ms") is not None]
            itl = [value for s in successes for value in s.get("inter_token_latency_ms", [])]
            latency = [s["duration_ms"] for s in successes]
            token_rates = [
                s["tokens_per_second"] for s in successes if s.get("tokens_per_second") is not None
            ]
            token_counts = [
                s["completion_tokens"] for s in successes if s.get("completion_tokens") is not None
            ]
            vram_samples.append(_hardware_snapshot())
            used = [
                value for snapshot in vram_samples if (value := _used_vram(snapshot)) is not None
            ]
            results = {
                "request_count": request.requests,
                "successful_requests": len(successes),
                "error_count": len(failures),
                "error_rate": round(len(failures) / request.requests, 6),
                "concurrency": request.concurrency,
                "warmup_requests": request.warmup_requests,
                "elapsed_seconds": round(elapsed, 4),
                "requests_per_second": round(len(successes) / elapsed, 4) if elapsed else None,
                "ttft_ms": _percentiles(ttft),
                "inter_token_latency_ms": _percentiles(itl),
                "end_to_end_latency_ms": _percentiles(latency),
                "tokens_per_second": _percentiles(token_rates),
                "total_completion_tokens": sum(token_counts) if token_counts else None,
                "throughput_tokens_per_second": round(sum(token_counts) / elapsed, 4)
                if token_counts and elapsed
                else None,
                "vram": {
                    "before": vram_samples[0],
                    "after": vram_samples[-1],
                    "peak_used_mb": max(used) if used else None,
                },
                "kv_cache_utilization": {
                    "source": "vllm_prometheus" if request.provider == "vllm" else None,
                    "peak_percent": max(kv_samples) if kv_samples else None,
                    "available": bool(kv_samples),
                },
                "failures": [s["error"] for s in failures[:10]],
            }
            with Session(engine) as db:
                record = db.get(ServingBenchmark, benchmark_id)
                record.status, record.results, record.completed_at = "completed", results, _utcnow()
                record.hardware = {"before": vram_samples[0], "after": vram_samples[-1]}
                db.add(record)
                db.commit()
        except asyncio.CancelledError:
            with Session(engine) as db:
                record = db.get(ServingBenchmark, benchmark_id)
                record.status, record.error, record.completed_at = (
                    "cancelled",
                    "Benchmark cancelled.",
                    _utcnow(),
                )
                db.add(record)
                db.commit()
            raise
        except Exception as exc:
            with Session(engine) as db:
                record = db.get(ServingBenchmark, benchmark_id)
                record.status, record.error, record.completed_at = "failed", str(exc), _utcnow()
                db.add(record)
                db.commit()
        finally:
            cancelled.set()
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)

    async def cancel(self, benchmark_id: int) -> bool:
        task = self.tasks.get(benchmark_id)
        if not task:
            return False
        self.cancel_events[benchmark_id].set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def shutdown(self):
        await asyncio.gather(*(self.cancel(identifier) for identifier in list(self.tasks)))

    def recover(self):
        with Session(engine) as db:
            for record in db.exec(
                select(ServingBenchmark).where(ServingBenchmark.status == "running")
            ):
                record.status, record.error, record.completed_at = (
                    "failed",
                    "API restarted during benchmark.",
                    _utcnow(),
                )
                db.add(record)
            db.commit()


manager = ServingBenchmarkManager()
