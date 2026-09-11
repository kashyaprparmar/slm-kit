"""Application settings and filesystem layout.

Everything SLM Kit writes (SQLite DB, downloaded/trained models, datasets, run
workdirs) lives under a single home directory so the whole install is easy to
back up or wipe. Override any field via environment variables prefixed with
``SLMKIT_`` or a ``.env`` file in the backend working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SLMKIT_", env_file=".env", extra="ignore"
    )

    # Where all persistent state lives. Default keeps it out of the repo.
    home: Path = Path.home() / ".slmkit"

    # API server
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_mb: int = 512

    # Optional external integrations (app works fully without any of these).
    hf_token: str | None = None
    judge_provider: str = "anthropic"      # anthropic | openai
    judge_api_key: str | None = None
    judge_model: str = "claude-sonnet-5"

    # Some HF repositories require executable custom modeling code. Disabled by
    # default for safety; operators can opt in for a repository they trust.
    trust_remote_code: bool = False

    # llmfit: shelled out via JSON. If the binary is missing we fall back to
    # the internal VRAM estimator, so this is best-effort.
    llmfit_bin: str = "llmfit"

    # Playground serving engine: "auto" uses vLLM if installed (warm, persistent
    # server — no per-request model reload), else falls back to the transformers
    # subprocess. vLLM is Linux-first; on native Windows it's normally absent, so
    # "auto" degrades to "transformers" there with no behavior change.
    serve_engine: str = "auto"             # auto | vllm | transformers
    vllm_port: int = 8801
    vllm_gpu_memory_utilization: float = 0.55  # conservative default for 8GB cards
    vllm_max_model_len: int = 4096
    vllm_idle_timeout_seconds: float = 600.0   # auto-stop a warm server after 10 idle min

    # Managed local deployment endpoint. This is deliberately separate from the
    # playground's short-lived/warm vLLM process: it serves one selected model
    # through an OpenAI-compatible API until explicitly stopped.
    deploy_host: str = "127.0.0.1"
    deploy_port: int = 8802
    deploy_startup_timeout_seconds: float = 180.0
    ollama_url: str = "http://127.0.0.1:11434"

    # Hardware polling cadence (seconds). Faster while a GPU job is running.
    idle_poll_seconds: float = 3.0
    active_poll_seconds: float = 1.5

    # Target hardware budget. Defaults tuned for an RTX 4060 8GB / 16GB RAM box.
    # Used by validators and the "fits / tight / won't fit" indicator.
    vram_budget_mb: int = 8192
    ram_budget_mb: int = 16384
    # Fraction of VRAM we consider "safe" to plan against (leaves headroom for
    # the desktop compositor, driver, and fragmentation).
    vram_safe_fraction: float = 0.90

    # ---- Derived paths -------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.home / "slmkit.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    @property
    def datasets_dir(self) -> Path:
        return self.home / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def hf_cache_dir(self) -> Path:
        return self.home / "hf-cache"

    @property
    def deployments_dir(self) -> Path:
        return self.home / "deployments"

    def ensure_dirs(self) -> None:
        for d in (
            self.home,
            self.datasets_dir,
            self.models_dir,
            self.runs_dir,
            self.hf_cache_dir,
            self.deployments_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
