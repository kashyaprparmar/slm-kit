"""Hugging Face Hub client: model metadata, listing, publish, import.

All functions are best-effort and offline-safe: anything that needs the network
degrades to a sensible fallback rather than raising, so the core app keeps
working without connectivity or a token.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings
from app.integrations.estimator import ModelSpec, spec_from_name

_settings = get_settings()


def _token() -> Optional[str]:
    return _settings.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _hf_home() -> dict[str, str]:
    # Keep HF's cache under our home dir so everything is in one place.
    return {"HF_HOME": str(_settings.hf_cache_dir)}


# --------------------------------------------------------------------------- #
# Model metadata → ModelSpec (drives the fallback estimator)
# --------------------------------------------------------------------------- #
from functools import lru_cache


@lru_cache(maxsize=64)
def get_model_spec(repo_or_path: str) -> ModelSpec:
    """Resolve architecture + parameter count for a repo id or local dir.

    Order: local config.json → HF Hub config.json + safetensors index →
    name-based heuristic. Cached — the estimate endpoint calls this on every
    debounced config change.
    """
    cfg = _load_config(repo_or_path)
    n_params = _param_count(repo_or_path, cfg)
    if cfg:
        hidden = int(cfg.get("hidden_size", 2048))
        layers = int(cfg.get("num_hidden_layers", 24))
        kv_heads = int(cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 8)))
        n_heads = int(cfg.get("num_attention_heads", 16))
        head_dim = int(cfg.get("head_dim", max(1, hidden // max(1, n_heads))))
        if n_params:
            return ModelSpec(n_params, hidden, layers, kv_heads, head_dim)
    # Fallback purely from the name.
    return spec_from_name(repo_or_path)


def _load_config(repo_or_path: str) -> Optional[dict[str, Any]]:
    local = Path(repo_or_path) / "config.json"
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            return None
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_or_path,
            filename="config.json",
            token=_token(),
            cache_dir=str(_settings.hf_cache_dir),
        )
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _param_count(repo_or_path: str, cfg: Optional[dict]) -> Optional[int]:
    # Prefer the exact count HF exposes via safetensors metadata.
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_or_path, token=_token(), files_metadata=False)
        st = getattr(info, "safetensors", None)
        if st and getattr(st, "total", None):
            return int(st.total)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Registry operations
# --------------------------------------------------------------------------- #
def list_my_models(limit: int = 100) -> list[dict[str, Any]]:
    """List repos owned by the authenticated user. Empty list if no token."""
    tok = _token()
    if not tok:
        return []
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        me = api.whoami(token=tok)
        author = me.get("name") if isinstance(me, dict) else None
        repos = api.list_models(author=author, token=tok, limit=limit)
        return [
            {
                "repo_id": r.id,
                "private": getattr(r, "private", None),
                "downloads": getattr(r, "downloads", None),
                "likes": getattr(r, "likes", None),
                "updated": str(getattr(r, "last_modified", "")),
            }
            for r in repos
        ]
    except Exception:
        return []


def publish_model(
    local_dir: str,
    repo_id: str,
    private: bool = True,
    model_card: Optional[str] = None,
) -> str:
    """Upload a local model dir to HF Hub; returns the repo URL. Requires token."""
    tok = _token()
    if not tok:
        raise RuntimeError("No Hugging Face token configured (set SLMKIT_HF_TOKEN).")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, token=tok, exist_ok=True)
    if model_card:
        (Path(local_dir) / "README.md").write_text(model_card, encoding="utf-8")
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, token=tok)
    return f"https://huggingface.co/{repo_id}"


def import_model(repo_id: str, dest_dir: str) -> str:
    """Download a repo into ``dest_dir`` for further training/testing."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=dest_dir,
        token=_token(),
        cache_dir=str(_settings.hf_cache_dir),
    )
    return path
