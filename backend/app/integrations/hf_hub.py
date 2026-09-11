"""Hugging Face Hub client: model metadata, listing, publish, import.

All functions are best-effort and offline-safe: anything that needs the network
degrades to a sensible fallback rather than raising, so the core app keeps
working without connectivity or a token.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.integrations.estimator import ModelSpec, spec_from_name

_settings = get_settings()


def _token() -> str | None:
    return _settings.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _hf_home() -> dict[str, str]:
    # Keep HF's cache under our home dir so everything is in one place.
    return {"HF_HOME": str(_settings.hf_cache_dir)}


# --------------------------------------------------------------------------- #
# Model metadata → ModelSpec (drives the fallback estimator)
# --------------------------------------------------------------------------- #
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


def _load_config(repo_or_path: str) -> dict[str, Any] | None:
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


def _param_count(repo_or_path: str, cfg: dict | None) -> int | None:
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


def inspect_model(repo_or_path: str, revision: str | None = None) -> dict[str, Any]:
    """Return lightweight compatibility metadata without loading model weights.

    Local directories are inspected directly. Hub repositories use only small
    metadata files and the model-info API, so this remains safe to call before
    a user commits to downloading a checkpoint.
    """
    root = Path(repo_or_path)
    cfg: dict[str, Any] = {}
    tokenizer_cfg: dict[str, Any] = {}
    adapter_cfg: dict[str, Any] = {}
    scratch_cfg: dict[str, Any] = {}
    files: list[str] = []
    resolved_revision = revision

    if root.is_dir():
        for name, target in (("config.json", cfg), ("tokenizer_config.json", tokenizer_cfg), ("adapter_config.json", adapter_cfg), ("arch.json", scratch_cfg)):
            path = root / name
            if path.is_file():
                try:
                    target.update(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    pass
        files = [p.name for p in root.iterdir() if p.is_file()]
    else:
        try:
            from huggingface_hub import HfApi, hf_hub_download

            info = HfApi().model_info(
                repo_or_path,
                revision=revision,
                token=_token(),
                files_metadata=False,
            )
            resolved_revision = getattr(info, "sha", None) or revision
            files = [s.rfilename for s in (getattr(info, "siblings", None) or [])]
            for name, target in (("config.json", cfg), ("tokenizer_config.json", tokenizer_cfg), ("adapter_config.json", adapter_cfg)):
                try:
                    path = hf_hub_download(
                        repo_id=repo_or_path,
                        filename=name,
                        revision=revision,
                        token=_token(),
                        cache_dir=str(_settings.hf_cache_dir),
                    )
                    target.update(json.loads(Path(path).read_text(encoding="utf-8")))
                except Exception:
                    pass
        except Exception as exc:
            return {
                "model_ref": repo_or_path,
                "reachable": False,
                "error": str(exc),
                "suggestions": ["Check the repository ID, network connection, revision, and HF token for private models."],
            }

    adapter = "adapter_config.json" in files
    scratch = "model.pt" in files and "arch.json" in files
    architectures = cfg.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    context_length = next(
        (cfg.get(k) for k in ("max_position_embeddings", "n_positions", "seq_length", "model_max_length") if cfg.get(k)),
        tokenizer_cfg.get("model_max_length") or scratch_cfg.get("block_size"),
    )
    params = _param_count(repo_or_path, cfg)
    quantization = cfg.get("quantization_config") or {}
    model_type = cfg.get("model_type")
    causal = bool(
        scratch
        or any("CausalLM" in str(name) for name in architectures)
        or cfg.get("is_decoder")
        or (model_type and not cfg.get("is_encoder_decoder", False))
    )
    warnings: list[str] = []
    if not (cfg or adapter or scratch):
        warnings.append("No recognized Transformers, PEFT adapter, or SLM Kit scratch metadata was found.")
    if cfg.get("is_encoder_decoder"):
        warnings.append("Encoder-decoder models are not supported by the current causal-LM training backend.")
    if not any("tokenizer" in name.lower() or name in {"vocab.json", "merges.txt"} for name in files):
        warnings.append("Tokenizer files were not found in this revision; loading may depend on a base model.")

    return {
        "model_ref": repo_or_path,
        "reachable": True,
        "revision": resolved_revision,
        "kind": "scratch" if scratch else "adapter" if adapter else "transformers",
        "adapter_base_model": adapter_cfg.get("base_model_name_or_path"),
        "architecture": architectures,
        "model_type": model_type,
        "parameters": params,
        "context_length": context_length,
        "vocab_size": cfg.get("vocab_size") or scratch_cfg.get("vocab_size"),
        "tokenizer_class": tokenizer_cfg.get("tokenizer_class"),
        "quantization": quantization,
        "supports_causal_lm": causal,
        "supports_lora": causal and not scratch,
        "supports_full_training": causal,
        "supports_4bit": causal and not scratch,
        "suggested_target_modules": ["all-linear"] if causal and not scratch else [],
        "warnings": warnings,
    }


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
    model_card: str | None = None,
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
