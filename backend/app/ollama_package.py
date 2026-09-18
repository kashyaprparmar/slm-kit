"""Lightweight, deterministic Ollama package policy rendering.

This deliberately reads tokenizer/generation metadata only; tokenizer rendering
remains owned by train_entry.tokenization at training time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def source_policy(source: Path, lineage: dict[str, Any]) -> dict[str, Any]:
    """Recover safe defaults from a local HF source or recorded lineage."""
    meta = lineage.get("source_meta") or {}
    policy = dict(meta.get("ollama_policy") or {})
    root = source if source.is_dir() else source.parent
    for name, key in (
        ("tokenizer_config.json", "tokenizer"),
        ("generation_config.json", "generation"),
    ):
        path = root / name
        if path.is_file():
            try:
                policy[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
    tokenizer = policy.get("tokenizer") or {}
    generation = policy.get("generation") or {}
    return {
        "chat_template": tokenizer.get("chat_template"),
        "stop_tokens": [tokenizer["eos_token"]]
        if isinstance(tokenizer.get("eos_token"), str)
        else [],
        "generation_defaults": {
            key: generation[key]
            for key in ("temperature", "top_p", "top_k", "repeat_penalty", "num_predict")
            if generation.get(key) is not None
        },
    }


def render_modelfile(gguf_filename: str, policy: dict[str, Any]) -> str:
    """Render an injection-safe relative-path Modelfile for an exported GGUF."""
    if not gguf_filename or Path(gguf_filename).name != gguf_filename:
        raise ValueError("Ollama package requires one GGUF filename in its package directory.")
    template = policy.get("chat_template")
    if template is not None and '"""' in template:
        raise ValueError(
            "The chat template contains an unsupported triple-quote sequence for an Ollama Modelfile."
        )
    lines = [f'FROM "./{gguf_filename}"']
    if template:
        lines.extend(['TEMPLATE """', template, '"""'])
    for token in policy.get("stop_tokens") or []:
        if not isinstance(token, str) or "\n" in token or '"' in token:
            raise ValueError("Ollama stop tokens must be single-line strings without quotes.")
        lines.append(f'PARAMETER stop "{token}"')
    for name, value in (policy.get("generation_defaults") or {}).items():
        if name not in {
            "temperature",
            "top_p",
            "top_k",
            "repeat_penalty",
            "num_predict",
        } or not isinstance(value, (int, float)):
            raise ValueError("Unsupported Ollama generation default.")
        lines.append(f"PARAMETER {name} {value}")
    return "\n".join(lines) + "\n"
