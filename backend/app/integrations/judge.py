"""Optional LLM-as-judge scoring via an external API (Anthropic or OpenAI).

Fully key-gated: with no key configured, ``judge_available`` is False and the app
works normally without it. Used by the eval harness to score free-form answers
that automatic metrics judge poorly.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from app.config import get_settings

_settings = get_settings()

_PROMPT = """You are grading an AI model's answer.

Question:
{question}

{reference_block}Model answer:
{answer}

Rate the model answer's quality from 1 to 10 (10 = fully correct, complete, and \
relevant). Respond with ONLY a JSON object: {{"score": <int 1-10>, "reason": "<short>"}}."""


def judge_available() -> bool:
    return bool(_settings.judge_api_key)


def judge_answer(question: str, answer: str, reference: Optional[str] = None) -> Optional[dict]:
    """Return {"score": 0..1, "raw": 1..10, "reason": str} or None if unavailable."""
    if not judge_available():
        return None
    ref_block = f"Reference answer:\n{reference}\n\n" if reference else ""
    prompt = _PROMPT.format(question=question, answer=answer, reference_block=ref_block)
    try:
        if _settings.judge_provider == "openai":
            text = _call_openai(prompt)
        else:
            text = _call_anthropic(prompt)
    except Exception:
        return None
    return _parse(text)


def _parse(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        raw = int(obj["score"])
    except Exception:
        return None
    return {"score": round(max(1, min(10, raw)) / 10.0, 3), "raw": raw, "reason": obj.get("reason", "")}


def _call_anthropic(prompt: str) -> str:
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": _settings.judge_api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": _settings.judge_model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_openai(prompt: str) -> str:
    import httpx

    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {_settings.judge_api_key}", "content-type": "application/json"},
        json={
            "model": _settings.judge_model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
