"""Isolated tokenizer profiler. Emits one JSON result and imports no model weights."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low, high = math.floor(index), math.ceil(index)
    return float(ordered[low] if low == high else ordered[low] * (high - index) + ordered[high] * (index - low))


def _script(character: str) -> str:
    if character.isspace() or character.isdigit() or unicodedata.category(character).startswith("P"):
        return "Common"
    name = unicodedata.name(character, "")
    for token, label in (
        ("DEVANAGARI", "Devanagari"), ("GUJARATI", "Gujarati"),
        ("ARABIC", "Arabic"), ("CYRILLIC", "Cyrillic"),
        ("HANGUL", "Hangul"), ("HIRAGANA", "Hiragana"),
        ("KATAKANA", "Katakana"), ("CJK", "Han"),
        ("IDEOGRAPH", "Han"), ("LATIN", "Latin"),
    ):
        if token in name:
            return label
    return "Other"


def _fingerprint_tokenizer(tokenizer, model_ref: str, revision: str | None) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model_ref}\0{revision or ''}\0".encode())
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: (item[1], item[0])):
        digest.update(str(index).encode() + b":" + token.encode("utf-8", errors="surrogatepass") + b"\0")
    payload = {"special_tokens": tokenizer.special_tokens_map,
               "chat_template": getattr(tokenizer, "chat_template", None),
               "class": tokenizer.__class__.__name__}
    digest.update(json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode())
    return digest.hexdigest()


def profile(config: dict) -> dict:
    from transformers import AutoTokenizer

    from app.datasets.validate import _iter_records
    from app.integrations.hf_hub import _token
    from app.train_entry.tokenization import TokenizerDataError, render_and_tokenize

    common = {"token": _token(), "cache_dir": config["cache_dir"],
              "trust_remote_code": bool(config.get("trust_remote_code"))}
    if config.get("revision"):
        common["revision"] = config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["model_ref"], **common)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths: list[int] = []
    unknown, total, chars, words, truncated = 0, 0, 0, 0, 0
    scripts = Counter()
    by_script = defaultdict(lambda: {"samples": 0, "tokens": 0, "characters": 0, "unknown_tokens": 0, "truncated": 0})
    previews, errors = [], []
    path = Path(config["dataset_path"])
    for line, row in _iter_records(path, config["dataset_format"]):
        if len(lengths) >= config["sample_limit"]:
            break
        if not isinstance(row, dict):
            errors.append({"line": line, "message": "Row is not an object."})
            continue
        try:
            item = render_and_tokenize(
                row, tokenizer, max_length=config["max_length"],
                loss_policy=config["loss_policy"], chat_template=config.get("chat_template"),
            )
        except TokenizerDataError as exc:
            errors.append({"line": line, "message": str(exc)})
            if len(errors) >= 20:
                break
            continue
        length = item["original_length"]
        lengths.append(length)
        total += length
        truncated += int(item["truncated"])
        text = item["rendered"]
        chars += len(text)
        words += len(text.split())
        sample_scripts = Counter(_script(character) for character in text)
        scripts.update(sample_scripts)
        dominant = next((name for name, _ in sample_scripts.most_common() if name != "Common"), "Common")
        unk_id = tokenizer.unk_token_id
        sample_unknown = sum(token == unk_id for token in item["input_ids"]) if unk_id is not None else 0
        unknown += sample_unknown
        sliced = by_script[dominant]
        sliced["samples"] += 1
        sliced["tokens"] += length
        sliced["characters"] += len(text)
        sliced["unknown_tokens"] += sample_unknown
        sliced["truncated"] += int(item["truncated"])
        if len(previews) < 5:
            previews.append({
                "line": line, "source_format": item["source_format"],
                "rendered": text, "tokens": tokenizer.convert_ids_to_tokens(item["input_ids"][:256]),
                "input_ids": item["input_ids"][:256], "loss_mask": item["loss_mask"][:256],
                "original_length": length, "truncated": item["truncated"],
            })
    if not lengths:
        raise ValueError("No rows could be profiled. " + (errors[0]["message"] if errors else "The dataset is empty."))
    maximum = config["max_length"]
    packed_capacity = math.ceil(total / maximum) * maximum if total else 0
    for value in by_script.values():
        value["tokens_per_character"] = round(value["tokens"] / max(1, value["characters"]), 4)
        value["unknown_token_rate"] = round(value["unknown_tokens"] / max(1, value["tokens"]), 6)
        value["truncation_rate"] = round(value["truncated"] / max(1, value["samples"]), 4)
    resolved = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or config.get("revision")
    fingerprint = _fingerprint_tokenizer(tokenizer, config["model_ref"], resolved)
    return {
        "tokenizer": {"model_ref": config["model_ref"], "revision": config.get("revision"),
                      "resolved_revision": resolved, "fingerprint": fingerprint,
                      "class": tokenizer.__class__.__name__, "vocab_size": len(tokenizer),
                      "model_max_length": tokenizer.model_max_length if tokenizer.model_max_length < 10**9 else None,
                      "chat_template": bool(getattr(tokenizer, "chat_template", None)),
                      "special_tokens": tokenizer.special_tokens_map},
        "stats": {"sampled_rows": len(lengths), "total_tokens": total, "min": min(lengths),
                  "max": max(lengths), "mean": round(statistics.mean(lengths), 2),
                  "median": round(statistics.median(lengths), 2),
                  "p90": round(_percentile(lengths, .90), 2), "p95": round(_percentile(lengths, .95), 2),
                  "p99": round(_percentile(lengths, .99), 2), "histogram": lengths[:500],
                  "samples_exceeding_max_length": sum(value > maximum for value in lengths),
                  "truncation_percentage": round(100 * truncated / len(lengths), 2),
                  "tokens_per_example": round(total / len(lengths), 2),
                  "tokens_per_character": round(total / max(1, chars), 4),
                  "tokens_per_word": round(total / max(1, words), 4),
                  "unknown_token_rate": round(unknown / max(1, total), 6),
                  "padding_overhead_percentage": round(100 * (maximum * len(lengths) - sum(min(x, maximum) for x in lengths)) / max(1, maximum * len(lengths)), 2),
                  "packing_efficiency_percentage": round(100 * total / max(1, packed_capacity), 2),
                  "script_characters": dict(scripts), "by_dominant_script": dict(by_script)},
        "previews": previews, "errors": errors,
    }


def main(path: str) -> int:
    try:
        print(json.dumps({"ok": True, "result": profile(json.loads(Path(path).read_text(encoding="utf-8")))}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
