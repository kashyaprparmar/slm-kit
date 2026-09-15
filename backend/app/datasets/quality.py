"""Bounded, read-only dataset quality diagnostics with example evidence."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from app.datasets.adapters import DatasetSchemaError, canonicalize
from app.datasets.validate import _iter_records

_HTML = re.compile(r"<[^>]{1,200}>")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


def _normalized(text: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _script(character: str) -> str:
    name = unicodedata.name(character, "")
    for token, label in (("DEVANAGARI", "Devanagari"), ("GUJARATI", "Gujarati"),
                         ("ARABIC", "Arabic"), ("CYRILLIC", "Cyrillic"),
                         ("HANGUL", "Hangul"), ("HIRAGANA", "Hiragana"),
                         ("KATAKANA", "Katakana"), ("CJK", "Han"),
                         ("IDEOGRAPH", "Han"), ("LATIN", "Latin")):
        if token in name:
            return label
    return "Common" if character.isspace() or unicodedata.category(character)[0] in "PN" else "Other"


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(_normalized(text)))


def _simhash(tokens: set[str]) -> int:
    scores = [0] * 64
    for token in tokens:
        value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            scores[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, score in enumerate(scores) if score >= 0)


def _warning(store, code: str, line: int, example: str, *, details=None):
    item = store.setdefault(code, {"code": code, "count": 0, "examples": []})
    item["count"] += 1
    if len(item["examples"]) < 5:
        item["examples"].append({"line": line, "text": example[:500], **(details or {})})


def _comparison_hashes(sources: list[tuple[str, str]], max_rows: int) -> set[str]:
    values = set()
    for path, fmt in sources:
        for _line, row in _iter_records(Path(path), fmt):
            if len(values) >= max_rows:
                return values
            if isinstance(row, dict):
                try:
                    text = canonicalize(row).content_text()
                    values.add(hashlib.sha256(_normalized(text).encode()).hexdigest())
                except DatasetSchemaError:
                    continue
    return values


def profile(path: str, fmt: str, *, max_rows=100_000, near_duplicate_threshold=.9,
            compare_sources: list[tuple[str, str]] | None = None) -> dict:
    warnings = {}
    exact, normalized = set(), set()
    prompt_answers = defaultdict(set)
    simhash_buckets: dict[int, list[tuple[set[str], int, str]]] = defaultdict(list)
    scripts = Counter()
    boilerplate = Counter()
    comparison = _comparison_hashes(compare_sources or [], max_rows)
    rows = valid = 0
    for line, row in _iter_records(Path(path), fmt):
        if rows >= max_rows:
            break
        rows += 1
        if not isinstance(row, dict):
            _warning(warnings, "invalid_row", line, "Row is not an object")
            continue
        try:
            record = canonicalize(row)
        except DatasetSchemaError as exc:
            _warning(warnings, "schema_error", line, str(exc))
            continue
        valid += 1
        text = record.content_text()
        raw_hash = hashlib.sha256(json.dumps(row, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()
        norm = _normalized(text)
        norm_hash = hashlib.sha256(norm.encode()).hexdigest()
        was_exact = raw_hash in exact
        if was_exact:
            _warning(warnings, "exact_duplicate", line, text)
        exact.add(raw_hash)
        if norm_hash in normalized and not was_exact:
            _warning(warnings, "normalized_duplicate", line, text)
        normalized.add(norm_hash)
        if norm_hash in comparison:
            _warning(warnings, "split_leakage", line, text)
        if not norm:
            _warning(warnings, "empty_value", line, text)
        if len(norm) < 20:
            _warning(warnings, "extremely_short", line, text)
        if len(text) > 50_000:
            _warning(warnings, "extremely_long", line, text)
        if "\x00" in text:
            _warning(warnings, "null_or_binary", line, text)
        if any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in text):
            _warning(warnings, "control_characters", line, text)
        if len(_HTML.findall(text)) * 20 > max(1, len(text)):
            _warning(warnings, "markup_density", line, text)
        if (len(text) > 30 and len(text) - len(text.rstrip()) > 10) or "   " in text:
            _warning(warnings, "excessive_whitespace", line, text)
        if "�" in text or any(unicodedata.category(char) in {"Cs", "Cn"} for char in text):
            _warning(warnings, "unicode_anomaly", line, text)
        if _EMAIL.search(text) or _PHONE.search(text):
            _warning(warnings, "possible_pii", line, text)
        scripts.update(_script(character) for character in text)
        for segment in filter(None, (_normalized(value) for value in text.splitlines())):
            if len(segment) >= 30:
                boilerplate[segment] += 1
                if boilerplate[segment] == 3:
                    _warning(warnings, "repeated_boilerplate", line, segment)
        if record.kind == "conversation":
            prompts, answer = record.training_pair()
            prompt_hash = hashlib.sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            prompt_answers[prompt_hash].add(_normalized(answer))
            if len(prompt_answers[prompt_hash]) == 2:
                _warning(warnings, "conflicting_answers", line, text)
        words = _tokens(text)
        if len(words) >= 4:
            signature = _simhash(words)
            bucket = signature >> 48
            for candidate, candidate_line, candidate_text in simhash_buckets[bucket][-100:]:
                similarity = len(words & candidate) / max(1, len(words | candidate))
                if similarity >= near_duplicate_threshold and norm_hash not in comparison:
                    _warning(warnings, "near_duplicate", line, text,
                             details={"similar_to_line": candidate_line, "similarity": round(similarity, 3),
                                      "similar_text": candidate_text[:300]})
                    break
            simhash_buckets[bucket].append((words, line, text))
    return {
        "sampled_rows": rows, "valid_rows": valid, "max_rows": max_rows,
        "truncated_scan": rows >= max_rows,
        "script_distribution": dict(scripts),
        "warnings": sorted(warnings.values(), key=lambda item: (-item["count"], item["code"])),
        "destructive_changes": False,
    }
