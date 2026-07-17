"""Automatic evaluation metrics.

Pure-Python where possible (exact match), with graceful optional dependencies for
ROUGE/BLEU (the ``[eval]`` extra). Perplexity needs the model itself and is
computed inside the eval subprocess, not here.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Optional


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def exact_match(preds: list[str], refs: list[str]) -> float:
    if not preds:
        return 0.0
    return mean(1.0 if _norm(p) == _norm(r) else 0.0 for p, r in zip(preds, refs))


def token_f1(preds: list[str], refs: list[str]) -> float:
    """Average token-overlap F1 — a softer signal than exact match."""
    def f1(p: str, r: str) -> float:
        pt, rt = _norm(p).split(), _norm(r).split()
        if not pt or not rt:
            return 0.0
        common = 0
        rt_pool = list(rt)
        for t in pt:
            if t in rt_pool:
                rt_pool.remove(t)
                common += 1
        if common == 0:
            return 0.0
        prec, rec = common / len(pt), common / len(rt)
        return 2 * prec * rec / (prec + rec)

    if not preds:
        return 0.0
    return mean(f1(p, r) for p, r in zip(preds, refs))


def rouge_l(preds: list[str], refs: list[str]) -> Optional[float]:
    try:
        from rouge_score import rouge_scorer
    except Exception:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return mean(scores) if scores else 0.0


def bleu(preds: list[str], refs: list[str]) -> Optional[float]:
    try:
        import sacrebleu
    except Exception:
        return None
    if not preds:
        return 0.0
    return sacrebleu.corpus_bleu(preds, [refs]).score / 100.0


def compute(preds: list[str], refs: list[str], wanted: list[str]) -> dict[str, float]:
    """Compute the requested metrics that are available; skip unavailable ones."""
    out: dict[str, float] = {}
    if "exact_match" in wanted:
        out["exact_match"] = round(exact_match(preds, refs), 4)
    if "token_f1" in wanted:
        out["token_f1"] = round(token_f1(preds, refs), 4)
    if "rouge_l" in wanted:
        r = rouge_l(preds, refs)
        if r is not None:
            out["rouge_l"] = round(r, 4)
    if "bleu" in wanted:
        b = bleu(preds, refs)
        if b is not None:
            out["bleu"] = round(b, 4)
    return out
