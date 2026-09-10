"""Deterministic answer normalization. Same family as current QA metrics."""

from __future__ import annotations

import re
from typing import Iterable

from experiments.qa_chunk_selection.metrics import char_f1, normalized_answer, token_f1


_HYPHEN = re.compile(r"[\u2010-\u2015\-]+")
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_predicted_answer(text: str) -> str:
    raw = _HYPHEN.sub("-", str(text or ""))
    raw = _PUNCT.sub(" ", raw)
    return normalized_answer(raw)


def answers_match(left: str, right: str) -> bool:
    return bool(normalize_predicted_answer(left)) and normalize_predicted_answer(left) == normalize_predicted_answer(right)


def pair_token_f1(left: str, right: str) -> float:
    return float(token_f1(left or "", [right or ""]))


def pair_char_f1(left: str, right: str) -> float:
    return float(char_f1(left or "", [right or ""]))


def majority_answer(answers: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    first: dict[str, str] = {}
    for raw in answers:
        key = normalize_predicted_answer(raw)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, str(raw or ""))
    if not counts:
        return ""
    best = max(counts.items(), key=lambda item: (item[1], -len(item[0])))[0]
    return first[best]
