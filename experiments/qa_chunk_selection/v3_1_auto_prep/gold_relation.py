"""Deterministic gold/span relation. No semantic encoder / NLI / generative judge."""

from __future__ import annotations

from typing import Sequence

from experiments.qa_chunk_selection.metrics import exact_match, normalized_answer


def _tokens(text: str) -> set[str]:
    return {part for part in normalized_answer(str(text or "")).split() if part}


def local_token_f1(value: str, gold_answers: Sequence[str]) -> float:
    pred = _tokens(value)
    if not pred:
        return 0.0
    best = 0.0
    for gold in gold_answers:
        gold_tok = _tokens(str(gold))
        if not gold_tok:
            continue
        overlap = len(pred & gold_tok)
        if overlap <= 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(gold_tok)
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def local_char_f1(value: str, gold_answers: Sequence[str]) -> float:
    pred = normalized_answer(value)
    if not pred:
        return 0.0
    best = 0.0
    for gold in gold_answers:
        ref = normalized_answer(str(gold))
        if not ref:
            continue
        inter = sum(1 for char in set(pred) if char in ref)
        if inter <= 0:
            continue
        precision = inter / max(len(set(pred)), 1)
        recall = inter / max(len(set(ref)), 1)
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def gold_relation(value: str, gold_answers: Sequence[str], *, f1_threshold: float = 0.50) -> dict[str, object]:
    raw = str(value or "").strip()
    golds = [str(item) for item in gold_answers if str(item).strip()]
    token = local_token_f1(raw, golds)
    char = local_char_f1(raw, golds)
    exact = float(exact_match(raw, golds)) if raw and golds else 0.0
    norm = bool(raw and golds and any(normalized_answer(raw) == normalized_answer(gold) for gold in golds))
    if not raw or not golds:
        relation = "no_gold_relation"
    elif exact >= 1.0:
        relation = "exact_match_gold"
    elif norm:
        relation = "normalized_match_gold"
    elif token >= float(f1_threshold) or char >= float(f1_threshold):
        relation = "overlap_match_gold"
    else:
        relation = "new_value_candidate"
    return {
        "qa_gold_relation": relation,
        "qa_exact_match_gold": exact,
        "qa_token_f1_gold": token,
        "qa_char_overlap_gold": char,
        "qa_normalized_match_gold": float(norm),
        "allowed_for_primary_te": relation in {"exact_match_gold", "normalized_match_gold", "overlap_match_gold"},
        "new_value_candidate": relation == "new_value_candidate",
    }
