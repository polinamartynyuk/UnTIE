"""Source-only answer matching.

Primary relaxed criterion: gold span containment after identity_key.
Exact identity remains a special case (zero extra tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from untie.aspect_qa.normalize import identity_key
from untie.extraction_metrics import token_f1

EXACT_MATCH_METHOD = "normalized_exact_match"
CONTAINMENT_MATCH_METHOD = "gold_span_containment"
SUPPORTED_MATCH_METHODS = (EXACT_MATCH_METHOD, CONTAINMENT_MATCH_METHOD)
DEFAULT_MAX_EXTRA_TOKENS = 5


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    matched_gold: str | None
    quality: float
    match_type: str
    extra_tokens: int | None


def key_tokens(value: str) -> tuple[str, ...]:
    key = identity_key(value)
    if not key:
        return ()
    return tuple(token for token in key.split() if token)


def _unigram_in_pred(gold: str, pred_tokens: Sequence[str]) -> bool:
    for token in pred_tokens:
        if token == gold:
            return True
        if gold and gold in token.split("-"):
            return True
    return False


def gold_tokens_in_pred(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> bool:
    """True iff gold is a contiguous token sequence in pred, or a unigram hyphen-part."""
    if not gold_tokens or not pred_tokens:
        return False
    if len(gold_tokens) == 1:
        return _unigram_in_pred(gold_tokens[0], pred_tokens)
    width = len(gold_tokens)
    if width > len(pred_tokens):
        return False
    gold = tuple(gold_tokens)
    for start in range(len(pred_tokens) - width + 1):
        if tuple(pred_tokens[start : start + width]) == gold:
            return True
    return False


def extra_token_count(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> int:
    return max(0, len(pred_tokens) - len(gold_tokens))


def normalized_exact_match(prediction: str, gold_answers: Sequence[str]) -> str | None:
    predicted_key = identity_key(prediction)
    if not predicted_key:
        return None
    for gold in gold_answers:
        if identity_key(gold) == predicted_key:
            return str(gold)
    return None


def gold_span_containment(
    prediction: str,
    gold_answers: Sequence[str],
    *,
    max_extra_tokens: int = DEFAULT_MAX_EXTRA_TOKENS,
) -> tuple[str | None, int | None]:
    """Return gold if Norm(gold) is a token subsequence of Norm(pred) with a bounded over-extraction.

    Direction is gold ⊆ pred only. Under-extraction (pred ⊆ gold) is not a hit.
    Character substrings such as ``ap`` in ``paper`` do not count.
    """
    exact = normalized_exact_match(prediction, gold_answers)
    if exact is not None:
        return exact, 0

    pred_tokens = key_tokens(prediction)
    if not pred_tokens:
        return None, None

    best_gold: str | None = None
    best_extra: int | None = None
    best_gold_len = -1
    cap = int(max_extra_tokens)
    for gold in gold_answers:
        gold_toks = key_tokens(gold)
        if not gold_tokens_in_pred(pred_tokens, gold_toks):
            continue
        extra = extra_token_count(pred_tokens, gold_toks)
        if extra > cap:
            continue
        gold_len = len(gold_toks)
        if gold_len > best_gold_len or (
            gold_len == best_gold_len and best_extra is not None and extra < best_extra
        ):
            best_gold = str(gold)
            best_extra = extra
            best_gold_len = gold_len
    return best_gold, best_extra


def classify_match(
    prediction: str,
    extractable_gold_answers: Sequence[str],
    *,
    method: str = CONTAINMENT_MATCH_METHOD,
    max_extra_tokens: int = DEFAULT_MAX_EXTRA_TOKENS,
) -> MatchResult:
    gold = [str(item) for item in extractable_gold_answers if str(item).strip()]
    quality = float(token_f1(prediction, gold)) / 100.0 if gold else 0.0
    if method == EXACT_MATCH_METHOD:
        matched = normalized_exact_match(prediction, gold)
        return MatchResult(
            matched=matched is not None,
            matched_gold=matched,
            quality=quality,
            match_type="exact" if matched is not None else "none",
            extra_tokens=0 if matched is not None else None,
        )
    if method != CONTAINMENT_MATCH_METHOD:
        raise ValueError(f"unsupported Phase 1 match method: {method}")
    matched, extra = gold_span_containment(
        prediction, gold, max_extra_tokens=max_extra_tokens
    )
    if matched is None:
        return MatchResult(False, None, quality, "none", None)
    match_type = "exact" if extra == 0 and identity_key(prediction) == identity_key(matched) else "containment"
    # extra==0 can also be hyphen-unigram containment inside a compound token.
    if identity_key(prediction) == identity_key(matched):
        match_type = "exact"
    return MatchResult(True, matched, quality, match_type, extra)


def match_extractable(
    prediction: str,
    extractable_gold_answers: Sequence[str],
    *,
    method: str = CONTAINMENT_MATCH_METHOD,
    max_extra_tokens: int = DEFAULT_MAX_EXTRA_TOKENS,
) -> tuple[bool, str | None, float]:
    result = classify_match(
        prediction,
        extractable_gold_answers,
        method=method,
        max_extra_tokens=max_extra_tokens,
    )
    return result.matched, result.matched_gold, result.quality


def gold_keys(values: Sequence[str]) -> set[str]:
    return {identity_key(item) for item in values if str(item).strip()}
