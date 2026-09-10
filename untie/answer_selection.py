"""Question-conditioned answer selection helpers.

These replace a field-specific 'task-like' lexicon. Eligibility and
near-duplicate merging must work for any QA question (task, dataset,
method, …) using only the question text plus the candidate spans.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .domain import Answer, FinalAnswer
from .protocols import SentenceEncoder


_EN_QUESTION_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "been",
        "being",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "please",
        "solved",
        "the",
        "this",
        "that",
        "to",
        "used",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    }
)
_RU_QUESTION_STOP = frozenset(
    {
        "в",
        "во",
        "был",
        "была",
        "было",
        "были",
        "для",
        "как",
        "какая",
        "какие",
        "какой",
        "кем",
        "кто",
        "на",
        "о",
        "об",
        "по",
        "решена",
        "решён",
        "решен",
        "с",
        "что",
        "чем",
        "эта",
        "это",
        "этот",
        "используется",
        "использован",
        "использована",
    }
)


def normalize_answer_key(text: str) -> str:
    """Canonical key so hyphen, spacing, and case variants collapse."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u00ad", "")
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    value = value.casefold()
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def question_focus(question: str, *, language: str = "en") -> str:
    """Content words of the question: the aspect cue without a field lexicon."""
    stops = _RU_QUESTION_STOP if str(language).casefold().startswith("ru") else _EN_QUESTION_STOP
    tokens = [
        token
        for token in re.findall(r"\w+", str(question or "").casefold(), flags=re.UNICODE)
        if token not in stops and len(token) > 1
    ]
    return " ".join(tokens)


def query_prototypes(question: str, *, language: str = "en") -> tuple[str, ...]:
    """Texts that define 'what kind of answer we want' from the QA question alone."""
    cleaned = re.sub(r"\s+", " ", str(question or "")).strip()
    if not cleaned:
        return ()
    focus = question_focus(cleaned, language=language)
    if focus and focus.casefold() != cleaned.casefold():
        return (cleaned, focus)
    return (cleaned,)


@dataclass(frozen=True)
class AggregationSettings:
    """All knobs. Defaults reproduce the historical centroid-of-all-spans vote."""

    merge_near_duplicates: bool = False
    query_quantile: float | None = None
    min_keep: int = 3
    language: str = "en"

    def uses_query_filter(self) -> bool:
        return self.query_quantile is not None

    def is_legacy(self) -> bool:
        return (not self.merge_near_duplicates) and self.query_quantile is None


@dataclass(frozen=True)
class AggregationStats:
    n_input: int
    n_after_merge: int
    n_after_filter: int
    query_threshold: float | None
    fallback: str
    selected_query_similarity: float | None
    prototypes: tuple[str, ...]


def _encode(encoder: SentenceEncoder, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=float)
    return np.asarray(encoder.encode(list(texts)))


def merge_near_duplicate_answers(answers: Sequence[Answer]) -> list[list[Answer]]:
    """Group answers that match after hyphen/case/spacing normalization."""
    groups: dict[str, list[Answer]] = {}
    order: list[str] = []
    for answer in answers:
        key = normalize_answer_key(answer.text) or f"__empty_{id(answer)}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(answer)
    return [groups[key] for key in order]


def _representative(group: Sequence[Answer]) -> Answer:
    counts: dict[str, int] = {}
    for item in group:
        counts[item.text] = counts.get(item.text, 0) + 1
    best_text = max(counts, key=lambda text: (counts[text], len(text)))
    return next(item for item in group if item.text == best_text)


def _query_similarities(
    answer_embeddings: np.ndarray,
    prototype_embeddings: np.ndarray,
) -> np.ndarray:
    if answer_embeddings.size == 0:
        return np.zeros((0,), dtype=float)
    if prototype_embeddings.size == 0:
        return np.zeros((answer_embeddings.shape[0],), dtype=float)
    return cosine_similarity(answer_embeddings, prototype_embeddings).max(axis=1)


def _keep_by_query_quantile(
    similarities: np.ndarray,
    *,
    quantile: float,
    min_keep: int,
) -> np.ndarray:
    n = int(similarities.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    keep_at_least = min(n, max(1, int(min_keep)))
    q = min(max(float(quantile), 0.0), 0.99)
    threshold = float(np.quantile(similarities, q))
    mask = similarities >= threshold
    if int(mask.sum()) >= keep_at_least:
        return mask
    order = np.argsort(-similarities)
    mask = np.zeros(n, dtype=bool)
    mask[order[:keep_at_least]] = True
    return mask


def select_typical_span(
    answers: Sequence[Answer],
    encoder: SentenceEncoder,
    *,
    question: str | None = None,
    settings: AggregationSettings | None = None,
) -> tuple[Answer | None, tuple[Answer, ...], AggregationStats]:
    """Centroid vote, optionally after duplicate merge and question-similarity filter."""
    settings = settings or AggregationSettings()
    items = list(answers)
    stats = AggregationStats(
        n_input=len(items),
        n_after_merge=len(items),
        n_after_filter=len(items),
        query_threshold=None,
        fallback="none",
        selected_query_similarity=None,
        prototypes=(),
    )
    if not items:
        return None, (), stats

    if settings.merge_near_duplicates:
        groups = merge_near_duplicate_answers(items)
        representatives = [_representative(group) for group in groups]
        texts = [item.text for item in representatives]
        embeddings = np.vstack(
            [
                _encode(encoder, [member.text for member in group]).mean(axis=0, keepdims=True)
                for group in groups
            ]
        )
        members = groups
    else:
        representatives = items
        texts = [item.text for item in items]
        embeddings = _encode(encoder, texts)
        members = [[item] for item in items]
    stats = AggregationStats(
        n_input=stats.n_input,
        n_after_merge=len(representatives),
        n_after_filter=len(representatives),
        query_threshold=None,
        fallback="none",
        selected_query_similarity=None,
        prototypes=(),
    )

    query_sims: np.ndarray | None = None
    prototypes = query_prototypes(question or "", language=settings.language) if question else ()
    if settings.uses_query_filter() and prototypes:
        proto_embeddings = _encode(encoder, list(prototypes))
        query_sims = _query_similarities(embeddings, proto_embeddings)
        mask = _keep_by_query_quantile(
            query_sims,
            quantile=float(settings.query_quantile),
            min_keep=settings.min_keep,
        )
        if not bool(mask.any()):
            stats = AggregationStats(
                n_input=stats.n_input,
                n_after_merge=stats.n_after_merge,
                n_after_filter=stats.n_after_merge,
                query_threshold=None,
                fallback="empty_filter_kept_all",
                selected_query_similarity=None,
                prototypes=prototypes,
            )
        else:
            embeddings = embeddings[mask]
            representatives = [item for item, kept in zip(representatives, mask) if kept]
            members = [group for group, kept in zip(members, mask) if kept]
            query_sims = query_sims[mask]
            tau = float(np.min(query_sims)) if len(query_sims) else None
            stats = AggregationStats(
                n_input=stats.n_input,
                n_after_merge=stats.n_after_merge,
                n_after_filter=len(representatives),
                query_threshold=tau,
                fallback="none",
                selected_query_similarity=None,
                prototypes=prototypes,
            )
    elif settings.uses_query_filter() and not prototypes:
        stats = AggregationStats(
            n_input=stats.n_input,
            n_after_merge=stats.n_after_merge,
            n_after_filter=stats.n_after_merge,
            query_threshold=None,
            fallback="missing_question",
            selected_query_similarity=None,
            prototypes=(),
        )

    similarities = cosine_similarity(embeddings)
    best_index = int(np.argmax(similarities.mean(axis=1)))
    best = representatives[best_index]
    if settings.merge_near_duplicates:
        supporting = tuple(members[best_index])
    else:
        # Historical Q0: all exact-text copies of the centroid winner.
        supporting = tuple(
            item for group in members for item in group if item.text == best.text
        )
    selected_q = None if query_sims is None else float(query_sims[best_index])
    stats = AggregationStats(
        n_input=stats.n_input,
        n_after_merge=stats.n_after_merge,
        n_after_filter=stats.n_after_filter,
        query_threshold=stats.query_threshold,
        fallback=stats.fallback,
        selected_query_similarity=selected_q,
        prototypes=stats.prototypes or prototypes,
    )
    return best, supporting, stats


def select_max_query_similarity(
    answers: Sequence[Answer],
    encoder: SentenceEncoder,
    *,
    question: str,
    settings: AggregationSettings | None = None,
) -> tuple[Answer | None, tuple[Answer, ...], AggregationStats]:
    """Keep the span most similar to the question (after optional duplicate merge)."""
    settings = settings or AggregationSettings(merge_near_duplicates=True)
    items = list(answers)
    if not items:
        empty = AggregationStats(0, 0, 0, None, "empty", None, ())
        return None, (), empty
    if settings.merge_near_duplicates:
        groups = merge_near_duplicate_answers(items)
        representatives = [_representative(group) for group in groups]
        members = groups
        embeddings = np.vstack(
            [
                _encode(encoder, [member.text for member in group]).mean(axis=0, keepdims=True)
                for group in groups
            ]
        )
    else:
        representatives = items
        members = [[item] for item in items]
        embeddings = _encode(encoder, [item.text for item in items])
    prototypes = query_prototypes(question, language=settings.language)
    if not prototypes:
        selected, supporting, stats = select_typical_span(
            answers, encoder, question=question, settings=settings
        )
        return selected, supporting, stats
    sims = _query_similarities(embeddings, _encode(encoder, list(prototypes)))
    best_index = int(np.argmax(sims))
    best = representatives[best_index]
    stats = AggregationStats(
        n_input=len(items),
        n_after_merge=len(representatives),
        n_after_filter=1,
        query_threshold=float(sims[best_index]),
        fallback="query_argmax",
        selected_query_similarity=float(sims[best_index]),
        prototypes=prototypes,
    )
    return best, tuple(members[best_index]), stats
