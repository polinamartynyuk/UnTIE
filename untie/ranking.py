from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .domain import ScoredChunk, TextChunk


@dataclass(frozen=True)
class WeightedKeyword:
    word: str
    lemma: str
    stem: str
    attention_weight: float = 1.0
    score_difference: float = 1.0


def filter_chunks(
    chunks: list[TextChunk],
    keywords: list[WeightedKeyword],
    *,
    minimum_matches: int = 1,
) -> list[TextChunk]:
    terms = {term.lower() for keyword in keywords for term in (keyword.lemma, keyword.stem)}
    if not terms:
        return []
    pattern = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, sorted(terms)))))
    return [
        chunk
        for chunk in chunks
        if len(set(pattern.findall(chunk.text.lower()))) >= minimum_matches
    ]


def score_chunks(
    chunks: list[TextChunk],
    keywords: list[WeightedKeyword],
    *,
    weight_ratio: float = 0.5,
    minimum_matches: int = 1,
    position_weight: float = 0.3,
    frequency_weight: float = 0.7,
    keep_unmatched: bool = False,
    unmatched_floor: float = 0.0,
) -> list[ScoredChunk]:
    if not chunks:
        return []
    if not keywords:
        if keep_unmatched:
            return [
                ScoredChunk(chunk=chunk, score=unmatched_floor)
                for chunk in chunks
            ]
        return []
    if not 0 <= weight_ratio <= 1:
        raise ValueError("weight_ratio must be between 0 and 1")
    if unmatched_floor < 0:
        raise ValueError("unmatched_floor cannot be negative")

    term_to_keyword: dict[str, WeightedKeyword] = {}
    for keyword in keywords:
        term_to_keyword[keyword.lemma.lower()] = keyword
        term_to_keyword[keyword.stem.lower()] = keyword
    pattern = re.compile(
        r"\b(?:{})\b".format("|".join(map(re.escape, sorted(term_to_keyword))))
    )
    present_keywords = {
        term_to_keyword[match.group()]
        for chunk in chunks
        for match in pattern.finditer(chunk.text.lower())
    }

    scored: list[ScoredChunk] = []
    for chunk in chunks:
        matches = list(pattern.finditer(chunk.text.lower()))
        matched = {term_to_keyword[match.group()] for match in matches}
        if len(matched) < minimum_matches:
            if not keep_unmatched:
                continue
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=unmatched_floor,
                    matched_keywords=(),
                    keyword_scores={},
                    original_weights={},
                )
            )
            continue
        base = sum(
            keyword.attention_weight * weight_ratio
            + keyword.score_difference * (1 - weight_ratio)
            for keyword in matched
        )
        average_position = sum(1 - match.start() / max(1, len(chunk.text)) for match in matches) / len(matches)
        position_bonus = 1 + position_weight * average_position
        frequency_bonus = 1 + frequency_weight * math.log1p(len(matches))
        # Only keywords occurring somewhere in the document belong in the
        # normalization pool. An unrelated, absent keyword must not dilute
        # otherwise identical chunk scores.
        uniqueness_bonus = 1 + len(matched) / len(present_keywords) * 0.5
        score = base * position_bonus * frequency_bonus * uniqueness_bonus
        scored.append(
            ScoredChunk(
                chunk=chunk,
                score=score,
                matched_keywords=tuple(sorted(keyword.word for keyword in matched)),
                keyword_scores={
                    keyword.word: sum(
                        1 for match in matches if term_to_keyword[match.group()] == keyword
                    )
                    for keyword in matched
                },
                original_weights={
                    keyword.word: {
                        "weight": keyword.attention_weight,
                        "score_diff": keyword.score_difference,
                    }
                    for keyword in matched
                },
            )
        )
    return sorted(scored, key=lambda item: item.score, reverse=True)


def normalize_nonnegative_scores(values: Sequence[float]) -> list[float]:
    """Min-max scale to [0, 1]. All-zero or empty input stays zeros."""
    if not values:
        return []
    numbers = [max(0.0, float(value)) for value in values]
    peak = max(numbers)
    if peak <= 0.0:
        return [0.0] * len(numbers)
    floor = min(numbers)
    span = peak - floor
    if span == 0.0:
        return [1.0] * len(numbers)
    return [(value - floor) / span for value in numbers]


def minmax_scale(values: Sequence[float]) -> list[float]:
    """Min-max scale to [0, 1] without clipping negatives (cross-encoder logits)."""
    if not values:
        return []
    numbers = [float(value) for value in values]
    peak = max(numbers)
    floor = min(numbers)
    span = peak - floor
    if span == 0.0:
        return [0.0] * len(numbers)
    return [(value - floor) / span for value in numbers]


def mix_score_channels(
    channels: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[float]:
    """Weighted sum of independently min-max scaled channels."""
    if len(channels) != len(weights):
        raise ValueError("channels and weights must have the same length")
    if not channels:
        return []
    length = len(channels[0])
    if any(len(channel) != length for channel in channels):
        raise ValueError("each score channel must have the same length")
    mixed = [0.0] * length
    for channel, weight in zip(channels, weights):
        scale = float(weight)
        for index, value in enumerate(minmax_scale(channel)):
            mixed[index] += scale * value
    return mixed


def chunk_query_similarities(
    encoder: object,
    query: str,
    texts: Sequence[str],
) -> list[float]:
    """Cosine similarity of each text to ``query``; negatives are clipped to 0."""
    if not texts:
        return []
    if not str(query).strip():
        return [0.0] * len(texts)
    encode = getattr(encoder, "encode")
    matrix = np.asarray(encode([query, *texts]), dtype=float)
    query_vec = matrix[0]
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0.0:
        return [0.0] * len(texts)
    scores = []
    for vector in matrix[1:]:
        other_norm = float(np.linalg.norm(vector))
        if other_norm == 0.0:
            scores.append(0.0)
            continue
        cosine = float(np.dot(query_vec, vector) / (query_norm * other_norm))
        scores.append(max(0.0, cosine))
    return scores


def combine_scored_chunk_channels(
    scored: Sequence[ScoredChunk],
    channels: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[ScoredChunk]:
    """Replace chunk scores with a weighted sum of min-max normalized channels."""
    if len(channels) != len(weights):
        raise ValueError("channels and weights must have the same length")
    if any(len(channel) != len(scored) for channel in channels):
        raise ValueError("each score channel must align with scored chunks")
    combined = [0.0] * len(scored)
    diagnostics: list[list[float]] = [[] for _ in scored]
    for channel, weight in zip(channels, weights):
        scale = float(weight)
        normalized = normalize_nonnegative_scores(channel)
        for index, value in enumerate(normalized):
            combined[index] += scale * value
            diagnostics[index].append(value)
    result = []
    for item, score, parts in zip(scored, combined, diagnostics):
        keyword_scores = dict(item.keyword_scores)
        keyword_scores["soft_combined"] = float(score)
        for index, value in enumerate(parts):
            keyword_scores[f"soft_channel_{index}"] = float(value)
        result.append(replace(item, score=float(score), keyword_scores=keyword_scores))
    return sorted(result, key=lambda item: item.score, reverse=True)
