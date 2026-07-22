from __future__ import annotations

import math
import re
from dataclasses import dataclass

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
) -> list[ScoredChunk]:
    if not chunks or not keywords:
        return []
    if not 0 <= weight_ratio <= 1:
        raise ValueError("weight_ratio must be between 0 and 1")

    term_to_keyword: dict[str, WeightedKeyword] = {}
    for keyword in keywords:
        term_to_keyword[keyword.lemma.lower()] = keyword
        term_to_keyword[keyword.stem.lower()] = keyword
    pattern = re.compile(
        r"\b(?:{})\b".format("|".join(map(re.escape, sorted(term_to_keyword))))
    )

    scored: list[ScoredChunk] = []
    for chunk in chunks:
        matches = list(pattern.finditer(chunk.text.lower()))
        matched = {term_to_keyword[match.group()] for match in matches}
        if len(matched) < minimum_matches:
            continue
        base = sum(
            keyword.attention_weight * weight_ratio
            + keyword.score_difference * (1 - weight_ratio)
            for keyword in matched
        )
        average_position = sum(1 - match.start() / max(1, len(chunk.text)) for match in matches) / len(matches)
        position_bonus = 1 + position_weight * average_position
        frequency_bonus = 1 + frequency_weight * math.log1p(len(matches))
        uniqueness_bonus = 1 + len(matched) / len(keywords) * 0.5
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
