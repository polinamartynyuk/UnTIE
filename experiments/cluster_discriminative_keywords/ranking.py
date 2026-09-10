"""Chunk ranking with frozen score_chunks modifiers or a pure discriminative sum."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from experiments.keybert_topic_clustering.routing import RoutingResult
from untie.domain import Sentence, TextChunk
from untie.keyword_evidence import CachedDocumentEvidence
from untie.ranking import WeightedKeyword, score_chunks

from .discrimination import KeywordProfile, keyword_in_text


def profiles_to_keywords(
    profiles: Sequence[KeywordProfile],
    *,
    weight_attr: str = "final_weight",
    keep_negative: bool = True,
) -> list[WeightedKeyword]:
    keywords = []
    for profile in profiles:
        weight = float(getattr(profile, weight_attr))
        if not keep_negative and weight <= 0:
            continue
        keywords.append(
            WeightedKeyword(
                word=profile.keyword,
                lemma=profile.keyword,
                stem=profile.keyword,
                attention_weight=weight,
                score_difference=max(weight, 0.0),
            )
        )
    return keywords


def _text_chunks(evidence: CachedDocumentEvidence) -> list[TextChunk]:
    return [
        TextChunk(
            sentences=(Sentence(text=item.text, number=item.index, tokens=tuple(item.text.split())),),
            text=item.text,
            token_count=item.token_count,
        )
        for item in evidence.chunks
    ]


def score_document_chunks(
    evidence: CachedDocumentEvidence,
    keywords: Sequence[WeightedKeyword],
    *,
    use_modifiers: bool = True,
) -> list[float]:
    chunks = _text_chunks(evidence)
    n = len(chunks)
    if not keywords:
        return [0.0] * n
    if use_modifiers:
        scored = score_chunks(
            chunks, list(keywords), weight_ratio=1.0, keep_unmatched=True, unmatched_floor=0.0
        )
        by_index = {item.chunk.start_sentence: float(item.score) for item in scored}
        return [float(by_index.get(chunk.start_sentence, 0.0)) for chunk in chunks]
    scores = []
    for chunk in chunks:
        total = 0.0
        for keyword in keywords:
            if keyword_in_text(keyword.word, chunk.text):
                total += float(keyword.attention_weight)
        scores.append(total)
    return scores


def mix_cluster_scores(
    per_cluster: Mapping[int, Sequence[float]],
    route: RoutingResult,
    *,
    soft: bool,
) -> list[float]:
    if not per_cluster:
        return []
    length = len(next(iter(per_cluster.values())))
    if not soft:
        chosen = per_cluster.get(int(route.top1))
        return list(chosen) if chosen is not None else [0.0] * length
    mixed = np.zeros(length, dtype=float)
    weight_map = dict(zip(route.top_ids, route.weights))
    mass = 0.0
    for cluster_id, scores in per_cluster.items():
        weight = float(weight_map.get(int(cluster_id), 0.0))
        if weight <= 0:
            continue
        mixed += weight * np.asarray(scores, dtype=float)
        mass += weight
    if mass <= 0 and int(route.top1) in per_cluster:
        return list(per_cluster[int(route.top1)])
    return [float(value) for value in mixed]


def gold_chunk_index(evidence: CachedDocumentEvidence) -> int | None:
    from untie.extraction_metrics import char_f1

    if not evidence.references or not evidence.chunks:
        return None
    scores = [char_f1(item.answer, list(evidence.references)) for item in evidence.chunks]
    if max(scores) <= 0:
        return None
    return int(np.argmax(scores))


def rank_metrics(scores: Sequence[float], gold_index: int | None) -> dict[str, float]:
    n = len(scores)
    if gold_index is None or n == 0 or gold_index < 0 or gold_index >= n:
        return {
            "evidence_recall@1": 0.0,
            "evidence_recall@3": 0.0,
            "evidence_recall@5": 0.0,
            "evidence_recall@10": 0.0,
            "mrr": 0.0,
            "mean_rank": float(n + 1 if n else 0),
            "gold_rank": -1.0,
            "has_gold_chunk": 0.0,
        }
    order = sorted(range(n), key=lambda index: (-float(scores[index]), index))
    rank = order.index(int(gold_index)) + 1
    return {
        "evidence_recall@1": float(rank == 1),
        "evidence_recall@3": float(rank <= 3),
        "evidence_recall@5": float(rank <= 5),
        "evidence_recall@10": float(rank <= 10),
        "mrr": float(1.0 / rank),
        "mean_rank": float(rank),
        "gold_rank": float(rank),
        "has_gold_chunk": 1.0,
    }


def summarize_ranks(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    usable = [row for row in rows if float(row.get("has_gold_chunk", 0.0)) > 0]
    if not usable:
        return {
            "n": 0,
            "n_with_gold": 0,
            "evidence_recall@1": 0.0,
            "evidence_recall@3": 0.0,
            "evidence_recall@5": 0.0,
            "evidence_recall@10": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
        }
    ranks = [float(row["gold_rank"]) for row in usable]
    n_all = len(rows)
    return {
        "n": n_all,
        "n_with_gold": len(usable),
        "evidence_recall@1": float(np.mean([row["evidence_recall@1"] for row in usable])),
        "evidence_recall@3": float(np.mean([row["evidence_recall@3"] for row in usable])),
        "evidence_recall@5": float(np.mean([row["evidence_recall@5"] for row in usable])),
        "evidence_recall@10": float(np.mean([row["evidence_recall@10"] for row in usable])),
        "mrr": float(np.mean([row["mrr"] for row in usable])),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }
