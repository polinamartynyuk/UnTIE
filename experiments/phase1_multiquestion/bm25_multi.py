"""BM25 over every question variant, plus document-level aggregations."""

from __future__ import annotations

from typing import Sequence

from experiments.phase1_scirex.bm25 import bm25_scores


def min_max_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low <= 1e-12:
        return [0.0 for _ in values]
    return [(float(item) - low) / (high - low) for item in values]


def ranks_descending(values: Sequence[float]) -> list[int]:
    """Rank 1 = highest score. Ties broken by earlier index."""
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def rrf_scores(rank_rows: Sequence[Sequence[int]], *, k: int = 60) -> list[float]:
    if not rank_rows:
        return []
    n_docs = len(rank_rows[0])
    scores = [0.0] * n_docs
    for ranks in rank_rows:
        for index, rank in enumerate(ranks):
            scores[index] += 1.0 / (float(k) + float(rank))
    return scores


def multi_question_bm25(
    questions: Sequence[str],
    chunk_texts: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    rrf_k: int = 60,
) -> dict[str, list[list[float]] | list[float] | list[list[int]]]:
    per_question_scores: list[list[float]] = []
    per_question_norm: list[list[float]] = []
    per_question_ranks: list[list[int]] = []
    for question in questions:
        scores = bm25_scores(question, chunk_texts, k1=k1, b=b)
        per_question_scores.append(scores)
        per_question_norm.append(min_max_normalize(scores))
        per_question_ranks.append(ranks_descending(scores))
    n_chunks = len(chunk_texts)
    n_questions = max(len(questions), 1)
    mean_normalized = [
        sum(row[index] for row in per_question_norm) / n_questions for index in range(n_chunks)
    ]
    mean_rank = [
        sum(row[index] for row in per_question_ranks) / n_questions for index in range(n_chunks)
    ]
    rrf = rrf_scores(per_question_ranks, k=rrf_k)
    return {
        "bm25_by_question": per_question_scores,
        "bm25_norm_by_question": per_question_norm,
        "rank_by_question": per_question_ranks,
        "mean_normalized": mean_normalized,
        "mean_rank": mean_rank,
        "rrf": rrf,
    }


def aggregate_hardness(payload: dict[str, list], *, method: str) -> list[float]:
    """Higher is harder. mean_rank is inverted so that rank 1 is the largest value."""
    if method == "mean_normalized":
        return list(payload["mean_normalized"])
    if method == "rrf":
        return list(payload["rrf"])
    if method == "mean_rank":
        n = len(payload["mean_rank"])
        return [float(n + 1) - float(rank) for rank in payload["mean_rank"]]
    raise ValueError(f"unknown BM25 aggregate: {method}")
