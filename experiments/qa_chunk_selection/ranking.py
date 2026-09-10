"""Ranking signals: dense cosine, BM25, generic CE, RRF, lexical overlap (smoke)."""

from __future__ import annotations

import math
import re
from typing import Callable, Protocol, Sequence

import numpy as np

from experiments.phase1_scirex.bm25 import bm25_scores

from .schema import ChunkRecord, InstanceRecord

ScoreFn = Callable[[str, Sequence[ChunkRecord]], list[float]]


class PairwiseScorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        ...


_TOKEN = re.compile(r"\w+", re.UNICODE)


def rank_ids(chunks: Sequence[ChunkRecord], scores: Sequence[float]) -> list[int]:
    if len(chunks) != len(scores):
        raise ValueError("scores must align with chunks")
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index].chunk_id),
    )
    return [int(chunks[index].chunk_id) for index in order]


def score_map(chunks: Sequence[ChunkRecord], scores: Sequence[float]) -> dict[int, float]:
    return {int(chunk.chunk_id): float(score) for chunk, score in zip(chunks, scores)}


def min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(high, low):
        return [0.0 for _ in values]
    return [(float(item) - low) / (high - low) for item in values]


def cosine_scores(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> list[float]:
    if chunk_vecs.size == 0:
        return []
    left = np.asarray(query_vec, dtype=float).reshape(-1)
    right = np.asarray(chunk_vecs, dtype=float)
    if right.ndim == 1:
        right = right.reshape(1, -1)
    denom = np.linalg.norm(left) * np.linalg.norm(right, axis=1)
    dots = right @ left
    scores = np.divide(dots, np.maximum(denom, 1e-12))
    return [float(item) for item in scores]


def lexical_overlap_scores(query: str, chunks: Sequence[ChunkRecord]) -> list[float]:
    query_tokens = set(_TOKEN.findall(query.casefold()))
    scores: list[float] = []
    for chunk in chunks:
        tokens = _TOKEN.findall(chunk.text.casefold())
        if not query_tokens or not tokens:
            scores.append(0.0)
            continue
        overlap = sum(token in query_tokens for token in tokens)
        scores.append(float(overlap) / float(len(tokens)))
    return scores


def bm25_chunk_scores(
    query: str,
    chunks: Sequence[ChunkRecord],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    return bm25_scores(query, [chunk.text for chunk in chunks], k1=k1, b=b)


def rrf_fuse(
    rankings: Sequence[Sequence[int]],
    *,
    k: int = 60,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[int(chunk_id)] = scores.get(int(chunk_id), 0.0) + 1.0 / float(k + rank)
    return scores


def dense_chunk_scores(
    query: str,
    chunks: Sequence[ChunkRecord],
    encode: Callable[[Sequence[str]], np.ndarray],
) -> list[float]:
    if not chunks:
        return []
    matrix = np.asarray(encode([query, *[chunk.text for chunk in chunks]]), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return cosine_scores(matrix[0], matrix[1:])


def cross_encoder_scores(
    query: str,
    chunks: Sequence[ChunkRecord],
    reranker: PairwiseScorer,
) -> list[float]:
    if not chunks:
        return []
    pairs = [(query, chunk.text) for chunk in chunks]
    raw = list(reranker.predict(pairs))
    if len(raw) != len(chunks):
        raise ValueError("cross-encoder must return one score per chunk")
    return [float(item) for item in raw]


def section_scores(query: str, chunks: Sequence[ChunkRecord], reranker: PairwiseScorer | None = None) -> list[float]:
    titles = [chunk.section_title.strip() or chunk.section_id for chunk in chunks]
    if reranker is None:
        fake = [ChunkRecord(
            chunk_id=chunk.chunk_id,
            text=title,
            label=chunk.label,
            primary_class=chunk.primary_class,
            position=chunk.position,
            section_id=chunk.section_id,
            section_title=chunk.section_title,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            token_start=chunk.token_start,
            token_end=chunk.token_end,
            gold_supporting=chunk.gold_supporting,
            previous_chunk_id=chunk.previous_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
        ) for chunk, title in zip(chunks, titles)]
        return lexical_overlap_scores(query, fake)
    pairs = [(query, title if title else " ") for title in titles]
    return [float(item) for item in reranker.predict(pairs)]


def mix_normalized(
    channels: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[float]:
    if not channels:
        return []
    length = len(channels[0])
    mixed = [0.0] * length
    for channel, weight in zip(channels, weights):
        scaled = min_max(channel)
        for index, value in enumerate(scaled):
            mixed[index] += float(weight) * value
    return mixed


def score_instance(
    instance: InstanceRecord,
    score_fn: ScoreFn,
) -> tuple[list[int], dict[int, float], list[float]]:
    scores = score_fn(instance.question, instance.chunks)
    return rank_ids(instance.chunks, scores), score_map(instance.chunks, scores), scores
