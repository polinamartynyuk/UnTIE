"""Reuse E0 retrieval signals on gold-free RuntimeChunk objects."""

from __future__ import annotations

from typing import Sequence

from experiments.qa_chunk_selection.ranking import (
    bm25_chunk_scores,
    cosine_scores,
    cross_encoder_scores,
    min_max,
    rrf_fuse,
)
from experiments.qa_chunk_selection.schema import ChunkRecord

from .qa import lexical_overlap
from .schemas import RuntimeChunk, RuntimeInstance, rank_ids


def _as_text_chunks(chunks: Sequence[RuntimeChunk]) -> list[ChunkRecord]:
    """Duck-type text/id only. Labels are dummy and must not be read by callers."""
    return [
        ChunkRecord(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            label="easy_negative",
            primary_class="easy_negative",
            position=chunk.position,
            section_id=chunk.section_id,
            section_title=chunk.section_title,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            token_start=chunk.token_start,
            token_end=chunk.token_end,
            gold_supporting=False,
            previous_chunk_id=chunk.previous_chunk_id,
            next_chunk_id=chunk.next_chunk_id,
        )
        for chunk in chunks
    ]


def dense_scores(instance: RuntimeInstance, encode) -> list[float]:
    texts = [instance.question, *[chunk.text for chunk in instance.chunks]]
    if len(texts) == 1:
        return []
    import numpy as np

    matrix = np.asarray(encode(texts), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return cosine_scores(matrix[0], matrix[1:])


def relevance_scores(instance: RuntimeInstance, reranker) -> list[float]:
    return cross_encoder_scores(instance.question, _as_text_chunks(instance.chunks), reranker)


def bm25_scores(instance: RuntimeInstance) -> list[float]:
    return bm25_chunk_scores(instance.question, _as_text_chunks(instance.chunks))


def rrf_dense_bm25(instance: RuntimeInstance, encode, *, rrf_k: int = 60) -> list[float]:
    dense = dense_scores(instance, encode)
    bm25 = bm25_scores(instance)
    fused = rrf_fuse(
        [rank_ids(instance.chunks, dense), rank_ids(instance.chunks, bm25)],
        k=rrf_k,
    )
    return [float(fused.get(chunk.chunk_id, 0.0)) for chunk in instance.chunks]


def section_title_scores(instance: RuntimeInstance) -> list[float]:
    scores = []
    missing = []
    for chunk in instance.chunks:
        title = str(chunk.section_title or "").strip()
        missing.append(not bool(title))
        scores.append(lexical_overlap(instance.question, title) if title else 0.0)
    return scores


def section_id_scores(instance: RuntimeInstance) -> list[float]:
    return [lexical_overlap(instance.question, str(chunk.section_id or "")) for chunk in instance.chunks]


def section_title_missing(instance: RuntimeInstance) -> list[bool]:
    return [not bool(str(chunk.section_title or "").strip()) for chunk in instance.chunks]


def fuse_minmax(channels: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    mixed = [0.0] * len(channels[0])
    for channel, weight in zip(channels, weights):
        scaled = min_max(channel)
        for index, value in enumerate(scaled):
            mixed[index] += float(weight) * float(value)
    return mixed


def fuse_ranks(rankings: Sequence[Sequence[int]], *, rrf_k: int = 60) -> dict[int, float]:
    return rrf_fuse(list(rankings), k=rrf_k)
