"""Aspect-specific Phase 1 class assignment. No attention/KeyBERT/diff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .bm25 import bm25_scores
from .config import Phase1Config
from .gap_kmeans import ClusterResult, cluster_passages, gumbel_labels


PRIMARY_CLASSES = (
    "qa_positive",
    "evidence_positive_qa_failed",
    "confirmed_false_negative",
    "false_negative_candidate",
    "hard_negative",
    "easy_negative",
)


@dataclass
class ChunkState:
    chunk_id: int
    text: str
    gold_supporting: bool
    qa_positive: bool
    predicted_answer: str
    matched_gold_answer: str | None
    qa_confidence: float
    answer_quality: float
    start: int | None
    end: int | None
    start_logit: float | None
    end_logit: float | None
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    section_type: str
    section_title: str
    eligible_for_qa: bool = True
    bm25_score: float = 0.0
    dense_question_similarity: float = 0.0
    cluster_id: int | None = None
    same_cluster_as_positive: bool = False
    primary_class: str = "easy_negative"
    match_type: str = "none"
    extra_tokens: int | None = None


@dataclass
class InstanceResult:
    document_id: str
    aspect: str
    qa_instance_id: str
    split: str
    question: str
    status: str
    instance_status: str
    enrichment_mode: str
    extractable_gold_answers: list[str]
    unextractable_gold_answers: list[str]
    chunks: list[ChunkState]
    fn_filter_status: str
    selected_k: int
    gap_values: tuple[float, ...] = ()
    gap_sk: tuple[float, ...] = ()
    cluster_sizes: tuple[int, ...] = ()
    problems: list[str] = field(default_factory=list)

    @property
    def positive_ids(self) -> list[int]:
        return [item.chunk_id for item in self.chunks if item.primary_class == "qa_positive"]

    @property
    def evidence_failed_ids(self) -> list[int]:
        return [item.chunk_id for item in self.chunks if item.primary_class == "evidence_positive_qa_failed"]

    @property
    def fn_ids(self) -> list[int]:
        return [item.chunk_id for item in self.chunks if item.primary_class == "false_negative_candidate"]

    @property
    def confirmed_fn_ids(self) -> list[int]:
        return [item.chunk_id for item in self.chunks if item.primary_class == "confirmed_false_negative"]

    @property
    def hard_negative_ids(self) -> list[int]:
        return [item.chunk_id for item in self.chunks if item.primary_class == "hard_negative"]

    @property
    def easy_negative_ids(self) -> list[int]:
        return [item.chunk_id for item in self.chunks if item.primary_class == "easy_negative"]


def _negative_pool(chunks: Sequence[ChunkState]) -> list[ChunkState]:
    return [
        item
        for item in chunks
        if item.eligible_for_qa and not item.qa_positive and not item.gold_supporting
    ]


def _assign_bm25(chunks: Sequence[ChunkState], question: str, config: Phase1Config) -> None:
    texts = [item.text for item in chunks]
    scores = bm25_scores(question, texts, k1=config.bm25_k1, b=config.bm25_b)
    for item, score in zip(chunks, scores):
        item.bm25_score = float(score)


def _top_n_negatives(pool: Sequence[ChunkState], top_n: int) -> list[ChunkState]:
    ranked = sorted(pool, key=lambda item: (-item.bm25_score, item.chunk_id))
    return ranked[: min(int(top_n), len(ranked))]


def _apply_clusters(
    positives: Sequence[ChunkState],
    candidates: Sequence[ChunkState],
    embeddings_by_id: dict[int, np.ndarray],
    config: Phase1Config,
) -> ClusterResult:
    members = [item for item in (*positives, *candidates)]
    if not members:
        return ClusterResult("skipped_insufficient_data", 0, (), None, (), (), ())
    matrix = np.vstack([embeddings_by_id[item.chunk_id] for item in members])
    result = cluster_passages(
        matrix,
        k_max_cap=config.k_max_cap,
        bootstrap=config.gap_bootstrap,
        seed=config.seed,
        min_points=config.min_cluster_points,
    )
    if result.status == "applied":
        labels = list(result.labels)
        if config.gumbel_ablation and result.centers is not None:
            labels = [
                int(item)
                for item in gumbel_labels(
                    matrix,
                    result.centers,
                    temperature=config.gumbel_temperature,
                    seed=config.seed,
                    runs=1,
                )
            ]
            result = ClusterResult(
                status=result.status,
                selected_k=result.selected_k,
                labels=tuple(labels),
                centers=result.centers,
                gap_values=result.gap_values,
                gap_sk=result.gap_sk,
                cluster_sizes=result.cluster_sizes,
            )
        for item, label in zip(members, result.labels):
            item.cluster_id = int(label)
        positive_labels = {item.cluster_id for item in positives if item.cluster_id is not None}
        for item in members:
            item.same_cluster_as_positive = item.cluster_id in positive_labels
    return result


def assign_instance(
    *,
    document_id: str,
    aspect: str,
    split: str,
    question: str,
    status: str,
    extractable_gold_answers: Sequence[str],
    unextractable_gold_answers: Sequence[str],
    chunks: list[ChunkState],
    embeddings_by_id: dict[int, np.ndarray],
    config: Phase1Config,
    gold_mention_chunk_ids: set[int] | None = None,
) -> InstanceResult:
    _assign_bm25(chunks, question, config)
    positives = [item for item in chunks if item.qa_positive]
    failed = [item for item in chunks if item.gold_supporting and not item.qa_positive]
    pool = _negative_pool(chunks)
    mention_ids = gold_mention_chunk_ids or set()
    problems: list[str] = []

    if not positives:
        for item in chunks:
            if item.qa_positive:
                item.primary_class = "qa_positive"
            elif item.gold_supporting:
                item.primary_class = "evidence_positive_qa_failed"
            else:
                item.primary_class = "easy_negative"
        return InstanceResult(
            document_id=document_id,
            aspect=aspect,
            qa_instance_id=f"{document_id}::{aspect}",
            split=split,
            question=question,
            status=status,
            instance_status="no_qa_positive",
            enrichment_mode=config.enrichment_mode,
            extractable_gold_answers=list(extractable_gold_answers),
            unextractable_gold_answers=list(unextractable_gold_answers),
            chunks=chunks,
            fn_filter_status="skipped_no_qa_positive",
            selected_k=0,
            problems=problems,
        )

    hn_candidates = _top_n_negatives(pool, config.hard_negative_top_n)
    cluster = _apply_clusters(positives, hn_candidates, embeddings_by_id, config)

    fn_ids: set[int] = set()
    if cluster.status == "applied":
        fn_ids = {
            item.chunk_id
            for item in hn_candidates
            if item.same_cluster_as_positive
        }
    confirmed: set[int] = set()
    for chunk_id in list(fn_ids):
        if chunk_id in mention_ids:
            confirmed.add(chunk_id)
            problems.append(
                f"{document_id}::{aspect} chunk {chunk_id}: FN candidate has gold mention"
            )

    hn_ids = {
        item.chunk_id
        for item in hn_candidates
        if item.chunk_id not in fn_ids and item.chunk_id not in confirmed
    }

    for item in chunks:
        if item.qa_positive:
            item.primary_class = "qa_positive"
        elif item.gold_supporting:
            item.primary_class = "evidence_positive_qa_failed"
        elif item.chunk_id in confirmed:
            item.primary_class = "confirmed_false_negative"
        elif item.chunk_id in fn_ids:
            item.primary_class = "false_negative_candidate"
        elif item.chunk_id in hn_ids:
            item.primary_class = "hard_negative"
        elif not item.qa_positive and not item.gold_supporting:
            item.primary_class = "easy_negative"
        else:
            item.primary_class = "easy_negative"

    del failed
    return InstanceResult(
        document_id=document_id,
        aspect=aspect,
        qa_instance_id=f"{document_id}::{aspect}",
        split=split,
        question=question,
        status=status,
        instance_status="ok",
        enrichment_mode=config.enrichment_mode,
        extractable_gold_answers=list(extractable_gold_answers),
        unextractable_gold_answers=list(unextractable_gold_answers),
        chunks=chunks,
        fn_filter_status=cluster.status,
        selected_k=cluster.selected_k,
        gap_values=cluster.gap_values,
        gap_sk=cluster.gap_sk,
        cluster_sizes=cluster.cluster_sizes,
        problems=problems,
    )
