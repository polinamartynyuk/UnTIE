"""Assign robust_positive / question_sensitive / HN. Any-correct is not the positive rule."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from experiments.phase1_scirex.gap_kmeans import ClusterResult, cluster_passages, gumbel_labels

from .bm25_multi import aggregate_hardness, multi_question_bm25
from .config import MultiQuestionConfig
from .consistency import consistency_class, qa_consistency


PRIMARY_CLASSES = (
    "robust_positive",
    "question_sensitive",
    "gold_supporting_robust_negative",
    "false_negative_candidate",
    "confirmed_false_negative",
    "hard_negative",
    "easy_negative",
)


@dataclass
class QuestionPrediction:
    question_id: int
    question: str
    predicted_answer: str
    matched_gold_answer: str | None
    qa_correct: bool
    qa_confidence: float
    answer_quality: float
    match_type: str
    extra_tokens: int | None
    start: int | None
    end: int | None
    start_logit: float | None
    end_logit: float | None
    bm25_score: float = 0.0
    bm25_normalized: float = 0.0
    bm25_rank: int = 0


@dataclass
class ChunkState:
    chunk_id: int
    text: str
    gold_supporting: bool
    predictions: list[QuestionPrediction]
    n_correct: int
    n_questions: int
    qa_consistency: float
    consistency_class: str
    any_correct: bool
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    section_type: str
    section_title: str
    eligible_for_qa: bool = True
    hardness: float = 0.0
    mean_normalized_bm25: float = 0.0
    mean_rank: float = 0.0
    rrf_score: float = 0.0
    dense_question_similarity: float = 0.0
    cluster_id: int | None = None
    same_cluster_as_positive: bool = False
    primary_class: str = "easy_negative"


@dataclass
class InstanceResult:
    document_id: str
    aspect: str
    qa_instance_id: str
    split: str
    canonical_question: str
    questions: list[str]
    n_questions: int
    status: str
    instance_status: str
    enrichment_mode: str
    extractable_gold_answers: list[str]
    unextractable_gold_answers: list[str]
    robust_positive_threshold: float
    chunks: list[ChunkState]
    fn_filter_status: str
    selected_k: int
    bm25_aggregate: str
    gap_values: tuple[float, ...] = ()
    gap_sk: tuple[float, ...] = ()
    cluster_sizes: tuple[int, ...] = ()
    problems: list[str] = field(default_factory=list)

    @property
    def robust_positive_ids(self) -> list[int]:
        return [item.chunk_id for item in self.chunks if item.primary_class == "robust_positive"]

    @property
    def question_sensitive_ids(self) -> list[int]:
        return [item.chunk_id for item in self.chunks if item.primary_class == "question_sensitive"]

    @property
    def gold_failed_ids(self) -> list[int]:
        return [
            item.chunk_id
            for item in self.chunks
            if item.primary_class == "gold_supporting_robust_negative"
        ]

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


def _apply_clusters(
    positives: Sequence[ChunkState],
    candidates: Sequence[ChunkState],
    embeddings_by_id: dict[int, np.ndarray],
    config: MultiQuestionConfig,
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
    questions: Sequence[str],
    status: str,
    extractable_gold_answers: Sequence[str],
    unextractable_gold_answers: Sequence[str],
    chunks: list[ChunkState],
    embeddings_by_id: dict[int, np.ndarray],
    config: MultiQuestionConfig,
    threshold: float,
    gold_mention_chunk_ids: set[int] | None = None,
) -> InstanceResult:
    n_questions = len(questions)
    for item in chunks:
        item.cluster_id = None
        item.same_cluster_as_positive = False
        item.primary_class = "easy_negative"
    texts = [item.text for item in chunks]
    bm25 = multi_question_bm25(
        questions,
        texts,
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    hardness = aggregate_hardness(bm25, method=config.bm25_aggregate)
    for index, item in enumerate(chunks):
        item.n_questions = n_questions
        item.n_correct = sum(1 for pred in item.predictions if pred.qa_correct)
        item.any_correct = item.n_correct > 0
        item.qa_consistency = qa_consistency(item.n_correct, n_questions)
        item.consistency_class = consistency_class(item.qa_consistency, threshold=threshold)
        item.mean_normalized_bm25 = float(bm25["mean_normalized"][index])
        item.mean_rank = float(bm25["mean_rank"][index])
        item.rrf_score = float(bm25["rrf"][index])
        item.hardness = float(hardness[index])
        for pred in item.predictions:
            qid = pred.question_id
            pred.bm25_score = float(bm25["bm25_by_question"][qid][index])
            pred.bm25_normalized = float(bm25["bm25_norm_by_question"][qid][index])
            pred.bm25_rank = int(bm25["rank_by_question"][qid][index])

    positives = [item for item in chunks if item.consistency_class == "robust_positive"]
    pool = [
        item
        for item in chunks
        if item.eligible_for_qa
        and item.consistency_class == "robust_negative"
        and not item.gold_supporting
    ]
    mention_ids = gold_mention_chunk_ids or set()
    problems: list[str] = []

    if not positives:
        for item in chunks:
            item.primary_class = _class_without_hn(item)
        return InstanceResult(
            document_id=document_id,
            aspect=aspect,
            qa_instance_id=f"{document_id}::{aspect}",
            split=split,
            canonical_question=str(questions[0]),
            questions=list(questions),
            n_questions=n_questions,
            status=status,
            instance_status="no_robust_positive",
            enrichment_mode=config.enrichment_mode,
            extractable_gold_answers=list(extractable_gold_answers),
            unextractable_gold_answers=list(unextractable_gold_answers),
            robust_positive_threshold=float(threshold),
            chunks=chunks,
            fn_filter_status="skipped_no_robust_positive",
            selected_k=0,
            bm25_aggregate=config.bm25_aggregate,
            problems=problems,
        )

    ranked_pool = sorted(pool, key=lambda item: (-item.hardness, item.chunk_id))
    hn_candidates = ranked_pool[: min(int(config.hard_negative_top_n), len(ranked_pool))]
    cluster = _apply_clusters(positives, hn_candidates, embeddings_by_id, config)
    fn_ids: set[int] = set()
    if cluster.status == "applied":
        fn_ids = {item.chunk_id for item in hn_candidates if item.same_cluster_as_positive}
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
        if item.consistency_class == "robust_positive":
            item.primary_class = "robust_positive"
        elif item.consistency_class == "question_sensitive":
            item.primary_class = "question_sensitive"
        elif item.gold_supporting:
            item.primary_class = "gold_supporting_robust_negative"
        elif item.chunk_id in confirmed:
            item.primary_class = "confirmed_false_negative"
        elif item.chunk_id in fn_ids:
            item.primary_class = "false_negative_candidate"
        elif item.chunk_id in hn_ids:
            item.primary_class = "hard_negative"
        else:
            item.primary_class = "easy_negative"

    return InstanceResult(
        document_id=document_id,
        aspect=aspect,
        qa_instance_id=f"{document_id}::{aspect}",
        split=split,
        canonical_question=str(questions[0]),
        questions=list(questions),
        n_questions=n_questions,
        status=status,
        instance_status="ok",
        enrichment_mode=config.enrichment_mode,
        extractable_gold_answers=list(extractable_gold_answers),
        unextractable_gold_answers=list(unextractable_gold_answers),
        robust_positive_threshold=float(threshold),
        chunks=chunks,
        fn_filter_status=cluster.status,
        selected_k=cluster.selected_k,
        bm25_aggregate=config.bm25_aggregate,
        gap_values=cluster.gap_values,
        gap_sk=cluster.gap_sk,
        cluster_sizes=cluster.cluster_sizes,
        problems=problems,
    )


def _class_without_hn(item: ChunkState) -> str:
    if item.consistency_class == "robust_positive":
        return "robust_positive"
    if item.consistency_class == "question_sensitive":
        return "question_sensitive"
    if item.gold_supporting:
        return "gold_supporting_robust_negative"
    return "easy_negative"


def relabel_with_threshold(
    result: InstanceResult,
    *,
    embeddings_by_id: dict[int, np.ndarray],
    config: MultiQuestionConfig,
    threshold: float,
    gold_mention_chunk_ids: set[int] | None = None,
) -> InstanceResult:
    """Recompute classes from stored per-question predictions. No new QA."""
    return assign_instance(
        document_id=result.document_id,
        aspect=result.aspect,
        split=result.split,
        questions=result.questions,
        status=result.status,
        extractable_gold_answers=result.extractable_gold_answers,
        unextractable_gold_answers=result.unextractable_gold_answers,
        chunks=result.chunks,
        embeddings_by_id=embeddings_by_id,
        config=config,
        threshold=threshold,
        gold_mention_chunk_ids=gold_mention_chunk_ids,
    )
