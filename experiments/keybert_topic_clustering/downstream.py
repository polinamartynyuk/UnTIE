"""Downstream Q0/Q1/D0/D1/D2 on frozen clustering. Reuses extractive QA evidence."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from untie.attention import AttentionKeywordExtractor
from untie.cli import load_static_keywords
from untie.config import PipelineConfig
from untie.extraction_metrics import char_f1
from untie.keyword_evidence import (
    CachedDocumentEvidence,
    EvidenceStore,
    ExtractionMetricCache,
    _cluster_indices,
    collect_document_evidence,
    rerank_cached_document,
    select_cached_answer,
)
from untie.keyword_training import StrategyConfig, TrainingConfig, tune_global_keywords
from untie.models import ModelFactory, profile_for_language
from untie.pipelines import DocumentProcessor
from untie.ranking import WeightedKeyword, score_chunks

from .documents import PaperDocument
from .routing import RoutingResult


ASPECT_ID = "task"
ASPECT_REFERENCE = "task. main task being solved in paper"
QUESTION = "Which task was solved?"


def select_frequency_cluster(
    answers: list,
    similarity: np.ndarray,
    *,
    threshold: float = 0.75,
):
    if not answers:
        return None
    if len(answers) == 1:
        return answers[0]
    clusters = _cluster_indices(similarity, threshold)
    best = max(clusters, key=len)
    texts = [answers[index].text for index in best]
    winner_text, _ = Counter(texts).most_common(1)[0]
    tied = [index for index in best if answers[index].text == winner_text]
    return answers[max(tied, key=lambda index: answers[index].confidence)]


def collect_evidence(
    papers: Sequence[PaperDocument],
    *,
    store: EvidenceStore,
    processor,
    answerer,
    encoder,
    attention,
    language: str,
    collect_candidates_splits: set[str],
    question: str = QUESTION,
) -> dict[str, CachedDocumentEvidence]:
    by_id: dict[str, CachedDocumentEvidence] = {}
    for paper in papers:
        mine = paper.split in collect_candidates_splits
        existing = store.load(paper.doc_id)
        if existing is not None and (existing.candidates or not mine):
            by_id[paper.doc_id] = existing
            continue
        evidence = collect_document_evidence(
            doc_id=paper.doc_id,
            text=paper.text,
            aspect_id=ASPECT_ID,
            aspect_reference=ASPECT_REFERENCE,
            question=question,
            references=paper.references,
            processor=processor,
            answerer=answerer,
            encoder=encoder,
            attention_extractor=attention,
            language=language,
            collect_candidates=mine,
            contrast_against_gold=True,
        )
        store.save(evidence)
        by_id[paper.doc_id] = evidence
    return by_id


def keywords_from_outcome(outcome) -> list[WeightedKeyword]:
    meta = {item.word: item for item in outcome.keyword_metadata}
    keywords = []
    for word in outcome.keywords:
        item = meta.get(word)
        keywords.append(
            WeightedKeyword(
                word=word,
                lemma=(item.lemma if item and item.lemma else word.lower()),
                stem=(item.stem if item and item.stem else word.lower()),
                attention_weight=item.attention_weight if item else 1.0,
                score_difference=item.score_difference if item else 0.0,
            )
        )
    return keywords


def fit_cluster_dictionary(
    cluster_id: int,
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    evidence: Mapping[str, CachedDocumentEvidence],
    *,
    strategy: StrategyConfig,
    language: str,
    seed: int,
    metric_cache: ExtractionMetricCache,
) -> tuple[list[WeightedKeyword], dict[str, object]]:
    usable_train = [doc_id for doc_id in train_ids if evidence[doc_id].candidates]
    if len(usable_train) < 2:
        return [], {"skipped": True, "reason": "too_few_train", "cluster_id": cluster_id}
    usable_dev = [doc_id for doc_id in dev_ids if doc_id in evidence]
    if not usable_dev:
        usable_dev = usable_train[-max(1, len(usable_train) // 5) :]
        usable_train = [doc_id for doc_id in usable_train if doc_id not in set(usable_dev)] or usable_train
    docs = [evidence[doc_id] for doc_id in [*usable_train, *usable_dev] if doc_id in evidence]
    config = TrainingConfig(
        language=language,
        max_keywords=10,
        min_keywords=min(4, max(1, len(usable_train))),
        selection_policy="union",
        stability_threshold=0.15,
        seed=seed + cluster_id,
        include_bertscore=False,
        strategies=(strategy,),
        evaluation_budget=120,
        max_candidates=80,
        enrich_train_references=False,
    )
    try:
        outcome = tune_global_keywords(
            docs,
            config=config,
            metric_cache=metric_cache,
            split={
                "train": tuple(usable_train),
                "dev": tuple(usable_dev),
                "test": (),
            },
            candidate_doc_ids=tuple(usable_train),
        )
    except (ValueError, RuntimeError) as error:
        return [], {
            "skipped": True,
            "reason": str(error),
            "cluster_id": cluster_id,
        }
    return keywords_from_outcome(outcome), {
        "skipped": False,
        "cluster_id": cluster_id,
        "keywords": list(outcome.keywords),
        "objective": outcome.objective,
        "strategy": strategy.name,
        "train_documents": len(usable_train),
        "dev_documents": len(usable_dev),
        "keyword_metadata": [item.to_dict() for item in outcome.keyword_metadata],
    }


def _answers_from_evidence(evidence: CachedDocumentEvidence):
    from untie.domain import Answer, Sentence, TextChunk

    answers = []
    chunks = []
    for item in evidence.chunks:
        chunk = TextChunk(
            sentences=(Sentence(text=item.text, number=item.index, tokens=tuple(item.text.split())),),
            text=item.text,
            token_count=item.token_count,
        )
        chunks.append(chunk)
        answers.append(
            Answer(
                text=item.answer,
                chunk=chunk,
                confidence=item.confidence,
                start_pos=item.start_pos,
                end_pos=item.end_pos,
                metadata={"chunk_score": item.confidence, "chunk_index": item.index},
            )
        )
    return answers, chunks


def predict_q0(evidence: CachedDocumentEvidence) -> tuple[str | None, dict[str, Any]]:
    return evidence.baseline_answer, {
        "fallback": evidence.baseline_answer is None,
        "matched_chunks": len(evidence.chunks),
        "qa_chunks": len(evidence.chunks),
        "qa_tokens": sum(item.token_count for item in evidence.chunks),
        "mode": "Q0",
    }


def predict_q1(
    evidence: CachedDocumentEvidence,
    keywords: Sequence[WeightedKeyword],
    strategy: Mapping[str, object],
) -> tuple[str | None, dict[str, Any]]:
    prediction, meta = rerank_cached_document(
        evidence,
        list(keywords),
        weight_ratio=float(
            {"only_weight": 1.0, "only_score_diff": 0.0, "equal_weight_score_diff": 0.5}.get(
                str(strategy.get("score_chunk_strategy", "equal_weight_score_diff")),
                0.5,
            )
        ),
        cluster_strategy=str(strategy.get("choose_cluster_strategy", "weighted_score")),
        answer_strategy=str(strategy.get("choose_answer_strategy", "combined_score")),
    )
    meta.update(
        {
            "qa_chunks": meta.get("matched_chunks", 0),
            "qa_tokens": sum(item.token_count for item in evidence.chunks),
            "mode": "Q1",
        }
    )
    return prediction, meta


def predict_d0(evidence: CachedDocumentEvidence) -> tuple[str | None, dict[str, Any]]:
    answers, _ = _answers_from_evidence(evidence)
    similarity = np.asarray(evidence.answer_similarity, dtype=float)
    selected = select_frequency_cluster(answers, similarity)
    return (
        None if selected is None else selected.text,
        {
            "fallback": selected is None,
            "matched_chunks": len(answers),
            "qa_chunks": len(answers),
            "qa_tokens": sum(item.token_count for item in evidence.chunks),
            "mode": "D0",
            "answer_cluster_size": None,
        },
    )


def _soft_keywords(
    route: RoutingResult,
    dictionaries: Mapping[int, list[WeightedKeyword]],
    global_keywords: Sequence[WeightedKeyword],
) -> list[tuple[list[WeightedKeyword], float]]:
    mixed = []
    for cluster_id, weight in zip(route.top_ids, route.weights):
        keywords = dictionaries.get(int(cluster_id)) or list(global_keywords)
        mixed.append((keywords, float(weight)))
    return mixed


def predict_keyword_mode(
    evidence: CachedDocumentEvidence,
    route: RoutingResult,
    dictionaries: Mapping[int, list[WeightedKeyword]],
    global_keywords: Sequence[WeightedKeyword],
    *,
    weight_ratio: float,
    hard: bool,
    cluster_strategy: str = "weighted_score",
    answer_strategy: str = "combined_score",
) -> tuple[str | None, dict[str, Any], list[dict[str, Any]]]:
    answers, chunks = _answers_from_evidence(evidence)
    if hard:
        keywords = dictionaries.get(int(route.top1)) or list(global_keywords)
        fallback = int(route.top1) not in dictionaries
        scored = score_chunks(
            chunks, keywords, weight_ratio=weight_ratio, keep_unmatched=True
        )
        used = [(keywords, 1.0)]
    else:
        used = _soft_keywords(route, dictionaries, global_keywords)
        fallback = any(int(cluster) not in dictionaries for cluster in route.top_ids)
        scores = np.zeros(len(chunks), dtype=float)
        for keywords, pi in used:
            ranked = score_chunks(
                chunks, keywords, weight_ratio=weight_ratio, keep_unmatched=True
            )
            by_index = {item.chunk.start_sentence: item.score for item in ranked}
            for index, chunk in enumerate(chunks):
                scores[index] += pi * float(by_index.get(chunk.start_sentence, 0.0))
        from untie.domain import ScoredChunk

        scored = [
            ScoredChunk(chunk=chunk, score=float(score))
            for chunk, score in zip(chunks, scores)
        ]
    score_by_index = {item.chunk.start_sentence: float(item.score) for item in scored}
    for answer in answers:
        answer.metadata["chunk_score"] = score_by_index.get(
            answer.chunk.start_sentence, 0.0
        )
    similarity = np.asarray(evidence.answer_similarity, dtype=float)
    selected = select_cached_answer(
        answers,
        similarity,
        similarity_threshold=0.75,
        cluster_strategy=cluster_strategy,
        answer_strategy=answer_strategy,
    )
    chunk_rows = []
    gold_index = _gold_chunk_index(evidence)
    order = sorted(range(len(answers)), key=lambda index: -float(answers[index].metadata["chunk_score"]))
    ranks = {index: rank + 1 for rank, index in enumerate(order)}
    for index, answer in enumerate(answers):
        chunk_rows.append(
            {
                "chunk_id": index,
                "candidate_answer": answer.text,
                "CE_confidence": float(evidence.chunks[index].confidence),
                "chunk_keyword_score": float(answer.metadata["chunk_score"]),
                "selected": bool(selected is not None and selected is answer),
                "gold_chunk": index == gold_index,
                "rank": ranks[index],
            }
        )
    prediction = None if selected is None else selected.text
    if prediction is None:
        prediction = evidence.baseline_answer
        fallback = True
    return (
        prediction,
        {
            "fallback": fallback,
            "matched_chunks": len(scored),
            "qa_chunks": len(answers),
            "qa_tokens": sum(item.token_count for item in evidence.chunks),
            "routed_cluster": int(route.top1),
            "routing_similarity": float(route.top1_similarity),
            "hard": hard,
            "gold_chunk_rank": ranks.get(gold_index, -1) if gold_index is not None else -1,
        },
        chunk_rows,
    )


def _gold_chunk_index(evidence: CachedDocumentEvidence) -> int | None:
    if not evidence.references or not evidence.chunks:
        return None
    scores = [
        char_f1(item.answer, list(evidence.references)) for item in evidence.chunks
    ]
    if max(scores) <= 0:
        return None
    return int(np.argmax(scores))


def retrieval_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    gold = next((row for row in rows if row.get("gold_chunk")), None)
    if gold is None:
        return {
            "evidence_recall@1": 0.0,
            "evidence_recall@3": 0.0,
            "evidence_recall@5": 0.0,
            "mrr_evidence": 0.0,
            "gold_chunk_rank": -1.0,
        }
    rank = int(gold["rank"])
    return {
        "evidence_recall@1": float(rank == 1),
        "evidence_recall@3": float(rank <= 3),
        "evidence_recall@5": float(rank <= 5),
        "mrr_evidence": float(1.0 / rank) if rank > 0 else 0.0,
        "gold_chunk_rank": float(rank),
    }


def bootstrap_mean_ci(values: Sequence[float], *, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(1000):
        sample = rng.choice(array, size=len(array), replace=True)
        means.append(float(sample.mean()))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(array.mean()), float(lo), float(hi)
