"""Offline KeyBERT-guided hard negatives. Does not retrain RoBERTa."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from experiments.keybert_topic_clustering.documents import PaperDocument
from experiments.keybert_topic_clustering.keyphrases import Keyphrase
from untie.extraction_metrics import char_f1
from untie.keyword_evidence import CachedDocumentEvidence

from .config import ExperimentParams
from .scoring import keybert_chunk_scores


def contains_gold_span(
    text: str,
    references: Sequence[str],
    *,
    min_chars: int,
) -> bool:
    haystack = str(text).casefold()
    for reference in references:
        needle = str(reference).strip().casefold()
        if len(needle) < min_chars:
            continue
        if needle and needle in haystack:
            return True
    return False


def hard_negatives_for_document(
    paper: PaperDocument,
    evidence: CachedDocumentEvidence,
    phrases: Sequence[Keyphrase],
    chunk_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
    params: ExperimentParams,
) -> dict[str, object]:
    """Thematically close chunks that do not contain a gold answer span."""
    weights = [float(item.weight) for item in phrases]
    topic = keybert_chunk_scores(
        chunk_embeddings,
        phrase_embeddings,
        weights,
        variant=params.keybert_topic_variant,
        mean_top_n=params.keybert_mean_top_n,
    )
    gold_hits = []
    negatives = []
    for index, (chunk, score) in enumerate(zip(evidence.chunks, topic)):
        has_gold = contains_gold_span(
            chunk.text,
            paper.references,
            min_chars=params.gold_substring_min_chars,
        )
        answer_f1 = char_f1(chunk.answer, list(paper.references))
        row = {
            "chunk_index": int(chunk.index),
            "keybert_score": float(score),
            "qa_confidence": float(chunk.confidence),
            "contains_gold_span": has_gold,
            "answer_char_f1": float(answer_f1),
            "token_count": int(chunk.token_count),
        }
        if has_gold or answer_f1 >= 0.5:
            gold_hits.append(row)
        else:
            negatives.append(row)
    negatives.sort(key=lambda item: -float(item["keybert_score"]))
    if negatives:
        median = float(np.median([item["keybert_score"] for item in negatives]))
        hard = [item for item in negatives if float(item["keybert_score"]) >= median]
    else:
        hard = []
    sampled = {}
    for ratio in params.hard_negative_ratios:
        k = int(round(ratio * max(len(hard), 0)))
        sampled[str(ratio)] = hard[: max(k, 1 if hard else 0)]
    return {
        "document_id": paper.doc_id,
        "split": paper.split,
        "n_chunks": len(evidence.chunks),
        "n_gold_like": len(gold_hits),
        "n_hard_negatives": len(hard),
        "gold_like": gold_hits,
        "hard_negatives": hard,
        "sampled_by_ratio": sampled,
    }


def build_hard_negative_table(
    papers: Sequence[PaperDocument],
    evidence: Mapping[str, CachedDocumentEvidence],
    phrases: Mapping[str, Sequence[Keyphrase]],
    chunk_embeddings: Mapping[str, np.ndarray],
    phrase_embeddings: Mapping[str, np.ndarray],
    params: ExperimentParams,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for paper in papers:
        if paper.split != "train" or paper.doc_id not in evidence:
            continue
        payload = hard_negatives_for_document(
            paper,
            evidence[paper.doc_id],
            phrases.get(paper.doc_id, ()),
            chunk_embeddings[paper.doc_id],
            phrase_embeddings.get(paper.doc_id, np.zeros((0, 0), dtype=np.float32)),
            params,
        )
        for item in payload["hard_negatives"]:
            rows.append(
                {
                    "document_id": paper.doc_id,
                    "chunk_index": item["chunk_index"],
                    "keybert_score": item["keybert_score"],
                    "qa_confidence": item["qa_confidence"],
                    "answer_char_f1": item["answer_char_f1"],
                    "n_hard_negatives": payload["n_hard_negatives"],
                    "n_gold_like": payload["n_gold_like"],
                }
            )
    return rows
