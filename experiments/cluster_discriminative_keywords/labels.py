"""Positive / easy-negative / hard-negative chunk labels. Train-only for dictionaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from experiments.keybert_qa_integration.hard_negatives import contains_gold_span
from untie.extraction_metrics import char_f1
from untie.keyword_evidence import CachedDocumentEvidence
from untie.ranking import WeightedKeyword, score_chunks

from .config import ExperimentParams, GOLD_ANSWER_F1, GOLD_SUBSTRING_MIN_CHARS


def _chunks_as_textchunks(evidence: CachedDocumentEvidence):
    from untie.domain import Sentence, TextChunk

    return [
        TextChunk(
            sentences=(Sentence(text=item.text, number=item.index, tokens=tuple(item.text.split())),),
            text=item.text,
            token_count=item.token_count,
        )
        for item in evidence.chunks
    ]


def gold_offsets(text: str, references: Sequence[str]) -> tuple[int | None, int | None]:
    haystack = str(text or "")
    folded = haystack.casefold()
    for reference in references:
        needle = str(reference).strip()
        if not needle:
            continue
        index = folded.find(needle.casefold())
        if index >= 0:
            return index, index + len(needle)
    return None, None


def label_document_chunks(
    evidence: CachedDocumentEvidence,
    *,
    cluster_id: int,
    split: str,
    params: ExperimentParams | None = None,
) -> list[dict[str, object]]:
    params = params or ExperimentParams()
    refs = list(evidence.references)
    rows = []
    for chunk in evidence.chunks:
        has_gold = contains_gold_span(
            chunk.text, refs, min_chars=params.gold_substring_min_chars
        )
        answer_f1 = float(char_f1(chunk.answer, refs))
        is_positive = bool(has_gold or answer_f1 >= params.gold_answer_f1)
        start, end = gold_offsets(chunk.text, refs)
        rows.append(
            {
                "document_id": evidence.doc_id,
                "chunk_id": int(chunk.index),
                "split": split,
                "cluster_id": int(cluster_id),
                "gold_span": " | ".join(refs),
                "global_span_start": start if start is not None else "",
                "global_span_end": end if end is not None else "",
                "contains_gold_span": int(has_gold),
                "answer_char_f1": answer_f1,
                "qa_confidence": float(chunk.confidence),
                "is_positive": int(is_positive),
                "token_count": int(chunk.token_count),
                "answer": chunk.answer,
            }
        )
    if rows:
        best = max(rows, key=lambda item: float(item["answer_char_f1"]))
        for item in rows:
            item["is_gold_chunk"] = int(
                item["chunk_id"] == best["chunk_id"] and float(best["answer_char_f1"]) > 0
            )
    return rows


def _top_indices(scores: Sequence[float], mask: Sequence[bool], k: int) -> list[int]:
    eligible = [index for index, keep in enumerate(mask) if keep]
    eligible.sort(key=lambda index: -float(scores[index]))
    return eligible[: max(0, int(k))]


def hard_negative_indices(
    evidence: CachedDocumentEvidence,
    labels: Sequence[Mapping[str, object]],
    *,
    global_keywords: Sequence[WeightedKeyword] | None = None,
    chunk_embeddings: np.ndarray | None = None,
    question_embedding: np.ndarray | None = None,
    params: ExperimentParams | None = None,
) -> dict[str, list[int]]:
    """HN1 QA-conf, HN2 keyword score, HN3 sim-to-positive, HN4 sim-to-question, HN5 union."""
    params = params or ExperimentParams()
    k = params.hn_top_n
    n = len(labels)
    if n == 0:
        return {name: [] for name in ("HN1", "HN2", "HN3", "HN4", "HN5", "easy")}
    is_pos = [bool(int(item["is_positive"])) for item in labels]
    not_pos = [not flag for flag in is_pos]
    conf = [float(item["qa_confidence"]) for item in labels]
    hn1 = _top_indices(conf, not_pos, k)

    kw_scores = [0.0] * n
    if global_keywords:
        chunks = _chunks_as_textchunks(evidence)
        scored = score_chunks(
            chunks, list(global_keywords), keep_unmatched=True, unmatched_floor=0.0
        )
        by_index = {item.chunk.start_sentence: float(item.score) for item in scored}
        kw_scores = [float(by_index.get(int(item["chunk_id"]), 0.0)) for item in labels]
    hn2 = _top_indices(kw_scores, not_pos, k)

    hn3: list[int] = []
    if chunk_embeddings is not None and chunk_embeddings.shape[0] == n and any(is_pos):
        pos_vec = chunk_embeddings[np.asarray(is_pos, dtype=bool)].mean(axis=0, keepdims=True)
        sims = (chunk_embeddings @ pos_vec.T).reshape(-1)
        hn3 = _top_indices([float(value) for value in sims], not_pos, k)

    hn4: list[int] = []
    if (
        chunk_embeddings is not None
        and question_embedding is not None
        and chunk_embeddings.shape[0] == n
    ):
        q = np.asarray(question_embedding).reshape(-1)
        sims_q = chunk_embeddings @ q
        hn4 = _top_indices([float(value) for value in sims_q], not_pos, k)

    union = []
    seen = set()
    for index in (*hn1, *hn2, *hn3, *hn4):
        if index not in seen:
            seen.add(index)
            union.append(index)
    easy = [index for index, flag in enumerate(not_pos) if flag]
    return {
        "HN1": hn1,
        "HN2": hn2,
        "HN3": hn3,
        "HN4": hn4,
        "HN5": union,
        "easy": easy,
    }


def flatten_hard_negative_table(
    by_doc: Mapping[str, Mapping[str, list[int]]],
    labels_by_doc: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    strategy: str = "HN5",
) -> list[dict[str, object]]:
    rows = []
    for doc_id, strategies in by_doc.items():
        labels = labels_by_doc[doc_id]
        for index in strategies.get(strategy, ()):
            item = labels[index]
            rows.append(
                {
                    "document_id": doc_id,
                    "chunk_id": item["chunk_id"],
                    "cluster_id": item["cluster_id"],
                    "split": item["split"],
                    "strategy": strategy,
                    "qa_confidence": item["qa_confidence"],
                    "answer_char_f1": item["answer_char_f1"],
                }
            )
    return rows


def cluster_chunk_groups(
    labels: Sequence[Mapping[str, object]],
    hard_by_doc: Mapping[str, Mapping[str, list[int]]],
    *,
    strategy: str,
) -> dict[int, dict[str, list[tuple[str, int]]]]:
    """cluster_id -> {positive, hard, easy} lists of (doc_id, chunk_id)."""
    by_doc: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in labels:
        by_doc[str(row["document_id"])].append(row)
    grouped: dict[int, dict[str, list[tuple[str, int]]]] = defaultdict(
        lambda: {"positive": [], "hard": [], "easy": []}
    )
    for doc_id, rows in by_doc.items():
        rows_sorted = sorted(rows, key=lambda item: int(item["chunk_id"]))
        cluster_id = int(rows_sorted[0]["cluster_id"])
        strategies = hard_by_doc.get(doc_id, {})
        hard_chunk_ids = {
            int(rows[index]["chunk_id"])
            for index in strategies.get(strategy, ())
            if 0 <= int(index) < len(rows)
        }
        easy_chunk_ids = {
            int(rows[index]["chunk_id"])
            for index in strategies.get("easy", ())
            if 0 <= int(index) < len(rows)
        }
        for row in rows_sorted:
            pair = (doc_id, int(row["chunk_id"]))
            if int(row["is_positive"]):
                grouped[cluster_id]["positive"].append(pair)
            elif int(row["chunk_id"]) in hard_chunk_ids:
                grouped[cluster_id]["hard"].append(pair)
            elif int(row["chunk_id"]) in easy_chunk_ids:
                grouped[cluster_id]["easy"].append(pair)
    return grouped
