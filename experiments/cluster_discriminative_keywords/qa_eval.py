"""QA evaluation on cached RoBERTa spans. Does not call the reader."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from untie.keyword_evidence import CachedDocumentEvidence, select_cached_answer
from untie.ranking import WeightedKeyword

from .ranking import score_document_chunks


def _norm(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    numbers = [float(item) for item in values]
    lo, hi = min(numbers), max(numbers)
    if hi - lo <= 1e-12:
        return [0.0] * len(numbers)
    return [(item - lo) / (hi - lo) for item in numbers]


def centroid_index(similarity: np.ndarray, indices: Sequence[int]) -> int:
    if not indices:
        return 0
    if len(indices) == 1:
        return int(indices[0])
    ix = np.asarray(list(indices), dtype=int)
    sub = similarity[np.ix_(ix, ix)]
    return int(ix[int(np.argmax(sub.mean(axis=1)))])


def predict_q0(evidence: CachedDocumentEvidence) -> tuple[str | None, dict[str, object]]:
    n = len(evidence.chunks)
    tokens = sum(item.token_count for item in evidence.chunks)
    return evidence.baseline_answer, {
        "fallback": evidence.baseline_answer is None,
        "qa_chunks": n,
        "qa_tokens": tokens,
        "mode": "Q0",
    }


def predict_consensus(
    evidence: CachedDocumentEvidence,
    chunk_scores: Sequence[float],
    *,
    indices: Sequence[int] | None = None,
) -> tuple[str | None, dict[str, object]]:
    from untie.domain import Answer, Sentence, TextChunk

    keep = list(indices) if indices is not None else list(range(len(evidence.chunks)))
    if not keep:
        return predict_q0(evidence)
    answers = []
    for index in keep:
        chunk = evidence.chunks[index]
        text_chunk = TextChunk(
            sentences=(Sentence(text=chunk.text, number=chunk.index, tokens=tuple(chunk.text.split())),),
            text=chunk.text,
            token_count=chunk.token_count,
        )
        answers.append(
            Answer(
                text=chunk.answer,
                chunk=text_chunk,
                confidence=chunk.confidence,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                metadata={"chunk_score": float(chunk_scores[index])},
            )
        )
    similarity = np.asarray(evidence.answer_similarity, dtype=float)
    if similarity.size == 0 or similarity.shape[0] != len(evidence.chunks):
        best = keep[int(np.argmax([float(chunk_scores[index]) for index in keep]))]
        pred = evidence.chunks[best].answer
        tokens = sum(evidence.chunks[index].token_count for index in keep)
        return pred, {
            "fallback": False,
            "qa_chunks": len(keep),
            "qa_tokens": tokens,
            "mode": "consensus_score_fallback",
        }
    subset = similarity[np.ix_(keep, keep)]
    selected = select_cached_answer(
        answers,
        subset,
        similarity_threshold=0.75,
        cluster_strategy="weighted_score",
        answer_strategy="combined_score",
    )
    pred = selected.text if selected is not None else evidence.baseline_answer
    tokens = sum(evidence.chunks[index].token_count for index in keep)
    return pred, {
        "fallback": selected is None,
        "qa_chunks": len(keep),
        "qa_tokens": tokens,
        "mode": "consensus",
    }


def predict_linear(
    evidence: CachedDocumentEvidence,
    chunk_scores: Sequence[float],
    *,
    alpha: float,
    beta: float,
    indices: Sequence[int] | None = None,
) -> tuple[str | None, dict[str, object]]:
    keep = list(indices) if indices is not None else list(range(len(evidence.chunks)))
    if not keep:
        return predict_q0(evidence)
    conf = _norm([float(evidence.chunks[index].confidence) for index in keep])
    scores = _norm([float(chunk_scores[index]) for index in keep])
    mixed = [alpha * c + beta * s for c, s in zip(conf, scores)]
    best = keep[int(np.argmax(mixed))]
    tokens = sum(evidence.chunks[index].token_count for index in keep)
    return evidence.chunks[best].answer, {
        "fallback": False,
        "qa_chunks": len(keep),
        "qa_tokens": tokens,
        "mode": "linear",
        "selected_chunk": best,
    }


def predict_prefilter_centroid(
    evidence: CachedDocumentEvidence,
    chunk_scores: Sequence[float],
    *,
    top_k: int,
) -> tuple[str | None, dict[str, object]]:
    n = len(evidence.chunks)
    if n == 0:
        return None, {"fallback": True, "qa_chunks": 0, "qa_tokens": 0, "mode": "prefilter"}
    order = sorted(range(n), key=lambda index: (-float(chunk_scores[index]), index))
    keep = order[: max(1, min(int(top_k), n))]
    similarity = np.asarray(evidence.answer_similarity, dtype=float)
    if similarity.size == 0:
        pred = evidence.chunks[keep[0]].answer
    else:
        chosen = centroid_index(similarity, keep)
        pred = evidence.chunks[chosen].answer
    tokens = sum(evidence.chunks[index].token_count for index in keep)
    return pred, {
        "fallback": False,
        "qa_chunks": len(keep),
        "qa_tokens": tokens,
        "mode": "prefilter_centroid",
        "top_k": int(top_k),
    }


def predict_hybrid(
    evidence: CachedDocumentEvidence,
    chunk_scores: Sequence[float],
    *,
    top_k: int,
    alpha: float,
    beta: float,
    use_consensus: bool,
) -> tuple[str | None, dict[str, object]]:
    n = len(evidence.chunks)
    order = sorted(range(n), key=lambda index: (-float(chunk_scores[index]), index))
    keep = order[: max(1, min(int(top_k), n))]
    if use_consensus:
        pred, meta = predict_consensus(evidence, chunk_scores, indices=keep)
    else:
        pred, meta = predict_linear(
            evidence, chunk_scores, alpha=alpha, beta=beta, indices=keep
        )
    meta["mode"] = "hybrid"
    meta["top_k"] = int(top_k)
    return pred, meta


def keyword_chunk_scores(
    evidence: CachedDocumentEvidence,
    keywords: Sequence[WeightedKeyword],
    *,
    use_modifiers: bool,
) -> list[float]:
    return score_document_chunks(evidence, keywords, use_modifiers=use_modifiers)
