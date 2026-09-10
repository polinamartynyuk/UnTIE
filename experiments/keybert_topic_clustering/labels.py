"""Automatic cluster labels from nearest documents' KeyBERT phrases. No LLM."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .documents import PaperDocument
from .keyphrases import Keyphrase, KeyphraseConfig, extract_document_keyphrases
from .encoder import CachedSentenceEncoder


def representative_terms(
    train_papers: Sequence[PaperDocument],
    train_vectors: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    *,
    encoder: CachedSentenceEncoder,
    phrases: dict[str, tuple[Keyphrase, ...]],
    nearest: int = 8,
    top_terms: int = 8,
) -> dict[int, list[dict[str, object]]]:
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_cluster[int(label)].append(index)
    output: dict[int, list[dict[str, object]]] = {}
    for cluster, member_indices in by_cluster.items():
        members = train_vectors[member_indices]
        sims = cosine_similarity(centroids[cluster].reshape(1, -1), members)[0]
        order = np.argsort(-sims)[: min(nearest, len(member_indices))]
        scores: dict[str, float] = defaultdict(float)
        support: dict[str, int] = defaultdict(int)
        for local in order:
            paper = train_papers[member_indices[int(local)]]
            for item in phrases.get(paper.doc_id, ()):
                key = item.phrase.casefold()
                scores[key] += float(item.weight)
                support[key] += 1
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_terms]
        output[cluster] = [
            {
                "term": term,
                "weight": float(weight),
                "document_support": int(support[term]),
            }
            for term, weight in ranked
        ]
    return output


def nearest_papers(
    train_papers: Sequence[PaperDocument],
    train_vectors: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    *,
    nearest: int = 3,
) -> dict[int, list[str]]:
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_cluster[int(label)].append(index)
    output: dict[int, list[str]] = {}
    for cluster, member_indices in by_cluster.items():
        members = train_vectors[member_indices]
        sims = cosine_similarity(centroids[cluster].reshape(1, -1), members)[0]
        order = np.argsort(-sims)[: min(nearest, len(member_indices))]
        output[cluster] = [train_papers[member_indices[int(local)]].doc_id for local in order]
        output[cluster] = [
            train_papers[member_indices[int(local)]].title or train_papers[member_indices[int(local)]].doc_id
            for local in order
        ]
    return output
