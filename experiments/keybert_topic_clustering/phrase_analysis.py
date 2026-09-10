"""Separate clustering of KeyBERT phrases. Not mixed into document clustering."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .documents import PaperDocument
from .encoder import CachedSentenceEncoder, l2_normalize
from .keyphrases import Keyphrase


def cluster_keyphrases(
    phrases: dict[str, tuple[Keyphrase, ...]],
    encoder: CachedSentenceEncoder,
    *,
    k: int = 16,
    seed: int = 42,
) -> dict[str, object]:
    unique: dict[str, float] = {}
    owners: dict[str, set[str]] = defaultdict(set)
    for doc_id, items in phrases.items():
        for item in items:
            key = item.phrase.strip()
            if not key:
                continue
            unique[key] = max(unique.get(key, 0.0), float(item.weight))
            owners[key].add(doc_id)
    terms = sorted(unique)
    if len(terms) < max(4, k):
        k = max(2, min(k, max(2, len(terms) // 2)))
    vectors = l2_normalize(encoder.encode(terms))
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(vectors)
    grouped: dict[int, list[str]] = defaultdict(list)
    for term, label in zip(terms, labels):
        grouped[int(label)].append(term)
    return {
        "k": int(k),
        "phrases": [
            {
                "phrase": term,
                "cluster": int(label),
                "weight": unique[term],
                "document_support": len(owners[term]),
            }
            for term, label in zip(terms, labels)
        ],
        "cluster_sizes": {str(cluster): len(items) for cluster, items in grouped.items()},
    }


def document_phrase_overlap(
    papers: Sequence[PaperDocument],
    doc_labels: dict[str, int],
    phrase_rows: list[dict[str, object]],
    phrases: dict[str, tuple[Keyphrase, ...]],
) -> list[dict[str, object]]:
    phrase_cluster = {str(row["phrase"]).casefold(): int(row["cluster"]) for row in phrase_rows}
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for paper in papers:
        if paper.doc_id not in doc_labels:
            continue
        doc_cluster = doc_labels[paper.doc_id]
        for item in phrases.get(paper.doc_id, ()):
            cluster = phrase_cluster.get(item.phrase.casefold())
            if cluster is not None:
                counts[doc_cluster][cluster] += 1
    rows = []
    for doc_cluster, counter in sorted(counts.items()):
        total = sum(counter.values()) or 1
        for phrase_cluster_id, count in counter.most_common(5):
            rows.append(
                {
                    "document_cluster": int(doc_cluster),
                    "keyphrase_cluster": int(phrase_cluster_id),
                    "count": int(count),
                    "share": float(count / total),
                }
            )
    return rows
