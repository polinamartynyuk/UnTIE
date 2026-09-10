"""Hard/soft routing of held-out documents to frozen train centroids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RoutingResult:
    doc_id: str
    split: str
    top1: int
    top1_similarity: float
    top_ids: tuple[int, ...]
    similarities: tuple[float, ...]
    weights: tuple[float, ...]


def route_matrix(
    vectors: np.ndarray,
    centroids: np.ndarray,
    doc_ids: list[str],
    splits: list[str],
    *,
    top_n: int = 3,
    temperature: float = 0.15,
) -> list[RoutingResult]:
    if len(vectors) == 0:
        return []
    sims = cosine_similarity(vectors, centroids)
    results: list[RoutingResult] = []
    k = sims.shape[1]
    take = min(top_n, k)
    for row, doc_id, split in zip(sims, doc_ids, splits):
        order = np.argsort(-row)
        chosen = order[:take]
        scores = row[chosen]
        scaled = np.exp(scores / max(temperature, 1e-6))
        weights = scaled / scaled.sum()
        results.append(
            RoutingResult(
                doc_id=str(doc_id),
                split=str(split),
                top1=int(order[0]),
                top1_similarity=float(row[order[0]]),
                top_ids=tuple(int(index) for index in chosen),
                similarities=tuple(float(value) for value in scores),
                weights=tuple(float(value) for value in weights),
            )
        )
    return results
