"""Clustering algorithms, internal metrics, stability. Train vectors only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from .encoder import l2_normalize


ALGORITHMS = ("BisectingKMeans", "KMeans", "MiniBatchKMeans")


def _estimator(name: str, k: int, seed: int):
    if name == "BisectingKMeans":
        return BisectingKMeans(n_clusters=k, random_state=seed, n_init=1)
    if name == "KMeans":
        return KMeans(n_clusters=k, random_state=seed, n_init=10)
    if name == "MiniBatchKMeans":
        return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=32)
    raise ValueError(f"unsupported clustering algorithm: {name}")


@dataclass(frozen=True)
class ClusterModel:
    algorithm: str
    k: int
    seed: int
    labels: np.ndarray
    centroids: np.ndarray
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    size_min: int
    size_max: int
    size_mean: float
    size_median: float

    def sizes(self) -> dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(cluster): int(count) for cluster, count in zip(unique, counts)}


def fit_clusters(train_vectors: np.ndarray, *, algorithm: str, k: int, seed: int) -> ClusterModel:
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(train_vectors) < k:
        raise ValueError("not enough train documents for requested k")
    model = _estimator(algorithm, k, seed)
    labels = np.asarray(model.fit_predict(train_vectors))
    centroids = []
    for cluster in range(k):
        members = train_vectors[labels == cluster]
        if len(members) == 0:
            centroids.append(np.zeros(train_vectors.shape[1], dtype=np.float32))
        else:
            centroids.append(members.mean(axis=0))
    centroids_arr = l2_normalize(np.vstack(centroids).astype(np.float32))
    sizes = np.bincount(labels, minlength=k)
    sil = float("nan")
    if len(set(labels.tolist())) > 1 and min(sizes[sizes > 0]) > 1:
        sil = float(silhouette_score(train_vectors, labels, metric="cosine"))
    db = float(davies_bouldin_score(train_vectors, labels))
    ch = float(calinski_harabasz_score(train_vectors, labels))
    return ClusterModel(
        algorithm=algorithm,
        k=k,
        seed=seed,
        labels=labels,
        centroids=centroids_arr,
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        size_min=int(sizes[sizes > 0].min()),
        size_max=int(sizes.max()),
        size_mean=float(sizes.mean()),
        size_median=float(np.median(sizes)),
    )


def stability_scores(
    train_vectors: np.ndarray,
    *,
    algorithm: str,
    k: int,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, float]:
    label_sets = [
        fit_clusters(train_vectors, algorithm=algorithm, k=k, seed=seed).labels
        for seed in seeds
    ]
    aris: list[float] = []
    nmis: list[float] = []
    for left in range(len(label_sets)):
        for right in range(left + 1, len(label_sets)):
            aris.append(float(adjusted_rand_score(label_sets[left], label_sets[right])))
            nmis.append(
                float(normalized_mutual_info_score(label_sets[left], label_sets[right]))
            )
    return {
        "mean_ari": float(np.mean(aris)) if aris else float("nan"),
        "mean_nmi": float(np.mean(nmis)) if nmis else float("nan"),
        "runs": float(len(seeds)),
    }
