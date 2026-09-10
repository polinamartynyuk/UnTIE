"""K-means + Gap Statistic. Hard assignment is the main FN filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class ClusterResult:
    status: str
    selected_k: int
    labels: tuple[int, ...]
    centers: np.ndarray | None
    gap_values: tuple[float, ...]
    gap_sk: tuple[float, ...]
    cluster_sizes: tuple[int, ...]


def _kmeans(data: np.ndarray, k: int, seed: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(data)


def _log_inertia(model: KMeans) -> float:
    return float(np.log(max(float(model.inertia_), 1e-12)))


def _reference_sample(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lows = data.min(axis=0)
    highs = data.max(axis=0)
    span = np.maximum(highs - lows, 1e-6)
    return lows + rng.random(data.shape) * span


def gap_statistic(
    data: np.ndarray,
    *,
    k_max: int,
    bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(data.shape[0])
    k_max = max(1, min(int(k_max), n))
    rng = np.random.default_rng(seed)
    gaps = np.zeros(k_max)
    sk = np.zeros(k_max)
    log_w = np.zeros(k_max)
    for k in range(1, k_max + 1):
        observed = _kmeans(data, k, seed)
        log_w[k - 1] = _log_inertia(observed)
        refs = []
        for draw in range(bootstrap):
            sample = _reference_sample(data, rng)
            refs.append(_log_inertia(_kmeans(sample, k, seed + draw + 1)))
        refs_arr = np.asarray(refs, dtype=float)
        gaps[k - 1] = float(refs_arr.mean() - log_w[k - 1])
        sk[k - 1] = float(refs_arr.std(ddof=1) * math_sqrt(1.0 + 1.0 / bootstrap)) if bootstrap > 1 else 0.0
    return gaps, sk, log_w


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def choose_k(gaps: np.ndarray, sk: np.ndarray) -> int:
    """Smallest k with Gap(k) >= Gap(k+1) - s_{k+1}; else argmax Gap."""
    if len(gaps) == 1:
        return 1
    for index in range(len(gaps) - 1):
        if gaps[index] >= gaps[index + 1] - sk[index + 1]:
            return index + 1
    return int(np.argmax(gaps) + 1)


def cluster_passages(
    embeddings: np.ndarray,
    *,
    k_max_cap: int,
    bootstrap: int,
    seed: int,
    min_points: int = 3,
) -> ClusterResult:
    data = np.asarray(embeddings, dtype=float)
    if data.ndim != 2 or data.shape[0] < min_points:
        n = int(data.shape[0]) if data.ndim == 2 else 0
        return ClusterResult(
            status="skipped_insufficient_data",
            selected_k=0,
            labels=tuple(0 for _ in range(n)),
            centers=None,
            gap_values=(),
            gap_sk=(),
            cluster_sizes=(),
        )
    k_max = min(k_max_cap, max(1, int(np.floor(np.sqrt(data.shape[0])))))
    k_max = max(1, min(k_max, data.shape[0]))
    gaps, sk, _ = gap_statistic(data, k_max=k_max, bootstrap=bootstrap, seed=seed)
    selected = choose_k(gaps, sk)
    if selected <= 1:
        return ClusterResult(
            status="no_cluster_structure",
            selected_k=1,
            labels=tuple(0 for _ in range(data.shape[0])),
            centers=None,
            gap_values=tuple(float(item) for item in gaps),
            gap_sk=tuple(float(item) for item in sk),
            cluster_sizes=(int(data.shape[0]),),
        )
    model = _kmeans(data, selected, seed)
    labels = tuple(int(item) for item in model.labels_)
    sizes = tuple(int(np.sum(model.labels_ == index)) for index in range(selected))
    return ClusterResult(
        status="applied",
        selected_k=selected,
        labels=labels,
        centers=np.asarray(model.cluster_centers_),
        gap_values=tuple(float(item) for item in gaps),
        gap_sk=tuple(float(item) for item in sk),
        cluster_sizes=sizes,
    )


def gumbel_labels(
    embeddings: np.ndarray,
    centers: np.ndarray,
    *,
    temperature: float,
    seed: int,
    runs: int = 1,
) -> np.ndarray:
    """Optional ablation only. No temperature decay."""
    sims = embeddings @ centers.T
    logits = sims - sims.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed)
    assigned = []
    for _ in range(max(1, runs)):
        gumbel = -np.log(-np.log(rng.uniform(size=probs.shape).clip(1e-12, 1 - 1e-12)))
        y = np.exp((np.log(probs.clip(1e-12, 1.0)) + gumbel) / max(temperature, 1e-6))
        y = y / y.sum(axis=1, keepdims=True)
        assigned.append(y.argmax(axis=1))
    return np.asarray(assigned[0], dtype=int)
