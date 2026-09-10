"""KeyBERT topic scores, keyword filters, mixes, and cluster priors.

Every numeric coefficient is passed in; nothing experimental is hardcoded.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from untie.ranking import WeightedKeyword, mix_score_channels, score_chunks

from .config import ExperimentParams, ModeFlags


def aligned_keyword_scores(
    chunks,
    keywords: Sequence[WeightedKeyword],
    *,
    weight_ratio: float,
    keep_unmatched: bool,
) -> list[float]:
    scored = score_chunks(
        list(chunks),
        list(keywords),
        weight_ratio=weight_ratio,
        keep_unmatched=keep_unmatched,
    )
    by_index = {item.chunk.start_sentence: float(item.score) for item in scored}
    return [float(by_index.get(chunk.start_sentence, 0.0)) for chunk in chunks]


def keybert_chunk_scores(
    chunk_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
    phrase_weights: Sequence[float],
    *,
    variant: str,
    mean_top_n: int,
) -> np.ndarray:
    """Thematic prior for each chunk vs document (or chunk) KeyBERT phrases."""
    n_chunks = int(chunk_embeddings.shape[0]) if chunk_embeddings.ndim == 2 else 0
    if n_chunks == 0:
        return np.zeros((0,), dtype=np.float32)
    if phrase_embeddings.size == 0 or phrase_embeddings.shape[0] == 0:
        return np.zeros((n_chunks,), dtype=np.float32)
    similarity = chunk_embeddings @ phrase_embeddings.T
    name = str(variant)
    if name == "max":
        return similarity.max(axis=1).astype(np.float32)
    if name == "mean_top_n":
        width = similarity.shape[1]
        take = max(1, min(int(mean_top_n), width))
        partitioned = np.partition(similarity, kth=width - take, axis=1)[:, width - take :]
        return partitioned.mean(axis=1).astype(np.float32)
    if name == "weighted":
        weights = np.asarray([max(float(value), 0.0) for value in phrase_weights], dtype=np.float32)
        if weights.size != similarity.shape[1]:
            raise ValueError("phrase_weights length must match the number of phrases")
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights)
        weights = weights / float(weights.sum())
        return (similarity @ weights).astype(np.float32)
    raise ValueError(f"unknown KeyBERT topic variant: {variant}")


def max_phrase_similarities(
    item_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
) -> np.ndarray:
    """Max cosine of each row to any KeyBERT phrase. Empty phrases → zeros."""
    if item_embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32)
    n_items = int(item_embeddings.shape[0])
    if phrase_embeddings.size == 0 or phrase_embeddings.shape[0] == 0:
        return np.zeros((n_items,), dtype=np.float32)
    similarity = item_embeddings @ phrase_embeddings.T
    return np.clip(similarity.max(axis=1), 0.0, 1.0).astype(np.float32)


def reweight_attention(
    keywords: Sequence[WeightedKeyword],
    similarities: Sequence[float],
    *,
    mode: str,
    threshold: float,
    floor: float,
) -> list[WeightedKeyword]:
    """Soft thematic filter. Never deletes a keyword just because it is missing from KeyBERT."""
    if mode not in {"none", "soft", "hard_threshold", "semantic"}:
        raise ValueError(f"unknown filter mode: {mode}")
    if floor <= 0.0:
        raise ValueError("filter floor must be positive so keywords are not zeroed")
    if len(keywords) != len(similarities):
        raise ValueError("keywords and similarities must have the same length")
    updated: list[WeightedKeyword] = []
    for keyword, raw_sim in zip(keywords, similarities):
        sim = float(np.clip(raw_sim, 0.0, 1.0))
        attention = float(keyword.attention_weight)
        if mode in {"soft", "semantic"}:
            attention = attention * (0.5 + 0.5 * sim)
        elif mode == "hard_threshold":
            if sim < threshold:
                attention = attention * floor
        updated.append(replace(keyword, attention_weight=attention))
    return updated


def apply_global_local(
    keywords: Sequence[WeightedKeyword],
    local_scores: Sequence[float],
    global_frequency: Mapping[str, float],
    *,
    alpha: float,
    beta: float,
) -> list[WeightedKeyword]:
    """keyword_score = att * (1 + α * global) * (1 + β * local). Train-only frequencies."""
    if len(keywords) != len(local_scores):
        raise ValueError("keywords and local_scores must have the same length")
    updated: list[WeightedKeyword] = []
    for keyword, local in zip(keywords, local_scores):
        key = keyword.lemma.casefold() or keyword.word.casefold()
        global_score = float(np.clip(global_frequency.get(key, 0.0), 0.0, 1.0))
        local_score = float(np.clip(local, 0.0, 1.0))
        attention = float(keyword.attention_weight) * (1.0 + alpha * global_score) * (
            1.0 + beta * local_score
        )
        updated.append(replace(keyword, attention_weight=attention))
    return updated


def cluster_phrase_labels(
    phrase_embeddings: np.ndarray,
    *,
    cluster_k: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster KeyBERT phrase embeddings. Returns (labels, centroids)."""
    if phrase_embeddings.size == 0 or phrase_embeddings.shape[0] == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 0), dtype=np.float32)
    n_phrases = int(phrase_embeddings.shape[0])
    k = max(1, min(int(cluster_k), n_phrases))
    if k == 1 or n_phrases == 1:
        centroid = phrase_embeddings.mean(axis=0, keepdims=True)
        return np.zeros((n_phrases,), dtype=int), centroid.astype(np.float32)
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = np.asarray(model.fit_predict(phrase_embeddings), dtype=int)
    centroids = np.asarray(model.cluster_centers_, dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)
    return labels, centroids


def cluster_relevance(
    attention_embeddings: np.ndarray,
    attention_weights: Sequence[float],
    centroids: np.ndarray,
    *,
    top_n: int,
    sim_threshold: float,
) -> np.ndarray:
    """Weight each phrase cluster by attention mass, count, and semantic similarity."""
    if centroids.size == 0 or centroids.shape[0] == 0:
        return np.zeros((0, centroids.shape[1] if centroids.ndim == 2 else 0), dtype=np.float32)
    n_clusters = int(centroids.shape[0])
    if attention_embeddings.size == 0 or attention_embeddings.shape[0] == 0:
        return centroids[:0]
    similarity = attention_embeddings @ centroids.T
    assigned = similarity.argmax(axis=1)
    scores = np.zeros((n_clusters,), dtype=np.float64)
    weights = np.asarray(list(attention_weights), dtype=np.float64)
    for cluster_id in range(n_clusters):
        mask = assigned == cluster_id
        count = int(mask.sum())
        if count == 0:
            continue
        member_sim = similarity[mask, cluster_id]
        mean_sim = float(np.clip(member_sim.mean(), 0.0, 1.0))
        if mean_sim < sim_threshold and count < 2:
            continue
        scores[cluster_id] = count * float(weights[mask].sum()) * (mean_sim + 1e-6)
    if float(scores.max()) <= 0.0:
        order = np.argsort(-np.linalg.norm(centroids, axis=1))[: max(1, top_n)]
        return centroids[order]
    order = np.argsort(-scores)[: max(1, min(int(top_n), n_clusters))]
    return centroids[order]


def cluster_chunk_scores(
    chunk_embeddings: np.ndarray,
    selected_centroids: np.ndarray,
) -> np.ndarray:
    if chunk_embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32)
    n_chunks = int(chunk_embeddings.shape[0])
    if selected_centroids.size == 0 or selected_centroids.shape[0] == 0:
        return np.zeros((n_chunks,), dtype=np.float32)
    similarity = chunk_embeddings @ selected_centroids.T
    return similarity.max(axis=1).astype(np.float32)


def mix_named_channels(
    channels: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
) -> list[float]:
    """Minmax each named channel, then weighted sum. Zero-weight channels are dropped."""
    active_channels: list[list[float]] = []
    active_weights: list[float] = []
    length: int | None = None
    for name, values in channels.items():
        scale = float(weights.get(name, 0.0))
        series = [float(value) for value in values]
        if length is None:
            length = len(series)
        elif len(series) != length:
            raise ValueError("score channels must have the same length")
        if scale == 0.0:
            continue
        active_channels.append(series)
        active_weights.append(scale)
    if not active_channels:
        return [0.0] * (length or 0)
    return mix_score_channels(active_channels, active_weights)


def consensus_scores(
    qa_confidence: Sequence[float],
    attention_scores: Sequence[float],
    keybert_scores: Sequence[float],
    *,
    qa_weight: float,
    attention_weight: float,
    keybert_weight: float,
) -> list[float]:
    return mix_named_channels(
        {
            "qa": qa_confidence,
            "attention": attention_scores,
            "keybert": keybert_scores,
        },
        {
            "qa": qa_weight,
            "attention": attention_weight,
            "keybert": keybert_weight,
        },
    )


def select_channel_plan(
    flags: ModeFlags,
    params: ExperimentParams,
) -> dict[str, float]:
    """Which mix weights apply for this flag combination."""
    weights = {
        "attention": 0.0,
        "diff": 0.0,
        "bundled": 0.0,
        "keybert": 0.0,
        "qa": 0.0,
    }
    independent = bool(flags.mix_diff_independent)
    bundled = flags.use_attention and flags.use_diff and not independent
    if bundled:
        weights["bundled"] = float(params.attention_weight)
    else:
        if flags.use_attention:
            weights["attention"] = float(params.attention_weight)
        if flags.use_diff:
            weights["diff"] = float(params.diff_weight)
    if flags.use_keybert_chunk_score or flags.use_keybert_clusters:
        weights["keybert"] = float(params.keybert_weight)
    if flags.use_answer_consensus:
        weights["qa"] = float(params.qa_weight if params.qa_weight else params.consensus_qa_weight)
    return weights
