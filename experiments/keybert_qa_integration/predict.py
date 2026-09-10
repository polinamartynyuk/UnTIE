"""Build chunk scores and pick an answer from cached extractive QA evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from experiments.keybert_topic_clustering.keyphrases import Keyphrase
from untie.domain import Answer
from untie.keyword_evidence import CachedDocumentEvidence, _rebuild_chunks, select_cached_answer
from untie.ranking import WeightedKeyword

from .config import ExperimentParams, ModeFlags
from .scoring import (
    aligned_keyword_scores,
    apply_global_local,
    cluster_chunk_scores,
    cluster_phrase_labels,
    cluster_relevance,
    consensus_scores,
    keybert_chunk_scores,
    max_phrase_similarities,
    mix_named_channels,
    reweight_attention,
    select_channel_plan,
)


@dataclass(frozen=True)
class DocumentFeatures:
    evidence: CachedDocumentEvidence
    phrases: tuple[Keyphrase, ...]
    phrase_embeddings: np.ndarray
    chunk_embeddings: np.ndarray
    attention_keywords: tuple[WeightedKeyword, ...]
    attention_embeddings: np.ndarray
    chunk_keybert_scores: np.ndarray | None = None


def _slice_keybert(
    features: DocumentFeatures,
    params: ExperimentParams,
) -> tuple[tuple[Keyphrase, ...], np.ndarray, np.ndarray]:
    top_k = max(1, int(params.keybert_top_k))
    phrases = features.phrases[:top_k]
    embeddings = features.phrase_embeddings[:top_k] if features.phrase_embeddings.size else features.phrase_embeddings
    weights = np.asarray([float(item.weight) for item in phrases], dtype=np.float32)
    return phrases, embeddings, weights


def prepare_attention_keywords(
    features: DocumentFeatures,
    flags: ModeFlags,
    params: ExperimentParams,
    *,
    static_keywords: Sequence[WeightedKeyword],
    global_frequency: Mapping[str, float],
) -> list[WeightedKeyword]:
    if flags.attention_source == "static":
        keywords = list(static_keywords)
    else:
        keywords = list(features.attention_keywords)
    if not keywords:
        return []
    need_sim = flags.use_keybert_filter or flags.use_global_local or flags.use_keybert_clusters
    similarities: list[float] = [0.0] * len(keywords)
    embeddings = features.attention_embeddings
    if need_sim and embeddings.size and embeddings.shape[0] == len(keywords):
        _, phrase_emb, _ = _slice_keybert(features, params)
        similarities = [float(value) for value in max_phrase_similarities(embeddings, phrase_emb)]
    if flags.use_keybert_filter:
        keywords = reweight_attention(
            keywords,
            similarities,
            mode=flags.filter_mode,
            threshold=params.keybert_sim_threshold,
            floor=params.filter_floor,
        )
    if flags.use_global_local:
        keywords = apply_global_local(
            keywords,
            similarities,
            global_frequency,
            alpha=params.global_alpha,
            beta=params.local_beta,
        )
    return keywords


def keybert_channel(
    features: DocumentFeatures,
    flags: ModeFlags,
    params: ExperimentParams,
    *,
    attention_keywords: Sequence[WeightedKeyword],
    seed: int = 42,
) -> list[float]:
    chunks = _rebuild_chunks(features.evidence)
    if flags.keybert_scope == "chunk" and features.chunk_keybert_scores is not None:
        scores = np.asarray(features.chunk_keybert_scores, dtype=float)
        if len(scores) == len(chunks):
            return [float(value) for value in scores]
    phrases, phrase_emb, phrase_weights = _slice_keybert(features, params)
    if flags.use_keybert_clusters:
        _labels, centroids = cluster_phrase_labels(
            phrase_emb,
            cluster_k=params.cluster_k,
            seed=seed,
        )
        att_emb = features.attention_embeddings
        att_weights = [float(item.attention_weight) for item in attention_keywords]
        if att_emb.size == 0 or att_emb.shape[0] != len(att_weights):
            att_emb = phrase_emb
            att_weights = [float(item.weight) for item in phrases] or [1.0]
        selected = cluster_relevance(
            att_emb,
            att_weights,
            centroids,
            top_n=params.cluster_top_n,
            sim_threshold=params.cluster_sim_threshold,
        )
        return [float(value) for value in cluster_chunk_scores(features.chunk_embeddings, selected)]
    raw = keybert_chunk_scores(
        features.chunk_embeddings,
        phrase_emb,
        phrase_weights,
        variant=params.keybert_topic_variant,
        mean_top_n=params.keybert_mean_top_n,
    )
    return [float(value) for value in raw]


def chunk_channel_bundle(
    features: DocumentFeatures,
    flags: ModeFlags,
    params: ExperimentParams,
    *,
    static_keywords: Sequence[WeightedKeyword],
    global_frequency: Mapping[str, float],
    seed: int = 42,
) -> dict[str, list[float]]:
    chunks = _rebuild_chunks(features.evidence)
    qa_conf = [float(item.confidence) for item in features.evidence.chunks]
    keywords = prepare_attention_keywords(
        features,
        flags,
        params,
        static_keywords=static_keywords,
        global_frequency=global_frequency,
    )
    attention_for_score = keywords
    att_scores = aligned_keyword_scores(
        chunks,
        attention_for_score,
        weight_ratio=1.0,
        keep_unmatched=params.keep_unmatched,
    )
    diff_scores = aligned_keyword_scores(
        chunks,
        list(static_keywords),
        weight_ratio=0.0,
        keep_unmatched=params.keep_unmatched,
    )
    bundled = aligned_keyword_scores(
        chunks,
        list(static_keywords),
        weight_ratio=params.static_weight_ratio,
        keep_unmatched=params.keep_unmatched,
    )
    keybert_scores = keybert_channel(
        features,
        flags,
        params,
        attention_keywords=attention_for_score or list(features.attention_keywords),
        seed=seed,
    )
    if len(keybert_scores) != len(chunks):
        keybert_scores = [0.0] * len(chunks)
    return {
        "qa": qa_conf,
        "attention": att_scores,
        "diff": diff_scores,
        "bundled": bundled,
        "keybert": keybert_scores,
    }


def mixed_chunk_scores(
    bundle: Mapping[str, Sequence[float]],
    flags: ModeFlags,
    params: ExperimentParams,
) -> list[float]:
    plan = select_channel_plan(flags, params)
    return mix_named_channels(bundle, plan)


def predict_document(
    features: DocumentFeatures,
    flags: ModeFlags,
    params: ExperimentParams,
    *,
    static_keywords: Sequence[WeightedKeyword],
    global_frequency: Mapping[str, float],
    seed: int = 42,
) -> tuple[str | None, dict[str, object]]:
    evidence = features.evidence
    if flags.is_baseline() or not evidence.chunks:
        return evidence.baseline_answer, {
            "fallback": evidence.baseline_answer is None,
            "matched_chunks": len(evidence.chunks),
            "selection": "baseline_aggregator",
            "keyword_count": 0,
        }
    chunks = _rebuild_chunks(evidence)
    bundle = chunk_channel_bundle(
        features,
        flags,
        params,
        static_keywords=static_keywords,
        global_frequency=global_frequency,
        seed=seed,
    )
    if flags.use_answer_consensus:
        scores = consensus_scores(
            bundle["qa"],
            bundle["attention"],
            bundle["keybert"],
            qa_weight=params.consensus_qa_weight,
            attention_weight=params.consensus_attention_weight,
            keybert_weight=params.consensus_keybert_weight,
        )
    else:
        scores = mixed_chunk_scores(bundle, flags, params)
    answers: list[Answer] = []
    for item, chunk, score in zip(evidence.chunks, chunks, scores):
        answers.append(
            Answer(
                text=item.answer,
                chunk=chunk,
                confidence=item.confidence,
                start_pos=item.start_pos,
                end_pos=item.end_pos,
                metadata={"chunk_score": float(score), "chunk_index": item.index},
            )
        )
    similarity = np.asarray(evidence.answer_similarity, dtype=float)
    selected = select_cached_answer(
        answers,
        similarity,
        similarity_threshold=params.similarity_threshold,
        cluster_strategy=params.cluster_strategy,
        answer_strategy=params.answer_strategy,
    )
    fallback = selected is None
    prediction = None if selected is None else selected.text
    if prediction is None:
        prediction = evidence.baseline_answer
        fallback = True
    keywords = prepare_attention_keywords(
        features,
        flags,
        params,
        static_keywords=static_keywords,
        global_frequency=global_frequency,
    )
    return prediction, {
        "fallback": fallback,
        "matched_chunks": len(answers),
        "selection": "answer_consensus" if flags.use_answer_consensus else "select_clustered",
        "keyword_count": len(keywords),
        "mean_chunk_score": float(np.mean(scores)) if scores else 0.0,
        "selected_chunk_index": None
        if selected is None
        else int(selected.metadata.get("chunk_index", -1)),
    }
