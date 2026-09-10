"""V1: KeyBERT document compatibility rerank of B0 top-K."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord

from .config import FUSION_LAMBDAS
from .heads import LinearHead, rank_by_scores
from .keybert_features import compatibility_features
from .shortlist import scores_from_row, top_k_ids


VARIANTS = ("V1-A0", "V1-A1", "V1-A2", "V1-A3", "V1-A4", "V1-A5", "V1-A6")
MODE = {
    "V1-A0": None,
    "V1-A1": "global",
    "V1-A2": "loco",
    "V1-A3": "losco",
    "V1-A4": "losco",
    "V1-A5": "losco",
    "V1-A6": "losco",
}


def doc_signal(variant: str, features: Mapping[str, float]) -> float:
    if variant == "V1-A1":
        return float(features.get("s_overlap") or 0.0)
    if variant == "V1-A2":
        return float(features.get("s_overlap") or 0.0)
    if variant == "V1-A3":
        return 0.5 * float(features.get("s_overlap") or 0.0) + 0.5 * float(features.get("s_embedding") or 0.0)
    if variant == "V1-A4":
        return float(features.get("s_support") or 0.0)
    if variant == "V1-A5":
        return float(features.get("s_idf") or 0.0)
    if variant == "V1-A6":
        return float(features.get("s_aspect") or 0.0)
    return 0.0


def candidate_features(
    item: InstanceRecord,
    chunk_id: int,
    phrases_by_chunk: Mapping[int, Sequence[Mapping[str, Any]]],
    embeddings: Mapping[str, Sequence[float]],
    idf: Mapping[str, float],
    *,
    min_overlap: float,
    mode: str,
    aspect_embedding: Sequence[float] | None,
    b0_score: float,
    b0_rank: int,
) -> dict[str, float]:
    feats = compatibility_features(
        item,
        chunk_id,
        phrases_by_chunk,
        embeddings,
        idf,
        min_overlap=min_overlap,
        mode=mode,
        aspect_embedding=aspect_embedding,
    )
    chunk = item.chunk_by_id(int(chunk_id))
    n_chunks = max(len(item.chunks), 1)
    feats.update(
        {
            "b0_score": float(b0_score),
            "b0_rank": float(b0_rank),
            "candidate_length": float(len(chunk.text)),
            "relative_position": float(chunk.position) / float(n_chunks),
        }
    )
    return feats


def rerank_variant(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    *,
    variant: str,
    k: int,
    lam: float,
    phrases_by_chunk: Mapping[int, Sequence[Mapping[str, Any]]],
    embeddings: Mapping[str, Sequence[float]],
    idf: Mapping[str, float],
    min_overlap: float,
    aspect_embedding: Sequence[float] | None = None,
    head: LinearHead | None = None,
) -> tuple[list[int], dict[int, float], dict[int, dict[str, float]]]:
    ranked = top_k_ids(b0_row, k)
    b0_scores = scores_from_row(b0_row)
    if variant == "V1-A0" or not ranked:
        full = {int(chunk.chunk_id): float(b0_scores.get(int(chunk.chunk_id), 0.0)) for chunk in item.chunks}
        return ranked_full(item, full), full, {}
    mode = MODE[variant]
    feat_map: dict[int, dict[str, float]] = {}
    fused: dict[int, float] = {}
    for rank, cid in enumerate(ranked, start=1):
        feats = candidate_features(
            item,
            cid,
            phrases_by_chunk,
            embeddings,
            idf,
            min_overlap=min_overlap,
            mode=mode or "losco",
            aspect_embedding=aspect_embedding,
            b0_score=float(b0_scores.get(int(cid), 0.0)),
            b0_rank=rank,
        )
        feat_map[int(cid)] = feats
        if head is not None:
            names = head.feature_names or list(feats)
            fused[int(cid)] = head.score([float(feats.get(name, 0.0)) for name in names])
        else:
            fused[int(cid)] = float(b0_scores.get(int(cid), 0.0)) + float(lam) * doc_signal(variant, feats)
    for chunk in item.chunks:
        fused.setdefault(int(chunk.chunk_id), float(b0_scores.get(int(chunk.chunk_id), -1e9)))
    return ranked_full(item, fused), fused, feat_map


def ranked_full(item: InstanceRecord, scores: Mapping[int, float]) -> list[int]:
    ids = [int(chunk.chunk_id) for chunk in item.chunks]
    vals = [float(scores.get(cid, 0.0)) for cid in ids]
    return rank_by_scores(ids, vals)


def choose_lambda(
    cal_items: Sequence[InstanceRecord],
    b0_by_qid: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
    k: int,
    feature_fn,
) -> float:
    best_lam = 0.5
    best = (-1.0, -1.0)
    for lam in FUSION_LAMBDAS:
        hits = 0
        f1s = 0.0
        n = 0
        for item in cal_items:
            row = b0_by_qid.get(item.question_id)
            if row is None:
                continue
            ranked, _scores, _feats = rerank_variant(item, row, variant=variant, k=k, lam=lam, **feature_fn(item))
            n += 1
            pos = {int(chunk.chunk_id) for chunk in item.positives}
            hits += int(bool(ranked and ranked[0] in pos))
            f1s += float(row.get("token_f1") or 0.0) if ranked and ranked[0] == int(row.get("selected_chunk_id") or -1) else 0.0
        score = (hits / max(n, 1), f1s / max(n, 1))
        if score > best:
            best = score
            best_lam = float(lam)
    return best_lam
