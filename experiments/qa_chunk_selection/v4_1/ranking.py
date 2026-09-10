"""Shortlist-isolated ranking: outside B0 top-K never wins argmax."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v4.heads import rank_by_scores
from experiments.qa_chunk_selection.v4.shortlist import ranked_ids_from_row, scores_from_row, top_k_ids
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features


def v6_block_scores(features: Mapping[str, float]) -> dict[str, float]:
    exact = float(features.get("exact_freq") or 0.0)
    token = float(features.get("mean_token_f1") or 0.0)
    majority = float(features.get("majority") or 0.0)
    support = float(features.get("source_clusters") or 0.0)
    return {
        "exact_freq": exact,
        "mean_token_f1": token,
        "majority": majority,
        "source_clusters": support,
        "S_answer": exact + token + majority,
        "S_support": support,
        "b0_score": float(features.get("b0_score") or 0.0),
    }


def v6_fused_score(
    features: Mapping[str, float],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    knockout: str | None = None,
) -> float:
    blocks = v6_block_scores(features)
    exact = 0.0 if knockout == "A1" else blocks["exact_freq"]
    token = 0.0 if knockout == "A2" else blocks["mean_token_f1"]
    majority = 0.0 if knockout == "A3" else blocks["majority"]
    support = 0.0 if knockout in {"A4", "A5"} else blocks["source_clusters"]
    answer = exact + token + majority
    if knockout == "A6":
        answer = 0.0
    a = 0.0 if knockout == "A7" else float(alpha)
    b = 0.0 if knockout == "A6" else float(beta)
    g = 0.0 if knockout in {"A4", "A5"} else float(gamma)
    if knockout == "A0":
        return blocks["b0_score"]
    # A4 (γ=0) and A5 (B0 + agreement, no support) are the same at frozen weights.
    return a * blocks["b0_score"] + b * answer + g * support


def isolate_shortlist_ranking(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    shortlist_scores: Mapping[int, float],
    *,
    k: int,
) -> tuple[list[int], dict[int, float]]:
    topk = [int(cid) for cid in top_k_ids(b0_row, k)]
    top_set = set(topk)
    if not topk:
        b0 = scores_from_row(b0_row)
        ids = [int(chunk.chunk_id) for chunk in item.chunks]
        return rank_by_scores(ids, [float(b0.get(cid, 0.0)) for cid in ids]), dict(b0)
    ordered_short = rank_by_scores(topk, [float(shortlist_scores.get(cid, 0.0)) for cid in topk])
    rest = [int(cid) for cid in ranked_ids_from_row(b0_row) if int(cid) not in top_set]
    seen = set(ordered_short) | set(rest)
    rest.extend(int(chunk.chunk_id) for chunk in item.chunks if int(chunk.chunk_id) not in seen)
    ranked = ordered_short + rest
    fused = {int(cid): float(shortlist_scores.get(cid, 0.0)) for cid in ordered_short}
    floor = min(fused.values()) if fused else 0.0
    for offset, cid in enumerate(rest, start=1):
        fused[int(cid)] = floor - float(offset)
    return ranked, fused


def shortlist_set_ok(b0_row: Mapping[str, Any], ranked_ids: Sequence[int], *, k: int) -> bool:
    expected = {int(cid) for cid in top_k_ids(b0_row, k)}
    got = {int(cid) for cid in list(ranked_ids)[: max(int(k), 0)]}
    return expected == got


def apply_v6_isolated(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    k: int,
    min_overlap: float,
    knockout: str | None = None,
) -> tuple[list[int], dict[int, float], dict[int, dict[str, float]]]:
    feats = agreement_features(item, b0_row, qa_by_chunk, k=k, min_overlap=min_overlap)
    short_scores = {
        int(cid): v6_fused_score(row, alpha=alpha, beta=beta, gamma=gamma, knockout=knockout)
        for cid, row in feats.items()
    }
    ranked, scores = isolate_shortlist_ranking(item, b0_row, short_scores, k=k)
    return ranked, scores, feats
