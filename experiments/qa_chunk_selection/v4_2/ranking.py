"""CORE and ATT fusion on an isolated B0 top-K shortlist. Do not retune V6."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v4.heads import rank_by_scores
from experiments.qa_chunk_selection.v4.shortlist import top_k_ids
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4_1.ranking import apply_v6_isolated, isolate_shortlist_ranking, v6_fused_score

from .contract import Z_EPS
from .features import matrix_at, zscore_map


def v6_shortlist_scores(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    k: int,
    min_overlap: float,
) -> tuple[list[int], dict[int, float], dict[int, dict[str, float]]]:
    return apply_v6_isolated(
        item,
        b0_row,
        qa_by_chunk,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        k=k,
        min_overlap=min_overlap,
    )


def core_shortlist_scores(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    k: int,
    min_overlap: float,
    variant: str,
    lambda_m: float,
    lambda_t: float = 0.0,
) -> dict[int, float]:
    feats = agreement_features(item, b0_row, qa_by_chunk, k=k, min_overlap=min_overlap)
    top = top_k_ids(b0_row, k)
    b0 = {int(cid): float((feats.get(cid) or {}).get("b0_score") or 0.0) for cid in top}
    z_b0 = zscore_map(b0)
    out: dict[int, float] = {}
    for cid in top:
        row = feats.get(int(cid)) or {}
        majority = float(row.get("majority") or 0.0)
        token = float(row.get("mean_token_f1") or 0.0) / 100.0
        if variant == "CORE1":
            out[int(cid)] = z_b0.get(int(cid), 0.0) + float(lambda_m) * majority
        elif variant == "CORE3":
            out[int(cid)] = float(lambda_m) * majority + float(lambda_t) * token
        else:
            out[int(cid)] = z_b0.get(int(cid), 0.0) + float(lambda_m) * majority + float(lambda_t) * token
    return out


def isolate(item: InstanceRecord, b0_row: Mapping[str, Any], shortlist_scores: Mapping[int, float], *, k: int):
    return isolate_shortlist_ranking(item, b0_row, shortlist_scores, k=k)


def head_ids(n_layers: int, n_heads: int) -> list[tuple[int, int]]:
    return [(layer, head) for layer in range(int(n_layers)) for head in range(int(n_heads))]


def bidirectional_shortlist(
    compact_by_chunk: Mapping[int, Mapping[str, Any]],
    selected: Sequence[tuple[int, int]],
) -> dict[int, float]:
    """Z-norm enrichment and span→Q density inside the current candidate set, then mean heads."""
    chunks = [int(cid) for cid in compact_by_chunk]
    if not chunks or not selected:
        return {cid: 0.0 for cid in chunks}
    per_head: dict[tuple[int, int], dict[int, float]] = {}
    for layer, head in selected:
        enrich = {cid: matrix_at(compact_by_chunk[cid], "q_to_span_enrichment", layer, head) for cid in chunks}
        dens = {cid: matrix_at(compact_by_chunk[cid], "span_to_q_density", layer, head) for cid in chunks}
        z_e = zscore_map(enrich)
        z_d = zscore_map(dens)
        per_head[(int(layer), int(head))] = {cid: 0.5 * (z_e[cid] + z_d[cid]) for cid in chunks}
    raw: dict[int, float] = {}
    n = max(len(selected), 1)
    for cid in chunks:
        raw[cid] = float(sum(per_head[key][cid] for key in per_head)) / float(n)
    return raw


def relative_attention(raw: Mapping[int, float]) -> dict[int, float]:
    return zscore_map(raw)


def att1_scores(v6_scores: Mapping[int, float], a_rel: Mapping[int, float], *, delta: float) -> dict[int, float]:
    z_v6 = zscore_map(v6_scores)
    return {int(cid): float(z_v6.get(int(cid), 0.0)) + float(delta) * float(a_rel.get(int(cid), 0.0)) for cid in v6_scores}


def att2_scores(v6_scores: Mapping[int, float], a_rel: Mapping[int, float], *, delta: float) -> dict[int, float]:
    return att1_scores(v6_scores, a_rel, delta=delta)


def att3_scores(v6_scores: Mapping[int, float], a_rel: Mapping[int, float], *, tau: float, lambda_v: float) -> dict[int, float]:
    z_v6 = zscore_map(v6_scores)
    out = {}
    for cid in v6_scores:
        penalty = float(lambda_v) * max(0.0, float(tau) - float(a_rel.get(int(cid), 0.0)))
        out[int(cid)] = float(z_v6.get(int(cid), 0.0)) - penalty
    return out


def ranking_from_shortlist(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    shortlist_scores: Mapping[int, float],
    *,
    k: int,
) -> tuple[list[int], dict[int, float]]:
    return isolate(item, b0_row, shortlist_scores, k=k)


def v6_only_formula_unchanged() -> dict[str, float]:
    sample = {"b0_score": 1.0, "exact_freq": 2.0, "mean_token_f1": 80.0, "majority": 1.0, "source_clusters": 3.0}
    return {"score": v6_fused_score(sample, alpha=2.0, beta=0.25, gamma=0.5), "eps": Z_EPS}


def rank_ids(ids: Sequence[int], scores: Mapping[int, float]) -> list[int]:
    return rank_by_scores(list(ids), [float(scores.get(int(cid), 0.0)) for cid in ids])
