"""SEM1 / LEXCORE / SEMCORE / ADAPT / SEMCTX ranking. No gold. Do not retune V6."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v4.shortlist import top_k_ids
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4_1.ranking import apply_v6_isolated, isolate_shortlist_ranking
from experiments.qa_chunk_selection.v4_2.features import zscore_map

from .contract import ALWAYS_TAU, NEVER_TAU


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


def b0_rank_index(b0_row: Mapping[str, Any], k: int) -> dict[int, int]:
    return {int(cid): index for index, cid in enumerate(top_k_ids(b0_row, k))}


def rank_ids_b0_tiebreak(ids: Sequence[int], scores: Mapping[int, float], b0_row: Mapping[str, Any], *, k: int) -> list[int]:
    order = b0_rank_index(b0_row, k)
    return sorted(
        [int(cid) for cid in ids],
        key=lambda cid: (-float(scores.get(int(cid), 0.0)), int(order.get(int(cid), 10**9)), int(cid)),
    )


def isolate_with_b0_tiebreak(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    shortlist_scores: Mapping[int, float],
    *,
    k: int,
) -> tuple[list[int], dict[int, float], bool]:
    topk = [int(cid) for cid in top_k_ids(b0_row, k)]
    if not topk:
        ranked, fused = isolate_shortlist_ranking(item, b0_row, shortlist_scores, k=k)
        return ranked, fused, False
    values = [float(shortlist_scores.get(cid, 0.0)) for cid in topk]
    top_score = max(values) if values else 0.0
    tied = sum(1 for value in values if abs(value - top_score) < 1e-12) >= 2
    ordered_short = rank_ids_b0_tiebreak(topk, shortlist_scores, b0_row, k=k)
    unique = {int(cid): float(len(ordered_short) - index) for index, cid in enumerate(ordered_short)}
    ranked, _order_fused = isolate_shortlist_ranking(item, b0_row, unique, k=k)
    fused: dict[int, float] = {int(cid): float(shortlist_scores.get(int(cid), 0.0)) for cid in ordered_short}
    floor = min(fused.values()) if fused else 0.0
    for offset, cid in enumerate([int(item_id) for item_id in ranked if int(item_id) not in fused], start=1):
        fused[int(cid)] = floor - float(offset)
    return ranked, fused, bool(tied)


def sem1_scores(v6_scores: Mapping[int, float], sem_rel: Mapping[int, float], *, delta: float) -> dict[int, float]:
    z_v6 = zscore_map(v6_scores)
    return {int(cid): float(z_v6.get(int(cid), 0.0)) + float(delta) * float(sem_rel.get(int(cid), 0.0)) for cid in v6_scores}


def lexcore_scores(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    k: int,
    min_overlap: float,
    lambda_m: float,
    lambda_t: float,
) -> dict[int, float]:
    feats = agreement_features(item, b0_row, qa_by_chunk, k=k, min_overlap=min_overlap)
    out: dict[int, float] = {}
    for cid in top_k_ids(b0_row, k):
        row = feats.get(int(cid)) or {}
        token = float(row.get("mean_token_f1") or 0.0) / 100.0
        out[int(cid)] = float(lambda_m) * float(row.get("majority") or 0.0) + float(lambda_t) * token
    return out


def semcore_scores(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    sem_rel: Mapping[int, float],
    *,
    k: int,
    min_overlap: float,
    lambda_m: float,
    lambda_s: float,
) -> dict[int, float]:
    feats = agreement_features(item, b0_row, qa_by_chunk, k=k, min_overlap=min_overlap)
    out: dict[int, float] = {}
    for cid in top_k_ids(b0_row, k):
        row = feats.get(int(cid)) or {}
        out[int(cid)] = float(lambda_m) * float(row.get("majority") or 0.0) + float(lambda_s) * float(sem_rel.get(int(cid), 0.0))
    return out


def semctx_scores(sem1_scores_map: Mapping[int, float], support_rel: Mapping[int, float], *, delta_c: float) -> dict[int, float]:
    z_sem1 = zscore_map(sem1_scores_map)
    return {int(cid): float(z_sem1.get(int(cid), 0.0)) + float(delta_c) * float(support_rel.get(int(cid), 0.0)) for cid in sem1_scores_map}


def routing_available(*, adapt_answers_ok: bool, selected_cid: int | None, best_triple: Sequence[int], consensus_conf) -> bool:
    if not adapt_answers_ok:
        return False
    if consensus_conf is None:
        return False
    if selected_cid is None:
        return False
    return int(selected_cid) in {int(cid) for cid in best_triple}


def adapt_use_semcore(*, confidence, tau: float, available: bool) -> bool:
    if not available:
        return False
    if tau == NEVER_TAU:
        return False
    if tau == ALWAYS_TAU:
        return True
    return float(confidence) >= float(tau)


def finite_tau(value: float) -> float:
    if value == NEVER_TAU:
        return 1e9
    if value == ALWAYS_TAU:
        return -1e9
    return float(value)
