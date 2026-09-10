"""CAL weight / lambda / eta selection with HN safeguard. Never peeks at EVAL."""

from __future__ import annotations

from itertools import product
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v4.shortlist import top_k_ids

from .evaluate import paired_delta, summarize_rows
from .ranking import apply_v6_isolated


def _qa_map(item: InstanceRecord, b0_row: Mapping[str, Any], qa_store, k: int) -> dict[int, dict[str, Any]]:
    return {
        cid: qa_store[(item.question_id, cid)]
        for cid in top_k_ids(b0_row, k)
        if (item.question_id, cid) in qa_store
    }


def mean_token_f1_across_seeds(rows: Sequence[Mapping[str, Any]]) -> float:
    by_q: dict[str, list[float]] = {}
    for row in rows:
        by_q.setdefault(str(row["question_id"]), []).append(float(row.get("token_f1") or 0.0))
    if not by_q:
        return 0.0
    return float(mean(mean(values) for values in by_q.values()))


def choose_v6_weights(
    cal_items: Sequence[InstanceRecord],
    b0_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    qa_store,
    b0_cal_rows: Sequence[Mapping[str, Any]],
    *,
    k: int,
    min_overlap: float,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
    gamma_grid: Sequence[float],
    epsilon_hn: float,
    predict_fn: Callable[..., dict[str, Any]],
    seeds: Sequence[int],
    progress=None,
) -> dict[str, Any]:
    b0_hn = float((summarize_rows(list(b0_cal_rows)).get("hn@1") or 0.0))
    ranked: list[dict[str, Any]] = []
    grid = list(product(alpha_grid, beta_grid, gamma_grid))
    for index, (alpha, beta, gamma) in enumerate(grid, start=1):
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            b0_map = b0_by_seed[int(seed)]
            for item in cal_items:
                row = b0_map.get(item.question_id)
                if row is None:
                    continue
                qa_by_chunk = _qa_map(item, row, qa_store, k)
                ranked_ids, scores, _feats = apply_v6_isolated(
                    item, row, qa_by_chunk, alpha=float(alpha), beta=float(beta), gamma=float(gamma), k=k, min_overlap=min_overlap
                )
                pred = predict_fn(item, scores, qa_by_chunk, ranked_ids)
                pred["seed"] = int(seed)
                rows.append(pred)
        summary = summarize_rows(rows)
        hn = float(summary.get("hn@1") or 0.0)
        admissible = hn <= b0_hn + float(epsilon_hn)
        f1 = mean_token_f1_across_seeds(rows)
        hit = float(summary.get("hit@1") or 0.0)
        ranked.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "cal_token_f1": f1,
                "cal_hn@1": hn,
                "cal_hit@1": hit,
                "admissible": admissible,
                "reader_mass": float(beta) + float(gamma),
            }
        )
        if progress:
            progress.count(index, len(grid), item="v6_grid")
    admissible_rows = [row for row in ranked if row["admissible"]] or ranked
    admissible_rows.sort(
        key=lambda row: (
            -float(row["cal_token_f1"]),
            float(row["cal_hn@1"]),
            -float(row["cal_hit@1"]),
            float(row["reader_mass"]),
            float(row["alpha"]),
            float(row["beta"]),
            float(row["gamma"]),
        )
    )
    best = admissible_rows[0]
    return {"selected": best, "b0_cal_hn@1": b0_hn, "n_grid": len(ranked), "tried": ranked}


def choose_lambda_agg(
    cal_items: Sequence[InstanceRecord],
    b0_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    qa_store,
    *,
    weights: Mapping[str, float],
    k: int,
    min_overlap: float,
    grid: Sequence[float],
    aggregate_fn: Callable[..., dict[str, Any]],
    seeds: Sequence[int],
    progress=None,
) -> dict[str, Any]:
    best = None
    tried = []
    for index, lam in enumerate(grid, start=1):
        rows = []
        for seed in seeds:
            b0_map = b0_by_seed[int(seed)]
            for item in cal_items:
                row = b0_map.get(item.question_id)
                if row is None:
                    continue
                rows.append(aggregate_fn(item, row, lam=float(lam), seed=int(seed)))
        f1 = mean_token_f1_across_seeds(rows)
        payload = {"lambda_agg": float(lam), "cal_token_f1": f1}
        tried.append(payload)
        if best is None or f1 > float(best["cal_token_f1"]) + 1e-12:
            best = payload
        if progress:
            progress.count(index, max(len(grid), 1), item="lambda_agg_grid")
    return {"selected": best, "tried": tried}
