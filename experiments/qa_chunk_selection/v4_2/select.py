"""CAL-only λ / δ selection. Never peeks at EVAL."""

from __future__ import annotations

from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord

from .evaluate import summarize_rows


def mean_token_f1_across_seeds(rows: Sequence[Mapping[str, Any]]) -> float:
    by_q: dict[str, list[float]] = {}
    for row in rows:
        by_q.setdefault(str(row["question_id"]), []).append(float(row.get("token_f1") or 0.0))
    if not by_q:
        return 0.0
    return float(mean(mean(values) for values in by_q.values()))


def choose_by_cal(
    cal_items: Sequence[InstanceRecord],
    *,
    grid: Sequence[Mapping[str, float]],
    predict_rows: Callable[[Mapping[str, float]], list[dict[str, Any]]],
    b0_hn: float,
    epsilon_hn: float,
    prefer_smaller: Sequence[str],
    progress=None,
    item_name: str = "cal_grid",
) -> dict[str, Any]:
    ranked = []
    for index, params in enumerate(grid, start=1):
        rows = predict_rows(params)
        summary = summarize_rows(rows)
        hn = float(summary.get("hn@1") or 0.0)
        f1 = mean_token_f1_across_seeds(rows)
        payload = {
            **{str(k): float(v) for k, v in params.items()},
            "cal_token_f1": f1,
            "cal_hn@1": hn,
            "admissible": hn <= float(b0_hn) + float(epsilon_hn),
        }
        ranked.append(payload)
        if progress:
            progress.count(index, max(len(grid), 1), item=item_name)
    admissible = [row for row in ranked if row["admissible"]]
    if admissible:
        def key(row: Mapping[str, Any]) -> tuple:
            smaller = tuple(float(row.get(name) or 0.0) for name in prefer_smaller)
            return (-float(row["cal_token_f1"]), float(row["cal_hn@1"]), *smaller)
        pool = admissible
    else:
        def key(row: Mapping[str, Any]) -> tuple:
            smaller = tuple(float(row.get(name) or 0.0) for name in prefer_smaller)
            return (float(row["cal_hn@1"]), *smaller)
        pool = ranked
    pool.sort(key=key)
    selected = dict(pool[0])
    if not admissible:
        selected["admissible"] = False
        selected["forced_safeguard_fallback"] = True
    return {"selected": selected, "b0_cal_hn@1": float(b0_hn), "tried": ranked, "n_grid": len(ranked), "n_admissible": len(admissible)}
