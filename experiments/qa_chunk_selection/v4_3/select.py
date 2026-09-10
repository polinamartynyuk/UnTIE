"""CAL-only selection helpers. EVAL is never used for tuning."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v4_2.select import choose_by_cal, mean_token_f1_across_seeds

from .evaluate import summarize_rows

__all__ = [
    "choose_admissible_cal",
    "choose_by_cal",
    "choose_max_f1",
    "mean_token_f1_across_seeds",
]


def choose_max_f1(
    tried: Sequence[Mapping[str, Any]],
    *,
    prefer_smaller: Sequence[str],
) -> dict[str, Any]:
    ranked = [dict(row) for row in tried]
    def key(row: Mapping[str, Any]) -> tuple:
        smaller = tuple(float(row.get(name) or 0.0) for name in prefer_smaller)
        return (-float(row.get("cal_token_f1") or 0.0), *smaller, float(row.get("cal_hn@1") or 0.0))
    ranked.sort(key=key)
    selected = dict(ranked[0]) if ranked else {}
    return {"selected": selected, "tried": ranked, "n_grid": len(ranked)}


def choose_admissible_cal(
    cal_items: Sequence[InstanceRecord],
    *,
    grid: Sequence[Mapping[str, float]],
    predict_rows: Callable[[Mapping[str, float]], list[dict[str, Any]]],
    hn_ref: float,
    epsilon_hn: float,
    prefer_smaller: Sequence[str],
    progress=None,
    item_name: str = "cal_grid",
) -> dict[str, Any]:
    """HN safeguard, then max F1, then smaller params, then smaller HN.

    Protocol SEM1 tie-break is smaller δ before HN, unlike V4.2 choose_by_cal.
    """
    del cal_items
    ranked = []
    for index, params in enumerate(grid, start=1):
        rows = predict_rows(params)
        hn_values = [float(row["hn@1"]) for row in rows if "hn@1" in row]
        hn = float(sum(hn_values) / len(hn_values)) if hn_values else float(summarize_rows(rows).get("hn@1") or 0.0)
        f1 = mean_token_f1_across_seeds(rows)
        payload = {
            **{str(k): float(v) for k, v in params.items()},
            "cal_token_f1": f1,
            "cal_hn@1": hn,
            "admissible": hn <= float(hn_ref) + float(epsilon_hn),
        }
        ranked.append(payload)
        if progress:
            progress.count(index, max(len(grid), 1), item=item_name)
    admissible = [row for row in ranked if row["admissible"]]
    if admissible:
        def key(row: Mapping[str, Any]) -> tuple:
            smaller = tuple(float(row.get(name) or 0.0) for name in prefer_smaller)
            return (-float(row["cal_token_f1"]), *smaller, float(row["cal_hn@1"]))
        pool = admissible
    else:
        def key(row: Mapping[str, Any]) -> tuple:
            smaller = tuple(float(row.get(name) or 0.0) for name in prefer_smaller)
            return (float(row["cal_hn@1"]), *smaller)
        pool = ranked
    pool.sort(key=key)
    selected = dict(pool[0]) if pool else {}
    if pool and not admissible:
        selected["admissible"] = False
        selected["forced_safeguard_fallback"] = True
    return {
        "selected": selected,
        "hn_ref": float(hn_ref),
        "tried": ranked,
        "n_grid": len(ranked),
        "n_admissible": len(admissible),
    }
