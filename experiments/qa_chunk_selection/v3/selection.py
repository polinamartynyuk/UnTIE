"""Constrained multi-objective selection vs V3-B0. Not pairwise_hn lexicographic."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _f(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def passes_safeguards(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    epsilon_hn: float,
    epsilon_pairwise_hn: float,
    min_coverage: float,
) -> dict[str, Any]:
    hn_ok = _f(candidate, "hn@1") <= _f(baseline, "hn@1") + float(epsilon_hn)
    pair_ok = _f(candidate, "pairwise_clean_hn") >= _f(baseline, "pairwise_clean_hn") - float(epsilon_pairwise_hn)
    coverage = _f(candidate, "target_status_coverage@1", 0.0)
    tm_hard = False
    notes = []
    if coverage >= float(min_coverage) and candidate.get("target_mismatch@1") is not None:
        tm_hard = True
        notes.append("TM@1 available as diagnostic with sufficient coverage")
    else:
        notes.append("TM@1 diagnostic only (coverage below safeguard threshold)")
    ok = bool(hn_ok and pair_ok)
    return {
        "ok": ok,
        "hn_ok": hn_ok,
        "pairwise_hn_ok": pair_ok,
        "tm_as_hard_safeguard": tm_hard,
        "notes": notes,
    }


def select_among(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_name: str,
    epsilon_hn: float,
    epsilon_pairwise_hn: float,
    min_coverage: float,
) -> dict[str, Any]:
    by_name = {str(row["name"]): dict(row) for row in rows}
    baseline = by_name[baseline_name]
    admissible = []
    rejected = []
    for row in rows:
        gate = passes_safeguards(
            row,
            baseline,
            epsilon_hn=epsilon_hn,
            epsilon_pairwise_hn=epsilon_pairwise_hn,
            min_coverage=min_coverage,
        )
        item = {**dict(row), "safeguards": gate}
        if gate["ok"] or str(row["name"]) == baseline_name:
            admissible.append(item)
        else:
            rejected.append(item)
    admissible.sort(
        key=lambda row: (
            -_f(row, "token_f1", -1.0),
            -_f(row, "hit@1_original", -1.0),
            -_f(row, "target_evidence_hit@1_v3", -1.0),
            _f(row, "hn@1", 1.0),
        )
    )
    selected = admissible[0] if admissible else dict(baseline)
    return {
        "baseline": baseline_name,
        "selected": selected.get("name"),
        "selected_row": selected,
        "admissible": [row["name"] for row in admissible],
        "rejected": [row["name"] for row in rejected],
        "table": admissible + rejected,
        "primary_criterion": "document-aspect Token F1 among systems passing HN safeguards vs B0",
    }


def pareto_frontier(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = (
                _f(other, "token_f1") >= _f(row, "token_f1")
                and _f(other, "hit@1_original") >= _f(row, "hit@1_original")
                and _f(other, "target_evidence_hit@1_v3") >= _f(row, "target_evidence_hit@1_v3")
                and _f(other, "hn@1") <= _f(row, "hn@1")
            )
            strictly_better = (
                _f(other, "token_f1") > _f(row, "token_f1")
                or _f(other, "hit@1_original") > _f(row, "hit@1_original")
                or _f(other, "target_evidence_hit@1_v3") > _f(row, "target_evidence_hit@1_v3")
                or _f(other, "hn@1") < _f(row, "hn@1")
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            names.append(str(row["name"]))
    return names


def deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any], keys: Sequence[str]) -> dict[str, float]:
    out = {}
    for key in keys:
        cv = _f(candidate, key)
        bv = _f(baseline, key)
        if math.isnan(cv) or math.isnan(bv):
            out[f"delta_{key}"] = float("nan")
        else:
            out[f"delta_{key}"] = float(cv - bv)
    return out
