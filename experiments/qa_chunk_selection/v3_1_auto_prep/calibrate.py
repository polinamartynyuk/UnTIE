"""TRAIN-only score normalization and CAL-only thresholds. No EVAL leakage."""

from __future__ import annotations

from typing import Any, Sequence

MINING_SCORE_VERSION = "raw_margin_mean_cdf_1"


def empirical_cdf(value: float, reference: Sequence[float]) -> float:
    rows = [float(item) for item in reference]
    if not rows:
        return 0.5
    return sum(1.0 for item in rows if item <= float(value)) / float(len(rows))


def median(values: Sequence[float]) -> float:
    rows = sorted(float(item) for item in values)
    if not rows:
        return 0.0
    mid = len(rows) // 2
    if len(rows) % 2:
        return rows[mid]
    return 0.5 * (rows[mid - 1] + rows[mid])


def iqr(values: Sequence[float]) -> float:
    rows = sorted(float(item) for item in values)
    if len(rows) < 4:
        return 1.0
    q1 = rows[len(rows) // 4]
    q3 = rows[(3 * len(rows)) // 4]
    return max(q3 - q1, 1e-6)


def plausibility_score(raw: float, train_positive_scores: Sequence[float]) -> float:
    return empirical_cdf(raw, train_positive_scores)


def mining_score(
    raw: float,
    margin: float,
    train_raw: Sequence[float],
    train_margin: Sequence[float],
) -> dict[str, float | str]:
    raw_cdf = empirical_cdf(raw, train_raw)
    margin_cdf = empirical_cdf(margin, train_margin)
    return {
        "qa_raw_cdf": raw_cdf,
        "qa_margin_cdf": margin_cdf,
        "qa_plausibility_score": 0.5 * raw_cdf + 0.5 * margin_cdf,
        "mining_score_version": MINING_SCORE_VERSION,
    }


def select_stage_a_threshold(
    cal_positive: Sequence[float],
    cal_easy: Sequence[float],
    *,
    r_min: float,
    max_easy_pass: float,
) -> dict[str, Any]:
    candidates = _candidate_taus(cal_positive)
    if not candidates:
        return {
            "tau_qa": 0.0,
            "cal_positive_recall": 0.0,
            "cal_easy_pass": 0.0,
            "ok": False,
            "reason": "empty_cal_positive",
            "n_cal_pos": 0,
            "n_cal_easy": len(cal_easy),
            "r_min": float(r_min),
            "max_easy_pass": float(max_easy_pass),
            "selection_rule": "none",
            "pareto": [],
        }
    feasible: list[tuple[float, float, float]] = []
    recall_ok: list[tuple[float, float, float]] = []
    for tau in candidates:
        rec = _rate(cal_positive, tau)
        easy = _rate(cal_easy, tau)
        if rec + 1e-12 >= float(r_min):
            recall_ok.append((tau, rec, easy))
            if easy <= float(max_easy_pass):
                feasible.append((tau, rec, easy))
    if feasible:
        tau, rec, easy = feasible[0]
        rule = "most_inclusive_joint_gate"
        ok = True
        reason = "joint_gate"
    elif recall_ok:
        tau, rec, easy = recall_ok[-1]
        rule = "most_selective_meeting_r_min"
        ok = False
        reason = "no_feasible_tau"
    else:
        tau = float(candidates[-1])
        rec = _rate(cal_positive, tau)
        easy = _rate(cal_easy, tau)
        rule = "no_r_min_operating_point"
        ok = False
        reason = "r_min_unattainable"
    return {
        "tau_qa": float(tau),
        "cal_positive_recall": float(rec),
        "cal_easy_pass": float(easy),
        "ok": ok,
        "reason": reason,
        "selection_rule": rule,
        "n_cal_pos": len(cal_positive),
        "n_cal_easy": len(cal_easy),
        "r_min": float(r_min),
        "max_easy_pass": float(max_easy_pass),
        "n_feasible_tau": len(feasible),
        "pareto": _pareto_sample(cal_positive, cal_easy),
    }


def select_consistency_threshold(pos_agreements: Sequence[float], hn_agreements: Sequence[float]) -> float:
    del hn_agreements
    grid = (0.50, 0.67, 0.75)
    if not pos_agreements:
        return 0.50
    mean_pos = sum(pos_agreements) / len(pos_agreements)
    for tau in grid:
        if mean_pos + 1e-12 >= tau:
            return float(tau)
    return 0.50


def select_ownership_delta(pos_margins: Sequence[float]) -> float:
    grid = (0.0, 0.25, 0.50)
    if not pos_margins:
        return 0.25
    mid = median(pos_margins)
    chosen = 0.25
    for delta in grid:
        if mid + 1e-12 >= delta:
            chosen = float(delta)
    return chosen


def _candidate_taus(cal_positive: Sequence[float]) -> list[float]:
    grid = {round(index / 100.0, 2) for index in range(0, 101)}
    grid.update(round(float(item), 4) for item in cal_positive)
    return sorted(grid)


def _pareto_sample(cal_positive: Sequence[float], cal_easy: Sequence[float]) -> list[dict[str, float]]:
    rows = []
    for index in range(0, 21):
        tau = index / 20.0
        rows.append(
            {
                "tau_qa": tau,
                "cal_positive_recall": _rate(cal_positive, tau),
                "cal_easy_pass": _rate(cal_easy, tau),
            }
        )
    return rows


def _rate(scores: Sequence[float], tau: float) -> float:
    if not scores:
        return 0.0
    return sum(1.0 for item in scores if float(item) >= float(tau)) / float(len(scores))
