"""V4.1 evaluation: seed-aware pairing, seed-averaged clustered bootstrap, role reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import clustered_bootstrap_ci
from experiments.qa_chunk_selection.v3.split import records_for_role
from experiments.qa_chunk_selection.v4.evaluate import (
    aggregation_prediction,
    chunk_prediction,
    qa_fields_from_row,
    summarize_rows,
    transitions as transitions_qid_only,
)

ROLE_ORDER = ("train", "cal", "eval", "cal_eval", "all")


def row_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["question_id"]), int(row.get("seed") or 0)


def index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {row_key(row): dict(row) for row in rows}


def paired_delta(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    metric: str = "token_f1",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Match (question_id, seed). Average per-question across seeds, then bootstrap documents."""
    left = index_rows(baseline)
    per_q: dict[str, list[float]] = defaultdict(list)
    doc_of: dict[str, str] = {}
    wins = loses = ties = 0
    n_matched = 0
    for row in candidate:
        prev = left.get(row_key(row))
        if not prev or metric not in row or metric not in prev:
            continue
        delta = float(row[metric]) - float(prev[metric])
        per_q[str(row["question_id"])].append(delta)
        doc_of[str(row["question_id"])] = str(row.get("document_id") or prev.get("document_id") or "")
        n_matched += 1
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            loses += 1
        else:
            ties += 1
    question_rows = [
        {"document_id": doc_of[qid], "question_id": qid, "delta": float(mean(values))}
        for qid, values in per_q.items()
        if values
    ]
    boot = (
        clustered_bootstrap_ci(question_rows, "delta", n_boot=n_boot, seed=seed)
        if question_rows
        else {"mean": 0.0, "low": 0.0, "high": 0.0}
    )
    return {
        "n_rows": n_matched,
        "n_questions": len(question_rows),
        "mean_delta": boot.get("mean"),
        "ci95_low": boot.get("low"),
        "ci95_high": boot.get("high"),
        "win": wins,
        "lose": loses,
        "tie": ties,
        "pairing": "question_id+seed",
        "bootstrap": "mean_over_seeds_then_cluster_document_id",
    }


def transitions(baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    left = index_rows(baseline)
    counts: Counter[str] = Counter()
    for row in candidate:
        prev = left.get(row_key(row))
        if not prev:
            continue
        a = str(prev.get("selected_chunk_label") or "none")
        b = str(row.get("selected_chunk_label") or "none")
        ap = str(prev.get("selected_chunk_primary_class") or "")
        counts[f"{a}->{b}"] += 1
        if ap == "gold_supporting_robust_negative" and b == "positive":
            counts["GSRN->positive"] += 1
        if a == "positive" and b == "hard_negative":
            counts["positive->HN"] += 1
    pos_gain = (
        counts.get("ambiguous->positive", 0)
        + counts.get("hard_negative->positive", 0)
        + counts.get("easy_negative->positive", 0)
    )
    pos_loss = (
        counts.get("positive->ambiguous", 0)
        + counts.get("positive->hard_negative", 0)
        + counts.get("positive->easy_negative", 0)
    )
    counts["net_positive_gain"] = pos_gain - pos_loss
    return dict(counts)


def role_of_document(document_id: str, splits: Mapping[str, Any], run: int) -> str:
    assignment = splits.get("assignment") or {}
    fold = int(assignment.get(document_id, -1))
    spec = splits["runs"][int(run)]
    if fold in {int(item) for item in spec["train"]}:
        return "train"
    if fold in {int(item) for item in spec["calibration"]}:
        return "cal"
    if fold in {int(item) for item in spec["eval"]}:
        return "eval"
    return "other"


def split_by_role(
    rows: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Any],
    run: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {key: [] for key in ("train", "cal", "eval", "other")}
    for row in rows:
        out[role_of_document(str(row["document_id"]), splits, run)].append(dict(row))
    out["cal_eval"] = list(out["cal"]) + list(out["eval"])
    out["all"] = list(rows)
    return out


def role_report(
    rows: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]] | None,
    splits: Mapping[str, Any],
    run: int,
    *,
    n_boot: int,
    seed: int,
    in_sample_roles: Sequence[str] = (),
) -> dict[str, Any]:
    grouped = split_by_role(rows, splits, run)
    payload: dict[str, Any] = {}
    for role in ROLE_ORDER:
        subset = grouped.get(role) or []
        summary = summarize_rows(subset) if subset else {"n": 0}
        entry: dict[str, Any] = {"metrics": summary, "n": len(subset)}
        if baseline and subset:
            entry["delta"] = paired_delta(baseline, subset, n_boot=n_boot, seed=seed)
        payload[role] = entry
    payload["IN_SAMPLE_CONTAMINATED"] = bool(in_sample_roles) and bool(grouped.get("train"))
    payload["in_sample_roles"] = list(in_sample_roles)
    payload["note"] = "ALL includes TRAIN; do not use ALL for trainable selection"
    return payload


def seed_design(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeds = sorted({int(row.get("seed") or 0) for row in rows})
    n_q = len({str(row["question_id"]) for row in rows})
    return {
        "seeds": seeds,
        "n_rows": len(rows),
        "n_questions": n_q,
        "three_seed": seeds == [42, 43, 44] and len(rows) == n_q * 3,
        "single_seed": len(seeds) == 1,
    }


def comparable_to_b0(candidate: Mapping[str, Any], b0: Mapping[str, Any]) -> bool:
    if candidate.get("IN_SAMPLE_CONTAMINATED"):
        return False
    if candidate.get("three_seed") != b0.get("three_seed"):
        return candidate.get("single_seed") and set(candidate.get("seeds") or []) <= set(b0.get("seeds") or [])
    return True


def filter_rows_for_role(unlocked, splits, run: int, role: str):
    if role == "calibration":
        role = "cal"
    mapped = {"cal": "calibration", "eval": "eval", "train": "train"}.get(role, role)
    return records_for_role(unlocked, splits, run=run, role=mapped)


def metric_close(left: float, right: float, *, tol: float) -> bool:
    return abs(float(left) - float(right)) <= float(tol)


__all__ = [
    "aggregation_prediction",
    "chunk_prediction",
    "comparable_to_b0",
    "filter_rows_for_role",
    "index_rows",
    "metric_close",
    "paired_delta",
    "qa_fields_from_row",
    "role_of_document",
    "role_report",
    "row_key",
    "seed_design",
    "split_by_role",
    "summarize_rows",
    "transitions",
    "transitions_qid_only",
]
