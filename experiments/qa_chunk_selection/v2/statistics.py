"""Paired clustered bootstrap. Seeds are averaged per question before clustering."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import clustered_bootstrap_ci, paired_delta


def mean_std(values: Sequence[float]) -> dict[str, float]:
    clean = [float(item) for item in values]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(clean) / len(clean)
    var = sum((item - mean) ** 2 for item in clean) / max(len(clean) - 1, 1)
    return {"mean": float(mean), "std": float(var ** 0.5), "n": len(clean)}


def average_across_seeds(
    rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    docs: dict[tuple[str, str], str] = {}
    for seed, rows in rows_by_seed.items():
        del seed
        for row in rows:
            key = (str(row["document_id"]), str(row["question_id"]))
            if metric not in row or row[metric] is None:
                continue
            grouped[key].append(float(row[metric]))
            docs[key] = str(row["document_id"])
    merged = []
    for (document_id, question_id), values in grouped.items():
        merged.append(
            {
                "document_id": document_id,
                "question_id": question_id,
                metric: sum(values) / len(values),
            }
        )
    return merged


def paired_delta_multiseed(
    left_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    right_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    metric: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    left = average_across_seeds(left_by_seed, metric)
    right = average_across_seeds(right_by_seed, metric)
    payload = paired_delta(left, right, metric, n_boot=n_boot, seed=seed)
    payload["number_of_seeds"] = len(set(left_by_seed) | set(right_by_seed))
    payload["aggregation_across_seeds"] = "mean_per_question_then_clustered_bootstrap_by_document"
    payload["n_documents"] = len({row["document_id"] for row in left})
    return payload


def clustered_ci_rows(rows: Sequence[Mapping[str, Any]], metric: str, *, n_boot: int, seed: int) -> dict[str, float]:
    return clustered_bootstrap_ci(rows, metric, n_boot=n_boot, seed=seed)
