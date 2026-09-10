"""Overlap clusters so one source region is not oversampled."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def overlap_ratio(left: tuple[int, int], right: tuple[int, int]) -> float:
    inter = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    span = max(min(left[1] - left[0], right[1] - right[0]), 1)
    return float(inter) / float(span)


def assign_clusters(rows: Sequence[Mapping[str, Any]], *, min_overlap: float = 0.6) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(dict(row))
    out: list[dict[str, Any]] = []
    for question_id, items in grouped.items():
        clusters: list[list[dict[str, Any]]] = []
        for row in items:
            span = _span(row)
            placed = False
            if span is not None:
                for cluster in clusters:
                    if any(other is not None and overlap_ratio(span, other) >= min_overlap for other in (_span(item) for item in cluster)):
                        cluster.append(row)
                        placed = True
                        break
            if not placed:
                clusters.append([row])
        for index, cluster in enumerate(clusters):
            cluster_id = f"{question_id}::c{index}"
            for row in cluster:
                row["evidence_cluster_id"] = cluster_id
                row["evidence_cluster_size"] = len(cluster)
                out.append(row)
    return out


def _span(row: Mapping[str, Any]) -> tuple[int, int] | None:
    start = row.get("qa_span_start", row.get("char_start"))
    end = row.get("qa_span_end", row.get("char_end"))
    if start is None or end is None:
        return None
    start_i, end_i = int(start), int(end)
    if end_i <= start_i:
        return None
    return start_i, end_i
