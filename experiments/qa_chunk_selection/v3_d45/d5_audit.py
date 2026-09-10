"""D5 audit: independent gold-positive clusters after overlap dedup. No aggregator."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3_1_auto_prep.cluster import assign_clusters


def run_d5_audit(records: Sequence[InstanceRecord], *, min_overlap: float = 0.6) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in records:
        positives = [
            {
                "question_id": item.question_id,
                "chunk_id": chunk.chunk_id,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "qa_span_start": chunk.char_start,
                "qa_span_end": chunk.char_end,
                "text": chunk.text,
            }
            for chunk in item.positives
        ]
        clustered = assign_clusters(positives, min_overlap=min_overlap) if positives else []
        n_clusters = len({row.get("evidence_cluster_id") for row in clustered}) if clustered else 0
        rows.append(
            {
                "document_id": item.document_id,
                "question_id": item.question_id,
                "aspect": item.aspect,
                "split": item.split,
                "n_positive_chunks": len(item.positives),
                "n_independent_clusters": n_clusters,
                "has_ge2_positives": len(item.positives) >= 2,
                "has_ge2_independent_clusters": n_clusters >= 2,
                "cluster_ids": sorted({str(row.get("evidence_cluster_id")) for row in clustered}),
            }
        )
    return rows, summarize_d5(rows)


def summarize_d5(rows: Sequence[dict[str, Any]], *, min_multi_cluster_rate: float = 0.20) -> dict[str, Any]:
    n = len(rows)
    ge2_pos = sum(1 for row in rows if row.get("has_ge2_positives"))
    ge2_ind = sum(1 for row in rows if row.get("has_ge2_independent_clusters"))
    by_aspect = defaultdict(list)
    for row in rows:
        by_aspect[str(row.get("aspect"))].append(int(row.get("n_independent_clusters") or 0))
    rate = (ge2_ind / n) if n else 0.0
    return {
        "n_questions": n,
        "n_with_ge2_positive_chunks": ge2_pos,
        "n_with_ge2_independent_clusters": ge2_ind,
        "rate_ge2_positives": (ge2_pos / n) if n else 0.0,
        "rate_ge2_independent_clusters": rate,
        "mean_clusters_by_aspect": {aspect: (sum(vals) / len(vals) if vals else 0.0) for aspect, vals in by_aspect.items()},
        "cluster_size_hist": dict(Counter(int(row.get("n_independent_clusters") or 0) for row in rows)),
        "aggregation_allowed": rate >= float(min_multi_cluster_rate),
        "min_multi_cluster_rate": float(min_multi_cluster_rate),
        "note": "overlapping positive chunks are one cluster; no semantic answer clustering",
    }
