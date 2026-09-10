"""Runtime source-overlap clusters. Not D5 gold-evidence clusters."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3_1_auto_prep.cluster import assign_clusters, overlap_ratio


def chunk_span(chunk: Mapping[str, Any] | Any) -> tuple[int, int] | None:
    start = getattr(chunk, "char_start", None)
    end = getattr(chunk, "char_end", None)
    if start is None:
        start = chunk.get("char_start") if isinstance(chunk, Mapping) else None
        end = chunk.get("char_end") if isinstance(chunk, Mapping) else None
    if start is None or end is None:
        return None
    start_i, end_i = int(start), int(end)
    if end_i <= start_i:
        return None
    return start_i, end_i


def overlapping_ids(
    item: InstanceRecord,
    chunk_id: int,
    *,
    min_overlap: float,
    pool_ids: Sequence[int] | None = None,
) -> set[int]:
    target = item.chunk_by_id(int(chunk_id))
    span = chunk_span(target)
    allowed = {int(cid) for cid in (pool_ids if pool_ids is not None else [c.chunk_id for c in item.chunks])}
    out = {int(chunk_id)}
    if span is None:
        return out
    for chunk in item.chunks:
        if int(chunk.chunk_id) not in allowed:
            continue
        other = chunk_span(chunk)
        if other is None:
            continue
        if overlap_ratio(span, other) >= float(min_overlap):
            out.add(int(chunk.chunk_id))
    return out


def document_source_clusters(item: InstanceRecord, *, min_overlap: float) -> list[dict[str, Any]]:
    rows = [
        {
            "question_id": item.question_id,
            "document_id": item.document_id,
            "chunk_id": chunk.chunk_id,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }
        for chunk in item.chunks
    ]
    return assign_clusters(rows, min_overlap=min_overlap)


def cluster_id_by_chunk(cluster_rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    return {int(row["chunk_id"]): str(row["evidence_cluster_id"]) for row in cluster_rows}


def shortlist_source_clusters(
    item: InstanceRecord,
    chunk_ids: Sequence[int],
    *,
    min_overlap: float,
) -> list[dict[str, Any]]:
    rows = []
    for cid in chunk_ids:
        chunk = item.chunk_by_id(int(cid))
        rows.append(
            {
                "question_id": item.question_id,
                "document_id": item.document_id,
                "chunk_id": chunk.chunk_id,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
        )
    return assign_clusters(rows, min_overlap=min_overlap)
