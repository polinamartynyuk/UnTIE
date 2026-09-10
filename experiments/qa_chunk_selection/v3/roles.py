"""Original vs V3-expanded positives and verified target-mismatch sets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import ChunkRecord, InstanceRecord
from experiments.qa_chunk_selection.v2.hn_flags import is_training_eligible_hn

from .contract import GSRN_CLASS, PRIMARY_GOLD_RELATIONS

Overlay = Mapping[tuple[str, int], Mapping[str, Any]]


def overlay_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["question_id"]), int(row["chunk_id"])): dict(row) for row in rows}


def overlay_row(overlay: Overlay, question_id: str, chunk_id: int) -> Mapping[str, Any] | None:
    return overlay.get((str(question_id), int(chunk_id)))


def original_positives(item: InstanceRecord) -> tuple[ChunkRecord, ...]:
    return item.positives


def v3_target_positives(item: InstanceRecord, overlay: Overlay) -> tuple[ChunkRecord, ...]:
    selected = []
    seen = set()
    for chunk in item.positives:
        selected.append(chunk)
        seen.add(chunk.chunk_id)
    for chunk in item.chunks:
        if chunk.chunk_id in seen:
            continue
        row = overlay_row(overlay, item.question_id, chunk.chunk_id)
        if not row:
            continue
        if row.get("v3_target_status") != "target_evidence":
            continue
        if row.get("v3_gold_relation") not in PRIMARY_GOLD_RELATIONS:
            continue
        selected.append(chunk)
        seen.add(chunk.chunk_id)
    return tuple(selected)


def verified_target_mismatch(item: InstanceRecord, overlay: Overlay) -> tuple[ChunkRecord, ...]:
    selected = []
    for chunk in item.chunks:
        row = overlay_row(overlay, item.question_id, chunk.chunk_id)
        if row and row.get("v3_target_status") == "target_mismatch":
            selected.append(chunk)
    return tuple(selected)


def gsrn_chunks(item: InstanceRecord) -> tuple[ChunkRecord, ...]:
    return tuple(chunk for chunk in item.chunks if chunk.primary_class == GSRN_CLASS)


def ambiguous_chunks(item: InstanceRecord) -> tuple[ChunkRecord, ...]:
    return tuple(chunk for chunk in item.chunks if chunk.label == "ambiguous")


def known_target_status(item: InstanceRecord, overlay: Overlay, chunk: ChunkRecord) -> str | None:
    if chunk.label == "positive":
        return "target_evidence"
    if chunk.label == "hard_negative":
        return "known_non_target"
    if chunk.label == "easy_negative":
        return "known_non_target"
    row = overlay_row(overlay, item.question_id, chunk.chunk_id)
    if not row:
        return None
    status = row.get("v3_target_status")
    return str(status) if status else None


def train_role(
    item: InstanceRecord,
    chunk: ChunkRecord,
    overlay: Overlay,
    flags,
) -> str | None:
    """Supervised role for B1/B2/B3. partial_support and unresolved are masked."""
    pos_ids = {chunk.chunk_id for chunk in v3_target_positives(item, overlay)}
    if chunk.chunk_id in pos_ids:
        return "positive"
    if any(tm.chunk_id == chunk.chunk_id for tm in verified_target_mismatch(item, overlay)):
        return "target_mismatch"
    if chunk.label == "hard_negative" and is_training_eligible_hn(flags, item.question_id, chunk.chunk_id):
        return "hard_negative"
    if chunk.label == "easy_negative":
        return "easy_negative"
    return None


def labeled_candidates(
    item: InstanceRecord,
    overlay: Overlay,
    flags,
    *,
    easy_cap: int = 4,
) -> list[tuple[ChunkRecord, str]]:
    chosen: list[tuple[ChunkRecord, str]] = []
    easies = 0
    for chunk in item.chunks:
        role = train_role(item, chunk, overlay, flags)
        if role is None:
            continue
        if role == "easy_negative":
            if easies >= int(easy_cap):
                continue
            easies += 1
        chosen.append((chunk, role))
    return chosen
