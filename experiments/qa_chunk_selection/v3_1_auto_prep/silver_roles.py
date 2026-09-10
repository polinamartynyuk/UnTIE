"""P_silver_target / TM_silver from high-confidence QA-based labels."""

from __future__ import annotations

from typing import Any, Mapping

from experiments.qa_chunk_selection.schema import InstanceRecord

from .contract import PRIMARY_GOLD_RELATIONS

Overlay = Mapping[tuple[str, int], Mapping[str, Any]]


def overlay_index(rows) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["question_id"]), int(row["chunk_id"])): dict(row) for row in rows}


def p_silver_target_ids(item: InstanceRecord, overlay: Overlay) -> set[int]:
    selected = {chunk.chunk_id for chunk in item.positives}
    for chunk in item.chunks:
        row = overlay.get((item.question_id, chunk.chunk_id))
        if not row or row.get("silver_confidence") != "high":
            continue
        if row.get("silver_target_status") != "silver_target_evidence":
            continue
        relation = row.get("qa_gold_relation") or row.get("silver_gold_relation")
        if relation in PRIMARY_GOLD_RELATIONS or not relation:
            selected.add(chunk.chunk_id)
    return selected


def tm_silver_ids(item: InstanceRecord, overlay: Overlay) -> set[int]:
    return {
        chunk.chunk_id
        for chunk in item.chunks
        if (overlay.get((item.question_id, chunk.chunk_id)) or {}).get("silver_target_status") == "silver_target_mismatch"
        and (overlay.get((item.question_id, chunk.chunk_id)) or {}).get("silver_confidence") == "high"
    }


def silver_to_v3_training_overlay(rows: list[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    relation_map = {
        "exact_match_gold": "matches_existing_gold",
        "normalized_match_gold": "matches_existing_gold",
        "overlap_match_gold": "semantically_equivalent_to_existing_gold",
        "new_value_candidate": "new_valid_value_candidate",
        "no_gold_relation": "no_gold_match_expected",
    }
    mapped: dict[tuple[str, int], dict[str, Any]] = {}
    clusters: dict[tuple[str, str, str], list[tuple[tuple[str, int], dict[str, Any]]]] = {}
    for row in rows:
        if row.get("silver_confidence") != "high":
            continue
        status = row.get("silver_target_status")
        if status not in {"silver_target_evidence", "silver_target_mismatch"}:
            continue
        if status == "silver_target_evidence" and row.get("new_value_candidate"):
            continue
        v3_status = "target_evidence" if status == "silver_target_evidence" else "target_mismatch"
        relation = relation_map.get(str(row.get("qa_gold_relation") or ""), "unclear")
        if v3_status == "target_evidence" and relation not in {
            "matches_existing_gold",
            "semantically_equivalent_to_existing_gold",
        }:
            continue
        key = (str(row["question_id"]), int(row["chunk_id"]))
        payload = {
            "question_id": row["question_id"],
            "chunk_id": row["chunk_id"],
            "document_id": row.get("document_id"),
            "source_label": row.get("source_label"),
            "v3_target_status": v3_status,
            "v3_gold_relation": relation,
            "v3_annotation_confidence": "high",
            "silver_target_status": status,
            "evidence_cluster_id": row.get("evidence_cluster_id"),
            "label_status": "silver",
            "chunk_text": row.get("chunk_text") or row.get("qa_predicted_span") or "",
            "outer_run_id": row.get("outer_run_id"),
        }
        cluster = str(row.get("evidence_cluster_id") or f"{key[0]}:{key[1]}")
        clusters.setdefault((key[0], cluster, v3_status), []).append((key, payload))
    for items in clusters.values():
        key, row = max(items, key=lambda item: len(str(item[1].get("chunk_text") or "")))
        mapped[key] = row
    return mapped
