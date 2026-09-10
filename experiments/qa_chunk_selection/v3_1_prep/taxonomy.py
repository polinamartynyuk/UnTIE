"""V3.1 overlay row schema. Reuses V3 target-status / gold-relation enums."""

from __future__ import annotations

from typing import Any, Mapping

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3.contract import GOLD_RELATIONS, PRIMARY_GOLD_RELATIONS, TARGET_STATUS
from experiments.qa_chunk_selection.v3.overlay import validate_overlay_row as validate_v3_overlay_row

from .contract import (
    ANNOTATION_VERSION,
    CONFIDENCE,
    FORBIDDEN_ANNOTATOR_FIELDS,
    MISMATCH_TYPES,
    SAMPLING_STRATA,
    SUPPORT_SCOPE,
    TAXONOMY_VERSION,
)


def strip_model_scores(row: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(row)
    for key in FORBIDDEN_ANNOTATOR_FIELDS:
        clean.pop(key, None)
    return clean


def assert_no_model_scores(row: Mapping[str, Any]) -> None:
    present = [key for key in FORBIDDEN_ANNOTATOR_FIELDS if key in row and row.get(key) not in (None, "", [], {})]
    if present:
        raise ValueError(f"annotator-facing row contains hidden model fields: {present}")


def taxonomy_payload() -> dict[str, object]:
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "annotation_version": ANNOTATION_VERSION,
        "target_match_status": list(TARGET_STATUS),
        "v3_gold_relation": list(GOLD_RELATIONS),
        "primary_gold_relations": sorted(PRIMARY_GOLD_RELATIONS),
        "v3_mismatch_type": list(MISMATCH_TYPES),
        "v3_annotation_confidence": list(CONFIDENCE),
        "v3_support_scope": list(SUPPORT_SCOPE),
        "annotation_sampling_stratum": list(SAMPLING_STRATA),
        "annotator_must_not_see": list(FORBIDDEN_ANNOTATOR_FIELDS),
    }


def empty_fill_fields() -> dict[str, Any]:
    return {
        "v3_target_status": None,
        "v3_target_value": None,
        "v3_gold_relation": None,
        "v3_mismatch_type": None,
        "v3_owner_text": None,
        "v3_support_scope": None,
        "v3_evidence_sentence_ids": None,
        "v3_evidence_char_offsets": None,
        "v3_annotation_confidence": None,
        "v3_annotation_reason": None,
        "source_sentence": None,
        "target_entity_or_value": None,
        "mismatch_entity_or_value": None,
        "context_used": "chunk",
        "adjudicated_status": None,
        "adjudication_reason": None,
        "annotator_id": None,
        "annotation_timestamp": None,
        "annotation_version": ANNOTATION_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
    }


def overlay_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["question_id"]), int(row["chunk_id"])


def final_status(row: Mapping[str, Any]) -> str | None:
    adjudicated = row.get("adjudicated_status")
    if adjudicated not in (None, ""):
        return str(adjudicated)
    status = row.get("v3_target_status")
    if status in (None, ""):
        return None
    return str(status)


def primary_status(row: Mapping[str, Any], *, keep_low_confidence: bool) -> str | None:
    status = final_status(row)
    if status is None:
        return None
    confidence = str(row.get("v3_annotation_confidence") or "")
    if (not keep_low_confidence) and confidence == "low" and not row.get("adjudicated_status"):
        return "unresolved"
    return status


def is_filled(row: Mapping[str, Any]) -> bool:
    return final_status(row) in TARGET_STATUS


def validate_prep_row(
    row: Mapping[str, Any],
    gold: InstanceRecord | None,
    *,
    require_filled: bool = False,
) -> list[str]:
    errors: list[str] = []
    if gold is not None:
        try:
            validate_v3_overlay_row(row, gold)
        except ValueError as exc:
            errors.append(str(exc))
        chunk = gold.chunk_by_id(int(row["chunk_id"]))
        if str(row.get("source_label")) != chunk.label:
            errors.append("source_label was rewritten")
        expected_sub = chunk.primary_class if chunk.label == "ambiguous" else None
        if row.get("source_ambiguous_subtype") not in (None, "") and row.get("source_ambiguous_subtype") != expected_sub:
            errors.append(
                f"source_ambiguous_subtype mismatch overlay={row.get('source_ambiguous_subtype')} source={expected_sub}"
            )
    status = row.get("v3_target_status")
    if require_filled and status not in TARGET_STATUS:
        errors.append("v3_target_status is required")
    if status not in (None, "") and status not in TARGET_STATUS:
        errors.append(f"unknown v3_target_status: {status}")
    relation = row.get("v3_gold_relation")
    if relation not in (None, "") and relation not in GOLD_RELATIONS:
        errors.append(f"unknown v3_gold_relation: {relation}")
    mismatch = row.get("v3_mismatch_type")
    if mismatch not in (None, "") and mismatch not in MISMATCH_TYPES:
        errors.append(f"unknown v3_mismatch_type: {mismatch}")
    confidence = row.get("v3_annotation_confidence")
    if confidence not in (None, "") and confidence not in CONFIDENCE:
        errors.append(f"unknown v3_annotation_confidence: {confidence}")
    scope = row.get("v3_support_scope")
    if scope not in (None, "") and scope not in SUPPORT_SCOPE:
        errors.append(f"unknown v3_support_scope: {scope}")
    stratum = row.get("annotation_sampling_stratum")
    if stratum not in (None, "") and stratum not in SAMPLING_STRATA:
        errors.append(f"unknown annotation_sampling_stratum: {stratum}")
    if status == "target_mismatch" and mismatch in (None, ""):
        errors.append("target_mismatch requires v3_mismatch_type (use unclear_mismatch_type if unknown)")
    if status == "target_evidence" and relation in (None, ""):
        errors.append("target_evidence requires v3_gold_relation")
    if status == "target_evidence" and relation == "no_gold_match_expected":
        errors.append("target_evidence + no_gold_match_expected is inconsistent; use new_valid_value_candidate")
    if require_filled and not str(row.get("v3_annotation_reason") or "").strip():
        errors.append("v3_annotation_reason is required")
    if require_filled and confidence not in CONFIDENCE:
        errors.append("v3_annotation_confidence is required")
    return errors


def overlay_index(rows: list[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        index[overlay_key(row)] = dict(row)
    return index
