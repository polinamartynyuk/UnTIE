"""Sanity audit of frozen positive / hard-negative labels. Does not relabel."""

from __future__ import annotations

from typing import Any, Sequence

from untie.aspect_qa.normalize import identity_key

from .schema import InstanceRecord


def gold_string_in_text(gold_answers: Sequence[str], text: str) -> bool:
    haystack = identity_key(text)
    if not haystack:
        return False
    for gold in gold_answers:
        key = identity_key(gold)
        if key and key in haystack:
            return True
    return False


def audit_instance(instance: InstanceRecord) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    chunk_ids = [chunk.chunk_id for chunk in instance.chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        findings.append(
            {
                "question_id": instance.question_id,
                "check": "unique_chunk_id",
                "ok": False,
                "detail": "duplicate chunk_id",
            }
        )
    pos_ids = {chunk.chunk_id for chunk in instance.positives}
    hn_ids = {chunk.chunk_id for chunk in instance.hard_negatives}
    if pos_ids & hn_ids:
        findings.append(
            {
                "question_id": instance.question_id,
                "check": "positive_hn_disjoint",
                "ok": False,
                "detail": sorted(pos_ids & hn_ids),
            }
        )
    if instance.instance_status == "ok" and not instance.positives:
        findings.append(
            {
                "question_id": instance.question_id,
                "check": "answerable_has_positive",
                "ok": False,
                "detail": instance.instance_status,
            }
        )
    for chunk in instance.hard_negatives:
        present = gold_string_in_text(instance.gold_answers, chunk.text)
        if present:
            findings.append(
                {
                    "question_id": instance.question_id,
                    "chunk_id": chunk.chunk_id,
                    "check": "hn_literal_gold",
                    "ok": False,
                    "detail": "normalized gold string occurs in hard_negative text",
                    "label_original": chunk.primary_class,
                    "label_verified": chunk.label,
                }
            )
        if chunk.gold_supporting:
            findings.append(
                {
                    "question_id": instance.question_id,
                    "chunk_id": chunk.chunk_id,
                    "check": "hn_not_gold_supporting",
                    "ok": False,
                    "detail": "hard_negative marked gold_supporting",
                }
            )
    return findings


def audit_records(records: Sequence[InstanceRecord]) -> dict[str, Any]:
    keys = [(item.document_id, item.question_id) for item in records]
    duplicate_keys = len(keys) != len(set(keys))
    findings: list[dict[str, Any]] = []
    if duplicate_keys:
        findings.append({"check": "unique_question_id", "ok": False, "detail": "duplicate keys"})
    for item in records:
        findings.extend(audit_instance(item))
    hn_gold_hits = [row for row in findings if row.get("check") == "hn_literal_gold"]
    errors = [
        row
        for row in findings
        if row.get("check") in {"unique_chunk_id", "positive_hn_disjoint", "unique_question_id"}
        and row.get("ok") is False
    ]
    return {
        "n_instances": len(records),
        "n_findings": len(findings),
        "n_blocking_errors": len(errors),
        "n_hn_literal_gold_warnings": len(hn_gold_hits),
        "blocking_errors": errors,
        "hn_literal_gold_warnings": hn_gold_hits[:50],
        "note": (
            "Labels are frozen. Literal gold in a hard_negative is recorded as a "
            "warning and is not used to rewrite Phase 1 classes."
        ),
    }
