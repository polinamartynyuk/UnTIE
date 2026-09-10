"""HN quality flags. Labels are never rewritten."""

from __future__ import annotations

from typing import Any, Sequence

from experiments.qa_chunk_selection.sanity import gold_string_in_text
from experiments.qa_chunk_selection.schema import InstanceRecord


def hn_quality_rows(
    records: Sequence[InstanceRecord],
    *,
    include_hn_gold_warning: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in records:
        for chunk in item.chunks:
            warning = False
            eligible = False
            if chunk.label == "hard_negative":
                warning = gold_string_in_text(item.gold_answers, chunk.text)
                eligible = (not warning) or bool(include_hn_gold_warning)
            rows.append(
                {
                    "document_id": item.document_id,
                    "question_id": item.question_id,
                    "chunk_id": chunk.chunk_id,
                    "split": item.split,
                    "label": chunk.label,
                    "primary_class": chunk.primary_class,
                    "hn_gold_string_warning": warning,
                    "training_eligible_hn": eligible,
                }
            )
    return rows


def flag_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["question_id"]), int(row["chunk_id"])): dict(row) for row in rows}


def is_training_eligible_hn(
    flags: dict[tuple[str, int], dict[str, Any]],
    question_id: str,
    chunk_id: int,
) -> bool:
    row = flags.get((question_id, int(chunk_id)))
    return bool(row and row.get("training_eligible_hn"))
