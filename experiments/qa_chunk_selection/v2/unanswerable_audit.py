"""Unanswerable audit. Heuristic statuses are not new gold."""

from __future__ import annotations

from typing import Any, Sequence

from experiments.qa_chunk_selection.sanity import gold_string_in_text
from experiments.qa_chunk_selection.schema import InstanceRecord

from .io_util import write_json, write_jsonl


def run_unanswerable_audit(records: Sequence[InstanceRecord], output_dir) -> dict[str, Any]:
    rows = []
    for item in records:
        if item.positives:
            continue
        gold_in_chunks = any(gold_string_in_text(item.gold_answers, chunk.text) for chunk in item.chunks)
        if gold_in_chunks:
            status = "answer_exists_but_no_robust_positive"
        elif item.gold_answers:
            status = "unknown"
        else:
            status = "true_no_answer_candidate"
        rows.append(
            {
                "document_id": item.document_id,
                "question_id": item.question_id,
                "aspect": item.aspect,
                "split": item.split,
                "question": item.question,
                "gold_answers": list(item.gold_answers),
                "n_chunks": len(item.chunks),
                "gold_string_anywhere_in_document": None,
                "gold_string_anywhere_in_document_unavailable": True,
                "gold_string_anywhere_in_chunks": gold_in_chunks,
                "ambiguous_count": sum(chunk.label == "ambiguous" for chunk in item.chunks),
                "HN_count": len(item.hard_negatives),
                "heuristic_status": status,
                "status_is_heuristic": True,
            }
        )
    summary = {
        "n_excluded": len(rows),
        "by_status": {},
        "note": "heuristic statuses are not gold labels; no-answer model is not trained",
    }
    for row in rows:
        key = row["heuristic_status"]
        summary["by_status"][key] = summary["by_status"].get(key, 0) + 1
    write_jsonl(output_dir / "rows.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    return summary
