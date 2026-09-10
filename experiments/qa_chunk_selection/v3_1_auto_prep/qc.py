"""QC, gates, coverage, CV inventory, readiness for QA-based silver overlay."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS, POSTHOC_SPLITS
from experiments.qa_chunk_selection.v3.split import records_for_role

from .silver_roles import overlay_index, p_silver_target_ids, tm_silver_ids


def file_sha256(path) -> str:
    from pathlib import Path

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_ids(records: Sequence[InstanceRecord]) -> dict[str, set[str]]:
    docs: set[str] = set()
    questions: set[str] = set()
    for item in records:
        if item.split in LOCKED_SPLITS or item.split in POSTHOC_SPLITS:
            docs.add(item.document_id)
            questions.add(item.question_id)
    return {"documents": docs, "questions": questions}


def validate_rows(rows: Sequence[Mapping[str, Any]], records: Sequence[InstanceRecord], locked) -> dict[str, Any]:
    by_q = {item.question_id: item for item in records}
    errors = []
    for row in rows:
        item = by_q.get(str(row["question_id"]))
        if item is None:
            errors.append({"error": "question not in unlocked universe", "question_id": row.get("question_id")})
            continue
        if item.document_id in locked["documents"] or item.question_id in locked["questions"]:
            errors.append({"error": "INTERNAL_TEST leakage", "question_id": row.get("question_id")})
        chunk = item.chunk_by_id(int(row["chunk_id"]))
        if str(row.get("source_label")) != chunk.label:
            errors.append({"error": "source_label rewritten", "question_id": row.get("question_id")})
    return {"n_rows": len(rows), "n_errors": len(errors), "errors": errors[:100], "ok": not errors}


def control_metrics(rows: Sequence[Mapping[str, Any]], stage_a: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a_by_key = {(str(row["question_id"]), int(row["chunk_id"])): row for row in stage_a}
    by_role: dict[str, list] = defaultdict(list)
    for row in rows:
        by_role[str(row.get("universe_role") or "")].append(row)
    pos = by_role.get("control_positive") or []
    hn = by_role.get("control_hn") or []
    easy = by_role.get("control_easy") or []

    def pass_rate(group):
        hits = 0
        for row in group:
            arow = a_by_key.get((str(row["question_id"]), int(row["chunk_id"])))
            if arow and arow.get("qa_plausible"):
                hits += 1
        return (hits / len(group)) if group else None

    return {
        "StageA_PositiveRecall": pass_rate(pos),
        "StageA_HNPassRate": pass_rate(hn),
        "StageA_EasyPassRate": pass_rate(easy),
        "SilverTargetEvidenceRate": _rate(rows, "silver_target_evidence"),
        "SilverTargetMismatchRate": _rate(rows, "silver_target_mismatch"),
        "SilverPartialRate": _rate(rows, "silver_partial_support"),
        "SilverUnresolvedRate": _rate(rows, "silver_unresolved"),
        "HighConfidenceRate": (sum(1 for row in rows if row.get("silver_confidence") == "high") / len(rows)) if rows else None,
        "n_control_positive": len(pos),
        "n_control_hn": len(hn),
        "n_control_easy": len(easy),
    }


def _rate(rows, status: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get("silver_target_status") == status) / len(rows)


def silver_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status = Counter(str(row.get("silver_target_status")) for row in rows)
    aspect = Counter(f"{row.get('silver_target_status')}|{row.get('aspect')}" for row in rows)
    return {
        "n_annotated": len(rows),
        "n_by_status": dict(status),
        "n_high_silver_target_evidence": sum(
            1 for row in rows if row.get("silver_target_status") == "silver_target_evidence" and row.get("silver_confidence") == "high"
        ),
        "n_high_silver_target_mismatch": sum(
            1 for row in rows if row.get("silver_target_status") == "silver_target_mismatch" and row.get("silver_confidence") == "high"
        ),
        "status_x_aspect": dict(aspect),
    }


def cv_inventory(records, splits, overlays_by_run: Mapping[int, Mapping], progress: ProgressLogger | None = None) -> dict[str, Any]:
    log = progress or ProgressLogger()
    runs = []
    for run in splits.get("runs") or []:
        run_id = int(run["run"])
        log.log(f"silver cv inventory run={run_id}")
        overlay = overlays_by_run.get(run_id) or {}
        train = records_for_role(records, splits, run=run_id, role="train")
        ev = records_for_role(records, splits, run=run_id, role="eval")
        train_q = {item.question_id for item in train if tm_silver_ids(item, overlay)}
        eval_both = {item.question_id for item in ev if p_silver_target_ids(item, overlay) and tm_silver_ids(item, overlay)}
        runs.append(
            {
                "run": run_id,
                "n_train_questions_with_TM_silver": len(train_q),
                "n_train_TM_chunks": sum(len(tm_silver_ids(item, overlay)) for item in train),
                "n_train_silver_target_chunks": sum(len(p_silver_target_ids(item, overlay)) for item in train),
                "n_eval_questions_with_target_and_TM_silver": len(eval_both),
            }
        )
    return {"runs": runs}


def readiness_checklist(
    *,
    validation,
    gates,
    inventory,
    counts,
    n_ambiguous: int,
    n_stage_a: int,
    freeze_ready: bool,
    min_train: int,
    qa_frozen: bool,
) -> dict[str, Any]:
    train_ok = all(int(run["n_train_questions_with_TM_silver"]) >= int(min_train) for run in inventory.get("runs") or []) if inventory.get("runs") else False
    items = {
        "qa_frozen": qa_frozen,
        "all_ambiguous_stage_a": n_stage_a >= n_ambiguous and n_ambiguous > 0,
        "stage_a_gate": bool(gates.get("passed")),
        "schema_ok": bool(validation.get("ok")),
        "TM_silver_gt_0": int(counts.get("n_high_silver_target_mismatch") or 0) > 0,
        "each_outer_train_meets_min_TM_silver": train_ok,
        "snapshot_frozen": freeze_ready,
    }
    blocking = [key for key, ok in items.items() if key != "snapshot_frozen" and not ok]
    return {
        "items": items,
        "blocking": blocking,
        "v3_1_training_allowed": not blocking,
        "note": "QA-based silver is weak supervision, not verified gold",
    }
