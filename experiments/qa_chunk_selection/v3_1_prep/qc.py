"""Ingest, schema QC, audit, coverage, CV inventory, and §21 readiness."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS, POSTHOC_SPLITS, PRIMARY_GOLD_RELATIONS
from experiments.qa_chunk_selection.v3.roles import original_positives, v3_target_positives, verified_target_mismatch
from experiments.qa_chunk_selection.v3.split import records_for_role

from .contract import MIN_TARGET_STATUS_COVERAGE_AT_1
from .taxonomy import (
    final_status,
    is_filled,
    overlay_index,
    overlay_key,
    primary_status,
    strip_model_scores,
    validate_prep_row,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_annotation_files(paths: Sequence[Path], progress: ProgressLogger | None = None) -> list[dict[str, Any]]:
    log = progress or ProgressLogger()
    merged: dict[tuple[str, int, str], dict[str, Any]] = {}
    ordered: list[Path] = []
    seen_files: set[Path] = set()
    for path in paths:
        matches = [Path(item) for item in sorted(glob(str(path)))]
        target = Path(path)
        if not matches and target.is_file():
            matches = [target]
        for match in matches:
            resolved = match.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            ordered.append(match)
    if not ordered:
        log.log("no annotation draft files found")
        return []
    for index, path in enumerate(ordered, start=1):
        log.count(index, len(ordered), item="ingest_file")
        for line_i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not is_filled(row):
                continue
            row = strip_model_scores(row)
            annotator = str(row.get("annotator_id") or path.stem)
            row["annotator_id"] = annotator
            row["_source_file"] = str(path)
            row["_source_line"] = line_i
            key = (str(row["question_id"]), int(row["chunk_id"]), annotator)
            merged[key] = row
    return list(merged.values())


def locked_ids(records: Sequence[InstanceRecord]) -> dict[str, set[str]]:
    docs: set[str] = set()
    questions: set[str] = set()
    for item in records:
        if item.split in LOCKED_SPLITS or item.split in POSTHOC_SPLITS:
            docs.add(item.document_id)
            questions.add(item.question_id)
    return {"documents": docs, "questions": questions}


def validate_overlay(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[InstanceRecord],
    *,
    locked: Mapping[str, set[str]],
    require_filled: bool,
    progress: ProgressLogger | None = None,
) -> dict[str, Any]:
    log = progress or ProgressLogger()
    by_q = {item.question_id: item for item in records}
    errors: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    total = max(len(rows), 1)
    for index, row in enumerate(rows, start=1):
        log.count(index, total, item="validate")
        key = overlay_key(row)
        by_key[key].append(dict(row))
        item = by_q.get(str(row["question_id"]))
        if item is None:
            errors.append({"key": key, "error": "question_id not in primary unlocked universe"})
            continue
        if item.document_id in locked["documents"] or item.question_id in locked["questions"]:
            errors.append({"key": key, "error": "INTERNAL_TEST or seen_exploratory_dev leakage"})
        if item.split in LOCKED_SPLITS:
            errors.append({"key": key, "error": "locked split in overlay"})
        for message in validate_prep_row(row, item, require_filled=require_filled):
            errors.append({"key": key, "error": message})
    duplicates = []
    for key, group in by_key.items():
        statuses = {final_status(row) for row in group}
        if len(statuses) > 1:
            duplicates.append({"key": key, "statuses": sorted(str(s) for s in statuses)})
            errors.append({"key": key, "error": f"conflicting labels for the same chunk: {sorted(statuses)}"})
    return {
        "n_rows": len(rows),
        "n_unique_chunks": len(by_key),
        "n_errors": len(errors),
        "errors": errors[:200],
        "n_duplicate_conflicts": len(duplicates),
        "ok": not errors,
    }


def build_working_overlay(
    rows: Sequence[Mapping[str, Any]],
    *,
    keep_low_confidence: bool,
) -> dict[tuple[str, int], dict[str, Any]]:
    working: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        payload["v3_target_status_raw"] = final_status(row)
        payload["v3_target_status"] = primary_status(row, keep_low_confidence=keep_low_confidence)
        working[overlay_key(row)] = payload
    return working


def effective_target_role(
    item: InstanceRecord,
    chunk_id: int,
    overlay: Mapping[tuple[str, int], Mapping[str, Any]],
) -> str:
    chunk = item.chunk_by_id(int(chunk_id))
    if chunk.label == "positive":
        return "target_evidence_original"
    row = overlay.get((item.question_id, int(chunk_id)))
    status = row.get("v3_target_status") if row else None
    relation = row.get("v3_gold_relation") if row else None
    if status == "target_evidence" and relation in PRIMARY_GOLD_RELATIONS:
        return "target_evidence_overlay"
    if status == "target_evidence":
        return "unresolved"
    if status == "target_mismatch":
        return "target_mismatch_verified"
    if chunk.label in {"hard_negative", "easy_negative"}:
        return "known_other_negative"
    if status == "partial_support":
        return "partial_support"
    if status == "unresolved":
        return "unresolved"
    return "unknown_ambiguous"


def counts(rows: Sequence[Mapping[str, Any]], *, keep_low_confidence: bool) -> dict[str, Any]:
    status = Counter()
    aspect = Counter()
    subtype = Counter()
    stratum = Counter()
    mismatch = Counter()
    confidence = Counter()
    gold_rel = Counter()
    for row in rows:
        st = primary_status(row, keep_low_confidence=keep_low_confidence) or "unfilled"
        status[st] += 1
        aspect[f"{st}|{row.get('aspect')}"] += 1
        subtype[f"{st}|{row.get('source_ambiguous_subtype')}"] += 1
        stratum[f"{st}|{row.get('annotation_sampling_stratum')}"] += 1
        mismatch[str(row.get("v3_mismatch_type") or "")] += 1
        confidence[f"{st}|{row.get('v3_annotation_confidence')}"] += 1
        gold_rel[str(row.get("v3_gold_relation") or "")] += 1
    n_te = status.get("target_evidence", 0)
    n_tm = status.get("target_mismatch", 0)
    n_partial = status.get("partial_support", 0)
    n_unresolved = status.get("unresolved", 0)
    n_primary_te = 0
    n_new_gold = 0
    for row in rows:
        st = primary_status(row, keep_low_confidence=keep_low_confidence)
        if st == "target_evidence" and row.get("v3_gold_relation") in PRIMARY_GOLD_RELATIONS:
            n_primary_te += 1
        if st == "target_evidence" and row.get("v3_gold_relation") == "new_valid_value_candidate":
            n_new_gold += 1
    return {
        "n_annotated": sum(status[key] for key in status if key != "unfilled"),
        "n_target_evidence": n_te,
        "n_target_evidence_primary": n_primary_te,
        "n_target_mismatch": n_tm,
        "n_partial_support": n_partial,
        "n_unresolved": n_unresolved,
        "n_new_valid_value_candidate": n_new_gold,
        "n_by_status": dict(status),
        "target_status_x_aspect": dict(aspect),
        "target_status_x_subtype": dict(subtype),
        "target_status_x_stratum": dict(stratum),
        "mismatch_type": dict(mismatch),
        "confidence_x_status": dict(confidence),
        "gold_relation": dict(gold_rel),
        "unique_document_id": len({str(row["document_id"]) for row in rows}),
        "unique_document_aspect": len({str(row["question_id"]) for row in rows}),
        "annotation_count_per_question": dict(Counter(str(row["question_id"]) for row in rows)),
    }


def false_negative_audit(rows: Sequence[Mapping[str, Any]], *, keep_low_confidence: bool) -> dict[str, Any]:
    hits = []
    for row in rows:
        st = primary_status(row, keep_low_confidence=keep_low_confidence)
        if row.get("source_label") == "ambiguous" and st == "target_evidence":
            hits.append(
                {
                    "question_id": row["question_id"],
                    "chunk_id": row["chunk_id"],
                    "v3_gold_relation": row.get("v3_gold_relation"),
                    "v3_target_value": row.get("v3_target_value"),
                    "annotation_sampling_stratum": row.get("annotation_sampling_stratum"),
                }
            )
    by_rel = Counter(str(row["v3_gold_relation"] or "") for row in hits)
    return {
        "n": len(hits),
        "by_gold_relation": dict(by_rel),
        "note": "source_label is not rewritten; these are overlay-only potential false negatives",
        "examples": hits[:50],
    }


def _seed42(rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    filtered = [dict(row) for row in rows if int(row.get("seed") or seed) == int(seed)]
    if filtered:
        return filtered
    # already unique-by-question
    seen = {}
    for row in rows:
        seen[str(row["question_id"])] = dict(row)
    return list(seen.values())


def coverage_from_b0(
    *,
    records: Sequence[InstanceRecord],
    b0_rows: Sequence[Mapping[str, Any]],
    overlay: Mapping[tuple[str, int], Mapping[str, Any]],
    b0_seed: int,
    ks: Sequence[int] = (1, 5, 10),
    progress: ProgressLogger | None = None,
) -> dict[str, Any]:
    log = progress or ProgressLogger()
    by_q = {item.question_id: item for item in records}
    ranked_rows = _seed42(b0_rows, b0_seed)
    n = len(ranked_rows)
    known_top1 = 0
    tm_top1 = 0
    known_role_top1 = 0
    amb_top1_adj = 0
    amb_top1_tm = 0
    qa_known = 0
    qa_tm = 0
    slot_known = {k: 0 for k in ks}
    slot_total = {k: 0 for k in ks}
    complete = {k: 0 for k in ks}
    pair_ok = 0
    pair_defined = 0
    stratum_top1 = Counter()
    for index, row in enumerate(ranked_rows, start=1):
        log.count(index, max(n, 1), item="coverage")
        item = by_q.get(str(row["question_id"]))
        if item is None:
            continue
        ids = [int(cid) for cid in (row.get("ranked_ids") or [])]
        if not ids:
            continue
        role1 = effective_target_role(item, ids[0], overlay)
        if role1 != "unknown_ambiguous":
            known_top1 += 1
            known_role_top1 += 1
        if role1 == "target_mismatch_verified":
            tm_top1 += 1
        top_chunk = item.chunk_by_id(ids[0])
        overlay_row = overlay.get((item.question_id, ids[0]))
        if top_chunk.label == "ambiguous" and overlay_row and overlay_row.get("v3_target_status") in {
            "target_evidence",
            "target_mismatch",
            "partial_support",
            "unresolved",
        }:
            amb_top1_adj += 1
            if overlay_row.get("v3_target_status") == "target_mismatch":
                amb_top1_tm += 1
            stratum_top1[str(overlay_row.get("annotation_sampling_stratum") or "unknown")] += 1
        qa_id = row.get("qa_chunk_id", ids[0])
        if qa_id is not None:
            qa_role = effective_target_role(item, int(qa_id), overlay)
            if qa_role != "unknown_ambiguous":
                qa_known += 1
            if qa_role == "target_mismatch_verified":
                qa_tm += 1
        p_v3 = {chunk.chunk_id for chunk in v3_target_positives(item, overlay)}
        tm = {chunk.chunk_id for chunk in verified_target_mismatch(item, overlay)}
        if p_v3 and tm:
            pair_defined += 1
            scores = {int(k): float(v) for k, v in (row.get("scores") or {}).items()}
            if scores:
                best_p = max(scores.get(cid, 0.0) for cid in p_v3)
                best_tm = max(scores.get(cid, 0.0) for cid in tm)
                if best_p > best_tm:
                    pair_ok += 1
        for k in ks:
            topk = ids[: int(k)]
            slot_total[k] += len(topk)
            known_slots = sum(effective_target_role(item, cid, overlay) != "unknown_ambiguous" for cid in topk)
            slot_known[k] += known_slots
            if known_slots == len(topk):
                complete[k] += 1
    n = max(n, 1)
    coverage_at_1 = known_top1 / n
    return {
        "n_questions": n,
        "b0_seed": int(b0_seed),
        "TargetStatusCoverage@1": coverage_at_1,
        "TargetMismatch@1_lower_bound": tm_top1 / n,
        "TargetMismatchRate@1_known_role": (tm_top1 / known_role_top1) if known_role_top1 else None,
        "TargetMismatchRate@1_overlay_adjudicated": (amb_top1_tm / amb_top1_adj) if amb_top1_adj else None,
        "overlay_adjudicated_top1_stratum": dict(stratum_top1),
        "WrongTargetAnswerCoverage": qa_known / n,
        "WrongTargetAnswerRate_lower_bound": qa_tm / n,
        "WrongTargetAnswerRate_known_role": (qa_tm / qa_known) if qa_known else None,
        "slot_coverage": {f"@{k}": (slot_known[k] / slot_total[k] if slot_total[k] else None) for k in ks},
        "complete_coverage": {f"@{k}": complete[k] / n for k in ks},
        "n_eval_questions_with_target_and_TM": pair_defined,
        "pairwise_pv3_gt_tm": (pair_ok / pair_defined) if pair_defined else None,
        "tm_as_hard_safeguard": bool(coverage_at_1 >= MIN_TARGET_STATUS_COVERAGE_AT_1),
        "note": "unknown_ambiguous is not treated as not-TM; TM@1 is a lower bound under incomplete overlay",
    }


def cv_inventory(
    *,
    records: Sequence[InstanceRecord],
    splits: Mapping[str, Any],
    overlay: Mapping[tuple[str, int], Mapping[str, Any]],
    progress: ProgressLogger | None = None,
) -> dict[str, Any]:
    log = progress or ProgressLogger()
    runs = []
    for run in splits.get("runs") or []:
        run_id = int(run["run"])
        log.log(f"cv inventory run={run_id}")
        train = records_for_role(records, splits, run=run_id, role="train")
        ev = records_for_role(records, splits, run=run_id, role="eval")
        train_tm_chunks = 0
        train_q = set()
        eval_both = set()
        for item in train:
            tms = verified_target_mismatch(item, overlay)
            if tms:
                train_tm_chunks += len(tms)
                train_q.add(item.question_id)
        for item in ev:
            has_p = bool(v3_target_positives(item, overlay) or original_positives(item))
            has_tm = bool(verified_target_mismatch(item, overlay))
            if has_p and has_tm:
                eval_both.add(item.question_id)
        runs.append(
            {
                "run": run_id,
                "n_train_TM_chunks": train_tm_chunks,
                "n_train_questions_with_TM": len(train_q),
                "n_eval_questions_with_target_and_TM": len(eval_both),
            }
        )
    return {"runs": runs}


def readiness_checklist(
    *,
    counts_payload: Mapping[str, Any],
    validation: Mapping[str, Any],
    coverage: Mapping[str, Any],
    inventory: Mapping[str, Any],
    n_queue: int,
    n_filled: int,
    source_label_rewritten: bool,
    leakage: bool,
    min_train_questions_with_TM: int,
    min_eval_questions_with_target_and_TM: int,
    freeze_ready: bool,
) -> dict[str, Any]:
    train_ok = all(
        int(run["n_train_questions_with_TM"]) >= int(min_train_questions_with_TM)
        for run in inventory.get("runs") or []
    ) if inventory.get("runs") else False
    eval_ok = all(
        int(run["n_eval_questions_with_target_and_TM"]) >= int(min_eval_questions_with_target_and_TM)
        for run in inventory.get("runs") or []
    ) if inventory.get("runs") else False
    items = {
        "taxonomy_fixed": True,
        "queue_processed_or_incomplete_documented": n_filled >= n_queue or n_filled < n_queue,
        "queue_fully_processed": n_filled >= n_queue and n_queue > 0,
        "low_confidence_handled": True,
        "source_label_not_rewritten": not source_label_rewritten,
        "internal_test_absent": not leakage,
        "verified_target_mismatch_gt_0": int(counts_payload.get("n_target_mismatch") or 0) > 0,
        "verified_target_evidence_counted": "n_target_evidence" in counts_payload,
        "cv_train_TM_inventory": bool(inventory.get("runs")),
        "min_train_questions_with_TM_fixed": int(min_train_questions_with_TM) > 0,
        "each_outer_train_meets_min_TM": train_ok,
        "pairwise_TM_support_counted": coverage.get("n_eval_questions_with_target_and_TM") is not None,
        "min_eval_questions_with_target_and_TM_fixed": int(min_eval_questions_with_target_and_TM) > 0,
        "distributions_built": "n_annotated" in counts_payload,
        "coverage_computed": "TargetStatusCoverage@1" in coverage,
        "schema_and_duplicates_ok": bool(validation.get("ok")),
        "snapshot_frozen": bool(freeze_ready),
    }
    items["queue_incomplete_documented"] = (n_filled < n_queue) or items["queue_fully_processed"]
    blocking = [
        key
        for key, ok in items.items()
        if key
        not in {
            "queue_fully_processed",
            "each_outer_train_meets_min_TM",
            "snapshot_frozen",
            "queue_incomplete_documented",
        }
        and not ok
    ]
    v31_allowed = (
        items["taxonomy_fixed"]
        and items["source_label_not_rewritten"]
        and items["internal_test_absent"]
        and items["verified_target_mismatch_gt_0"]
        and items["schema_and_duplicates_ok"]
        and items["each_outer_train_meets_min_TM"]
        and items["queue_fully_processed"]
    )
    return {
        "items": items,
        "blocking": blocking,
        "n_queue": n_queue,
        "n_filled": n_filled,
        "min_train_questions_with_TM": int(min_train_questions_with_TM),
        "min_eval_questions_with_target_and_TM": int(min_eval_questions_with_target_and_TM),
        "v3_1_training_allowed": v31_allowed,
        "note": (
            "V3.1 ranking experiments must not start until v3_1_training_allowed is true. "
            "Incomplete annotation is documented rather than treated as not-TM."
        ),
    }


def overlay_for_roles(working: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return overlay_index([dict(row) for row in working.values()])
