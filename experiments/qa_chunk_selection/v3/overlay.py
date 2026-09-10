"""V3 target-status overlay. Never rewrites source_label."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord

from .contract import GSRN_CLASS, GOLD_RELATIONS, PRIMARY_GOLD_RELATIONS, TARGET_STATUS
from .io_util import read_jsonl, write_json, write_jsonl

TAXONOMY_VERSION = "v3-target-status-1"


def empty_overlay_row(item: InstanceRecord, chunk) -> dict[str, Any]:
    return {
        "document_id": item.document_id,
        "question_id": item.question_id,
        "chunk_id": chunk.chunk_id,
        "aspect": item.aspect,
        "source_label": chunk.label,
        "source_ambiguous_subtype": chunk.primary_class if chunk.label == "ambiguous" else None,
        "v3_target_status": None,
        "v3_target_value": None,
        "v3_gold_relation": None,
        "v3_annotation_confidence": None,
        "v3_annotation_reason": None,
        "annotation_sampling_stratum": None,
        "sampling_probability": None,
        "taxonomy_version": TAXONOMY_VERSION,
    }


def validate_overlay_row(row: Mapping[str, Any], gold: InstanceRecord) -> None:
    chunk = gold.chunk_by_id(int(row["chunk_id"]))
    if str(row.get("source_label")) != chunk.label:
        raise ValueError(
            f"overlay must not rewrite source_label: overlay={row.get('source_label')} source={chunk.label}"
        )
    status = row.get("v3_target_status")
    if status not in (None, "") and status not in TARGET_STATUS:
        raise ValueError(f"unknown v3_target_status: {status}")
    relation = row.get("v3_gold_relation")
    if relation not in (None, "") and relation not in GOLD_RELATIONS:
        raise ValueError(f"unknown v3_gold_relation: {relation}")


def load_overlay(path) -> dict[tuple[str, int], dict[str, Any]]:
    from pathlib import Path

    file = Path(path)
    if not file.exists():
        return {}
    rows = read_jsonl(file)
    index = {}
    for row in rows:
        key = (str(row["question_id"]), int(row["chunk_id"]))
        index[key] = dict(row)
    return index


def sample_annotation_queue(
    records: Sequence[InstanceRecord],
    *,
    scores: Mapping[tuple[str, int], float] | None = None,
    hard_n: int,
    repr_n: int,
    seed: int,
) -> dict[str, Any]:
    """Hard-case + representative strata. Scores stay in metadata, not annotator queue."""
    rng = random.Random(int(seed))
    ambiguous: list[dict[str, Any]] = []
    for item in records:
        scored = []
        if scores:
            for chunk in item.chunks:
                scored.append(float(scores.get((item.question_id, chunk.chunk_id), 0.0)))
        else:
            scored = [0.0] * len(item.chunks)
        order = sorted(range(len(item.chunks)), key=lambda i: -scored[i])
        top1 = item.chunks[order[0]].chunk_id if order else None
        top5 = {item.chunks[i].chunk_id for i in order[:5]}
        q_hi = _quantile(scored, 0.8) if scored else 0.0
        for chunk, score in zip(item.chunks, scored):
            if chunk.label != "ambiguous":
                continue
            row = empty_overlay_row(item, chunk)
            row.update(
                {
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "position": chunk.position,
                    "chunk_length": len(chunk.text),
                    "b_score": float(score),
                    "is_top1": chunk.chunk_id == top1,
                    "in_top5": chunk.chunk_id in top5,
                    "high_score": float(score) >= q_hi if scored else False,
                    "source_sentence": chunk.text,
                }
            )
            ambiguous.append(row)

    hard_keys = []
    hard_pool = []
    for row in ambiguous:
        hard = bool(
            row["is_top1"]
            or row["in_top5"]
            or row["high_score"]
            or row.get("source_ambiguous_subtype")
            in {GSRN_CLASS, "question_sensitive", "false_negative_candidate"}
        )
        if hard:
            hard_pool.append(row)
    hard_pool.sort(
        key=lambda row: (
            -int(row["is_top1"]),
            -int(row["in_top5"]),
            -float(row["b_score"]),
            row["question_id"],
            int(row["chunk_id"]),
        )
    )
    selected_hard = hard_pool[: int(hard_n)]
    hard_ids = {(row["question_id"], row["chunk_id"]) for row in selected_hard}

    remaining = [row for row in ambiguous if (row["question_id"], row["chunk_id"]) not in hard_ids]
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in remaining:
        key = f"{row['aspect']}|{row.get('source_ambiguous_subtype')}|len{ _len_bin(row['chunk_length'])}"
        strata[key].append(row)
    representative: list[dict[str, Any]] = []
    if remaining and repr_n > 0:
        keys = sorted(strata)
        per = max(1, int(math.ceil(repr_n / max(len(keys), 1))))
        for key in keys:
            pool = list(strata[key])
            rng.shuffle(pool)
            take = pool[:per]
            p = len(take) / max(len(pool), 1)
            for row in take:
                row = dict(row)
                row["sampling_probability"] = float(p)
                representative.append(row)
            if len(representative) >= repr_n:
                break
        representative = representative[: int(repr_n)]

    queue = []
    metadata = []
    for stratum, rows in (("hard_case", selected_hard), ("representative", representative)):
        for row in rows:
            meta = dict(row)
            meta["annotation_sampling_stratum"] = stratum
            metadata.append(meta)
            queue.append(
                {
                    "document_id": row["document_id"],
                    "question_id": row["question_id"],
                    "chunk_id": row["chunk_id"],
                    "aspect": row["aspect"],
                    "question": next(
                        item.question for item in records if item.question_id == row["question_id"]
                    ),
                    "chunk_text": next(
                        chunk.text
                        for item in records
                        if item.question_id == row["question_id"]
                        for chunk in item.chunks
                        if chunk.chunk_id == row["chunk_id"]
                    ),
                    "section_title": row.get("section_title"),
                    "source_label": row["source_label"],
                    "source_ambiguous_subtype": row.get("source_ambiguous_subtype"),
                    "gold_answers": list(
                        next(item.gold_answers for item in records if item.question_id == row["question_id"])
                    ),
                    "v3_target_status": None,
                    "v3_target_value": None,
                    "v3_gold_relation": None,
                    "v3_annotation_confidence": None,
                    "v3_annotation_reason": None,
                    "taxonomy_version": TAXONOMY_VERSION,
                }
            )
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "n_ambiguous": len(ambiguous),
        "n_hard_case": len(selected_hard),
        "n_representative": len(representative),
        "queue": queue,
        "metadata": metadata,
    }


def write_annotation_artifacts(output_dir, payload: Mapping[str, Any]) -> None:
    from pathlib import Path

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "queue.jsonl", payload["queue"])
    write_jsonl(root / "sampling_metadata.jsonl", payload["metadata"])
    write_jsonl(root / "target_status.jsonl", [])
    write_json(
        root / "taxonomy.json",
        {
            "taxonomy_version": TAXONOMY_VERSION,
            "target_match_status": list(TARGET_STATUS),
            "v3_gold_relation": list(GOLD_RELATIONS),
            "primary_gold_relations": sorted(PRIMARY_GOLD_RELATIONS),
            "annotator_must_not_see": ["b_score", "attention_score", "predicted_class", "model_rank"],
        },
    )
    write_json(
        root / "sampling_summary.json",
        {
            "taxonomy_version": payload["taxonomy_version"],
            "n_ambiguous": payload["n_ambiguous"],
            "n_hard_case": payload["n_hard_case"],
            "n_representative": payload["n_representative"],
        },
    )


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def _len_bin(length: int) -> str:
    if length < 80:
        return "short"
    if length < 240:
        return "medium"
    return "long"
