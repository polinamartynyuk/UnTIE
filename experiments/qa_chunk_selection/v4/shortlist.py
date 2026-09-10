"""Load historical V3-B0 OOF scores. Do not retrain B0-live."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3.split import records_for_role

from .io_guard import read_jsonl


def load_b0_oof(v3_dir: Path) -> list[dict[str, Any]]:
    path = Path(v3_dir) / "analysis" / "oof" / "b0" / "predictions_scored.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"V3-B0 OOF missing at {path}")
    return read_jsonl(path)


def index_b0_oof(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        out[(str(row["question_id"]), int(row["seed"]))] = dict(row)
    return out


def scores_from_row(row: Mapping[str, Any]) -> dict[int, float]:
    raw = row.get("scores") or {}
    return {int(key): float(value) for key, value in raw.items()}


def ranked_ids_from_row(row: Mapping[str, Any]) -> list[int]:
    if row.get("ranked_ids"):
        return [int(item) for item in row["ranked_ids"]]
    scored = sorted(scores_from_row(row).items(), key=lambda item: (-item[1], item[0]))
    return [cid for cid, _score in scored]


def top_k_ids(row: Mapping[str, Any], k: int) -> list[int]:
    return ranked_ids_from_row(row)[: max(int(k), 0)]


def b0_rows_for_seed(index: Mapping[tuple[str, int], Mapping[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    return {qid: dict(row) for (qid, row_seed), row in index.items() if int(row_seed) == int(seed)}


def eval_question_ids(unlocked: Sequence[InstanceRecord], splits: Mapping[str, Any], run: int) -> set[str]:
    return {item.question_id for item in records_for_role(unlocked, splits, run=run, role="eval")}


def role_records(unlocked: Sequence[InstanceRecord], splits: Mapping[str, Any], run: int, role: str):
    if not splits:
        return tuple(unlocked)
    return records_for_role(unlocked, splits, run=run, role=role)


def group_oof_by_run(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("cv_run") or 0)].append(dict(row))
    return dict(grouped)
