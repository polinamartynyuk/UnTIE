"""Load fold-specific QA-silver overlays. Never mix another run's CAL thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.qa_chunk_selection.v3_1_auto_prep.qc import file_sha256
from experiments.qa_chunk_selection.v3_1_auto_prep.silver_roles import overlay_index, silver_to_v3_training_overlay

from .io_guard import read_jsonl

REQUIRED_SILVER_FILES = (
    "annotation_qa_silver/qa_model_manifest.json",
    "annotation_qa_silver/aspect_questions.json",
    "annotation_qa_silver/attribution_questions.json",
    "annotation_qa_silver/stage_a_all_ambiguous.jsonl",
    "annotation_qa_silver/stage_b_paraphrases.jsonl",
    "annotation_qa_silver/stage_b_ownership.jsonl",
    "annotation_qa_silver/discourse_features.jsonl",
    "annotation_qa_silver/qc_summary.json",
    "annotation_qa_silver/silver_manifest.json",
)


def silver_root(silver_dir: Path) -> Path:
    return Path(silver_dir) / "annotation_qa_silver"


def missing_freeze_files(silver_dir: Path) -> list[str]:
    root = Path(silver_dir)
    missing = [name for name in REQUIRED_SILVER_FILES if not (root / name).exists()]
    oof = silver_root(root) / "oof"
    if not oof.exists() or not list(oof.glob("run*/silver_target_status.jsonl")):
        missing.append("annotation_qa_silver/oof/run*/silver_target_status.jsonl")
    return missing


def load_fold_overlays(silver_dir: Path) -> dict[int, list[dict[str, Any]]]:
    root = silver_root(silver_dir) / "oof"
    overlays: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("run*/silver_target_status.jsonl")):
        run_id = int(path.parent.name.replace("run", ""))
        overlays[run_id] = read_jsonl(path)
    if not overlays:
        raise FileNotFoundError(f"no fold-specific silver overlays under {root}")
    return overlays


def index_fold_overlays(rows_by_run: dict[int, list[dict[str, Any]]]) -> tuple[dict[int, dict], dict[int, dict]]:
    silver = {run_id: overlay_index(rows) for run_id, rows in rows_by_run.items()}
    training = {run_id: silver_to_v3_training_overlay(rows) for run_id, rows in rows_by_run.items()}
    return silver, training


def overlay_hashes(silver_dir: Path, rows_by_run: dict[int, list[dict[str, Any]]]) -> dict[str, str]:
    root = silver_root(silver_dir)
    payload = {
        "silver_manifest": file_sha256(root / "silver_manifest.json") if (root / "silver_manifest.json").exists() else "",
        "qc_summary": file_sha256(root / "qc_summary.json") if (root / "qc_summary.json").exists() else "",
        "qa_model_manifest": file_sha256(root / "qa_model_manifest.json") if (root / "qa_model_manifest.json").exists() else "",
    }
    for run_id in sorted(rows_by_run):
        path = root / "oof" / f"run{run_id}" / "silver_target_status.jsonl"
        if path.exists():
            payload[f"run{run_id}"] = file_sha256(path)
    return payload
