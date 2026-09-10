"""Immutable overlay snapshot. Never writes into the V3 artifact tree."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.v3.freeze import stable_json_hash

from .contract import ANNOTATION_VERSION, GUIDELINE_VERSION, TAXONOMY_VERSION
from .io_guard import write_json, write_jsonl
from .qc import file_sha256


def freeze_overlay(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    root = Path(output_dir) / "annotation"
    frozen = root / "target_status_v3_1_frozen.jsonl"
    summary_path = root / "annotation_summary_v3_1.json"
    manifest_path = root / "annotation_manifest_v3_1.json"
    if frozen.exists() and not manifest.get("allow_overwrite_freeze"):
        raise PermissionError(
            f"freeze already exists at {frozen}. Create a new overlay version instead of editing in place."
        )
    write_jsonl(frozen, rows)
    write_json(summary_path, summary)
    payload = dict(manifest)
    payload.update(
        {
            "taxonomy_version": TAXONOMY_VERSION,
            "annotation_guideline_version": GUIDELINE_VERSION,
            "annotation_version": ANNOTATION_VERSION,
            "n_annotated": len(rows),
            "overlay_file": str(frozen),
            "overlay_file_hash": file_sha256(frozen),
            "freeze_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "immutable": True,
        }
    )
    payload["manifest_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    write_json(manifest_path, payload)
    return {
        "frozen": str(frozen),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "overlay_file_hash": payload["overlay_file_hash"],
    }
