"""Freeze QA-silver rules/hashes. Never overwrite an existing freeze."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.freeze import git_commit, stable_json_hash

from .contract import ANNOTATION_VERSION, PROTOCOL, QUESTION_SET_VERSION, SILVER_RULE_VERSION
from .io_guard import write_json
from .qc import file_sha256


def freeze_silver(output_dir: Path, *, qc_summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, str]:
    root = Path(output_dir) / "annotation_qa_silver"
    manifest_path = root / "silver_manifest.json"
    qc_path = root / "qc_summary.json"
    if manifest_path.exists() and not manifest.get("allow_overwrite_freeze"):
        raise PermissionError(
            f"silver freeze already exists at {manifest_path}. "
            "Use a new output directory for a new version."
        )
    write_json(qc_path, qc_summary)
    payload = dict(manifest)
    payload.update(
        {
            "protocol": PROTOCOL,
            "annotation_version": ANNOTATION_VERSION,
            "silver_rule_version": SILVER_RULE_VERSION,
            "question_set_version": QUESTION_SET_VERSION,
            "qc_summary_hash": file_sha256(qc_path),
            "freeze_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(PROJECT_ROOT),
            "immutable": True,
            "label_status": "silver",
            "annotator": "frozen_extractive_qa",
        }
    )
    payload["manifest_hash"] = stable_json_hash({k: v for k, v in payload.items() if k != "manifest_hash"})
    write_json(manifest_path, payload)
    return {"manifest": str(manifest_path), "qc_summary": str(qc_path), "manifest_hash": payload["manifest_hash"]}
