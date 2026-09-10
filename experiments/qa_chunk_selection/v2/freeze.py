"""INTERNAL_TEST lock and freeze manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .config import ExperimentConfigV2
from .contract import EVAL_SPLITS_BEFORE_FREEZE
from .io_util import load_json, write_json


def git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def freeze_path(output_dir: Path) -> Path:
    return Path(output_dir) / "final_freeze_manifest.json"


def write_freeze_manifest(config: ExperimentConfigV2, extra: Mapping[str, Any]) -> Path:
    from .io_util import PROJECT_ROOT

    payload = {
        "config_hash": config.config_hash(),
        "resolved_config": config.to_dict(),
        "git_commit": git_commit(PROJECT_ROOT),
        "seeds": {
            "split_seed": config.split_seed,
            "model_seeds": list(config.model_seeds),
            "bootstrap_seed": config.bootstrap_seed,
        },
        **dict(extra),
    }
    path = freeze_path(config.output_dir)
    write_json(path, payload)
    return path


def load_freeze_manifest(output_dir: Path) -> dict[str, Any]:
    path = freeze_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"missing freeze manifest: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("freeze manifest must be an object")
    return payload


def assert_eval_split_allowed(
    split: str,
    *,
    allow_internal_test: bool,
    output_dir: Path,
    require_freeze: bool = True,
) -> None:
    if split != "internal_test":
        if split not in EVAL_SPLITS_BEFORE_FREEZE and split != "internal_test":
            return
        return
    if not allow_internal_test:
        raise PermissionError(
            "INTERNAL_TEST is locked. Pass --allow-internal-test after freeze-final."
        )
    if require_freeze and not freeze_path(output_dir).exists():
        raise PermissionError("INTERNAL_TEST requires artifacts/qa_chunk_selection_v2/final_freeze_manifest.json")


def verify_freeze_hash(config: ExperimentConfigV2) -> None:
    manifest = load_freeze_manifest(config.output_dir)
    expected = str(manifest.get("config_hash") or "")
    actual = config.config_hash()
    if expected and expected != actual:
        raise ValueError(
            f"freeze manifest config_hash {expected} does not match current config {actual}"
        )


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
