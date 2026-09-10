from __future__ import annotations

from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.io_util import load_json, read_jsonl, write_json, write_jsonl

from .contract import PROTECTED_WRITE_ROOTS, V3_D45_ARTIFACT_ROOT

__all__ = ["PROJECT_ROOT", "assert_isolated_output", "load_json", "read_jsonl", "write_json", "write_jsonl"]


def assert_isolated_output(path: Path, project_root: Path | None = None) -> Path:
    resolved = path.resolve()
    root = (project_root or PROJECT_ROOT).resolve()
    for relative in PROTECTED_WRITE_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(f"v3_d45 refuses to write into {resolved} (protected={relative})")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def default_output() -> Path:
    return PROJECT_ROOT / V3_D45_ARTIFACT_ROOT
