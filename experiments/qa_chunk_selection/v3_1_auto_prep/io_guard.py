"""Write isolation for QA-based silver prep."""

from __future__ import annotations

from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.io_util import (
    append_jsonl,
    load_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

from .contract import PROTECTED_WRITE_ROOTS, V3_1_PREP_ARTIFACT_ROOT, V3_1_RANKING_ARTIFACT_ROOT, V3_ARTIFACT_ROOT

__all__ = [
    "PROJECT_ROOT",
    "assert_isolated_output",
    "load_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
    "append_jsonl",
]


def assert_isolated_output(path: Path, project_root: Path | None = None) -> Path:
    resolved = path.resolve()
    root = (project_root or PROJECT_ROOT).resolve()
    for relative in PROTECTED_WRITE_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                "v3_1_auto_prep refuses to write into a protected directory: "
                f"{resolved} (protected={relative})"
            )
    for forbidden in (
        root / V3_ARTIFACT_ROOT,
        root / V3_1_PREP_ARTIFACT_ROOT,
        root / V3_1_RANKING_ARTIFACT_ROOT,
    ):
        target = forbidden.resolve()
        if resolved == target or target in resolved.parents:
            raise ValueError(f"refusing to write QA silver prep into {target}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
