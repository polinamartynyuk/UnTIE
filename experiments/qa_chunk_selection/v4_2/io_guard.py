"""Write isolation: never touch E0, V2, V3, V3.1, D45, V4, or V4.1 trees."""

from __future__ import annotations

from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v4.io_guard import file_sha256, load_json, read_jsonl, write_json, write_jsonl

from .contract import PROTECTED_WRITE_ROOTS, V4_2_ARTIFACT_ROOT

COMPLETED_MARKERS = (
    "final/model_selection.json",
    "reports/results_v4_2.md",
    "att1/metrics.json",
)

__all__ = [
    "COMPLETED_MARKERS",
    "PROJECT_ROOT",
    "assert_isolated_output",
    "file_sha256",
    "is_completed_output",
    "load_json",
    "read_jsonl",
    "refuse_completed_output",
    "write_json",
    "write_jsonl",
]


def assert_isolated_output(path: Path, project_root: Path | None = None) -> Path:
    resolved = path.resolve()
    root = (project_root or PROJECT_ROOT).resolve()
    for relative in PROTECTED_WRITE_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                "V4.2 refuses to write into a protected directory: "
                f"{resolved} (protected={relative})"
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def is_completed_output(path: Path) -> bool:
    root = Path(path)
    return any((root / marker).exists() for marker in COMPLETED_MARKERS)


def refuse_completed_output(path: Path, *, allow: bool) -> None:
    if allow:
        return
    if is_completed_output(path):
        raise ValueError(
            "V4.2 refuses to overwrite a completed run at "
            f"{path.resolve()}. Set a new --output-dir or ALLOW_EXISTING_OUT=1."
        )


def default_output() -> Path:
    return PROJECT_ROOT / V4_2_ARTIFACT_ROOT
