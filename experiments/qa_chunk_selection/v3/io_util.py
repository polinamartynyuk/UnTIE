"""V3 write isolation. E0 and V2 artifact trees are protected."""

from __future__ import annotations

from pathlib import Path

from experiments.qa_chunk_selection.io_util import (  # noqa: F401
    PROJECT_ROOT,
    append_jsonl,
    load_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

from .contract import E0_ARTIFACT_ROOT, PROTECTED_WRITE_ROOTS, V2_ARTIFACT_ROOT, V3_ARTIFACT_ROOT

E0_OUTPUT = PROJECT_ROOT / E0_ARTIFACT_ROOT
V2_OUTPUT = PROJECT_ROOT / V2_ARTIFACT_ROOT
DEFAULT_V3_OUTPUT = PROJECT_ROOT / V3_ARTIFACT_ROOT


def assert_isolated_output(path: Path, project_root: Path | None = None) -> Path:
    resolved = path.resolve()
    root = (project_root or PROJECT_ROOT).resolve()
    for relative in PROTECTED_WRITE_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                "qa_chunk_selection_v3 refuses to write into a protected directory: "
                f"{resolved}"
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
