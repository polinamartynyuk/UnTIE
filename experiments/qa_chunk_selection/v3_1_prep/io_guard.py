"""Write isolation: never touch E0, V2, V3, production, or Phase 1 trees."""

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

from .contract import PROTECTED_WRITE_ROOTS, V3_ARTIFACT_ROOT

__all__ = [
    "PROJECT_ROOT",
    "assert_isolated_output",
    "assert_v3_not_output",
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
                "v3_1_prep refuses to write into a protected directory: "
                f"{resolved} (protected={relative})"
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def assert_v3_not_output(output_dir: Path, v3_dir: Path) -> None:
    out = Path(output_dir).resolve()
    v3 = Path(v3_dir).resolve()
    if out == v3 or v3 in out.parents or out in v3.parents:
        raise ValueError(
            "V3.1 data-prep must not share an output tree with V3. "
            f"output={out} v3={v3}. Use artifacts/qa_chunk_selection_v3_1_prep/."
        )
    default_v3 = (PROJECT_ROOT / V3_ARTIFACT_ROOT).resolve()
    if out == default_v3 or default_v3 in out.parents:
        raise ValueError(f"refusing to write V3.1 prep into the V3 artifact tree: {out}")
