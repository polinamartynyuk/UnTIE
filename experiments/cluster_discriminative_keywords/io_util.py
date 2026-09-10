"""Output isolation for hyp4. Never write into frozen artifact trees."""

from __future__ import annotations

from pathlib import Path

from experiments.keybert_topic_clustering.io_util import (  # noqa: F401
    append_timing,
    write_csv,
    write_json,
)


PROTECTED_ROOTS = (
    "artifacts/hierarchical_topic_optimized",
    "artifacts/keybert_topic_clustering",
    "artifacts/keybert_qa_integration",
    "artifacts/answer_aggregation",
)


def assert_isolated_output(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    root = project_root.resolve()
    for relative in PROTECTED_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                "cluster discriminative experiment refuses to write into a "
                f"protected directory: {resolved}"
            )
        try:
            resolved.relative_to(protected)
        except ValueError:
            continue
        else:
            raise ValueError(
                "cluster discriminative experiment refuses to write into a "
                f"protected directory: {resolved}"
            )
    return resolved
