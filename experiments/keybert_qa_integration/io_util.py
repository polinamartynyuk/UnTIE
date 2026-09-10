"""Output isolation for hyp3. Never write into production or hyp2 artifact trees."""

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
    "en/evaluation",
    "en/refit",
    "en/refit_no_gold_contrast",
    "en/soft_rerank",
    "en/cross_encoder",
    "en/ce_topic_query",
    "en/ce_topic_query_body",
)


def assert_isolated_output(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    root = project_root.resolve()
    for relative in PROTECTED_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                "keybert QA integration refuses to write into a protected "
                f"directory: {resolved}"
            )
        try:
            resolved.relative_to(protected)
        except ValueError:
            continue
        else:
            raise ValueError(
                "keybert QA integration refuses to write into a protected "
                f"directory: {resolved}"
            )
    return resolved
