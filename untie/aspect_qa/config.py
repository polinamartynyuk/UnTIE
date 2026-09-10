"""Experiment configuration for SciREX aspect-level extractive QA preprocessing.

Questions and flags are experimental configuration, not additional document facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


ASPECT_FIELD_MAP: dict[str, str] = {
    "method": "methods",
    "task": "tasks",
    "metric": "metrics",
    "material": "materials",
}

ASPECT_QUESTIONS: dict[str, str] = {
    "method": "What method or model is used or proposed in the paper?",
    "task": "What task is addressed in the paper?",
    "metric": "What evaluation metric is used in the paper?",
    "material": "What dataset or material is used in the paper?",
}

EXPECTED_SOURCE_FIELDS: tuple[str, ...] = (
    "doc_id",
    "methods",
    "tasks",
    "metrics",
    "materials",
    "score",
    "original_text",
)

# SciREX serialization labels observed in original_text, plus the prompt set.
STRUCTURAL_MARKERS: tuple[str, ...] = (
    "document",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "bibliography",
    "appendix",
)

BIBLIOGRAPHY_MARKER = "bibliography"


@dataclass(frozen=True)
class AspectQAPreprocessConfig:
    aspect_questions: Mapping[str, str] = field(default_factory=lambda: dict(ASPECT_QUESTIONS))
    aspect_field_map: Mapping[str, str] = field(default_factory=lambda: dict(ASPECT_FIELD_MAP))
    enable_score_aspect: bool = False
    include_bibliography_in_qa: bool = False
    enrichment_mode: str = "E0"
    split_seed: int = 42
    train_fraction: float = 0.6
    dev_fraction: float = 0.2
    # When set, every document is assigned this official corpus split
    # instead of the hash-based internal split of a single JSONL file.
    fixed_split: str | None = None
    chunk_max_tokens: int | None = None
    overlap_tokens: int | None = None
    model_max_length: int = 512

    def __post_init__(self) -> None:
        if self.enrichment_mode != "E0":
            raise ValueError("baseline preprocessing must use enrichment_mode='E0'")
        if self.enable_score_aspect:
            raise ValueError("ENABLE_SCORE_ASPECT must stay False until relation labels exist")
        if self.fixed_split is not None and self.fixed_split not in {"train", "dev", "test"}:
            raise ValueError(f"fixed_split must be train, dev, or test, got {self.fixed_split!r}")


def default_output_dir(project_root: Path) -> Path:
    return project_root / "datasets" / "scirex_aspect_qa_preprocessing"


def official_split_output_dir(project_root: Path, split: str) -> Path:
    if split == "train":
        return default_output_dir(project_root)
    return project_root / "datasets" / f"scirex_aspect_qa_preprocessing_{split}"
