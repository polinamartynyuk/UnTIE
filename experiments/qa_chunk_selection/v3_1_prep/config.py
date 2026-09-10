"""CLI/config for the isolated V3.1 overlay preparation workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .contract import (
    DEFAULT_MIN_EVAL_QUESTIONS_WITH_TARGET_AND_TM,
    DEFAULT_MIN_TRAIN_QUESTIONS_WITH_TM,
    V3_1_PREP_ARTIFACT_ROOT,
    V3_ARTIFACT_ROOT,
)

DEFAULT_STAGES = (
    "init",
    "export-annotator",
    "ingest",
    "validate",
    "audit",
    "coverage",
    "cv-inventory",
    "readiness",
)

ALL_STAGES = DEFAULT_STAGES + ("expand-coverage", "freeze")


@dataclass(frozen=True)
class PrepConfig:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_1_PREP_ARTIFACT_ROOT)
    v3_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_ARTIFACT_ROOT)
    stages: tuple[str, ...] = DEFAULT_STAGES
    seed: int = 42
    b0_seed: int = 42
    annotator_id: str = "annotator_1"
    coverage_completion_n: int = 0
    coverage_completion_k: int = 10
    min_train_questions_with_TM: int = DEFAULT_MIN_TRAIN_QUESTIONS_WITH_TM
    min_eval_questions_with_target_and_TM: int = DEFAULT_MIN_EVAL_QUESTIONS_WITH_TARGET_AND_TM
    allow_partial_freeze: bool = False
    keep_low_confidence: bool = False
    annotations_glob: str = "drafts/*.jsonl"
    smoke: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["v3_artifact_dir"] = str(self.v3_artifact_dir)
        payload["stages"] = list(self.stages)
        return payload


def parse_stages(raw: str) -> tuple[str, ...]:
    text = (raw or "all").strip().lower()
    if text in {"all", "default"}:
        return DEFAULT_STAGES
    if text == "full":
        return DEFAULT_STAGES + ("freeze",)
    parts = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = [item for item in parts if item not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown V3.1 prep stages: {unknown}; allowed={list(ALL_STAGES)}")
    return parts


def v3_paths(v3_dir: Path) -> dict[str, Path]:
    root = Path(v3_dir)
    return {
        "root": root,
        "queue": root / "annotation" / "queue.jsonl",
        "metadata": root / "annotation" / "sampling_metadata.jsonl",
        "taxonomy": root / "annotation" / "taxonomy.json",
        "target_status": root / "annotation" / "target_status.jsonl",
        "sampling_summary": root / "annotation" / "sampling_summary.json",
        "dataset": root / "data" / "dataset.jsonl",
        "splits": root / "data" / "v3_cv_splits.json",
        "prepare_summary": root / "data" / "prepare_summary.json",
        "b0_oof": root / "analysis" / "oof" / "b0" / "predictions_scored.jsonl",
    }
