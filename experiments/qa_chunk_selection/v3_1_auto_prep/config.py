"""Config for isolated QA-based silver overlay preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .contract import (
    DEFAULT_MAX_EASY_PASS,
    DEFAULT_MAX_INVALID_SPAN,
    DEFAULT_MIN_EVAL_TARGET_AND_TM,
    DEFAULT_MIN_TRAIN_TM_QUESTIONS,
    DEFAULT_R_MIN,
    V3_1_AUTO_ARTIFACT_ROOT,
)
from experiments.qa_chunk_selection.v3.contract import V3_ARTIFACT_ROOT

DEFAULT_STAGES = (
    "init",
    "freeze-qa",
    "build-universe",
    "stage-a",
    "calibrate",
    "stage-a-gate",
    "stage-b",
    "discourse",
    "map-silver",
    "cluster",
    "qc",
    "stability",
    "readiness",
)
ALL_STAGES = DEFAULT_STAGES + ("freeze",)


@dataclass(frozen=True)
class AutoPrepConfig:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_1_AUTO_ARTIFACT_ROOT)
    v3_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_ARTIFACT_ROOT)
    stages: tuple[str, ...] = DEFAULT_STAGES
    seed: int = 42
    device: str = "auto"
    r_min: float = DEFAULT_R_MIN
    max_easy_pass_rate: float = DEFAULT_MAX_EASY_PASS
    max_invalid_span_rate: float = DEFAULT_MAX_INVALID_SPAN
    gold_f1_threshold: float = 0.50
    min_train_questions_with_TM_silver: int = DEFAULT_MIN_TRAIN_TM_QUESTIONS
    min_eval_questions_with_target_and_TM_silver: int = DEFAULT_MIN_EVAL_TARGET_AND_TM
    control_per_class: int = 40
    resume: bool = True
    smoke: bool = False
    skip_qa_models: bool = False
    allow_partial_freeze: bool = False
    allow_existing_output: bool = False
    limit_candidates: int = 0
    reuse_stage_a_dir: Optional[Path] = None
    skip_qa_until_needed: bool = True

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["v3_artifact_dir"] = str(self.v3_artifact_dir)
        payload["reuse_stage_a_dir"] = str(self.reuse_stage_a_dir) if self.reuse_stage_a_dir else None
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
        raise ValueError(f"unknown V3.1 QA-silver stages: {unknown}; allowed={list(ALL_STAGES)}")
    return parts


def v3_paths(v3_dir: Path) -> dict[str, Path]:
    root = Path(v3_dir)
    return {
        "root": root,
        "queue": root / "annotation" / "queue.jsonl",
        "dataset": root / "data" / "dataset.jsonl",
        "splits": root / "data" / "v3_cv_splits.json",
        "hn_flags": root / "data" / "hn_quality_flags.jsonl",
        "b0_oof": root / "analysis" / "oof" / "b0" / "predictions_scored.jsonl",
    }
