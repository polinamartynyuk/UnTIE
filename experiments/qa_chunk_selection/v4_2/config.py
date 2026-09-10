"""V4.2 config. Grids are frozen in the contract before any EVAL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.contract import V3_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4.contract import V4_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_1.contract import V4_1_ARTIFACT_ROOT

from .contract import (
    ATT3_LAMBDA_GRID,
    ATT3_TAU_GRID,
    ATTENTION_FEATURE_VERSION,
    CORE_MAJORITY_GRID,
    CORE_TOKEN_GRID,
    DELTA_GRID,
    EPSILON_HN,
    MIN_PAIRED_QUESTIONS,
    N_BOOTSTRAP,
    N_HEAD_GRID,
    PRIMARY_K,
    QA_INFERENCE_HASH,
    READER_EASY_THRESHOLD,
    V4_2_ARTIFACT_ROOT,
)

R0 = ("init", "r0")
CACHE = ("init", "r0", "cache")
PRIMARY = (
    "init",
    "r0",
    "cache",
    "core",
    "att-reg",
    "att-extract",
    "heads",
    "att1",
    "att2",
    "att3",
    "compare",
    "freeze",
    "report",
)
FULL = PRIMARY
ALL_STAGES = PRIMARY
DEFAULT_STAGES = PRIMARY


@dataclass(frozen=True)
class ExperimentConfigV42:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_2_ARTIFACT_ROOT)
    v3_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_ARTIFACT_ROOT)
    v4_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_ARTIFACT_ROOT)
    v4_1_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_1_ARTIFACT_ROOT)
    stages: tuple[str, ...] = DEFAULT_STAGES
    device: str = "auto"
    seed: int = 42
    model_seeds: tuple[int, ...] = (42, 43, 44)
    primary_k: int = PRIMARY_K
    smoke: bool = False
    allow_existing_output: bool = False
    allow_internal_test: bool = False
    skip_bertscore: bool = True
    skip_qa: bool = False
    reuse_v4_1_qa_cache: bool = True
    force_att2: bool = False
    force_att3: bool = False
    source_overlap: float = 0.6
    epsilon_hn: float = EPSILON_HN
    n_bootstrap: int = N_BOOTSTRAP
    reader_easy_threshold: float = READER_EASY_THRESHOLD
    min_paired_questions: int = MIN_PAIRED_QUESTIONS
    n_head_grid: tuple[int, ...] = N_HEAD_GRID
    delta_grid: tuple[float, ...] = DELTA_GRID
    core_majority_grid: tuple[float, ...] = CORE_MAJORITY_GRID
    core_token_grid: tuple[float, ...] = CORE_TOKEN_GRID
    att3_tau_grid: tuple[float, ...] = ATT3_TAU_GRID
    att3_lambda_grid: tuple[float, ...] = ATT3_LAMBDA_GRID
    qa_inference_hash: str = QA_INFERENCE_HASH
    attention_feature_version: str = ATTENTION_FEATURE_VERSION

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["v3_artifact_dir"] = str(self.v3_artifact_dir)
        payload["v4_artifact_dir"] = str(self.v4_artifact_dir)
        payload["v4_1_artifact_dir"] = str(self.v4_1_artifact_dir)
        payload["stages"] = list(self.stages)
        payload["model_seeds"] = list(self.model_seeds)
        payload["n_head_grid"] = list(self.n_head_grid)
        payload["delta_grid"] = list(self.delta_grid)
        payload["core_majority_grid"] = list(self.core_majority_grid)
        payload["core_token_grid"] = list(self.core_token_grid)
        payload["att3_tau_grid"] = list(self.att3_tau_grid)
        payload["att3_lambda_grid"] = list(self.att3_lambda_grid)
        return payload


def parse_stages(raw: str) -> tuple[str, ...]:
    text = (raw or "primary").strip().lower()
    if text in {"r0", "regression"}:
        return R0
    if text in {"cache", "phase0"}:
        return CACHE
    if text in {"primary", "default", "phase1", "all", "full"}:
        return PRIMARY
    parts = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = [item for item in parts if item not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown V4.2 stages: {unknown}")
    return parts
