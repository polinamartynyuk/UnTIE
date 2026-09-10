"""V4.3 config. Grids are frozen in the contract before any EVAL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.contract import V3_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4.contract import V4_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_1.contract import V4_1_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_2.contract import V4_2_ARTIFACT_ROOT

from .contract import (
    ADAPT_QUANTILES,
    CONTEXT_DELTA_GRID,
    DELTA_SEM_GRID,
    EPSILON_HN,
    LEXCORE_MAJORITY_GRID,
    LEXCORE_TOKEN_GRID,
    N_BOOTSTRAP,
    NORMALIZATION_VERSION,
    PRIMARY_K,
    QA_INFERENCE_HASH,
    SEMANTIC_MODEL,
    SEMCORE_MAJORITY_GRID,
    SEMCORE_SEM_GRID,
    V4_3_ARTIFACT_ROOT,
)

R0 = ("init", "r0")
CACHE = ("init", "r0", "cache")
PRIMARY = (
    "init",
    "r0",
    "cache",
    "embed",
    "sanity",
    "features",
    "sem1",
    "lexcore",
    "semcore",
    "complement",
    "adapt1",
    "semctx1",
    "compare",
    "freeze",
    "report",
)
FULL = PRIMARY
ALL_STAGES = PRIMARY
DEFAULT_STAGES = PRIMARY


@dataclass(frozen=True)
class ExperimentConfigV43:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_3_ARTIFACT_ROOT)
    v3_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_ARTIFACT_ROOT)
    v4_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_ARTIFACT_ROOT)
    v4_1_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_1_ARTIFACT_ROOT)
    v4_2_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_2_ARTIFACT_ROOT)
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
    force_semctx: bool = False
    source_overlap: float = 0.6
    epsilon_hn: float = EPSILON_HN
    n_bootstrap: int = N_BOOTSTRAP
    semantic_model: str = SEMANTIC_MODEL
    normalization_version: str = NORMALIZATION_VERSION
    delta_sem_grid: tuple[float, ...] = DELTA_SEM_GRID
    lexcore_majority_grid: tuple[float, ...] = LEXCORE_MAJORITY_GRID
    lexcore_token_grid: tuple[float, ...] = LEXCORE_TOKEN_GRID
    semcore_majority_grid: tuple[float, ...] = SEMCORE_MAJORITY_GRID
    semcore_sem_grid: tuple[float, ...] = SEMCORE_SEM_GRID
    adapt_quantiles: tuple[float, ...] = ADAPT_QUANTILES
    context_delta_grid: tuple[float, ...] = CONTEXT_DELTA_GRID
    qa_inference_hash: str = QA_INFERENCE_HASH

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["v3_artifact_dir"] = str(self.v3_artifact_dir)
        payload["v4_artifact_dir"] = str(self.v4_artifact_dir)
        payload["v4_1_artifact_dir"] = str(self.v4_1_artifact_dir)
        payload["v4_2_artifact_dir"] = str(self.v4_2_artifact_dir)
        payload["stages"] = list(self.stages)
        payload["model_seeds"] = list(self.model_seeds)
        payload["delta_sem_grid"] = list(self.delta_sem_grid)
        payload["lexcore_majority_grid"] = list(self.lexcore_majority_grid)
        payload["lexcore_token_grid"] = list(self.lexcore_token_grid)
        payload["semcore_majority_grid"] = list(self.semcore_majority_grid)
        payload["semcore_sem_grid"] = list(self.semcore_sem_grid)
        payload["adapt_quantiles"] = list(self.adapt_quantiles)
        payload["context_delta_grid"] = list(self.context_delta_grid)
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
        raise ValueError(f"unknown V4.3 stages: {unknown}")
    return parts
