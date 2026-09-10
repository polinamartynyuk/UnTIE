"""V4.1 config. Grids are frozen in the contract before any EVAL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.contract import V3_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4.contract import V4_ARTIFACT_ROOT

from .contract import (
    EPSILON_HN,
    ETA_GRID,
    K_CANDIDATES,
    LAMBDA_AGG_GRID,
    N_BOOTSTRAP,
    PRIMARY_K,
    QA_INFERENCE_HASH,
    UTILITY_F1_PCT,
    V4_1_ARTIFACT_ROOT,
    V4_V2A4_LAMBDA,
    V6_ALPHA_GRID,
    V6_BETA_GRID,
    V6_GAMMA_GRID,
)

R0 = ("init", "r0")
CACHE = ("init", "r0", "cache")
PRIMARY = (
    "init",
    "r0",
    "cache",
    "r0-v6-repro",
    "r1",
    "r2",
    "r3",
    "r4",
    "compare",
    "freeze",
    "report",
)
FULL = PRIMARY[:-2] + ("r5", "compare", "freeze", "report")
ALL_STAGES = (
    "init",
    "r0",
    "cache",
    "r0-v6-repro",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "compare",
    "freeze",
    "report",
)
DEFAULT_STAGES = PRIMARY


@dataclass(frozen=True)
class ExperimentConfigV41:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_1_ARTIFACT_ROOT)
    v3_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_ARTIFACT_ROOT)
    v4_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_ARTIFACT_ROOT)
    stages: tuple[str, ...] = DEFAULT_STAGES
    device: str = "auto"
    seed: int = 42
    model_seeds: tuple[int, ...] = (42, 43, 44)
    trainable_seeds: tuple[int, ...] = (42,)
    top_k_values: tuple[int, ...] = K_CANDIDATES
    primary_k: int = PRIMARY_K
    smoke: bool = False
    allow_existing_output: bool = False
    allow_internal_test: bool = False
    allow_r1_without_repro: bool = False
    skip_bertscore: bool = True
    skip_qa: bool = False
    reuse_v4_cache: bool = True
    source_overlap: float = 0.6
    utility_f1_pct: float = UTILITY_F1_PCT
    epsilon_hn: float = EPSILON_HN
    n_bootstrap: int = N_BOOTSTRAP
    v6_alpha_grid: tuple[float, ...] = V6_ALPHA_GRID
    v6_beta_grid: tuple[float, ...] = V6_BETA_GRID
    v6_gamma_grid: tuple[float, ...] = V6_GAMMA_GRID
    lambda_agg_grid: tuple[float, ...] = LAMBDA_AGG_GRID
    eta_grid: tuple[float, ...] = ETA_GRID
    v2_a4_lambda: float = V4_V2A4_LAMBDA
    qa_inference_hash: str = QA_INFERENCE_HASH
    b0_epochs: int = 5
    b0_patience: int = 2
    b0_learning_rate: float = 2e-5
    b0_max_length: int = 512
    listwise_temperature: float = 1.0
    easy_negative_cap: int = 4
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["v3_artifact_dir"] = str(self.v3_artifact_dir)
        payload["v4_artifact_dir"] = str(self.v4_artifact_dir)
        payload["stages"] = list(self.stages)
        payload["model_seeds"] = list(self.model_seeds)
        payload["trainable_seeds"] = list(self.trainable_seeds)
        payload["top_k_values"] = list(self.top_k_values)
        payload["v6_alpha_grid"] = list(self.v6_alpha_grid)
        payload["v6_beta_grid"] = list(self.v6_beta_grid)
        payload["v6_gamma_grid"] = list(self.v6_gamma_grid)
        payload["lambda_agg_grid"] = list(self.lambda_agg_grid)
        payload["eta_grid"] = list(self.eta_grid)
        return payload


def parse_stages(raw: str) -> tuple[str, ...]:
    text = (raw or "primary").strip().lower()
    if text in {"r0", "regression"}:
        return R0
    if text in {"cache", "phase0"}:
        return CACHE
    if text in {"primary", "default", "phase1"}:
        return PRIMARY
    if text in {"all", "full"}:
        return FULL
    parts = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = [item for item in parts if item not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown V4.1 stages: {unknown}")
    return parts
