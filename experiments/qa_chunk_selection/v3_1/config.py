from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from experiments.qa_chunk_selection.config import DEFAULT_CE_MODEL
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.config import ExperimentConfigV3
from experiments.qa_chunk_selection.v3.contract import V2_ARTIFACT_ROOT, V3_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import V3_1_AUTO_ARTIFACT_ROOT

from .contract import V3_1_ARTIFACT_ROOT

DEFAULT_STAGES = ("check-preconditions", "prepare", "s0-audit", "b0", "b1", "oof-compare", "g1", "report")
GATED = ("r1", "pairwise", "listwise", "attention-audit", "d4-oracle", "d5-audit", "freeze-final", "final-eval")
ALL_STAGES = DEFAULT_STAGES + GATED


@dataclass
class ExperimentConfigV31:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_1_ARTIFACT_ROOT)
    v3_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_ARTIFACT_ROOT)
    v2_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V2_ARTIFACT_ROOT)
    silver_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_1_AUTO_ARTIFACT_ROOT)
    seed: int = 42
    split_seed: int = 42
    bootstrap_seed: int = 42
    device: str = "auto"
    model_seeds: tuple[int, ...] = (42, 43, 44)
    r1_k_values: tuple[int, ...] = (5,)
    top_k_values: tuple[int, ...] = (1, 5, 10)
    n_bootstrap: int = 5000
    b0_epochs: int = 8
    b0_patience: int = 2
    b2_batch_size: int = 16
    b2_learning_rate: float = 2e-5
    b2_max_length: int = 512
    b2_pos_hn_easy: tuple[int, int, int] = (1, 3, 1)
    easy_negative_cap: int = 4
    epsilon_hn: float = 0.02
    epsilon_pairwise_hn: float = 0.03
    delta_f1_min: float = 1.0
    skip_bertscore: bool = True
    skip_qa: bool = False
    smoke: bool = False
    resume: bool = False
    force_r1: bool = False
    allow_internal_test: bool = False
    cross_encoder_model: str = DEFAULT_CE_MODEL
    stages: tuple[str, ...] = DEFAULT_STAGES

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, tuple):
                payload[key] = list(value)
        return payload

    def as_v3_config(self) -> ExperimentConfigV3:
        return ExperimentConfigV3(
            output_dir=self.output_dir,
            v2_artifact_dir=self.v2_artifact_dir,
            seed=self.seed,
            split_seed=self.split_seed,
            bootstrap_seed=self.bootstrap_seed,
            device=self.device,
            model_seeds=self.model_seeds,
            top_k_values=self.top_k_values,
            n_bootstrap=self.n_bootstrap,
            b2_epochs=self.b0_epochs,
            b2_patience=self.b0_patience,
            b2_batch_size=self.b2_batch_size,
            b2_learning_rate=self.b2_learning_rate,
            b2_max_length=self.b2_max_length,
            b2_pos_hn_easy=self.b2_pos_hn_easy,
            easy_negative_cap=self.easy_negative_cap,
            epsilon_hn=self.epsilon_hn,
            epsilon_pairwise_hn=self.epsilon_pairwise_hn,
            delta_f1_min=self.delta_f1_min,
            skip_bertscore=self.skip_bertscore,
            skip_qa=self.skip_qa,
            smoke=self.smoke,
            resume=self.resume,
            allow_internal_test=False,
            cross_encoder_model=self.cross_encoder_model,
            stages=(),
        )


def parse_stages(raw: Sequence[str] | str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_STAGES
    items = [item.strip() for item in (raw.split(",") if isinstance(raw, str) else raw) if str(item).strip()]
    if items == ["all"]:
        return DEFAULT_STAGES
    unknown = [item for item in items if item not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown V3.1 stages: {unknown}")
    return tuple(items)
