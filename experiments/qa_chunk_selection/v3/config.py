"""V3 configuration. Hyperparameters are chosen on CALIBRATION_FOLD / OOF only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from experiments.qa_chunk_selection.config import DEFAULT_CE_MODEL, DEFAULT_DEV_LABELS, DEFAULT_DEV_PREPROCESS, DEFAULT_TRAIN_LABELS, DEFAULT_TRAIN_PREPROCESS

from .io_util import DEFAULT_V3_OUTPUT, E0_OUTPUT, PROJECT_ROOT, V2_OUTPUT

STAGES = (
    "prepare",
    "split",
    "overlay-sample",
    "b0",
    "b1",
    "b2",
    "b3",
    "b4",
    "structure",
    "positive-quality",
    "at0",
    "at1",
    "at2",
    "at3",
    "at4",
    "at4-d",
    "reader-success",
    "span-ranker",
    "at5",
    "d4",
    "d5",
    "d3",
    "oof-compare",
    "report",
    "freeze-final",
    "final-eval",
)

DEFAULT_STAGES_V3 = (
    "prepare",
    "split",
    "overlay-sample",
    "b0",
    "b1",
    "b2",
    "b3",
    "structure",
    "positive-quality",
    "at0",
    "at1",
    "at2",
    "at3",
    "at4",
    "span-ranker",
    "oof-compare",
    "report",
)

GATED_STAGES = ("b4", "at4-d", "reader-success", "at5", "d4", "d5", "d3", "freeze-final", "final-eval")


@dataclass
class ExperimentConfigV3:
    train_label_dir: Path = DEFAULT_TRAIN_LABELS
    dev_label_dir: Path = DEFAULT_DEV_LABELS
    train_preprocess_dir: Path = DEFAULT_TRAIN_PREPROCESS
    dev_preprocess_dir: Path = DEFAULT_DEV_PREPROCESS
    output_dir: Path = DEFAULT_V3_OUTPUT
    v2_artifact_dir: Path = V2_OUTPUT
    e0_artifact_dir: Path = E0_OUTPUT
    seed: int = 42
    split_seed: int = 42
    bootstrap_seed: int = 42
    device: str = "auto"
    n_folds: int = 5
    model_seeds: tuple[int, ...] = (42, 43, 44)
    top_k_values: tuple[int, ...] = (1, 5, 10)
    b4_k_values: tuple[int, ...] = (3, 5, 8)
    span_k_values: tuple[int, ...] = (2, 3, 5)
    attention_k_values: tuple[int, ...] = (3, 5, 8)
    n_heads_grid: tuple[int, ...] = (1, 2, 4, 8, 16)
    n_bootstrap: int = 5000
    b2_epochs: int = 5
    b2_patience: int = 2
    b2_batch_size: int = 16
    b2_learning_rate: float = 2e-5
    b2_max_length: int = 512
    b2_pos_hn_easy: tuple[int, int, int] = (1, 3, 1)
    pairwise_p_tm: float = 0.50
    pairwise_p_hn: float = 0.40
    pairwise_p_easy: float = 0.10
    listwise_temperature: float = 1.0
    easy_negative_cap: int = 4
    epsilon_hn: float = 0.02
    epsilon_pairwise_hn: float = 0.03
    min_target_status_coverage: float = 0.95
    delta_f1_min: float = 1.0
    runtime_ratio_max: float = 3.0
    memory_ratio_max: float = 2.0
    overlay_hard_n: int = 200
    overlay_repr_n: int = 200
    skip_bertscore: bool = True
    skip_qa: bool = False
    smoke: bool = False
    debug_docs: int = 0
    limit_docs: int = 0
    allow_internal_test: bool = False
    resume: bool = False
    include_unanswerable: bool = False
    cross_encoder_model: str = DEFAULT_CE_MODEL
    stages: tuple[str, ...] = DEFAULT_STAGES_V3

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, tuple):
                payload[key] = list(value)
        return payload

    def stable_config_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        for key in (
            "stages",
            "resume",
            "allow_internal_test",
            "smoke",
            "debug_docs",
            "limit_docs",
            "device",
            "output_dir",
            "e0_artifact_dir",
            "v2_artifact_dir",
        ):
            payload.pop(key, None)
        return payload

    def config_hash(self) -> str:
        encoded = json.dumps(self.stable_config_dict(), sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def run_dir(self, experiment: str, seed: int | None = None) -> Path:
        used = self.seed if seed is None else int(seed)
        return self.output_dir / "runs" / experiment / str(used)


def parse_stages(raw: Sequence[str] | str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_STAGES_V3
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in raw if str(item).strip()]
    if items == ["all"]:
        return DEFAULT_STAGES_V3
    if items == ["full"]:
        return STAGES
    unknown = [item for item in items if item not in STAGES]
    if unknown:
        raise ValueError(f"unknown V3 stages: {unknown}")
    return tuple(items)
