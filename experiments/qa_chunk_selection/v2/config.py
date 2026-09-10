"""V2 experiment configuration. Hyperparameters are chosen on CALIBRATION/DEVELOPMENT only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from experiments.qa_chunk_selection.config import DEFAULT_CE_MODEL, DEFAULT_DEV_LABELS, DEFAULT_DEV_PREPROCESS, DEFAULT_TRAIN_LABELS, DEFAULT_TRAIN_PREPROCESS

from .io_util import DEFAULT_V2_OUTPUT, PROJECT_ROOT

DEFAULT_E0_ARTIFACTS = PROJECT_ROOT / "artifacts" / "qa_chunk_selection"

STAGES = (
    "prepare-v2",
    "split-v2",
    "e0-regression",
    "reader-oracle",
    "ambiguous-audit",
    "b2-train",
    "b2-eval",
    "ambiguous-audit-b2",
    "reader-aware",
    "c-sweep",
    "c-adaptive",
    "d2-ablate",
    "compare-v2",
    "report-v2",
    "freeze-final",
    "final-eval",
    "d4-train",
    "d4",
    "d5",
    "unanswerable-audit",
)

DEFAULT_STAGES_V2 = (
    "prepare-v2",
    "split-v2",
    "e0-regression",
    "reader-oracle",
    "ambiguous-audit",
    "b2-train",
    "b2-eval",
    "ambiguous-audit-b2",
    "reader-aware",
    "c-sweep",
    "c-adaptive",
    "d2-ablate",
    "compare-v2",
    "report-v2",
)

OPTIONAL_STAGES = ("freeze-final", "final-eval", "d4-train", "d4", "d5", "unanswerable-audit")


@dataclass
class ExperimentConfigV2:
    train_label_dir: Path = DEFAULT_TRAIN_LABELS
    dev_label_dir: Path = DEFAULT_DEV_LABELS
    train_preprocess_dir: Path = DEFAULT_TRAIN_PREPROCESS
    dev_preprocess_dir: Path = DEFAULT_DEV_PREPROCESS
    output_dir: Path = DEFAULT_V2_OUTPUT
    e0_artifact_dir: Path = DEFAULT_E0_ARTIFACTS
    seed: int = 42
    split_seed: int = 42
    bootstrap_seed: int = 42
    device: str = "auto"
    candidate_mode: str = "all"
    train_fraction: float = 0.60
    calibration_fraction: float = 0.15
    development_fraction: float = 0.15
    internal_test_fraction: float = 0.10
    model_seeds: tuple[int, ...] = (42, 43, 44)
    top_k_values: tuple[int, ...] = (1, 5, 10)
    passage_m_values: tuple[int, ...] = (1, 2, 3, 5, 8)
    default_passage_m: int = 5
    rrf_k: int = 60
    n_bootstrap: int = 5000
    regression_atol: float = 1e-6
    regression_rtol: float = 1e-5
    include_hn_gold_warning: bool = False
    b2_variants: tuple[str, ...] = ("b2a", "b2b", "b2c")
    b2_epochs: int = 5
    b2_patience: int = 2
    b2_batch_size: int = 16
    b2_learning_rate: float = 2e-5
    b2_max_length: int = 512
    b2_pos_hn_easy: tuple[int, int, int] = (1, 3, 1)
    r3_k_pre_values: tuple[int, ...] = (3, 5, 8)
    r3_default_k_pre: int = 5
    adaptive_m_uncertain_values: tuple[int, ...] = (3, 5)
    skip_bertscore: bool = True
    skip_qa: bool = False
    smoke: bool = False
    debug_docs: int = 0
    limit_docs: int = 0
    allow_internal_test: bool = False
    resume: bool = False
    include_unanswerable: bool = False
    replay_e0_inference: bool = True
    cross_encoder_model: str = DEFAULT_CE_MODEL
    stages: tuple[str, ...] = DEFAULT_STAGES_V2

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, tuple):
                payload[key] = list(value)
        return payload

    def stable_config_dict(self) -> dict[str, Any]:
        """Fields that define the scientific configuration, not a particular CLI invocation."""
        payload = self.to_dict()
        for key in (
            "stages",
            "resume",
            "allow_internal_test",
            "smoke",
            "debug_docs",
            "limit_docs",
            "replay_e0_inference",
            "device",
            "output_dir",
            "e0_artifact_dir",
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
        return DEFAULT_STAGES_V2
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in raw if str(item).strip()]
    if items == ["all"]:
        return DEFAULT_STAGES_V2
    if items == ["full"]:
        return STAGES
    unknown = [item for item in items if item not in STAGES]
    if unknown:
        raise ValueError(f"unknown V2 stages: {unknown}")
    return tuple(items)
