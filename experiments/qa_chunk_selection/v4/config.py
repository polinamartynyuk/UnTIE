"""V4 config. Hyperparameters are fixed a priori; CAL only tunes fusion weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.contract import V3_ARTIFACT_ROOT

from .contract import (
    KEYBERT_BACKEND,
    KEYBERT_N,
    KEYBERT_NGRAM,
    KEYBERT_USE_MMR,
    SOURCE_OVERLAP,
    UTILITY_ETA,
    UTILITY_F1,
    V4_ARTIFACT_ROOT,
)

PHASE0 = ("init", "freeze-b0", "shortlist", "clusters", "keybert", "qa-topk")
PHASE1 = PHASE0 + ("v1", "v2", "compare", "report")
FULL = PHASE0 + ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "compare", "report")
ALL_STAGES = FULL

DEFAULT_STAGES = PHASE1
FUSION_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ExperimentConfigV4:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V4_ARTIFACT_ROOT)
    v3_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_ARTIFACT_ROOT)
    stages: tuple[str, ...] = DEFAULT_STAGES
    device: str = "auto"
    seed: int = 42
    model_seeds: tuple[int, ...] = (42, 43, 44)
    trainable_seeds: tuple[int, ...] = (42,)
    top_k_values: tuple[int, ...] = (5, 10)
    primary_k: int = 5
    smoke: bool = False
    allow_existing_output: bool = False
    allow_internal_test: bool = False
    skip_bertscore: bool = True
    skip_qa: bool = False
    keybert_backend: str = KEYBERT_BACKEND
    keybert_n: int = KEYBERT_N
    keybert_ngram_low: int = KEYBERT_NGRAM[0]
    keybert_ngram_high: int = KEYBERT_NGRAM[1]
    keybert_use_mmr: bool = KEYBERT_USE_MMR
    source_overlap: float = SOURCE_OVERLAP
    utility_f1: float = UTILITY_F1
    utility_eta: float = UTILITY_ETA
    epsilon_hn: float = 0.02
    epsilon_pairwise: float = 0.03
    delta_f1_min: float = 0.0
    n_bootstrap: int = 5000
    b0_epochs: int = 5
    b0_patience: int = 2
    b0_batch_size: int = 16
    b0_learning_rate: float = 2e-5
    b0_max_length: int = 512
    b0_pos_hn_easy: tuple[int, int, int] = (1, 3, 1)
    listwise_temperature: float = 1.0
    easy_negative_cap: int = 4
    ownership_gate_pairwise: float = 0.55
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    # Protocol prefers inner grouped B0. This cycle reuses historical V3-B0 OOF
    # shortlists on TRAIN (no inner cross-fit; V3.1 ranking never ran).
    inner_crossfit: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["v3_artifact_dir"] = str(self.v3_artifact_dir)
        payload["stages"] = list(self.stages)
        payload["model_seeds"] = list(self.model_seeds)
        payload["trainable_seeds"] = list(self.trainable_seeds)
        payload["top_k_values"] = list(self.top_k_values)
        payload["b0_pos_hn_easy"] = list(self.b0_pos_hn_easy)
        return payload


def parse_stages(raw: str) -> tuple[str, ...]:
    text = (raw or "phase1").strip().lower()
    if text in {"phase0", "cache", "common"}:
        return PHASE0
    if text in {"phase1", "default"}:
        return PHASE1
    if text in {"all", "full"}:
        return FULL
    parts = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = [item for item in parts if item not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown V4 stages: {unknown}")
    return parts
