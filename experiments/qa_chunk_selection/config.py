"""Experiment configuration. Hyperparameters are chosen on DEV only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_LABELS = (
    PROJECT_ROOT / "artifacts" / "phase1_multiquestion" / "official_train_q5"
)
DEFAULT_DEV_LABELS = (
    PROJECT_ROOT / "artifacts" / "phase1_multiquestion" / "official_dev_q5"
)
DEFAULT_TRAIN_PREPROCESS = PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing"
DEFAULT_DEV_PREPROCESS = PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing_dev"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "qa_chunk_selection"
DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

STAGES = (
    "prepare",
    "a0",
    "a",
    "b-train",
    "b",
    "c",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "compare",
)

DEFAULT_STAGES = (
    "prepare",
    "a0",
    "a",
    "b-train",
    "b",
    "c",
    "d1",
    "d2",
    "compare",
)


@dataclass(frozen=True)
class ExperimentConfig:
    train_label_dir: Path = DEFAULT_TRAIN_LABELS
    dev_label_dir: Path = DEFAULT_DEV_LABELS
    train_preprocess_dir: Path = DEFAULT_TRAIN_PREPROCESS
    dev_preprocess_dir: Path = DEFAULT_DEV_PREPROCESS
    output_dir: Path = DEFAULT_OUTPUT
    seed: int = 42
    device: str = "auto"
    candidate_mode: str = "all"
    n_easy_negatives: int = 5
    top_k_values: tuple[int, ...] = (1, 5, 10)
    dense_top_k: int = 20
    ce_top_k: int = 10
    passage_m_values: tuple[int, ...] = (2, 3, 5, 8)
    default_passage_m: int = 5
    rrf_k: int = 60
    answerability_ratios: tuple[tuple[int, int, int], ...] = (
        (1, 1, 0),
        (1, 2, 1),
        (1, 3, 1),
    )
    b_loss: str = "binary"
    b_epochs: int = 3
    b_batch_size: int = 16
    b_learning_rate: float = 2e-5
    b_max_length: int = 512
    pairwise_margin: float = 1.0
    graph_hops: int = 1
    max_evidence_chunks: int = 8
    max_graph_neighbors: int = 3
    sentence_window: int = 0
    n_seed_chunks: int = 2
    skip_bertscore: bool = True
    skip_qa: bool = False
    smoke: bool = False
    debug_docs: int = 0
    limit_docs: int = 0
    allow_test: bool = False
    resume: bool = False
    train_reader: bool = False
    include_unanswerable: bool = False
    cross_encoder_model: str = DEFAULT_CE_MODEL
    stages: tuple[str, ...] = DEFAULT_STAGES
    seeds: tuple[int, ...] = (42,)

    def __post_init__(self) -> None:
        if self.candidate_mode not in {"all", "controlled"}:
            raise ValueError("candidate_mode must be 'all' or 'controlled'")
        if self.b_loss not in {"binary", "pairwise"}:
            raise ValueError("b_loss must be 'binary' or 'pairwise'")
        unknown = [item for item in self.stages if item not in STAGES]
        if unknown:
            raise ValueError(f"unknown stages: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, tuple):
                payload[key] = [list(item) if isinstance(item, tuple) else item for item in value]
        return payload

    def run_dir(self, experiment: str, seed: int | None = None) -> Path:
        used = self.seed if seed is None else int(seed)
        return self.output_dir / "runs" / experiment / str(used)


def parse_stages(raw: Sequence[str] | str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_STAGES
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in raw if str(item).strip()]
    if items == ["all"]:
        return DEFAULT_STAGES
    if items == ["full"]:
        return STAGES
    return tuple(items)
