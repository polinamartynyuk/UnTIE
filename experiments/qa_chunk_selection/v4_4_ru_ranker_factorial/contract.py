"""Frozen contract for the two-dataset Russian ranker factorial."""

from dataclasses import dataclass
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT


EXPERIMENT_ID = "v4_4_ru_ranker_factorial_v1"
SEEDS = (42, 43, 44)
K_VALUES = (3, 4, 5)
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
SOURCE_OVERLAP = 0.6
EPSILON_HN = 0.02
HIT1_TOLERANCE = 0.005
ROUTING_QUANTILE = 0.70
Q5_ROUTING_QUANTILE = 0.30
MIN_ROUTING_ROWS = 100
STRATEGIES = ("q0", "selected")
SYSTEMS = ("B0", "V6", "SEM", "LEX", "LEX-LEM", "ASEM", "ALEX", "ASEM-Q5", "ALEX-Q5")


@dataclass(frozen=True)
class RankerSpec:
    key: str
    model_name: str
    revision: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    prepared_artifact: Path
    reference_artifact: Path
    records_rel: str
    splits_rel: str
    flags_rel: str
    qcache_rel: str
    qpred_rel: str
    current_b0_rel: str
    chunk_labels_rel: str | None = None


# Revisions are fixed so a resumed run cannot silently use different weights.
RANKER_SPECS = {
    "modernbert_small": RankerSpec(
        "modernbert_small",
        "ARGA100/ru-reranker-modernbert-small",
        "8d4ea05d7c793bc812879ca18e7e310ac4cb228f",
    ),
    "dity_msmarco": RankerSpec(
        "dity_msmarco",
        "DiTy/cross-encoder-russian-msmarco",
        "9029bab08103ad171724b510d312befa5b476293",
    ),
}

_RUSERRC = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_ru_q5_b0_20260912_172431"
_SCIMDIX_DATA = PROJECT_ROOT / "artifacts/qa_chunk_selection_external_data_20260913/scimdix"
_SCIMDIX_REFERENCE = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_external_20260913_213335/scimdix"

DATASET_SPECS = {
    "russerrc": DatasetSpec(
        "russerrc", _RUSERRC, _RUSERRC,
        "data/answerable_dataset.jsonl", "data/answerable_cv_splits.json",
        "data/hn_quality_flags.jsonl", "q5/prediction_cache.jsonl",
        "q5/question_predictions.parquet", "b0/oof_predictions_scored.jsonl",
        "q5/chunk_labels.parquet",
    ),
    "scimdix": DatasetSpec(
        "scimdix", _SCIMDIX_DATA, _SCIMDIX_REFERENCE,
        "data/dataset.jsonl", "data/v3_cv_splits.json",
        "data/hn_quality_flags.jsonl", "q5/prediction_cache.jsonl",
        "q5/question_predictions.parquet", "b0/oof_predictions_scored.jsonl",
        None,
    ),
}

