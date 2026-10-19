"""Frozen scientific contract recovered from the completed V4.4 artifacts."""

from dataclasses import dataclass
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT


EXPERIMENT_ID = "v4_5_ru_compact_refinement_v1"
SEEDS = (42, 43, 44)
FOLDS = tuple(range(5))
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
SOURCE_OVERLAP = 0.6
ROUTING_QUANTILE = 0.70
MIN_ROUTING_ROWS = 100
EPSILON_HN = 0.02
HIT1_TOLERANCE = 0.005
BOOTSTRAP_SEED = 42
SYSTEMS = ("S0_historical_b0", "S1_dity_b0", "S2_v6_support", "S3_adaptive", "S4_bge_b0")


@dataclass(frozen=True)
class RankerSpec:
    key: str
    model_name: str
    revision: str
    status: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    prepared_artifact: Path
    reference_artifact: Path
    records_rel: str
    splits_rel: str
    flags_rel: str
    qcache_rel: str
    current_b0_rel: str
    k: int
    historical_factorial_root: Path
    historical_bge_oof: Path | None
    qa_model: Path


RANKERS = {
    "dity": RankerSpec(
        "dity", "DiTy/cross-encoder-russian-msmarco",
        "9029bab08103ad171724b510d312befa5b476293",
        "development_selected_after_v4_4",
    ),
    "bge": RankerSpec(
        "bge", "BAAI/bge-reranker-v2-m3",
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "strong_tested_reference",
    ),
}

_RU = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_ru_q5_b0_20260912_172431"
_SCI_DATA = PROJECT_ROOT / "artifacts/qa_chunk_selection_external_data_20260913/scimdix"
_SCI_REF = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_external_20260913_213335/scimdix"
_FACTORIAL = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_ru_rankers_20260914_203558"
_BGE = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_ru_ablation_20260912_230234"
_QA = PROJECT_ROOT / "scripts/models_processing/models/rubert_ru_qa_model"

DATASETS = {
    "russerrc": DatasetSpec(
        "russerrc", _RU, _RU,
        "data/answerable_dataset.jsonl", "data/answerable_cv_splits.json",
        "data/hn_quality_flags.jsonl", "q5/prediction_cache.jsonl",
        "b0/oof_predictions_scored.jsonl", 5, _FACTORIAL / "russerrc",
        _BGE / "cross_encoders/bge_m3/oof_predictions_scored.jsonl", _QA,
    ),
    "scimdix": DatasetSpec(
        "scimdix", _SCI_DATA, _SCI_REF,
        "data/dataset.jsonl", "data/v3_cv_splits.json",
        "data/hn_quality_flags.jsonl", "q5/prediction_cache.jsonl",
        "b0/oof_predictions_scored.jsonl", 4, _FACTORIAL / "scimdix",
        None, _QA,
    ),
}

TRAINING_RECIPE = {
    "optimizer": "AdamW",
    "learning_rate": 1e-5,
    "micro_batch_size": 2,
    "gradient_accumulation": 8,
    "effective_batch_size": 16,
    "max_length": 512,
    "precision": "bf16 on supported CUDA; fp16 otherwise on CUDA; fp32 on CPU",
    "max_epochs": 5,
    "early_stopping_patience": 2,
    "early_stopping_selection": ["pairwise_clean_hn", "pr_auc"],
    "warmup": "none",
    "scheduler": "none",
    "sampling": "unique question pairs; positive:hard-negative:easy = 1:3:1",
    "question_weighting": "equal total weight per question",
    "oversampling": "none",
}

STAGE2_CONTRACT = {
    "formula": "alpha*b0_score + beta*(exact_freq + mean_token_f1 + majority) + gamma*source_clusters",
    "beta": 0.0,
    "beta_fixed_from_v4_4_evidence": True,
    "alpha_grid": list(WEIGHT_GRID),
    "gamma_grid": list(WEIGHT_GRID),
    "normalization": "historical agreement_features scales; no additional normalization",
    "support_definition": f"source span cluster overlap >= {SOURCE_OVERLAP}",
    "tie_break": "descending stage2 score, then historical B0 order, then chunk id",
    "top_k": {key: spec.k for key, spec in DATASETS.items()},
    "question": "canonical q0",
    "question_selection": False,
    "q5_veto": False,
    "alternative_branch": "SEM family selected on TRAIN+CAL; zero weights are exact B0 via B0 tie-break",
    "router_feature": "consensus_conf of best coherent semantic core; selected alternative chunk must belong to core",
    "router_calibration": "Q70 on TRAIN+CAL available rows; unavailable or <100 rows falls back to V6",
    "runtime_gold_fields": False,
}

