"""Frozen, dataset-independent V10 / SRR scientific contract."""
from __future__ import annotations

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection_v8.contract import (
    CELLS, FAMILIES, H2_COLUMNS, LANGUAGES, OUTER_FOLDS, SEEDS,
)

V5_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v5/run_20260916_135455"
V8_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v8_dual/run_20260923_133853"
V9_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400"

INNER_FOLDS = 4
INNER_SPLIT_SEED = 42
PAIR_MIN_GAP = 1.0
INNER_EVIDENCE_MIN_ROWS = 5
INNER_EVIDENCE_MIN_QUESTIONS = 5
REFIT_EVIDENCE_MIN_ROWS = 10
REFIT_EVIDENCE_MIN_QUESTIONS = 5
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)
RANK_GRID = tuple(
    {"learning_rate": 0.05, "max_leaf_nodes": leaves, "max_iter": iterations,
     "l2_regularization": l2}
    for leaves in (7, 15) for iterations in (100, 200) for l2 in (1.0, 10.0)
)
TAU_SWITCH_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)
TAU_ANSWER_GRID = (0.20, 0.30, 0.40, 0.50)
TAU_EVIDENCE_GRID = (0.10, 0.20, 0.30, 0.40, 0.50)
SYSTEMS = ("S0", "S1", "S2", "A0", "A1", "A2")
ABLATIONS = {
    "A0": (False, False),
    "A1": (True, False),
    "A2": (False, True),
    "S2": (True, True),
}

# Exact project semantics. Unknown non-null labels are fatal; there is no
# substring/string-guess fallback.
EVIDENCE_MAPPING = {
    "sufficient_positive": "SAFE_EVIDENCE",
    "insufficient_evidence": "UNSAFE_EVIDENCE",
    "unresolved_for_training": "EXCLUDE",
    None: "EXCLUDE",
    "": "EXCLUDE",
}

QUESTION_EXTRA_COLUMNS = (
    "candidate_count", "valid_answer_count", "empty_answer_count",
    "unique_normalized_answer_count", "answer_diversity",
    "largest_exact_answer_cluster_ratio",
)
QUESTION_AGG_COLUMNS = tuple(
    f"{column}__{stat}"
    for column in H2_COLUMNS
    for stat in ("max", "min", "mean", "std", "range")
) + QUESTION_EXTRA_COLUMNS
SWITCH_COLUMNS = QUESTION_AGG_COLUMNS
RANK_COLUMNS = tuple(f"delta__{x}" for x in H2_COLUMNS) + QUESTION_AGG_COLUMNS
ANSWER_RISK_COLUMNS = RANK_COLUMNS
EVIDENCE_RISK_COLUMNS = (
    tuple(f"candidate__{x}" for x in H2_COLUMNS)
    + tuple(f"delta__{x}" for x in H2_COLUMNS)
    + QUESTION_AGG_COLUMNS
)


def frozen_contract() -> dict:
    return {
        "experiment": "V10 / SRR — Switch–Rank–Risk",
        "version": "10.0.1",
        "development_status": "development",
        "cells": list(CELLS), "families": FAMILIES, "languages": LANGUAGES,
        "systems": {"S0": "canonical B0", "S1": "exact frozen H2", "S2": "V10 SRR"},
        "diagnostic_ablations": {k: list(v) for k, v in ABLATIONS.items() if k != "S2"},
        "h2_feature_columns": list(H2_COLUMNS),
        "question_aggregate_columns": list(QUESTION_AGG_COLUMNS),
        "learned_h2_outputs_excluded": True,
        "forbidden_runtime_features": ["gold", "actual_f1", "utility", "evidence_label", "dataset_id", "language_id", "aspect_id"],
        "k1_policy": "B0/NO_ALTERNATIVE; excluded from all module training and inference",
        "switch": {"model": "StandardScaler+L2 LogisticRegression(liblinear)", "C": list(LOGISTIC_C_GRID), "target": "any alternative F1 > B0 F1"},
        "rank": {"model": "HistGradientBoostingClassifier", "grid": list(RANK_GRID), "pair_min_gap": PAIR_MIN_GAP, "symmetric_training": True, "antisymmetric_inference": True},
        "answer_risk": {"model": "StandardScaler+L2 LogisticRegression(liblinear)", "C": list(LOGISTIC_C_GRID), "target": "alternative F1 < B0 F1"},
        "evidence_risk": {"model": "StandardScaler+L2 LogisticRegression(liblinear)", "C": list(LOGISTIC_C_GRID), "mapping": {str(k): v for k, v in EVIDENCE_MAPPING.items()}, "inner_support": {"rows_per_class": INNER_EVIDENCE_MIN_ROWS, "questions_per_class": INNER_EVIDENCE_MIN_QUESTIONS}, "refit_support": {"rows_per_class": REFIT_EVIDENCE_MIN_ROWS, "questions_per_class": REFIT_EVIDENCE_MIN_QUESTIONS}},
        "inner_cv": {"folds": INNER_FOLDS, "group": "canonical_document_id", "stratification": "seed-42 question has UNSAFE_EVIDENCE alternative", "splitter": "StratifiedGroupKFold(shuffle=True)", "split_seed": INNER_SPLIT_SEED, "split_reference_model_seed": 42, "same_splits_all_modules_and_seeds": True},
        "protocol_amendment": "V10.0.1: inner evidence support 5/5; full refit remains 10 rows/5 questions because SciER384 has only 11 unsafe rows in one outer TRAIN/CAL",
        "model_seeds": list(SEEDS), "outer_folds": list(OUTER_FOLDS),
        "thresholds": {"switch": list(TAU_SWITCH_GRID), "answer_harm": list(TAU_ANSWER_GRID), "evidence_harm": list(TAU_EVIDENCE_GRID)},
        "threshold_objective": "maximum TRAIN-only OOF Token F1 subject to evidence safety",
        "safety": {"delta_hit1_min_pp": -0.5, "delta_hn1_max_pp": 1.0, "delta_sufficient_min_pp": -1.0},
        "fallbacks": ["NO_ALTERNATIVE", "NO_SWITCH", "NO_SAFE_CANDIDATE", "SELECTED_ALTERNATIVE", "OUTER_FOLD_B0_FALLBACK"],
        "bootstrap": {"iterations": 5000, "seed": 42, "cluster": "canonical_document_id", "seeds_are_repeated_predictions": True},
        "same_neural_forwards_s1_s2": True,
        "no_new_neural_models": True,
        "historical_roots_read_only": True,
    }

