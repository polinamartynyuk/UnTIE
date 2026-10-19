"""Frozen, dataset-independent scientific contract for V9 / H2-ACS."""
from __future__ import annotations

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection_v8.contract import (
    ALPHA_GRID,
    CELLS,
    FAMILIES,
    H2_COLUMNS,
    H3_COLUMNS,
    HARM_C_GRID,
    INNER_FOLDS,
    LANGUAGES,
    OUTER_FOLDS,
    SEEDS,
    TAU_HARM_GRID,
    TAU_UTILITY_GRID,
    V4_4_EXTERNAL_RUN,
    V4_4_RUN,
    V4_5_RUN,
    V5_RUN,
    V6_RUN,
    V7_RUN,
)

V8_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v8_dual/run_20260923_133853"

SELECT_FOLDS = 4
SELECT_SPLIT_SEED = 42
EPS_SELECT = 0.20
COMPLEXITY_TIE = 0.10
WORST_SELECT_FOLD = -0.50
INCREMENTAL_HIT_MIN_PP = -0.25
INCREMENTAL_HN_MAX_PP = 0.50
ABSOLUTE_HIT_MIN_PP = -0.50
ABSOLUTE_HN_MAX_PP = 1.00

CONFIGS = ("C0", "CA", "CP", "CAP")
CONFIG_COMPLEXITY = {"C0": 0, "CA": 1, "CP": 1, "CAP": 2}

# Exact V8 H3-H2 delta.
AGREEMENT_GROUP = tuple(x for x in H3_COLUMNS if x not in H2_COLUMNS)

# Exact V7 PairLEX-derived group.  V8 H2 already contains all three columns
# because H2 is defined as exact V7 S2 + FullLEX.  Set union is therefore
# idempotent: CP == C0 and CAP == CA.  Keeping this explicit avoids inventing
# a duplicate PairLEX representation or silently changing exact H2.
PAIRLEX_GROUP = ("pairlex", "pairlex_x_margin", "qarel_x_pairlex")
PAIRLEX_INCREMENTAL_GROUP = tuple(x for x in PAIRLEX_GROUP if x not in H2_COLUMNS)

CONFIG_COLUMNS = {
    "C0": tuple(H2_COLUMNS),
    "CA": tuple(H3_COLUMNS),
    "CP": tuple(dict.fromkeys((*H2_COLUMNS, *PAIRLEX_GROUP))),
    "CAP": tuple(dict.fromkeys((*H2_COLUMNS, *AGREEMENT_GROUP, *PAIRLEX_GROUP))),
}
SYSTEM_TO_CONFIG = {"S1": "C0", "D-A": "CA", "D-P": "CP", "D-AP": "CAP"}
PRIMARY_SYSTEMS = ("S0", "S1", "S2")
DIAGNOSTIC_SYSTEMS = ("D-A", "D-P", "D-AP")


def frozen_contract() -> dict:
    return {
        "experiment": "V9 / H2 Adaptive Component Selection",
        "version": "9.0.0",
        "development_status": "development",
        "primary_mode": "LOCAL",
        "cells": list(CELLS),
        "families": FAMILIES,
        "languages": LANGUAGES,
        "canonical_protocol": "exact V8 HRR",
        "systems": {"S0": "H0/B0", "S1": "fixed exact V8 H2", "S2": "H2-ACS"},
        "diagnostics": list(DIAGNOSTIC_SYSTEMS),
        "candidate_configs": {k: list(v) for k, v in CONFIG_COLUMNS.items()},
        "h2_columns": list(H2_COLUMNS),
        "agreement_group": list(AGREEMENT_GROUP),
        "pairlex_group": list(PAIRLEX_GROUP),
        "pairlex_incremental_group": list(PAIRLEX_INCREMENTAL_GROUP),
        "pairlex_already_in_exact_h2": not bool(PAIRLEX_INCREMENTAL_GROUP),
        "cap_rule": "set union only; no Agreement×PairLEX interaction",
        "target": "TokenF1(Ai,G)-TokenF1(A0,G)",
        "harm_target": "1[utility<0]",
        "question_total_weight": 1.0,
        "alpha_grid": list(ALPHA_GRID),
        "harm_C_grid": list(HARM_C_GRID),
        "tau_utility_grid": list(TAU_UTILITY_GRID),
        "tau_harm_grid": list(TAU_HARM_GRID),
        "outer_folds": len(OUTER_FOLDS),
        "config_tuning_meta_folds": INNER_FOLDS,
        "select_folds": SELECT_FOLDS,
        "select_split_seed": SELECT_SPLIT_SEED,
        "model_seeds": list(SEEDS),
        "selector": {
            "eps_select": EPS_SELECT,
            "nonnegative_folds_required": 3,
            "worst_fold_delta_min": WORST_SELECT_FOLD,
            "incremental_hit_min_pp": INCREMENTAL_HIT_MIN_PP,
            "incremental_hn_max_pp": INCREMENTAL_HN_MAX_PP,
            "absolute_hit_min_pp": ABSOLUTE_HIT_MIN_PP,
            "absolute_hn_max_pp": ABSOLUTE_HN_MAX_PP,
            "stable_change": "different from C0 in >=2/3 seeds",
            "minimum_activity": "max(5, ceil(1% TRAIN/CAL questions))",
            "complexity_tie_f1": COMPLEXITY_TIE,
        },
        "bootstrap": {"iterations": 5000, "seed": 42, "cluster": "canonical_document_id"},
        "no_manual_dataset_language_aspect_or_k_rules": True,
        "no_new_neural_models": True,
        "same_neural_forwards_s1_s2": True,
        "historical_roots_read_only": True,
    }

