"""Frozen scientific contract for V7/RAPR."""
from __future__ import annotations

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS

V5_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v5/run_20260916_135455"
V6_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v6_universal/run_20260922_132020"
CELLS = ("scirex", "scier_384", "scier_128", "russerrc", "scimdix")
FAMILIES = {"scirex":"scirex", "scier_384":"scier", "scier_128":"scier", "russerrc":"russerrc", "scimdix":"scimdix"}
LANGUAGES = {cell: V5_CELLS[cell].language for cell in CELLS}
SEEDS = (42, 43, 44)
OUTER_FOLDS = tuple(range(5))
INNER_FOLDS = 4
MAX_EXTRA_QA = 2
MAX_QA = 3
ALPHA_GRID = (0.1, 1.0, 10.0)
HARM_C_GRID = (0.1, 1.0, 10.0)
TAU_ADMIT_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
TAU_UTILITY_GRID = (0.00, 0.25, 0.50, 1.00, 2.00)
TAU_HARM_GRID = (0.20, 0.30, 0.40, 0.50)
MEAN_QA_BUDGET = 2.0

ADMISSION_NUMERIC = (
    "candidate_rank_norm", "b0_score_norm", "delta_b0_norm", "effective_k",
    "candidate_count", "gap12_norm", "gap13_norm", "b0_entropy_norm", "top1_z",
)
SELECTOR_NUMERIC = (
    "delta_b0_norm", "delta_qarel_raw", "pairlex", "candidate_rank_norm",
    "effective_k", "gap12_norm", "b0_entropy_norm", "answer_valid_i",
    "answer_valid_0", "both_empty", "qarel_x_margin", "pairlex_x_margin",
    "qarel_x_pairlex", "b0_x_margin",
)
SELECTOR_NOLEX_NUMERIC = tuple(x for x in SELECTOR_NUMERIC if x not in {
    "pairlex", "pairlex_x_margin", "qarel_x_pairlex"
})
SYSTEMS = {
    "S0": {"eligible": False, "full_qa": False, "harm": False, "admission": False},
    "S1": {"eligible": False, "full_qa": True, "harm": False, "admission": False},
    "S2": {"eligible": False, "full_qa": True, "harm": True, "admission": False},
    "S3": {"eligible": True, "full_qa": False, "harm": True, "admission": True},
    "S3-noLEX": {"eligible": False, "full_qa": False, "harm": True, "admission": True},
    "S3-B0-order": {"eligible": False, "full_qa": False, "harm": True, "admission": True},
    "S3-cap1": {"eligible": False, "full_qa": False, "harm": True, "admission": True},
}

def selector_columns(system: str) -> tuple[str, ...]:
    return SELECTOR_NOLEX_NUMERIC if system == "S3-noLEX" else SELECTOR_NUMERIC

def frozen_contract() -> dict:
    return {
        "name": "V7/RAPR", "version": "7.0.0", "development_status": "V7-A development",
        "cells": list(CELLS), "families": FAMILIES, "languages": LANGUAGES,
        "historical_v5_run": str(V5_RUN.resolve()), "historical_v6_run": str(V6_RUN.resolve()),
        "seeds": list(SEEDS), "outer_folds": 5, "inner_meta_folds": INNER_FOLDS,
        "primary_mode": "LOCAL", "secondary_modes": ["POOLED-simple", "LOCO-meta"],
        "systems": SYSTEMS, "primary_system": "S3",
        "max_extra_qa": MAX_EXTRA_QA, "max_qa_per_question": MAX_QA,
        "targets": {"utility": "TokenF1(Ai,G)-TokenF1(A0,G)", "harm": "1[utility<0]", "admission": "1[utility>0]"},
        "models": {"utility": "StandardScaler+Ridge", "harm": "StandardScaler+OneHot(language)+LogisticRegression(liblinear,L2)",
                   "admission": "StandardScaler+OneHot(language)+LogisticRegression(liblinear,L2,C=1)"},
        "alpha_grid": list(ALPHA_GRID), "harm_C_grid": list(HARM_C_GRID),
        "tau_admit_grid": list(TAU_ADMIT_GRID), "tau_utility_grid": list(TAU_UTILITY_GRID),
        "tau_harm_grid": list(TAU_HARM_GRID), "mean_qa_budget": MEAN_QA_BUDGET,
        "pairlex": "TokenF1(predicted_answer_i,predicted_answer_0)/100",
        "features": {"admission": list(ADMISSION_NUMERIC), "selector": list(SELECTOR_NUMERIC),
                     "categorical": ["language"], "forbidden": ["aspect", "source", "SEM", "QA confidence", "dataset_id"]},
        "interactions": ["delta_qarel_raw*gap12_norm", "pairlex*gap12_norm", "delta_qarel_raw*pairlex", "delta_b0_norm*gap12_norm"],
        "qarel_normalization": "raw oriented delta to A0, then TRAIN-fitted StandardScaler",
        "weights": ["question total=1", "SciER lineage total=1", "POOLED family totals equal", "seeds isolated"],
        "selection": "META_PRED_OOF only; staged alpha, harm C, then system-specific policy",
        "safety": {"delta_hit1_min_pp": -0.5, "delta_hn1_max_pp": 1.0},
        "fallback": "B0_FALLBACK when no safe/resource-feasible OOF policy",
        "empty_answer": {"one_empty_pairlex": 0.0, "both_empty_pairlex": 1.0, "unsupported_empty_qarel_delta": 0.0},
        "no_posthoc_substitution": True,
    }
