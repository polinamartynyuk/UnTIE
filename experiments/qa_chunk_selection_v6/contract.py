"""Frozen scientific contract for V6/UAR-G."""
from __future__ import annotations

from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS

V5_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v5/run_20260916_135455"
CELLS = ("scirex", "scier_384", "scier_128", "russerrc", "scimdix")
FAMILIES = {"scirex": "scirex", "scier_384": "scier", "scier_128": "scier", "russerrc": "russerrc", "scimdix": "scimdix"}
LANGUAGES = {key: V5_CELLS[key].language for key in CELLS}
SEEDS = (42, 43, 44)
OUTER_FOLDS = tuple(range(5))
META_FOLDS = 4
C_GRID = (0.1, 1.0, 10.0)
TAU_SWITCH_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
GATE_BUDGETS = (1.25, 1.50, 2.00)
PRIMARY_BUDGET = 2.0
EPSILON = 1e-6

ASPECT_MAP = {
    "method": "METHOD", "метод": "METHOD", "task": "TASK", "задача": "TASK",
    "material": "MATERIAL", "материал": "MATERIAL", "tool": "TOOL", "инструмент": "TOOL",
    "goal": "GOAL", "aim": "GOAL", "цель": "GOAL", "use": "USE", "usage": "USE",
    "использование": "USE", "metric": "METRIC", "метрика": "METRIC", "result": "RESULT",
    "результат": "RESULT", "conclusion": "CONCLUSION", "вывод": "CONCLUSION",
    "contribution": "CONTRIBUTION", "вклад": "CONTRIBUTION", "advantage": "ADVANTAGE",
    "преимущество": "ADVANTAGE", "example": "EXAMPLE", "пример": "EXAMPLE",
}

CONTEXT_NUMERIC = (
    "effective_k", "candidate_count", "gap12_norm", "gap13_norm", "top1_z",
    "b0_entropy_norm", "flat_score_vector", "has_rank3", "delta_answer_valid",
)
CORE_NUMERIC = (
    "delta_b0", "delta_qarel", *CONTEXT_NUMERIC,
    "qarel_x_k", "qarel_x_margin", "b0_x_margin", "qarel_x_entropy", "b0_x_qarel",
)
SOURCE_NUMERIC = (
    "delta_source", "source_x_k", "source_x_margin",
)
LEX_NUMERIC = ("delta_lex", "lex_x_k", "lex_x_margin")
GATE_NUMERIC = (
    "effective_k", "candidate_count", "gap12_norm", "gap13_norm", "has_rank3",
    "top1_z", "b0_entropy_norm", "flat_score_vector",
)
ASPECTS = tuple(sorted(set(ASPECT_MAP.values()) | {"OTHER"}))
ASPECT_INTERACTIONS = tuple(f"qarel_x_aspect__{name}" for name in ASPECTS)
SOURCE_ASPECT_INTERACTIONS = tuple(f"source_x_aspect__{name}" for name in ASPECTS)

VARIANTS = {
    "S1": {"target": "v5_evidence_mirrored", "gate": False, "signals": "core", "eligible": False},
    "S2": {"target": "positive_token_f1_utility", "gate": False, "signals": "core", "eligible": True},
    "S3": {"target": "positive_token_f1_utility", "gate": True, "signals": "core", "eligible": True},
    "S4-src": {"target": "positive_token_f1_utility", "gate": True, "signals": "source", "eligible": False},
    "S4-lex": {"target": "positive_token_f1_utility", "gate": True, "signals": "lex", "eligible": False},
    "S4": {"target": "positive_token_f1_utility", "gate": True, "signals": "source_lex", "eligible": True},
    "S5": {"target": "positive_token_f1_utility", "gate": True, "signals": "full_optional", "eligible": False},
}


def feature_columns(variant: str, *, aspect: bool = True) -> tuple[str, ...]:
    names = list(CORE_NUMERIC)
    if aspect:
        names += list(ASPECT_INTERACTIONS)
    signals = VARIANTS[variant]["signals"]
    if signals in {"source", "source_lex", "full_optional"}:
        names += list(SOURCE_NUMERIC)
        if aspect:
            names += list(SOURCE_ASPECT_INTERACTIONS)
    if signals in {"lex", "source_lex", "full_optional"}:
        names += list(LEX_NUMERIC)
    if signals == "full_optional":
        names += ["delta_sem", "delta_qaconf"]
    return tuple(names)


def frozen_contract() -> dict:
    return {
        "name": "V6/UAR-G", "version": "6.0.0", "cells": list(CELLS), "families": FAMILIES,
        "seeds": list(SEEDS), "outer_folds": 5, "inner_meta_folds": META_FOLDS,
        "historical_v5_run": str(V5_RUN.resolve()),
        "historical_models": {key: {"ranker": V5_CELLS[key].ranker_id, "revision": V5_CELLS[key].ranker_revision,
                                      "qa": str(V5_CELLS[key].qa_model.resolve()), "k": V5_CELLS[key].k,
                                      "question_policy": V5_CELLS[key].question_policy} for key in CELLS},
        "variants": VARIANTS, "C_grid": list(C_GRID), "tau_switch_grid": list(TAU_SWITCH_GRID),
        "gate_budgets": list(GATE_BUDGETS), "primary_budget": PRIMARY_BUDGET,
        "utility_target": "1[TokenF1(Ai,gold)-TokenF1(A0,gold)>0]",
        "calibration": "Platt on META_PRED_OOF", "normalization": "within-query z, std floor 1e-6",
        "weighting": ["question total=1", "SciER lineage total=1", "POOLED family totals equal", "seeds isolated"],
        "modes": ["LOCAL", "POOLED", "LOCO-meta"], "pooled_dataset_id": False,
        "interactions": {name: list(feature_columns(name)) for name in VARIANTS},
        "safety": {"delta_hit1_min_pp": -0.5, "delta_hn1_max_pp": 1.0},
        "missing": {"invalid_answer_qarel": "within-query minimum", "optional_signal": 0.0,
                    "single_class": "no switch/no route", "tie": "keep B0"},
        "s5_status": "diagnostic; SEM/QAConf only when exact historical signal is universally replayable",
    }

