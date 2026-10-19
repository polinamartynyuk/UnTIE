"""Frozen, dataset-independent scientific contract for V8."""
from __future__ import annotations

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection_v7.contract import CELLS, FAMILIES, LANGUAGES

V4_4_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4"
V4_4_EXTERNAL_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_external_20260913_213335"
V4_5_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_5_ru/run_20260915_155234"
V5_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v5/run_20260916_135455"
V6_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v6_universal/run_20260922_132020"
V7_RUN = PROJECT_ROOT / "artifacts/qa_chunk_selection_v7_rapr/run_20260922_181837"

SEEDS = (42, 43, 44)
OUTER_FOLDS = tuple(range(5))
INNER_FOLDS = 4
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
HARM_C_GRID = (0.01, 0.1, 1.0)
TAU_UTILITY_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
TAU_HARM_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
V44_WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)

# The exact V5/V7 canonical feature tables contain no historical semantic
# embeddings (sem_norm=0, sem_available=0).  Availability is decided before
# evaluation and never from quality results.
SEM_COMMON = False
SUPPORT_COMMON = True

H1_COLUMNS = (
    "delta_b0_norm", "delta_qarel_raw", "pairlex", "candidate_rank_norm",
    "effective_k", "gap12_norm", "b0_entropy_norm", "answer_valid_i",
    "answer_valid_0", "both_empty", "qarel_x_margin", "pairlex_x_margin",
    "qarel_x_pairlex", "b0_x_margin",
)
H2_COLUMNS = H1_COLUMNS + ("delta_fulllex", "fulllex_x_margin", "qarel_x_fulllex")
H3_COLUMNS = H2_COLUMNS + (
    "delta_agreement", "agreement_x_margin", "fulllex_x_agreement",
)
H4_COLUMNS = H3_COLUMNS + ("delta_sem",)

F_BASE = ("delta_s44", "candidate_rank_norm", "effective_k", "v44_margin", "b0_margin")
F1_COLUMNS = F_BASE + ("delta_fulllex", "fulllex_x_v44_margin")
F2_COLUMNS = F1_COLUMNS + ("delta_pairlex", "pairlex_x_v44_margin", "fulllex_x_pairlex")
F3_COLUMNS = F2_COLUMNS + ("delta_qarel", "qarel_x_v44_margin", "fulllex_x_qarel")
F4_COLUMNS = F3_COLUMNS + ("delta_sem", "sem_x_v44_margin", "fulllex_x_sem")

BRANCH_HRR_SYSTEMS = ("H0", "H1", "H2", "H3") + (("H4",) if SEM_COMMON else ())
BRANCH_AF_SYSTEMS = ("F0", "F1", "F2", "F3", "F5-core") if not SEM_COMMON else ("F0", "F1", "F2", "F3", "F4", "F5-sem")
PRIMARY_HRR = "H3"
PRIMARY_AF = "F5-sem" if SEM_COMMON else "F5-core"

def hrr_columns(system: str) -> tuple[str, ...]:
    return {"H1": H1_COLUMNS, "H2": H2_COLUMNS, "H3": H3_COLUMNS, "H4": H4_COLUMNS}[system]

def af_columns(system: str) -> tuple[str, ...]:
    return {"F1": F1_COLUMNS, "F2": F2_COLUMNS, "F3": F3_COLUMNS,
            "F4": F4_COLUMNS, "F5-core": F3_COLUMNS, "F5-sem": F4_COLUMNS}[system]

def frozen_contract(branch: str) -> dict:
    common = {
        "experiment": "V8 dual-branch universal reader reranking",
        "version": "8.0.0", "development_status": "development",
        "cells": list(CELLS), "families": FAMILIES, "languages": LANGUAGES,
        "canonical_baseline": "exact V7/V5 B0, question policy, candidates and K",
        "seeds": list(SEEDS), "outer_folds": 5, "inner_meta_folds": INNER_FOLDS,
        "primary_mode": "LOCAL", "secondary_modes": ["POOLED-simple", "LOCO-meta"],
        "alpha_grid": list(ALPHA_GRID), "harm_C_grid": list(HARM_C_GRID),
        "tau_utility_grid": list(TAU_UTILITY_GRID), "tau_harm_grid": list(TAU_HARM_GRID),
        "selection_source": "META_PRED_OOF only", "question_total_weight": 1.0,
        "safety": {"delta_hit1_min_pp": -0.5, "delta_hn1_max_pp": 1.0},
        "bootstrap": {"iterations": 5000, "seed": 42, "cluster": "canonical_document_id"},
        "sem_common": SEM_COMMON, "sem_decision_basis": "canonical V5/V7 runtime has sem_available=0 for every cell",
        "support_common": SUPPORT_COMMON, "no_manual_language_rules": True,
        "no_posthoc_feature_removal": True,
    }
    if branch == "hrr":
        return common | {
            "branch": "V8-HRR", "systems": list(BRANCH_HRR_SYSTEMS), "primary": PRIMARY_HRR,
            "features": {s: list(hrr_columns(s)) for s in BRANCH_HRR_SYSTEMS if s != "H0"},
            "target": "TokenF1(Ai,G)-TokenF1(A0,G)", "fallback": "H0/B0",
            "full_top_k_qa": True, "pre_qa_admission": False,
        }
    if branch == "v44_af":
        return common | {
            "branch": "V8-V44-AF", "systems": list(BRANCH_AF_SYSTEMS), "primary": PRIMARY_AF,
            "features": {s: list(af_columns(s)) for s in BRANCH_AF_SYSTEMS if s != "F0"},
            "v44_core": "alpha*B0 + beta*(exact_freq+mean_token_f1+majority) + gamma*source_clusters",
            "v44_weight_grid": list(V44_WEIGHT_GRID), "anchor": "argmax fitted V44-CORE score",
            "target": "TokenF1(Ai,G)-TokenF1(A44,G)", "fallback": "F0/C44",
            "pairlex_reference": "exact V7 B0 answer",
        }
    raise ValueError(branch)
