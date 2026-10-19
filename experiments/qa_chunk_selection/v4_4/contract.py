"""Pre-EVAL choices from the V4.4 protocol, with explicit ambiguity resolutions."""

from experiments.qa_chunk_selection.v4_3.contract import (
    PROTECTED_WRITE_ROOTS as PREVIOUS_ROOTS,
    SEMANTIC_MODEL, NORMALIZATION_VERSION,
)

ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v4_4"
PROTOCOL = "hypothesis/v4_4_adaptive_reader_consensus_protocol.md"
PROTECTED_WRITE_ROOTS = PREVIOUS_ROOTS + ("artifacts/qa_chunk_selection_v4_3",)
STATUS = "cross-fitted development stability replay; NOT independent confirmatory OOF"
SEEDS = (42, 43, 44)
K = 5
SEM_FIXED = {"lambda_m": 0.5, "lambda_s": 0.25}
LEX_FIXED = {"lambda_m": 0.25, "lambda_t": 2.0}
V6_GLOBAL = {"alpha": 0.0, "beta": 0.25, "gamma": 2.0}
FINAL_SEED = 42
MIN_ROUTING_CAL_ROWS = 100
MIN_SUPPORT_CAL_ROWS = 100
N_BOOTSTRAP = 5000


def contract():
    return {
        "protocol": PROTOCOL, "status": STATUS, "primary_k": K, "seeds": list(SEEDS),
        "SEM_FIXED": SEM_FIXED, "LEX_FIXED": LEX_FIXED,
        "semantic_model": SEMANTIC_MODEL, "normalization_version": NORMALIZATION_VERSION,
        "encoder_frozen": True, "v6_retuned": False,
        "routing_confidence": "best-triple cohesion minus between similarity",
        "min_valid_answers": 4, "min_routing_cal_rows": MIN_ROUTING_CAL_ROWS,
        "routing_threshold": "Q70; TRAIN+CAL; available rows only; no gold; linear quantile",
        "threshold_per_reader": True,
        "threshold_row_unit": "(question_id, seed), pooled fixed seeds; not independent samples",
        "hn_primary_margin": 0.01, "hit1_primary_margin": -0.005,
        "h2_superiority_hn_rule": "ALEX HN <= V6 HN + 0.01 (same absolute safety budget as H1)",
        "h2_noninferiority": "lower development CI > -0.5; HN <= ASEM; Hit1 >= ASEM - 0.005",
        "alex_when_h1_fails": "computed as required descriptive control; H2 skipped; cannot win",
        "support_threshold": "Q20 TRAIN+CAL available AND reader top1 != V6 top1; no Q70 filter; no gold",
        "min_support_cal_rows": MIN_SUPPORT_CAL_ROWS,
        "bootstrap": N_BOOTSTRAP, "pairing": "question_id+seed",
        "bootstrap_unit": "mean seeds per question, clustered document_id",
        "final_candidates": ["V6-K5", "ASEM", "ALEX"],
        "final_v6_global": V6_GLOBAL, "final_b0_seed_policy": {"kind": "single", "seed": FINAL_SEED},
        "final_threshold_source": "available cross-fitted rows, canonical seed 42; same minimum 100",
        "final_output": "draft only; B0-final refit and locked test are separate future operations",
        "internal_test": "locked; no runtime/test CLI; mixed snapshot filtered before record decoding",
        "protected_write_roots": list(PROTECTED_WRITE_ROOTS),
    }
