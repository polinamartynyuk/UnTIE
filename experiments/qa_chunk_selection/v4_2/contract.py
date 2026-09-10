"""V4.2 contract: isolated writes; V4.1 and all prior trees stay read-only."""

from __future__ import annotations

from experiments.qa_chunk_selection.v4_1.contract import (
    E0_ARTIFACT_ROOT,
    HISTORICAL_B0,
    SOURCE_OVERLAP,
    V2_ARTIFACT_ROOT,
    V3_ARTIFACT_ROOT,
    V4_ARTIFACT_ROOT,
    V4_1_ARTIFACT_ROOT,
    V4_V6_WEIGHTS,
    PROTECTED_WRITE_ROOTS as V4_1_PROTECTED_WRITE_ROOTS,
)

V4_2_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v4_2"
PROTOCOL = "hypothesis/v4_2_selected_head_attention_protocol.md"

PROTECTED_WRITE_ROOTS = V4_1_PROTECTED_WRITE_ROOTS + (V4_1_ARTIFACT_ROOT,)

HISTORICAL_V6_K5 = {
    "token_f1": 34.624867427228835,
    "exact_match": 0.25312729948491536,
    "hit@1": 0.4289919058130979,
    "hit@5": 0.7902869757174393,
    "hn@1": 0.054451802796173655,
    "n": 1359,
}

PRIMARY_K = 5
N_BOOTSTRAP = 5000
EPSILON_HN = 0.02
READER_EASY_THRESHOLD = 80.0
MIN_PAIRED_QUESTIONS = 30
N_HEAD_GRID = (1, 2, 4, 8)
DELTA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
CORE_MAJORITY_GRID = (0.25, 0.5, 1.0, 2.0)
CORE_TOKEN_GRID = (0.0, 0.25, 0.5, 1.0)
ATT3_TAU_GRID = (-1.0, -0.5, 0.0)
ATT3_LAMBDA_GRID = (0.1, 0.25, 0.5)
NONINFERIORITY_F1 = -0.5
ATTENTION_FEATURE_VERSION = "v4.2-bi-enrich-idf-v2"
QA_INFERENCE_HASH = "v4-topk"
Z_EPS = 1e-8
PRIMARY_FINAL_SYSTEMS = ("B0-live", "V6-K5", "CORE1", "CORE2", "ATT1")


def experiment_contract_v4_2() -> dict[str, object]:
    return {
        "prompt": PROTOCOL,
        "v4_2_artifacts": V4_2_ARTIFACT_ROOT,
        "v4_1_artifacts": V4_1_ARTIFACT_ROOT,
        "v4_artifacts": V4_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "prior_trees": "read-only; never overwrite V4.1 or earlier",
        "internal_test": "locked",
        "gold": "TRAIN-only reader-easy F1>=80 for head discovery; never runtime ranking",
        "ambiguous": "never negative supervision for heads / N / delta",
        "gsrn": "EVAL diagnostics and transitions only",
        "do_not_retune_v6": True,
        "v6_weights": "exact V4.1 per-run weights",
        "primary_k": PRIMARY_K,
        "k10_attention": "not in this experiment",
        "u80_utility": "not mixed into V4.2",
        "reader_easy_threshold": READER_EASY_THRESHOLD,
        "min_paired_questions": MIN_PAIRED_QUESTIONS,
        "n_head_grid": list(N_HEAD_GRID),
        "delta_grid": list(DELTA_GRID),
        "core_majority_grid": list(CORE_MAJORITY_GRID),
        "core_token_grid": list(CORE_TOKEN_GRID),
        "hn_epsilon": EPSILON_HN,
        "bootstrap": N_BOOTSTRAP,
        "primary_attention_feature": "bidirectional(Q->span enrichment, span->Q density) then shortlist z",
        "head_selection_universe": "unique TRAIN union of B0 top-5 across seeds",
        "primary_negative": "valid-span training-eligible HN",
        "fallback_negative": "valid-span high-ranked easy from same union",
        "source_overlap": SOURCE_OVERLAP,
        "v4_v6_weights_default": dict(V4_V6_WEIGHTS),
        "attention_feature_version": ATTENTION_FEATURE_VERSION,
        "att2_implementation": (
            "IDF-weighted Q->span: M = mean_q sum_s A(q,s)*idf(word(s)), then same |S|/|C| enrichment; "
            "span->Q stays unweighted. Per-span-token Q masses are stored; raw [L,H,T,T] tensors are not."
        ),
        "historical_b0": dict(HISTORICAL_B0),
        "historical_v6_k5": dict(HISTORICAL_V6_K5),
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "primary_final_systems": list(PRIMARY_FINAL_SYSTEMS),
        "reused_modules": [
            "experiments.qa_chunk_selection.v4_1.evaluate",
            "experiments.qa_chunk_selection.v4_1.ranking",
            "experiments.qa_chunk_selection.v3.attention.token_masks_from_encoding",
            "experiments.qa_chunk_selection.v4.v6_reader.agreement_features",
            "experiments.qa_chunk_selection.v4.shortlist",
            "experiments.qa_chunk_selection.v4.qa_shortlist",
        ],
    }
