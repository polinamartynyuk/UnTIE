"""V4.1 contract: isolated writes; V4 and all prior trees stay read-only."""

from __future__ import annotations

from experiments.qa_chunk_selection.v4.contract import (
    E0_ARTIFACT_ROOT,
    LOCKED_SPLITS,
    POSTHOC_SPLITS,
    PROTECTED_WRITE_ROOTS as V4_PROTECTED_WRITE_ROOTS,
    SOURCE_OVERLAP,
    UNLOCKED_V2_SPLITS,
    V2_ARTIFACT_ROOT,
    V3_ARTIFACT_ROOT,
    V4_ARTIFACT_ROOT,
)
from experiments.qa_chunk_selection.v4.config import FUSION_LAMBDAS

V4_1_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v4_1"
PROTOCOL = "hypothesis/v4_1_reader_centered_continuation_protocol.md"

PROTECTED_WRITE_ROOTS = V4_PROTECTED_WRITE_ROOTS + (V4_ARTIFACT_ROOT,)

# Historical V3-B0 / V4 numbers used as R0 regression targets (0-100 Token F1).
HISTORICAL_B0 = {
    "token_f1": 27.358413858227983,
    "exact_match": 0.20750551876379691,
    "hit@1": 0.3671817512877116,
    "hit@5": 0.7902869757174393,
    "hit@10": 0.9220014716703459,
    "hn@1": 0.04856512141280353,
    "ambiguous_at_1_original": 0.5084621044885945,
    "gsrn_at_1_original": 0.33922001471670343,
    "pairwise_clean_hn": 0.7513879233769745,
    "n": 1359,
}
HISTORICAL_V4_V6_TOKEN_F1 = 35.044000944588475
HISTORICAL_V4_V2A4_TOKEN_F1 = 30.255623156662235
V4_V6_WEIGHTS = {"alpha": 2.0, "beta": 0.25, "gamma": 0.5}
V4_V2A4_LAMBDA = 1.0
UTILITY_F1_PCT = 80.0
UTILITY_F1_UNIT_MISTAKE = 0.8
REPRO_F1_TOLERANCE = 1e-6
B0_IDENTITY_TOLERANCE = 1e-6
PAIRWISE_TOLERANCE = 1e-4
EPSILON_HN = 0.02
N_BOOTSTRAP = 5000
PRIMARY_K = 5
K_CANDIDATES = (5, 10)
V6_ALPHA_GRID = FUSION_LAMBDAS
V6_BETA_GRID = FUSION_LAMBDAS
V6_GAMMA_GRID = FUSION_LAMBDAS
LAMBDA_AGG_GRID = FUSION_LAMBDAS
ETA_GRID = (0.25, 0.5, 0.75)
QA_INFERENCE_HASH = "v4-topk"
PRIMARY_FINAL_SYSTEMS = ("B0-live", "V6-K5", "C1", "V6-K10", "C1-K10")


def experiment_contract_v4_1() -> dict[str, object]:
    return {
        "prompt": PROTOCOL,
        "v4_1_artifacts": V4_1_ARTIFACT_ROOT,
        "v4_artifacts": V4_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "prior_trees": "read-only; never overwrite V4 or earlier",
        "internal_test": "locked until freeze manifest; default run does not open it",
        "source_label": "never rewritten",
        "gold": "scored metrics and TRAIN-only U80 utility mining; never runtime ranking",
        "ambiguous": "unlabeled, never automatic TE/TM or strong negative",
        "b0_live": "historical V3-B0 max5 OOF jsonl; pairwise taken from that jsonl, not V4 rescored 0.41 scale",
        "pairing": "(question_id, seed) vs matching B0 seed",
        "bootstrap": "mean delta across seeds within question, then cluster document_id, n_boot>=5000",
        "shortlist": "argmax only inside exact B0 top-K; Hit@K invariant",
        "v6_feature_scale": "exactly V4 agreement_features: exact_freq count, mean_token_f1 0..100, majority 0/1, source_clusters count",
        "v6_alpha_grid": list(V6_ALPHA_GRID),
        "v6_beta_grid": list(V6_BETA_GRID),
        "v6_gamma_grid": list(V6_GAMMA_GRID),
        "lambda_agg_grid": list(LAMBDA_AGG_GRID),
        "epsilon_hn": EPSILON_HN,
        "source_overlap": SOURCE_OVERLAP,
        "primary_k": PRIMARY_K,
        "k_candidates": list(K_CANDIDATES),
        "primary_final_systems": list(PRIMARY_FINAL_SYSTEMS),
        "utility_threshold_pct": UTILITY_F1_PCT,
        "eta_grid": list(ETA_GRID),
        "v4_v6_weights": dict(V4_V6_WEIGHTS),
        "v4_v2a4_lambda": V4_V2A4_LAMBDA,
        "unlocked_v2_splits": list(UNLOCKED_V2_SPLITS),
        "locked_splits": list(LOCKED_SPLITS),
        "posthoc_splits": list(POSTHOC_SPLITS),
        "reused_modules": [
            "experiments.qa_chunk_selection.progress",
            "experiments.qa_chunk_selection.models.load_frozen_qa",
            "experiments.qa_chunk_selection.metrics",
            "experiments.qa_chunk_selection.v2.cache",
            "experiments.qa_chunk_selection.v3.data",
            "experiments.qa_chunk_selection.v3.split",
            "experiments.qa_chunk_selection.v4.shortlist",
            "experiments.qa_chunk_selection.v4.evaluate",
            "experiments.qa_chunk_selection.v4.v6_reader",
            "experiments.qa_chunk_selection.v4.v2_aggregate",
            "experiments.qa_chunk_selection.v4.answers",
            "experiments.qa_chunk_selection.v4.clusters",
            "experiments.qa_chunk_selection.v4.qa_shortlist",
            "experiments.qa_chunk_selection.v4.v3_utility",
            "experiments.qa_chunk_selection.v4.heads",
        ],
    }
