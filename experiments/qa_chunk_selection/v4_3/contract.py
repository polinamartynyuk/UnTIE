"""V4.3 contract: isolated writes; V4.2 and all prior trees stay read-only."""

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
)
from experiments.qa_chunk_selection.v4_2.contract import (
    HISTORICAL_V6_K5,
    V4_2_ARTIFACT_ROOT,
    PROTECTED_WRITE_ROOTS as V4_2_PROTECTED_WRITE_ROOTS,
)

V4_3_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v4_3"
PROTOCOL = "hypothesis/v4_3_semantic_reader_agreement_protocol.md"

PROTECTED_WRITE_ROOTS = V4_2_PROTECTED_WRITE_ROOTS + (V4_2_ARTIFACT_ROOT,)

HISTORICAL_CORE3 = {
    "token_f1": 35.443709671082736,
    "exact_match": 0.3208241353936718,
    "hit@1": 0.41206769683590877,
    "hn@1": 0.06181015452538632,
    "n": 1359,
}

PRIMARY_K = 5
N_BOOTSTRAP = 5000
EPSILON_HN = 0.02
Z_EPS = 1e-8
QA_INFERENCE_HASH = "v4-topk"
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NORMALIZATION_VERSION = "v4.3-nfc-ws-v1"
MIN_VALID_ANSWERS_SEM = 3
MIN_VALID_ANSWERS_ADAPT = 4
CONTEXT_TOKEN_WINDOW = 64
DELTA_SEM_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
LEXCORE_MAJORITY_GRID = (0.25, 0.5, 1.0, 2.0)
LEXCORE_TOKEN_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
SEMCORE_MAJORITY_GRID = (0.25, 0.5, 1.0, 2.0)
SEMCORE_SEM_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
ADAPT_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90)
CONTEXT_DELTA_GRID = (0.0, 0.1, 0.25, 0.5)
CLUSTER_THRESHOLDS = (0.70, 0.80, 0.85, 0.90)
READER_EASY_THRESHOLD = 80.0
NEVER_TAU = float("inf")
ALWAYS_TAU = float("-inf")
PRIMARY_FINAL_SYSTEMS = ("B0-live", "V6-K5", "SEM1")


def experiment_contract_v4_3() -> dict[str, object]:
    return {
        "prompt": PROTOCOL,
        "v4_3_artifacts": V4_3_ARTIFACT_ROOT,
        "v4_2_artifacts": V4_2_ARTIFACT_ROOT,
        "v4_1_artifacts": V4_1_ARTIFACT_ROOT,
        "v4_artifacts": V4_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "prior_trees": "read-only; never overwrite V4.2 or earlier",
        "internal_test": "locked",
        "gold": "CAL/EVAL metrics and TRAIN-only semantic sanity; never runtime ranking",
        "do_not_retune_v6": True,
        "v6_weights": "exact V4.1 per-run weights",
        "primary_k": PRIMARY_K,
        "semantic_model": SEMANTIC_MODEL,
        "encoder_frozen": True,
        "no_encoder_search": True,
        "sem_feature": "mean top-2 cosine similarities excluding self",
        "normalization_version": NORMALIZATION_VERSION,
        "min_valid_answers_sem": MIN_VALID_ANSWERS_SEM,
        "min_valid_answers_adapt": MIN_VALID_ANSWERS_ADAPT,
        "delta_sem_grid": list(DELTA_SEM_GRID),
        "lexcore_majority_grid": list(LEXCORE_MAJORITY_GRID),
        "lexcore_token_grid": list(LEXCORE_TOKEN_GRID),
        "semcore_majority_grid": list(SEMCORE_MAJORITY_GRID),
        "semcore_sem_grid": list(SEMCORE_SEM_GRID),
        "adapt_quantiles": list(ADAPT_QUANTILES),
        "adapt_controls": ["NEVER", "ALWAYS"],
        "context_delta_grid": list(CONTEXT_DELTA_GRID),
        "cluster_thresholds_diagnostic_only": list(CLUSTER_THRESHOLDS),
        "hn_epsilon": EPSILON_HN,
        "bootstrap": N_BOOTSTRAP,
        "k10": "not in V4.3",
        "u80_utility": "not mixed into V4.3",
        "attention": "not in V4.3",
        "hidden_state": "not in V4.3",
        "aspect_routing": "not in V4.3",
        "source_overlap": SOURCE_OVERLAP,
        "v4_v6_weights_default": dict(V4_V6_WEIGHTS),
        "historical_b0": dict(HISTORICAL_B0),
        "historical_v6_k5": dict(HISTORICAL_V6_K5),
        "historical_core3": dict(HISTORICAL_CORE3),
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "qa_inference_hash": QA_INFERENCE_HASH,
        "primary_final_systems": list(PRIMARY_FINAL_SYSTEMS),
        "reused_modules": [
            "experiments.qa_chunk_selection.v4_1.evaluate",
            "experiments.qa_chunk_selection.v4_1.ranking",
            "experiments.qa_chunk_selection.v4_2.cache_util",
            "experiments.qa_chunk_selection.v4_2.select",
            "experiments.qa_chunk_selection.v4_2.features.zscore_map",
            "experiments.qa_chunk_selection.v4.v6_reader.agreement_features",
            "experiments.qa_chunk_selection.v4.shortlist",
            "experiments.qa_chunk_selection.v4.qa_shortlist",
        ],
    }
