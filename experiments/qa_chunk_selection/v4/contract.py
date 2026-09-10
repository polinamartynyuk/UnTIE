"""V4 contract: isolated writes, prior experiment trees read-only."""

from __future__ import annotations

from experiments.qa_chunk_selection.v3.contract import (
    E0_ARTIFACT_ROOT,
    LOCKED_SPLITS,
    POSTHOC_SPLITS,
    PROTECTED_WRITE_ROOTS as V3_PROTECTED_WRITE_ROOTS,
    UNLOCKED_V2_SPLITS,
    V2_ARTIFACT_ROOT,
    V3_ARTIFACT_ROOT,
)
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import (
    V3_1_AUTO_ARTIFACT_ROOT,
    V3_1_RANKING_ARTIFACT_ROOT,
)
from experiments.qa_chunk_selection.v3_1_prep.contract import V3_1_PREP_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v3_d45.contract import V3_D45_ARTIFACT_ROOT

V3_1_AUTO_V2_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v3_1_auto_v2"
V4_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v4"

PROTECTED_WRITE_ROOTS = V3_PROTECTED_WRITE_ROOTS + (
    V3_ARTIFACT_ROOT,
    V3_1_PREP_ARTIFACT_ROOT,
    V3_1_AUTO_ARTIFACT_ROOT,
    V3_1_AUTO_V2_ARTIFACT_ROOT,
    V3_1_RANKING_ARTIFACT_ROOT,
    V3_D45_ARTIFACT_ROOT,
)

PROTOCOL = "hypothesis/v4_experiment_plan_7_strategies.md"
KEYBERT_BACKEND = "sentence-transformers/all-MiniLM-L6-v2"
KEYBERT_N = 10
KEYBERT_NGRAM = (1, 3)
KEYBERT_USE_MMR = False
SOURCE_OVERLAP = 0.6
UTILITY_F1 = 0.80
UTILITY_ETA = 0.50


def experiment_contract_v4() -> dict[str, object]:
    return {
        "prompt": PROTOCOL,
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "v3_1_prep_artifacts": V3_1_PREP_ARTIFACT_ROOT,
        "v3_1_auto_artifacts": V3_1_AUTO_ARTIFACT_ROOT,
        "v3_1_auto_v2_artifacts": V3_1_AUTO_V2_ARTIFACT_ROOT,
        "v3_1_ranking_artifacts": V3_1_RANKING_ARTIFACT_ROOT,
        "v3_d45_artifacts": V3_D45_ARTIFACT_ROOT,
        "v4_artifacts": V4_ARTIFACT_ROOT,
        "prior_trees": "read-only; never overwrite",
        "b0_live": "historical V3-B0 max5 OOF; not silently replaced by max8",
        "internal_test": "locked",
        "source_label": "never rewritten",
        "gold": "scored metrics and TRAIN-only utility mining only; never runtime ranking",
        "ambiguous": "unlabeled, never automatic TE/TM or strong negative",
        "qa_silver": "not reused as TE/TM supervision",
        "keybert_backend": KEYBERT_BACKEND,
        "keybert_n": KEYBERT_N,
        "keybert_ngram": list(KEYBERT_NGRAM),
        "keybert_mmr": KEYBERT_USE_MMR,
        "source_overlap_threshold": SOURCE_OVERLAP,
        "v1_primary": "LOSCO, not simple LOCO",
        "runtime_clusters": "source-overlap of all/shortlist chunks, not D5 gold clusters",
        "v4_role": "diagnostic only, not a production ranker",
        "posthoc_softmax": "invariance control only",
        "synthetic_wrong_document": "requires target-document KeyBERT anchor",
        "unlocked_v2_splits": list(UNLOCKED_V2_SPLITS),
        "locked_splits": list(LOCKED_SPLITS),
        "posthoc_splits": list(POSTHOC_SPLITS),
        "reused_modules": [
            "experiments.qa_chunk_selection.progress",
            "experiments.qa_chunk_selection.models.load_frozen_qa",
            "experiments.qa_chunk_selection.metrics",
            "experiments.qa_chunk_selection.v2.cache",
            "experiments.qa_chunk_selection.v2.qa",
            "experiments.qa_chunk_selection.v3.data",
            "experiments.qa_chunk_selection.v3.split",
            "experiments.qa_chunk_selection.v3.evaluate",
            "experiments.qa_chunk_selection.v3.train",
            "experiments.qa_chunk_selection.v3.metrics_v3",
            "experiments.qa_chunk_selection.v3_1_auto_prep.cluster",
            "experiments.keybert_topic_clustering.encoder",
        ],
    }
