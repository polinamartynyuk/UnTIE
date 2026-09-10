"""V3.1-Silver ranking contract. Isolated writes."""

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
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import V3_1_AUTO_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v3_1_prep.contract import V3_1_PREP_ARTIFACT_ROOT

V3_1_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v3_1"
PROTECTED_WRITE_ROOTS = V3_PROTECTED_WRITE_ROOTS + (
    V3_ARTIFACT_ROOT,
    V3_1_PREP_ARTIFACT_ROOT,
    V3_1_AUTO_ARTIFACT_ROOT,
    "artifacts/qa_chunk_selection_v4",
)
PROTOCOL = "hypothesis/v3_1_experiment_continuation_automated_overlay.md"


def experiment_contract_v3_1() -> dict[str, object]:
    return {
        "prompt": PROTOCOL,
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "v3_1_auto_artifacts": V3_1_AUTO_ARTIFACT_ROOT,
        "v3_1_artifacts": V3_1_ARTIFACT_ROOT,
        "annotator": "frozen extractive QA; no generative teacher",
        "cv": "reuse frozen V3 folds; silver labels are fold-specific TRAIN/CAL thresholds",
        "b0_reference": "B0-max8 fixed a priori",
        "primary_endpoint": "Token F1 on original SciREX gold",
        "internal_test": "locked",
        "source_label": "never rewritten",
        "unlocked_v2_splits": list(UNLOCKED_V2_SPLITS),
        "locked_splits": list(LOCKED_SPLITS),
        "posthoc_splits": list(POSTHOC_SPLITS),
    }
