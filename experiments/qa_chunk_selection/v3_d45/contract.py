"""Isolated D4 oracle + D5 audit. Does not write E0/V2/V3/V3.1 trees."""

from __future__ import annotations

from experiments.qa_chunk_selection.v3.contract import (
    E0_ARTIFACT_ROOT,
    PROTECTED_WRITE_ROOTS as V3_PROTECTED_WRITE_ROOTS,
    V2_ARTIFACT_ROOT,
    V3_ARTIFACT_ROOT,
)
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import V3_1_AUTO_ARTIFACT_ROOT, V3_1_RANKING_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v3_1_prep.contract import V3_1_PREP_ARTIFACT_ROOT

V3_D45_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v3_d45"
PROTECTED_WRITE_ROOTS = V3_PROTECTED_WRITE_ROOTS + (
    V3_ARTIFACT_ROOT,
    V3_1_PREP_ARTIFACT_ROOT,
    V3_1_AUTO_ARTIFACT_ROOT,
    V3_1_RANKING_ARTIFACT_ROOT,
    "artifacts/qa_chunk_selection_v4",
)
PROTOCOL = "hypothesis/v3_1_experiment_continuation_automated_overlay.md"


def experiment_contract_v3_d45() -> dict[str, object]:
    return {
        "prompt": PROTOCOL,
        "also": "hypothesis/qa_chunk_selection_experiments_v3_plan_attention.md §§27–28",
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "v3_d45_artifacts": V3_D45_ARTIFACT_ROOT,
        "d4": "oracle diagnostic only; no sentence-selector training",
        "d5": "independent-evidence audit after overlap dedup; no aggregation model",
        "internal_test": "locked",
        "source_label": "never rewritten",
        "gold": "diagnostic windows only, not runtime supervision for a selector",
        "reused_modules": [
            "experiments.qa_chunk_selection.progress",
            "experiments.qa_chunk_selection.models.load_frozen_qa",
            "experiments.qa_chunk_selection.v2.cache",
            "experiments.qa_chunk_selection.v2.qa",
            "experiments.qa_chunk_selection.v2.sentence_evidence",
            "experiments.qa_chunk_selection.v3.data",
            "experiments.qa_chunk_selection.v3.split",
            "experiments.qa_chunk_selection.v3.positive_quality",
            "experiments.qa_chunk_selection.v3_1_auto_prep.cluster",
        ],
    }
