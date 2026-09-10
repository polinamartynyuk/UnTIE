"""V3.1 QA-based silver overlay contract. Isolated from E0/V2/V3/manual prep."""

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
from experiments.qa_chunk_selection.v3_1_prep.contract import V3_1_PREP_ARTIFACT_ROOT

V3_1_AUTO_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v3_1_auto"
V3_1_RANKING_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v3_1"

PROTECTED_WRITE_ROOTS = V3_PROTECTED_WRITE_ROOTS + (
    V3_ARTIFACT_ROOT,
    V3_1_PREP_ARTIFACT_ROOT,
    V3_1_RANKING_ARTIFACT_ROOT,
    "artifacts/qa_chunk_selection_v4",
)

PROTOCOL = "hypothesis/v3_1_automated_data_preparation_protocol.md"
SILVER_RULE_VERSION = "v3.1-qa-silver-2"
QUESTION_SET_VERSION = "v3.1-qa-questions-1"
ANNOTATION_VERSION = "v3.1-qa-silver-auto-2"

SILVER_STATUS = (
    "silver_target_evidence",
    "silver_target_mismatch",
    "silver_partial_support",
    "silver_unresolved",
)
CONFIDENCE = ("high", "medium", "low")
PRIMARY_GOLD_RELATIONS = frozenset({"exact_match_gold", "normalized_match_gold", "overlap_match_gold"})

DEFAULT_R_MIN = 0.80
DEFAULT_MAX_EASY_PASS = 0.25
DEFAULT_MAX_INVALID_SPAN = 0.05
DEFAULT_MIN_TRAIN_TM_QUESTIONS = 10
DEFAULT_MIN_EVAL_TARGET_AND_TM = 10


def experiment_contract_v3_1_auto() -> dict[str, object]:
    return {
        "prompt": PROTOCOL,
        "continuation": "hypothesis/v3_1_experiment_continuation_automated_overlay.md",
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "v3_1_prep_artifacts": V3_1_PREP_ARTIFACT_ROOT,
        "v3_1_auto_artifacts": V3_1_AUTO_ARTIFACT_ROOT,
        "v3_1_ranking_artifacts": V3_1_RANKING_ARTIFACT_ROOT,
        "annotator": "frozen extractive QA already used in the pipeline; no generative teacher",
        "forbidden_models": ["generative LLM", "NLI", "semantic encoder", "learned annotator", "learned calibrator"],
        "source_label": "never rewritten",
        "label_status": "silver weak supervision, not verified gold",
        "internal_test": "locked; never sent through silver labeling",
        "v3_run": "read-only; never overwrite artifacts/qa_chunk_selection_v3/",
        "manual_prep": "read-only; never overwrite artifacts/qa_chunk_selection_v3_1_prep/",
        "cv": "raw QA cache is global; thresholds and silver labels are fold-specific TRAIN/CAL",
        "silver_status": list(SILVER_STATUS),
        "silver_rule_version": SILVER_RULE_VERSION,
        "question_set_version": QUESTION_SET_VERSION,
        "unlocked_v2_splits": list(UNLOCKED_V2_SPLITS),
        "locked_splits": list(LOCKED_SPLITS),
        "posthoc_splits": list(POSTHOC_SPLITS),
        "reused_modules": [
            "experiments.qa_chunk_selection.progress",
            "experiments.qa_chunk_selection.models",
            "experiments.qa_chunk_selection.metrics",
            "experiments.qa_chunk_selection.v2.qa",
            "experiments.qa_chunk_selection.v2.cache",
            "experiments.qa_chunk_selection.v2.schemas",
            "experiments.qa_chunk_selection.v2.hn_flags",
            "experiments.qa_chunk_selection.v3.data",
            "experiments.qa_chunk_selection.v3.split",
            "experiments.qa_chunk_selection.v3.freeze",
            "experiments.qa_chunk_selection.v3_1_prep.context",
        ],
    }
