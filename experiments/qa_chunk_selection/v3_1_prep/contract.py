"""V3.1 data-prep contract: isolated overlay workspace, V3/V2/E0 read-only."""

from __future__ import annotations

from experiments.qa_chunk_selection.v3.contract import (
    E0_ARTIFACT_ROOT,
    GOLD_RELATIONS,
    LOCKED_SPLITS,
    POSTHOC_SPLITS,
    PRIMARY_GOLD_RELATIONS,
    PROTECTED_WRITE_ROOTS as V3_PROTECTED_WRITE_ROOTS,
    TARGET_STATUS,
    UNLOCKED_V2_SPLITS,
    V2_ARTIFACT_ROOT,
    V3_ARTIFACT_ROOT,
)

V3_1_PREP_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v3_1_prep"

PROTECTED_WRITE_ROOTS = V3_PROTECTED_WRITE_ROOTS + (
    V3_ARTIFACT_ROOT,
    "artifacts/qa_chunk_selection_v4",
)

TAXONOMY_VERSION = "v3-target-status-1"
ANNOTATION_VERSION = "v3.1-overlay-1"
GUIDELINE_VERSION = "v3.1-data-preparation-protocol"

MISMATCH_TYPES = (
    "related_work",
    "external_cited_work",
    "baseline",
    "compared_method",
    "other_dataset",
    "other_benchmark",
    "other_experiment",
    "other_metric_context",
    "background_definition",
    "historical_result",
    "other",
    "unclear_mismatch_type",
)

CONFIDENCE = ("high", "medium", "low")
SUPPORT_SCOPE = ("same_sentence", "neighbor_sentence", "neighbor_chunk", "document_wide", "unclear")
SAMPLING_STRATA = ("hard_case", "representative", "coverage_completion")

FORBIDDEN_ANNOTATOR_FIELDS = (
    "b_score",
    "attention_score",
    "predicted_class",
    "model_rank",
    "is_top1",
    "in_top5",
    "high_score",
    "ranked_ids",
    "scores",
    "final_ranking_score",
    "selected_chunk_ids",
    "qa_span_score",
    "qa_confidence_proxy",
)

EFFECTIVE_ROLES = (
    "target_evidence_original",
    "target_evidence_overlay",
    "target_mismatch_verified",
    "known_other_negative",
    "partial_support",
    "unresolved",
    "unknown_ambiguous",
)

DEFAULT_MIN_TRAIN_QUESTIONS_WITH_TM = 10
DEFAULT_MIN_EVAL_QUESTIONS_WITH_TARGET_AND_TM = 10
MIN_TARGET_STATUS_COVERAGE_AT_1 = 0.95


def experiment_contract_v3_1_prep() -> dict[str, object]:
    return {
        "prompt": "hypothesis/v3_1_data_preparation_protocol.md",
        "v3_results": "hypothesis/qa_chunk_selection_experiments_v3_results.md",
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "v3_1_prep_artifacts": V3_1_PREP_ARTIFACT_ROOT,
        "production_pipeline": "unchanged; this package only prepares a target-status overlay",
        "source_label": "never rewritten; V3 target-status is a separate overlay",
        "internal_test": "locked; never used for annotation, taxonomy, or guideline checks",
        "seen_official_dev": "seen_exploratory_dev is not part of the primary overlay universe",
        "v3_run": "read-only; never overwrite artifacts/qa_chunk_selection_v3/",
        "annotator_must_not_see": list(FORBIDDEN_ANNOTATOR_FIELDS),
        "taxonomy_version": TAXONOMY_VERSION,
        "annotation_version": ANNOTATION_VERSION,
        "guideline_version": GUIDELINE_VERSION,
        "target_status": list(TARGET_STATUS),
        "gold_relations": list(GOLD_RELATIONS),
        "primary_gold_relations": sorted(PRIMARY_GOLD_RELATIONS),
        "mismatch_types": list(MISMATCH_TYPES),
        "unlocked_v2_splits": list(UNLOCKED_V2_SPLITS),
        "locked_splits": list(LOCKED_SPLITS),
        "posthoc_splits": list(POSTHOC_SPLITS),
        "min_target_status_coverage@1": MIN_TARGET_STATUS_COVERAGE_AT_1,
        "min_train_questions_with_TM_default": DEFAULT_MIN_TRAIN_QUESTIONS_WITH_TM,
        "min_eval_questions_with_target_and_TM_default": DEFAULT_MIN_EVAL_QUESTIONS_WITH_TARGET_AND_TM,
        "reused_modules": [
            "experiments.qa_chunk_selection.progress",
            "experiments.qa_chunk_selection.schema",
            "experiments.qa_chunk_selection.structure",
            "experiments.qa_chunk_selection.v3.contract",
            "experiments.qa_chunk_selection.v3.overlay",
            "experiments.qa_chunk_selection.v3.roles",
            "experiments.qa_chunk_selection.v3.data",
            "experiments.qa_chunk_selection.v3.split",
            "experiments.qa_chunk_selection.v3.structure",
            "experiments.qa_chunk_selection.v3.freeze",
        ],
    }
