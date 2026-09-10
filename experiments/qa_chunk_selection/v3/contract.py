"""V3 contract: isolated artifacts, V2/E0 read-only, gold never at ranking."""

from __future__ import annotations

from experiments.qa_chunk_selection.contract import COMPONENT_MAP, LABEL_MAP

E0_ARTIFACT_ROOT = "artifacts/qa_chunk_selection"
V2_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v2"
V3_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v3"

PROTECTED_WRITE_ROOTS = (
    "datasets",
    "untie",
    "artifacts/phase1_multiquestion",
    "artifacts/phase1_scirex",
    "artifacts/hierarchical_topic_optimized",
    "artifacts/keybert_topic_clustering",
    "artifacts/keybert_qa_integration",
    "artifacts/answer_aggregation",
    "artifacts/cluster_discriminative_keywords",
    E0_ARTIFACT_ROOT,
    V2_ARTIFACT_ROOT,
)

UNLOCKED_V2_SPLITS = ("train", "calibration", "development")
POSTHOC_SPLITS = ("seen_exploratory_dev",)
LOCKED_SPLITS = ("internal_test",)

TARGET_STATUS = (
    "target_evidence",
    "target_mismatch",
    "partial_support",
    "unresolved",
)

GOLD_RELATIONS = (
    "matches_existing_gold",
    "semantically_equivalent_to_existing_gold",
    "new_valid_value_candidate",
    "no_gold_match_expected",
    "unclear",
)

PRIMARY_GOLD_RELATIONS = frozenset(
    {"matches_existing_gold", "semantically_equivalent_to_existing_gold"}
)

GSRN_CLASS = "gold_supporting_robust_negative"

FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {
        "gold_answers",
        "gold_spans",
        "label",
        "primary_class",
        "gold_supporting",
        "selected_chunk_label",
        "qa_chunk_label",
        "em",
        "token_f1",
        "exact_match",
        "hn_gold_string_warning",
        "training_eligible_hn",
        "source_label",
        "source_ambiguous_subtype",
        "v3_target_status",
        "v3_target_value",
        "v3_gold_relation",
        "span_correct_target",
        "wrong_target",
        "attribution_aware_TokenF1",
    }
)


def experiment_contract_v3() -> dict[str, object]:
    return {
        "prompt": "hypothesis/qa_chunk_selection_experiments_v3_plan_attention.md",
        "v2_prompt": "hypothesis/qa_chunk_selection_experiments_v2.md",
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "v3_artifacts": V3_ARTIFACT_ROOT,
        "production_pipeline": "unchanged; this package is opt-in and isolated",
        "gold_at_inference": (
            "never used for ranking, shortlist, attention runtime features, "
            "span reranking, or aggregation; gold-span attention is diagnostic_only"
        ),
        "internal_test": "locked until freeze-final + --allow-internal-test + checklist",
        "seen_official_dev": "seen_exploratory_dev; post-hoc sanity only; never V3 model selection",
        "source_label": "never rewritten; V3 target-status is a separate overlay",
        "include_unanswerable": False,
        "d3_default": False,
        "d4_default": "gated",
        "d5_default": "gated",
        "raw_logit_C": "baseline only; not the V3 multi-candidate mechanism",
        "selection_key": "constrained Token F1 vs V3-B0; not pairwise_hn lexicographic",
        "attention_heads": "selected on TRAIN_FOLDS; N/thresholds on CALIBRATION; never EVAL/TEST",
        "component_map": dict(COMPONENT_MAP),
        "label_map": dict(LABEL_MAP),
        "reused_modules": [
            "experiments.qa_chunk_selection.dataset",
            "experiments.qa_chunk_selection.ranking",
            "experiments.qa_chunk_selection.metrics",
            "experiments.qa_chunk_selection.models",
            "experiments.qa_chunk_selection.multipassage",
            "experiments.qa_chunk_selection.answerability",
            "experiments.qa_chunk_selection.progress",
            "experiments.qa_chunk_selection.sanity",
            "experiments.qa_chunk_selection.structure",
            "experiments.qa_chunk_selection.v2.answerability_v2",
            "experiments.qa_chunk_selection.v2.evaluate_v2",
            "experiments.qa_chunk_selection.v2.qa",
            "experiments.qa_chunk_selection.v2.schemas",
            "experiments.qa_chunk_selection.v2.hn_flags",
            "experiments.qa_chunk_selection.v2.cache",
            "experiments.qa_chunk_selection.v2.statistics",
        ],
    }
