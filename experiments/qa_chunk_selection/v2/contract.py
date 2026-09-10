"""V2 contract: reuse E0 pieces, isolate artifacts, never touch production."""

from __future__ import annotations

from experiments.qa_chunk_selection.contract import COMPONENT_MAP, LABEL_MAP

E0_ARTIFACT_ROOT = "artifacts/qa_chunk_selection"
V2_ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v2"

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
)

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
    }
)

V2_SPLITS = (
    "train",
    "calibration",
    "development",
    "internal_test",
    "seen_exploratory_dev",
)

EVAL_SPLITS_BEFORE_FREEZE = ("train", "calibration", "development", "seen_exploratory_dev")


def experiment_contract_v2() -> dict[str, object]:
    return {
        "prompt": "hypothesis/qa_chunk_selection_experiments_v2.md",
        "e0_prompt": "hypothesis/qa_chunk_selection_experiments_v1.md",
        "e0_artifacts": E0_ARTIFACT_ROOT,
        "v2_artifacts": V2_ARTIFACT_ROOT,
        "production_pipeline": "unchanged; this package is opt-in and isolated",
        "gold_at_inference": "never used for ranking, filtering, top-M, adaptive M, reader-aware, or aggregation",
        "internal_test": "locked until freeze-final + --allow-internal-test",
        "seen_official_dev": "seen_exploratory_dev; not a clean TEST; not used for V2 parameter search",
        "d1_default": False,
        "d3_default": False,
        "qa_span_score_is": "qa_confidence_proxy, not extractability or P(correct)",
        "source_offsets": "chunk.char_start + chunk-local offsets; full document text is not loaded",
        "c_is_not": "trained Multi-passage BERT; raw-logit multi-passage baseline only",
        "component_map": dict(COMPONENT_MAP),
        "label_map": dict(LABEL_MAP),
        "reused_e0_modules": [
            "experiments.qa_chunk_selection.dataset",
            "experiments.qa_chunk_selection.ranking",
            "experiments.qa_chunk_selection.metrics",
            "experiments.qa_chunk_selection.models",
            "experiments.qa_chunk_selection.multipassage",
            "experiments.qa_chunk_selection.answerability",
            "experiments.qa_chunk_selection.progress",
            "experiments.qa_chunk_selection.sanity",
            "experiments.qa_chunk_selection.structure",
        ],
    }
