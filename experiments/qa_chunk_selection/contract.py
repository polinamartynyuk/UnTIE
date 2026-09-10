"""Architecture map: reuse existing UnTIE / Phase 1 pieces. Do not duplicate."""

from __future__ import annotations

COMPONENT_MAP = {
    "chunks": "datasets/scirex_aspect_qa_preprocessing{,_dev,_test} parquet (read-only)",
    "labels": "artifacts/phase1_multiquestion/official_{train,dev}_q5 qa_chunk_dataset.parquet",
    "questions": "untie.aspect_qa.config.ASPECT_QUESTIONS (canonical Q0)",
    "qa": "untie.models.ExtractiveQuestionAnswerer (frozen RoBERTa)",
    "dense": "untie.models.ModelFactory.sentence_encoder (english MiniLM profile)",
    "bm25": "experiments.phase1_scirex.bm25.bm25_scores",
    "cross_encoder_generic": "untie.cross_encoder_ranking.SentenceTransformerCrossEncoder",
    "match": "experiments.phase1_scirex.match.classify_match / identity_key",
    "qa_metrics": "untie.extraction_metrics char/token/rouge; identity_key exact match",
    "answer_aggregation": "untie.answer_selection.normalize_answer_key",
    "isolation": "writes only under artifacts/qa_chunk_selection/",
}

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
)

LABEL_MAP = {
    "robust_positive": "positive",
    "hard_negative": "hard_negative",
    "easy_negative": "easy_negative",
    "question_sensitive": "ambiguous",
    "gold_supporting_robust_negative": "ambiguous",
    "false_negative_candidate": "ambiguous",
    "confirmed_false_negative": "ambiguous",
}

AMBIGUOUS_EXCLUDED_FROM_TRAINING = frozenset(LABEL_MAP.keys()) - {
    "robust_positive",
    "hard_negative",
    "easy_negative",
}


def experiment_contract() -> dict[str, object]:
    return {
        "prompt": "hypothesis/qa_chunk_selection_experiments_v1.md",
        "labels_source": "docs/sredstva-otbora-relevantnyh-chankov.md §15 official_train_q5 / official_dev_q5",
        "production_pipeline": "unchanged; this package is opt-in and isolated",
        "mode": "controlled candidate-set (Mode II) plus optional full-pool C_all",
        "test_policy": "official SciREX TEST is refused until --allow-test; labels are not mined here",
        "gold_at_inference": "never used for ranking, filtering, or top-M selection",
        "hard_negative_mining": "not performed; frozen Phase 1 Q5 labels are used as-is",
        "experiment_a": "generic off-the-shelf relevance Cross-Encoder, no HN supervision",
        "experiment_b": "same CE backbone, trained on positive vs verified hard_negative",
        "experiment_c": "top-M from B with globally competing raw start/end logits (not per-passage softmax)",
        "experiment_d": "ablation ladder D1–D5 on the same frozen candidate sets",
        "component_map": COMPONENT_MAP,
        "label_map": dict(LABEL_MAP),
    }
