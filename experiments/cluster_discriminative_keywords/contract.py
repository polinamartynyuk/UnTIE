"""Architecture map: reuse existing UnTIE / hyp2 / hyp3 pieces. Do not duplicate."""

from __future__ import annotations

COMPONENT_MAP = {
    "qa": "untie.models.ExtractiveQuestionAnswerer via cached EvidenceStore (no re-decode)",
    "chunking": "frozen CachedChunkAnswer rows from hyp2 evidence",
    "baseline_q0": "untie.qa.AnswerAggregator stored as CachedDocumentEvidence.baseline_answer",
    "answer_consensus": "untie.keyword_evidence.select_cached_answer (AnswerConsensus analog)",
    "weighted_keyword": "untie.ranking.WeightedKeyword",
    "chunk_score": "untie.ranking.score_chunks (position/frequency/uniqueness modifiers unchanged)",
    "attention": "CachedDocumentEvidence.candidates.attention_weight from QA attention mining",
    "diff": "candidates.score_difference = cosine(aspect) - cosine(gold); semantics not rewritten",
    "global_keywords": "experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json field-id 1",
    "clustering": "frozen C_keybert E2+R2 ng1-2_top20_mmr0.50 | KMeans k=4 seed 42",
    "routing": "artifacts/keybert_topic_clustering/en_full/C_keybert_routing.json",
    "topic_phrases": "C_keybert_document_keyphrases.csv + chosen_configs representative_terms",
    "gold_chunk": "argmax char_f1(chunk.answer, references); positives also substring overlap",
    "keybert_role": "question-conditioned candidate SOURCE only; never replaces attention",
    "sentence_encoder_topic": "sentence-transformers/all-MiniLM-L6-v2",
}


def experiment_contract() -> dict[str, object]:
    return {
        "prompt": "hypothesis/hyp4_prompt4.md",
        "production_pipeline": "unchanged; this package is opt-in and isolated",
        "clustering": "frozen C_keybert; not re-tuned against TEST",
        "qa_reader": "RoBERTa extractive QA already cached; this run does not call it",
        "keybert_is_not": "replacement for attention keywords or a topic keyword auto-promoted to QA weight",
        "diff_is_not": "a topic keyword; negative diff is not treated as topic signal",
        "component_map": COMPONENT_MAP,
    }
