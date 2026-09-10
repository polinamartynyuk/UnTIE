"""Map hyp2 'CE' onto the existing extractive QA component.

The prompt uses CE for the project component that already forms candidate
answers. UnTIE does not have a class named CE. The production extractor is:

    untie.models.ExtractiveQuestionAnswerer
    wrapped by untie.qa.AnswerFinder

Input:  question string + chunk text (context).
Output: ``answer`` span, ``score`` (start_prob * end_prob), char offsets.
Attention: same RoBERTa QA weights via ``untie.attention.AttentionKeywordExtractor``
           when ``ModelProfile.attention_supported`` is true (English).

This experiment never substitutes MiniLM cross-encoder ranking for CE.
MiniLM-L6-v2 is used only as the cheap clustering/KeyBERT encoder.
"""

from __future__ import annotations

COMPONENT_MAP = {
    "documents": "datasets/scirex_structured.json (JSONL; doc_id, original_text, tasks)",
    "title_abstract": "SciREx metadata title/abstract/headings are empty; parsed from body",
    "language": "en",
    "split": "untie.keyword_tuning.deterministic_document_split seed=42 (0.6/0.2/0.2)",
    "ce": "untie.models.ExtractiveQuestionAnswerer via untie.qa.AnswerFinder",
    "ce_score": "start_logit_softmax * end_logit_softmax on the extracted span",
    "attention": "untie.attention.AttentionKeywordExtractor on the same QA model",
    "weighted_keyword": "untie.ranking.WeightedKeyword",
    "attention_weight": "median attention over train docs in a cluster (existing pool agg)",
    "score_difference": "untie.keywords.score_keyword_contrast; cosine(aspect)-cosine(gold)",
    "candidate_pool": "untie.keyword_tuning.aggregate_candidate_pool (train only)",
    "static_keywords": "experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json",
    "chunk_score": "untie.ranking.score_chunks (attention*lambda + diff*(1-lambda) * modifiers)",
    "minimum_matches": 1,
    "modifiers": "position_weight=0.3, frequency_weight=0.7, uniqueness_bonus",
    "fallback": "unmatched chunks drop; prediction falls back to AnswerAggregator baseline",
    "answer_consensus": "untie.qa.AnswerConsensus.select_clustered / keyword_evidence.select_cached_answer",
    "sffs": "untie.keyword_training.tune_global_keywords",
    "tuning_objective": "untie.keyword_tuning.evaluate_objective on train/dev evidence",
    "metrics": "untie.extraction_metrics + ExtractionMetricCache",
    "sentence_encoder_qa": "ModelProfile.english().sentence_model (local eng_sentence_transformer)",
    "sentence_encoder_topics": "sentence-transformers/all-MiniLM-L6-v2 (KeyBERT + clustering only)",
    "keybert": "keybert==0.9.0 from requirements/requirements.txt",
    "q0": "AnswerAggregator over all CE/QA chunk answers (production baseline)",
    "q1": "StaticKeywordRerankingPipeline semantics via rerank_cached_document",
}


def ce_contract() -> dict[str, object]:
    return {
        "prompt_name": "CE",
        "actual_class": "untie.models.ExtractiveQuestionAnswerer",
        "wrapper": "untie.qa.AnswerFinder",
        "inputs": ["question", "context_chunk"],
        "outputs": ["answer", "score", "start", "end"],
        "confidence": "span start_prob * end_prob",
        "attention": "untie.attention.AttentionKeywordExtractor on the same QA weights",
        "not_used_as_ce": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "component_map": COMPONENT_MAP,
    }
