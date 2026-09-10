"""Map the hyp3 prompt onto existing UnTIE extractive QA components."""

from __future__ import annotations


COMPONENT_MAP = {
    "qa": "untie.models.ExtractiveQuestionAnswerer via untie.qa.AnswerFinder",
    "qa_score": "start_prob * end_prob on the extracted span",
    "attention": "untie.attention.AttentionKeywordExtractor on the same RoBERTa weights",
    "online_attention": "top-M QA-confidence chunks; no gold filter at test",
    "diff": "frozen global static dictionary score_difference (not per-doc gold contrast)",
    "static_keywords": "experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json",
    "chunk_score": "untie.ranking.score_chunks; mixes via untie.ranking.mix_score_channels",
    "keybert": "thematic prior only; never replaces attention keywords",
    "answer_select": "untie.keyword_evidence.select_cached_answer (select_clustered)",
    "baseline": "AnswerAggregator over all chunk answers",
    "sentence_encoder_qa": "ModelProfile.english().sentence_model",
    "sentence_encoder_keybert": "sentence-transformers/all-MiniLM-L6-v2",
}


def qa_contract() -> dict[str, object]:
    return {
        "prompt_name": "extractive QA on RoBERTa",
        "actual_class": "untie.models.ExtractiveQuestionAnswerer",
        "attention": "untie.attention.AttentionKeywordExtractor",
        "keybert_role": "thematic prior / filter / extra score channel",
        "keybert_is_not": "replacement for attention keywords",
        "not_used_as_qa": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "training_default": "skipped (exp 6–7 data prep only unless --run-training)",
        "component_map": COMPONENT_MAP,
    }
