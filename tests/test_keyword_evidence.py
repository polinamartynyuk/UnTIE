from __future__ import annotations

import json

import numpy as np

from untie.config import ModelProfile, PipelineConfig
from untie.domain import Sentence, TextChunk
from untie.keyword_evidence import (
    CachedChunkAnswer,
    CachedDocumentEvidence,
    CandidateObservation,
    EvidenceStore,
    ExtractionMetricCache,
    collect_document_evidence,
    rerank_cached_document,
    stable_fingerprint,
    _candidates_without_gold_contrast,
    _contrast_against_references,
)
from untie.ranking import WeightedKeyword
from untie.pipelines import StaticKeywordRerankingPipeline


class FakeEncoder:
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        values = []
        for text in texts:
            normalized = str(text).lower()
            values.append(
                [
                    float("task" in normalized or "задач" in normalized),
                    float("semantic" in normalized),
                    float("baseline" in normalized),
                    max(1.0, float(len(normalized.split()))),
                ]
            )
        return np.asarray(values)


class FakeProcessor:
    def process(self, text: str) -> list[TextChunk]:
        parts = [part.strip() for part in text.split("|")]
        return [
            TextChunk(
                sentences=(
                    Sentence(part, index, tuple(part.split())),
                ),
                text=part,
                token_count=len(part.split()),
            )
            for index, part in enumerate(parts)
        ]


class FakeAnswerer:
    def __call__(self, *, question: str, context: str):
        answer = "semantic task" if "semantic" in context else "baseline"
        return {"answer": answer, "score": 0.9 if "semantic" in context else 0.4}


def test_evidence_store_roundtrip_and_fingerprint_rejection(tmp_path) -> None:
    evidence = _cached_document()
    store = EvidenceStore(tmp_path)
    path = store.save(evidence)

    assert path.is_file()
    loaded = store.load("doc/1", expected_fingerprint=evidence.fingerprint)
    assert loaded == evidence
    assert store.load("doc/1", expected_fingerprint="different") is None

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["doc_id"] == "doc/1"


def test_stable_fingerprint_ignores_mapping_order() -> None:
    assert stable_fingerprint({"a": 1, "b": 2}) == stable_fingerprint(
        {"b": 2, "a": 1}
    )


def test_extraction_metric_cache_persists_lightweight_scores(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    cache = ExtractionMetricCache(path)
    first = cache.score(
        "semantic task",
        ["semantic task"],
        language="en",
        include_bertscore=False,
    )
    cache.save()

    loaded = ExtractionMetricCache(path)
    second = loaded.score(
        "semantic task",
        ["semantic task"],
        language="en",
        include_bertscore=False,
    )
    assert second == first
    assert second["char_f1"] == 1.0


def test_cached_reranking_falls_back_when_no_keyword_matches() -> None:
    evidence = _cached_document()
    answer, diagnostics = rerank_cached_document(
        evidence,
        [WeightedKeyword("absent", "absent", "absent")],
    )
    assert answer == "baseline"
    assert diagnostics["fallback"] is True


def test_cached_reranking_selects_matching_answer() -> None:
    evidence = _cached_document()
    answer, diagnostics = rerank_cached_document(
        evidence,
        [WeightedKeyword("semantic", "semantic", "semantic", 1.0, 1.0)],
        cluster_strategy="highest_avg_score",
        answer_strategy="highest_chunk_score",
    )
    assert answer == "semantic task"
    assert diagnostics["matched_keywords"] == ["semantic"]


def test_cached_evaluator_matches_reference_free_pipeline() -> None:
    evidence = _cached_document()
    keyword = WeightedKeyword(
        "semantic", "semantic", "semantic", 1.0, 1.0
    )
    cached_answer, _ = rerank_cached_document(
        evidence,
        [keyword],
        cluster_strategy="highest_avg_score",
        answer_strategy="highest_chunk_score",
    )
    config = PipelineConfig(
        profile=ModelProfile.english(),
        chunk_max_tokens=20,
        overlap_tokens=2,
    )
    runtime = StaticKeywordRerankingPipeline(
        FakeProcessor(),  # type: ignore[arg-type]
        FakeAnswerer(),
        FakeEncoder(),
        config,
        [keyword],
    ).run(
        "unrelated baseline | semantic method solves task",
        "Which task?",
        cluster_strategy="highest_avg_score",
        answer_strategy="highest_chunk_score",
    )
    assert runtime.final_answer is not None
    assert runtime.final_answer.text == cached_answer


def test_collect_document_evidence_uses_all_references_and_caches_answers() -> None:
    evidence = collect_document_evidence(
        doc_id="train-1",
        text="semantic task solution | unrelated baseline",
        aspect_id="task",
        aspect_reference="task",
        question="Which task?",
        references=["semantic task", "task solution"],
        processor=FakeProcessor(),
        answerer=FakeAnswerer(),
        encoder=FakeEncoder(),
        attention_extractor=lambda question, context: [
            {"word": "semantic", "weight": 0.8},
            {"word": "solution", "weight": 0.4},
        ],
        language="en",
        strict_answer_threshold=0.0,
        min_answer_threshold=0.0,
    )

    assert evidence.references == ("semantic task", "task solution")
    assert len(evidence.chunks) == 2
    assert evidence.baseline_answer
    assert len(evidence.answer_similarity) == 2


def _collect_kwargs(**overrides):
    payload = {
        "doc_id": "train-1",
        "text": "semantic task solution | unrelated baseline",
        "aspect_id": "task",
        "aspect_reference": "task",
        "question": "Which task?",
        "references": ["semantic task"],
        "processor": FakeProcessor(),
        "answerer": FakeAnswerer(),
        "encoder": FakeEncoder(),
        "attention_extractor": lambda question, context: [
            {"word": "semantic", "weight": 0.9},
            {"word": "solution", "weight": 0.4},
        ],
        "language": "en",
        "strict_answer_threshold": 0.0,
        "min_answer_threshold": 0.0,
        "keyword_idf_threshold": 0.0,
    }
    payload.update(overrides)
    return payload


def test_gold_contrast_drops_answer_like_terms_by_default() -> None:
    evidence = collect_document_evidence(**_collect_kwargs())
    words = {item.word for item in evidence.candidates}
    assert "semantic" not in words
    assert evidence.fingerprint == collect_document_evidence(
        **_collect_kwargs(contrast_against_gold=True)
    ).fingerprint


def test_disabling_gold_contrast_keeps_answer_like_attention_terms() -> None:
    with_contrast = collect_document_evidence(**_collect_kwargs())
    without_contrast = collect_document_evidence(
        **_collect_kwargs(contrast_against_gold=False)
    )
    words = {item.word for item in without_contrast.candidates}
    assert "semantic" in words
    assert without_contrast.fingerprint != with_contrast.fingerprint
    assert all(item.score_difference == 0.0 for item in without_contrast.candidates)


def test_contrast_helpers_drop_or_keep_gold_like_terms() -> None:
    candidates = [{"word": "semantic", "weight": 0.9}]
    dropped = _contrast_against_references(
        candidates,
        aspect_reference="task",
        references=["semantic task"],
        encoder=FakeEncoder(),
    )
    kept = _candidates_without_gold_contrast(candidates)
    assert dropped == []
    assert kept == [{"word": "semantic", "weight": 0.9, "score_diff": 0.0}]


def test_default_gold_contrast_fingerprint_matches_legacy_payload() -> None:
    from untie.keyword_evidence import evidence_fingerprint_payload

    assert evidence_fingerprint_payload({"pipeline_version": "v1"}) == {
        "pipeline_version": "v1"
    }
    assert evidence_fingerprint_payload(
        {"pipeline_version": "v1"}, contrast_against_gold=False
    ) == {"pipeline_version": "v1", "contrast_against_gold": False}


def _cached_document() -> CachedDocumentEvidence:
    return CachedDocumentEvidence(
        doc_id="doc/1",
        aspect_id="task",
        question="Which task?",
        references=("semantic task",),
        chunks=(
            CachedChunkAnswer(0, "unrelated baseline", 2, "baseline", 0.4),
            CachedChunkAnswer(
                1, "semantic method solves task", 4, "semantic task", 0.9
            ),
        ),
        answer_similarity=((1.0, 0.1), (0.1, 1.0)),
        baseline_answer="baseline",
        baseline_confidence=0.4,
        candidates=(
            CandidateObservation(
                "semantic", "semantic", "semantic", 0.8, 0.5, (1,)
            ),
        ),
        fingerprint="fingerprint",
    )
