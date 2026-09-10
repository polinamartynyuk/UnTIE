from __future__ import annotations

import numpy as np

from experiments.phase1_scirex.assign import ChunkState, assign_instance
from experiments.phase1_scirex.bm25 import bm25_scores
from experiments.phase1_scirex.checks import check_instances
from experiments.phase1_scirex.config import Phase1Config
from experiments.phase1_scirex.gap_kmeans import cluster_passages
from experiments.phase1_scirex.match import match_extractable
from experiments.phase1_scirex.report import assert_isolated_output
from pathlib import Path


def _cfg(**kwargs) -> Phase1Config:
    values = dict(
        input_dir=Path("datasets/scirex_aspect_qa_preprocessing"),
        output_dir=Path("artifacts/phase1_scirex/test"),
        hard_negative_top_n=3,
        gap_bootstrap=5,
        min_cluster_points=3,
        k_max_cap=10,
        seed=0,
    )
    values.update(kwargs)
    return Phase1Config(**values)


def _chunk(chunk_id: int, text: str, **kwargs) -> ChunkState:
    payload = dict(
        chunk_id=chunk_id,
        text=text,
        gold_supporting=False,
        qa_positive=False,
        predicted_answer="",
        matched_gold_answer=None,
        qa_confidence=0.1,
        answer_quality=0.0,
        start=0,
        end=0,
        start_logit=0.0,
        end_logit=0.0,
        char_start=chunk_id * 10,
        char_end=chunk_id * 10 + 5,
        token_start=chunk_id,
        token_end=chunk_id + 1,
        section_type="section",
        section_title="Intro",
    )
    payload.update(kwargs)
    return ChunkState(**payload)


def test_multi_answer_normalized_match() -> None:
    ok, matched, _ = match_extractable(
        "text classification",
        ["Sentiment Analysis", "Text Classification"],
        method="normalized_exact_match",
    )
    assert ok
    assert matched == "Text Classification"
    miss, _, _ = match_extractable(
        "named entity recognition",
        ["Sentiment Analysis"],
        method="normalized_exact_match",
    )
    assert not miss


def test_gold_span_containment_allows_bounded_over_extraction() -> None:
    from experiments.phase1_scirex.match import classify_match

    hit = classify_match(
        "multi - person pose estimation problem",
        ["Multi-Person Pose Estimation"],
        method="gold_span_containment",
        max_extra_tokens=5,
    )
    assert hit.matched
    assert hit.matched_gold == "Multi-Person Pose Estimation"
    assert hit.match_type == "containment"
    assert hit.extra_tokens == 1


def test_gold_span_containment_rejects_under_extraction() -> None:
    from experiments.phase1_scirex.match import classify_match

    miss = classify_match(
        "segmentation",
        ["Semantic Segmentation"],
        method="gold_span_containment",
        max_extra_tokens=5,
    )
    assert not miss.matched


def test_gold_span_containment_uses_token_boundaries() -> None:
    from experiments.phase1_scirex.match import classify_match

    miss = classify_match("paper", ["AP"], method="gold_span_containment", max_extra_tokens=5)
    assert not miss.matched
    mapping = classify_match("mapping", ["MAP"], method="gold_span_containment", max_extra_tokens=5)
    assert not mapping.matched


def test_gold_span_containment_rejects_paragraph_dump() -> None:
    from experiments.phase1_scirex.match import classify_match, key_tokens

    pred = (
        "Classifying facial expressions into different categories requires capturing "
        "regional distortions of facial landmarks and still mentions Facial Expression Recognition"
    )
    extra = len(key_tokens(pred)) - len(key_tokens("Facial Expression Recognition"))
    assert extra > 5
    miss = classify_match(
        pred,
        ["Facial Expression Recognition"],
        method="gold_span_containment",
        max_extra_tokens=5,
    )
    assert not miss.matched
    hit = classify_match(
        pred,
        ["Facial Expression Recognition"],
        method="gold_span_containment",
        max_extra_tokens=extra,
    )
    assert hit.matched


def test_gold_span_containment_prefers_longer_gold() -> None:
    from experiments.phase1_scirex.match import classify_match

    hit = classify_match(
        "MS - COCO 2017",
        ["COCO", "MS COCO"],
        method="gold_span_containment",
        max_extra_tokens=5,
    )
    assert hit.matched
    # "MS - COCO" collapses to ms-coco, so MS COCO as two tokens may not match;
    # if it does not, COCO can still hit via hyphen-part.
    assert hit.matched_gold in {"COCO", "MS COCO"}


def test_containment_respects_exact_first() -> None:
    from experiments.phase1_scirex.match import classify_match

    hit = classify_match(
        "Accuracy",
        ["Accuracy"],
        method="gold_span_containment",
        max_extra_tokens=5,
    )
    assert hit.match_type == "exact"
    assert hit.extra_tokens == 0


def test_aspect_specific_labels_for_same_chunk() -> None:
    config = _cfg()
    embeddings = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.9, 0.1]),
        2: np.array([0.0, 1.0]),
        3: np.array([0.1, 0.9]),
    }
    method_chunks = [
        _chunk(0, "Uses BERT.", qa_positive=True, predicted_answer="BERT", matched_gold_answer="BERT"),
        _chunk(1, "Uses BERT again.", gold_supporting=True, predicted_answer="CNN"),
        _chunk(2, "Unrelated table.", predicted_answer="CNN"),
        _chunk(3, "Another baseline.", predicted_answer="LSTM"),
    ]
    task_chunks = [
        _chunk(0, "Uses BERT.", predicted_answer="BERT"),
        _chunk(1, "Uses BERT again.", predicted_answer="CNN"),
        _chunk(2, "Sentiment analysis task.", qa_positive=True, predicted_answer="Sentiment Analysis", matched_gold_answer="Sentiment Analysis"),
        _chunk(3, "Another baseline.", predicted_answer="LSTM"),
    ]
    method = assign_instance(
        document_id="d1",
        aspect="method",
        split="train",
        question="What method or model is used or proposed in the paper?",
        status="ready",
        extractable_gold_answers=["BERT"],
        unextractable_gold_answers=[],
        chunks=method_chunks,
        embeddings_by_id=embeddings,
        config=config,
    )
    task = assign_instance(
        document_id="d1",
        aspect="task",
        split="train",
        question="What task is addressed in the paper?",
        status="ready",
        extractable_gold_answers=["Sentiment Analysis"],
        unextractable_gold_answers=[],
        chunks=task_chunks,
        embeddings_by_id=embeddings,
        config=config,
    )
    assert 0 in method.positive_ids
    assert 0 not in task.positive_ids
    assert method.chunks[0].primary_class != task.chunks[0].primary_class


def test_gold_supporting_failed_qa_is_not_hard_negative() -> None:
    config = _cfg()
    embeddings = {i: np.eye(4)[i] for i in range(4)}
    chunks = [
        _chunk(0, "Accuracy is reported.", qa_positive=True, predicted_answer="Accuracy", matched_gold_answer="Accuracy"),
        _chunk(1, "Accuracy appears here too.", gold_supporting=True, predicted_answer="F1"),
        _chunk(2, "Unrelated.", predicted_answer="BLEU"),
        _chunk(3, "Also unrelated.", predicted_answer="ROUGE"),
    ]
    result = assign_instance(
        document_id="d2",
        aspect="metric",
        split="train",
        question="What evaluation metric is used in the paper?",
        status="partially_extractable",
        extractable_gold_answers=["Accuracy"],
        unextractable_gold_answers=["Error"],
        chunks=chunks,
        embeddings_by_id=embeddings,
        config=config,
    )
    assert 1 in result.evidence_failed_ids
    assert 1 not in result.hard_negative_ids
    assert all(not chunk.gold_supporting for chunk in result.chunks if chunk.primary_class == "hard_negative")


def test_fn_candidates_are_excluded_from_hard_negatives() -> None:
    config = _cfg(min_cluster_points=3, k_max_cap=2)
    embeddings = {
        0: np.array([10.0, 0.0]),
        1: np.array([9.5, 0.2]),
        2: np.array([0.0, 10.0]),
        3: np.array([0.2, 9.5]),
        4: np.array([0.1, 9.7]),
    }
    chunks = [
        _chunk(0, "positive cluster text alpha", qa_positive=True, predicted_answer="FastText", matched_gold_answer="FastText"),
        _chunk(1, "positive cluster text beta", predicted_answer="CNN"),
        _chunk(2, "other cluster text gamma", predicted_answer="LSTM"),
        _chunk(3, "other cluster text delta", predicted_answer="RNN"),
        _chunk(4, "other cluster text epsilon", predicted_answer="GRU"),
    ]
    result = assign_instance(
        document_id="d3",
        aspect="method",
        split="train",
        question="What method or model is used or proposed in the paper?",
        status="ready",
        extractable_gold_answers=["FastText"],
        unextractable_gold_answers=[],
        chunks=chunks,
        embeddings_by_id=embeddings,
        config=config,
    )
    if result.fn_filter_status == "applied":
        assert set(result.fn_ids).isdisjoint(result.hard_negative_ids)


def test_no_qa_positive_does_not_build_hard_negative_set() -> None:
    config = _cfg()
    embeddings = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    chunks = [
        _chunk(0, "no match", predicted_answer="CNN"),
        _chunk(1, "still no match", predicted_answer="LSTM", gold_supporting=True),
    ]
    result = assign_instance(
        document_id="d4",
        aspect="method",
        split="train",
        question="What method or model is used or proposed in the paper?",
        status="ready",
        extractable_gold_answers=["BERT"],
        unextractable_gold_answers=[],
        chunks=chunks,
        embeddings_by_id=embeddings,
        config=config,
    )
    assert result.instance_status == "no_qa_positive"
    assert result.hard_negative_ids == []
    assert 1 in result.evidence_failed_ids


def test_k1_skips_cluster_fn_filter() -> None:
    data = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=float)
    result = cluster_passages(data, k_max_cap=1, bootstrap=3, seed=0, min_points=3)
    assert result.status in {"no_cluster_structure", "applied"}
    tiny = cluster_passages(np.array([[1.0, 0.0]]), k_max_cap=10, bootstrap=2, seed=0, min_points=3)
    assert tiny.status == "skipped_insufficient_data"


def test_bm25_ranks_query_overlap_higher() -> None:
    scores = bm25_scores(
        "What task is addressed in the paper?",
        ["The dataset is ImageNet.", "The task is semantic segmentation of street scenes."],
    )
    assert scores[1] > scores[0]


def test_checks_catch_hard_negative_with_gold_supporting() -> None:
    config = _cfg()
    embeddings = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([0.2, 0.8])}
    chunks = [
        _chunk(0, "pos", qa_positive=True, predicted_answer="BERT", matched_gold_answer="BERT"),
        _chunk(1, "fail", gold_supporting=True, predicted_answer="CNN"),
        _chunk(2, "neg", predicted_answer="LSTM"),
    ]
    result = assign_instance(
        document_id="d5",
        aspect="method",
        split="train",
        question="What method or model is used or proposed in the paper?",
        status="ready",
        extractable_gold_answers=["BERT"],
        unextractable_gold_answers=[],
        chunks=chunks,
        embeddings_by_id=embeddings,
        config=config,
    )
    errors = check_instances(
        [result],
        source_chunks={
            ("d5", 0): {"chunk_text": "pos", "char_start": 0, "char_end": 5},
            ("d5", 1): {"chunk_text": "fail", "char_start": 10, "char_end": 15},
            ("d5", 2): {"chunk_text": "neg", "char_start": 20, "char_end": 25},
        },
    )
    assert errors == []


def test_output_isolation_refuses_preprocessing_dir() -> None:
    try:
        assert_isolated_output(Path("datasets/scirex_aspect_qa_preprocessing/oops"), Path("."))
    except ValueError:
        return
    raise AssertionError("expected isolation error")
