from __future__ import annotations

import pytest

from experiments.qa_chunk_selection.candidates import controlled_chunks, freeze_candidate_sets
from experiments.qa_chunk_selection.config import ExperimentConfig, parse_stages
from experiments.qa_chunk_selection.contract import experiment_contract
from experiments.qa_chunk_selection.dataset import map_primary_class
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT, assert_isolated_output
from experiments.qa_chunk_selection.metrics import (
    coverage_at_k,
    hit_at_k,
    hn_at_k,
    pairwise_pos_gt_hn,
    reciprocal_rank,
)
from experiments.qa_chunk_selection.multipassage import (
    PassageLogits,
    global_best_span,
    independent_best_span,
    independent_softmax_confidence,
)
from experiments.qa_chunk_selection.run import build_parser
from experiments.qa_chunk_selection.sanity import audit_records
from experiments.qa_chunk_selection.schema import ChunkRecord, InstanceRecord
from experiments.qa_chunk_selection.structure import aggregate_answers, split_sentences


def _chunk(chunk_id: int, text: str, label: str, **kwargs) -> ChunkRecord:
    payload = dict(
        chunk_id=chunk_id,
        text=text,
        label=label,
        primary_class={
            "positive": "robust_positive",
            "hard_negative": "hard_negative",
            "easy_negative": "easy_negative",
            "ambiguous": "question_sensitive",
        }[label],
        position=chunk_id,
        section_id="intro",
        section_title="Introduction",
        char_start=chunk_id * 20,
        char_end=chunk_id * 20 + len(text),
        token_start=chunk_id,
        token_end=chunk_id + 1,
        gold_supporting=label == "positive",
        previous_chunk_id=None if chunk_id == 0 else chunk_id - 1,
        next_chunk_id=chunk_id + 1,
    )
    payload.update(kwargs)
    return ChunkRecord(**payload)


def _instance(*chunks: ChunkRecord, document_id: str = "doc-a", split: str = "train") -> InstanceRecord:
    return InstanceRecord(
        document_id=document_id,
        question_id=f"{document_id}::task",
        aspect="task",
        question="What task is addressed in the paper?",
        gold_answers=("semantic segmentation",),
        split=split,
        chunks=chunks,
    )


def test_contract_keeps_production_pipeline_isolated() -> None:
    contract = experiment_contract()
    assert "unchanged" in str(contract["production_pipeline"])
    assert contract["gold_at_inference"].startswith("never")
    assert "not performed" in str(contract["hard_negative_mining"])


def test_label_map_matches_phase1_q5() -> None:
    assert map_primary_class("robust_positive") == "positive"
    assert map_primary_class("hard_negative") == "hard_negative"
    assert map_primary_class("question_sensitive") == "ambiguous"
    assert map_primary_class("gold_supporting_robust_negative") == "ambiguous"
    assert map_primary_class("false_negative_candidate") == "ambiguous"
    with pytest.raises(ValueError):
        map_primary_class("qa_positive")


def test_isolation_refuses_preprocessing_and_phase1() -> None:
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "phase1_multiquestion" / "official_train_q5")
    allowed = assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection" / "_isolation_test")
    assert allowed.name == "_isolation_test"


def test_retrieval_metrics_use_effective_k() -> None:
    ranked = [2, 1]
    positives = [1]
    hards = [2]
    assert hit_at_k(ranked, positives, 1) == 0.0
    assert hit_at_k(ranked, positives, 5) == 1.0
    assert coverage_at_k(ranked, positives, 5) == 1.0
    assert reciprocal_rank(ranked, positives) == pytest.approx(0.5)
    assert hn_at_k(ranked, hards, 5) == pytest.approx(0.5)
    acc, n_pairs = pairwise_pos_gt_hn({1: 0.2, 2: 0.9}, positives, hards)
    assert n_pairs == 1
    assert acc == 0.0


def test_controlled_candidates_are_deterministic() -> None:
    easy = [_chunk(i, f"easy {i}", "easy_negative") for i in range(3, 12)]
    instance = _instance(
        _chunk(0, "positive evidence semantic segmentation", "positive"),
        _chunk(1, "hard negative related work", "hard_negative"),
        _chunk(2, "another hard", "hard_negative"),
        *easy,
    )
    left = [chunk.chunk_id for chunk in controlled_chunks(instance, n_easy=3, seed=42)]
    right = [chunk.chunk_id for chunk in controlled_chunks(instance, n_easy=3, seed=42)]
    other = [chunk.chunk_id for chunk in controlled_chunks(instance, n_easy=3, seed=7)]
    assert left == right
    assert 0 in left and 1 in left and 2 in left
    assert len(left) == 6
    assert left != other or set(left) != set(other) or left == other


def test_document_level_split_rejects_leakage() -> None:
    from experiments.qa_chunk_selection.candidates import document_split_payload

    train = _instance(_chunk(0, "pos", "positive"), document_id="doc-1", split="train")
    leaked = _instance(_chunk(0, "pos", "positive"), document_id="doc-1", split="dev")
    with pytest.raises(ValueError):
        document_split_payload([train, leaked])


def test_sanity_keeps_frozen_labels_and_flags_gold_in_hn() -> None:
    instance = _instance(
        _chunk(0, "the task is semantic segmentation", "positive"),
        _chunk(1, "semantic segmentation appears in a hard negative", "hard_negative", gold_supporting=False),
    )
    findings = audit_records([instance])
    assert findings["n_blocking_errors"] == 0
    assert findings["n_hn_literal_gold_warnings"] == 1


def test_global_logits_are_not_independent_softmax() -> None:
    hn = PassageLogits(
        chunk_id=1,
        text="x",
        start_logits=(8.0,),
        end_logits=(8.0,),
        offsets=((0, 1),),
        token_is_context=(True,),
    )
    pos = PassageLogits(
        chunk_id=0,
        text="abcdef",
        start_logits=(12.0, 11.0, 11.0, 11.0, 11.0, 11.0),
        end_logits=(12.0, 11.0, 11.0, 11.0, 11.0, 11.0),
        offsets=tuple((i, i + 1) for i in range(6)),
        token_is_context=(True,) * 6,
    )
    global_span = global_best_span([hn, pos])
    assert global_span is not None
    assert global_span.chunk_id == 0
    hn_conf = independent_softmax_confidence(hn)
    pos_conf = independent_softmax_confidence(pos)
    assert hn_conf > pos_conf
    assert independent_best_span(pos).logit_sum > independent_best_span(hn).logit_sum


def test_answer_aggregation_deduplicates_overlapping_chunks() -> None:
    chunks = (
        _chunk(0, "semantic segmentation method", "positive", char_start=0, char_end=40),
        _chunk(1, "semantic segmentation method extra", "positive", char_start=0, char_end=40),
        _chunk(2, "cnn", "easy_negative", char_start=80, char_end=90),
    )
    spans = [
        {"chunk_id": 0, "text": "semantic segmentation", "start_char": 0, "end_char": 21, "score": 1.0},
        {"chunk_id": 1, "text": "semantic segmentation", "start_char": 0, "end_char": 21, "score": 1.0},
        {"chunk_id": 2, "text": "cnn", "start_char": 0, "end_char": 3, "score": 1.0},
    ]
    result = aggregate_answers(spans, chunks)
    assert result["n_unique_regions"] == 1
    assert result["normalized"] == "semantic segmentation"


def test_sentence_split_keeps_offsets() -> None:
    spans = split_sentences("Hello world. Next sentence!")
    assert len(spans) == 2
    assert spans[0][2].startswith("Hello")
    assert spans[1][0] > spans[0][0]


def test_parser_and_stages() -> None:
    parser = build_parser()
    args = parser.parse_args(["--smoke", "--stages", "prepare,a0,a", "--skip-qa"])
    assert args.smoke is True
    assert parse_stages("all")[0] == "prepare"
    assert "d5" in parse_stages("full")
    cfg = ExperimentConfig(smoke=True, stages=("prepare", "a0"))
    assert cfg.candidate_mode == "all"


def test_freeze_candidate_sets_all_keeps_ambiguous() -> None:
    instance = _instance(
        _chunk(0, "pos", "positive"),
        _chunk(1, "hn", "hard_negative"),
        _chunk(2, "easy", "easy_negative"),
        _chunk(3, "maybe", "ambiguous"),
    )
    rows = freeze_candidate_sets([instance], mode="all", n_easy=5, seed=42)
    assert rows[0]["n_ambiguous"] == 1
    assert rows[0]["chunk_ids"] == [0, 1, 2, 3]


def test_tfidf_answerability_learns_pos_vs_hn() -> None:
    from experiments.qa_chunk_selection.answerability import TfidfAnswerabilityScorer, collect_pairs

    train = [
        _instance(
            _chunk(0, "the method is a convolutional neural network", "positive"),
            _chunk(1, "related work discusses previous baselines", "hard_negative"),
            document_id="doc-train",
        ),
        _instance(
            _chunk(0, "we propose a convolutional neural network", "positive"),
            _chunk(1, "appendix lists hyperparameters", "hard_negative"),
            document_id="doc-train-2",
        ),
    ]
    pairs = collect_pairs(train, ratio=(1, 1, 0), seed=0)
    scorer = TfidfAnswerabilityScorer().fit(pairs)
    pos = scorer.predict([("What method is used?", "the method is a convolutional neural network")])[0]
    hn = scorer.predict([("What method is used?", "related work discusses previous baselines")])[0]
    assert pos > hn
