from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.phase1_multiquestion.assign import (
    ChunkState,
    QuestionPrediction,
    assign_instance,
)
from experiments.phase1_multiquestion.bm25_multi import multi_question_bm25, ranks_descending
from experiments.phase1_multiquestion.config import MultiQuestionConfig
from experiments.phase1_multiquestion.consistency import consistency_class, qa_consistency
from experiments.phase1_multiquestion.questions import ASPECT_QUESTION_VARIANTS, questions_for
from untie.aspect_qa.config import ASPECT_QUESTIONS


def _cfg(**kwargs) -> MultiQuestionConfig:
    values = dict(
        input_dir=Path("datasets/scirex_aspect_qa_preprocessing"),
        output_dir=Path("artifacts/phase1_multiquestion/test"),
        mode="Q3",
        n_questions=3,
        hard_negative_top_n=3,
        min_cluster_points=3,
        seed=0,
    )
    values.update(kwargs)
    return MultiQuestionConfig(**values)


def _pred(qid: int, question: str, *, correct: bool, answer: str = "BERT") -> QuestionPrediction:
    return QuestionPrediction(
        question_id=qid,
        question=question,
        predicted_answer=answer if correct else "CNN",
        matched_gold_answer="BERT" if correct else None,
        qa_correct=correct,
        qa_confidence=0.2 if correct else 0.01,
        answer_quality=1.0 if correct else 0.0,
        match_type="exact" if correct else "none",
        extra_tokens=0 if correct else None,
        start=0,
        end=4,
        start_logit=1.0,
        end_logit=1.0,
    )


def _chunk(chunk_id: int, text: str, preds: list[QuestionPrediction], **kwargs) -> ChunkState:
    payload = dict(
        chunk_id=chunk_id,
        text=text,
        gold_supporting=False,
        predictions=preds,
        n_correct=0,
        n_questions=len(preds),
        qa_consistency=0.0,
        consistency_class="robust_negative",
        any_correct=False,
        char_start=chunk_id * 10,
        char_end=chunk_id * 10 + 5,
        token_start=chunk_id,
        token_end=chunk_id + 1,
        section_type="section",
        section_title="Intro",
    )
    payload.update(kwargs)
    return ChunkState(**payload)


def test_canonical_question_is_baseline() -> None:
    for aspect, canonical in ASPECT_QUESTIONS.items():
        assert questions_for(aspect, 1) == (canonical,)
        assert ASPECT_QUESTION_VARIANTS[aspect][0] == canonical
        q3 = questions_for(aspect, 3)
        q5 = questions_for(aspect, 5)
        assert q3 == q5[:3]
        assert len(set(q5)) == 5


def test_qa_consistency_formula_and_exclusive_labels() -> None:
    assert qa_consistency(2, 5) == 0.4
    assert consistency_class(0.0, threshold=0.8) == "robust_negative"
    assert consistency_class(0.4, threshold=0.8) == "question_sensitive"
    assert consistency_class(0.8, threshold=0.8) == "robust_positive"
    assert consistency_class(1.0, threshold=0.8) == "robust_positive"
    # any-correct is not robust_positive when threshold is high
    assert consistency_class(1 / 3, threshold=1.0) == "question_sensitive"


def test_any_correct_is_not_the_positive_rule() -> None:
    questions = questions_for("method", 3)
    embeddings = {i: np.eye(4)[i] for i in range(4)}
    chunks = [
        _chunk(0, "Uses BERT.", [_pred(0, questions[0], correct=True), _pred(1, questions[1], correct=False), _pred(2, questions[2], correct=False)]),
        _chunk(1, "Unrelated A.", [_pred(q, questions[q], correct=False, answer="LSTM") for q in range(3)]),
        _chunk(2, "Unrelated B.", [_pred(q, questions[q], correct=False, answer="RNN") for q in range(3)]),
        _chunk(3, "Unrelated C.", [_pred(q, questions[q], correct=False, answer="GRU") for q in range(3)]),
    ]
    result = assign_instance(
        document_id="d1",
        aspect="method",
        split="train",
        questions=questions,
        status="ready",
        extractable_gold_answers=["BERT"],
        unextractable_gold_answers=[],
        chunks=chunks,
        embeddings_by_id=embeddings,
        config=_cfg(),
        threshold=1.0,
    )
    assert chunks[0].any_correct
    assert chunks[0].qa_consistency == 1 / 3
    assert 0 not in result.robust_positive_ids
    assert 0 in result.question_sensitive_ids
    assert result.instance_status == "no_robust_positive"
    assert result.hard_negative_ids == []


def test_gold_supporting_zero_consistency_is_not_hard_negative() -> None:
    questions = questions_for("metric", 3)
    embeddings = {i: np.eye(4)[i] for i in range(4)}
    pos_preds = [_pred(q, questions[q], correct=True, answer="Accuracy") for q in range(3)]
    fail_preds = [_pred(q, questions[q], correct=False, answer="F1") for q in range(3)]
    chunks = [
        _chunk(0, "Accuracy is reported.", pos_preds),
        _chunk(1, "Accuracy appears here too.", fail_preds, gold_supporting=True),
        _chunk(2, "Unrelated.", fail_preds),
        _chunk(3, "Also unrelated.", fail_preds),
    ]
    result = assign_instance(
        document_id="d2",
        aspect="metric",
        split="dev",
        questions=questions,
        status="ready",
        extractable_gold_answers=["Accuracy"],
        unextractable_gold_answers=[],
        chunks=chunks,
        embeddings_by_id=embeddings,
        config=_cfg(),
        threshold=1.0,
    )
    assert 0 in result.robust_positive_ids
    assert 1 in result.gold_failed_ids
    assert 1 not in result.hard_negative_ids
    assert all(not chunk.gold_supporting for chunk in result.chunks if chunk.primary_class == "hard_negative")
    assert all(chunk.qa_consistency == 0 for chunk in result.chunks if chunk.primary_class == "hard_negative")


def test_bm25_aggregation_keeps_per_question_ranks() -> None:
    payload = multi_question_bm25(
        ["What dataset is used in the paper?", "Which dataset does the paper use?"],
        ["The dataset is ImageNet.", "Unrelated table of contents.", "ImageNet classification dataset split."],
    )
    assert len(payload["bm25_by_question"]) == 2
    assert len(payload["rank_by_question"][0]) == 3
    assert payload["rank_by_question"][0][0] == 1 or payload["mean_normalized"][0] >= payload["mean_normalized"][1]
    ranks = ranks_descending([0.2, 0.9, 0.9])
    assert ranks[1] == 1
    assert ranks[2] == 2


def test_questions_contain_no_document_entities() -> None:
    banned = ("cifar", "imagenet", "bert", "cityscapes", "frrn", "bleu")
    for aspect, variants in ASPECT_QUESTION_VARIANTS.items():
        blob = " ".join(variants).casefold()
        assert aspect in {"method", "task", "metric", "material"}
        for token in banned:
            assert token not in blob


def test_official_dev_paths_do_not_overwrite_legacy_hash_split() -> None:
    from experiments.phase1_multiquestion import official_dev as official

    assert official.DEV_Q5.name == "official_dev_q5"
    assert official.DEV_Q3.name == "official_dev_q3"
    assert official.THRESHOLDS.name == "official_frozen_thresholds.json"
    assert official.CACHE.name == "official_dev_qa_prediction_cache.jsonl"
    assert official.PREPROCESS.name == "scirex_aspect_qa_preprocessing_dev"
    assert official.DEV_Q5 != official.LEGACY_ROOT / "dev_q5"
    assert official.THRESHOLDS != official.LEGACY_ROOT / "frozen_thresholds.json"
    official._assert_isolated()


def test_load_document_id_list_skips_comments(tmp_path: Path) -> None:
    from experiments.phase1_scirex.tables import load_document_id_list

    path = tmp_path / "ids.txt"
    path.write_text("# header\n\nabc\nabc\ndef\n", encoding="utf-8")
    assert load_document_id_list(path) == ("abc", "def")


def test_official_train_remainder_is_disjoint_and_isolated() -> None:
    from experiments.phase1_multiquestion import official_train_remainder as remainder

    ids = remainder.remaining_official_train_document_ids()
    labeled = set(
        __import__("pandas")
        .read_parquet(remainder.EXISTING_TRAIN_Q5 / "qa_instances.parquet")["document_id"]
        .astype(str)
    )
    assert ids
    assert not set(ids) & labeled
    assert remainder.Q5 != remainder.EXISTING_TRAIN_Q5
    assert remainder.PHASE1_OUT != remainder.PROJECT_ROOT / "artifacts" / "phase1_scirex" / "train"
    assert remainder.THRESHOLDS.name == "official_frozen_thresholds.json"
    assert remainder.CACHE.name == "official_train_remainder_qa_prediction_cache.jsonl"
    remainder._assert_isolated()
    thresholds = remainder._load_thresholds()
    assert thresholds["Q5"]["chosen_threshold"] == 0.6
    assert abs(thresholds["Q3"]["chosen_threshold"] - 2 / 3) < 1e-12


def test_merge_labeled_runs_rejects_overlap(tmp_path: Path) -> None:
    from experiments.phase1_multiquestion.official_train_remainder import merge_labeled_runs

    def _write(dir_path: Path, doc_id: str, threshold: float) -> None:
        dir_path.mkdir(parents=True)
        instances = __import__("pandas").DataFrame(
            {
                "document_id": [doc_id],
                "aspect": ["task"],
                "qa_instance_id": [f"{doc_id}::task"],
                "robust_positive_chunk_ids": [[1]],
                "hard_negative_chunk_ids": [[2]],
                "question_sensitive_chunk_ids": [[]],
                "instance_status": ["ok"],
            }
        )
        chunks = __import__("pandas").DataFrame(
            {
                "qa_instance_id": [f"{doc_id}::task", f"{doc_id}::task"],
                "chunk_id": [1, 2],
                "primary_class": ["robust_positive", "hard_negative"],
                "any_correct": [True, False],
                "n_correct": [5, 0],
                "n_questions": [5, 5],
                "qa_consistency": [1.0, 0.0],
                "consistency_class": ["robust_positive", "robust_negative"],
                "aspect": ["task", "task"],
            }
        )
        instances.to_parquet(dir_path / "qa_instances.parquet", index=False)
        chunks.to_parquet(dir_path / "qa_chunk_dataset.parquet", index=False)
        chunks.iloc[0:0].to_parquet(dir_path / "qa_question_predictions.parquet", index=False)
        for name in (
            "robust_positive_chunks",
            "question_sensitive_chunks",
            "gold_supporting_robust_negative",
            "false_negative_candidates",
            "hard_negative_chunks",
            "easy_negative_chunks",
        ):
            chunks.iloc[0:0].to_parquet(dir_path / f"{name}.parquet", index=False)
        (dir_path / "experiment_config.json").write_text(
            __import__("json").dumps({"robust_positive_threshold": threshold}) + "\n",
            encoding="utf-8",
        )

    left = tmp_path / "left"
    right = tmp_path / "right"
    _write(left, "doc-a", 0.6)
    _write(right, "doc-a", 0.6)
    try:
        merge_labeled_runs(left, right, tmp_path / "out")
        raise AssertionError("expected overlap to fail")
    except RuntimeError as exc:
        assert "overlap" in str(exc)
