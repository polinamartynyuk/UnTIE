from pathlib import Path

import numpy as np
import pandas as pd

from experiments.phase1_multiquestion.config import MultiQuestionConfig
from experiments.phase1_multiquestion.engine import PredictionCache
from experiments.qa_chunk_selection.schema import ChunkRecord, InstanceRecord
from experiments.qa_chunk_selection.v4_4_ru_data.config import ASPECT_SPECS
from experiments.qa_chunk_selection.v4_4_ru_q5_b0.data import validate_cv
from experiments.qa_chunk_selection.v4_4_ru_q5_b0.q5 import process_document
from experiments.qa_chunk_selection.v4_4_ru_q5_b0.questions import ASPECTS, QUESTIONS, validate_questions
from experiments.qa_chunk_selection.v3.split import build_cv_splits


def _chunk(cid, label):
    return ChunkRecord(
        chunk_id=cid, text=f"text {cid}", label=label, primary_class=label,
        position=cid, section_id="", section_title="", char_start=cid * 10,
        char_end=cid * 10 + 6, token_start=cid, token_end=cid + 1,
        gold_supporting=label == "positive", previous_chunk_id=None, next_chunk_id=None,
    )


def test_question_inventory_is_frozen_for_all_native_aspects():
    validate_questions()
    canonical = {aspect: question for _, (aspect, question) in ASPECT_SPECS.items()}
    assert set(ASPECTS) == set(canonical)
    assert all(QUESTIONS[aspect][0] == canonical[aspect] for aspect in ASPECTS)
    assert sum(len(value) for value in QUESTIONS.values()) == 45


def test_q5_processes_five_questions_and_uses_consistency(tmp_path):
    instances = pd.DataFrame([{
        "document_id": "d1", "aspect": "task", "split": "train", "status": "ready",
        "extractable_gold_answers": ["ответ"], "unextractable_gold_answers": [],
        "gold_supporting_chunk_ids": [0],
    }])
    chunks = pd.DataFrame([
        {"document_id": "d1", "chunk_id": 0, "chunk_text": "верный ответ", "char_start": 0,
         "char_end": 12, "token_start": 0, "token_end": 2, "eligible_for_qa": True},
        {"document_id": "d1", "chunk_id": 1, "chunk_text": "посторонний текст", "char_start": 13,
         "char_end": 30, "token_start": 2, "token_end": 4, "eligible_for_qa": True},
    ])
    mentions = pd.DataFrame(columns=["document_id", "aspect", "chunk_ids"])
    config = MultiQuestionConfig(input_dir=tmp_path, output_dir=tmp_path / "out", mode="Q5", n_questions=5)
    cache = PredictionCache(tmp_path / "predictions.jsonl")

    def answer(_question, text):
        return {"answer": "ответ" if "верный" in text else "мимо", "score": 1.0,
                "start": 7 if "верный" in text else 0, "end": 12 if "верный" in text else 4}

    def encode(texts):
        return np.asarray([[float(len(text)), 1.0] for text in texts])

    result = process_document("d1", instances, chunks, mentions, config, answer, encode, cache, 0.6)[0]
    assert result.n_questions == 5
    assert result.robust_positive_ids == [0]
    assert len(cache.rows) == 10


def test_answerable_only_grouped_cv_has_complete_disjoint_roles():
    records = tuple(
        InstanceRecord(f"d{i}", f"d{i}::task", "task", QUESTIONS["task"][0], ("answer",),
                       "train", (_chunk(0, "positive"), _chunk(1, "hard_negative")))
        for i in range(10)
    )
    cv = build_cv_splits(records, n_folds=5, seed=42)
    assert validate_cv(records, cv) == []
    for run in cv["runs"]:
        train, cal, ev = (set(run[name]) for name in ("train", "calibration", "eval"))
        assert not (train & cal or train & ev or cal & ev)
        assert train | cal | ev == set(range(5))


def test_b0_oof_reuses_canonical_q5_reader_trace():
    from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import score_eval

    good = ChunkRecord(
        chunk_id=0, text="верный ответ", label="positive", primary_class="robust_positive",
        position=0, section_id="", section_title="", char_start=10, char_end=22,
        token_start=0, token_end=2, gold_supporting=True, previous_chunk_id=None, next_chunk_id=1,
    )
    bad = ChunkRecord(
        chunk_id=1, text="другой текст", label="hard_negative", primary_class="hard_negative",
        position=1, section_id="", section_title="", char_start=23, char_end=35,
        token_start=2, token_end=4, gold_supporting=False, previous_chunk_id=0, next_chunk_id=None,
    )
    record = InstanceRecord("d", "d::task", "task", QUESTIONS["task"][0], ("ответ",),
                            "eval", (good, bad))

    class Scorer:
        def predict(self, pairs):
            return [0.9, 0.1]

    qa = {("d", "task", 0): {"qa_predicted_answer": "ответ", "answer_start": 7,
                                "answer_end": 12, "start_logit": 4.0, "end_logit": 3.0}}
    row = score_eval((record,), Scorer(), run=0, seed=42, flags=[], qa_lookup=qa)[0]
    assert row["qa_answer"] == "ответ"
    assert row["source_start_char"] == 17
    assert row["qa_source"] == "reused_q5_canonical_cache"
    assert row["token_f1"] == 100.0
