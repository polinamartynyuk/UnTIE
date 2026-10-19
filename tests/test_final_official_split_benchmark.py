from __future__ import annotations

from pathlib import Path

import pytest

from experiments.final_official_split_benchmark.config import Config
from experiments.final_official_split_benchmark.engine import Reader
from experiments.final_official_split_benchmark.data import (
    _internal_split,
    assert_disjoint,
    load_test_gold,
    load_test_runtime,
)
from experiments.final_official_split_benchmark.io_guard import write_json, write_jsonl
from experiments.final_official_split_benchmark.pipeline import Pipeline
from experiments.qa_chunk_selection.v2.schemas import assert_no_gold_fields, to_runtime
from experiments.qa_chunk_selection.v3.data import synthetic_records


def test_document_grouped_internal_split_is_disjoint():
    records = synthetic_records(n_docs=20)
    train, validation, test = _internal_split(records, test=True)
    assert_disjoint(train, validation, label="TRAIN/VAL")
    assert_disjoint(train, test, label="TRAIN/TEST")
    assert_disjoint(validation, test, label="VAL/TEST")
    assert {x.document_id for x in train} | {x.document_id for x in validation} | {x.document_id for x in test} == {x.document_id for x in records}


def test_runtime_projection_contains_no_gold_or_labels():
    runtime = to_runtime(synthetic_records(n_docs=1)[0]).to_dict()
    assert_no_gold_fields(runtime)
    text = str(runtime).lower()
    assert "gold_answers" not in text
    assert "primary_class" not in text


def test_smoke_cannot_read_test():
    with pytest.raises(RuntimeError, match="smoke mode cannot read TEST"):
        load_test_runtime("scirex", smoke=True)
    with pytest.raises(RuntimeError, match="smoke mode cannot read TEST gold"):
        load_test_gold("scirex", unlocked=True, smoke=True)


def test_gold_requires_unlock():
    with pytest.raises(RuntimeError, match="gold access requires"):
        load_test_gold("scirex", unlocked=False)


def test_immutable_prediction_rejects_change(tmp_path: Path):
    path = tmp_path / "prediction.jsonl"
    write_jsonl(path, [{"runtime": {"question_id": "q1", "answer": "a"}}], immutable=True)
    write_jsonl(path, [{"runtime": {"question_id": "q1", "answer": "a"}}], immutable=True)
    with pytest.raises(FileExistsError, match="immutable artifact"):
        write_jsonl(path, [{"runtime": {"question_id": "q1", "answer": "changed"}}], immutable=True)


def test_config_rejects_test_unlock_in_smoke(tmp_path: Path):
    with pytest.raises(ValueError, match="smoke mode can never unlock TEST"):
        Config(run_dir=tmp_path, phase="smoke", smoke=True, unlock_test=True).validate()


def test_resume_cannot_turn_smoke_into_test_run(tmp_path: Path):
    smoke = Config(run_dir=tmp_path, cells=("scirex",), seeds=(42,), phase="smoke", smoke=True)
    write_json(tmp_path / "manifest.json", {"config": smoke.payload()})
    resumed = Config(run_dir=tmp_path, cells=("scirex",), seeds=(42,), phase="all", smoke=False, resume=True, unlock_test=True)
    with pytest.raises(RuntimeError, match="resume fingerprint mismatch"):
        Pipeline(resumed)._assert_resume_fingerprint()


def test_reader_uses_direct_qa_model_without_pipeline_registry(monkeypatch, tmp_path: Path):
    import transformers
    import untie.models

    loaded = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded["tokenizer"] = (Path(path), kwargs)
            return object()

    class FakeModel:
        def to(self, device):
            loaded["device"] = device
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded["model"] = (Path(path), kwargs)
            return FakeModel()

    class FakeAnswerer:
        def __init__(self, **kwargs):
            loaded["answerer"] = kwargs

        def __call__(self, *, question, context):
            return {"answer": context[:4], "score": 0.75, "start": 0, "end": 4}

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForQuestionAnswering", FakeAutoModel)
    monkeypatch.setattr(untie.models, "ExtractiveQuestionAnswerer", FakeAnswerer)

    result = Reader(tmp_path, "cpu", smoke=False)("question", "answer text")

    assert result == {"answer": "answ", "score": 0.75, "start": 0, "end": 4, "error": None}
    assert loaded["tokenizer"][1] == {"local_files_only": True, "trust_remote_code": False}
    assert loaded["model"][1] == {"local_files_only": True, "trust_remote_code": False}
    assert loaded["device"] == "cpu"
    assert loaded["answerer"]["max_answer_tokens"] == 64


def test_qwen_reuses_complete_exact_local_snapshot(monkeypatch, tmp_path: Path):
    import huggingface_hub
    import torch
    import transformers
    from experiments.final_official_split_benchmark.generative import QwenBaseline

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        (snapshot / name).write_text("{}")
    calls = []

    class FakeTokenizer:
        eos_token_id = 0
        truncation_side = "left"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("tokenizer", str(source), kwargs))
            return FakeTokenizer()

    class FakeModel:
        def to(self, device):
            calls.append(("device", device, {}))
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("model", str(source), kwargs))
            return FakeModel()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot))
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    baseline = QwenBaseline("Qwen/example", "exact-revision", "cpu", allow_downloads=True)

    assert baseline.tokenizer.truncation_side == "right"
    assert calls[0] == ("tokenizer", str(snapshot), {"local_files_only": True, "trust_remote_code": False})
    assert calls[1][0:2] == ("model", str(snapshot))
    assert calls[1][2]["local_files_only"] is True
    assert "revision" not in calls[1][2]


def test_meta_policy_falls_back_for_single_candidate_question():
    from types import SimpleNamespace
    from experiments.final_official_split_benchmark.proposed import apply_runtime

    runtime = [{"dataset_cell": "russerrc", "question_id": "q-single", "b0_rank": 1, "chunk_id": 7}]
    policy = SimpleNamespace(selection={"fold_policy": "ACTIVE", "tau_utility": 0.1, "tau_harm": 0.2})

    result = apply_runtime(runtime, policy)

    assert result == {"q-single": {
        "selected_chunk_id": 7,
        "candidate_scores": {},
        "fold_policy": "B0_FALLBACK_NO_ALTERNATIVE",
        "thresholds": {"tau_utility": 0.1, "tau_harm": 0.2},
        "fallback_reason": "NO_ALTERNATIVE_CANDIDATE",
    }}


def test_evaluation_prediction_paths_exclude_runtime_features(tmp_path: Path):
    from experiments.final_official_split_benchmark.analyze import _prediction_paths

    (tmp_path / "test_predictions/P1").mkdir(parents=True)
    (tmp_path / "test_predictions/_runtime_features").mkdir(parents=True)
    expected = tmp_path / "test_predictions/P1/cell.jsonl"
    expected.write_text('{}\n')
    (tmp_path / "test_predictions/_runtime_features/cell.jsonl").write_text('{}\n')

    assert _prediction_paths(tmp_path, "cell") == [expected]
