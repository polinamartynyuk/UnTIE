"""Russian V4.4 metric, routing, and filesystem isolation checks."""
from pathlib import Path

import pytest

from experiments.qa_chunk_selection.v4_4_ru.contract import K, SEEDS, contract
from experiments.qa_chunk_selection.v4_4_ru.io_guard import claim_output
from experiments.qa_chunk_selection.v4_4_ru.metrics_ru import rouge_l_f1
from experiments.qa_chunk_selection.v4_4_ru.routing import RouteFeature, calibrate


def test_contract_is_russian_and_three_seed():
    spec = contract()
    assert spec["language"] == "ru"
    assert tuple(spec["seeds"]) == SEEDS == (42, 43, 44)
    assert K == 5
    assert spec["qa"].startswith("reuse")


def test_unicode_rouge_l_handles_cyrillic():
    assert rouge_l_f1("анализ поэтических текстов", "анализ поэтических текстов") == 1.0
    assert 0.0 < rouge_l_f1("анализ текстов", "анализ поэтических текстов") < 1.0
    assert rouge_l_f1("нет", "другое") == 0.0


def test_output_cannot_overlap_input_or_existing_result(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "artifacts" / "source"
    source.mkdir(parents=True)
    monkeypatch.setattr("experiments.qa_chunk_selection.v4_4_ru.io_guard.PROJECT_ROOT", project)
    with pytest.raises(ValueError, match="separate"):
        claim_output(source / "child", source)
    output = claim_output(project / "artifacts" / "fresh", source)
    (output / "result.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        claim_output(output, source)
    assert (output / "result.txt").read_text(encoding="utf-8") == "keep"


def test_routing_threshold_never_uses_eval():
    rows = [RouteFeature(f"q{i}", f"d{i}", 42, "train", i / 100, True, 1, 0)
            for i in range(100)]
    rows += [RouteFeature(f"e{i}", f"ed{i}", 42, "eval", 1000.0, True, 1, 0)
             for i in range(10)]
    result = calibrate(rows)
    assert result["enabled"]
    assert result["uses_gold"] is False
    assert all(not qid.startswith("e") for qid, _ in result["source_keys"])
