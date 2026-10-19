from pathlib import Path

import pytest

from experiments.qa_chunk_selection.v4_4_ru_data.config import ASPECT_SPECS, RuDataConfig
from experiments.qa_chunk_selection.v4_4_ru_data.official_source import (
    SourceDocument,
    answers_by_aspect,
    parse_tagged_text,
)


def test_nested_tags_keep_exact_offsets_and_all_aspects():
    raw = "<Contrib>Решён <Task>поиск ответа</Task> методом <Method>поиска</Method></Contrib>."
    clean, spans, warnings = parse_tagged_text(raw)
    assert warnings == ()
    assert "<" not in clean
    assert {span.tag for span in spans} == {"Contrib", "Task", "Method"}
    assert all(clean[span.start : span.end] == span.text for span in spans)


def test_malformed_crossing_tag_is_repaired_and_reported():
    clean, spans, warnings = parse_tagged_text("Для <Use>обучения <Adv>быстро</Use>.")
    assert clean == "Для обучения быстро."
    assert any("implicitly closed <Adv>" in warning for warning in warnings)
    assert {(span.tag, span.text) for span in spans} == {("Use", "обучения быстро"), ("Adv", "быстро")}


def test_answers_keep_native_taxonomy():
    clean, spans, warnings = parse_tagged_text("<Goal>проверить метод</Goal>")
    doc = SourceDocument("d", "x", "part.csv", "", clean, spans, warnings)
    assert answers_by_aspect(doc) == {"goal": ("проверить метод",)}


def test_default_config_selects_every_official_aspect(tmp_path: Path):
    config = RuDataConfig(output_dir=tmp_path / "new")
    assert set(config.aspects) == {value[0] for value in ASPECT_SPECS.values()}
    config.validate()


def test_source_modes_are_exclusive(tmp_path: Path):
    with pytest.raises(ValueError):
        RuDataConfig(output_dir=tmp_path / "x", source_dir=tmp_path, download=True).validate()
