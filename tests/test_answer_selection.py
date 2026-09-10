from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.answer_aggregation.io_util import assert_isolated_output
from untie.answer_selection import (
    AggregationSettings,
    merge_near_duplicate_answers,
    normalize_answer_key,
    query_prototypes,
    question_focus,
    select_typical_span,
)
from untie.domain import Answer, Sentence, TextChunk
from untie.qa import AnswerAggregator


def _chunk(index: int, text: str) -> TextChunk:
    return TextChunk(
        sentences=(Sentence(text=text, number=index, tokens=tuple(text.split())),),
        text=text,
        token_count=len(text.split()),
    )


def _answer(text: str, index: int = 0, confidence: float = 0.1) -> Answer:
    return Answer(text=text, chunk=_chunk(index, text), confidence=confidence)


class OverlapEncoder:
    """Bag-of-words cosine so tests do not depend on a task lexicon."""

    def encode(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        vocab = ("dataset", "optimizer", "task", "imagenet", "adam", "chunking", "coordinates")
        rows = []
        for text in texts:
            tokens = set(str(text).casefold().replace("-", " ").split())
            rows.append([1.0 if word in tokens else 0.0 for word in vocab])
        return np.asarray(rows, dtype=float)


def test_normalize_collapses_hyphen_and_spacing_variants() -> None:
    assert normalize_answer_key("Language Modeling") == normalize_answer_key("language modeling")
    assert normalize_answer_key("re - identification") == normalize_answer_key("re-identification")
    assert normalize_answer_key("image-to-image translation") == normalize_answer_key(
        "image - to - image translation"
    )


def test_merge_groups_near_duplicates_without_a_field_lexicon() -> None:
    answers = [
        _answer("coordinates", 0),
        _answer("coordinates", 1),
        _answer("coordinates", 2),
        _answer("Language modeling", 3),
        _answer("language modeling", 4),
        _answer("language-modeling", 5),
    ]
    groups = merge_near_duplicate_answers(answers)
    keys = {normalize_answer_key(group[0].text) for group in groups}
    assert keys == {"coordinates", "language modeling"}
    sizes = sorted(len(group) for group in groups)
    assert sizes == [3, 3]


def test_question_focus_is_derived_from_the_question_not_a_task_list() -> None:
    assert "task" in question_focus("Which task was solved?")
    assert "dataset" in question_focus("Which dataset is used?")
    assert "optimizer" in question_focus("Which optimizer is used?")
    assert "solved" not in question_focus("Which task was solved?")
    prototypes = query_prototypes("Which dataset is used?")
    assert prototypes[0] == "Which dataset is used?"
    assert "dataset" in prototypes[1]


def test_query_filter_follows_the_question() -> None:
    encoder = OverlapEncoder()
    answers = [
        _answer("ImageNet dataset", 0),
        _answer("Adam optimizer", 1),
        _answer("coordinates", 2),
        _answer("coordinates", 3),
        _answer("coordinates", 4),
    ]
    settings = AggregationSettings(
        merge_near_duplicates=True,
        query_quantile=0.7,
        min_keep=1,
        language="en",
    )
    dataset_pick, _support, _stats = select_typical_span(
        answers,
        encoder,
        question="Which dataset is used?",
        settings=settings,
    )
    optimizer_pick, _support, _stats = select_typical_span(
        answers,
        encoder,
        question="Which optimizer is used?",
        settings=settings,
    )
    assert dataset_pick is not None
    assert optimizer_pick is not None
    assert "dataset" in dataset_pick.text.casefold()
    assert "optimizer" in optimizer_pick.text.casefold()


def test_legacy_centroid_still_prefers_exact_duplicates() -> None:
    encoder = OverlapEncoder()
    answers = [
        _answer("coordinates", 0),
        _answer("coordinates", 1),
        _answer("coordinates", 2),
        _answer("coordinates", 3),
        _answer("chunking", 4),
    ]
    selected, supporting, _stats = select_typical_span(answers, encoder)
    assert selected is not None
    assert selected.text == "coordinates"
    assert len(supporting) == 4
    legacy = AnswerAggregator(encoder).aggregate(answers)
    assert legacy is not None
    assert legacy.text == "coordinates"


def test_answer_selection_module_has_no_task_span_lexicon() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "untie" / "answer_selection.py"
    ).read_text(encoding="utf-8")
    for banned in ("detection", "classification", "segmentation", "recognition", "chunking"):
        assert banned not in source.casefold()


def test_output_dir_refuses_protected_trees() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="refuses to write"):
        assert_isolated_output(root / "artifacts" / "keybert_qa_integration" / "en_full", root)
    allowed = assert_isolated_output(root / "artifacts" / "answer_aggregation" / "en", root)
    assert "answer_aggregation" in str(allowed)


def test_launch_script_points_at_isolated_output() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "answer_aggregation"
        / "launch_background.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "nohup" in text
    assert "artifacts/answer_aggregation" in text
    assert "untie_venv/bin/python" in text
    assert 'QUESTION="${QUESTION:-Which task was solved?}"' in text
    assert ' --question "$QUESTION"' in text
    assert "hierarchical_topic_optimized" not in text.split("OUT_DIR=", 1)[1][:400]
