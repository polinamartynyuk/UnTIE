from __future__ import annotations

import pandas as pd
import pytest

from untie.extraction_metrics import (
    METRIC_COLUMNS,
    bertscore_metrics,
    char_f1,
    compute_metrics_rowwise,
    compute_row_metrics,
    normalize_gold,
    normalize_pred,
    rouge_l_f1,
    token_f1,
)


def test_normalize_pred_and_gold() -> None:
    assert normalize_pred(None) == ""
    assert normalize_pred("  answer  ") == "answer"
    assert normalize_gold("single") == ["single"]
    assert normalize_gold([" first ", None, "", "second"]) == ["first", "second"]


def test_char_f1_exact_and_empty_cases() -> None:
    assert char_f1("Paris", "Paris") == pytest.approx(1.0)
    assert char_f1("", "Paris") == 0.0
    assert char_f1("Paris", []) == 0.0
    assert char_f1("Paris", ["London", "Paris"]) == pytest.approx(1.0)
    assert char_f1("abc", "xyz") == 0.0


def test_char_f1_partial_overlap() -> None:
    score = char_f1("hello", "hello world")
    assert 0.0 < score < 1.0


def test_rouge_l_f1_exact_match() -> None:
    pytest.importorskip("rouge_score")
    assert rouge_l_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l_f1("", "the cat sat") == 0.0


def test_token_f1_exact_match() -> None:
    pytest.importorskip("evaluate")
    assert token_f1("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(100.0)


def test_compute_row_metrics_without_bertscore() -> None:
    pytest.importorskip("rouge_score")
    pytest.importorskip("evaluate")
    metrics = compute_row_metrics(
        "the cat sat on the mat",
        "the cat sat on the mat",
        include_bertscore=False,
    )
    assert set(metrics) == {"char_f1", "token_f1", "rouge_l_f1"}
    assert metrics["char_f1"] == pytest.approx(1.0)
    assert metrics["rouge_l_f1"] == pytest.approx(1.0)


def test_compute_metrics_rowwise_adds_columns() -> None:
    pytest.importorskip("rouge_score")
    pytest.importorskip("evaluate")
    frame = pd.DataFrame(
        {
            "corrected_answer": ["Paris", "London"],
            "tasks_cleaned": [["Paris"], ["Paris"]],
        }
    )
    scored = compute_metrics_rowwise(
        frame,
        include_bertscore=False,
        show_progress=False,
    )
    assert list(scored.columns[-3:]) == ["char_f1", "token_f1", "rouge_l_f1"]
    assert scored.loc[0, "char_f1"] == pytest.approx(1.0)
    assert scored.loc[1, "char_f1"] == pytest.approx(0.0)


def test_bertscore_metrics_runs_when_available() -> None:
    pytest.importorskip("bert_score")
    result = bertscore_metrics("Paris", "Paris", lang="en")
    assert set(result) == {"p", "r", "f1"}
    assert all(key in METRIC_COLUMNS for key in ("bertscore_p", "bertscore_r", "bertscore_f1"))
