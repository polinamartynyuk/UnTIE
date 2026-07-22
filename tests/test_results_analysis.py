from __future__ import annotations

import pandas as pd

from untie.results_analysis import (
    parse_metrics,
    split_evaluable_rows,
    strategy_applicability_summary,
)


def test_parse_metrics_supports_current_and_legacy_formats() -> None:
    expected = (0.9, 2.0)
    assert parse_metrics({"cosine_sim": 0.9, "lev_dist": 2}) == expected
    assert parse_metrics("{'cosine_sim': 0.9, 'lev_dist': 2}") == expected


def test_evaluable_split_uses_ru_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": ["a", "a", "b"],
            "strategy_applicable": [True, False, False],
        }
    )
    evaluable, degenerate = split_evaluable_rows(frame)
    assert len(evaluable) == 1
    assert len(degenerate) == 2
    assert strategy_applicability_summary(frame) == {
        "total_documents": 2,
        "evaluable_documents": 1,
        "evaluable_documents_pct": 50.0,
    }


def test_old_en_results_remain_fully_evaluable() -> None:
    frame = pd.DataFrame({"doc_id": ["en-1", "en-2"]})
    evaluable, degenerate = split_evaluable_rows(frame)
    assert evaluable.equals(frame)
    assert degenerate.empty
    assert strategy_applicability_summary(frame)["evaluable_documents_pct"] == 100.0
