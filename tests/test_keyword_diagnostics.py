from __future__ import annotations

from pathlib import Path

import pytest

from untie.keyword_diagnostics import (
    audit_pool_aggregation,
    parse_tuning_trace,
    sffs_trace_rows,
    stability_selection_rows,
)
from untie.keyword_tuning import KeywordEvidence


def test_audit_pool_rejects_stopwords_and_low_support() -> None:
    evidence = [
        KeywordEvidence("the", "train-1", 1.0, 1.0, chunk_support_rate=1.0),
        KeywordEvidence("useful term", "train-1", 0.2, 0.3, chunk_support_rate=1.0),
        KeywordEvidence("useful term", "train-2", 0.4, 0.5, chunk_support_rate=1.0),
        KeywordEvidence("rare term", "train-1", 0.9, 0.9, chunk_support_rate=1.0),
    ]
    rows = audit_pool_aggregation(evidence, ("train-1", "train-2"), min_document_support=2)
    by_term = {row["term"]: row for row in rows}
    assert by_term["the"]["decision"] == "rejected"
    assert by_term["useful term"]["decision"] == "kept"
    assert by_term["rare term"]["decision"] == "rejected"
    assert by_term["rare term"]["reason"] == "low_document_support"


@pytest.mark.skipif(
    not Path(
        "experiments/analysis_results/keyword_tuning_task/en/full_large_v3/scart_tuned_model.tuning.json"
    ).exists(),
    reason="v3 tuning trace not available",
)
def test_trace_parsers_load_v3_artifact() -> None:
    payload = parse_tuning_trace(
        "experiments/analysis_results/keyword_tuning_task/en/full_large_v3/scart_tuned_model.tuning.json"
    )
    sffs = sffs_trace_rows(payload)
    stability = stability_selection_rows(payload, stability_threshold=0.2)
    assert payload["keywords"]
    assert sffs
    assert any(row["stage"] == "stability_threshold" for row in stability)
