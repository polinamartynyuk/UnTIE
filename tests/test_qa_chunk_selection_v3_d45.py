from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v3_d45.config import D45Config
from experiments.qa_chunk_selection.v3_d45.d5_audit import run_d5_audit
from experiments.qa_chunk_selection.v3_d45.io_guard import assert_isolated_output
from experiments.qa_chunk_selection.v3_d45.pipeline import PipelineD45
from experiments.qa_chunk_selection.v3_d45.windows import locate_gold, windows_for_positive


def test_d45_refuses_protected_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1_auto" / "oops")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v3_d45")
    assert allowed.name == "qa_chunk_selection_v3_d45"


def test_d4_windows_localize_synthetic_gold() -> None:
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    chunk = item.positives[0]
    located = locate_gold(chunk, item)
    assert located is not None
    windows = windows_for_positive(item, chunk)
    assert windows["full_chunk"]["localized"] is True
    assert windows["gold_sentence"]["localized"] is True
    assert item.gold_answers[0] in windows["gold_sentence"]["text"]
    assert windows["minimal_gold_window"]["localized"] is True


def test_d5_overlap_is_not_independent() -> None:
    records = [item for item in synthetic_records(n_docs=4) if item.positives]
    _rows, summary = run_d5_audit(records)
    assert summary["n_questions"] == len(records)
    assert "n_with_ge2_independent_clusters" in summary
    assert summary["rate_ge2_independent_clusters"] <= summary["rate_ge2_positives"] + 1e-12


def test_d45_smoke_pipeline(tmp_path: Path) -> None:
    config = D45Config(
        output_dir=tmp_path / "qa_chunk_selection_v3_d45",
        v3_artifact_dir=tmp_path / "missing_v3",
        smoke=True,
    )
    assert PipelineD45(config).run(list(config.stages)) == 0
    root = config.output_dir
    assert (root / "progress.jsonl").exists()
    assert (root / "analysis" / "d4" / "oracle_summary.json").exists()
    assert (root / "analysis" / "d5" / "audit.json").exists()
    assert (root / "reports" / "results_v3_d45.json").exists()
