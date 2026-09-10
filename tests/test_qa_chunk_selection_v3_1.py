from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v3_1.config import ExperimentConfigV31, parse_stages
from experiments.qa_chunk_selection.v3_1.contract import V3_1_ARTIFACT_ROOT, experiment_contract_v3_1
from experiments.qa_chunk_selection.v3_1.io_guard import assert_isolated_output
from experiments.qa_chunk_selection.v3_1.pipeline import PipelineV31
from experiments.qa_chunk_selection.v3_1_auto_prep.silver_roles import (
    p_silver_target_ids,
    silver_to_v3_training_overlay,
    tm_silver_ids,
)


def test_v31_contract_isolates_prior_artifacts() -> None:
    contract = experiment_contract_v3_1()
    assert contract["v3_1_artifacts"] == V3_1_ARTIFACT_ROOT
    assert contract["b0_reference"].startswith("B0-max8")
    assert "never create new outer folds" in str(contract["cv"])
    assert contract["internal_test"].startswith("locked")


def test_v31_refuses_protected_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1_prep" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1_auto" / "oops")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v3_1")
    assert allowed.name == "qa_chunk_selection_v3_1"


def test_b0_max8_is_fixed_before_eval() -> None:
    config = ExperimentConfigV31()
    assert config.b0_epochs == 8
    assert config.b0_patience == 2
    assert config.as_v3_config().b2_epochs == 8
    assert parse_stages("all")[0] == "check-preconditions"


def test_silver_roles_high_confidence_only() -> None:
    item = synthetic_records(n_docs=1)[0]
    amb = next(chunk for chunk in item.chunks if chunk.label == "ambiguous")
    overlay = {
        (item.question_id, amb.chunk_id): {
            "silver_target_status": "silver_target_mismatch",
            "silver_confidence": "medium",
            "silver_gold_relations": ["no_gold_relation"],
        }
    }
    assert not tm_silver_ids(item, overlay)
    overlay[(item.question_id, amb.chunk_id)]["silver_confidence"] = "high"
    assert amb.chunk_id in tm_silver_ids(item, overlay)
    assert item.positives[0].chunk_id in p_silver_target_ids(item, overlay)


def test_training_overlay_maps_silver_not_verified() -> None:
    rows = [
        {
            "question_id": "q1",
            "chunk_id": 3,
            "document_id": "d1",
            "source_label": "ambiguous",
            "silver_target_status": "silver_target_evidence",
            "silver_confidence": "high",
            "silver_gold_relations": ["normalized_match_gold"],
            "evidence_cluster_id": "q1::c0",
            "chunk_text": "aaaa",
        },
        {
            "question_id": "q1",
            "chunk_id": 4,
            "document_id": "d1",
            "source_label": "ambiguous",
            "silver_target_status": "silver_target_evidence",
            "silver_confidence": "high",
            "silver_gold_relations": ["normalized_match_gold"],
            "evidence_cluster_id": "q1::c0",
            "chunk_text": "aa",
        },
    ]
    mapped = silver_to_v3_training_overlay(rows)
    assert len(mapped) == 1
    row = next(iter(mapped.values()))
    assert row["v3_target_status"] == "target_evidence"
    assert row["v3_gold_relation"] == "matches_existing_gold"
    assert row["label_status"] == "silver"


def test_non_smoke_requires_frozen_overlay(tmp_path: Path) -> None:
    config = ExperimentConfigV31(
        output_dir=tmp_path / "qa_chunk_selection_v3_1",
        silver_dir=tmp_path / "missing_silver",
        stages=("check-preconditions",),
        smoke=False,
    )
    with pytest.raises(PermissionError):
        PipelineV31(config).run()


def test_smoke_ranking_pipeline(tmp_path: Path) -> None:
    out = tmp_path / "qa_chunk_selection_v3_1"
    config = ExperimentConfigV31(
        output_dir=out,
        stages=("check-preconditions", "prepare", "s0-audit", "b0", "b1", "oof-compare", "g1", "report"),
        smoke=True,
        skip_qa=True,
        n_bootstrap=32,
    )
    code = PipelineV31(config).run()
    assert code == 0
    assert (out / "progress.jsonl").exists()
    assert (out / "analysis" / "s0_audit.json").exists()
    assert (out / "analysis" / "oof" / "b0" / "metrics.json").exists()
    assert (out / "analysis" / "oof" / "b1" / "metrics.json").exists()
    assert (out / "analysis" / "g1.json").exists()
    assert (out / "reports" / "results_v3_1.md").exists()
    g1 = (out / "analysis" / "g1.json").read_text(encoding="utf-8")
    assert "G1-" in g1
