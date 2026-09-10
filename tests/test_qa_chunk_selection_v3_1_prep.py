from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v3.overlay import sample_annotation_queue
from experiments.qa_chunk_selection.v3.split import build_cv_splits
from experiments.qa_chunk_selection.v3_1_prep.annotator import export_annotator_queue
from experiments.qa_chunk_selection.v3_1_prep.config import PrepConfig, parse_stages
from experiments.qa_chunk_selection.v3_1_prep.contract import FORBIDDEN_ANNOTATOR_FIELDS, V3_ARTIFACT_ROOT, experiment_contract_v3_1_prep
from experiments.qa_chunk_selection.v3_1_prep.io_guard import assert_isolated_output, assert_v3_not_output, load_json
from experiments.qa_chunk_selection.v3_1_prep.pipeline import PipelinePrep
from experiments.qa_chunk_selection.v3_1_prep.qc import effective_target_role, overlay_for_roles, validate_overlay
from experiments.qa_chunk_selection.v3_1_prep.taxonomy import assert_no_model_scores, validate_prep_row


def test_contract_keeps_v3_read_only() -> None:
    contract = experiment_contract_v3_1_prep()
    assert contract["v3_artifacts"] == V3_ARTIFACT_ROOT
    assert "never overwrite" in str(contract["v3_run"])
    assert "b_score" in contract["annotator_must_not_see"]


def test_refuses_e0_v2_v3_and_production_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v2" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "untie" / "x")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v3_1_prep")
    assert allowed.name == "qa_chunk_selection_v3_1_prep"
    with pytest.raises(ValueError):
        assert_v3_not_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3", PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3")


def test_annotator_pack_hides_scores() -> None:
    records = synthetic_records(n_docs=4)
    sampled = sample_annotation_queue(records, scores={(item.question_id, 1): 0.99 for item in records}, hard_n=4, repr_n=2, seed=42)
    payload = export_annotator_queue(
        records=records,
        queue=sampled["queue"],
        metadata=sampled["metadata"],
        annotator_id="annotator_1",
    )
    assert payload["n"] == 6
    for row in payload["queue"]:
        assert_no_model_scores(row)
        for field in FORBIDDEN_ANNOTATOR_FIELDS:
            assert field not in row
        assert row["chunk_text"]
        assert row["question"]
        assert "previous_sentence" in row
        assert row["v3_target_status"] is None
        assert row["source_label"] == "ambiguous"


def test_source_label_cannot_be_rewritten() -> None:
    records = synthetic_records(n_docs=2)
    item = records[0]
    amb = next(chunk for chunk in item.chunks if chunk.label == "ambiguous")
    row = {
        "question_id": item.question_id,
        "chunk_id": amb.chunk_id,
        "source_label": "positive",
        "v3_target_status": "target_mismatch",
        "v3_gold_relation": "no_gold_match_expected",
        "v3_mismatch_type": "related_work",
        "v3_annotation_confidence": "high",
        "v3_annotation_reason": "other paper",
    }
    errors = validate_prep_row(row, item, require_filled=True)
    assert any("source_label" in message for message in errors)


def test_internal_test_leakage_rejected() -> None:
    records = synthetic_records(n_docs=2)
    item = records[0]
    amb = next(chunk for chunk in item.chunks if chunk.label == "ambiguous")
    row = {
        "document_id": item.document_id,
        "question_id": item.question_id,
        "chunk_id": amb.chunk_id,
        "aspect": item.aspect,
        "source_label": amb.label,
        "source_ambiguous_subtype": amb.primary_class,
        "v3_target_status": "target_mismatch",
        "v3_mismatch_type": "baseline",
        "v3_gold_relation": "no_gold_match_expected",
        "v3_annotation_confidence": "high",
        "v3_annotation_reason": "baseline",
        "annotation_sampling_stratum": "hard_case",
    }
    report = validate_overlay(
        [row],
        records,
        locked={"documents": {item.document_id}, "questions": set()},
        require_filled=True,
    )
    assert report["ok"] is False
    assert any("leakage" in err["error"] for err in report["errors"])


def test_effective_target_role_uses_overlay_not_source_rewrite() -> None:
    records = synthetic_records(n_docs=1)
    item = records[0]
    amb = next(chunk for chunk in item.chunks if chunk.label == "ambiguous")
    pos = next(chunk for chunk in item.chunks if chunk.label == "positive")
    overlay = overlay_for_roles(
        {
            (item.question_id, amb.chunk_id): {
                "question_id": item.question_id,
                "chunk_id": amb.chunk_id,
                "v3_target_status": "target_mismatch",
                "v3_gold_relation": "matches_existing_gold",
            }
        }
    )
    assert effective_target_role(item, pos.chunk_id, overlay) == "target_evidence_original"
    assert effective_target_role(item, amb.chunk_id, overlay) == "target_mismatch_verified"
    assert item.chunks[amb.chunk_id].label == "ambiguous"


def test_parse_stages_default_excludes_freeze() -> None:
    stages = parse_stages("all")
    assert "freeze" not in stages
    assert "init" in stages
    assert "export-annotator" in stages


def test_smoke_pipeline_writes_only_to_tmp(tmp_path: Path) -> None:
    out = tmp_path / "prep"
    config = PrepConfig(output_dir=out, v3_artifact_dir=tmp_path / "v3-missing", stages=("init", "export-annotator", "ingest", "validate", "audit", "coverage", "cv-inventory", "readiness"), smoke=True)
    PipelinePrep(config).run(list(config.stages))
    assert (out / "annotator" / "to_fill.jsonl").exists()
    assert (out / "annotator" / "index.html").exists()
    assert (out / "analysis" / "readiness.json").exists()
    assert (out / "progress.jsonl").exists()
    readiness = (out / "analysis" / "readiness.json").read_text(encoding="utf-8")
    assert "v3_1_training_allowed" in readiness
    assert not (tmp_path / "v3-missing").exists()


def test_freeze_skipped_when_overlay_empty(tmp_path: Path) -> None:
    out = tmp_path / "prep"
    config = PrepConfig(
        output_dir=out,
        v3_artifact_dir=tmp_path / "v3-missing",
        stages=("init", "export-annotator", "ingest", "validate", "audit", "coverage", "cv-inventory", "readiness", "freeze"),
        smoke=True,
    )
    code = PipelinePrep(config).run(list(config.stages))
    assert code == 2
    assert not (out / "annotation" / "target_status_v3_1_frozen.jsonl").exists()
    skipped = (out / "analysis" / "freeze_skipped.json").read_text(encoding="utf-8")
    assert "no verified target_mismatch" in skipped or "overlay not ready" in skipped
    readiness = load_json(out / "analysis" / "readiness.json")
    assert readiness["items"]["distributions_built"] is True
    assert readiness["items"]["verified_target_mismatch_gt_0"] is False


def test_cv_splits_untouched_by_prep() -> None:
    records = synthetic_records(n_docs=10)
    payload = build_cv_splits(records, n_folds=5, seed=42)
    assert payload["frozen"] is True
    assert payload["split_payload_hash"]
