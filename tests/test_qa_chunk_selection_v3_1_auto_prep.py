from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v3_1_auto_prep.config import AutoPrepConfig
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import (
    V3_1_AUTO_ARTIFACT_ROOT,
    experiment_contract_v3_1_auto,
)
from experiments.qa_chunk_selection.v3_1_auto_prep.gold_relation import gold_relation
from experiments.qa_chunk_selection.v3_1_auto_prep.grounding import validate_candidate
from experiments.qa_chunk_selection.v3_1_auto_prep.io_guard import assert_isolated_output
from experiments.qa_chunk_selection.v3_1_auto_prep.mapping import map_silver_status
from experiments.qa_chunk_selection.v3_1_auto_prep.pipeline import PipelineAutoPrep
from experiments.qa_chunk_selection.v3_1_auto_prep.schema import validate_silver_row
from experiments.qa_chunk_selection.v3_1_auto_prep.teacher import RuleTeacher, annotate_candidate


def test_auto_contract_keeps_prior_trees_read_only() -> None:
    contract = experiment_contract_v3_1_auto()
    assert contract["v3_1_auto_artifacts"] == V3_1_AUTO_ARTIFACT_ROOT
    assert "never overwrite" in str(contract["v3_run"])
    assert "never overwrite" in str(contract["manual_prep"])
    assert contract["label_status"].startswith("silver")


def test_auto_refuses_protected_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v2" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1_prep" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "untie" / "x")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v3_1_auto")
    assert allowed.name == "qa_chunk_selection_v3_1_auto"


def test_grounding_rejects_hallucinated_evidence() -> None:
    source = "We propose a semantic segmentation system."
    ok = validate_candidate(
        {"value": "semantic segmentation", "evidence_text": "semantic segmentation", "evidence_char_start": 13, "evidence_char_end": 34},
        source,
    )
    assert ok["valid"]
    bad = validate_candidate(
        {"value": "BERT", "evidence_text": "BERT was trained on Wikipedia", "evidence_char_start": 0, "evidence_char_end": 4},
        source,
    )
    assert not bad["valid"]
    assert "evidence_not_in_source" in bad["flags"]


def test_gold_relation_cascade() -> None:
    assert gold_relation("Foo", ["Foo"]) == "exact_match_gold"
    assert gold_relation("foo.", ["Foo"]) in {"normalized_match_gold", "exact_match_gold", "semantic_equivalent_gold"}
    assert gold_relation("completely other", ["Foo"]) == "new_value_candidate"


def test_mapping_is_conservative_and_mixed_prefers_current() -> None:
    extracted = [
        {"value": "seg", "evidence_text": "seg", "valid": True, "evidence_char_start": 0, "evidence_char_end": 3},
        {"value": "old", "evidence_text": "old", "valid": True, "evidence_char_start": 4, "evidence_char_end": 7},
    ]
    owners = [
        {"value": "seg", "owner_role": "current_paper", "owner_text": "we", "owner_evidence": "we", "grounded": True},
        {"value": "old", "owner_role": "baseline", "owner_text": "baseline", "owner_evidence": "baseline", "grounded": True},
    ]
    suff = [
        {"value": "seg", "sufficiency": "sufficient"},
        {"value": "old", "sufficiency": "insufficient"},
    ]
    decision = map_silver_status(
        extracted=extracted,
        owners=owners,
        sufficiency=suff,
        gold_relations=["exact_match_gold", "no_gold_relation"],
        contradiction_flags=[],
        source_text="we propose seg versus baseline old",
    )
    assert decision["silver_target_status"] == "silver_target_evidence"
    assert decision["contains_competing_values"] is True
    mixed = map_silver_status(
        extracted=extracted,
        owners=owners,
        sufficiency=[{"value": "seg", "sufficiency": "partial"}, {"value": "old", "sufficiency": "sufficient"}],
        gold_relations=["exact_match_gold", "no_gold_relation"],
        contradiction_flags=[],
        source_text="partial current and strong baseline old",
    )
    assert mixed["silver_target_status"] == "silver_unresolved"


def test_source_label_cannot_be_rewritten() -> None:
    records = synthetic_records(n_docs=2)
    item = records[0]
    amb = next(chunk for chunk in item.chunks if chunk.label == "ambiguous")
    errors = validate_silver_row(
        {
            "question_id": item.question_id,
            "chunk_id": amb.chunk_id,
            "source_label": "positive",
            "silver_target_status": "silver_target_mismatch",
            "silver_confidence": "high",
            "universe_role": "primary_ambiguous",
        },
        item,
    )
    assert any("source_label" in message for message in errors)


def test_rule_teacher_is_source_grounded() -> None:
    item = synthetic_records(n_docs=1)[0]
    chunk = item.chunks[0]
    payload = annotate_candidate(
        RuleTeacher(),
        {
            "aspect": item.aspect,
            "question": item.question,
            "document_id": item.document_id,
            "section_title": chunk.section_title,
            "chunk_text": chunk.text,
            "gold_answers": list(item.gold_answers),
        },
    )
    assert payload["extract"]["candidates"]
    assert all(row["valid"] for row in payload["extract"]["candidates"])


def test_smoke_auto_prep_pipeline_and_freeze(tmp_path: Path) -> None:
    out = tmp_path / "qa_chunk_selection_v3_1_auto"
    config = AutoPrepConfig(
        output_dir=out,
        stages=(
            "init",
            "freeze-teacher",
            "build-universe",
            "teacher-pass",
            "gold-relation",
            "contradiction",
            "map-silver",
            "cluster",
            "qc-controls",
            "qc-gates",
            "coverage",
            "cv-inventory",
            "readiness",
            "freeze",
        ),
        smoke=True,
        min_train_questions_with_TM_silver=1,
        min_eval_questions_with_target_and_TM_silver=0,
        min_teacher_positive_recall=0.0,
        min_teacher_hn_reject_rate=0.0,
        min_teacher_easy_reject_rate=0.0,
    )
    code = PipelineAutoPrep(config).run(list(config.stages))
    assert (out / "progress.jsonl").exists()
    assert (out / "annotation_auto" / "teacher_config.json").exists()
    assert (out / "annotation_auto" / "silver_target_status_all.jsonl").exists()
    readiness = (out / "analysis" / "readiness.json").read_text(encoding="utf-8")
    assert "v3_1_training_allowed" in readiness
    if code == 0:
        assert (out / "annotation_auto" / "silver_target_status_v3_1_frozen.jsonl").exists()
    else:
        assert (out / "analysis" / "freeze_skipped.json").exists()


def test_freeze_does_not_overwrite(tmp_path: Path) -> None:
    from experiments.qa_chunk_selection.v3_1_auto_prep.freeze import freeze_silver

    out = tmp_path / "qa_chunk_selection_v3_1_auto"
    freeze_silver(out, rows=[{"question_id": "q", "chunk_id": 0}], qc_summary={}, manifest={})
    with pytest.raises(PermissionError):
        freeze_silver(out, rows=[{"question_id": "q", "chunk_id": 1}], qc_summary={}, manifest={})
