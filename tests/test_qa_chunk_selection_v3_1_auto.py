from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v3_1.config import ExperimentConfigV31
from experiments.qa_chunk_selection.v3_1.contract import experiment_contract_v3_1
from experiments.qa_chunk_selection.v3_1.io_guard import assert_isolated_output as assert_rank_isolated
from experiments.qa_chunk_selection.v3_1.pipeline import PipelineV31
from experiments.qa_chunk_selection.v3_1_auto_prep.calibrate import mining_score, select_stage_a_threshold
from experiments.qa_chunk_selection.v3_1_auto_prep.run import main as auto_main
from experiments.qa_chunk_selection.v3_1_auto_prep.config import AutoPrepConfig, parse_stages
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import V3_1_AUTO_ARTIFACT_ROOT, experiment_contract_v3_1_auto
from experiments.qa_chunk_selection.v3_1_auto_prep.gold_relation import gold_relation
from experiments.qa_chunk_selection.v3_1_auto_prep.io_guard import assert_isolated_output, load_json, read_jsonl
from experiments.qa_chunk_selection.v3_1_auto_prep.mapping import map_silver
from experiments.qa_chunk_selection.v3_1_auto_prep.pipeline import PipelineAutoPrep
from experiments.qa_chunk_selection.v3_1_auto_prep.qc import locked_ids, validate_rows


def test_auto_prep_loads_frozen_qa_without_cross_encoder() -> None:
    from experiments.qa_chunk_selection import models as model_mod
    from experiments.qa_chunk_selection.v3_1_auto_prep import pipeline as auto_pipeline

    source = Path(auto_pipeline.__file__).read_text(encoding="utf-8")
    assert "cross_encoder_model=\"unused\"" not in source
    assert "load_frozen_qa" in source
    assert hasattr(model_mod, "load_frozen_qa")
    bundle = model_mod.load_frozen_qa(device="cpu", smoke=True)
    assert bundle.smoke is True
    assert bundle.ce_model_name == "lexical-overlap"
    assert bundle.qa_answerer is None


def test_auto_contract_forbids_teacher_and_keeps_prior_trees() -> None:
    contract = experiment_contract_v3_1_auto()
    assert contract["v3_1_auto_artifacts"] == V3_1_AUTO_ARTIFACT_ROOT
    assert "never overwrite" in str(contract["v3_run"])
    assert "never overwrite" in str(contract["manual_prep"])
    assert "generative teacher" in str(contract["annotator"]).lower() or "no generative" in str(contract["annotator"]).lower()
    assert "generative LLM" in contract["forbidden_models"]
    assert "NLI" in contract["forbidden_models"]
    ranking = experiment_contract_v3_1()
    assert ranking["annotator"].startswith("frozen extractive QA")
    assert ranking["primary_endpoint"].startswith("Token F1")


def test_auto_and_ranking_refuse_protected_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v2" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1_prep" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1" / "oops")
    with pytest.raises(ValueError):
        assert_rank_isolated(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3" / "oops")
    with pytest.raises(ValueError):
        assert_rank_isolated(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1_auto" / "oops")
    with pytest.raises(ValueError):
        assert_rank_isolated(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3_1_prep" / "oops")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v3_1_auto")
    assert allowed.name == "qa_chunk_selection_v3_1_auto"


def test_gold_relation_is_string_only_and_not_ownership() -> None:
    exact = gold_relation("ResNet-50", ["ResNet-50"])
    assert exact["qa_gold_relation"] == "exact_match_gold"
    assert exact["allowed_for_primary_te"] is True
    overlap = gold_relation("resnet 50 backbone", ["ResNet-50"])
    assert overlap["qa_gold_relation"] in {"normalized_match_gold", "overlap_match_gold"}
    novel = gold_relation("completely different phrase", ["ResNet-50"])
    assert novel["qa_gold_relation"] == "new_value_candidate"
    assert novel["allowed_for_primary_te"] is False
    assert novel["new_value_candidate"] is True


def test_mapping_is_conservative() -> None:
    unresolved = map_silver(
        qa_plausible=False,
        consistency=1.0,
        target_score=2.0,
        external_score=0.0,
        gold={"allowed_for_primary_te": True, "new_value_candidate": False, "qa_gold_relation": "exact_match_gold"},
        discourse={"current_paper_marker_score": 2, "independent_external_context": False},
        paraphrase_spans=["ResNet"],
        canonical_span="ResNet",
        target_span="ResNet",
        external_span="",
        tau_consistency=0.5,
        delta=0.25,
        far_from_threshold=True,
        near_threshold=False,
    )
    assert unresolved["silver_target_status"] == "silver_unresolved"
    assert unresolved["abstained"] is True

    novel = map_silver(
        qa_plausible=True,
        consistency=1.0,
        target_score=2.0,
        external_score=0.0,
        gold={"allowed_for_primary_te": False, "new_value_candidate": True, "qa_gold_relation": "new_value_candidate"},
        discourse={"current_paper_marker_score": 2, "independent_external_context": False},
        paraphrase_spans=["new value"],
        canonical_span="new value",
        target_span="new value",
        external_span="",
        tau_consistency=0.5,
        delta=0.25,
        far_from_threshold=True,
        near_threshold=False,
    )
    assert novel["silver_target_status"] == "silver_unresolved"
    assert novel["new_value_candidate"] is True

    te = map_silver(
        qa_plausible=True,
        consistency=1.0,
        target_score=2.0,
        external_score=0.0,
        gold={"allowed_for_primary_te": True, "new_value_candidate": False, "qa_gold_relation": "exact_match_gold"},
        discourse={"current_paper_marker_score": 2, "independent_external_context": False},
        paraphrase_spans=["ResNet"],
        canonical_span="ResNet",
        target_span="ResNet",
        external_span="",
        tau_consistency=0.5,
        delta=0.25,
        far_from_threshold=True,
        near_threshold=False,
    )
    assert te["silver_target_status"] == "silver_target_evidence"
    assert te["silver_confidence"] == "high"

    tm = map_silver(
        qa_plausible=True,
        consistency=1.0,
        target_score=0.0,
        external_score=2.0,
        gold={"allowed_for_primary_te": False, "new_value_candidate": False, "qa_gold_relation": "no_gold_relation"},
        discourse={"current_paper_marker_score": 0, "independent_external_context": True},
        paraphrase_spans=["BERT"],
        canonical_span="BERT",
        target_span="",
        external_span="BERT",
        tau_consistency=0.5,
        delta=0.25,
        far_from_threshold=True,
        near_threshold=False,
    )
    assert tm["silver_target_status"] == "silver_target_mismatch"
    assert tm["silver_confidence"] == "high"


def test_stage_a_threshold_is_recall_oriented() -> None:
    chosen = select_stage_a_threshold([0.9, 0.8, 0.7], [0.1, 0.2], r_min=0.80, max_easy_pass=0.25)
    assert chosen["ok"] is True
    assert chosen["cal_positive_recall"] + 1e-12 >= 0.80
    assert chosen["cal_easy_pass"] <= 0.25


def test_stage_a_threshold_does_not_fall_back_to_zero() -> None:
    pos = [0.10, 0.20, 0.30, 0.40, 0.50]
    easy = [0.08, 0.18, 0.28, 0.38, 0.48]
    chosen = select_stage_a_threshold(pos, easy, r_min=0.80, max_easy_pass=0.25)
    assert chosen["ok"] is False
    assert chosen["reason"] == "no_feasible_tau"
    assert float(chosen["tau_qa"]) > 0.0
    assert chosen["cal_positive_recall"] + 1e-12 >= 0.80


def test_mining_score_uses_raw_and_margin() -> None:
    mined = mining_score(3.0, 1.0, [0.0, 3.0, 6.0], [0.0, 0.5, 1.0])
    assert mined["qa_raw_cdf"] == 2 / 3
    assert mined["qa_margin_cdf"] == 1.0
    assert abs(float(mined["qa_plausibility_score"]) - (0.5 * 2 / 3 + 0.5)) < 1e-12


def test_internal_test_is_not_labeled(tmp_path: Path) -> None:
    config = AutoPrepConfig(
        output_dir=tmp_path / "qa_chunk_selection_v3_1_auto",
        v3_artifact_dir=tmp_path / "missing_v3",
        stages=("init", "build-universe"),
        smoke=True,
    )
    pipe = PipelineAutoPrep(config)
    assert pipe.run(["init", "build-universe"]) == 0
    locked = replace(pipe.unlocked[0], split="internal_test", question_id="internal_test::task")
    pipe.unlocked = pipe.unlocked + (locked,)
    pipe.records = pipe.records + (locked,)
    pipe.stage_build_universe()
    assert all(row["question_id"] != locked.question_id for row in pipe.universe)
    validation = validate_rows(
        [
            {
                "question_id": locked.question_id,
                "chunk_id": locked.chunks[0].chunk_id,
                "source_label": locked.chunks[0].label,
            }
        ],
        (locked,),
        locked_ids((locked,)),
    )
    assert validation["ok"] is False
    assert any("INTERNAL_TEST" in str(error["error"]) for error in validation["errors"])


def test_auto_prep_smoke_and_freeze_no_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "qa_chunk_selection_v3_1_auto"
    config = AutoPrepConfig(
        output_dir=out,
        v3_artifact_dir=tmp_path / "missing_v3",
        stages=parse_stages("full"),
        smoke=True,
        allow_partial_freeze=True,
        resume=False,
    )
    code = PipelineAutoPrep(config).run(list(config.stages))
    assert code in {0, 2}
    assert (out / "progress.jsonl").exists()
    silver = out / "annotation_qa_silver"
    assert (silver / "qa_model_manifest.json").exists()
    assert (silver / "stage_a_all_ambiguous.jsonl").exists()
    assert (silver / "discourse_features.jsonl").exists()
    assert (out / "analysis" / "readiness.json").exists()
    rows = []
    for path in silver.glob("oof/run*/silver_target_status.jsonl"):
        rows.extend(read_jsonl(path))
    assert rows
    records = {item.question_id: item for item in synthetic_records(n_docs=8) if item.positives}
    for row in rows:
        item = records.get(str(row["question_id"]))
        if item is None:
            continue
        chunk = item.chunk_by_id(int(row["chunk_id"]))
        assert row["source_label"] == chunk.label
        assert row["source_label"] != row.get("silver_target_status")
    assert load_json(silver / "qa_model_manifest.json")["qa_model"] == "lexical-smoke"
    if (silver / "silver_manifest.json").exists():
        from experiments.qa_chunk_selection.v3_1_auto_prep.freeze import freeze_silver

        with pytest.raises(PermissionError):
            freeze_silver(out, qc_summary={}, manifest={})


def test_ranking_refuses_missing_freeze(tmp_path: Path) -> None:
    config = ExperimentConfigV31(
        output_dir=tmp_path / "qa_chunk_selection_v3_1",
        v3_artifact_dir=tmp_path / "missing_v3",
        silver_dir=tmp_path / "empty_silver",
        smoke=False,
        skip_qa=True,
        stages=("check-preconditions",),
    )
    with pytest.raises(PermissionError):
        PipelineV31(config).run()


def test_ranking_smoke_s0_b0_b1_g1_report(tmp_path: Path) -> None:
    config = ExperimentConfigV31(
        output_dir=tmp_path / "qa_chunk_selection_v3_1",
        v3_artifact_dir=tmp_path / "missing_v3",
        silver_dir=tmp_path / "empty_silver",
        smoke=True,
        skip_qa=True,
        stages=("check-preconditions", "prepare", "s0-audit", "b0", "b1", "oof-compare", "g1", "report"),
    )
    code = PipelineV31(config).run()
    assert code == 0
    root = config.output_dir
    assert (root / "progress.jsonl").exists()
    assert (root / "analysis" / "s0_audit.json").exists()
    assert (root / "analysis" / "oof" / "b0" / "metrics.json").exists()
    assert (root / "analysis" / "oof" / "b1" / "metrics.json").exists()
    g1 = load_json(root / "analysis" / "g1.json")
    assert g1["decision"] in {"G1-A", "G1-B", "G1-STOP"}
    assert (root / "reports" / "results_v3_1.json").exists()
    assert (root / "annotation" / "oof" / "run0" / "silver_target_status.jsonl").exists()
    assert load_json(root / "data" / "prepare_summary.json")["source_label_rewritten"] is False
    assert load_json(root / "data" / "prepare_summary.json")["fold_specific_silver"] is True


def test_auto_prep_refuses_completed_output(tmp_path: Path) -> None:
    out = tmp_path / "qa_chunk_selection_v3_1_auto"
    (out / "analysis").mkdir(parents=True)
    (out / "analysis" / "readiness.json").write_text("{\"v3_1_training_allowed\": false}\n", encoding="utf-8")
    assert auto_main(["--output-dir", str(out), "--smoke", "--stages", "init"]) == 2
    assert auto_main(["--output-dir", str(out), "--smoke", "--stages", "init", "--allow-existing-output"]) == 0


def test_auto_prep_reuses_stage_a(tmp_path: Path) -> None:
    first = tmp_path / "qa_chunk_selection_v3_1_auto_v1"
    second = tmp_path / "qa_chunk_selection_v3_1_auto_v2"
    cfg1 = AutoPrepConfig(
        output_dir=first,
        v3_artifact_dir=tmp_path / "missing_v3",
        stages=("init", "freeze-qa", "build-universe", "stage-a"),
        smoke=True,
        resume=False,
    )
    assert PipelineAutoPrep(cfg1).run(list(cfg1.stages)) == 0
    src = first / "annotation_qa_silver" / "stage_a_all_ambiguous.jsonl"
    assert src.exists()
    cfg2 = AutoPrepConfig(
        output_dir=second,
        v3_artifact_dir=tmp_path / "missing_v3",
        stages=("init", "freeze-qa", "build-universe", "stage-a", "calibrate"),
        smoke=True,
        resume=False,
        reuse_stage_a_dir=first,
        skip_qa_until_needed=True,
    )
    assert PipelineAutoPrep(cfg2).run(list(cfg2.stages)) == 0
    imported = load_json(second / "analysis" / "reused_stage_a.json")
    assert imported["n_imported"] > 0
    assert (second / "analysis" / "stage_a_pareto.json").exists()
    cost = load_json(second / "annotation_qa_silver" / "stage_a_cost.json")
    assert cost["n_uncached_calls"] == 0
