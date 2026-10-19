import json
from pathlib import Path

from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v4_4_ru.io_guard import read_jsonl
from experiments.qa_chunk_selection.v4_5_ru_compact_refinement.audit import audit_dataset
from experiments.qa_chunk_selection.v4_5_ru_compact_refinement.config import Config
from experiments.qa_chunk_selection.v4_5_ru_compact_refinement.contract import (
    DATASETS, RANKERS, STAGE2_CONTRACT, SYSTEMS, TRAINING_RECIPE,
)
from experiments.qa_chunk_selection.v4_5_ru_compact_refinement.io_guard import (
    null_undefined_pairwise, validate_finite, validate_inputs,
)
from experiments.qa_chunk_selection.v4_5_ru_compact_refinement.serialization import strict_prediction


def test_contract_is_compact_frozen_and_dataset_specific():
    assert RANKERS["dity"].model_name == "DiTy/cross-encoder-russian-msmarco"
    assert len(RANKERS["dity"].revision) == 40
    assert RANKERS["bge"].model_name == "BAAI/bge-reranker-v2-m3"
    assert len(RANKERS["bge"].revision) == 40
    assert DATASETS["russerrc"].k == 5
    assert DATASETS["scimdix"].k == 4
    assert STAGE2_CONTRACT["beta"] == 0.0
    assert STAGE2_CONTRACT["question"] == "canonical q0"
    assert STAGE2_CONTRACT["question_selection"] is False
    assert STAGE2_CONTRACT["q5_veto"] is False
    assert SYSTEMS == ("S0_historical_b0", "S1_dity_b0", "S2_v6_support", "S3_adaptive", "S4_bge_b0")


def test_full_config_rejects_training_recipe_drift(tmp_path):
    Config(output_dir=tmp_path / "run").validate()
    changed = Config(output_dir=tmp_path / "run", learning_rate=2e-5)
    try:
        changed.validate()
    except ValueError as exc:
        assert "frozen historical training recipe" in str(exc)
    else:
        raise AssertionError("scientific recipe drift was accepted")
    assert TRAINING_RECIPE["effective_batch_size"] == 16


def test_smoke_contract_is_one_seed_and_one_dataset(tmp_path):
    config = Config(output_dir=tmp_path / "run", datasets=("russerrc",), seeds=(42,), smoke=True)
    config.validate()


def test_completed_historical_inputs_are_present_and_read_only_contract():
    for spec in DATASETS.values():
        paths = validate_inputs(spec)
        assert all(Path(path).is_file() for path in paths.values())


def test_protocol_recovery_confirms_historical_branch_and_q5_replay():
    for spec in DATASETS.values():
        audit = audit_dataset(spec)
        replay = audit["q5_replay"]
        assert replay["alternative_sem_branch_exact_b0"] is True
        assert replay["descriptive_replay_only"] is True
        assert audit["candidate_check"]["paired_quality_valid"] is True


def test_strict_serializer_aligns_scores_ranking_and_q0():
    spec = DATASETS["russerrc"]
    paths = validate_inputs(spec)
    item = records_from_jsonl(paths["records"])[0]
    cache = read_jsonl(paths["qcache"])
    source = next(row for row in cache if str(row["document_id"]) == item.document_id
                  and str(row["aspect"]) == item.aspect and int(row["chunk_id"]) == int(item.chunks[0].chunk_id)
                  and int(row["question_id"]) == 0)
    order = [int(chunk.chunk_id) for chunk in item.chunks]
    qa = {"qa_answer": str(source.get("qa_predicted_answer") or ""),
          "qa_chunk_id": order[0], "qa_chunk_start_char": int(source.get("answer_start") or 0),
          "qa_chunk_end_char": int(source.get("answer_end") or 0)}
    row = strict_prediction(item, ranked_ids=order, qa=qa, flags={}, method="test",
                            extra={"seed": 42, "cv_run": 0})
    assert row["selected_chunk_id"] == order[0] == row["ranked_ids"][0]
    assert int(max(row["scores"], key=row["scores"].get)) == order[0]
    assert set(map(int, row["scores"])) == set(order)


def test_background_launcher_uses_unique_v45_run_directory():
    text = Path("experiments/qa_chunk_selection/v4_5_ru_compact_refinement/launch_background.sh").read_text()
    assert "artifacts/qa_chunk_selection_v4_5_ru/run_$STAMP" in text
    assert "qa_chunk_selection_v4_5_ru_$STAMP.log" in text
    assert "nohup" in text and "--phase all" in text


def test_undefined_pairwise_is_null_without_masking_other_nonfinite_values():
    row = null_undefined_pairwise({
        "pairwise_pos_gt_hn": float("nan"),
        "n_pairs": 0,
        "nested": {"pairwise_pos_gt_hn": float("nan")},
        "pairwise_all_hn": float("nan"),
        "pairwise_clean_hn": float("nan"),
        "score": 0.25,
    })
    assert row["pairwise_pos_gt_hn"] is None
    assert row["nested"]["pairwise_pos_gt_hn"] is None
    assert row["pairwise_all_hn"] is None
    assert row["pairwise_clean_hn"] is None
    validate_finite(row)

    invalid = null_undefined_pairwise({"score": float("nan")})
    try:
        validate_finite(invalid)
    except ValueError as exc:
        assert "root.score" in str(exc)
    else:
        raise AssertionError("a non-finite model score was incorrectly hidden")
