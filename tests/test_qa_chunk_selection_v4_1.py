from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v4.contract import V4_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_1.config import ExperimentConfigV41, parse_stages
from experiments.qa_chunk_selection.v4_1.contract import (
    PRIMARY_FINAL_SYSTEMS,
    UTILITY_F1_PCT,
    UTILITY_F1_UNIT_MISTAKE,
    experiment_contract_v4_1,
)
from experiments.qa_chunk_selection.v4_1.evaluate import paired_delta
from experiments.qa_chunk_selection.v4_1.io_guard import assert_isolated_output, refuse_completed_output
from experiments.qa_chunk_selection.v4_1.pipeline import PipelineV41
from experiments.qa_chunk_selection.v4_1.ranking import isolate_shortlist_ranking, v6_fused_score
from experiments.qa_chunk_selection.v4_1.run import main as v41_main


PROTECTED = (
    "artifacts/qa_chunk_selection",
    "artifacts/qa_chunk_selection_v2",
    "artifacts/qa_chunk_selection_v3",
    "artifacts/qa_chunk_selection_v3_1_prep",
    "artifacts/qa_chunk_selection_v3_1_auto",
    "artifacts/qa_chunk_selection_v3_1_auto_v2",
    "artifacts/qa_chunk_selection_v3_1",
    "artifacts/qa_chunk_selection_v3_d45",
    "artifacts/qa_chunk_selection_v4",
)


def test_v4_1_contract_keeps_v4_readonly() -> None:
    contract = experiment_contract_v4_1()
    assert contract["v4_1_artifacts"] == "artifacts/qa_chunk_selection_v4_1"
    assert contract["v4_artifacts"] == "artifacts/qa_chunk_selection_v4"
    assert "never overwrite V4" in str(contract["prior_trees"])
    assert contract["utility_threshold_pct"] == 80.0
    assert UTILITY_F1_PCT == 80.0
    assert UTILITY_F1_UNIT_MISTAKE == 0.8
    assert contract["v6_feature_scale"].startswith("exactly V4")
    assert "B0-live" in contract["primary_final_systems"]
    assert "C1" in PRIMARY_FINAL_SYSTEMS
    from experiments.qa_chunk_selection.v4_1.contract import PROTECTED_WRITE_ROOTS

    assert V4_ARTIFACT_ROOT in PROTECTED_WRITE_ROOTS


def test_v4_1_refuses_protected_writes_including_v4(tmp_path: Path) -> None:
    for relative in PROTECTED:
        with pytest.raises(ValueError):
            assert_isolated_output(PROJECT_ROOT / relative / "oops")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v4_1")
    assert allowed.name == "qa_chunk_selection_v4_1"


def test_v4_1_cli_refuses_v4_and_completed(tmp_path: Path) -> None:
    assert v41_main(["--output-dir", str(PROJECT_ROOT / V4_ARTIFACT_ROOT), "--smoke"]) == 2
    assert v41_main(["--output-dir", str(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3"), "--smoke"]) == 2
    completed = tmp_path / "qa_chunk_selection_v4_1"
    (completed / "r1_v6_oof").mkdir(parents=True)
    (completed / "r1_v6_oof" / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        refuse_completed_output(completed, allow=False)
    assert v41_main(["--output-dir", str(completed), "--smoke"]) == 2


def test_paired_delta_uses_question_and_seed() -> None:
    baseline = [
        {"question_id": "q1", "document_id": "d1", "seed": 42, "token_f1": 10.0},
        {"question_id": "q1", "document_id": "d1", "seed": 43, "token_f1": 30.0},
        {"question_id": "q2", "document_id": "d2", "seed": 42, "token_f1": 50.0},
    ]
    candidate = [
        {"question_id": "q1", "document_id": "d1", "seed": 42, "token_f1": 20.0},
        {"question_id": "q1", "document_id": "d1", "seed": 43, "token_f1": 40.0},
        {"question_id": "q2", "document_id": "d2", "seed": 42, "token_f1": 50.0},
    ]
    identity = paired_delta(baseline, baseline, n_boot=20, seed=0)
    assert identity["win"] == 0 and identity["lose"] == 0
    assert abs(float(identity["mean_delta"] or 0.0)) <= 1e-12
    delta = paired_delta(baseline, candidate, n_boot=20, seed=0)
    assert delta["n_rows"] == 3
    assert delta["n_questions"] == 2
    assert delta["pairing"] == "question_id+seed"
    assert float(delta["mean_delta"]) > 0


def test_shortlist_isolation_outside_topk_cannot_win() -> None:
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    ids = [int(chunk.chunk_id) for chunk in item.chunks]
    b0_row = {
        "ranked_ids": ids,
        "scores": {str(cid): float(10 - cid) for cid in ids},
    }
    k = 2
    shortlist = {ids[0]: 1.0, ids[1]: 2.0, ids[-1]: 999.0}
    ranked, scores = isolate_shortlist_ranking(item, b0_row, shortlist, k=k)
    assert ranked[0] != ids[-1]
    assert set(ranked[:k]) == set(ids[:k])
    assert max(scores, key=lambda cid: (scores[cid], -cid)) in set(ids[:k])


def test_v6_formula_keeps_v4_scale() -> None:
    score = v6_fused_score(
        {"b0_score": 1.0, "exact_freq": 2.0, "mean_token_f1": 80.0, "majority": 1.0, "source_clusters": 3.0},
        alpha=2.0,
        beta=0.25,
        gamma=0.5,
    )
    assert abs(score - (2.0 * 1.0 + 0.25 * (2.0 + 80.0 + 1.0) + 0.5 * 3.0)) < 1e-12


def test_rank_modules_do_not_read_gold() -> None:
    from experiments.qa_chunk_selection.v4_1 import features, ranking, select

    for module in (features, ranking, select):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "gold_answers" not in source
        assert "gold_spans" not in source


def test_parse_stages_aliases() -> None:
    assert parse_stages("primary")[0] == "init"
    assert "r5" not in parse_stages("primary")
    assert "r5" in parse_stages("full")
    assert "r1" in parse_stages("primary")
    with pytest.raises(ValueError):
        parse_stages("not-a-stage")


def test_v4_1_smoke_primary(tmp_path: Path) -> None:
    config = ExperimentConfigV41(
        output_dir=tmp_path / "qa_chunk_selection_v4_1",
        v3_artifact_dir=tmp_path / "missing_v3",
        v4_artifact_dir=tmp_path / "missing_v4",
        stages=parse_stages("primary"),
        smoke=True,
        n_bootstrap=20,
        v6_alpha_grid=(0.0, 2.0),
        v6_beta_grid=(0.25,),
        v6_gamma_grid=(0.5,),
        lambda_agg_grid=(1.0,),
        reuse_v4_cache=False,
    )
    assert PipelineV41(config).run(list(config.stages)) == 0
    root = config.output_dir
    assert (root / "progress.jsonl").exists()
    assert (root / "contract.json").exists()
    assert (root / "analysis" / "compare.json").exists()
    assert (root / "analysis" / "selection.json").exists()
    assert (root / "reports" / "results_v4_1.md").exists()
    assert (root / "r1_v6_oof" / "metrics.json").exists()
    assert (root / "final" / "freeze_manifest.json").exists()
    freeze = (root / "final" / "freeze_manifest.json").read_text(encoding="utf-8")
    assert "locked" in freeze
    progress = (root / "progress.jsonl").read_text(encoding="utf-8")
    assert "STAGE" in progress or "stage" in progress.lower()
    v4_tree = PROJECT_ROOT / V4_ARTIFACT_ROOT
    if v4_tree.exists():
        assert not any(path.name == "v4_1" for path in v4_tree.rglob("results_v4_1.md"))
