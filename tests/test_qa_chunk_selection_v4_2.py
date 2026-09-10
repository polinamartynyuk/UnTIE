from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v4.contract import V4_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_1.contract import V4_1_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_1.ranking import isolate_shortlist_ranking
from experiments.qa_chunk_selection.v4_2.attention import span_length_bias_probe, synthetic_compact
from experiments.qa_chunk_selection.v4_2.config import ExperimentConfigV42, parse_stages
from experiments.qa_chunk_selection.v4_2.contract import (
    PRIMARY_FINAL_SYSTEMS,
    V4_2_ARTIFACT_ROOT,
    experiment_contract_v4_2,
)
from experiments.qa_chunk_selection.v4_2.features import content_masks, zscore_map
from experiments.qa_chunk_selection.v4_2.heads import build_pools, union_topk
from experiments.qa_chunk_selection.v4_2.io_guard import assert_isolated_output, refuse_completed_output
from experiments.qa_chunk_selection.v4_2.pipeline import PipelineV42
from experiments.qa_chunk_selection.v4_2.ranking import att1_scores, rank_ids, relative_attention
from experiments.qa_chunk_selection.v4_2.run import main as v42_main


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
    "artifacts/qa_chunk_selection_v4_1",
)


def test_v4_2_contract_keeps_prior_readonly() -> None:
    contract = experiment_contract_v4_2()
    assert contract["v4_2_artifacts"] == "artifacts/qa_chunk_selection_v4_2"
    assert contract["v4_1_artifacts"] == "artifacts/qa_chunk_selection_v4_1"
    assert contract["v4_artifacts"] == "artifacts/qa_chunk_selection_v4"
    assert "never overwrite V4.1" in str(contract["prior_trees"])
    assert contract["do_not_retune_v6"] is True
    assert contract["internal_test"] == "locked"
    assert "ATT1" in PRIMARY_FINAL_SYSTEMS
    assert "ATT2" not in PRIMARY_FINAL_SYSTEMS
    from experiments.qa_chunk_selection.v4_2.contract import PROTECTED_WRITE_ROOTS

    assert V4_ARTIFACT_ROOT in PROTECTED_WRITE_ROOTS
    assert V4_1_ARTIFACT_ROOT in PROTECTED_WRITE_ROOTS
    assert V4_2_ARTIFACT_ROOT not in PROTECTED_WRITE_ROOTS


def test_v4_2_refuses_protected_writes_including_v4_and_v4_1(tmp_path: Path) -> None:
    for relative in PROTECTED:
        with pytest.raises(ValueError):
            assert_isolated_output(PROJECT_ROOT / relative / "oops")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v4_2")
    assert allowed.name == "qa_chunk_selection_v4_2"


def test_v4_2_cli_refuses_v4_v4_1_and_completed(tmp_path: Path) -> None:
    assert v42_main(["--output-dir", str(PROJECT_ROOT / V4_ARTIFACT_ROOT), "--smoke"]) == 2
    assert v42_main(["--output-dir", str(PROJECT_ROOT / V4_1_ARTIFACT_ROOT), "--smoke"]) == 2
    assert v42_main(["--output-dir", str(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3"), "--smoke"]) == 2
    completed = tmp_path / "qa_chunk_selection_v4_2"
    (completed / "att1").mkdir(parents=True)
    (completed / "att1" / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        refuse_completed_output(completed, allow=False)
    assert v42_main(["--output-dir", str(completed), "--smoke"]) == 2


def test_delta0_reproduces_v6_ranking() -> None:
    v6 = {0: 3.0, 1: 1.0, 2: 2.0}
    noisy = {0: 99.0, 1: -4.0, 2: 7.0}
    fused = att1_scores(v6, noisy, delta=0.0)
    assert rank_ids(list(v6), fused) == rank_ids(list(v6), v6)
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    ids = [int(chunk.chunk_id) for chunk in item.chunks]
    b0_row = {"ranked_ids": ids, "scores": {str(cid): float(10 - cid) for cid in ids}}
    short = {ids[0]: 4.0, ids[1]: 1.0, ids[2]: 2.0}
    ranked_v6, _ = isolate_shortlist_ranking(item, b0_row, short, k=3)
    ranked_d0, _ = isolate_shortlist_ranking(item, b0_row, att1_scores(short, {cid: 9.0 for cid in short}, delta=0.0), k=3)
    assert ranked_v6[0] == ranked_d0[0]
    assert set(ranked_d0[:3]) == set(ids[:3])


def test_zero_variance_shortlist_is_zero_not_nan() -> None:
    rel = relative_attention({1: 2.5, 2: 2.5, 3: 2.5})
    assert rel == {1: 0.0, 2: 0.0, 3: 0.0}
    z = zscore_map({10: 1.0})
    assert z[10] == 0.0
    assert all(value == value for value in rel.values())


def test_span_length_bias_and_special_masks() -> None:
    probe = span_length_bias_probe(seed=0)
    assert probe["ok"]
    assert probe["enrichment_near_uniform"]
    q = np.array([True, True, False, False, False])
    c = np.array([False, False, False, True, True])
    spec = np.array([True, False, True, False, True])
    q2, c2 = content_masks(q, c, spec)
    assert q2.tolist() == [False, True, False, False, False]
    assert c2.tolist() == [False, False, False, True, False]


def test_invalid_span_hn_excluded_from_head_pools() -> None:
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    hn = next(chunk for chunk in item.chunks if chunk.label == "hard_negative")
    pos = next(chunk for chunk in item.chunks if chunk.label == "positive")
    compact = {
        (item.question_id, int(pos.chunk_id)): {"valid_span": True, "n_layers": 1, "n_heads": 1},
        (item.question_id, int(hn.chunk_id)): {"valid_span": False, "n_layers": 1, "n_heads": 1},
    }
    union = {item.question_id: {int(pos.chunk_id), int(hn.chunk_id)}}
    flags = {(item.question_id, int(hn.chunk_id)): {"training_eligible_hn": True}}
    pools, audit = build_pools(
        [item],
        union,
        compact,
        flags,
        reader_easy_threshold=0.0,
        gold_f1=lambda _item, _cid: 100.0,
        fallback=False,
        b0_by_seed={42: {item.question_id: {"ranked_ids": [pos.chunk_id, hn.chunk_id]}}},
        seeds=(42,),
        k=5,
    )
    assert item.question_id not in pools
    assert audit["n_questions_with_valid_pos_and_HN"] == 0


def test_seed_union_dedupes_question_chunk() -> None:
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    ids = [int(chunk.chunk_id) for chunk in item.chunks]
    b0 = {
        42: {item.question_id: {"ranked_ids": ids}},
        43: {item.question_id: {"ranked_ids": list(reversed(ids))}},
    }
    union = union_topk([item], b0, (42, 43), k=2)
    assert item.question_id in union
    assert len(union[item.question_id]) == len(set(ids[:2]) | set(ids[-2:]))


def test_rank_modules_do_not_read_gold() -> None:
    from experiments.qa_chunk_selection.v4_2 import features, ranking, select

    for module in (features, ranking, select):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "gold_answers" not in source
        assert "gold_spans" not in source


def test_parse_stages_aliases() -> None:
    assert parse_stages("primary")[0] == "init"
    assert "att1" in parse_stages("primary")
    assert "report" in parse_stages("primary")
    with pytest.raises(ValueError):
        parse_stages("not-a-stage")


def test_synthetic_compact_has_no_raw_tensor() -> None:
    payload = synthetic_compact("q", "context tokens here", predicted_start=0, predicted_end=4, seed=1)
    assert "q_to_span_enrichment" in payload
    assert "q_to_span_token_mean" in payload
    assert "attentions" not in payload
    assert payload["feature_version"].startswith("v4.2")


def test_empty_span_features_are_zero() -> None:
    from experiments.qa_chunk_selection.v4_2.features import compact_head_features

    attn = np.ones((2, 2, 8, 8), dtype=float) / 8.0
    q_idx = np.arange(0, 3)
    c_idx = np.arange(4, 8)
    empty = compact_head_features(attn, q_idx=q_idx, c_idx=c_idx, s_idx=np.array([], dtype=int))
    assert empty["span_length_tokens"] == 0
    assert empty["valid_span"] is False
    assert float(np.mean(empty["q_to_span_enrichment"])) == 0.0
    assert empty["q_to_span_token_mean"][0][0] == []


def test_idf_uses_per_token_mass_not_mean_scale() -> None:
    from experiments.qa_chunk_selection.v4_2.features import compact_head_features, idf_q_to_span_enrichment

    n, layers, heads = 8, 1, 1
    attn = np.zeros((layers, heads, n, n), dtype=float)
    q_idx = np.array([0, 1])
    c_idx = np.array([3, 4, 5, 6])
    s_idx = np.array([3, 4])
    attn[:, :, 0, 3] = 1.0
    attn[:, :, 1, 3] = 1.0
    feat = compact_head_features(attn, q_idx=q_idx, c_idx=c_idx, s_idx=s_idx)
    feat["span_word_keys"] = ["rare", "common"]
    idf = {"rare": 4.0, "common": 1.0}
    weighted = idf_q_to_span_enrichment(feat, idf)
    unweighted = float(feat["q_to_span_enrichment"][0][0])
    mean_scaled = unweighted * ((4.0 + 1.0) / 2.0)
    assert weighted[0][0] > unweighted
    assert abs(weighted[0][0] - mean_scaled) > 0.05


def test_cal_does_not_select_hn_violating_high_f1() -> None:
    from experiments.qa_chunk_selection.v4_2.select import choose_by_cal

    def predict(params):
        delta = float(params["delta"])
        hn = 0.20 if delta > 0 else 0.05
        f1 = 90.0 if delta > 0 else 10.0
        return [
            {"question_id": "q1", "seed": 42, "token_f1": f1, "hn@1": hn, "mrr": 0.0},
            {"question_id": "q1", "seed": 43, "token_f1": f1, "hn@1": hn, "mrr": 0.0},
        ]

    choice = choose_by_cal(
        [],
        grid=[{"delta": 0.0}, {"delta": 1.0}],
        predict_rows=predict,
        b0_hn=0.05,
        epsilon_hn=0.02,
        prefer_smaller=("delta",),
    )
    assert choice["selected"]["delta"] == 0.0
    assert choice["n_admissible"] == 1


def test_v4_2_smoke_primary(tmp_path: Path) -> None:
    config = ExperimentConfigV42(
        output_dir=tmp_path / "qa_chunk_selection_v4_2",
        v3_artifact_dir=tmp_path / "missing_v3",
        v4_artifact_dir=tmp_path / "missing_v4",
        v4_1_artifact_dir=tmp_path / "missing_v4_1",
        stages=parse_stages("primary"),
        smoke=True,
        n_bootstrap=20,
        n_head_grid=(1, 2),
        delta_grid=(0.0, 0.5),
        core_majority_grid=(0.5, 1.0),
        core_token_grid=(0.0, 0.5),
        min_paired_questions=1,
        reader_easy_threshold=0.0,
        reuse_v4_1_qa_cache=False,
    )
    assert PipelineV42(config).run(list(config.stages)) == 0
    root = config.output_dir
    assert (root / "progress.jsonl").exists()
    assert (root / "contract.json").exists()
    assert (root / "analysis" / "compare.json").exists()
    assert (root / "analysis" / "selection.json").exists()
    assert (root / "reports" / "results_v4_2.md").exists()
    assert (root / "att1" / "metrics.json").exists()
    assert (root / "core" / "oof_core1.jsonl").exists()
    assert (root / "final" / "freeze_manifest.json").exists()
    freeze = (root / "final" / "freeze_manifest.json").read_text(encoding="utf-8")
    assert "locked" in freeze
    assert "v6_retuned" in freeze
    progress = (root / "progress.jsonl").read_text(encoding="utf-8")
    assert "STAGE" in progress
    for prior in (PROJECT_ROOT / V4_ARTIFACT_ROOT, PROJECT_ROOT / V4_1_ARTIFACT_ROOT):
        if prior.exists():
            assert not any(path.name == "results_v4_2.md" for path in prior.rglob("results_v4_2.md"))
