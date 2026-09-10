from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v4.contract import V4_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_1.contract import V4_1_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_1.ranking import isolate_shortlist_ranking as isolate
from experiments.qa_chunk_selection.v4_2.contract import V4_2_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_3.config import ExperimentConfigV43, parse_stages
from experiments.qa_chunk_selection.v4_3.contract import (
    ALWAYS_TAU,
    NEVER_TAU,
    PRIMARY_FINAL_SYSTEMS,
    V4_3_ARTIFACT_ROOT,
    experiment_contract_v4_3,
)
from experiments.qa_chunk_selection.v4_3.encoder import HashEncoder, SemanticEncoder
from experiments.qa_chunk_selection.v4_3.features import semantic_pack, support_gain, support_rel
from experiments.qa_chunk_selection.v4_3.io_guard import assert_isolated_output, refuse_completed_output
from experiments.qa_chunk_selection.v4_3.pipeline import PipelineV43
from experiments.qa_chunk_selection.v4_3.ranking import (
    adapt_use_semcore,
    isolate_with_b0_tiebreak,
    rank_ids_b0_tiebreak,
    routing_available,
    sem1_scores,
)
from experiments.qa_chunk_selection.v4_3.run import main as v43_main
from experiments.qa_chunk_selection.v4_3.select import choose_admissible_cal, choose_max_f1
from experiments.qa_chunk_selection.v4_3.text import cosine, is_valid_answer, l2_normalize, local_evidence_sentence, semantic_normalize
from experiments.qa_chunk_selection.v4_2.features import zscore_map

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
    "artifacts/qa_chunk_selection_v4_2",
)


def test_v4_3_contract_keeps_prior_readonly() -> None:
    contract = experiment_contract_v4_3()
    assert contract["v4_3_artifacts"] == "artifacts/qa_chunk_selection_v4_3"
    assert contract["v4_2_artifacts"] == "artifacts/qa_chunk_selection_v4_2"
    assert "never overwrite V4.2" in str(contract["prior_trees"])
    assert contract["do_not_retune_v6"] is True
    assert contract["internal_test"] == "locked"
    assert contract["encoder_frozen"] is True
    assert "SEM1" in PRIMARY_FINAL_SYSTEMS
    assert "SEMCORE" not in PRIMARY_FINAL_SYSTEMS
    from experiments.qa_chunk_selection.v4_3.contract import PROTECTED_WRITE_ROOTS

    assert V4_ARTIFACT_ROOT in PROTECTED_WRITE_ROOTS
    assert V4_1_ARTIFACT_ROOT in PROTECTED_WRITE_ROOTS
    assert V4_2_ARTIFACT_ROOT in PROTECTED_WRITE_ROOTS
    assert V4_3_ARTIFACT_ROOT not in PROTECTED_WRITE_ROOTS


def test_v4_3_refuses_protected_writes_including_v4_2(tmp_path: Path) -> None:
    for relative in PROTECTED:
        with pytest.raises(ValueError):
            assert_isolated_output(PROJECT_ROOT / relative / "oops")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v4_3")
    assert allowed.name == "qa_chunk_selection_v4_3"


def test_v4_3_cli_refuses_prior_and_completed(tmp_path: Path) -> None:
    assert v43_main(["--output-dir", str(PROJECT_ROOT / V4_ARTIFACT_ROOT), "--smoke"]) == 2
    assert v43_main(["--output-dir", str(PROJECT_ROOT / V4_1_ARTIFACT_ROOT), "--smoke"]) == 2
    assert v43_main(["--output-dir", str(PROJECT_ROOT / V4_2_ARTIFACT_ROOT), "--smoke"]) == 2
    completed = tmp_path / "qa_chunk_selection_v4_3"
    (completed / "sem1").mkdir(parents=True)
    (completed / "sem1" / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        refuse_completed_output(completed, allow=False)
    assert v43_main(["--output-dir", str(completed), "--smoke"]) == 2


def test_deterministic_embeddings_and_self_sim() -> None:
    enc = HashEncoder(dim=16)
    a = enc.encode(["graph neural network"])[0]
    b = enc.encode(["graph neural network"])[0]
    c = enc.encode(["unrelated answer"])[0]
    assert np.allclose(a, b)
    assert abs(cosine(a, a) - 1.0) < 1e-6
    assert cosine(a, c) < 0.999
    unit = l2_normalize(a)
    assert abs(float(np.linalg.norm(unit)) - 1.0) < 1e-5


def test_identical_normalized_answers_share_embedding(tmp_path: Path) -> None:
    encoder = SemanticEncoder(cache_root=tmp_path / "cache", smoke=True)
    left = encoder.embed_one("  Graph   Neural Network ", store="answer")
    right = encoder.embed_one("Graph Neural Network", store="answer")
    assert np.allclose(left, right)
    assert semantic_normalize("  Graph   Neural Network ") == "Graph Neural Network"


def test_self_sim_excluded_from_semtop2() -> None:
    vecs = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.9, 0.1]),
        2: np.array([0.0, 1.0]),
        3: np.array([0.1, 0.9]),
    }
    for cid in list(vecs):
        vecs[cid] = l2_normalize(vecs[cid])
    pack = semantic_pack({cid: f"ans-{cid}" for cid in vecs}, vecs, top_ids=list(vecs))
    assert pack["semantic_available"] is True
    assert pack["self_sim_excluded"] is True
    # sim(i,i) exists in matrix but must not be used in SemTop2
    self_sim = pack["sim"]["0|0"]
    assert abs(self_sim - 1.0) < 1e-6
    assert float(pack["sem_top2"]["0"]) < 0.999


def test_fewer_than_three_valid_answers_disables_semantic() -> None:
    vecs = {0: l2_normalize(np.array([1.0, 0.0])), 1: l2_normalize(np.array([0.0, 1.0]))}
    pack = semantic_pack({0: "ok", 1: "also", 2: "!!!", 3: ""}, vecs, top_ids=[0, 1, 2, 3])
    assert pack["semantic_available"] is False
    assert pack["sem_rel"]["0"] == 0.0
    v6 = {0: 3.0, 1: 1.0, 2: 2.0}
    fused = sem1_scores(v6, {cid: 9.0 for cid in v6}, delta=0.0)
    assert list(fused) == list(v6)


def test_delta0_sem1_matches_v6_order() -> None:
    v6 = {0: 3.0, 1: 1.0, 2: 2.0}
    noisy = {0: 99.0, 1: -4.0, 2: 7.0}
    fused = sem1_scores(v6, noisy, delta=0.0)
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    ids = [int(chunk.chunk_id) for chunk in item.chunks]
    b0_row = {"ranked_ids": ids, "scores": {str(cid): float(10 - cid) for cid in ids}}
    ranked_v6, _ = isolate(item, b0_row, {ids[0]: 4.0, ids[1]: 1.0, ids[2]: 2.0}, k=3)
    ranked_d0, _ = isolate(item, b0_row, sem1_scores({ids[0]: 4.0, ids[1]: 1.0, ids[2]: 2.0}, {cid: 9.0 for cid in ids[:3]}, delta=0.0), k=3)
    assert ranked_v6[0] == ranked_d0[0]


def test_zero_semtop2_variance_is_zero_not_nan() -> None:
    rel = zscore_map({1: 0.5, 2: 0.5, 3: 0.5})
    assert rel == {1: 0.0, 2: 0.0, 3: 0.0}
    assert all(value == value for value in rel.values())


def test_reader_only_tie_break_uses_b0_rank() -> None:
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    ids = [int(chunk.chunk_id) for chunk in item.chunks]
    b0_row = {"ranked_ids": ids, "scores": {str(cid): float(len(ids) - i) for i, cid in enumerate(ids)}}
    tied = {cid: 1.0 for cid in ids[:3]}
    ranked = rank_ids_b0_tiebreak(ids[:3], tied, b0_row, k=3)
    assert ranked == ids[:3]
    isolated, _fused, is_tie = isolate_with_b0_tiebreak(item, b0_row, tied, k=3)
    assert is_tie is True
    assert isolated[0] == ids[0]


def test_adapt_valid_and_triple_fallbacks() -> None:
    assert routing_available(adapt_answers_ok=False, selected_cid=1, best_triple=[1, 2, 3], consensus_conf=0.4) is False
    assert routing_available(adapt_answers_ok=True, selected_cid=9, best_triple=[1, 2, 3], consensus_conf=0.4) is False
    assert routing_available(adapt_answers_ok=True, selected_cid=1, best_triple=[1, 2, 3], consensus_conf=0.4) is True
    assert adapt_use_semcore(confidence=0.9, tau=NEVER_TAU, available=True) is False
    assert adapt_use_semcore(confidence=0.9, tau=ALWAYS_TAU, available=True) is True
    assert adapt_use_semcore(confidence=0.9, tau=ALWAYS_TAU, available=False) is False
    assert adapt_use_semcore(confidence=0.2, tau=0.5, available=True) is False
    assert adapt_use_semcore(confidence=0.8, tau=0.5, available=True) is True


def test_support_gain_identity_and_zero_variance() -> None:
    base = {1: 0.2, 2: 0.4, 3: 0.1}
    ans = {1: 0.5, 2: 0.7, 3: 0.4}
    gain = support_gain(base, ans, [1, 2, 3])
    assert gain[1] == pytest.approx(0.3)
    assert gain[2] == pytest.approx(0.3)
    rel = support_rel(gain)
    assert rel == {1: 0.0, 2: 0.0, 3: 0.0}


def test_local_sentence_contains_span() -> None:
    text = "First sentence is background. We propose a GCN-based model here. Third sentence."
    start = text.index("GCN-based")
    end = start + len("GCN-based model")
    payload = local_evidence_sentence(text, start_char=start, end_char=end, predicted_answer="GCN-based model")
    assert "GCN-based model" in payload["text"]
    assert payload["span_in_sentence"] is True
    assert is_valid_answer("GCN-based model")
    assert not is_valid_answer("   ")
    assert not is_valid_answer("!!!")


def test_local_sentence_prefers_span_location_over_earlier_string_mention() -> None:
    text = "We mention GCN-based model early. Later we propose a GCN-based model here."
    start = text.rindex("GCN-based model")
    end = start + len("GCN-based model")
    payload = local_evidence_sentence(text, start_char=start, end_char=end, predicted_answer="GCN-based model")
    assert "Later we propose" in payload["text"]
    assert payload["span_in_sentence"] is True


def test_sem1_cal_tie_break_prefers_smaller_delta_before_hn() -> None:
    def predict(params):
        hn = 0.06 if float(params["delta"]) == 0.0 else 0.01
        return [{"question_id": "q", "seed": 42, "token_f1": 30.0, "hn@1": hn}]

    choice = choose_admissible_cal(
        [],
        grid=[{"delta": 0.0}, {"delta": 1.0}],
        predict_rows=predict,
        hn_ref=0.10,
        epsilon_hn=0.02,
        prefer_smaller=("delta",),
    )
    assert choice["selected"]["delta"] == 0.0
    assert choice["n_admissible"] == 2


def test_semcore_and_lexcore_share_smaller_lambda_m_first() -> None:
    tried = [
        {"lambda_m": 1.0, "lambda_s": 0.0, "cal_token_f1": 20.0, "cal_hn@1": 0.0},
        {"lambda_m": 0.5, "lambda_s": 1.0, "cal_token_f1": 20.0, "cal_hn@1": 0.0},
    ]
    choice = choose_max_f1(tried, prefer_smaller=("lambda_m", "lambda_s"))
    assert choice["selected"]["lambda_m"] == 0.5


def test_rank_modules_do_not_read_gold() -> None:
    from experiments.qa_chunk_selection.v4_3 import features, ranking, select, text

    for module in (features, ranking, select, text):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "gold_answers" not in source
        assert "gold_spans" not in source


def test_parse_stages_aliases() -> None:
    assert parse_stages("primary")[0] == "init"
    assert "sem1" in parse_stages("primary")
    assert "adapt1" in parse_stages("primary")
    assert "report" in parse_stages("primary")
    with pytest.raises(ValueError):
        parse_stages("not-a-stage")


def test_v4_3_smoke_primary(tmp_path: Path) -> None:
    config = ExperimentConfigV43(
        output_dir=tmp_path / "qa_chunk_selection_v4_3",
        v3_artifact_dir=tmp_path / "missing_v3",
        v4_artifact_dir=tmp_path / "missing_v4",
        v4_1_artifact_dir=tmp_path / "missing_v4_1",
        v4_2_artifact_dir=tmp_path / "missing_v4_2",
        stages=parse_stages("primary"),
        smoke=True,
        n_bootstrap=20,
        delta_sem_grid=(0.0, 0.5),
        lexcore_majority_grid=(0.5, 1.0),
        lexcore_token_grid=(0.0, 0.5),
        semcore_majority_grid=(0.5, 1.0),
        semcore_sem_grid=(0.0, 0.5),
        adapt_quantiles=(0.50, 0.80),
        context_delta_grid=(0.0, 0.25),
        reuse_v4_1_qa_cache=False,
        force_semctx=True,
    )
    assert PipelineV43(config).run(list(config.stages)) == 0
    root = config.output_dir
    assert (root / "progress.jsonl").exists()
    assert (root / "contract.json").exists()
    assert (root / "analysis" / "compare.json").exists()
    assert (root / "sem1" / "metrics.json").exists()
    assert (root / "sem1" / "run0" / "delta0_identity.json").exists()
    assert (root / "lexcore" / "oof_predictions.jsonl").exists()
    assert (root / "semcore" / "oof_predictions.jsonl").exists()
    assert (root / "adapt1" / "oof_predictions.jsonl").exists()
    assert (root / "adapt1" / "run0" / "always_identity.json").exists()
    assert (root / "adapt1" / "run0" / "never_identity.json").exists()
    assert (root / "analysis" / "b0_margin_diagnostic.json").exists()
    assert (root / "analysis" / "context_to_context.json").exists()
    assert (root / "final" / "freeze_manifest.json").exists()
    assert (root / "reports" / "results_v4_3.md").exists()
    freeze = (root / "final" / "freeze_manifest.json").read_text(encoding="utf-8")
    assert "locked" in freeze
    assert "v6_retuned" in freeze
    progress = (root / "progress.jsonl").read_text(encoding="utf-8")
    assert "STAGE" in progress
    for prior in (PROJECT_ROOT / V4_ARTIFACT_ROOT, PROJECT_ROOT / V4_1_ARTIFACT_ROOT, PROJECT_ROOT / V4_2_ARTIFACT_ROOT):
        if prior.exists():
            assert not any(path.name == "results_v4_3.md" for path in prior.rglob("results_v4_3.md"))
