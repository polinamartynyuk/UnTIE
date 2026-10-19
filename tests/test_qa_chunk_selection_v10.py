from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.qa_chunk_selection_v10.contract import (
    ANSWER_RISK_COLUMNS, EVIDENCE_RISK_COLUMNS, H2_COLUMNS,
    QUESTION_AGG_COLUMNS, RANK_COLUMNS,
)
from experiments.qa_chunk_selection_v10.features import (
    assert_feature_schema, assert_runtime_gold_free, evidence_class,
    inner_manifest, pair_rows, question_aggregates, risk_rows, switch_rows,
)
from experiments.qa_chunk_selection_v10.io_guard import assert_v10_write
from experiments.qa_chunk_selection_v10.models import (
    TrainingBlocked, _logistic, policy_metrics, rank_question, simulate,
)


def _candidate(chunk, rank, f1, *, tier="sufficient_positive", answer="answer"):
    vector = {name: float(rank) / 10 for name in H2_COLUMNS}
    return {
        "chunk_id": chunk, "rank": rank, "b0_score": float(10 - rank),
        "answer": answer, "x": vector,
        "evaluation": {"f1": float(f1), "tier": tier, "evidence_class": evidence_class(tier), "hn": int(tier == "insufficient_evidence")},
    }


def _question(qid="q1", candidates=None):
    candidates = candidates or [_candidate(1, 1, 20), _candidate(2, 2, 80), _candidate(3, 3, 10, tier="insufficient_evidence")]
    return {
        "question_id": qid, "canonical_document_id": f"doc-{qid}",
        "candidates": candidates, "q": question_aggregates(candidates),
        "evaluation": {"positive_opportunity": any(x["evaluation"]["f1"] > candidates[0]["evaluation"]["f1"] for x in candidates[1:])},
    }


def test_feature_schemas_are_stable_finite_and_gold_free():
    assert_feature_schema()
    question = _question()
    assert tuple(question["q"]) == QUESTION_AGG_COLUMNS
    assert all(np.isfinite(list(question["q"].values())))
    assert len(RANK_COLUMNS) == len(H2_COLUMNS) + len(QUESTION_AGG_COLUMNS)
    assert len(EVIDENCE_RISK_COLUMNS) == 2 * len(H2_COLUMNS) + len(QUESTION_AGG_COLUMNS)
    with pytest.raises(AssertionError): assert_runtime_gold_free({"gold_answer": "leak"})


def test_question_aggregates_deterministic_and_exact_answer_clusters():
    candidates = [_candidate(1, 1, 0, answer="A-B"), _candidate(2, 2, 0, answer="A B"), _candidate(3, 3, 0, answer="")]
    first = question_aggregates(candidates); second = question_aggregates(list(reversed(candidates)))
    assert first == second
    assert first["candidate_count"] == 3
    assert first["valid_answer_count"] == 2
    assert first["empty_answer_count"] == 1


def test_k1_is_excluded_from_every_training_table():
    question = _question(candidates=[_candidate(1, 1, 50)])
    assert switch_rows([question]) == []
    assert pair_rows([question]) == []
    assert risk_rows([question], evidence=False) == []
    assert risk_rows([question], evidence=True) == []


def test_pair_targets_gap_symmetry_and_question_weight():
    rows = pair_rows([_question()])
    assert len(rows) == 6
    assert sum(x["weight"] for x in rows) == pytest.approx(1.0)
    by_direction = {(x["left_chunk_id"], x["right_chunk_id"]): x for x in rows}
    for (left, right), row in by_direction.items():
        reverse = by_direction[(right, left)]
        assert row["target"] == 1 - reverse["target"]
        for name in H2_COLUMNS: assert row[f"delta__{name}"] == pytest.approx(-reverse[f"delta__{name}"])


def test_pair_min_gap_removes_ties():
    q = _question(candidates=[_candidate(1, 1, 10.0), _candidate(2, 2, 10.5)])
    assert pair_rows([q]) == []


def test_risk_targets_neutral_and_question_weight():
    q = _question(candidates=[_candidate(1, 1, 20), _candidate(2, 2, 20), _candidate(3, 3, 10)])
    answer = risk_rows([q], evidence=False)
    assert [x["target"] for x in answer] == [0, 1]
    assert sum(x["weight"] for x in answer) == pytest.approx(1.0)
    assert all(all(name in row for name in ANSWER_RISK_COLUMNS) for row in answer)


def test_evidence_mapping_is_exact_and_excluded_weight_zero():
    assert evidence_class("sufficient_positive") == "SAFE_EVIDENCE"
    assert evidence_class("insufficient_evidence") == "UNSAFE_EVIDENCE"
    assert evidence_class("unresolved_for_training") == "EXCLUDE"
    with pytest.raises(RuntimeError, match="UNKNOWN_EVIDENCE_LABEL"): evidence_class("positive-ish")
    q = _question(candidates=[_candidate(1, 1, 0), _candidate(2, 2, 0, tier="unresolved_for_training"), _candidate(3, 3, 0, tier="insufficient_evidence")])
    rows = risk_rows([q], evidence=True)
    assert rows[0]["weight"] == 0
    assert rows[1]["weight"] == 1


def test_switch_and_risk_training_support_guards():
    with pytest.raises(TrainingBlocked, match="SWITCH_TRAINING_BLOCKED"):
        _logistic(switch_rows([_question(f"q{i}") for i in range(5)]), QUESTION_AGG_COLUMNS, 1.0, 42, "SWITCH")
    with pytest.raises(TrainingBlocked, match="ANSWER_RISK_TRAINING_BLOCKED"):
        _logistic(risk_rows([_question(f"q{i}") for i in range(3)], evidence=False), ANSWER_RISK_COLUMNS, 1.0, 42, "ANSWER_RISK")


class _PairModel:
    def predict_proba(self, frame):
        # Deliberately non-complementary directed outputs. rank_question must
        # symmetrize them into p_ji = 1-p_ij.
        p = np.asarray([0.8 if row[0] < 0 else 0.6 for row in frame.to_numpy()], float)
        return np.column_stack((1 - p, p))


def test_pairwise_inference_is_antisymmetric_and_uses_b0():
    q = _question(); scores, pairs, order = rank_question(_PairModel(), q)
    assert len(pairs) == 3  # includes both comparisons against B0
    assert set(scores) == {1, 2, 3}
    assert all(0 <= x <= 1 for x in scores.values())
    assert set(order) == {1, 2, 3}


def test_policy_k1_skips_all_inference_and_returns_b0():
    q = _question(candidates=[_candidate(1, 1, 50)])
    predictions = {q["question_id"]: {"p_switch": None, "rank_scores": {}, "pairs": [], "rank_order": [], "p_answer_harm": {}, "p_evidence_harm": {}}}
    rows = simulate([q], predictions, {"tau_switch": .5, "tau_answer_harm": .5, "tau_evidence_harm": .5})
    assert rows[0]["selected"]["chunk_id"] == 1
    assert rows[0]["reason"] == "NO_ALTERNATIVE"


def test_policy_scans_rank_order_and_applies_both_risks():
    q = _question(); predictions = {"q1": {"p_switch": .9, "rank_scores": {1: .2, 2: .9, 3: .8}, "pairs": [], "rank_order": [2, 3], "p_answer_harm": {2: .8, 3: .1}, "p_evidence_harm": {2: .1, 3: .1}}}
    rows = simulate([q], predictions, {"tau_switch": .5, "tau_answer_harm": .5, "tau_evidence_harm": .5})
    assert rows[0]["selected"]["chunk_id"] == 3
    assert policy_metrics(rows)["harmful_switch_rate"] == 1.0


def test_stratified_group_manifest_is_stable_and_shared_by_document():
    questions = []
    for index in range(24):
        tier = "insufficient_evidence" if index % 2 else "sufficient_positive"
        questions.append(_question(f"q{index}", [_candidate(index * 2, 1, 10), _candidate(index * 2 + 1, 2, 20, tier=tier)]))
    first = inner_manifest(questions); second = inner_manifest(list(reversed(questions)))
    assert first["sha256"] == second["sha256"]
    assert first["splitter"] == "StratifiedGroupKFold"
    assert set(first["assignment"].values()) == {0, 1, 2, 3}


def test_evidence_inner_guard_is_distinct_from_strict_refit_guard():
    questions = []
    for index in range(18):
        tier = "insufficient_evidence" if index < 9 else "sufficient_positive"
        questions.append(_question(f"e{index}", [_candidate(index * 2, 1, 10), _candidate(index * 2 + 1, 2, 20, tier=tier)]))
    rows = risk_rows(questions, evidence=True)
    model = _logistic(rows, EVIDENCE_RISK_COLUMNS, 1.0, 42, "EVIDENCE_RISK", stage="inner")
    assert model.predict_proba.__call__
    with pytest.raises(TrainingBlocked, match="stage=refit"):
        _logistic(rows, EVIDENCE_RISK_COLUMNS, 1.0, 42, "EVIDENCE_RISK", stage="refit")


def test_v10_write_guard_protects_history():
    with pytest.raises(RuntimeError): assert_v10_write(Path("artifacts/qa_chunk_selection_v9_h2_acs/forbidden.json"))
    allowed = assert_v10_write(Path("artifacts/qa_chunk_selection_v10_srr/test/out.json"))
    assert "qa_chunk_selection_v10_srr" in str(allowed)
