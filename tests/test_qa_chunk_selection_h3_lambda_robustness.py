from __future__ import annotations

import experiments.qa_chunk_selection_h3_lambda_robustness.models as robust
import experiments.qa_chunk_selection_h3_shrinkage_ablation.models as shrinkage
from experiments.qa_chunk_selection_h3_lambda_robustness.models import (
    SEEDS_10,
    SEEDS_3,
    build_policies,
    lambda_evidence,
    prepare_states,
    select_shared,
    stability_decision,
)
from experiments.qa_chunk_selection_v8.contract import H3_COLUMNS


def _rows():
    rows = []
    for question in range(15):
        base_f1 = float((question % 4) * 10)
        for candidate in (1, 2):
            index = question * 2 + candidate
            row = {
                name: float(((index * (offset + 3) + offset) % 17) - 8) / 8.0
                for offset, name in enumerate(H3_COLUMNS)
            }
            candidate_f1 = base_f1 + (10.0 if candidate == 1 and question % 3 else -5.0)
            row.update(
                canonical_document_id=f"doc-{question}",
                dataset_cell="synthetic",
                corpus_family="synthetic",
                question_id=f"q-{question}",
                question_lineage_id=f"lineage-{question}",
                chunk_id=candidate,
                base_chunk_id=0,
                b0_rank=candidate + 1,
                candidate_f1=candidate_f1,
                base_f1=base_f1,
                candidate_tier="sufficient_positive" if candidate_f1 > base_f1 else "ambiguous",
                base_tier="ambiguous",
                candidate_hn=int(candidate == 2 and question % 4 == 0),
                base_hn=0,
                utility=candidate_f1 - base_f1,
                harm_target=int(candidate_f1 < base_f1),
                weight=0.5,
            )
            rows.append(row)
    return rows


def _small(monkeypatch):
    monkeypatch.setattr(shrinkage, "ALPHA_GRID", (1.0,))
    monkeypatch.setattr(shrinkage, "HARM_C_GRID", (0.1,))
    monkeypatch.setattr(shrinkage, "TAU_UTILITY_GRID", (0.0, 0.5))
    monkeypatch.setattr(shrinkage, "TAU_HARM_GRID", (0.2, 0.5))
    monkeypatch.setattr(robust, "LAMBDA_GRID", (0.0, 0.5, 1.0))


def test_frozen_seed_sets_are_nested_and_have_ten_unique_values():
    assert SEEDS_3 == SEEDS_10[:3]
    assert len(SEEDS_10) == len(set(SEEDS_10)) == 10


def test_stability_gate_defaults_to_h3_on_mismatch_or_uncertainty():
    passed = {"gate_pass": True}
    uncertain = {"gate_pass": False}
    mismatch = stability_decision({"selected_lambda": 0.0}, {"selected_lambda": 0.5}, passed)
    assert mismatch["accepted_lambda"] == 1.0
    assert mismatch["reason"] == "UNSTABLE_TRAIN_REFIT_SELECTION"
    unstable = stability_decision({"selected_lambda": 0.0}, {"selected_lambda": 0.0}, uncertain)
    assert unstable["accepted_lambda"] == 1.0
    assert unstable["reason"] == "OOF_UNCERTAINTY_OVERLAPS_ZERO"
    accepted = stability_decision({"selected_lambda": 0.0}, {"selected_lambda": 0.0}, passed)
    assert accepted["accepted_lambda"] == 0.0
    assert accepted["gate_pass"] is True


def test_prepare_select_evidence_and_policy_build_complete(monkeypatch):
    _small(monkeypatch)
    states = prepare_states(_rows(), SEEDS_3, folds=3)
    selection = select_shared(states, SEEDS_3)
    evidence = lambda_evidence(states, SEEDS_3, selection["selected_lambda"], iterations=200)
    gate = stability_decision(selection, selection, evidence)
    policies = build_policies(
        states,
        SEEDS_3,
        gate["accepted_lambda"],
        "TEST-SYSTEM",
        selection,
        gate,
    )
    assert set(policies) == set(SEEDS_3)
    assert all(policy.lambda_value == gate["accepted_lambda"] for policy in policies.values())
    assert evidence["iterations"] == 200
