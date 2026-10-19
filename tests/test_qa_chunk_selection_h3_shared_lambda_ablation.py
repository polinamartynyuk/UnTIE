from __future__ import annotations

import numpy as np

import experiments.qa_chunk_selection_h3_shared_lambda_ablation.models as shared
import experiments.qa_chunk_selection_h3_shrinkage_ablation.models as shrinkage
from experiments.qa_chunk_selection_h3_shared_lambda_ablation.models import (
    SYSTEM_SHARED,
    fit_shared_lambda_policies,
)
from experiments.qa_chunk_selection_v8.contract import H3_COLUMNS


def _rows():
    values = []
    for question in range(18):
        base_f1 = float((question % 5) * 10)
        for candidate in (1, 2):
            index = question * 2 + candidate
            row = {
                name: float(((index * (offset + 3) + offset) % 17) - 8) / 8.0
                for offset, name in enumerate(H3_COLUMNS)
            }
            candidate_f1 = base_f1 + (15.0 if candidate == 1 and question % 3 else -5.0)
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
            values.append(row)
    return values


def test_one_lambda_is_shared_by_all_seed_policies(monkeypatch):
    monkeypatch.setattr(shrinkage, "ALPHA_GRID", (1.0,))
    monkeypatch.setattr(shrinkage, "HARM_C_GRID", (0.1,))
    monkeypatch.setattr(shrinkage, "TAU_UTILITY_GRID", (0.0, 0.5))
    monkeypatch.setattr(shrinkage, "TAU_HARM_GRID", (0.2, 0.5))
    monkeypatch.setattr(shared, "LAMBDA_GRID", (0.0, 0.5, 1.0))
    policies, selection = fit_shared_lambda_policies(_rows(), (42, 43, 44), folds=3)
    selected = selection["selected_lambda"]
    assert set(policies) == {42, 43, 44}
    assert all(policy.lambda_value == selected for policy in policies.values())
    assert all(policy.system == SYSTEM_SHARED for policy in policies.values())
    assert len(selection["aggregate_trials"]) == 3
    for trial in selection["aggregate_trials"]:
        assert set(trial["per_seed"]) == {"42", "43", "44"}
        assert np.isfinite(trial["mean_token_f1"])
