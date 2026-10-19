from __future__ import annotations

import numpy as np

import experiments.qa_chunk_selection_h3_shrinkage_ablation.models as shrinkage
from experiments.qa_chunk_selection_h3_shrinkage_ablation.models import (
    AGREEMENT_COLUMNS,
    AGREEMENT_PENALTY_GRID,
    LAMBDA_GRID,
    BlendPolicy,
    fit_blend_policy,
    fit_group_harm,
    fit_group_policy,
    fit_group_ridge,
)
from experiments.qa_chunk_selection_v8.contract import H2_COLUMNS, H3_COLUMNS
from experiments.qa_chunk_selection_v8.models import (
    ConstantClassifier,
    ConstantRegressor,
    fit_harm,
    fit_ridge,
    predict_harm,
    predict_reg,
)


def _feature_rows(count=32):
    rows = []
    for index in range(count):
        row = {
            name: float(((index * (offset + 2) + offset) % 13) - 6) / 6.0
            for offset, name in enumerate(H3_COLUMNS)
        }
        row.update(
            utility=float((index % 7) - 3),
            harm_target=int(index % 4 == 0),
            weight=1.0,
        )
        rows.append(row)
    return rows


def _policy_rows():
    rows = []
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
            rows.append(row)
    return rows


def _small_grid(monkeypatch):
    monkeypatch.setattr(shrinkage, "ALPHA_GRID", (1.0,))
    monkeypatch.setattr(shrinkage, "HARM_C_GRID", (0.1,))
    monkeypatch.setattr(shrinkage, "TAU_UTILITY_GRID", (0.0, 0.5))
    monkeypatch.setattr(shrinkage, "TAU_HARM_GRID", (0.2, 0.5))


def test_agreement_columns_are_exact_h3_extension():
    assert AGREEMENT_COLUMNS == tuple(name for name in H3_COLUMNS if name not in H2_COLUMNS)
    assert AGREEMENT_COLUMNS == ("delta_agreement", "agreement_x_margin", "fulllex_x_agreement")


def test_blend_prediction_has_exact_endpoints_and_midpoint():
    rows = [{name: 0.0 for name in H3_COLUMNS} for _ in range(3)]
    for value, expected_utility, expected_harm in ((0.0, 2.0, 0.2), (0.25, 3.0, 0.3), (1.0, 6.0, 0.6)):
        policy = BlendPolicy(
            ConstantRegressor(2.0),
            ConstantClassifier(0.2),
            ConstantRegressor(6.0),
            ConstantClassifier(0.6),
            value,
            {},
        )
        utility, harm = policy.predict(rows)
        assert np.allclose(utility, expected_utility)
        assert np.allclose(harm, expected_harm)


def test_group_penalty_one_exactly_reproduces_original_h3_models():
    rows = _feature_rows()
    original_ridge = fit_ridge(rows, H3_COLUMNS, 1.0)
    grouped_ridge = fit_group_ridge(rows, 1.0, 1.0)
    assert np.allclose(
        predict_reg(original_ridge, rows, H3_COLUMNS),
        predict_reg(grouped_ridge, rows, H3_COLUMNS),
        atol=1e-12,
    )
    original_harm = fit_harm(rows, H3_COLUMNS, 0.1)
    grouped_harm = fit_group_harm(rows, 0.1, 1.0)
    assert np.allclose(
        predict_harm(original_harm, rows, H3_COLUMNS),
        predict_harm(grouped_harm, rows, H3_COLUMNS),
        atol=1e-12,
    )


def test_both_train_oof_selectors_complete_on_synthetic_data(monkeypatch):
    _small_grid(monkeypatch)
    rows = _policy_rows()
    blend = fit_blend_policy(rows, seed=42, folds=3)
    grouped = fit_group_policy(rows, seed=42, folds=3)
    assert blend.lambda_value in LAMBDA_GRID
    assert blend.selection["selection_source"] == "META_PRED_OOF only"
    assert grouped.agreement_penalty in AGREEMENT_PENALTY_GRID
    assert grouped.selection["selection_source"] == "META_PRED_OOF only"
    assert blend.selection["policy_trials"]
    assert grouped.selection["policy_trials"]
