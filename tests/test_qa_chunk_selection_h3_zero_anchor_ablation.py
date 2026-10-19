from __future__ import annotations

import numpy as np

from experiments.qa_chunk_selection_h3_zero_anchor_ablation.models import (
    AGREEMENT_COLUMNS,
    fit_harm,
    fit_ridge,
)
from experiments.qa_chunk_selection_v8.contract import H2_COLUMNS, H3_COLUMNS


def _rows():
    values = []
    for index in range(12):
        row = {name: float(((index + offset) % 5) - 2) for offset, name in enumerate(H3_COLUMNS)}
        row.update(
            utility=float((index % 4) - 1),
            harm_target=int(index % 3 == 0),
            weight=1.0,
        )
        values.append(row)
    return values


def _agreement_zero_transform(model):
    import pandas as pd

    transformer = model.named_steps["scale"].named_transformers_["agreement_zero_anchor"]
    zero = pd.DataFrame([[0.0] * len(AGREEMENT_COLUMNS)], columns=list(AGREEMENT_COLUMNS))
    return np.asarray(transformer.transform(zero)).reshape(-1)


def test_incremental_columns_are_exact_h3_minus_h2():
    assert AGREEMENT_COLUMNS == tuple(name for name in H3_COLUMNS if name not in H2_COLUMNS)
    assert AGREEMENT_COLUMNS == ("delta_agreement", "agreement_x_margin", "fulllex_x_agreement")


def test_raw_zero_has_zero_direct_transform_for_both_models():
    ridge = fit_ridge(_rows(), alpha=1.0)
    harm = fit_harm(_rows(), c_value=0.1)
    assert np.array_equal(_agreement_zero_transform(ridge), np.zeros(len(AGREEMENT_COLUMNS)))
    assert np.array_equal(_agreement_zero_transform(harm), np.zeros(len(AGREEMENT_COLUMNS)))


def test_h2_columns_keep_centered_standard_scaling():
    model = fit_ridge(_rows(), alpha=1.0)
    transformer = model.named_steps["scale"].named_transformers_["h2"]
    assert transformer.with_mean is True
    assert transformer.with_std is True
