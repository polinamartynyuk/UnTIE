"""Exact H3 policy with zero-anchored scaling for agreement-only columns.

H2 columns retain the original StandardScaler behavior.  The three incremental
H3 agreement columns are divided by their training standard deviation without
mean subtraction, so an observed raw zero has exactly zero direct contribution.
"""
from __future__ import annotations

import copy
from collections import defaultdict
import math

import numpy as np

from experiments.qa_chunk_selection_v8.contract import (
    ALPHA_GRID,
    H2_COLUMNS,
    H3_COLUMNS,
    HARM_C_GRID,
    TAU_HARM_GRID,
    TAU_UTILITY_GRID,
)
from experiments.qa_chunk_selection_v8.features import normalize_weights
from experiments.qa_chunk_selection_v8.models import (
    ConstantClassifier,
    ConstantRegressor,
    Policy,
    fold_for,
    metrics,
    predict_harm,
    predict_reg,
    simulate,
)


AGREEMENT_COLUMNS = tuple(name for name in H3_COLUMNS if name not in H2_COLUMNS)
SYSTEM = "H3-ZERO-ANCHOR"


def _frame(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def _pipeline(estimator):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scale = ColumnTransformer(
        [
            ("h2", StandardScaler(), list(H2_COLUMNS)),
            ("agreement_zero_anchor", StandardScaler(with_mean=False), list(AGREEMENT_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("scale", scale), ("model", estimator)])


def fit_ridge(rows, alpha):
    from sklearn.linear_model import Ridge

    if not rows:
        return ConstantRegressor(0.0)
    target = np.asarray([row["utility"] for row in rows], dtype=float)
    if np.ptp(target) < 1e-12:
        return ConstantRegressor(float(target.mean()))
    model = _pipeline(Ridge(alpha=float(alpha)))
    model.fit(
        _frame(rows)[list(H3_COLUMNS)],
        target,
        model__sample_weight=np.asarray([row["weight"] for row in rows], dtype=float),
    )
    return model


def fit_harm(rows, c_value):
    from sklearn.linear_model import LogisticRegression

    if not rows:
        return ConstantClassifier(0.0)
    target = np.asarray([row["harm_target"] for row in rows], dtype=int)
    if len(set(target.tolist())) < 2:
        return ConstantClassifier(float(target[0]))
    model = _pipeline(
        LogisticRegression(C=float(c_value), solver="liblinear", class_weight=None, max_iter=3000)
    )
    model.fit(
        _frame(rows)[list(H3_COLUMNS)],
        target,
        model__sample_weight=np.asarray([row["weight"] for row in rows], dtype=float),
    )
    return model


def _oof(rows, seed, folds, fit, predict):
    assignment = [fold_for(row["canonical_document_id"], seed, folds) for row in rows]
    output = np.full(len(rows), np.nan)
    for held in sorted(set(assignment)):
        train = [row for row, fold in zip(rows, assignment) if fold != held]
        indexes = [index for index, fold in enumerate(assignment) if fold == held]
        if not train or not indexes:
            continue
        model = fit(train)
        output[indexes] = predict(model, [rows[index] for index in indexes])
    if np.isnan(output).any():
        raise RuntimeError("META_PRED_OOF coverage incomplete")
    return output, assignment


def _coefficient_map(model):
    if not hasattr(model, "named_steps"):
        return {name: 0.0 for name in H3_COLUMNS}
    values = np.asarray(model.named_steps["model"].coef_, dtype=float).reshape(-1)
    ordered = (*H2_COLUMNS, *AGREEMENT_COLUMNS)
    return {name: float(value) for name, value in zip(ordered, values)}


def _zero_anchor_audit(model):
    if not hasattr(model, "named_steps"):
        return {"agreement_transform_of_raw_zero": [0.0] * len(AGREEMENT_COLUMNS), "passed": True}
    transformer = model.named_steps["scale"].named_transformers_["agreement_zero_anchor"]
    import pandas as pd

    zero = pd.DataFrame([[0.0] * len(AGREEMENT_COLUMNS)], columns=list(AGREEMENT_COLUMNS))
    observed = np.asarray(transformer.transform(zero), dtype=float).reshape(-1)
    return {
        "agreement_transform_of_raw_zero": [float(value) for value in observed],
        "passed": bool(np.allclose(observed, 0.0, atol=0.0, rtol=0.0)),
    }


def fit_policy(rows: list[dict], seed: int, folds: int = 4) -> Policy:
    values = copy.deepcopy(rows)
    normalize_weights(values, pooled=False)
    documents = {row["canonical_document_id"] for row in values}
    folds = max(2, min(int(folds), 4, len(documents)))
    weights = np.asarray([row["weight"] for row in values], dtype=float)
    truth = np.asarray([row["utility"] for row in values], dtype=float)

    ridge_trials = []
    for alpha in ALPHA_GRID:
        predicted, assignment = _oof(
            values,
            seed,
            folds,
            lambda part, alpha=alpha: fit_ridge(part, alpha),
            lambda model, part: predict_reg(model, part, H3_COLUMNS),
        )
        mae = float(np.average(np.abs(predicted - truth), weights=weights))
        ridge_trials.append((alpha, predicted, mae, assignment))
    best_mae = min(value[2] for value in ridge_trials)
    alpha, utility_oof, mae, assignment = max(
        [value for value in ridge_trials if value[2] <= best_mae + 0.05], key=lambda value: value[0]
    )

    from sklearn.metrics import log_loss

    harm_trials = []
    for c_value in HARM_C_GRID:
        predicted, _ = _oof(
            values,
            seed,
            folds,
            lambda part, c_value=c_value: fit_harm(part, c_value),
            lambda model, part: predict_harm(model, part, H3_COLUMNS),
        )
        loss = float(
            log_loss(
                [row["harm_target"] for row in values],
                np.clip(predicted, 1e-8, 1 - 1e-8),
                sample_weight=weights,
                labels=[0, 1],
            )
        )
        harm_trials.append((c_value, predicted, loss))
    best_loss = min(value[2] for value in harm_trials)
    c_harm, harm_oof, loss = min(
        [value for value in harm_trials if value[2] <= best_loss + 0.001], key=lambda value: value[0]
    )

    threshold_trials = []
    for tau_utility in TAU_UTILITY_GRID:
        for tau_harm in TAU_HARM_GRID:
            score = metrics(simulate(values, utility_oof, harm_oof, True, tau_utility, tau_harm))
            score.update(
                tau_utility=tau_utility,
                tau_harm=tau_harm,
                feasible=score["delta_hit1_pp"] >= -0.5 - 1e-12
                and score["delta_hn1_pp"] <= 1.0 + 1e-12,
            )
            threshold_trials.append(score)
    feasible = [value for value in threshold_trials if value["feasible"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda value: (
                value["token_f1"],
                -value["switch_rate"],
                value["tau_utility"],
                -value["tau_harm"],
            ),
        )
        fallback = False
    else:
        chosen = {
            "tau_utility": "inf",
            "tau_harm": 0.0,
            "token_f1": metrics(simulate(values, utility_oof, harm_oof, True, math.inf, 0.0))["token_f1"],
            "delta_f1": 0.0,
            "delta_hit1_pp": 0.0,
            "delta_hn1_pp": 0.0,
            "switch_rate": 0.0,
            "feasible": True,
        }
        fallback = True

    utility_model = fit_ridge(values, alpha)
    harm_model = fit_harm(values, c_harm)
    utility_audit = _zero_anchor_audit(utility_model)
    harm_audit = _zero_anchor_audit(harm_model)
    if not utility_audit["passed"] or not harm_audit["passed"]:
        raise AssertionError("zero-anchor invariant failed")
    selection = {
        "ablation": SYSTEM,
        "only_change": "agreement columns use StandardScaler(with_mean=False)",
        "alpha": alpha,
        "ridge_oof_mae": mae,
        "C_harm": c_harm,
        "harm_oof_log_loss": loss,
        "tau_utility": chosen["tau_utility"],
        "tau_harm": chosen["tau_harm"],
        "fold_policy": "BASE_FALLBACK" if fallback else "ACTIVE",
        "selected_metrics": chosen,
        "ridge_trials": [{"alpha": value[0], "mae": value[2]} for value in ridge_trials],
        "harm_trials": [{"C": value[0], "log_loss": value[2]} for value in harm_trials],
        "policy_trials": threshold_trials,
        "meta_fold": assignment,
        "inner_folds_used": folds,
        "selection_source": "META_PRED_OOF",
        "standardized_coefficients": _coefficient_map(utility_model),
        "zero_anchor_audit": {"utility": utility_audit, "harm": harm_audit},
    }
    return Policy(SYSTEM, tuple(H3_COLUMNS), utility_model, harm_model, selection)

