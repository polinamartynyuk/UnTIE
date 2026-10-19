"""TRAIN-OOF tuning for two compact H3 shrinkage variants."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
    fit_harm as fit_original_harm,
    fit_ridge as fit_original_ridge,
    fold_for,
    metrics,
    predict_harm,
    predict_reg,
    simulate,
)


SYSTEM_BLEND = "H3-BLEND"
SYSTEM_GROUP = "H3-GROUP-RIDGE"
AGREEMENT_COLUMNS = tuple(name for name in H3_COLUMNS if name not in H2_COLUMNS)
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
AGREEMENT_PENALTY_GRID = (1.0, 2.0, 4.0, 8.0, 16.0)


def _frame(rows):
    import pandas as pd

    return pd.DataFrame(rows)


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


def _select_ridge(rows, columns, seed, folds, fit):
    weights = np.asarray([row["weight"] for row in rows], dtype=float)
    truth = np.asarray([row["utility"] for row in rows], dtype=float)
    trials = []
    for alpha in ALPHA_GRID:
        prediction, assignment = _oof(
            rows,
            seed,
            folds,
            lambda part, alpha=alpha: fit(part, alpha),
            lambda model, part: predict_reg(model, part, columns),
        )
        mae = float(np.average(np.abs(prediction - truth), weights=weights))
        trials.append((alpha, prediction, mae, assignment))
    best = min(value[2] for value in trials)
    alpha, prediction, mae, assignment = max(
        [value for value in trials if value[2] <= best + 0.05], key=lambda value: value[0]
    )
    return {
        "alpha": float(alpha),
        "prediction": prediction,
        "mae": mae,
        "assignment": assignment,
        "trials": [{"alpha": float(value[0]), "mae": value[2]} for value in trials],
    }


def _select_harm(rows, columns, seed, folds, fit):
    from sklearn.metrics import log_loss

    weights = np.asarray([row["weight"] for row in rows], dtype=float)
    target = [row["harm_target"] for row in rows]
    trials = []
    for c_value in HARM_C_GRID:
        prediction, _ = _oof(
            rows,
            seed,
            folds,
            lambda part, c_value=c_value: fit(part, c_value),
            lambda model, part: predict_harm(model, part, columns),
        )
        loss = float(
            log_loss(
                target,
                np.clip(prediction, 1e-8, 1 - 1e-8),
                sample_weight=weights,
                labels=[0, 1],
            )
        )
        trials.append((c_value, prediction, loss))
    best = min(value[2] for value in trials)
    c_value, prediction, loss = min(
        [value for value in trials if value[2] <= best + 0.001], key=lambda value: value[0]
    )
    return {
        "C": float(c_value),
        "prediction": prediction,
        "log_loss": loss,
        "trials": [{"C": float(value[0]), "log_loss": value[2]} for value in trials],
    }


def _threshold_trials(rows, utility, harm, **parameters):
    output = []
    for tau_utility in TAU_UTILITY_GRID:
        for tau_harm in TAU_HARM_GRID:
            score = metrics(simulate(rows, utility, harm, True, tau_utility, tau_harm))
            score.update(
                tau_utility=tau_utility,
                tau_harm=tau_harm,
                feasible=score["delta_hit1_pp"] >= -0.5 - 1e-12
                and score["delta_hn1_pp"] <= 1.0 + 1e-12,
                **parameters,
            )
            output.append(score)
    return output


def _fallback(rows, **parameters):
    score = metrics(simulate(rows, np.zeros(len(rows)), np.ones(len(rows)), True, math.inf, 0.0))
    score.update(
        tau_utility="inf",
        tau_harm=0.0,
        delta_f1=0.0,
        delta_hit1_pp=0.0,
        delta_hn1_pp=0.0,
        switch_rate=0.0,
        feasible=True,
        **parameters,
    )
    return score


@dataclass
class BlendPolicy:
    utility_h2: object
    harm_h2: object
    utility_h3: object
    harm_h3: object
    lambda_value: float
    selection: dict
    system: str = SYSTEM_BLEND

    def predict(self, rows):
        utility_h2 = predict_reg(self.utility_h2, rows, H2_COLUMNS)
        harm_h2 = predict_harm(self.harm_h2, rows, H2_COLUMNS)
        utility_h3 = predict_reg(self.utility_h3, rows, H3_COLUMNS)
        harm_h3 = predict_harm(self.harm_h3, rows, H3_COLUMNS)
        value = float(self.lambda_value)
        return utility_h2 + value * (utility_h3 - utility_h2), harm_h2 + value * (harm_h3 - harm_h2)


def fit_blend_policy(rows: list[dict], seed: int, folds: int = 4, progress=None) -> BlendPolicy:
    values = copy.deepcopy(rows)
    normalize_weights(values, pooled=False)
    documents = {row["canonical_document_id"] for row in values}
    folds = max(2, min(int(folds), 4, len(documents)))
    components = {}
    for label, columns in (("H2", H2_COLUMNS), ("H3", H3_COLUMNS)):
        if progress:
            progress("blend component", component=label)
        ridge = _select_ridge(
            values,
            columns,
            seed,
            folds,
            lambda part, alpha, columns=columns: fit_original_ridge(part, columns, alpha),
        )
        harm = _select_harm(
            values,
            columns,
            seed,
            folds,
            lambda part, c_value, columns=columns: fit_original_harm(part, columns, c_value),
        )
        components[label] = {"columns": columns, "ridge": ridge, "harm": harm}

    trials = []
    for value in LAMBDA_GRID:
        if progress:
            progress("blend lambda", lambda_value=value)
        utility = components["H2"]["ridge"]["prediction"] + value * (
            components["H3"]["ridge"]["prediction"] - components["H2"]["ridge"]["prediction"]
        )
        harm = components["H2"]["harm"]["prediction"] + value * (
            components["H3"]["harm"]["prediction"] - components["H2"]["harm"]["prediction"]
        )
        trials.extend(_threshold_trials(values, utility, harm, lambda_value=value))
    feasible = [trial for trial in trials if trial["feasible"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda trial: (
                trial["token_f1"],
                -trial["switch_rate"],
                trial["tau_utility"],
                -trial["tau_harm"],
                trial["lambda_value"],
            ),
        )
        fallback = False
    else:
        chosen = _fallback(values, lambda_value=1.0)
        fallback = True

    h2 = components["H2"]
    h3 = components["H3"]
    utility_h2 = fit_original_ridge(values, H2_COLUMNS, h2["ridge"]["alpha"])
    harm_h2 = fit_original_harm(values, H2_COLUMNS, h2["harm"]["C"])
    utility_h3 = fit_original_ridge(values, H3_COLUMNS, h3["ridge"]["alpha"])
    harm_h3 = fit_original_harm(values, H3_COLUMNS, h3["harm"]["C"])
    selection = {
        "system": SYSTEM_BLEND,
        "selection_source": "META_PRED_OOF only",
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_value": float(chosen["lambda_value"]),
        "tau_utility": chosen["tau_utility"],
        "tau_harm": chosen["tau_harm"],
        "fold_policy": "BASE_FALLBACK" if fallback else "ACTIVE",
        "selected_metrics": chosen,
        "policy_trials": trials,
        "components": {
            label: {
                "columns": list(component["columns"]),
                "alpha": component["ridge"]["alpha"],
                "ridge_oof_mae": component["ridge"]["mae"],
                "C_harm": component["harm"]["C"],
                "harm_oof_log_loss": component["harm"]["log_loss"],
                "ridge_trials": component["ridge"]["trials"],
                "harm_trials": component["harm"]["trials"],
            }
            for label, component in components.items()
        },
        "inner_folds_used": folds,
    }
    return BlendPolicy(
        utility_h2,
        harm_h2,
        utility_h3,
        harm_h3,
        float(chosen["lambda_value"]),
        selection,
    )


def apply_blend_policy(rows, policy: BlendPolicy):
    utility, harm = policy.predict(rows)
    selection = policy.selection
    tau_utility = math.inf if selection["tau_utility"] == "inf" else float(selection["tau_utility"])
    if selection["fold_policy"] == "BASE_FALLBACK":
        tau_utility = math.inf
    choices = simulate(rows, utility, harm, True, tau_utility, float(selection["tau_harm"]))
    return choices, (utility, harm)


class MultiplyBy(TransformerMixin, BaseEstimator):
    """Small joblib-safe sklearn transformer."""

    def __init__(self, factor=1.0):
        self.factor = float(factor)

    def fit(self, values, target=None):
        array = np.asarray(values)
        self.n_features_in_ = int(array.shape[1]) if array.ndim > 1 else 1
        return self

    def transform(self, values):
        return np.asarray(values, dtype=float) * self.factor


def _group_pipeline(estimator, agreement_penalty):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    agreement = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("penalty_scale", MultiplyBy(1.0 / math.sqrt(float(agreement_penalty)))),
        ]
    )
    transform = ColumnTransformer(
        [
            ("h2", StandardScaler(), list(H2_COLUMNS)),
            ("agreement", agreement, list(AGREEMENT_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("scale", transform), ("model", estimator)])


def fit_group_ridge(rows, alpha, agreement_penalty):
    from sklearn.linear_model import Ridge

    if not rows:
        return ConstantRegressor(0.0)
    target = np.asarray([row["utility"] for row in rows], dtype=float)
    if np.ptp(target) < 1e-12:
        return ConstantRegressor(float(target.mean()))
    model = _group_pipeline(Ridge(alpha=float(alpha)), agreement_penalty)
    model.fit(
        _frame(rows)[list(H3_COLUMNS)],
        target,
        model__sample_weight=np.asarray([row["weight"] for row in rows], dtype=float),
    )
    return model


def fit_group_harm(rows, c_value, agreement_penalty):
    from sklearn.linear_model import LogisticRegression

    if not rows:
        return ConstantClassifier(0.0)
    target = np.asarray([row["harm_target"] for row in rows], dtype=int)
    if len(set(target.tolist())) < 2:
        return ConstantClassifier(float(target[0]))
    model = _group_pipeline(
        LogisticRegression(C=float(c_value), solver="liblinear", class_weight=None, max_iter=3000),
        agreement_penalty,
    )
    model.fit(
        _frame(rows)[list(H3_COLUMNS)],
        target,
        model__sample_weight=np.asarray([row["weight"] for row in rows], dtype=float),
    )
    return model


@dataclass
class GroupPenaltyPolicy:
    utility_model: object
    harm_model: object
    agreement_penalty: float
    selection: dict
    system: str = SYSTEM_GROUP

    def predict(self, rows):
        return predict_reg(self.utility_model, rows, H3_COLUMNS), predict_harm(self.harm_model, rows, H3_COLUMNS)


def fit_group_policy(rows: list[dict], seed: int, folds: int = 4, progress=None) -> GroupPenaltyPolicy:
    values = copy.deepcopy(rows)
    normalize_weights(values, pooled=False)
    documents = {row["canonical_document_id"] for row in values}
    folds = max(2, min(int(folds), 4, len(documents)))
    candidates = []
    trials = []
    for penalty in AGREEMENT_PENALTY_GRID:
        if progress:
            progress("agreement penalty", agreement_penalty=penalty)
        ridge = _select_ridge(
            values,
            H3_COLUMNS,
            seed,
            folds,
            lambda part, alpha, penalty=penalty: fit_group_ridge(part, alpha, penalty),
        )
        harm = _select_harm(
            values,
            H3_COLUMNS,
            seed,
            folds,
            lambda part, c_value, penalty=penalty: fit_group_harm(part, c_value, penalty),
        )
        candidates.append({"agreement_penalty": penalty, "ridge": ridge, "harm": harm})
        trials.extend(
            _threshold_trials(
                values,
                ridge["prediction"],
                harm["prediction"],
                agreement_penalty=penalty,
            )
        )
    feasible = [trial for trial in trials if trial["feasible"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda trial: (
                trial["token_f1"],
                -trial["switch_rate"],
                trial["tau_utility"],
                -trial["tau_harm"],
                -trial["agreement_penalty"],
            ),
        )
        fallback = False
    else:
        chosen = _fallback(values, agreement_penalty=1.0)
        fallback = True
    selected = next(value for value in candidates if value["agreement_penalty"] == chosen["agreement_penalty"])
    penalty = float(chosen["agreement_penalty"])
    utility_model = fit_group_ridge(values, selected["ridge"]["alpha"], penalty)
    harm_model = fit_group_harm(values, selected["harm"]["C"], penalty)
    selection = {
        "system": SYSTEM_GROUP,
        "selection_source": "META_PRED_OOF only",
        "agreement_penalty_grid": list(AGREEMENT_PENALTY_GRID),
        "agreement_penalty": penalty,
        "tau_utility": chosen["tau_utility"],
        "tau_harm": chosen["tau_harm"],
        "fold_policy": "BASE_FALLBACK" if fallback else "ACTIVE",
        "selected_metrics": chosen,
        "policy_trials": trials,
        "candidate_diagnostics": [
            {
                "agreement_penalty": value["agreement_penalty"],
                "alpha": value["ridge"]["alpha"],
                "ridge_oof_mae": value["ridge"]["mae"],
                "C_harm": value["harm"]["C"],
                "harm_oof_log_loss": value["harm"]["log_loss"],
                "ridge_trials": value["ridge"]["trials"],
                "harm_trials": value["harm"]["trials"],
            }
            for value in candidates
        ],
        "inner_folds_used": folds,
    }
    return GroupPenaltyPolicy(utility_model, harm_model, penalty, selection)


def apply_group_policy(rows, policy: GroupPenaltyPolicy):
    utility, harm = policy.predict(rows)
    selection = policy.selection
    tau_utility = math.inf if selection["tau_utility"] == "inf" else float(selection["tau_utility"])
    if selection["fold_policy"] == "BASE_FALLBACK":
        tau_utility = math.inf
    choices = simulate(rows, utility, harm, True, tau_utility, float(selection["tau_harm"]))
    return choices, (utility, harm)


APPLIERS = {SYSTEM_BLEND: apply_blend_policy, SYSTEM_GROUP: apply_group_policy}
FITTERS = {SYSTEM_BLEND: fit_blend_policy, SYSTEM_GROUP: fit_group_policy}

