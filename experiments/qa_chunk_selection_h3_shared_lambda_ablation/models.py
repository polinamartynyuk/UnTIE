"""Select one H2/H3 blend coefficient per dataset across all model seeds."""
from __future__ import annotations

import copy

import numpy as np

from experiments.qa_chunk_selection_v8.contract import H2_COLUMNS, H3_COLUMNS
from experiments.qa_chunk_selection_v8.features import normalize_weights
from experiments.qa_chunk_selection_v8.models import fit_harm, fit_ridge
from experiments.qa_chunk_selection_h3_shrinkage_ablation.models import (
    BlendPolicy,
    LAMBDA_GRID,
    _fallback,
    _select_harm,
    _select_ridge,
    _threshold_trials,
    apply_blend_policy,
)


SYSTEM_SHARED = "H3-SHARED-LAMBDA"


def _seed_rows(rows: list[dict], seed: int) -> list[dict]:
    values = copy.deepcopy(rows)
    for row in values:
        row["seed"] = int(seed)
    normalize_weights(values, pooled=False)
    return values


def _best_threshold(rows, trials: list[dict], lambda_value: float) -> dict:
    feasible = [value for value in trials if value["feasible"]]
    if not feasible:
        return _fallback(rows, lambda_value=lambda_value) | {"fallback": True}
    chosen = max(
        feasible,
        key=lambda value: (
            value["token_f1"],
            -value["switch_rate"],
            value["tau_utility"],
            -value["tau_harm"],
        ),
    )
    return dict(chosen) | {"fallback": False}


def fit_shared_lambda_policies(
    rows: list[dict],
    seeds: tuple[int, ...],
    folds: int = 4,
    progress=None,
) -> tuple[dict[int, BlendPolicy], dict]:
    """Fit seed-specific H2/H3 models but choose one lambda across seeds.

    For every seed, ridge/logistic hyperparameters and thresholds remain
    seed-specific.  For each lambda, the best feasible threshold pair is found
    independently inside that seed's grouped OOF.  Lambda is then selected by
    mean OOF Token F1 across all seeds, with lower mean switching as the first
    tie-break and the original H3 endpoint as the final tie-break.
    """
    if not rows:
        raise ValueError("no training alternatives")
    if not seeds:
        raise ValueError("no seeds")

    seed_state = {}
    for seed in seeds:
        values = _seed_rows(rows, seed)
        documents = {row["canonical_document_id"] for row in values}
        used_folds = max(2, min(int(folds), 4, len(documents)))
        components = {}
        for label, columns in (("H2", H2_COLUMNS), ("H3", H3_COLUMNS)):
            if progress:
                progress("shared-lambda component", seed=seed, component=label)
            ridge = _select_ridge(
                values,
                columns,
                seed,
                used_folds,
                lambda part, alpha, columns=columns: fit_ridge(part, columns, alpha),
            )
            harm = _select_harm(
                values,
                columns,
                seed,
                used_folds,
                lambda part, c_value, columns=columns: fit_harm(part, columns, c_value),
            )
            components[label] = {"columns": columns, "ridge": ridge, "harm": harm}

        by_lambda = {}
        for lambda_value in LAMBDA_GRID:
            h2, h3 = components["H2"], components["H3"]
            utility = h2["ridge"]["prediction"] + lambda_value * (
                h3["ridge"]["prediction"] - h2["ridge"]["prediction"]
            )
            harm = h2["harm"]["prediction"] + lambda_value * (
                h3["harm"]["prediction"] - h2["harm"]["prediction"]
            )
            trials = _threshold_trials(values, utility, harm, lambda_value=lambda_value)
            by_lambda[lambda_value] = {
                "chosen": _best_threshold(values, trials, lambda_value),
                "trials": trials,
            }
        seed_state[seed] = {
            "rows": values,
            "folds": used_folds,
            "components": components,
            "by_lambda": by_lambda,
        }

    aggregate_trials = []
    for lambda_value in LAMBDA_GRID:
        selected = [seed_state[seed]["by_lambda"][lambda_value]["chosen"] for seed in seeds]
        aggregate_trials.append(
            {
                "lambda_value": float(lambda_value),
                "mean_token_f1": float(np.mean([value["token_f1"] for value in selected])),
                "mean_delta_f1": float(np.mean([value["delta_f1"] for value in selected])),
                "mean_switch_rate": float(np.mean([value["switch_rate"] for value in selected])),
                "mean_delta_hit1_pp": float(np.mean([value["delta_hit1_pp"] for value in selected])),
                "mean_delta_hn1_pp": float(np.mean([value["delta_hn1_pp"] for value in selected])),
                "fallback_seeds": [
                    int(seed)
                    for seed, value in zip(seeds, selected)
                    if value.get("fallback", False)
                ],
                "per_seed": {str(seed): value for seed, value in zip(seeds, selected)},
            }
        )
    shared = max(
        aggregate_trials,
        key=lambda value: (
            value["mean_token_f1"],
            -value["mean_switch_rate"],
            value["lambda_value"],
        ),
    )
    chosen_lambda = float(shared["lambda_value"])
    if progress:
        progress(
            "shared lambda selected",
            lambda_value=chosen_lambda,
            mean_oof_token_f1=shared["mean_token_f1"],
        )

    policies = {}
    seed_selections = {}
    for seed in seeds:
        state = seed_state[seed]
        values = state["rows"]
        h2, h3 = state["components"]["H2"], state["components"]["H3"]
        chosen = state["by_lambda"][chosen_lambda]["chosen"]
        policy = BlendPolicy(
            fit_ridge(values, H2_COLUMNS, h2["ridge"]["alpha"]),
            fit_harm(values, H2_COLUMNS, h2["harm"]["C"]),
            fit_ridge(values, H3_COLUMNS, h3["ridge"]["alpha"]),
            fit_harm(values, H3_COLUMNS, h3["harm"]["C"]),
            chosen_lambda,
            {
                "system": SYSTEM_SHARED,
                "selection_source": "mean document-grouped META_PRED_OOF across all declared seeds",
                "lambda_grid": list(LAMBDA_GRID),
                "lambda_value": chosen_lambda,
                "tau_utility": chosen["tau_utility"],
                "tau_harm": chosen["tau_harm"],
                "fold_policy": "BASE_FALLBACK" if chosen.get("fallback") else "ACTIVE",
                "selected_metrics": chosen,
                "policy_trials": state["by_lambda"][chosen_lambda]["trials"],
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
                    for label, component in state["components"].items()
                },
                "inner_folds_used": state["folds"],
                "shared_selection": {
                    "seeds": list(seeds),
                    "selected_lambda": chosen_lambda,
                    "aggregate_selected_metrics": shared,
                },
            },
            system=SYSTEM_SHARED,
        )
        policies[seed] = policy
        seed_selections[str(seed)] = policy.selection

    selection = {
        "system": SYSTEM_SHARED,
        "selection_unit": "one lambda per dataset and phase",
        "selection_source": "mean document-grouped META_PRED_OOF across seeds",
        "seeds": list(seeds),
        "lambda_grid": list(LAMBDA_GRID),
        "selected_lambda": chosen_lambda,
        "aggregate_trials": aggregate_trials,
        "seed_selections": seed_selections,
    }
    return policies, selection


__all__ = ["SYSTEM_SHARED", "fit_shared_lambda_policies", "apply_blend_policy"]
