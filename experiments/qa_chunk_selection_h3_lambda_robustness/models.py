"""Reusable OOF state for seed-count and stability-gate ablations."""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np

from experiments.qa_chunk_selection_v8.contract import H2_COLUMNS, H3_COLUMNS
from experiments.qa_chunk_selection_v8.features import normalize_weights
from experiments.qa_chunk_selection_v8.models import fit_harm, fit_ridge, simulate
from experiments.qa_chunk_selection_h3_shrinkage_ablation.models import (
    BlendPolicy,
    LAMBDA_GRID,
    _fallback,
    _select_harm,
    _select_ridge,
    _threshold_trials,
    apply_blend_policy,
)


SYSTEM_SHARED_10 = "H3-SHARED-10"
SYSTEM_STABILITY_3 = "H3-STABILITY-3"
SYSTEM_STABILITY_10 = "H3-STABILITY-10"
SYSTEMS = (SYSTEM_SHARED_10, SYSTEM_STABILITY_3, SYSTEM_STABILITY_10)
SEEDS_3 = (42, 43, 44)
SEEDS_10 = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)


def _seed_rows(rows, seed):
    values = copy.deepcopy(rows)
    for row in values:
        row["seed"] = int(seed)
    normalize_weights(values, pooled=False)
    return values


def _best(rows, trials, lambda_value):
    feasible = [value for value in trials if value["feasible"]]
    if not feasible:
        return _fallback(rows, lambda_value=lambda_value) | {"fallback": True}
    return dict(
        max(
            feasible,
            key=lambda value: (
                value["token_f1"],
                -value["switch_rate"],
                value["tau_utility"],
                -value["tau_harm"],
            ),
        )
    ) | {"fallback": False}


def prepare_states(rows: list[dict], seeds: tuple[int, ...], folds: int, progress=None) -> dict:
    """Fit every seed once and retain all lambda-specific OOF decisions."""
    states = {}
    for seed in seeds:
        values = _seed_rows(rows, seed)
        documents = {row["canonical_document_id"] for row in values}
        used_folds = max(2, min(int(folds), 4, len(documents)))
        components = {}
        for label, columns in (("H2", H2_COLUMNS), ("H3", H3_COLUMNS)):
            if progress:
                progress("robust-lambda component", seed=seed, component=label)
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
            chosen = _best(values, trials, lambda_value)
            tau_utility = float("inf") if chosen["tau_utility"] == "inf" else float(chosen["tau_utility"])
            choices = simulate(
                values,
                utility,
                harm,
                True,
                tau_utility,
                float(chosen["tau_harm"]),
            )
            by_lambda[float(lambda_value)] = {
                "chosen": chosen,
                "trials": trials,
                "choices": choices,
            }
        states[int(seed)] = {
            "rows": values,
            "folds": used_folds,
            "components": components,
            "by_lambda": by_lambda,
        }
    return states


def select_shared(states: dict, seeds: tuple[int, ...]) -> dict:
    aggregate = []
    for lambda_value in LAMBDA_GRID:
        values = [states[seed]["by_lambda"][float(lambda_value)]["chosen"] for seed in seeds]
        aggregate.append(
            {
                "lambda_value": float(lambda_value),
                "mean_token_f1": float(np.mean([value["token_f1"] for value in values])),
                "mean_delta_f1": float(np.mean([value["delta_f1"] for value in values])),
                "mean_switch_rate": float(np.mean([value["switch_rate"] for value in values])),
                "mean_delta_hit1_pp": float(np.mean([value["delta_hit1_pp"] for value in values])),
                "mean_delta_hn1_pp": float(np.mean([value["delta_hn1_pp"] for value in values])),
                "fallback_seeds": [
                    int(seed)
                    for seed, value in zip(seeds, values)
                    if value.get("fallback", False)
                ],
                "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
            }
        )
    selected = max(
        aggregate,
        key=lambda value: (
            value["mean_token_f1"],
            -value["mean_switch_rate"],
            value["lambda_value"],
        ),
    )
    return {
        "seeds": list(seeds),
        "lambda_grid": list(LAMBDA_GRID),
        "selected_lambda": float(selected["lambda_value"]),
        "selected_metrics": selected,
        "aggregate_trials": aggregate,
    }


def lambda_evidence(states: dict, seeds: tuple[int, ...], candidate: float, iterations=5000) -> dict:
    """Document-clustered OOF bootstrap for candidate lambda minus H3."""
    candidate = float(candidate)
    if candidate == 1.0:
        return {
            "candidate_lambda": 1.0,
            "reference_lambda": 1.0,
            "delta_token_f1": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "standard_error": 0.0,
            "documents": len(
                {row["canonical_document_id"] for row in states[seeds[0]]["rows"]}
            ),
            "iterations": iterations,
            "gate_pass": True,
            "reason": "H3_DEFAULT",
        }
    by_document = defaultdict(list)
    for seed in seeds:
        state = states[seed]
        docs = {
            (row["dataset_cell"], row["question_id"]): row["canonical_document_id"]
            for row in state["rows"]
        }
        candidate_choices = {
            (row["dataset_cell"], row["question_id"]): row
            for row in state["by_lambda"][candidate]["choices"]
        }
        reference_choices = {
            (row["dataset_cell"], row["question_id"]): row
            for row in state["by_lambda"][1.0]["choices"]
        }
        if set(candidate_choices) != set(reference_choices):
            raise RuntimeError("lambda OOF question universes differ")
        for key in candidate_choices:
            by_document[docs[key]].append(
                float(candidate_choices[key]["f1"] - reference_choices[key]["f1"])
            )
    documents = sorted(by_document)
    observed = float(np.mean([value for doc in documents for value in by_document[doc]]))
    rng = np.random.default_rng(1042)
    samples = []
    for _ in range(int(iterations)):
        draw = rng.choice(documents, len(documents), replace=True)
        samples.append(float(np.mean([value for doc in draw for value in by_document[doc]])))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "candidate_lambda": candidate,
        "reference_lambda": 1.0,
        "delta_token_f1": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "standard_error": float(np.std(samples, ddof=1)),
        "documents": len(documents),
        "iterations": int(iterations),
        "cluster": "canonical document across all declared seeds",
        "gate_pass": bool(low > 0.0),
        "reason": "LOWER_95_CI_ABOVE_ZERO" if low > 0.0 else "UNCERTAINTY_OVERLAPS_ZERO",
    }


def stability_decision(development: dict, refit: dict, evidence: dict) -> dict:
    dev_lambda = float(development["selected_lambda"])
    refit_lambda = float(refit["selected_lambda"])
    stable = dev_lambda == refit_lambda
    accepted = refit_lambda == 1.0 or (stable and evidence["gate_pass"])
    return {
        "development_lambda": dev_lambda,
        "refit_lambda": refit_lambda,
        "exactly_stable": stable,
        "evidence": evidence,
        "accepted_lambda": refit_lambda if accepted else 1.0,
        "gate_pass": accepted,
        "fallback_to_h3": not accepted,
        "reason": (
            "H3_DEFAULT"
            if refit_lambda == 1.0
            else "STABLE_AND_OOF_SIGNIFICANT"
            if accepted
            else "UNSTABLE_TRAIN_REFIT_SELECTION"
            if not stable
            else "OOF_UNCERTAINTY_OVERLAPS_ZERO"
        ),
    }


def build_policies(states: dict, seeds: tuple[int, ...], lambda_value: float, system: str, shared: dict, gate=None):
    policies = {}
    lambda_value = float(lambda_value)
    for seed in seeds:
        state = states[seed]
        values = state["rows"]
        h2, h3 = state["components"]["H2"], state["components"]["H3"]
        chosen = state["by_lambda"][lambda_value]["chosen"]
        selection = {
            "system": system,
            "selection_source": "mean document-grouped META_PRED_OOF across declared seeds",
            "lambda_grid": list(LAMBDA_GRID),
            "lambda_value": lambda_value,
            "tau_utility": chosen["tau_utility"],
            "tau_harm": chosen["tau_harm"],
            "fold_policy": "BASE_FALLBACK" if chosen.get("fallback") else "ACTIVE",
            "selected_metrics": chosen,
            "policy_trials": state["by_lambda"][lambda_value]["trials"],
            "components": {
                label: {
                    "columns": list(component["columns"]),
                    "alpha": component["ridge"]["alpha"],
                    "ridge_oof_mae": component["ridge"]["mae"],
                    "C_harm": component["harm"]["C"],
                    "harm_oof_log_loss": component["harm"]["log_loss"],
                }
                for label, component in state["components"].items()
            },
            "inner_folds_used": state["folds"],
            "shared_selection": shared,
            "stability_gate": gate,
        }
        policies[seed] = BlendPolicy(
            fit_ridge(values, H2_COLUMNS, h2["ridge"]["alpha"]),
            fit_harm(values, H2_COLUMNS, h2["harm"]["C"]),
            fit_ridge(values, H3_COLUMNS, h3["ridge"]["alpha"]),
            fit_harm(values, H3_COLUMNS, h3["harm"]["C"]),
            lambda_value,
            selection,
            system=system,
        )
    return policies


__all__ = [
    "SYSTEMS",
    "SYSTEM_SHARED_10",
    "SYSTEM_STABILITY_3",
    "SYSTEM_STABILITY_10",
    "SEEDS_3",
    "SEEDS_10",
    "prepare_states",
    "select_shared",
    "lambda_evidence",
    "stability_decision",
    "build_policies",
    "apply_blend_policy",
]
