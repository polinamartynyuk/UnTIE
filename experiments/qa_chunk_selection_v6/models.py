"""Strict META_PRED_OOF fitting, Platt calibration and safe threshold selection."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .contract import C_GRID, GATE_NUMERIC, PRIMARY_BUDGET, TAU_SWITCH_GRID, feature_columns


class ConstantModel:
    def __init__(self, probability: float):
        self.probability = float(probability)
    def decision_function(self, frame):
        p = min(max(self.probability, 1e-8), 1 - 1e-8)
        return np.full(len(frame), math.log(p / (1 - p)), dtype=float)
    def predict_proba(self, frame):
        p = 1 / (1 + np.exp(-self.decision_function(frame)))
        return np.column_stack([1 - p, p])


class IdentityCalibrator:
    status = "unavailable_single_class"
    def predict(self, scores):
        scores = np.asarray(scores, dtype=float)
        return 1 / (1 + np.exp(-np.clip(scores, -40, 40)))


class PlattCalibrator:
    status = "fitted_on_meta_pred_oof"
    def __init__(self, model): self.model = model
    def predict(self, scores):
        return self.model.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]


def _pipeline(columns, seed: int, *, c: float, balanced: bool, categorical=("language", "canonical_aspect")):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return Pipeline([
        ("pre", ColumnTransformer([
            ("continuous", StandardScaler(), list(columns)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
        ])),
        ("model", LogisticRegression(C=float(c), penalty="l2", solver="liblinear", class_weight="balanced" if balanced else None,
                                     max_iter=3000, random_state=int(seed))),
    ])


def fit_model(rows, columns, seed: int, *, c=1.0, balanced=False, categorical=("language", "canonical_aspect")):
    import pandas as pd
    if not rows:
        return ConstantModel(0.0)
    frame = pd.DataFrame(rows); y = frame.target.astype(int)
    if y.nunique() < 2:
        return ConstantModel(float(y.iloc[0]))
    model = _pipeline(columns, seed, c=c, balanced=balanced, categorical=categorical)
    fit = {"model__sample_weight": frame.weight.astype(float)} if "weight" in frame else {}
    model.fit(frame, y, **fit)
    return model


def decision(model, rows) -> np.ndarray:
    import pandas as pd
    frame = pd.DataFrame(rows)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(frame), dtype=float)
    p = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    return np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))


def fit_platt(scores, targets):
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(targets, dtype=int)
    if len(set(y.tolist())) < 2:
        return IdentityCalibrator()
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=3000)
    model.fit(np.asarray(scores, dtype=float).reshape(-1, 1), y)
    return PlattCalibrator(model)


def fold_for(document_id: str, seed: int, folds: int) -> int:
    raw = hashlib.sha256(f"{seed}|{document_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % folds


def meta_oof(rows: list[dict], columns, seed: int, *, c: float, balanced: bool, folds: int, categorical=("language", "canonical_aspect")) -> tuple[np.ndarray, list[int]]:
    predictions = np.full(len(rows), np.nan); assignment = [fold_for(r["canonical_document_id"], seed, folds) for r in rows]
    for fold in sorted(set(assignment)):
        train = [row for row, value in zip(rows, assignment) if value != fold]
        indices = [i for i, value in enumerate(assignment) if value == fold]
        validation = [rows[i] for i in indices]
        if not train or not validation:
            continue
        model = fit_model(train, columns, seed + fold, c=c, balanced=balanced, categorical=categorical)
        predictions[indices] = decision(model, validation)
    if np.isnan(predictions).any():
        raise RuntimeError("META_PRED_OOF coverage incomplete")
    return predictions, assignment


def _question_choices(rows: list[dict], probabilities, tau: float, *, gate=None, gate_tau=-math.inf):
    groups = defaultdict(list)
    for index, row in enumerate(rows): groups[row["question_id"]].append((row, float(probabilities[index])))
    selected = []
    for qid, candidates in groups.items():
        first = candidates[0][0]
        routed = gate is None or (float(gate.get(qid, 0.0)) >= gate_tau and int(first["effective_k"]) > 1)
        best, p = max(candidates, key=lambda value: (value[1], -int(value[0]["chunk_id"])))
        switch = bool(routed and p > tau)
        selected.append({"question_id": qid, "f1": best["candidate_f1"] if switch else first["baseline_f1"],
                         "hit1": int((best["candidate_tier"] if switch else first["baseline_tier"]) == "sufficient_positive"),
                         "hn1": int(best["candidate_hn"] if switch else first["baseline_hn"]),
                         "base_f1": first["baseline_f1"], "base_hit1": int(first["baseline_tier"] == "sufficient_positive"),
                         "base_hn1": int(first["baseline_hn"]), "switch": switch,
                         "effective_k": int(first["effective_k"]), "route": routed})
    return selected


def metrics(choices) -> dict:
    mean = lambda key: float(np.mean([float(row[key]) for row in choices])) if choices else 0.0
    return {"token_f1": mean("f1"), "baseline_token_f1": mean("base_f1"),
            "delta_f1": mean("f1") - mean("base_f1"), "hit1": mean("hit1"),
            "delta_hit1_pp": 100 * (mean("hit1") - mean("base_hit1")), "hn1": mean("hn1"),
            "delta_hn1_pp": 100 * (mean("hn1") - mean("base_hn1")),
            "switch_rate": mean("switch"), "routing_rate": mean("route"),
            "qa_forwards": float(np.mean([r["effective_k"] if r["route"] else 1 for r in choices])) if choices else 1.0}


def safe(value: dict) -> bool:
    return value["delta_hit1_pp"] >= -0.5 - 1e-12 and value["delta_hn1_pp"] <= 1.0 + 1e-12


def select_switch(rows, probabilities) -> tuple[float, dict, list[dict]]:
    trials = []
    for tau in TAU_SWITCH_GRID:
        score = metrics(_question_choices(rows, probabilities, tau))
        trials.append({"tau": tau, **score, "safe": safe(score)})
    feasible = [row for row in trials if row["safe"]]
    if not feasible:
        fallback = metrics(_question_choices(rows, probabilities, math.inf))
        chosen = {"tau": "inf", **fallback, "safe": True, "fallback": "no safe nontrivial switch threshold"}
        return math.inf, chosen, [*trials, chosen]
    chosen = max(feasible, key=lambda row: (row["token_f1"], row["tau"], -row["switch_rate"]))
    return float(chosen["tau"]), chosen, trials


@dataclass
class FittedUtility:
    model: object
    calibrator: object
    columns: tuple[str, ...]
    c: float
    tau: float
    oof_probability: np.ndarray
    meta_fold: list[int]
    selection: dict

    def probabilities(self, rows): return self.calibrator.predict(decision(self.model, rows))


def tune_utility(rows: list[dict], variant: str, seed: int, folds: int, *, aspect=True) -> FittedUtility:
    columns = feature_columns(variant, aspect=aspect); trials = []
    categorical = ("language", "canonical_aspect") if aspect else ("language",)
    for c in C_GRID:
        raw, assignment = meta_oof(rows, columns, seed, c=c, balanced=False, folds=folds, categorical=categorical)
        calibrator = fit_platt(raw, [r["target"] for r in rows]); probability = calibrator.predict(raw)
        tau, score, thresholds = select_switch(rows, probability)
        trials.append({"c": c, "raw": raw, "assignment": assignment, "calibrator": calibrator,
                       "probability": probability, "tau": tau, "score": score, "thresholds": thresholds})
    feasible = [trial for trial in trials if trial["score"]["safe"]]
    chosen = max(feasible or trials, key=lambda x: (x["score"]["token_f1"], -x["c"]))
    model = fit_model(rows, columns, seed, c=chosen["c"], balanced=False, categorical=categorical)
    summary = {"selected_C": chosen["c"], "tau_switch": chosen["tau"], "calibration": chosen["calibrator"].status,
               "C_trials": [{"C": x["c"], "selected_threshold_metrics": x["score"], "threshold_trials": x["thresholds"]} for x in trials]}
    return FittedUtility(model, chosen["calibrator"], columns, chosen["c"], chosen["tau"],
                         chosen["probability"], chosen["assignment"], summary)


@dataclass
class FittedGate:
    model: object
    calibrator: object
    tau: float
    probabilities_oof: dict[str, float]
    meta_fold: list[int]
    selection: dict
    def probabilities(self, rows): return self.calibrator.predict(decision(self.model, rows))


def tune_gate(gate_rows: list[dict], utility_rows: list[dict], utility_oof, tau_switch: float,
              seed: int, folds: int, *, budget=PRIMARY_BUDGET) -> FittedGate:
    raw, assignment = meta_oof(gate_rows, GATE_NUMERIC, seed, c=1.0, balanced=True, folds=folds)
    calibrator = fit_platt(raw, [r["target"] for r in gate_rows]); probability = calibrator.predict(raw)
    by_question = {row["question_id"]: float(p) for row, p in zip(gate_rows, probability)}
    candidates = sorted({math.inf, *probability.tolist()}, reverse=True)
    trials = []
    for tau in candidates:
        score = metrics(_question_choices(utility_rows, utility_oof, tau_switch, gate=by_question, gate_tau=tau))
        trials.append({"tau": float(tau) if math.isfinite(tau) else "inf", **score,
                       "feasible": score["qa_forwards"] <= budget + 1e-9 and safe(score)})
    feasible = [row for row in trials if row["feasible"]]
    if feasible:
        chosen = max(feasible, key=lambda row: (row["token_f1"], -row["qa_forwards"],
                                                float(row["tau"]) if row["tau"] != "inf" else math.inf,
                                                -row["routing_rate"]))
        tau = math.inf if chosen["tau"] == "inf" else float(chosen["tau"])
    else:
        tau = math.inf; chosen = next(row for row in trials if row["tau"] == "inf")
    model = fit_model(gate_rows, GATE_NUMERIC, seed, c=1.0, balanced=True)
    return FittedGate(model, calibrator, tau, by_question, assignment,
                      {"budget": budget, "tau_gate": "inf" if not math.isfinite(tau) else tau,
                       "calibration": calibrator.status, "selected": chosen, "trials": trials})


def normalize_training_weights(rows: list[dict], *, pooled: bool) -> dict:
    """Normalize questions, duplicate SciER lineages, then families without mixing seeds."""
    by_question = defaultdict(list)
    for row in rows: by_question[(row["dataset_cell"], row["question_id"])].append(row)
    for values in by_question.values():
        total = sum(float(r.get("weight", 1.0)) for r in values) or 1.0
        for row in values: row["weight"] = float(row.get("weight", 1.0)) / total
    by_lineage = defaultdict(list)
    for row in rows: by_lineage[row["question_lineage_id"]].append(row)
    for values in by_lineage.values():
        cells = {r["dataset_cell"] for r in values}
        divisor = len(cells) if any(c.startswith("scier_") for c in cells) else 1
        for row in values: row["weight"] /= divisor
    if pooled:
        by_family = defaultdict(list)
        for row in rows: by_family[row["corpus_family"]].append(row)
        totals = {family: sum(float(r["weight"]) for r in values) for family, values in by_family.items()}
        for family, values in by_family.items():
            for row in values: row["weight"] /= max(totals[family], 1e-12)
    audit = {"questions": {f"{cell}:{qid}": sum(r["weight"] for r in values) for (cell, qid), values in by_question.items()},
             "lineages": {key: sum(r["weight"] for r in values) for key, values in by_lineage.items()},
             "families": {family: sum(r["weight"] for r in rows if r["corpus_family"] == family) for family in sorted({r["corpus_family"] for r in rows})}}
    return audit

