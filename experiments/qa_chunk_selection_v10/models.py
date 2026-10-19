"""Honest inner-OOF training and V10 SRR policy simulation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np
import pandas as pd

from .contract import (
    ABLATIONS, ANSWER_RISK_COLUMNS, EVIDENCE_RISK_COLUMNS, H2_COLUMNS,
    INNER_EVIDENCE_MIN_QUESTIONS, INNER_EVIDENCE_MIN_ROWS,
    LOGISTIC_C_GRID, QUESTION_AGG_COLUMNS, RANK_COLUMNS, RANK_GRID,
    REFIT_EVIDENCE_MIN_QUESTIONS, REFIT_EVIDENCE_MIN_ROWS,
    SWITCH_COLUMNS, TAU_ANSWER_GRID, TAU_EVIDENCE_GRID, TAU_SWITCH_GRID,
)
from .features import pair_rows, risk_rows, switch_rows


class TrainingBlocked(RuntimeError):
    pass


def _frame(rows, columns):
    frame = pd.DataFrame(rows, columns=list(columns))
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all(): raise ValueError("non-finite model feature")
    return frame


def _logistic(rows, columns, c, seed, module, stage="refit"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    used = [x for x in rows if float(x.get("weight", 1.0)) > 0]
    y = np.asarray([int(x["target"]) for x in used])
    questions = {klass: {x["question_id"] for x in used if int(x["target"]) == klass} for klass in (0, 1)}
    counts = {klass: int((y == klass).sum()) for klass in (0, 1)}
    if module == "SWITCH":
        if min(counts.values(), default=0) < 10: raise TrainingBlocked(f"SWITCH_TRAINING_BLOCKED: {counts}")
    elif module == "EVIDENCE_RISK":
        min_rows = INNER_EVIDENCE_MIN_ROWS if stage == "inner" else REFIT_EVIDENCE_MIN_ROWS
        min_questions = INNER_EVIDENCE_MIN_QUESTIONS if stage == "inner" else REFIT_EVIDENCE_MIN_QUESTIONS
        if min(counts.values(), default=0) < min_rows or min((len(x) for x in questions.values()), default=0) < min_questions:
            raise TrainingBlocked(
                f"{module}_TRAINING_BLOCKED: stage={stage}, rows={counts}, "
                f"questions={dict((k,len(v)) for k,v in questions.items())}, "
                f"required_rows={min_rows}, required_questions={min_questions}"
            )
    else:
        if min(counts.values(), default=0) < 10 or min((len(x) for x in questions.values()), default=0) < 5:
            raise TrainingBlocked(f"{module}_TRAINING_BLOCKED: rows={counts}, questions={dict((k,len(v)) for k,v in questions.items())}")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=float(c), solver="liblinear", class_weight=None, random_state=seed, max_iter=3000)),
    ])
    model.fit(_frame(used, columns), y, model__sample_weight=np.asarray([x["weight"] for x in used], float))
    return model


def _ranker(rows, params, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    used = [x for x in rows if float(x["weight"]) > 0]
    unique_questions = {x["question_id"] for x in used}
    unordered = len(used) // 2
    if len(unique_questions) < 10 or unordered < 20:
        raise TrainingBlocked(f"RANK_TRAINING_BLOCKED: questions={len(unique_questions)}, unordered_pairs={unordered}")
    model = HistGradientBoostingClassifier(random_state=seed, **params)
    model.fit(
        _frame(used, RANK_COLUMNS), np.asarray([x["target"] for x in used], int),
        sample_weight=np.asarray([x["weight"] for x in used], float),
    )
    return model


def _predict_probability(model, rows, columns):
    if not rows: return np.asarray([], dtype=float)
    values = np.asarray(model.predict_proba(_frame(rows, columns))[:, 1], float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("invalid model probability")
    return values


def _fold_rows(rows, manifest, valid_fold):
    assignment = manifest["assignment"]
    return (
        [x for x in rows if assignment[x["canonical_document_id"]] != valid_fold],
        [x for x in rows if assignment[x["canonical_document_id"]] == valid_fold],
    )


def _oof_logistic(rows, columns, c, seed, manifest, module):
    output = {}
    for fold in range(manifest["folds"]):
        train, valid = _fold_rows(rows, manifest, fold)
        model = _logistic(train, columns, c, seed, module, stage="inner")
        probability = _predict_probability(model, valid, columns)
        for row, value in zip(valid, probability):
            key = (row["question_id"], row.get("chunk_id"))
            if key in output: raise AssertionError(f"duplicate OOF key: {key}")
            output[key] = float(value)
    expected = {(x["question_id"], x.get("chunk_id")) for x in rows}
    if set(output) != expected: raise RuntimeError(f"{module} OOF coverage incomplete")
    return output


def _rank_feature(question, left, right):
    row = {f"delta__{name}": float(left["x"][name] - right["x"][name]) for name in H2_COLUMNS}
    row.update({name: float(question["q"][name]) for name in QUESTION_AGG_COLUMNS})
    return row


def rank_question(model, question):
    candidates = question["candidates"]
    if len(candidates) < 2: return {candidates[0]["chunk_id"]: 0.0}, [], [candidates[0]["chunk_id"]]
    directed_rows = []; directed_keys = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            directed_rows.extend((_rank_feature(question, candidates[i], candidates[j]), _rank_feature(question, candidates[j], candidates[i])))
            directed_keys.extend(((i, j), (j, i)))
    raw_values = _predict_probability(model, directed_rows, RANK_COLUMNS)
    raw = {key: float(value) for key, value in zip(directed_keys, raw_values)}
    sym = {}; pair_payload = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            probability = (raw[(i, j)] + (1.0 - raw[(j, i)])) / 2.0
            sym[(i, j)] = probability; sym[(j, i)] = 1.0 - probability
            pair_payload.append({"left": candidates[i]["chunk_id"], "right": candidates[j]["chunk_id"], "p_left_gt_right": probability})
    scores = {
        candidate["chunk_id"]: float(np.mean([sym[(i, j)] for j in range(len(candidates)) if j != i]))
        for i, candidate in enumerate(candidates)
    }
    order = [x["chunk_id"] for x in sorted(candidates, key=lambda x: (-scores[x["chunk_id"]], -x["b0_score"], x["rank"], x["chunk_id"]))]
    return scores, pair_payload, order


def _oof_rank(questions, params, seed, manifest):
    all_rows = pair_rows(questions); output = {}
    for fold in range(manifest["folds"]):
        train = [x for x in all_rows if manifest["assignment"][x["canonical_document_id"]] != fold]
        valid_questions = [x for x in questions if manifest["assignment"][x["canonical_document_id"]] == fold and len(x["candidates"]) >= 2]
        model = _ranker(train, params, seed)
        for question in valid_questions:
            output[question["question_id"]] = rank_question(model, question)
    expected = {x["question_id"] for x in questions if len(x["candidates"]) >= 2}
    if set(output) != expected: raise RuntimeError("RANK OOF coverage incomplete")
    return output


def _weighted_log_loss(rows, predictions):
    from sklearn.metrics import log_loss
    used = [x for x in rows if float(x.get("weight", 1)) > 0]
    return float(log_loss(
        [x["target"] for x in used],
        np.clip([predictions[(x["question_id"], x.get("chunk_id"))] for x in used], 1e-8, 1 - 1e-8),
        sample_weight=[x["weight"] for x in used], labels=[0, 1],
    ))


def _choose_logistic(questions, evidence, switch, seed, manifest):
    if switch:
        rows, columns, module = switch_rows(questions), SWITCH_COLUMNS, "SWITCH"
    else:
        rows = risk_rows(questions, evidence=evidence)
        columns = EVIDENCE_RISK_COLUMNS if evidence else ANSWER_RISK_COLUMNS
        module = "EVIDENCE_RISK" if evidence else "ANSWER_RISK"
    trials = []
    for c in LOGISTIC_C_GRID:
        predictions = _oof_logistic(rows, columns, c, seed, manifest, module)
        trials.append({"C": c, "log_loss": _weighted_log_loss(rows, predictions), "predictions": predictions})
    best = min(x["log_loss"] for x in trials)
    selected = min((x for x in trials if x["log_loss"] <= best + 0.001), key=lambda x: x["C"])
    return selected, [{"C": x["C"], "log_loss": x["log_loss"]} for x in trials], rows, columns, module


def _choose_rank(questions, seed, manifest):
    trials = []
    lookup = {x["question_id"]: x for x in questions}
    for params in RANK_GRID:
        predictions = _oof_rank(questions, params, seed, manifest)
        f1 = []
        for question_id, (_, _, order) in predictions.items():
            question = lookup[question_id]; by_id = {x["chunk_id"]: x for x in question["candidates"]}
            best_alt = next(chunk for chunk in order if chunk != question["candidates"][0]["chunk_id"])
            f1.append(by_id[best_alt]["evaluation"]["f1"])
        trials.append({"params": dict(params), "top_alternative_f1": float(np.mean(f1)), "predictions": predictions})
    best = max(x["top_alternative_f1"] for x in trials)
    eligible = [x for x in trials if x["top_alternative_f1"] >= best - 0.10]
    selected = min(eligible, key=lambda x: (x["params"]["max_leaf_nodes"], x["params"]["max_iter"], -x["params"]["l2_regularization"]))
    return selected, [{"params": x["params"], "top_alternative_f1": x["top_alternative_f1"]} for x in trials]


def _risk_feature(question, candidate, evidence):
    base = question["candidates"][0]; row = {}
    if evidence: row.update({f"candidate__{name}": float(candidate["x"][name]) for name in H2_COLUMNS})
    row.update({f"delta__{name}": float(candidate["x"][name] - base["x"][name]) for name in H2_COLUMNS})
    row.update({name: float(question["q"][name]) for name in QUESTION_AGG_COLUMNS})
    return row


def predict_all(models, questions):
    output = {}
    for question in questions:
        qid = question["question_id"]; candidates = question["candidates"]
        if len(candidates) == 1:
            output[qid] = {"p_switch": None, "rank_scores": {}, "pairs": [], "rank_order": [], "p_answer_harm": {}, "p_evidence_harm": {}}
            continue
        p_switch = float(_predict_probability(models.switch, [question["q"]], SWITCH_COLUMNS)[0])
        scores, pairs, order = rank_question(models.rank, question)
        alternatives = candidates[1:]
        answer = _predict_probability(models.answer_risk, [_risk_feature(question, x, False) for x in alternatives], ANSWER_RISK_COLUMNS)
        evidence = _predict_probability(models.evidence_risk, [_risk_feature(question, x, True) for x in alternatives], EVIDENCE_RISK_COLUMNS)
        output[qid] = {
            "p_switch": p_switch, "rank_scores": scores, "pairs": pairs,
            "rank_order": [x for x in order if x != candidates[0]["chunk_id"]],
            "p_answer_harm": {x["chunk_id"]: float(v) for x, v in zip(alternatives, answer)},
            "p_evidence_harm": {x["chunk_id"]: float(v) for x, v in zip(alternatives, evidence)},
        }
    return output


def _oof_bundle(questions, switch_prediction, rank_prediction, answer_prediction, evidence_prediction):
    output = {}
    for question in questions:
        qid = question["question_id"]
        if len(question["candidates"]) == 1:
            output[qid] = {"p_switch": None, "rank_scores": {}, "pairs": [], "rank_order": [], "p_answer_harm": {}, "p_evidence_harm": {}}
            continue
        scores, pairs, order = rank_prediction[qid]; base = question["candidates"][0]["chunk_id"]
        output[qid] = {
            "p_switch": switch_prediction[(qid, None)], "rank_scores": scores, "pairs": pairs,
            "rank_order": [x for x in order if x != base],
            "p_answer_harm": {x["chunk_id"]: answer_prediction[(qid, x["chunk_id"])] for x in question["candidates"][1:]},
            "p_evidence_harm": {x["chunk_id"]: evidence_prediction[(qid, x["chunk_id"])] for x in question["candidates"][1:]},
        }
    return output


def simulate(questions, predictions, thresholds, use_answer=True, use_evidence=True, fold_fallback=False):
    rows = []
    for question in questions:
        candidates = question["candidates"]; base = candidates[0]; pred = predictions[question["question_id"]]
        if len(candidates) == 1:
            selected, reason = base, "NO_ALTERNATIVE"
        elif fold_fallback:
            selected, reason = base, "OUTER_FOLD_B0_FALLBACK"
        elif pred["p_switch"] < thresholds["tau_switch"]:
            selected, reason = base, "NO_SWITCH"
        else:
            by_id = {x["chunk_id"]: x for x in candidates}; selected = None
            for chunk_id in pred["rank_order"]:
                if use_answer and pred["p_answer_harm"][chunk_id] > thresholds["tau_answer_harm"]: continue
                if use_evidence and pred["p_evidence_harm"][chunk_id] > thresholds["tau_evidence_harm"]: continue
                selected = by_id[chunk_id]; break
            if selected is None: selected, reason = base, "NO_SAFE_CANDIDATE"
            else: reason = "SELECTED_ALTERNATIVE"
        rows.append({"question": question, "selected": selected, "base": base, "reason": reason, "prediction": pred})
    return rows


def policy_metrics(rows):
    if not rows: return {"token_f1": 0.0, "delta_hit1_pp": 0.0, "delta_hn1_pp": 0.0, "delta_sufficient_pp": 0.0, "switch_rate": 0.0, "positive_capture": 0.0, "harmful_switch_rate": 0.0}
    mean = lambda values: float(np.mean(list(values)))
    selected_f1 = mean(x["selected"]["evaluation"]["f1"] for x in rows)
    hit = mean(x["selected"]["evaluation"]["tier"] == "sufficient_positive" for x in rows)
    base_hit = mean(x["base"]["evaluation"]["tier"] == "sufficient_positive" for x in rows)
    hn = mean(x["selected"]["evaluation"]["hn"] for x in rows); base_hn = mean(x["base"]["evaluation"]["hn"] for x in rows)
    switched = [x for x in rows if x["selected"]["chunk_id"] != x["base"]["chunk_id"]]
    opportunities = [x for x in rows if x["question"]["evaluation"]["positive_opportunity"]]
    return {
        "token_f1": selected_f1, "delta_hit1_pp": 100 * (hit - base_hit), "delta_hn1_pp": 100 * (hn - base_hn),
        "delta_sufficient_pp": 100 * (hit - base_hit), "switch_rate": len(switched) / len(rows),
        "positive_capture": (mean(x["selected"]["evaluation"]["f1"] > x["base"]["evaluation"]["f1"] for x in opportunities) if opportunities else 0.0),
        "harmful_switch_rate": (mean(x["selected"]["evaluation"]["f1"] < x["base"]["evaluation"]["f1"] for x in switched) if switched else 0.0),
    }


def _better_trial(candidate, current):
    if current is None: return True
    if candidate["token_f1"] > current["token_f1"] + 0.10: return True
    if candidate["token_f1"] < current["token_f1"] - 0.10: return False
    left = (-candidate["delta_hn1_pp"], candidate["delta_sufficient_pp"], candidate["positive_capture"], -candidate["harmful_switch_rate"], candidate["thresholds"]["tau_switch"], -candidate["switch_rate"])
    right = (-current["delta_hn1_pp"], current["delta_sufficient_pp"], current["positive_capture"], -current["harmful_switch_rate"], current["thresholds"]["tau_switch"], -current["switch_rate"])
    return left > right


def calibrate(questions, predictions, system):
    use_answer, use_evidence = ABLATIONS[system]
    answer_grid = TAU_ANSWER_GRID if use_answer else (1.0,)
    evidence_grid = TAU_EVIDENCE_GRID if use_evidence else (1.0,)
    trials = []; selected = None
    for ts, ta, te in product(TAU_SWITCH_GRID, answer_grid, evidence_grid):
        thresholds = {"tau_switch": ts, "tau_answer_harm": ta, "tau_evidence_harm": te}
        metrics = policy_metrics(simulate(questions, predictions, thresholds, use_answer, use_evidence))
        feasible = metrics["delta_hit1_pp"] >= -0.5 - 1e-12 and metrics["delta_hn1_pp"] <= 1.0 + 1e-12 and metrics["delta_sufficient_pp"] >= -1.0 - 1e-12
        trial = {**metrics, "thresholds": thresholds, "feasible": feasible}; trials.append(trial)
        if feasible and _better_trial(trial, selected): selected = trial
    if selected is None:
        return {"fold_policy": "B0_FALLBACK", "thresholds": {"tau_switch": 1.0, "tau_answer_harm": 0.0, "tau_evidence_harm": 0.0}, "selected_metrics": policy_metrics(simulate(questions, predictions, {"tau_switch": 1.0, "tau_answer_harm": 0.0, "tau_evidence_harm": 0.0}, fold_fallback=True)), "trials": trials}
    return {"fold_policy": "ACTIVE", "thresholds": selected["thresholds"], "selected_metrics": {k: v for k, v in selected.items() if k != "thresholds"}, "trials": trials}


@dataclass
class SRRModels:
    switch: object
    rank: object
    answer_risk: object
    evidence_risk: object


def train_srr(questions, seed, manifest, progress=None):
    if progress: progress.log("tune SWITCH", seed=seed)
    switch, switch_trials, switch_data, _, _ = _choose_logistic(questions, False, True, seed, manifest)
    if progress: progress.log("tune PAIRWISE RANK", seed=seed, configurations=len(RANK_GRID))
    rank, rank_trials = _choose_rank(questions, seed, manifest)
    if progress: progress.log("tune ANSWER-RISK", seed=seed)
    answer, answer_trials, answer_data, _, _ = _choose_logistic(questions, False, False, seed, manifest)
    if progress: progress.log("tune EVIDENCE-RISK", seed=seed)
    evidence, evidence_trials, evidence_data, _, _ = _choose_logistic(questions, True, False, seed, manifest)
    oof = _oof_bundle(questions, switch["predictions"], rank["predictions"], answer["predictions"], evidence["predictions"])
    policies = {system: calibrate(questions, oof, system) for system in ABLATIONS}
    if progress: progress.log("TRAIN-only OOF thresholds calibrated", seed=seed, policies={k: v["fold_policy"] for k, v in policies.items()})
    models = SRRModels(
        switch=_logistic(switch_data, SWITCH_COLUMNS, switch["C"], seed, "SWITCH"),
        rank=_ranker(pair_rows(questions), rank["params"], seed),
        answer_risk=_logistic(answer_data, ANSWER_RISK_COLUMNS, answer["C"], seed, "ANSWER_RISK"),
        evidence_risk=_logistic(evidence_data, EVIDENCE_RISK_COLUMNS, evidence["C"], seed, "EVIDENCE_RISK"),
    )
    selection = {
        "switch": {"C": switch["C"], "oof_log_loss": switch["log_loss"], "trials": switch_trials},
        "rank": {"params": rank["params"], "oof_top_alternative_f1": rank["top_alternative_f1"], "trials": rank_trials},
        "answer_risk": {"C": answer["C"], "oof_log_loss": answer["log_loss"], "trials": answer_trials},
        "evidence_risk": {"C": evidence["C"], "oof_log_loss": evidence["log_loss"], "trials": evidence_trials},
        "policies": policies, "inner_split_sha256": manifest["sha256"],
        "training_support_contract": {"evidence_inner": {"rows_per_class": INNER_EVIDENCE_MIN_ROWS, "questions_per_class": INNER_EVIDENCE_MIN_QUESTIONS}, "evidence_refit": {"rows_per_class": REFIT_EVIDENCE_MIN_ROWS, "questions_per_class": REFIT_EVIDENCE_MIN_QUESTIONS}},
    }
    return models, selection, oof
