"""Document-grouped nested component selection for V9.

Only TRAIN/CAL SELECT_OOF rows are accepted here.  Outer-EVAL rows deliberately
use a different split_role and are rejected by the selector.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from experiments.qa_chunk_selection_v8.models import fold_for
from .contract import (
    ABSOLUTE_HIT_MIN_PP,
    ABSOLUTE_HN_MAX_PP,
    COMPLEXITY_TIE,
    CONFIG_COMPLEXITY,
    CONFIGS,
    EPS_SELECT,
    INCREMENTAL_HIT_MIN_PP,
    INCREMENTAL_HN_MAX_PP,
    SELECT_FOLDS,
    SELECT_SPLIT_SEED,
    WORST_SELECT_FOLD,
)
from .io_guard import sha256_json


def make_select_split_manifest(rows: list[dict]) -> dict:
    documents = sorted({str(x["canonical_document_id"]) for x in rows})
    if len(documents) < SELECT_FOLDS:
        raise ValueError(f"SELECT_FOLDS={SELECT_FOLDS} requires at least four document groups")
    assignment = {doc: fold_for(doc, SELECT_SPLIT_SEED, SELECT_FOLDS) for doc in documents}
    counts = Counter(assignment.values())
    missing = [fold for fold in range(SELECT_FOLDS) if counts.get(fold, 0) == 0]
    if missing:
        raise RuntimeError(f"deterministic V8-family SELECT splitter produced empty folds: {missing}")
    questions = defaultdict(set)
    for row in rows:
        questions[str(row["canonical_document_id"])].add(str(row["question_id"]))
    entries = [
        {
            "canonical_document_id": doc,
            "select_fold": assignment[doc],
            "question_ids": sorted(questions[doc]),
        }
        for doc in documents
    ]
    payload = {
        "splitter": "exact V8 fold_for(document_id, seed, folds)",
        "select_split_seed": SELECT_SPLIT_SEED,
        "select_folds": SELECT_FOLDS,
        "documents": entries,
        "fold_document_counts": {str(k): counts[k] for k in range(SELECT_FOLDS)},
    }
    payload["sha256"] = sha256_json(payload)
    return payload


def split_rows(rows: list[dict], manifest: dict, held_fold: int) -> tuple[list[dict], list[dict]]:
    assignment = {x["canonical_document_id"]: int(x["select_fold"]) for x in manifest["documents"]}
    train = [x for x in rows if assignment[str(x["canonical_document_id"])] != held_fold]
    valid = [x for x in rows if assignment[str(x["canonical_document_id"])] == held_fold]
    if not train or not valid:
        raise RuntimeError(f"empty SELECT train/valid partition for fold {held_fold}")
    train_docs = {str(x["canonical_document_id"]) for x in train}
    valid_docs = {str(x["canonical_document_id"]) for x in valid}
    if train_docs & valid_docs:
        raise AssertionError("document leakage between SELECT_TRAIN and SELECT_VALID")
    return train, valid


def split_runtime(runtime: list[dict], targets: dict, manifest: dict, held_fold: int):
    assignment = {x["canonical_document_id"]: int(x["select_fold"]) for x in manifest["documents"]}
    valid = [x for x in runtime if assignment[str(x["canonical_document_id"])] == held_fold]
    keys = {(str(x["dataset_cell"]), str(x["question_id"])) for x in valid}
    valid_targets = {key: value for key, value in targets.items() if key in keys}
    if not valid or not valid_targets:
        raise RuntimeError(f"missing SELECT_VALID runtime/targets for fold {held_fold}")
    return valid, valid_targets


def _index(rows: list[dict]) -> dict[tuple[str, int], dict]:
    result = {}
    for row in rows:
        runtime = row["runtime"]
        if runtime.get("split_role") != "SELECT_OOF":
            raise AssertionError("component selector accepts SELECT_OOF rows only")
        key = (str(runtime["question_id"]), int(runtime["seed"]))
        if key in result:
            raise AssertionError(f"duplicate SELECT_OOF question/seed: {key}")
        result[key] = row
    return result


def _question_average(index: dict, seeds: tuple[int, ...]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for (question_id, seed), row in index.items():
        grouped[question_id].append((seed, row))
    result = {}
    for question_id, values in grouped.items():
        seen = {seed for seed, _ in values}
        if seen != set(seeds):
            raise AssertionError(f"incomplete seed coverage for {question_id}: {sorted(seen)}")
        evaluations = [x["evaluation"] for _, x in values]
        runtime = values[0][1]["runtime"]
        result[question_id] = {
            "f1": float(np.mean([x["token_f1"] for x in evaluations])),
            "hit1": float(np.mean([x["hit@1"] for x in evaluations])),
            "hn1": float(np.mean([x["hn@1"] for x in evaluations])),
            "h0_f1": float(np.mean([x["baseline_token_f1"] for x in evaluations])),
            "h0_hit1": float(np.mean([x["baseline_hit@1"] for x in evaluations])),
            "h0_hn1": float(np.mean([x["baseline_hn@1"] for x in evaluations])),
            "select_fold": int(runtime["select_fold"]),
            "document_id": str(runtime["canonical_document_id"]),
            "selected": {int(seed): int(row["runtime"]["selected_chunk_id"]) for seed, row in values},
        }
    return result


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def select_configuration(predictions: dict[str, list[dict]], seeds: tuple[int, ...]) -> dict:
    if tuple(predictions) != CONFIGS:
        raise ValueError(f"selector requires frozen config order {CONFIGS}, got {tuple(predictions)}")
    indexes = {config: _index(rows) for config, rows in predictions.items()}
    reference_keys = set(indexes["C0"])
    for config, index in indexes.items():
        if set(index) != reference_keys:
            raise AssertionError(f"SELECT_OOF coverage differs for {config}")
    questions = {config: _question_average(index, seeds) for config, index in indexes.items()}
    reference_questions = set(questions["C0"])
    if any(set(value) != reference_questions for value in questions.values()):
        raise AssertionError("seed-averaged question coverage differs across configurations")

    required_seed_changes = 2 if len(seeds) == 3 else max(1, math.ceil(2 * len(seeds) / 3))
    required_activity = max(5, math.ceil(0.01 * len(reference_questions)))
    candidates = {}
    for config in CONFIGS:
        current = questions[config]
        base = questions["C0"]
        f1 = _mean(current[q]["f1"] for q in reference_questions)
        base_f1 = _mean(base[q]["f1"] for q in reference_questions)
        h0_f1 = _mean(base[q]["h0_f1"] for q in reference_questions)
        hit = _mean(current[q]["hit1"] for q in reference_questions)
        base_hit = _mean(base[q]["hit1"] for q in reference_questions)
        h0_hit = _mean(base[q]["h0_hit1"] for q in reference_questions)
        hn = _mean(current[q]["hn1"] for q in reference_questions)
        base_hn = _mean(base[q]["hn1"] for q in reference_questions)
        h0_hn = _mean(base[q]["h0_hn1"] for q in reference_questions)
        fold_deltas = []
        for fold in range(SELECT_FOLDS):
            ids = [q for q in reference_questions if current[q]["select_fold"] == fold]
            if not ids:
                raise AssertionError(f"SELECT_OOF fold {fold} has no questions")
            fold_deltas.append(_mean(current[q]["f1"] - base[q]["f1"] for q in ids))
        stable_changed = 0
        changed = []
        for question_id in sorted(reference_questions):
            count = sum(
                current[question_id]["selected"][seed] != base[question_id]["selected"][seed]
                for seed in seeds
            )
            if count >= required_seed_changes:
                stable_changed += 1
                changed.append(question_id)
        seed_delta = {
            str(seed): _mean(
                indexes[config][(q, seed)]["evaluation"]["token_f1"]
                - indexes["C0"][(q, seed)]["evaluation"]["token_f1"]
                for q in reference_questions
            )
            for seed in seeds
        }
        delta = f1 - base_f1
        metrics = {
            "config": config,
            "complexity": CONFIG_COMPLEXITY[config],
            "questions": len(reference_questions),
            "seed_averaged_select_oof_f1": f1,
            "c0_f1": base_f1,
            "h0_f1": h0_f1,
            "delta_f1_vs_c0": delta,
            "per_seed_delta_f1_vs_c0": seed_delta,
            "selection_fold_deltas": fold_deltas,
            "nonnegative_selection_folds": sum(x >= -1e-12 for x in fold_deltas),
            "worst_selection_fold_delta": min(fold_deltas),
            "hit1": hit,
            "hn1": hn,
            "delta_hit1_vs_c0_pp": 100.0 * (hit - base_hit),
            "delta_hn1_vs_c0_pp": 100.0 * (hn - base_hn),
            "delta_hit1_vs_h0_pp": 100.0 * (hit - h0_hit),
            "delta_hn1_vs_h0_pp": 100.0 * (hn - h0_hn),
            "stable_changed_questions": stable_changed,
            "required_stable_changed_questions": required_activity,
            "stable_changed_question_ids": changed,
            "switch_rate_vs_c0": stable_changed / max(len(reference_questions), 1),
        }
        checks = {
            "minimum_gain": delta >= EPS_SELECT - 1e-12,
            "fold_sign_stability": metrics["nonnegative_selection_folds"] >= 3,
            "worst_fold_guard": metrics["worst_selection_fold_delta"] >= WORST_SELECT_FOLD - 1e-12,
            "incremental_hit_safety": metrics["delta_hit1_vs_c0_pp"] >= INCREMENTAL_HIT_MIN_PP - 1e-12,
            "incremental_hn_safety": metrics["delta_hn1_vs_c0_pp"] <= INCREMENTAL_HN_MAX_PP + 1e-12,
            "absolute_hit_safety": metrics["delta_hit1_vs_h0_pp"] >= ABSOLUTE_HIT_MIN_PP - 1e-12,
            "absolute_hn_safety": metrics["delta_hn1_vs_h0_pp"] <= ABSOLUTE_HN_MAX_PP + 1e-12,
            "activity": stable_changed >= required_activity,
        }
        metrics["checks"] = checks
        metrics["eligible"] = config == "C0" or all(checks.values())
        candidates[config] = metrics

    eligible = [candidates[x] for x in CONFIGS[1:] if candidates[x]["eligible"]]
    if not eligible:
        selected = "C0"
        reason = "no optional configuration passed all frozen SELECT_OOF gates"
    else:
        best_f1 = max(x["seed_averaged_select_oof_f1"] for x in eligible)
        tied = [x for x in eligible if best_f1 - x["seed_averaged_select_oof_f1"] < COMPLEXITY_TIE - 1e-12]
        tied.sort(key=lambda x: (
            x["complexity"],
            x["hn1"],
            -x["hit1"],
            x["switch_rate_vs_c0"],
            x["config"],
        ))
        selected = tied[0]["config"]
        reason = "maximum eligible SELECT_OOF F1 with frozen complexity/evidence tie-breaks"

    sensitivity = {}
    for eps in (0.0, 0.1, 0.2, 0.3):
        sensitivity[str(eps)] = [
            config for config in CONFIGS[1:]
            if candidates[config]["delta_f1_vs_c0"] >= eps - 1e-12
            and all(value for name, value in candidates[config]["checks"].items() if name != "minimum_gain")
        ]
    return {
        "selected_config": selected,
        "selection_reason": reason,
        "selection_source": "seed-averaged TRAIN-only SELECT_OOF",
        "outer_eval_accessed": False,
        "seeds_are_not_independent_rows": True,
        "model_seeds": list(seeds),
        "required_seed_changes": required_seed_changes,
        "candidates": candidates,
        "eps_sensitivity_diagnostic_only": sensitivity,
    }

