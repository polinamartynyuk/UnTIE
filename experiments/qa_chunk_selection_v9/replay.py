"""Exact H0/H1/H2/H3 replay against immutable V8 predictions."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np

from experiments.qa_chunk_selection_v8.models import Policy
from experiments.qa_chunk_selection_v8.pipeline import build_predictions
from .data import DataStore
from .io_guard import read_jsonl, write_json


def _rows_by_question(rows: list[dict]) -> dict[str, dict]:
    return {str(x["runtime"]["question_id"]): x for x in rows}


def _compare(expected: list[dict], observed: list[dict]) -> dict:
    left = _rows_by_question(expected)
    right = _rows_by_question(observed)
    if set(left) != set(right):
        raise RuntimeError("replay question coverage mismatch")
    mismatch = defaultdict(int)
    for question_id in left:
        a, b = left[question_id], right[question_id]
        if int(a["runtime"]["selected_chunk_id"]) != int(b["runtime"]["selected_chunk_id"]):
            mismatch["selected_chunk_id"] += 1
        if str(a["runtime"]["final_answer"]) != str(b["runtime"]["final_answer"]):
            mismatch["final_answer"] += 1
        if int(a["evaluation"]["changed_top1"]) != int(b["evaluation"]["changed_top1"]):
            mismatch["changed_top1"] += 1
        if abs(float(a["evaluation"]["token_f1"]) - float(b["evaluation"]["token_f1"])) > 1e-10:
            mismatch["token_f1"] += 1
    return {
        "rows": len(left),
        "mismatches": dict(mismatch),
        "status": "PASS" if not mismatch else "FAIL",
        "token_f1": float(np.mean([x["evaluation"]["token_f1"] for x in observed])),
    }


def _load_policy(root: Path, system: str) -> Policy:
    payload = json.loads((root / "selection.json").read_text())
    return Policy(
        system=system,
        columns=tuple(payload["feature_columns"]),
        utility_model=joblib.load(root / "utility_model.joblib"),
        harm_model=joblib.load(root / "harm_model.joblib"),
        selection=payload["selection"],
    )


def replay(config, output: Path, progress) -> dict:
    store = DataStore(config, progress)
    result = {system: {"cells": {}, "status": "PASS"} for system in ("H0", "H1", "H2", "H3")}
    totals = defaultdict(list)
    units = len(config.datasets) * len(config.outer_folds) * len(config.seeds)
    counter = 0
    for cell in config.datasets:
        per_cell = defaultdict(list)
        for fold in config.outer_folds:
            for seed in config.seeds:
                counter += 1
                progress.count(counter, units, item="exact H0/H1/H2/H3 replay", extra={"cell": cell, "fold": fold, "seed": seed})
                runtime, hrows, targets = store.load(cell, fold, seed, "eval")
                v8_predictions = config.v8_run / "branch_hrr/predictions/local" / cell / cell / f"outer_fold_{fold}" / f"seed_{seed}"
                generated_h0 = build_predictions("hrr", cell, runtime, hrows, targets, "H0", "local", fold, seed)
                comparisons = {"H0": _compare(read_jsonl(v8_predictions / "H0.jsonl"), generated_h0)}
                v7_h1 = config.v7_run / "predictions/local" / cell / cell / f"outer_fold_{fold}" / f"seed_{seed}" / "S2.jsonl"
                comparisons["H1"] = _compare(read_jsonl(v8_predictions / "H1.jsonl"), read_jsonl(v7_h1))
                for system in ("H2", "H3"):
                    model_root = config.v8_run / "branch_hrr/models/local" / cell / f"outer_fold_{fold}" / f"seed_{seed}" / system
                    policy = _load_policy(model_root, system)
                    generated = build_predictions("hrr", cell, runtime, hrows, targets, system, "local", fold, seed, policy)
                    comparisons[system] = _compare(read_jsonl(v8_predictions / f"{system}.jsonl"), generated)
                for system, comparison in comparisons.items():
                    if comparison["status"] != "PASS":
                        raise RuntimeError(f"{system}_REPLAY_FAILED: {cell} fold={fold} seed={seed}: {comparison['mismatches']}")
                    per_cell[system].append(comparison)
                    totals[system].append(comparison)
        for system in result:
            rows = per_cell[system]
            result[system]["cells"][cell] = {
                "units": len(rows),
                "rows": sum(x["rows"] for x in rows),
                "status": "PASS",
                "mean_unit_token_f1": float(np.mean([x["token_f1"] for x in rows])),
            }
    for system in result:
        result[system]["rows"] = sum(x["rows"] for x in totals[system])
        write_json(output / f"{system.lower()}_replay.json", result[system])
    return result

