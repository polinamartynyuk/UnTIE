"""Leakage-separated V10 feature and target construction."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Iterable

import numpy as np

from experiments.qa_chunk_selection.v4.answers import normalize_predicted_answer
from .contract import (
    ANSWER_RISK_COLUMNS, EVIDENCE_MAPPING, EVIDENCE_RISK_COLUMNS, H2_COLUMNS,
    INNER_FOLDS, INNER_SPLIT_SEED, PAIR_MIN_GAP, QUESTION_AGG_COLUMNS,
    RANK_COLUMNS, SWITCH_COLUMNS,
)
from .io_guard import sha256_json


RUNTIME_FORBIDDEN = (
    "gold", "actual_f1", "actual_utility", "utility", "evidence_label",
    "oracle", "dataset_id", "language_id", "aspect_id",
)


def grouped(rows: Iterable[dict]) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for row in rows: result[str(row["question_id"])].append(row)
    return {key: sorted(value, key=lambda x: (int(x["b0_rank"]), int(x["chunk_id"]))) for key, value in result.items()}


def assert_runtime_gold_free(value, path="runtime"):
    if isinstance(value, dict):
        for key, child in value.items():
            low = str(key).lower()
            if any(token in low for token in RUNTIME_FORBIDDEN):
                raise AssertionError(f"forbidden field leaked at {path}.{key}")
            assert_runtime_gold_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value): assert_runtime_gold_free(child, f"{path}[{index}]")


def evidence_class(label) -> str:
    if label not in EVIDENCE_MAPPING:
        raise RuntimeError(f"UNKNOWN_EVIDENCE_LABEL: {label!r}")
    return EVIDENCE_MAPPING[label]


def _base_vector(top: dict) -> dict[str, float]:
    valid = int(bool(str(top.get("qa_answer") or "").strip()))
    gap = float(top["gap12_norm"])
    vector = {
        "delta_b0_norm": 0.0, "delta_qarel_raw": 0.0, "pairlex": 1.0,
        "candidate_rank_norm": 0.0, "effective_k": float(top["effective_k"]),
        "gap12_norm": gap, "b0_entropy_norm": float(top["b0_entropy_norm"]),
        "answer_valid_i": float(valid), "answer_valid_0": float(valid),
        "both_empty": float(not valid), "qarel_x_margin": 0.0,
        "pairlex_x_margin": gap, "qarel_x_pairlex": 0.0,
        "b0_x_margin": 0.0, "delta_fulllex": 0.0,
        "fulllex_x_margin": 0.0, "qarel_x_fulllex": 0.0,
    }
    return {key: float(vector[key]) for key in H2_COLUMNS}


def _finite_vector(value: dict, columns: tuple[str, ...], label: str) -> None:
    missing = [name for name in columns if name not in value]
    if missing: raise KeyError(f"{label} missing columns: {missing}")
    bad = {name: value[name] for name in columns if not math.isfinite(float(value[name]))}
    if bad: raise ValueError(f"{label} non-finite features: {bad}")


def question_aggregates(candidates: list[dict]) -> dict[str, float]:
    if not candidates: raise ValueError("question has no candidates")
    candidates = sorted(candidates, key=lambda x: (int(x["rank"]), int(x["chunk_id"])))
    output = {}
    for column in H2_COLUMNS:
        values = np.asarray([x["x"][column] for x in candidates], dtype=float)
        output.update({
            f"{column}__max": float(values.max()), f"{column}__min": float(values.min()),
            f"{column}__mean": float(values.mean()), f"{column}__std": float(values.std()),
            f"{column}__range": float(values.max() - values.min()),
        })
    normalized = [normalize_predicted_answer(x["answer"]) for x in candidates if str(x["answer"]).strip()]
    counts = Counter(normalized)
    valid = len(normalized)
    output.update({
        "candidate_count": float(len(candidates)),
        "valid_answer_count": float(valid),
        "empty_answer_count": float(len(candidates) - valid),
        "unique_normalized_answer_count": float(len(counts)),
        "answer_diversity": float(len(counts) / max(valid, 1)),
        "largest_exact_answer_cluster_ratio": float(max(counts.values(), default=0) / max(valid, 1)),
    })
    _finite_vector(output, QUESTION_AGG_COLUMNS, "question aggregate")
    return output


def make_questions(runtime: list[dict], hrr_rows: list[dict], targets: dict) -> list[dict]:
    """Build all candidate vectors. Targets stay under evaluation-only keys."""
    runtime_groups = grouped(runtime)
    row_groups = grouped(hrr_rows)
    output = []
    for question_id, values in sorted(runtime_groups.items()):
        key = (str(values[0]["dataset_cell"]), question_id)
        target = targets[key]
        hrows = {int(x["chunk_id"]): x for x in row_groups.get(question_id, [])}
        candidates = []
        for index, raw in enumerate(values):
            chunk_id = int(raw["chunk_id"])
            if index == 0:
                vector = _base_vector(raw)
            else:
                if chunk_id not in hrows: raise RuntimeError(f"missing exact H2 row: {key}/{chunk_id}")
                vector = {name: float(hrows[chunk_id][name]) for name in H2_COLUMNS}
            _finite_vector(vector, H2_COLUMNS, "H2 candidate")
            tier = target["label_tier"].get(str(chunk_id))
            evidence = evidence_class(tier)
            candidates.append({
                "chunk_id": chunk_id, "rank": int(raw["b0_rank"]),
                "b0_score": float(raw["b0_score_raw"]),
                "answer": str(raw.get("qa_answer") or ""), "x": vector,
                "evaluation": {
                    "f1": float(target["candidate_f1"][str(chunk_id)]),
                    "tier": tier, "evidence_class": evidence,
                    "hn": int(target["hard_negative"][str(chunk_id)]),
                    "chunk_label": target.get("chunk_label", {}).get(str(chunk_id)),
                    "primary_class": target.get("primary_class", {}).get(str(chunk_id)),
                },
            })
        q = question_aggregates(candidates)
        base_f1 = candidates[0]["evaluation"]["f1"]
        question = {
            "cell": key[0], "question_id": question_id,
            "document_id": str(values[0]["document_id"]),
            "canonical_document_id": str(values[0]["canonical_document_id"]),
            "document_sha256": str(values[0]["document_sha256"]),
            "question_lineage_id": str(values[0]["question_lineage_id"]),
            "canonical_aspect": str(values[0]["canonical_aspect"]),
            "language": str(values[0]["language"]),
            "outer_fold": int(values[0]["outer_fold"]), "seed": int(values[0]["seed"]),
            "candidates": candidates, "q": q,
            "evaluation": {
                "gold_answers": list(target["gold_answers"]),
                "positive_opportunity": bool(any(x["evaluation"]["f1"] > base_f1 for x in candidates[1:])),
            },
        }
        output.append(question)
    return output


def runtime_question_view(question: dict) -> dict:
    value = {key: question[key] for key in (
        "cell", "question_id", "document_id", "canonical_document_id",
        "document_sha256", "question_lineage_id", "outer_fold", "seed",
    )}
    value["candidate_ids"] = [x["chunk_id"] for x in question["candidates"]]
    value["candidate_answers"] = [x["answer"] for x in question["candidates"]]
    value["h2_vectors"] = [{name: x["x"][name] for name in H2_COLUMNS} for x in question["candidates"]]
    value["q"] = dict(question["q"])
    assert_runtime_gold_free(value)
    return value


def inner_manifest(questions: list[dict], folds=INNER_FOLDS) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold

    documents = sorted({x["canonical_document_id"] for x in questions})
    if len(documents) < folds: raise RuntimeError(f"INNER_SPLIT_BLOCKED: only {len(documents)} documents")
    ordered = sorted(questions, key=lambda x: (x["canonical_document_id"], x["question_id"]))
    labels = np.asarray([
        int(any(c["evaluation"]["evidence_class"] == "UNSAFE_EVIDENCE" for c in q["candidates"][1:]))
        for q in ordered
    ], dtype=int)
    groups = np.asarray([q["canonical_document_id"] for q in ordered])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=INNER_SPLIT_SEED)
    assignment = {}
    for fold, (_, valid_indices) in enumerate(splitter.split(np.zeros((len(ordered), 1)), labels, groups)):
        for index in valid_indices:
            document = str(groups[index])
            previous = assignment.setdefault(document, fold)
            if previous != fold: raise AssertionError("document assigned to multiple inner folds")
    if set(assignment) != set(documents): raise RuntimeError("INNER_SPLIT_BLOCKED: incomplete document assignment")
    missing = set(range(folds)) - set(assignment.values())
    if missing: raise RuntimeError(f"INNER_SPLIT_BLOCKED: empty folds {sorted(missing)}")
    payload = {
        "folds": folds, "split_seed": INNER_SPLIT_SEED, "reference_model_seed": 42,
        "splitter": "StratifiedGroupKFold",
        "stratification_target": "question_has_unsafe_evidence_alternative",
        "assignment": assignment,
        "fold_document_counts": {str(f): int(sum(v == f for v in assignment.values())) for f in range(folds)},
        "fold_question_counts": {str(f): int(sum(assignment[q["canonical_document_id"]] == f for q in ordered)) for f in range(folds)},
        "fold_unsafe_question_counts": {str(f): int(sum(assignment[q["canonical_document_id"]] == f and int(label) == 1 for q, label in zip(ordered, labels))) for f in range(folds)},
    }
    payload["sha256"] = sha256_json(payload)
    return payload


def feature_row(prefix: str, vector: dict) -> dict:
    return {f"{prefix}{name}": float(vector[name]) for name in H2_COLUMNS}


def switch_rows(questions: list[dict]) -> list[dict]:
    rows = []
    for question in questions:
        if len(question["candidates"]) < 2: continue
        row = {name: float(question["q"][name]) for name in SWITCH_COLUMNS}
        row.update(question_id=question["question_id"], canonical_document_id=question["canonical_document_id"],
                   target=int(question["evaluation"]["positive_opportunity"]), weight=1.0)
        rows.append(row)
    return rows


def pair_rows(questions: list[dict]) -> list[dict]:
    rows = []
    for question in questions:
        candidates = question["candidates"]
        unordered = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                gap = candidates[i]["evaluation"]["f1"] - candidates[j]["evaluation"]["f1"]
                if abs(gap) < PAIR_MIN_GAP: continue
                unordered.append((candidates[i], candidates[j], int(gap > 0)))
        if not unordered: continue
        weight = 1.0 / (2.0 * len(unordered))
        for left, right, target in unordered:
            for a, b, y in ((left, right, target), (right, left, 1 - target)):
                row = {f"delta__{name}": float(a["x"][name] - b["x"][name]) for name in H2_COLUMNS}
                row.update({name: float(question["q"][name]) for name in QUESTION_AGG_COLUMNS})
                row.update(question_id=question["question_id"], canonical_document_id=question["canonical_document_id"],
                           left_chunk_id=a["chunk_id"], right_chunk_id=b["chunk_id"], target=y, weight=weight)
                _finite_vector(row, RANK_COLUMNS, "rank row")
                rows.append(row)
    return rows


def risk_rows(questions: list[dict], evidence=False) -> list[dict]:
    rows = []
    for question in questions:
        candidates = question["candidates"]
        if len(candidates) < 2: continue
        base = candidates[0]
        eligible = [x for x in candidates[1:] if not evidence or x["evaluation"]["evidence_class"] != "EXCLUDE"]
        denominator = max(len(eligible), 1)
        for candidate in candidates[1:]:
            if evidence:
                klass = candidate["evaluation"]["evidence_class"]
                sample_weight = 0.0 if klass == "EXCLUDE" else 1.0 / denominator
                target = int(klass == "UNSAFE_EVIDENCE") if klass != "EXCLUDE" else 0
                row = feature_row("candidate__", candidate["x"])
            else:
                sample_weight = 1.0 / len(candidates[1:])
                target = int(candidate["evaluation"]["f1"] < base["evaluation"]["f1"])
                row = {}
            row.update({f"delta__{name}": float(candidate["x"][name] - base["x"][name]) for name in H2_COLUMNS})
            row.update({name: float(question["q"][name]) for name in QUESTION_AGG_COLUMNS})
            row.update(question_id=question["question_id"], canonical_document_id=question["canonical_document_id"],
                       chunk_id=candidate["chunk_id"], target=target, weight=sample_weight)
            _finite_vector(row, EVIDENCE_RISK_COLUMNS if evidence else ANSWER_RISK_COLUMNS, "risk row")
            rows.append(row)
    return rows


def assert_feature_schema():
    for columns in (H2_COLUMNS, QUESTION_AGG_COLUMNS, SWITCH_COLUMNS, RANK_COLUMNS, ANSWER_RISK_COLUMNS, EVIDENCE_RISK_COLUMNS):
        if len(columns) != len(set(columns)): raise AssertionError("duplicate feature column")
    forbidden = ("dataset", "language", "aspect", "gold", "oracle", "utility", "evidence_label")
    for columns in (SWITCH_COLUMNS, RANK_COLUMNS, ANSWER_RISK_COLUMNS, EVIDENCE_RISK_COLUMNS):
        if any(any(token in name.lower() for token in forbidden) for name in columns):
            raise AssertionError(f"forbidden feature in schema: {columns}")
