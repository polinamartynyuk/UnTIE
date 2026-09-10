"""Document-level grouped 5-fold CV for unlocked V2 documents."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Sequence

from experiments.qa_chunk_selection.metrics import hn_bucket
from experiments.qa_chunk_selection.schema import InstanceRecord

from .io_util import write_json


def document_features(records: Sequence[InstanceRecord]) -> dict[str, dict[str, Any]]:
    by_doc: dict[str, list[InstanceRecord]] = defaultdict(list)
    for item in records:
        by_doc[item.document_id].append(item)
    features: dict[str, dict[str, Any]] = {}
    for document_id, items in by_doc.items():
        n_answerable = sum(1 for item in items if item.positives)
        n_positive = sum(len(item.positives) for item in items)
        n_ambiguous = sum(1 for item in items for chunk in item.chunks if chunk.label == "ambiguous")
        n_hn = max((len(item.hard_negatives) for item in items), default=0)
        aspect_counts = Counter(item.aspect for item in items)
        dominant = sorted(aspect_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if aspect_counts else "none"
        features[document_id] = {
            "document_id": document_id,
            "n_questions": len(items),
            "n_answerable": n_answerable,
            "n_robust_positive": n_positive,
            "n_ambiguous": n_ambiguous,
            "hn_bucket": hn_bucket(n_hn),
            "dominant_aspect": dominant,
            "aspects": dict(aspect_counts),
            "stratum": _stratum(n_answerable, dominant, n_positive, hn_bucket(n_hn), n_ambiguous),
        }
    return features


def _stratum(n_answerable: int, aspect: str, n_pos: int, hn: str, n_amb: int) -> str:
    ans = "1" if n_answerable <= 1 else ("2" if n_answerable == 2 else "3+")
    pos = "0" if n_pos <= 0 else ("1" if n_pos == 1 else ("2-3" if n_pos <= 3 else "4+"))
    amb = "0" if n_amb <= 0 else ("1-2" if n_amb <= 2 else "3+")
    return f"ans{ans}|{aspect}|pos{pos}|hn{hn}|amb{amb}"


def assign_folds(
    document_ids: Sequence[str],
    features: dict[str, dict[str, Any]],
    *,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    n_folds = max(int(n_folds), 2)
    grouped: dict[str, list[str]] = defaultdict(list)
    for document_id in document_ids:
        grouped[features[document_id]["stratum"]].append(document_id)
    assignment: dict[str, int] = {}
    fold_sizes = [0] * n_folds
    # Deterministic: sort strata, then document_id. Greedy to the current smallest fold,
    # breaking ties by fold index — not by a hash of seed-only which would ignore ids.
    rng_order = sorted(grouped)
    cursor = int(seed) % n_folds
    for stratum in rng_order:
        docs = sorted(grouped[stratum])
        for document_id in docs:
            fold = min(range(n_folds), key=lambda i: (fold_sizes[i], (i + cursor) % n_folds))
            assignment[document_id] = fold
            fold_sizes[fold] += 1
    return assignment


def cyclic_roles(n_folds: int) -> list[dict[str, list[int]]]:
    """Match the protocol example: eval=F_(4+r)%5, cal=F_(3+r)%5 for 5 folds."""
    runs = []
    n = int(n_folds)
    for run in range(n):
        eval_fold = (n - 1 + run) % n
        cal_fold = (n - 2 + run) % n
        train = [fold for fold in range(n) if fold not in {eval_fold, cal_fold}]
        runs.append({"run": run, "train": train, "calibration": [cal_fold], "eval": [eval_fold]})
    return runs


def build_cv_splits(
    records: Sequence[InstanceRecord],
    *,
    n_folds: int,
    seed: int,
    output_path=None,
) -> dict[str, Any]:
    docs = sorted({item.document_id for item in records})
    features = document_features(records)
    n_folds_eff = min(max(int(n_folds), 2), max(len(docs), 2))
    assignment = assign_folds(docs, features, n_folds=n_folds_eff, seed=seed)
    roles = cyclic_roles(n_folds_eff)
    folds: dict[str, list[str]] = {str(i): [] for i in range(n_folds_eff)}
    for document_id, fold in assignment.items():
        folds[str(fold)].append(document_id)
    for key in folds:
        folds[key] = sorted(folds[key])
    doc_hash = hashlib.sha256("\n".join(docs).encode("utf-8")).hexdigest()
    payload = {
        "split_seed": int(seed),
        "n_folds": n_folds_eff,
        "n_documents": len(docs),
        "algorithm": "greedy_grouped_stratification_by_stratum_then_smallest_fold",
        "grouping_unit": "document_id",
        "document_id_hash": doc_hash,
        "folds": folds,
        "assignment": assignment,
        "runs": roles,
        "per_fold": {
            str(i): _fold_summary(features, folds[str(i)]) for i in range(n_folds_eff)
        },
        "frozen": True,
    }
    encoded = json.dumps(
        {"seed": seed, "docs": docs, "folds": folds, "runs": roles, "algorithm": payload["algorithm"]},
        sort_keys=True,
    )
    payload["split_payload_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def _fold_summary(features: dict[str, dict[str, Any]], docs: Sequence[str]) -> dict[str, Any]:
    items = [features[doc] for doc in docs if doc in features]
    aspects: Counter[str] = Counter()
    for row in items:
        aspects.update(row.get("aspects") or {})
    return {
        "n_documents": len(docs),
        "n_answerable": int(sum(row["n_answerable"] for row in items)),
        "n_robust_positive": int(sum(row["n_robust_positive"] for row in items)),
        "n_ambiguous": int(sum(row["n_ambiguous"] for row in items)),
        "aspects": dict(aspects),
        "hn_buckets": dict(Counter(row["hn_bucket"] for row in items)),
    }


def records_for_role(
    records: Sequence[InstanceRecord],
    splits: MappingLike,
    *,
    run: int,
    role: str,
) -> tuple[InstanceRecord, ...]:
    fold_ids = {int(item) for item in splits["runs"][int(run)][role]}
    assignment = splits["assignment"]
    return tuple(item for item in records if int(assignment[item.document_id]) in fold_ids)


# typing alias without importing Mapping in signature noise
MappingLike = dict[str, Any]
