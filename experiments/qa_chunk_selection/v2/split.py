"""Document-level V2 splits from official TRAIN before answerable filtering."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from experiments.qa_chunk_selection.metrics import hn_bucket
from experiments.qa_chunk_selection.schema import InstanceRecord

from .contract import V2_SPLITS

ROLE_ORDER = ("train", "calibration", "development", "internal_test")


def split_document_ids(
    records: Sequence[InstanceRecord],
    *,
    seed: int,
    fractions: tuple[float, float, float, float],
) -> dict[str, list[str]]:
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("V2 split fractions must sum to 1")
    by_doc: dict[str, list[InstanceRecord]] = defaultdict(list)
    for item in records:
        by_doc[item.document_id].append(item)
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for document_id, items in by_doc.items():
        aspects = Counter(item.aspect for item in items)
        aspect = aspects.most_common(1)[0][0] if aspects else "unknown"
        answerable = int(any(item.positives for item in items))
        strata[(aspect, answerable)].append(document_id)
    rng = np.random.default_rng(int(seed))
    assigned: dict[str, str] = {}
    for key in sorted(strata):
        docs = sorted(strata[key])
        rng.shuffle(docs)
        n = len(docs)
        cuts = _prefix_cuts(n, fractions)
        for role, (lo, hi) in zip(ROLE_ORDER, cuts):
            for document_id in docs[lo:hi]:
                assigned[document_id] = role
    payload = {role: sorted(doc for doc, role_name in assigned.items() if role_name == role) for role in ROLE_ORDER}
    payload = _ensure_nonempty_roles(payload)
    _assert_no_overlap(payload)
    return payload


def _ensure_nonempty_roles(payload: dict[str, list[str]]) -> dict[str, list[str]]:
    """Keep group-by-document assignment; steal from TRAIN if a holdout role is empty."""
    for role in ("calibration", "development", "internal_test"):
        if payload[role]:
            continue
        if len(payload["train"]) <= 1:
            continue
        moved = payload["train"].pop()
        payload[role].append(moved)
        payload[role] = sorted(payload[role])
        payload["train"] = sorted(payload["train"])
    return payload


def _prefix_cuts(n: int, fractions: Sequence[float]) -> list[tuple[int, int]]:
    raw = [int(round(n * float(item))) for item in fractions]
    drift = n - sum(raw)
    # Adjust the last bucket, then earlier ones if still off.
    raw[-1] = max(0, raw[-1] + drift)
    if sum(raw) != n:
        raw[0] += n - sum(raw)
    raw = [max(0, item) for item in raw]
    while sum(raw) > n:
        for index in range(len(raw) - 1, -1, -1):
            if raw[index] > 0:
                raw[index] -= 1
                break
    while sum(raw) < n:
        raw[0] += 1
    cuts = []
    cursor = 0
    for count in raw:
        cuts.append((cursor, cursor + count))
        cursor += count
    return cuts


def _assert_no_overlap(payload: dict[str, list[str]]) -> None:
    seen: set[str] = set()
    for role, docs in payload.items():
        overlap = seen & set(docs)
        if overlap:
            raise ValueError(f"document leakage into {role}: {sorted(overlap)[:5]}")
        seen.update(docs)


def apply_document_roles(
    records: Sequence[InstanceRecord],
    roles: dict[str, list[str]],
    *,
    official_dev: Sequence[InstanceRecord],
) -> tuple[InstanceRecord, ...]:
    lookup = {document_id: role for role, docs in roles.items() for document_id in docs}
    out: list[InstanceRecord] = []
    for item in records:
        role = lookup.get(item.document_id)
        if role is None:
            raise KeyError(f"document {item.document_id} has no V2 role")
        out.append(_with_split(item, role))
    for item in official_dev:
        out.append(_with_split(item, "seen_exploratory_dev"))
    return tuple(out)


def _with_split(item: InstanceRecord, split: str) -> InstanceRecord:
    return InstanceRecord(
        document_id=item.document_id,
        question_id=item.question_id,
        aspect=item.aspect,
        question=item.question,
        gold_answers=item.gold_answers,
        split=split,
        chunks=item.chunks,
        gold_spans=item.gold_spans,
        instance_status=item.instance_status,
        source_split=item.source_split,
    )


def split_sanity(records: Sequence[InstanceRecord]) -> dict[str, Any]:
    by_split: dict[str, list[InstanceRecord]] = defaultdict(list)
    for item in records:
        by_split[item.split].append(item)
    summary: dict[str, Any] = {}
    for split in (*ROLE_ORDER, "seen_exploratory_dev"):
        rows = by_split.get(split, [])
        docs = {item.document_id for item in rows}
        answerable = [item for item in rows if item.positives]
        summary[split] = {
            "n_documents": len(docs),
            "n_questions": len(rows),
            "n_answerable": len(answerable),
            "n_no_robust_positive": len(rows) - len(answerable),
            "aspects": dict(Counter(item.aspect for item in rows)),
            "n_positive": sum(len(item.positives) for item in rows),
            "n_hard_negative": sum(len(item.hard_negatives) for item in rows),
            "n_easy_negative": sum(len(item.easy_negatives) for item in rows),
            "n_ambiguous": sum(sum(chunk.label == "ambiguous" for chunk in item.chunks) for item in rows),
            "hn_buckets": dict(Counter(hn_bucket(len(item.hard_negatives)) for item in answerable)),
        }
    docs_by_split = {split: {item.document_id for item in rows} for split, rows in by_split.items()}
    for left in V2_SPLITS:
        for right in V2_SPLITS:
            if left >= right:
                continue
            overlap = docs_by_split.get(left, set()) & docs_by_split.get(right, set())
            if overlap:
                raise ValueError(f"document_id overlap {left} ∩ {right}: {sorted(overlap)[:5]}")
    return summary
