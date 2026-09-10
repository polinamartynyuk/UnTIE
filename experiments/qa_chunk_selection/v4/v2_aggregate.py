"""V2: deterministic answer aggregation over B0 top-K."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord

from .answers import normalize_predicted_answer
from .clusters import cluster_id_by_chunk, shortlist_source_clusters
from .config import FUSION_LAMBDAS
from .heads import softmax
from .shortlist import scores_from_row, top_k_ids


VARIANTS = ("V2-A0", "V2-A1", "V2-A2", "V2-A3", "V2-A4", "V2-A5")


def aggregate(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    variant: str,
    k: int,
    lam: float,
    min_overlap: float,
) -> dict[str, Any]:
    ranked = top_k_ids(b0_row, k)
    b0 = scores_from_row(b0_row)
    if variant == "V2-A0":
        cid = int(ranked[0]) if ranked else int(item.chunks[0].chunk_id)
        qa = qa_by_chunk.get(cid) or {}
        return {
            "answer": str(qa.get("qa_answer") or ""),
            "supporting_chunk_ids": [cid],
            "provenance_chunk_id": cid,
            "n_source_clusters": 1,
            "source_labels": [item.chunk_by_id(cid).label],
            "scores": b0,
            "qa": dict(qa),
        }
    groups: dict[str, list[int]] = defaultdict(list)
    for cid in ranked:
        qa = qa_by_chunk.get(int(cid)) or {}
        key = str(qa.get("normalized_answer") or normalize_predicted_answer(str(qa.get("qa_answer") or "")))
        if not key:
            continue
        groups[key].append(int(cid))
    if not groups:
        return aggregate(item, b0_row, qa_by_chunk, variant="V2-A0", k=k, lam=lam, min_overlap=min_overlap)
    rel_qa = softmax([float((qa_by_chunk.get(cid) or {}).get("qa_raw_score") or 0.0) for cid in ranked])
    rel_map = {int(cid): float(score) for cid, score in zip(ranked, rel_qa)}
    scored: list[tuple[float, str, list[int]]] = []
    for key, cids in groups.items():
        clusters = shortlist_source_clusters(item, cids, min_overlap=min_overlap)
        n_clusters = len({row["evidence_cluster_id"] for row in clusters})
        if variant == "V2-A1":
            score = float(len(cids))
        elif variant == "V2-A2":
            score = float(n_clusters)
        elif variant == "V2-A3":
            score = float(sum(b0.get(cid, 0.0) for cid in cids))
        elif variant == "V2-A4":
            score = max((b0.get(cid, 0.0) for cid in cids), default=0.0) + float(lam) * __import__("math").log(1.0 + n_clusters)
        else:
            score = max((rel_map.get(cid, 0.0) for cid in cids), default=0.0) + float(lam) * __import__("math").log(1.0 + n_clusters)
        scored.append((score, key, cids))
    scored.sort(key=lambda item: (-item[0], item[1]))
    _score, key, cids = scored[0]
    provenance = max(cids, key=lambda cid: b0.get(int(cid), 0.0))
    answer = str((qa_by_chunk.get(int(cids[0])) or {}).get("qa_answer") or "")
    answer_cid = int(cids[0])
    for cid in cids:
        raw = str((qa_by_chunk.get(int(cid)) or {}).get("qa_answer") or "")
        if normalize_predicted_answer(raw) == key and raw:
            answer = raw
            answer_cid = int(cid)
            break
    clusters = shortlist_source_clusters(item, cids, min_overlap=min_overlap)
    return {
        "answer": answer,
        "supporting_chunk_ids": cids,
        "provenance_chunk_id": int(provenance),
        "n_source_clusters": len({row["evidence_cluster_id"] for row in clusters}),
        "source_labels": [item.chunk_by_id(int(cid)).label for cid in cids],
        "scores": b0,
        "qa": dict(qa_by_chunk.get(int(answer_cid)) or {}),
    }
