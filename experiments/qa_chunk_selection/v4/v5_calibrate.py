"""V5: lightweight learned cross-passage calibration. Softmax-only is a control."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.hn_flags import is_training_eligible_hn as hn_eligible

from .heads import LinearHead, rank_by_scores, softmax
from .shortlist import scores_from_row, top_k_ids


FEATURE_NAMES = (
    "b0_score",
    "b0_rank",
    "qa_raw",
    "qa_margin",
    "rel_qa",
    "rel_b0",
    "span_len",
    "cand_len",
    "rel_pos",
)


def features_for_shortlist(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    k: int,
) -> dict[int, list[float]]:
    ranked = top_k_ids(b0_row, k)
    b0 = scores_from_row(b0_row)
    rel_b0 = softmax([b0.get(int(cid), 0.0) for cid in ranked])
    rel_qa = softmax([float((qa_by_chunk.get(int(cid)) or {}).get("qa_raw_score") or 0.0) for cid in ranked])
    out: dict[int, list[float]] = {}
    n_chunks = max(len(item.chunks), 1)
    for rank, cid in enumerate(ranked):
        chunk = item.chunk_by_id(int(cid))
        qa = qa_by_chunk.get(int(cid)) or {}
        start = float(qa.get("qa_chunk_start_char") or 0.0)
        end = float(qa.get("qa_chunk_end_char") or 0.0)
        out[int(cid)] = [
            float(b0.get(int(cid), 0.0)),
            float(rank + 1),
            float(qa.get("qa_raw_score") or 0.0),
            float(qa.get("qa_margin") or 0.0),
            float(rel_qa[rank]) if rank < len(rel_qa) else 0.0,
            float(rel_b0[rank]) if rank < len(rel_b0) else 0.0,
            max(end - start, 0.0),
            float(len(chunk.text)),
            float(chunk.position) / float(n_chunks),
        ]
    return out


def collect_training(
    records: Sequence[InstanceRecord],
    b0_by_qid: Mapping[str, Mapping[str, Any]],
    qa_store: Mapping[tuple[str, int], Mapping[str, Any]],
    flags,
    *,
    k: int,
    easy_cap: int,
) -> tuple[list[list[float]], list[int]]:
    matrix: list[list[float]] = []
    labels: list[int] = []
    for item in records:
        row = b0_by_qid.get(item.question_id)
        if row is None:
            continue
        qa_by_chunk = {cid: qa_store[(item.question_id, cid)] for cid in top_k_ids(row, k) if (item.question_id, cid) in qa_store}
        feats = features_for_shortlist(item, row, qa_by_chunk, k=k)
        easy_used = 0
        for cid, vec in feats.items():
            chunk = item.chunk_by_id(int(cid))
            if chunk.label == "positive":
                matrix.append(vec)
                labels.append(1)
            elif chunk.label == "hard_negative" and (not flags or hn_eligible(flags, item.question_id, int(cid))):
                matrix.append(vec)
                labels.append(0)
            elif chunk.label == "easy_negative" and easy_used < int(easy_cap):
                matrix.append(vec)
                labels.append(0)
                easy_used += 1
            # ambiguous stays unlabeled
    return matrix, labels


def apply_v5(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    variant: str,
    k: int,
    head: LinearHead | None,
) -> tuple[list[int], dict[int, float]]:
    b0 = scores_from_row(b0_row)
    ranked = top_k_ids(b0_row, k)
    if variant == "V5-A0":
        scores = {int(cid): float((qa_by_chunk.get(int(cid)) or {}).get("qa_raw_score") or 0.0) for cid in ranked}
    elif variant == "V5-A1":
        # monotone: ranking identical to B0
        scores = {int(cid): float(b0.get(int(cid), 0.0)) for cid in ranked}
        _ = softmax(list(scores.values()))
    elif variant == "V5-A2":
        rel_qa = softmax([float((qa_by_chunk.get(int(cid)) or {}).get("qa_raw_score") or 0.0) for cid in ranked])
        scores = {int(cid): float(b0.get(int(cid), 0.0)) + 0.5 * rel for cid, rel in zip(ranked, rel_qa)}
    else:
        feats = features_for_shortlist(item, b0_row, qa_by_chunk, k=k)
        scores = {cid: float(head.score(vec) if head is not None else 0.0) for cid, vec in feats.items()}
    for chunk in item.chunks:
        scores.setdefault(int(chunk.chunk_id), float(b0.get(int(chunk.chunk_id), -1e9)))
    ids = [int(chunk.chunk_id) for chunk in item.chunks]
    return rank_by_scores(ids, [scores[cid] for cid in ids]), scores
