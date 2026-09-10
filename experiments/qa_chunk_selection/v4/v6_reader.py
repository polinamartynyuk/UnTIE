"""V6: reader-centered passage rerank from answer agreement."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord

from .answers import answers_match, majority_answer, pair_char_f1, pair_token_f1
from .clusters import shortlist_source_clusters
from .config import FUSION_LAMBDAS
from .heads import LinearHead, rank_by_scores
from .shortlist import scores_from_row, top_k_ids


def agreement_features(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    k: int,
    min_overlap: float,
) -> dict[int, dict[str, float]]:
    ranked = top_k_ids(b0_row, k)
    b0 = scores_from_row(b0_row)
    answers = [str((qa_by_chunk.get(int(cid)) or {}).get("qa_answer") or "") for cid in ranked]
    majority = majority_answer(answers)
    out: dict[int, dict[str, float]] = {}
    for cid in ranked:
        qa = qa_by_chunk.get(int(cid)) or {}
        ans = str(qa.get("qa_answer") or "")
        others = [str((qa_by_chunk.get(int(other)) or {}).get("qa_answer") or "") for other in ranked if int(other) != int(cid)]
        exact = sum(1.0 for other in others if answers_match(ans, other))
        tok = [pair_token_f1(ans, other) for other in others]
        ch = [pair_char_f1(ans, other) for other in others]
        same = [other for other in ranked if answers_match(ans, str((qa_by_chunk.get(int(other)) or {}).get("qa_answer") or ""))]
        clusters = shortlist_source_clusters(item, same or [cid], min_overlap=min_overlap)
        out[int(cid)] = {
            "b0_score": float(b0.get(int(cid), 0.0)),
            "b0_rank": float(ranked.index(int(cid)) + 1),
            "qa_raw": float(qa.get("qa_raw_score") or 0.0),
            "qa_margin": float(qa.get("qa_margin") or 0.0),
            "exact_freq": exact,
            "mean_token_f1": float(sum(tok) / len(tok)) if tok else 0.0,
            "mean_char_f1": float(sum(ch) / len(ch)) if ch else 0.0,
            "source_clusters": float(len({row["evidence_cluster_id"] for row in clusters})),
            "majority": float(answers_match(ans, majority)),
            "unique": float(exact <= 0),
        }
    return out


def apply_v6(
    item: InstanceRecord,
    b0_row: Mapping[str, Any],
    qa_by_chunk: Mapping[int, Mapping[str, Any]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    k: int,
    min_overlap: float,
    head: LinearHead | None = None,
) -> tuple[list[int], dict[int, float]]:
    b0 = scores_from_row(b0_row)
    feats = agreement_features(item, b0_row, qa_by_chunk, k=k, min_overlap=min_overlap)
    scores: dict[int, float] = {}
    for cid, row in feats.items():
        if head is not None:
            names = head.feature_names or list(row)
            scores[int(cid)] = head.score([float(row.get(name, 0.0)) for name in names])
        else:
            scores[int(cid)] = (
                float(alpha) * row["b0_score"]
                + float(beta) * (row["exact_freq"] + row["mean_token_f1"] + row["majority"])
                + float(gamma) * row["source_clusters"]
            )
    for chunk in item.chunks:
        scores.setdefault(int(chunk.chunk_id), float(b0.get(int(chunk.chunk_id), -1e9)))
    ids = [int(chunk.chunk_id) for chunk in item.chunks]
    return rank_by_scores(ids, [scores[cid] for cid in ids]), scores


def choose_v6_weights(cal_items, b0_by_qid, qa_store, *, k: int, min_overlap: float) -> tuple[float, float, float]:
    best = (1.0, 1.0, 0.5)
    best_hit = -1.0
    for alpha in FUSION_LAMBDAS:
        for beta in FUSION_LAMBDAS:
            gamma = 0.5
            hits = 0
            n = 0
            for item in cal_items:
                row = b0_by_qid.get(item.question_id)
                if row is None:
                    continue
                qa_by_chunk = {
                    cid: qa_store[(item.question_id, cid)]
                    for cid in top_k_ids(row, k)
                    if (item.question_id, cid) in qa_store
                }
                ranked, _ = apply_v6(item, row, qa_by_chunk, alpha=alpha, beta=beta, gamma=gamma, k=k, min_overlap=min_overlap)
                n += 1
                pos = {int(chunk.chunk_id) for chunk in item.positives}
                hits += int(bool(ranked and ranked[0] in pos))
            score = hits / max(n, 1)
            if score > best_hit:
                best_hit = score
                best = (float(alpha), float(beta), float(gamma))
    return best
