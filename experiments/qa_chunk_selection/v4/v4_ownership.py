"""V4 diagnostic: anchored own-document vs synthetic wrong-document contrast."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import binary_pr_auc, binary_roc_auc
from experiments.qa_chunk_selection.schema import InstanceRecord

from .keybert_features import cosine, document_profile_phrases
from .qa_shortlist import answer_one


def profile_vector(phrases: Sequence[str], embeddings: Mapping[str, Sequence[float]]) -> list[float]:
    vecs = [list(embeddings[p]) for p in phrases if p in embeddings]
    if not vecs:
        return []
    dim = len(vecs[0])
    return [sum(row[i] for row in vecs) / float(len(vecs)) for i in range(dim)]


def pair_documents(
    records: Sequence[InstanceRecord],
    profiles: Mapping[str, Sequence[float]],
    *,
    seed: int,
) -> list[tuple[InstanceRecord, InstanceRecord, float]]:
    by_aspect_split: dict[tuple[str, str], list[InstanceRecord]] = {}
    for item in records:
        if not item.positives:
            continue
        by_aspect_split.setdefault((item.aspect, item.split), []).append(item)
    pairs = []
    for _key, items in by_aspect_split.items():
        for item in items:
            best = None
            best_sim = -1.0
            a = profiles.get(item.document_id) or []
            for other in items:
                if other.document_id == item.document_id:
                    continue
                if other.split != item.split:
                    continue
                sim = cosine(a, profiles.get(other.document_id) or [])
                if sim > best_sim:
                    best_sim = sim
                    best = other
            if best is not None:
                pairs.append((item, best, float(best_sim)))
    pairs.sort(key=lambda row: (-row[2], row[0].question_id, row[1].question_id))
    del seed
    return pairs


def anchor_prefix(phrases: Sequence[str]) -> str:
    joined = "; ".join(phrases[:12])
    return f"Target-document keyphrases: {joined}. Question: "


def probe_pair(
    target: InstanceRecord,
    candidate: InstanceRecord,
    candidate_chunk_id: int,
    answer_fn: Callable[[str, str], dict[str, Any]],
    *,
    cache,
    model_id: str,
    inference_hash: str,
    phrases: Sequence[str],
) -> dict[str, Any]:
    question = target.question
    anchored = anchor_prefix(phrases) + question
    plain = answer_one(
        candidate,
        candidate_chunk_id,
        answer_fn,
        cache=cache,
        model_id=model_id,
        inference_hash=inference_hash + ":plain",
        question=question,
    )
    focused = answer_one(
        candidate,
        candidate_chunk_id,
        answer_fn,
        cache=cache,
        model_id=model_id,
        inference_hash=inference_hash + ":anchor",
        question=anchored,
    )
    return {
        "plain_score": float(plain.get("qa_raw_score") or 0.0),
        "anchored_score": float(focused.get("qa_raw_score") or 0.0),
        "plain_answer": plain.get("qa_answer"),
        "anchored_answer": focused.get("qa_answer"),
    }


def summarize_diagnostic(target_scores: Sequence[float], mismatch_scores: Sequence[float]) -> dict[str, Any]:
    labels = [1] * len(target_scores) + [0] * len(mismatch_scores)
    scores = list(target_scores) + list(mismatch_scores)
    pairwise = 0.0
    n = 0
    for t in target_scores:
        for m in mismatch_scores:
            n += 1
            pairwise += float(t > m) + 0.5 * float(t == m)
    return {
        "n_target": len(target_scores),
        "n_mismatch": len(mismatch_scores),
        "auroc": binary_roc_auc(labels, scores),
        "auprc": binary_pr_auc(labels, scores),
        "pairwise_target_gt_mismatch": pairwise / max(n, 1),
        "mean_target": sum(target_scores) / max(len(target_scores), 1),
        "mean_mismatch": sum(mismatch_scores) / max(len(mismatch_scores), 1),
    }
