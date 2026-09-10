"""Shortlist semantic features. Gold is not used."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.qa_chunk_selection.v4.answers import answers_match, pair_token_f1
from experiments.qa_chunk_selection.v4_2.features import zscore_map

from .contract import (
    CLUSTER_THRESHOLDS,
    MIN_VALID_ANSWERS_ADAPT,
    MIN_VALID_ANSWERS_SEM,
    Z_EPS,
)
from .text import cosine, is_valid_answer, semantic_normalize


def similarity_matrix(embeddings: Mapping[int, np.ndarray], ids: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    valid = [int(cid) for cid in ids if int(cid) in embeddings]
    for i, left in enumerate(valid):
        for right in valid[i + 1 :]:
            sim = cosine(embeddings[left], embeddings[right], eps=Z_EPS)
            out[f"{left}|{right}"] = sim
            out[f"{right}|{left}"] = sim
        out[f"{left}|{left}"] = cosine(embeddings[left], embeddings[left], eps=Z_EPS)
    return out


def sim_at(matrix: Mapping[str, float], left: int, right: int) -> float:
    return float(matrix.get(f"{int(left)}|{int(right)}") or 0.0)


def sem_top2(matrix: Mapping[str, float], cid: int, valid_ids: Sequence[int]) -> float | None:
    others = [int(other) for other in valid_ids if int(other) != int(cid)]
    if len(others) < 2:
        return None
    sims = sorted((sim_at(matrix, cid, other) for other in others), reverse=True)
    return float((sims[0] + sims[1]) / 2.0)


def sem_mean(matrix: Mapping[str, float], cid: int, valid_ids: Sequence[int]) -> float | None:
    others = [int(other) for other in valid_ids if int(other) != int(cid)]
    if not others:
        return None
    return float(sum(sim_at(matrix, cid, other) for other in others) / len(others))


def best_coherent_triple(matrix: Mapping[str, float], valid_ids: Sequence[int]) -> dict[str, Any]:
    ids = [int(cid) for cid in valid_ids]
    if len(ids) < 3:
        return {"triple": [], "cohesion": None, "between": None, "consensus_conf": None}
    best = None
    best_key = None
    for triple in combinations(sorted(ids), 3):
        pairs = [sim_at(matrix, triple[0], triple[1]), sim_at(matrix, triple[0], triple[2]), sim_at(matrix, triple[1], triple[2])]
        coh = float(sum(pairs) / 3.0)
        key = (-coh, triple)
        if best is None or key < best_key:
            best = (triple, coh)
            best_key = key
    triple, coh = best
    outside = [cid for cid in ids if cid not in set(triple)]
    if not outside:
        between = None
        conf = None
    else:
        cross = [sim_at(matrix, inner, outer) for inner in triple for outer in outside]
        between = float(sum(cross) / len(cross))
        conf = float(coh - between)
    return {"triple": list(triple), "cohesion": coh, "between": between, "consensus_conf": conf}


def semantic_pack(
    answers: Mapping[int, str],
    embeddings: Mapping[int, np.ndarray],
    *,
    top_ids: Sequence[int],
) -> dict[str, Any]:
    valid_ids = [int(cid) for cid in top_ids if is_valid_answer(answers.get(int(cid), ""))]
    n_valid = len(valid_ids)
    semantic_available = n_valid >= int(MIN_VALID_ANSWERS_SEM)
    adapt_n_ok = n_valid >= int(MIN_VALID_ANSWERS_ADAPT)
    matrix = similarity_matrix(embeddings, valid_ids) if semantic_available else {}
    top2: dict[int, float] = {}
    mean_s: dict[int, float] = {}
    if semantic_available:
        for cid in valid_ids:
            value = sem_top2(matrix, cid, valid_ids)
            if value is None:
                semantic_available = False
                top2 = {}
                break
            top2[int(cid)] = float(value)
            mean_val = sem_mean(matrix, cid, valid_ids)
            if mean_val is not None:
                mean_s[int(cid)] = float(mean_val)
    rel = zscore_map(top2) if semantic_available and top2 else {int(cid): 0.0 for cid in top_ids}
    if not semantic_available:
        rel = {int(cid): 0.0 for cid in top_ids}
        top2 = {}
        mean_s = {}
        matrix = matrix if n_valid >= 2 else {}
    triple = best_coherent_triple(matrix, valid_ids) if adapt_n_ok and matrix else {"triple": [], "cohesion": None, "between": None, "consensus_conf": None}
    return {
        "valid_ids": valid_ids,
        "n_valid": n_valid,
        "semantic_available": bool(semantic_available),
        "adapt_answers_ok": bool(adapt_n_ok),
        "sem_top2": {str(k): v for k, v in top2.items()},
        "sem_mean": {str(k): v for k, v in mean_s.items()},
        "sem_rel": {str(k): float(rel.get(int(k), 0.0)) for k in top_ids},
        "sim": matrix,
        "cohesion": triple.get("cohesion"),
        "between": triple.get("between"),
        "consensus_conf": triple.get("consensus_conf"),
        "best_triple": triple.get("triple") or [],
        "self_sim_excluded": True,
    }


def lexical_pair_rows(answers: Mapping[int, str], matrix: Mapping[str, float], valid_ids: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    ids = [int(cid) for cid in valid_ids]
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            a = answers.get(left, "")
            b = answers.get(right, "")
            tok = float(pair_token_f1(a, b))
            sem = sim_at(matrix, left, right)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "token_f1": tok,
                    "semantic_cosine": sem,
                    "exact_match": bool(answers_match(a, b)),
                    "high_semantic_low_f1": bool(sem >= 0.7 and tok < 50.0),
                    "high_both": bool(sem >= 0.7 and tok >= 50.0),
                    "low_both": bool(sem < 0.4 and tok < 50.0),
                    "high_f1_low_semantic": bool(tok >= 50.0 and sem < 0.4),
                    "left_norm": semantic_normalize(a),
                    "right_norm": semantic_normalize(b),
                }
            )
    return rows


def cluster_answers(matrix: Mapping[str, float], valid_ids: Sequence[int], *, threshold: float) -> list[list[int]]:
    ids = [int(cid) for cid in valid_ids]
    parent = {cid: cid for cid in ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if sim_at(matrix, left, right) >= float(threshold):
                union(left, right)
    groups: dict[int, list[int]] = {}
    for cid in ids:
        groups.setdefault(find(cid), []).append(cid)
    return [sorted(group) for group in groups.values()]


def cluster_diagnostics(matrix: Mapping[str, float], valid_ids: Sequence[int]) -> dict[str, Any]:
    payload = {}
    for threshold in CLUSTER_THRESHOLDS:
        clusters = cluster_answers(matrix, valid_ids, threshold=threshold)
        sizes = [len(group) for group in clusters]
        cohesions = []
        seps = []
        for group in clusters:
            if len(group) >= 2:
                pairs = [sim_at(matrix, a, b) for i, a in enumerate(group) for b in group[i + 1 :]]
                cohesions.append(float(sum(pairs) / max(len(pairs), 1)))
            outside = [cid for cid in valid_ids if cid not in set(group)]
            if group and outside:
                cross = [sim_at(matrix, a, b) for a in group for b in outside]
                seps.append(float(sum(cross) / len(cross)))
        payload[str(threshold)] = {
            "n_clusters": len(clusters),
            "sizes": sizes,
            "mean_cohesion": float(sum(cohesions) / max(len(cohesions), 1)) if cohesions else None,
            "mean_separation": float(sum(seps) / max(len(seps), 1)) if seps else None,
        }
    return payload


def support_gain(base_fit: Mapping[int, float], answer_fit: Mapping[int, float], ids: Sequence[int]) -> dict[int, float]:
    return {int(cid): float(answer_fit.get(int(cid), 0.0)) - float(base_fit.get(int(cid), 0.0)) for cid in ids}


def support_rel(gain: Mapping[int, float]) -> dict[int, float]:
    return zscore_map(gain)


def b0_margin(b0_scores: Sequence[float]) -> float | None:
    ordered = sorted((float(item) for item in b0_scores), reverse=True)
    if len(ordered) < 2:
        return None
    return float(ordered[0] - ordered[1])
