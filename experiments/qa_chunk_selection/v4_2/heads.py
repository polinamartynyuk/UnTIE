"""TRAIN-only contrastive head discovery on B0 top-5 union. No CAL/EVAL labels."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.hn_flags import is_training_eligible_hn
from experiments.qa_chunk_selection.v4.shortlist import top_k_ids

from .features import matrix_at, zscore_map
from .ranking import bidirectional_shortlist, head_ids


def union_topk(
    items: Sequence[InstanceRecord],
    b0_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    seeds: Sequence[int],
    *,
    k: int,
) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    wanted = {item.question_id for item in items}
    for seed in seeds:
        for qid, row in b0_by_seed[int(seed)].items():
            if qid in wanted:
                out[qid].update(int(cid) for cid in top_k_ids(row, k))
    return {qid: set(cids) for qid, cids in out.items()}


def b0_rank_in_union(b0_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]], question_id: str, chunk_id: int, seeds: Sequence[int], *, k: int) -> float:
    ranks = []
    for seed in seeds:
        row = b0_by_seed[int(seed)].get(question_id)
        if row is None:
            continue
        top = [int(cid) for cid in top_k_ids(row, k)]
        if int(chunk_id) in top:
            ranks.append(top.index(int(chunk_id)) + 1)
    return float(min(ranks)) if ranks else 1e9


def _compact_for_question(qid: str, compact: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(cid): row for (question_id, cid), row in compact.items() if question_id == qid}


def _ah_for_question(compact_by_cid: Mapping[int, Mapping[str, Any]], layer: int, head: int) -> dict[int, float]:
    enrich = {int(cid): matrix_at(row, "q_to_span_enrichment", layer, head) for cid, row in compact_by_cid.items()}
    dens = {int(cid): matrix_at(row, "span_to_q_density", layer, head) for cid, row in compact_by_cid.items()}
    z_e = zscore_map(enrich)
    z_d = zscore_map(dens)
    return {cid: 0.5 * (z_e[cid] + z_d[cid]) for cid in compact_by_cid}


def contrastive_r(
    by_question: Mapping[str, Mapping[str, list[int]]],
    compact: Mapping[tuple[str, int], Mapping[str, Any]],
    selected: Sequence[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], float]:
    sample = next(iter(compact.values()), {})
    n_layers = int(sample.get("n_layers") or 0)
    n_heads = int(sample.get("n_heads") or 0)
    heads = list(selected) if selected is not None else head_ids(n_layers, n_heads)
    per_head: dict[tuple[int, int], list[float]] = {key: [] for key in heads}
    for qid, pools in by_question.items():
        pos = sorted({int(cid) for cid in pools.get("pos") or [] if (qid, int(cid)) in compact})
        neg = sorted({int(cid) for cid in pools.get("neg") or [] if (qid, int(cid)) in compact})
        if not pos or not neg:
            continue
        local = _compact_for_question(qid, compact)
        if not local:
            continue
        for layer, head in heads:
            scores = _ah_for_question(local, layer, head)
            pairs = []
            for p in pos:
                for n in neg:
                    a, b = scores.get(p, 0.0), scores.get(n, 0.0)
                    if a > b + 1e-12:
                        pairs.append(1.0)
                    elif abs(a - b) <= 1e-12:
                        pairs.append(0.5)
                    else:
                        pairs.append(0.0)
            if pairs:
                per_head[(layer, head)].append(float(mean(pairs)))
    return {key: float(mean(vals)) if vals else 0.5 for key, vals in per_head.items()}


def stable_heads(
    fold_scores: Sequence[Mapping[tuple[int, int], float]],
    overall: Mapping[tuple[int, int], float],
    *,
    min_folds: int = 2,
) -> list[tuple[int, int]]:
    stable = []
    for key, value in overall.items():
        wins = sum(1 for fold in fold_scores if float(fold.get(key, 0.5)) > 0.5)
        if float(value) > 0.5 and wins >= int(min_folds):
            stable.append(key)
    stable.sort(key=lambda key: (-float(overall[key]), key[0], key[1]))
    return stable


def aggregate_pairwise(
    by_question: Mapping[str, Mapping[str, list[int]]],
    compact: Mapping[tuple[str, int], Mapping[str, Any]],
    selected: Sequence[tuple[int, int]],
) -> float:
    """One selected-head score per question, then mean positive>negative pairwise."""
    per_q: list[float] = []
    for qid, pools in by_question.items():
        pos = sorted({int(cid) for cid in pools.get("pos") or [] if (qid, int(cid)) in compact})
        neg = sorted({int(cid) for cid in pools.get("neg") or [] if (qid, int(cid)) in compact})
        if not pos or not neg or not selected:
            continue
        local = _compact_for_question(qid, compact)
        if not local:
            continue
        raw = bidirectional_shortlist(local, selected)
        pairs = []
        for p in pos:
            for n in neg:
                a, b = float(raw.get(p, 0.0)), float(raw.get(n, 0.0))
                if a > b + 1e-12:
                    pairs.append(1.0)
                elif abs(a - b) <= 1e-12:
                    pairs.append(0.5)
                else:
                    pairs.append(0.0)
        if pairs:
            per_q.append(float(mean(pairs)))
    return float(mean(per_q)) if per_q else 0.5


def choose_n(
    train_fold_pools: Sequence[Mapping[str, Mapping[str, list[int]]]],
    compact: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    n_grid: Sequence[int],
    min_folds: int = 2,
) -> dict[str, Any]:
    tried = []
    best = None
    for n in n_grid:
        inner = []
        for hold in range(len(train_fold_pools)):
            train_maps = [train_fold_pools[index] for index in range(len(train_fold_pools)) if index != hold]
            if not train_maps:
                inner.append(0.5)
                continue
            merged: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"pos": [], "neg": []})
            fold_r = []
            for pool in train_maps:
                fold_r.append(contrastive_r(pool, compact))
                for qid, rows in pool.items():
                    merged[qid]["pos"].extend(rows.get("pos") or [])
                    merged[qid]["neg"].extend(rows.get("neg") or [])
            unique_merged = {
                qid: {"pos": sorted(set(rows["pos"])), "neg": sorted(set(rows["neg"]))}
                for qid, rows in merged.items()
            }
            overall = contrastive_r(unique_merged, compact)
            ranked = stable_heads(fold_r, overall, min_folds=min(min_folds, max(len(train_maps), 1)))
            if not ranked:
                ranked = sorted(overall, key=lambda key: (-float(overall[key]), key[0], key[1]))
            chosen = ranked[: max(int(n), 1)]
            inner.append(aggregate_pairwise(train_fold_pools[hold], compact, chosen))
        payload = {"n": int(n), "mean_inner_pairwise": float(mean(inner)) if inner else 0.5, "folds": inner}
        tried.append(payload)
        if best is None or payload["mean_inner_pairwise"] > best["mean_inner_pairwise"] + 1e-12:
            best = payload
        elif best is not None and abs(payload["mean_inner_pairwise"] - best["mean_inner_pairwise"]) <= 1e-12 and int(n) < int(best["n"]):
            best = payload
    return {"selected": best, "tried": tried}


def build_pools(
    items: Sequence[InstanceRecord],
    union: Mapping[str, set[int]],
    compact: Mapping[tuple[str, int], Mapping[str, Any]],
    flags,
    *,
    reader_easy_threshold: float,
    gold_f1,
    fallback: bool,
    b0_by_seed,
    seeds,
    k: int,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, Any]]:
    pools: dict[str, dict[str, list[int]]] = {}
    n_pos = n_hn = n_easy = 0
    n_invalid_pos = n_invalid_hn = n_invalid_easy = 0
    paired = 0
    for item in items:
        cids = union.get(item.question_id) or set()
        pos = []
        hn = []
        easy = []
        for cid in cids:
            chunk = item.chunk_by_id(int(cid))
            feat = compact.get((item.question_id, int(cid)))
            if feat is None:
                continue
            if not feat.get("valid_span"):
                if chunk.primary_class == "robust_positive" or chunk.label == "positive":
                    n_invalid_pos += 1
                elif chunk.label == "hard_negative":
                    n_invalid_hn += 1
                elif chunk.label == "easy_negative":
                    n_invalid_easy += 1
                continue
            if chunk.primary_class == "robust_positive":
                f1 = float(gold_f1(item, int(cid)))
                if f1 >= float(reader_easy_threshold):
                    pos.append(int(cid))
            elif chunk.label == "hard_negative" and is_training_eligible_hn(flags, item.question_id, int(cid)):
                hn.append(int(cid))
            elif chunk.label == "easy_negative":
                easy.append(int(cid))
        easy.sort(key=lambda cid: b0_rank_in_union(b0_by_seed, item.question_id, cid, seeds, k=k))
        neg = list(hn)
        used_fallback = False
        if fallback:
            extra = [cid for cid in easy if cid not in neg]
            need = max(len(pos), 1)
            take = extra[:need]
            if take:
                neg.extend(take)
                used_fallback = True
        if pos and hn:
            paired += 1
        if pos:
            n_pos += len(set(pos))
        n_hn += len(set(hn))
        n_easy += len(set(easy))
        if pos and neg:
            pools[item.question_id] = {"pos": sorted(set(pos)), "neg": sorted(set(neg)), "hn": sorted(set(hn)), "fallback": used_fallback}
    audit = {
        "n_questions_with_valid_pos_and_HN": paired,
        "n_unique_positive_chunks": n_pos,
        "n_unique_HN_chunks": n_hn,
        "n_easy_available": n_easy,
        "n_pooled_questions": len(pools),
        "n_invalid_positive": n_invalid_pos,
        "n_invalid_hn": n_invalid_hn,
        "n_invalid_easy": n_invalid_easy,
        "negative_fallback": bool(fallback) and any(row.get("fallback") for row in pools.values()),
    }
    return pools, audit
