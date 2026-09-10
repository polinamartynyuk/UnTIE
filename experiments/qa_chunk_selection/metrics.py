"""Retrieval, answerability, and QA metrics. Labels are used only after scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from untie.answer_selection import normalize_answer_key
from untie.aspect_qa.normalize import identity_key
from untie.extraction_metrics import char_f1, rouge_l_f1, token_f1

from .schema import InstanceRecord


def _rank_index(ranked_ids: Sequence[int], wanted: set[int]) -> int | None:
    for index, chunk_id in enumerate(ranked_ids, start=1):
        if int(chunk_id) in wanted:
            return index
    return None


def hit_at_k(ranked_ids: Sequence[int], positive_ids: Sequence[int], k: int) -> float:
    if not positive_ids:
        return 0.0
    top = {int(item) for item in ranked_ids[: max(int(k), 0)]}
    return float(bool(top & {int(item) for item in positive_ids}))


def coverage_at_k(ranked_ids: Sequence[int], positive_ids: Sequence[int], k: int) -> float:
    positives = {int(item) for item in positive_ids}
    if not positives:
        return 0.0
    top = {int(item) for item in ranked_ids[: max(int(k), 0)]}
    return float(len(top & positives) / len(positives))


def reciprocal_rank(ranked_ids: Sequence[int], positive_ids: Sequence[int]) -> float:
    if not positive_ids:
        return 0.0
    rank = _rank_index(ranked_ids, {int(item) for item in positive_ids})
    return 0.0 if rank is None else 1.0 / float(rank)


def hn_at_k(ranked_ids: Sequence[int], hn_ids: Sequence[int], k: int) -> float:
    pool = list(ranked_ids)
    k_eff = min(max(int(k), 0), len(pool))
    if k_eff <= 0:
        return 0.0
    top = {int(item) for item in pool[:k_eff]}
    return float(len(top & {int(item) for item in hn_ids}) / k_eff)


def pairwise_pos_gt_hn(
    scores: Mapping[int, float],
    positive_ids: Sequence[int],
    hn_ids: Sequence[int],
) -> tuple[float, int]:
    wins = 0
    total = 0
    for pos in positive_ids:
        for hn in hn_ids:
            total += 1
            if float(scores.get(int(pos), float("-inf"))) > float(
                scores.get(int(hn), float("-inf"))
            ):
                wins += 1
    if total == 0:
        return float("nan"), 0
    return float(wins / total), total


def ranking_row(
    instance: InstanceRecord,
    ranked_ids: Sequence[int],
    scores: Mapping[int, float],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    pos = [chunk.chunk_id for chunk in instance.positives]
    hn = [chunk.chunk_id for chunk in instance.hard_negatives]
    selected = int(ranked_ids[0]) if ranked_ids else None
    selected_label = None
    if selected is not None:
        try:
            selected_label = instance.chunk_by_id(selected).label
        except KeyError:
            selected_label = None
    pair_acc, pair_n = pairwise_pos_gt_hn(scores, pos, hn)
    row: dict[str, Any] = {
        "document_id": instance.document_id,
        "question_id": instance.question_id,
        "aspect": instance.aspect,
        "split": instance.split,
        "n_candidates": len(instance.chunks),
        "n_positive": len(pos),
        "n_hard_negative": len(hn),
        "selected_chunk_id": selected,
        "selected_chunk_label": selected_label,
        "selected_is_hn": selected_label == "hard_negative",
        "mrr": reciprocal_rank(ranked_ids, pos),
        "pairwise_pos_gt_hn": pair_acc,
        "n_pairs": pair_n,
        "hn_bucket": hn_bucket(len(hn)),
    }
    for k in ks:
        row[f"hit@{k}"] = hit_at_k(ranked_ids, pos, k)
        row[f"coverage@{k}"] = coverage_at_k(ranked_ids, pos, k)
        row[f"hn@{k}"] = hn_at_k(ranked_ids, hn, k)
    return row


def hn_bucket(n_hard: int) -> str:
    if n_hard <= 0:
        return "0"
    if n_hard == 1:
        return "1"
    if n_hard <= 3:
        return "2-3"
    return "4+"


def exact_match(prediction: str | None, gold_answers: Sequence[str]) -> float:
    predicted = identity_key(prediction or "")
    if not predicted:
        return 0.0
    return float(any(predicted == identity_key(item) for item in gold_answers if str(item).strip()))


def qa_row(prediction: str | None, gold_answers: Sequence[str], *, skip_bertscore: bool = True) -> dict[str, float]:
    pred = prediction or ""
    gold = list(gold_answers)
    row = {
        "exact_match": exact_match(pred, gold),
        "token_f1": float(token_f1(pred, gold)),
        "char_f1": float(char_f1(pred, gold)),
        "rouge_l_f1": float(rouge_l_f1(pred, gold)),
        "no_answer": float(not str(pred).strip()),
        "coverage": float(bool(str(pred).strip())),
    }
    if not skip_bertscore:
        from untie.extraction_metrics import bertscore_metrics

        bert = bertscore_metrics(pred, gold, lang="en")
        row["bertscore_f1"] = float(bert["f1"])
    return row


def mean_or_nan(values: Sequence[float]) -> float:
    clean = [float(item) for item in values if item is not None and not (isinstance(item, float) and math.isnan(item))]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def summarize_ranking(rows: Sequence[Mapping[str, Any]], ks: Sequence[int] = (1, 5, 10)) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    summary: dict[str, Any] = {
        "n": len(rows),
        "mrr": mean_or_nan([float(row["mrr"]) for row in rows]),
        "pairwise_pos_gt_hn": mean_or_nan(
            [float(row["pairwise_pos_gt_hn"]) for row in rows if row.get("n_pairs", 0)]
        ),
        "p_selected_hn": mean_or_nan([float(bool(row.get("selected_is_hn"))) for row in rows]),
        "qa_coverage": mean_or_nan([float(row["coverage"]) for row in rows if "coverage" in row]),
        "exact_match": mean_or_nan([float(row["exact_match"]) for row in rows if "exact_match" in row]),
        "token_f1": mean_or_nan([float(row["token_f1"]) for row in rows if "token_f1" in row]),
        "char_f1": mean_or_nan([float(row["char_f1"]) for row in rows if "char_f1" in row]),
        "rouge_l_f1": mean_or_nan([float(row["rouge_l_f1"]) for row in rows if "rouge_l_f1" in row]),
    }
    for k in ks:
        summary[f"hit@{k}"] = mean_or_nan([float(row[f"hit@{k}"]) for row in rows if f"hit@{k}" in row])
        summary[f"coverage@{k}"] = mean_or_nan(
            [float(row[f"coverage@{k}"]) for row in rows if f"coverage@{k}" in row]
        )
        summary[f"hn@{k}"] = mean_or_nan([float(row[f"hn@{k}"]) for row in rows if f"hn@{k}" in row])
    by_bucket: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row.get("hn_bucket") or "unknown")].append(row)
    summary["by_hn_bucket"] = {
        bucket: {
            "n": len(items),
            "hit@1": mean_or_nan([float(item.get("hit@1", float("nan"))) for item in items]),
            "hn@1": mean_or_nan([float(item.get("hn@1", float("nan"))) for item in items]),
            "pairwise_pos_gt_hn": mean_or_nan(
                [float(item["pairwise_pos_gt_hn"]) for item in items if item.get("n_pairs", 0)]
            ),
        }
        for bucket, items in sorted(by_bucket.items())
    }
    return summary


def binary_pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if not labels or len(set(int(item) for item in labels)) < 2:
        return float("nan")
    from sklearn.metrics import average_precision_score, roc_auc_score

    del roc_auc_score
    return float(average_precision_score(list(labels), list(scores)))


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if not labels or len(set(int(item) for item in labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(list(labels), list(scores)))


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def threshold_f1(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, float]:
    pred = [int(float(score) > float(threshold)) for score in scores]
    tp = sum(int(y == 1 and p == 1) for y, p in zip(labels, pred))
    fp = sum(int(y == 0 and p == 1) for y, p in zip(labels, pred))
    fn = sum(int(y == 1 and p == 0) for y, p in zip(labels, pred))
    tn = sum(int(y == 0 and p == 0) for y, p in zip(labels, pred))
    return {
        "threshold": float(threshold),
        "precision": tp / float(tp + fp) if tp + fp else 0.0,
        "recall": tp / float(tp + fn) if tp + fn else 0.0,
        "f1": f1_from_counts(tp, fp, fn),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def best_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    unique = sorted(set(float(item) for item in scores))
    if not unique:
        return threshold_f1(labels, scores, 0.5)
    candidates = unique + [unique[-1] + 1.0]
    scored = [threshold_f1(labels, scores, value) for value in candidates]
    return max(scored, key=lambda item: (item["f1"], item["threshold"]))


def clustered_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    by_doc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if metric not in row:
            continue
        value = row[metric]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        by_doc[str(row["document_id"])].append(float(value))
    docs = sorted(by_doc)
    if not docs:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(docs, size=len(docs), replace=True)
        values = [item for doc in sampled for item in by_doc[str(doc)]]
        stats.append(float(sum(values) / len(values)) if values else float("nan"))
    clean = [item for item in stats if not math.isnan(item)]
    if not clean:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    return {
        "mean": float(sum(clean) / len(clean)),
        "low": float(np.percentile(clean, 2.5)),
        "high": float(np.percentile(clean, 97.5)),
    }


def paired_delta(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    left = {(row["document_id"], row["question_id"]): row for row in left_rows}
    right = {(row["document_id"], row["question_id"]): row for row in right_rows}
    keys = sorted(set(left) & set(right))
    merged = []
    for document_id, question_id in keys:
        lv = left[(document_id, question_id)].get(metric)
        rv = right[(document_id, question_id)].get(metric)
        if lv is None or rv is None:
            continue
        merged.append(
            {
                "document_id": document_id,
                "question_id": question_id,
                "delta": float(rv) - float(lv),
            }
        )
    ci = clustered_bootstrap_ci(merged, "delta", n_boot=n_boot, seed=seed)
    return {"n": len(merged), **ci}


def normalized_answer(text: str) -> str:
    return normalize_answer_key(text)
