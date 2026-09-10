"""V4 metrics. Chunk systems reuse V3 enrich; aggregation uses answer-level fields."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import clustered_bootstrap_ci, qa_row
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.evaluate_v2 import build_raw_prediction
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.v2.schemas import rank_ids, to_runtime
from experiments.qa_chunk_selection.v3.metrics_v3 import enrich_v3, summarize_v3

from .answers import answers_match
from .clusters import cluster_id_by_chunk, shortlist_source_clusters

QA_RUNTIME_FIELDS = (
    "qa_answer",
    "qa_chunk_id",
    "qa_chunk_start_char",
    "qa_chunk_end_char",
    "qa_start",
    "qa_end",
    "source_start_char",
    "source_end_char",
    "qa_span_score",
    "qa_span_margin",
    "qa_raw_score",
    "qa_start_logit",
    "qa_end_logit",
    "qa_confidence_proxy",
    "used_m",
    "reader_tokens",
    "qa_forward_passes",
)


def score_map(chunk_ids: Sequence[int], scores: Sequence[float] | Mapping[int, float]) -> dict[int, float]:
    if isinstance(scores, Mapping):
        return {int(key): float(value) for key, value in scores.items()}
    return {int(cid): float(score) for cid, score in zip(chunk_ids, scores)}


def qa_fields_from_row(row: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = {}
    for key in QA_RUNTIME_FIELDS:
        if row is not None and key in row and row[key] is not None:
            payload[key] = row[key]
    return payload


def _chunk_or_none(item: InstanceRecord, chunk_id: Any):
    if chunk_id is None:
        return None
    try:
        return item.chunk_by_id(int(chunk_id))
    except (KeyError, TypeError, ValueError):
        return None


def coerce_extractive_qa(
    item: InstanceRecord,
    qa: Mapping[str, Any] | None,
    *,
    scores: Mapping[int, float] | None = None,
    extra_chunk_ids: Sequence[int] | None = None,
) -> tuple[dict[str, Any], str]:
    """Return a gold-free QA payload whose offsets reconstruct the answer.

    The leftover string is the original answer when offsets cannot be repaired;
    callers may restore it after build_raw_prediction so Token F1 is not dropped.
    """
    hint = qa_fields_from_row(qa)
    answer = str(hint.get("qa_answer") or "")
    scores = scores or {}
    ordered: list[int] = []
    for cid in (hint.get("qa_chunk_id"), *(extra_chunk_ids or []), *(int(chunk.chunk_id) for chunk in item.chunks)):
        if cid is None:
            continue
        value = int(cid)
        if value not in ordered:
            ordered.append(value)
    if not answer:
        if hint.get("qa_chunk_id") is None and ordered:
            hint["qa_chunk_id"] = ordered[0]
        hint.setdefault("qa_chunk_start_char", 0)
        hint.setdefault("qa_chunk_end_char", 0)
        return hint, ""
    for cid in ordered:
        chunk = _chunk_or_none(item, cid)
        if chunk is None:
            continue
        local_start = int(hint.get("qa_chunk_start_char") if hint.get("qa_chunk_start_char") is not None else hint.get("qa_start") or -1)
        local_end = int(hint.get("qa_chunk_end_char") if hint.get("qa_chunk_end_char") is not None else hint.get("qa_end") or -1)
        if 0 <= local_start < local_end <= len(chunk.text) and chunk.text[local_start:local_end] == answer:
            start, end = local_start, local_end
        else:
            start = chunk.text.find(answer)
            if start < 0:
                continue
            end = start + len(answer)
        out = dict(hint)
        out["qa_answer"] = answer
        out["qa_chunk_id"] = int(cid)
        out["qa_chunk_start_char"] = int(start)
        out["qa_chunk_end_char"] = int(end)
        out["qa_span_score"] = float(out.get("qa_span_score") or scores.get(int(cid), 0.0) or 0.0)
        return out, ""
    fallback = dict(hint)
    fallback["qa_answer"] = ""
    fallback.setdefault("qa_chunk_id", ordered[0] if ordered else None)
    fallback["qa_chunk_start_char"] = 0
    fallback["qa_chunk_end_char"] = 0
    return fallback, answer


def chunk_prediction(
    item: InstanceRecord,
    *,
    ranking_method: str,
    scores: Mapping[int, float],
    qa: Mapping[str, Any] | None,
    flags: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    skip_bertscore: bool = True,
) -> dict[str, Any]:
    runtime = to_runtime(item)
    ordered = [int(cid) for cid in rank_ids(runtime.chunks, [float(scores.get(int(chunk.chunk_id), 0.0)) for chunk in runtime.chunks])]
    qa_safe, leftover_answer = coerce_extractive_qa(item, qa, scores=scores, extra_chunk_ids=ordered[:1])
    raw = build_raw_prediction(
        runtime,
        ranking_method=ranking_method,
        scores={str(cid): float(scores.get(int(cid), 0.0)) for cid in ordered},
        ranked_ids=ordered,
        qa=qa_safe,
    )
    if leftover_answer and not raw.get("qa_answer"):
        raw["qa_answer"] = leftover_answer
    raw["seed"] = raw.get("seed")
    return enrich_v3(raw, item, {}, flags=flags or {}, skip_bertscore=skip_bertscore, ks=(1, 5, 10))


def _extractive_qa_payload(
    item: InstanceRecord,
    *,
    answer: str,
    supporting_chunk_ids: Sequence[int],
    provenance_chunk_id: int | None,
    scores: Mapping[int, float],
    qa_hint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hint = dict(qa_hint or {})
    ordered: list[int] = []
    for cid in (provenance_chunk_id, hint.get("qa_chunk_id"), *supporting_chunk_ids):
        if cid is None:
            continue
        value = int(cid)
        if value not in ordered:
            ordered.append(value)
    text = str(answer or hint.get("qa_answer") or "")
    for cid in ordered:
        chunk = item.chunk_by_id(int(cid))
        local_start = int(hint.get("qa_chunk_start_char") or hint.get("qa_start") or -1)
        local_end = int(hint.get("qa_chunk_end_char") or hint.get("qa_end") or -1)
        if text and 0 <= local_start < local_end and chunk.text[local_start:local_end] == text:
            start, end = local_start, local_end
        else:
            start = chunk.text.find(text) if text else 0
            end = start + len(text) if start >= 0 and text else 0
            if text and start < 0:
                continue
        return {
            "qa_answer": text,
            "qa_chunk_id": int(cid),
            "qa_chunk_start_char": int(start),
            "qa_chunk_end_char": int(end),
            "qa_span_score": float(hint.get("qa_span_score") or scores.get(int(cid), 0.0)),
            "qa_span_margin": hint.get("qa_span_margin"),
            "qa_raw_score": hint.get("qa_raw_score") or hint.get("qa_span_score") or scores.get(int(cid), 0.0),
        }
    return {
        "qa_answer": "",
        "qa_chunk_id": provenance_chunk_id,
        "qa_chunk_start_char": 0,
        "qa_chunk_end_char": 0,
        "qa_span_score": float(scores.get(int(provenance_chunk_id), 0.0)) if provenance_chunk_id is not None else 0.0,
    }


def aggregation_prediction(
    item: InstanceRecord,
    *,
    ranking_method: str,
    answer: str,
    supporting_chunk_ids: Sequence[int],
    provenance_chunk_id: int | None,
    n_source_clusters: int,
    source_labels: Sequence[str],
    scores: Mapping[int, float],
    skip_bertscore: bool = True,
    min_overlap: float = 0.6,
    qa_hint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    qa = _extractive_qa_payload(
        item,
        answer=answer,
        supporting_chunk_ids=supporting_chunk_ids,
        provenance_chunk_id=provenance_chunk_id,
        scores=scores,
        qa_hint=qa_hint,
    )
    row = chunk_prediction(item, ranking_method=ranking_method, scores=scores, qa=qa, skip_bertscore=skip_bertscore)
    pos_ids = {int(chunk.chunk_id) for chunk in item.positives}
    has_pos = bool(pos_ids & {int(cid) for cid in supporting_chunk_ids})
    gold_match = float(row.get("exact_match") or 0.0) >= 1.0 or float(row.get("token_f1") or 0.0) >= 50.0
    row["final_answer_support_count"] = len(supporting_chunk_ids)
    row["n_independent_source_clusters"] = int(n_source_clusters)
    row["final_answer_has_original_positive_support"] = float(has_pos)
    row["gold_match_from_non_original_positive"] = float(gold_match and not has_pos)
    mix = Counter(str(label) for label in source_labels)
    row["support_source_label_mix"] = dict(mix)
    row["aggregation_provenance_chunk_id"] = provenance_chunk_id
    if supporting_chunk_ids:
        clusters = shortlist_source_clusters(item, supporting_chunk_ids, min_overlap=min_overlap)
        row["n_source_clusters_of_winning_answer"] = len({row_c["evidence_cluster_id"] for row_c in clusters})
    else:
        row["n_source_clusters_of_winning_answer"] = 0
    return row


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = summarize_v3(list(rows), ks=(1, 5, 10))
    if rows and "final_answer_has_original_positive_support" in rows[0]:
        n = max(len(rows), 1)
        summary["final_answer_has_original_positive_support"] = sum(
            float(row.get("final_answer_has_original_positive_support") or 0.0) for row in rows
        ) / n
        summary["mean_final_answer_support_count"] = sum(float(row.get("final_answer_support_count") or 0.0) for row in rows) / n
        summary["mean_n_independent_source_clusters"] = sum(
            float(row.get("n_independent_source_clusters") or 0.0) for row in rows
        ) / n
    return summary


def transitions(baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    left = {row["question_id"]: row for row in baseline}
    counts = Counter()
    for row in candidate:
        prev = left.get(row["question_id"])
        if not prev:
            continue
        a = str(prev.get("selected_chunk_label") or "none")
        b = str(row.get("selected_chunk_label") or "none")
        ap = str(prev.get("selected_chunk_primary_class") or "")
        bp = str(row.get("selected_chunk_primary_class") or "")
        counts[f"{a}->{b}"] += 1
        if ap == "gold_supporting_robust_negative" and b == "positive":
            counts["GSRN->positive"] += 1
        if a == "positive" and b == "hard_negative":
            counts["positive->HN"] += 1
    return dict(counts)


def paired_delta(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    metric: str = "token_f1",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    left = {row["question_id"]: row for row in baseline}
    deltas = []
    wins = loses = ties = 0
    for row in candidate:
        prev = left.get(row["question_id"])
        if not prev or metric not in row or metric not in prev:
            continue
        delta = float(row[metric]) - float(prev[metric])
        deltas.append({"document_id": row["document_id"], "question_id": row["question_id"], "delta": delta})
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            loses += 1
        else:
            ties += 1
    boot = clustered_bootstrap_ci(deltas, "delta", n_boot=n_boot, seed=seed) if deltas else {"mean": 0.0, "low": 0.0, "high": 0.0}
    return {
        "n": len(deltas),
        "mean_delta": boot.get("mean"),
        "ci95_low": boot.get("low"),
        "ci95_high": boot.get("high"),
        "win": wins,
        "lose": loses,
        "tie": ties,
    }


def flag_map(rows: Sequence[Mapping[str, Any]] | None):
    return flag_index(list(rows or []))
