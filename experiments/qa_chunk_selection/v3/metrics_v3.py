"""V3 ranking / attribution metrics. Overlay never overwrites source_label."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import (
    coverage_at_k,
    hit_at_k,
    hn_at_k,
    mean_or_nan,
    pairwise_pos_gt_hn,
    qa_row,
    reciprocal_rank,
)
from experiments.qa_chunk_selection.schema import InstanceRecord

from .roles import (
    ambiguous_chunks,
    gsrn_chunks,
    known_target_status,
    original_positives,
    v3_target_positives,
    verified_target_mismatch,
)


def _mean(values: Sequence[Any]) -> float:
    clean = [float(item) for item in values if item is not None and not (isinstance(item, float) and math.isnan(item))]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def enrich_v3(
    raw: Mapping[str, Any],
    gold: InstanceRecord,
    overlay: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    flags: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    skip_bertscore: bool = True,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    ranked = [int(item) for item in raw.get("ranked_ids") or []]
    scores = {int(key): float(value) for key, value in (raw.get("scores") or {}).items()}
    p_orig = [chunk.chunk_id for chunk in original_positives(gold)]
    p_v3 = [chunk.chunk_id for chunk in v3_target_positives(gold, overlay)]
    tm = [chunk.chunk_id for chunk in verified_target_mismatch(gold, overlay)]
    amb = [chunk.chunk_id for chunk in ambiguous_chunks(gold)]
    gsrn = [chunk.chunk_id for chunk in gsrn_chunks(gold)]
    hn_all = [chunk.chunk_id for chunk in gold.hard_negatives]
    hn_clean = [
        chunk.chunk_id
        for chunk in gold.hard_negatives
        if not (flags or {}).get((gold.question_id, chunk.chunk_id), {}).get("hn_gold_string_warning")
    ]
    selected = raw.get("selected_chunk_id")
    selected_chunk = gold.chunk_by_id(int(selected)) if selected is not None else None
    selected_label = selected_chunk.label if selected_chunk else None
    selected_primary = selected_chunk.primary_class if selected_chunk else None
    selected_status = (
        known_target_status(gold, overlay, selected_chunk) if selected_chunk is not None else None
    )
    pair_hn, n_hn = pairwise_pos_gt_hn(scores, p_orig, hn_all)
    pair_clean, n_clean = pairwise_pos_gt_hn(scores, p_orig, hn_clean)
    pair_amb, n_amb = pairwise_pos_gt_hn(scores, p_orig, amb)
    pair_gsrn, n_gsrn = pairwise_pos_gt_hn(scores, p_orig, gsrn)
    pair_tm, n_tm = pairwise_pos_gt_hn(scores, p_v3, tm)
    qa_chunk_id = raw.get("qa_chunk_id", selected)
    qa_chunk = None
    if qa_chunk_id is not None:
        try:
            qa_chunk = gold.chunk_by_id(int(qa_chunk_id))
        except KeyError:
            qa_chunk = None
    qa_status = known_target_status(gold, overlay, qa_chunk) if qa_chunk is not None else None
    qa_metrics = qa_row(raw.get("qa_answer"), gold.gold_answers, skip_bertscore=skip_bertscore)
    gold_match = float(qa_metrics.get("exact_match") or 0.0) >= 1.0 or float(qa_metrics.get("token_f1") or 0.0) >= 50.0
    source_orig_pos = bool(qa_chunk is not None and qa_chunk.label == "positive")
    wrong_target = bool(qa_status == "target_mismatch")
    attribution_f1 = float(qa_metrics.get("token_f1") or 0.0)
    if qa_status == "target_mismatch" or (qa_chunk is not None and qa_chunk.chunk_id not in set(p_v3) and qa_status == "known_non_target"):
        attribution_f1 = 0.0
    elif qa_chunk is not None and qa_chunk.chunk_id in set(p_v3):
        attribution_f1 = float(qa_metrics.get("token_f1") or 0.0)
    row = dict(raw)
    row.update(
        {
            "split": gold.split,
            "gold_answers": list(gold.gold_answers),
            "selected_chunk_label": selected_label,
            "selected_chunk_primary_class": selected_primary,
            "selected_v3_target_status": selected_status,
            "qa_chunk_label": qa_chunk.label if qa_chunk is not None else None,
            "qa_v3_target_status": qa_status,
            "n_positive": len(p_orig),
            "n_positive_v3": len(p_v3),
            "n_hard_negative": len(hn_all),
            "n_target_mismatch_verified": len(tm),
            "mrr": reciprocal_rank(ranked, p_orig),
            "pairwise_pos_gt_hn": pair_hn,
            "pairwise_all_hn": pair_hn,
            "pairwise_clean_hn": pair_clean,
            "n_pairs": n_hn,
            "n_pairs_clean": n_clean,
            "pairwise_pos_gt_amb_original": pair_amb,
            "n_pairs_amb_original": n_amb,
            "pairwise_pos_gt_gsrn_original": pair_gsrn,
            "n_pairs_gsrn_original": n_gsrn,
            "pairwise_pv3_gt_tm": pair_tm,
            "n_pairs_tm": n_tm,
            "selected_is_hn": selected_label == "hard_negative",
            "ambiguous_at_1_original": float(selected_label == "ambiguous"),
            "gsrn_at_1_original": float(selected_primary == "gold_supporting_robust_negative"),
            "target_mismatch_at_1": float(selected_status == "target_mismatch"),
            "target_status_known_at_1": float(selected_status is not None),
            "wrong_target_answer": float(wrong_target),
            "wrong_target_known": float(qa_status is not None),
            "gold_match_from_non_original_positive": float(gold_match and not source_orig_pos),
            "attribution_aware_token_f1": attribution_f1,
            **qa_metrics,
        }
    )
    for k in ks:
        top = ranked[: max(int(k), 0)]
        known = 0
        for chunk_id in top:
            try:
                chunk = gold.chunk_by_id(int(chunk_id))
            except KeyError:
                continue
            if known_target_status(gold, overlay, chunk) is not None:
                known += 1
        row[f"hit@{k}"] = hit_at_k(ranked, p_orig, k)
        row[f"hit@{k}_original"] = hit_at_k(ranked, p_orig, k)
        row[f"target_evidence_hit@{k}_v3"] = hit_at_k(ranked, p_v3, k)
        row[f"coverage@{k}"] = coverage_at_k(ranked, p_orig, k)
        row[f"hn@{k}"] = hn_at_k(ranked, hn_all, k)
        row[f"target_status_coverage@{k}"] = float(known / max(len(top), 1)) if top else 0.0
    if selected_label == "positive":
        row["f1_given_selected_positive"] = row.get("token_f1")
    else:
        row["f1_given_selected_positive"] = None
    return row


def summarize_v3(rows: Sequence[Mapping[str, Any]], ks: Sequence[int] = (1, 5, 10)) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    known1 = [row for row in rows if row.get("target_status_known_at_1")]
    tm_count = sum(1 for row in rows if row.get("target_mismatch_at_1"))
    coverage1 = _mean([row.get("target_status_known_at_1") for row in rows])
    summary: dict[str, Any] = {
        "n": n,
        "mrr": mean_or_nan([float(row["mrr"]) for row in rows]),
        "token_f1": mean_or_nan([float(row["token_f1"]) for row in rows if "token_f1" in row]),
        "exact_match": mean_or_nan([float(row["exact_match"]) for row in rows if "exact_match" in row]),
        "char_f1": mean_or_nan([float(row["char_f1"]) for row in rows if "char_f1" in row]),
        "rouge_l_f1": mean_or_nan([float(row["rouge_l_f1"]) for row in rows if "rouge_l_f1" in row]),
        "attribution_aware_token_f1": mean_or_nan(
            [float(row["attribution_aware_token_f1"]) for row in rows if "attribution_aware_token_f1" in row]
        ),
        "pairwise_pos_gt_hn": mean_or_nan(
            [float(row["pairwise_pos_gt_hn"]) for row in rows if row.get("n_pairs")]
        ),
        "pairwise_clean_hn": mean_or_nan(
            [float(row["pairwise_clean_hn"]) for row in rows if row.get("n_pairs_clean")]
        ),
        "pairwise_pos_gt_amb_original": mean_or_nan(
            [float(row["pairwise_pos_gt_amb_original"]) for row in rows if row.get("n_pairs_amb_original")]
        ),
        "pairwise_pos_gt_gsrn_original": mean_or_nan(
            [float(row["pairwise_pos_gt_gsrn_original"]) for row in rows if row.get("n_pairs_gsrn_original")]
        ),
        "pairwise_pv3_gt_tm": mean_or_nan(
            [float(row["pairwise_pv3_gt_tm"]) for row in rows if row.get("n_pairs_tm")]
        ),
        "ambiguous_at_1_original": _mean([row.get("ambiguous_at_1_original") for row in rows]),
        "gsrn_at_1_original": _mean([row.get("gsrn_at_1_original") for row in rows]),
        "target_status_coverage@1": coverage1,
        "target_mismatch@1_lower_bound": float(tm_count / n) if n else float("nan"),
        "target_mismatch_rate@1_adjudicated": float(sum(1 for row in known1 if row.get("target_mismatch_at_1")) / len(known1))
        if known1
        else float("nan"),
        "wrong_target_answer_rate_lower_bound": _mean([row.get("wrong_target_answer") for row in rows]),
        "wrong_target_coverage": _mean([row.get("wrong_target_known") for row in rows]),
        "gold_match_from_non_original_positive": _mean(
            [row.get("gold_match_from_non_original_positive") for row in rows]
        ),
        "f1_given_selected_positive": _mean(
            [row.get("f1_given_selected_positive") for row in rows if row.get("selected_chunk_label") == "positive"]
        ),
    }
    if coverage1 == coverage1 and coverage1 >= 0.999:
        summary["target_mismatch@1"] = summary["target_mismatch_rate@1_adjudicated"]
    else:
        summary["target_mismatch@1"] = None
        summary["target_mismatch@1_note"] = "incomplete coverage; use lower_bound and adjudicated rate"
    for k in ks:
        summary[f"hit@{k}"] = mean_or_nan([float(row[f"hit@{k}"]) for row in rows if f"hit@{k}" in row])
        summary[f"hit@{k}_original"] = mean_or_nan(
            [float(row[f"hit@{k}_original"]) for row in rows if f"hit@{k}_original" in row]
        )
        summary[f"target_evidence_hit@{k}_v3"] = mean_or_nan(
            [float(row[f"target_evidence_hit@{k}_v3"]) for row in rows if f"target_evidence_hit@{k}_v3" in row]
        )
        summary[f"hn@{k}"] = mean_or_nan([float(row[f"hn@{k}"]) for row in rows if f"hn@{k}" in row])
        summary[f"target_status_coverage@{k}"] = mean_or_nan(
            [float(row[f"target_status_coverage@{k}"]) for row in rows if f"target_status_coverage@{k}" in row]
        )
    return summary
